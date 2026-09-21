"""FastAPI application: the HTTP face of the optimizer.

Run with: ``uv run uvicorn qgati.api.main:app --reload``

Endpoints
---------
``GET  /health``
    Liveness probe.
``GET  /solvers``
    What can be run, and which one is the production default.
``POST /scenarios``
    Create a scenario — generated at random, or described explicitly — and get
    back an id to optimize.
``GET  /scenarios``
    List stored scenarios.
``GET  /scenarios/{scenario_id}``
    Fetch one.
``POST /optimize/{scenario_id}``
    Run one solver on a stored scenario. Defaults to the registry's production
    default, **ACO**.
``GET  /optimize/{scenario_id}/compare``
    Run every solver on the same cost matrix and return the comparison — the
    Phase 4 benchmark, scoped to one scenario. This is where QPSO's research
    value sits next to the production default.
``GET  /graph/delhi``
    The road network as GeoJSON, so a map renders real street geometry rather
    than straight lines between stops. Scope it to a scenario to keep it small.

Three design points worth knowing
---------------------------------
**The graph is a dependency, not a global.** :func:`get_graph` is a FastAPI
dependency, so tests can substitute a synthetic graph through
``app.dependency_overrides`` and exercise the whole API without the cached Delhi
extract or the network. It is loaded lazily and cached, because
``load_delhi_graph`` is ~20 s on a cold cache and ~0.2 s warm.

**Responses resolve indices back into ids.** Solvers return
:class:`~qgati.optimizer.models.Solution`, which holds *positions* in the
scenario's delivery list — meaningless to a client. Every response here maps them
back to delivery ids and road-graph nodes, and optionally to a road polyline.

**The scenario is named by the URL, never the body.** ``POST /optimize/{scenario_id}``
takes only solver parameters; anything that identifies *what* to solve is a path
parameter. That keeps the two kinds of input from drifting into one payload.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Annotated

import networkx as nx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from qgati.api.schemas import (
    CompareResponse,
    DeliveryOut,
    DepotOut,
    OptimizeRequest,
    OptimizeResponse,
    RouteOut,
    ScenarioCreateRequest,
    ScenarioResponse,
    ScenarioSummary,
    SolverResultOut,
    StopOut,
    VehicleOut,
)
from qgati.api.store import ScenarioNotFound, ScenarioStore
from qgati.graph import (
    DEFAULT_PADDING_M,
    bbox_around_nodes,
    build_cost_matrix,
    graph_to_geojson,
    load_delhi_graph,
    nearest_node,
    parse_bbox,
    route_polyline,
)
from qgati.optimizer import (
    DEFAULT_SOLVER_KEY,
    MAX_EXACT_DELIVERIES,
    Delivery,
    Depot,
    Evaluation,
    Scenario,
    Solution,
    Vehicle,
    build_random_scenario,
    evaluate,
    get_solver,
    servable_nodes,
)
from qgati.optimizer.registry import DEFAULT_ITERATIONS, DEFAULT_POPULATION, SOLVERS

__all__ = ["app", "create_app"]

LOGGER = logging.getLogger(__name__)

#: Browser origins allowed to call this API. The Next.js dev server runs on port
#: 3000, and both spellings are listed because a browser treats ``localhost`` and
#: ``127.0.0.1`` as distinct origins.
ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

#: Ceiling on a generated instance. A cost matrix is quadratic in stops and the
#: solvers are iterated in Python, so an unbounded request could pin a worker for
#: minutes. 200 stops is far beyond the sizes this project benchmarks.
MAX_GENERATED_DELIVERIES = 200

#: Ceiling on a request-time bounding-box clip. A bbox covering the whole
#: extract is the unscoped request, which is allowed — this only bounds a bbox
#: so wide it would be a coordinate typo (e.g. a lat/lon swap).
MAX_BBOX_SPAN = 1.0

#: Starlette renamed this constant (422 UNPROCESSABLE_ENTITY -> ..._CONTENT), so
#: accept whichever the installed version defines rather than emitting a
#: deprecation warning on every validation failure.
UNPROCESSABLE = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)

STORE = ScenarioStore()


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _load_graph() -> nx.Graph:
    """Load the Delhi graph once per process.

    ``lru_cache`` rather than a module-level call so the ~20 s cold-cache fetch
    happens on first request, not at import — which keeps the test suite and the
    OpenAPI schema generation fast.
    """
    return load_delhi_graph()


def get_graph() -> nx.Graph:
    """The road graph, as an overridable dependency."""
    return _load_graph()


def get_store() -> ScenarioStore:
    """The scenario store, as an overridable dependency."""
    return STORE


GraphDep = Annotated[nx.Graph, Depends(get_graph)]
StoreDep = Annotated[ScenarioStore, Depends(get_store)]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_location(
    graph: nx.Graph,
    node: int | str | None,
    lat: float | None,
    lon: float | None,
    what: str,
) -> int | str:
    """Turn either a node id or a coordinate into a node id."""
    if node is not None:
        if node not in graph:
            raise HTTPException(
                status_code=UNPROCESSABLE,
                detail=f"{what}: node {node!r} is not in the road graph",
            )
        return node
    assert lat is not None and lon is not None  # guaranteed by the schema
    return nearest_node(graph, lat=lat, lon=lon)  # type: ignore[arg-type]


def _coordinates(graph: nx.Graph, node: int | str) -> tuple[float, float]:
    """Lat/lon for a node, from either an OSMnx or a synthetic graph layout."""
    data = graph.nodes[node]
    if "y" in data and "x" in data:  # OSMnx
        return float(data["y"]), float(data["x"])
    if "pos" in data:  # synthetic test graph stores (x, y)
        x, y = data["pos"]
        return float(y), float(x)
    raise HTTPException(
        status_code=UNPROCESSABLE,
        detail=f"road-graph node {node!r} carries no coordinates",
    )


def _not_found(scenario_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"no scenario with id {scenario_id!r}",
    )


def _scenario_response(record) -> ScenarioResponse:
    scenario = record.scenario
    return ScenarioResponse(
        scenario_id=record.scenario_id,
        depot=DepotOut(
            node=scenario.depot.node, lat=scenario.depot.lat, lon=scenario.depot.lon
        ),
        deliveries=[
            DeliveryOut(id=item.id, node=item.node, demand=item.demand)
            for item in scenario.deliveries
        ],
        vehicles=[
            VehicleOut(id=vehicle.id, capacity=vehicle.capacity)
            for vehicle in scenario.vehicles
        ],
        total_demand=scenario.total_demand,
        total_capacity=scenario.total_capacity,
        n_deliveries=scenario.n_deliveries,
        n_vehicles=scenario.n_vehicles,
        exactly_solvable=scenario.n_deliveries <= MAX_EXACT_DELIVERIES,
    )


def _routes_out(
    record,
    solution: Solution,
    evaluation: Evaluation,
    graph: nx.Graph,
    include_geometry: bool = True,
) -> list[RouteOut]:
    """Resolve a solution's delivery indices into ids, nodes, loads and costs.

    ``Solution`` holds *positions* in the scenario's delivery list, which mean
    nothing to a client; this is where they become delivery ids and graph nodes.

    With ``include_geometry``, each route is also traced along the road network.
    The depot is prepended and appended by this function because a
    :class:`Solution` route lists only its stops — the tour's return leg is
    implicit until something has to draw it.
    """
    scenario = record.scenario
    depot_node = scenario.depot.node
    routes: list[RouteOut] = []

    for position, route in enumerate(solution.routes):
        vehicle = scenario.vehicles[position]
        stop_nodes = [scenario.deliveries[index].node for index in route]

        # An unused vehicle has no tour to draw; emitting the depot twice would
        # render as a one-point line rather than as nothing.
        geometry = (
            route_polyline(graph, [depot_node, *stop_nodes, depot_node])
            if include_geometry and stop_nodes
            else []
        )

        routes.append(
            RouteOut(
                vehicle_id=vehicle.id,
                stops=[
                    StopOut(
                        delivery_id=scenario.deliveries[index].id,
                        node=scenario.deliveries[index].node,
                    )
                    for index in route
                ],
                load=evaluation.route_loads[position],
                capacity=vehicle.capacity,
                travel_cost=evaluation.route_costs[position],
                geometry=geometry,
            )
        )
    return routes


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    application = FastAPI(
        title="Q-Gati API",
        version="0.1.0",
        description=(
            "Multi-algorithm vehicle routing for Delhi. Six solvers behind one "
            "contract; ACO is the production default, chosen on Phase 4's "
            "measured benchmark results."
        ),
    )

    # The frontend is a separate origin (the Next.js dev server on port 3000),
    # so the browser needs explicit permission before it may read a response.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        """Last-resort handler: log the detail, return none of it.

        Without this, an unexpected error surfaces as Starlette's bare
        ``Internal Server Error`` text — not JSON, so a client's error handling
        breaks on the one response it most needs to parse. The traceback goes to
        the server log, where it is useful, and never to the client, where it
        would leak file paths and internals.

        Expected failures never reach here: they are raised as ``HTTPException``
        with a status the client can act on.
        """
        LOGGER.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    @application.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @application.get("/solvers", tags=["meta"])
    def list_solvers() -> dict:
        """Every registered solver, and which one is the production default."""
        return {
            "default": DEFAULT_SOLVER_KEY,
            "default_iterations": DEFAULT_ITERATIONS,
            "default_population": DEFAULT_POPULATION,
            "solvers": [spec.to_dict() for spec in SOLVERS],
        }

    # -- scenarios ---------------------------------------------------------- #
    @application.post(
        "/scenarios", response_model=ScenarioResponse, status_code=status.HTTP_201_CREATED,
        tags=["scenarios"],
    )
    def create_scenario(
        request: ScenarioCreateRequest, graph: GraphDep, store: StoreDep
    ) -> ScenarioResponse:
        """Create a scenario, generated at random or described explicitly."""
        if request.kind == "generate":
            assert request.n_deliveries is not None and request.n_vehicles is not None
            if request.n_deliveries > MAX_GENERATED_DELIVERIES:
                raise HTTPException(
                    status_code=UNPROCESSABLE,
                    detail=(
                        f"n_deliveries must be at most {MAX_GENERATED_DELIVERIES}, "
                        f"got {request.n_deliveries}"
                    ),
                )
            try:
                # build_random_scenario samples only from the largest strongly-
                # connected subgraph, so the instance is servable by construction.
                scenario = build_random_scenario(
                    graph,
                    request.n_deliveries,
                    request.n_vehicles,
                    seed=request.seed,
                )
            except ValueError as error:
                raise HTTPException(
                    status_code=UNPROCESSABLE, detail=str(error)
                ) from None
        else:
            assert request.depot is not None
            assert request.deliveries is not None
            assert request.vehicles is not None

            depot_node = _resolve_location(
                graph, request.depot.node, request.depot.lat, request.depot.lon, "depot"
            )
            depot_lat, depot_lon = _coordinates(graph, depot_node)

            deliveries = []
            for stop in request.deliveries:
                node = _resolve_location(
                    graph, stop.node, stop.lat, stop.lon, f"delivery {stop.id!r}"
                )
                deliveries.append(Delivery(id=stop.id, node=node, demand=stop.demand))

            try:
                scenario = Scenario(
                    depot=Depot(node=depot_node, lat=depot_lat, lon=depot_lon),
                    deliveries=tuple(deliveries),
                    vehicles=tuple(
                        Vehicle(id=vehicle.id, capacity=vehicle.capacity)
                        for vehicle in request.vehicles
                    ),
                )
            except ValueError as error:
                # E.g. demand exceeding fleet capacity — a client error, not a bug.
                raise HTTPException(
                    status_code=UNPROCESSABLE, detail=str(error)
                ) from None

        try:
            cost_matrix = build_cost_matrix(graph, scenario)
        except ValueError as error:
            # Unreachable node pairs. build_cost_matrix refuses to substitute
            # inf, because that would let an optimizer return a confidently wrong
            # answer; the message tells the client to use the servable subgraph.
            raise HTTPException(
                status_code=UNPROCESSABLE, detail=str(error)
            ) from None

        return _scenario_response(store.add(scenario, cost_matrix))

    @application.get(
        "/scenarios", response_model=list[ScenarioSummary], tags=["scenarios"]
    )
    def list_scenarios(store: StoreDep) -> list[ScenarioSummary]:
        """Every stored scenario, oldest first."""
        return [
            ScenarioSummary(
                scenario_id=record.scenario_id,
                n_deliveries=record.scenario.n_deliveries,
                n_vehicles=record.scenario.n_vehicles,
                total_demand=record.scenario.total_demand,
            )
            for record in store.list()
        ]

    @application.get(
        "/scenarios/{scenario_id}", response_model=ScenarioResponse, tags=["scenarios"]
    )
    def get_scenario(scenario_id: str, store: StoreDep) -> ScenarioResponse:
        """Fetch one stored scenario."""
        try:
            return _scenario_response(store.get(scenario_id))
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

    @application.get("/graph/servable", tags=["meta"])
    def servable_count(graph: GraphDep) -> dict:
        """How many mutually-reachable nodes this graph offers for scenarios."""
        return {"servable_nodes": len(servable_nodes(graph))}

    @application.get("/graph/delhi", tags=["graph"])
    def graph_geojson(
        graph: GraphDep,
        store: StoreDep,
        scenario_id: str | None = None,
        bbox: str | None = None,
        padding_m: float = Query(default=DEFAULT_PADDING_M, ge=0, le=5000),
        include_nodes: bool = True,
    ) -> dict:
        """The road network as GeoJSON, for drawing a map.

        Unscoped, this is the whole 2033-node / 4778-edge extract, about 1 MB of
        JSON — fine as a one-time basemap fetch, wasteful for a single delivery
        round. Scope it to keep it small:

        * ``?scenario_id=...`` — the box around that scenario's stops, padded by
          ``padding_m`` (default 400 m) so the streets a route might use are all
          present. This is the one the frontend wants.
        * ``?bbox=min_lon,min_lat,max_lon,max_lat`` — an explicit box.

        The two are mutually exclusive: a scenario already defines a box, and
        silently letting one override the other would make a request that looks
        specific quietly return something else.

        Edges are emitted as ``LineString`` features using OSMnx's own geometry
        where an edge has one, so curved roads draw as curves. Each carries its
        endpoints as ``u``/``v``, which is what lets the frontend highlight the
        edges a solved route actually used rather than redrawing the geometry.
        """
        clip = None
        if scenario_id is not None and bbox is not None:
            raise HTTPException(
                status_code=UNPROCESSABLE,
                detail="give either 'scenario_id' or 'bbox', not both",
            )

        if bbox is not None:
            try:
                clip = parse_bbox(bbox)
            except ValueError as error:
                raise HTTPException(
                    status_code=UNPROCESSABLE, detail=str(error)
                ) from None
            if (clip[2] - clip[0]) > MAX_BBOX_SPAN or (clip[3] - clip[1]) > MAX_BBOX_SPAN:
                raise HTTPException(
                    status_code=UNPROCESSABLE,
                    detail=(
                        f"bbox spans more than {MAX_BBOX_SPAN} degrees; this "
                        "extract covers a few kilometres around Connaught Place"
                    ),
                )
        elif scenario_id is not None:
            try:
                record = store.get(scenario_id)
            except ScenarioNotFound:
                raise _not_found(scenario_id) from None

            scenario = record.scenario
            clip = bbox_around_nodes(
                graph,
                [scenario.depot.node] + [stop.node for stop in scenario.deliveries],
                padding_m=padding_m,
            )

        return graph_to_geojson(graph, bbox=clip, include_nodes=include_nodes)

    # -- optimize ----------------------------------------------------------- #
    @application.post(
        "/optimize/{scenario_id}", response_model=OptimizeResponse, tags=["optimize"]
    )
    def optimize(
        scenario_id: str,
        graph: GraphDep,
        store: StoreDep,
        payload: OptimizeRequest | None = None,
    ) -> OptimizeResponse:
        """Run one solver on a stored scenario.

        Defaults to the production default solver (ACO) at the benchmark's
        default search effort, so an API result is comparable with a benchmark
        result. The body is optional: ``POST /optimize/{id}`` with no payload at
        all runs the default.
        """
        request = payload or OptimizeRequest()

        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        try:
            spec = get_solver(request.solver or DEFAULT_SOLVER_KEY)
        except KeyError as error:
            raise HTTPException(
                status_code=UNPROCESSABLE, detail=str(error)
            ) from None

        iterations = request.iterations or DEFAULT_ITERATIONS
        population = request.population or DEFAULT_POPULATION

        if spec.is_exact and record.scenario.n_deliveries > (spec.exact_limit or 0):
            raise HTTPException(
                status_code=UNPROCESSABLE,
                detail=(
                    f"{spec.name} is limited to {spec.exact_limit} deliveries; this "
                    f"scenario has {record.scenario.n_deliveries}. Use a heuristic."
                ),
            )

        start = time.perf_counter()
        solution, cost, convergence = spec(
            record.cost_matrix,
            record.scenario,
            seed=request.seed,
            population=population,
            iterations=iterations,
        )
        runtime_ms = (time.perf_counter() - start) * 1000.0

        evaluation = evaluate(solution, record.scenario, record.cost_matrix)

        return OptimizeResponse(
            scenario_id=record.scenario_id,
            solver=spec.key,
            solver_name=spec.name,
            seed=request.seed,
            iterations=iterations if spec.is_stochastic else None,
            population=population if spec.is_stochastic else None,
            cost=cost,
            travel_cost=evaluation.travel_cost,
            feasible=evaluation.feasible,
            runtime_ms=runtime_ms,
            routes=_routes_out(
                record,
                solution,
                evaluation,
                graph,
                include_geometry=request.include_geometry,
            ),
            convergence=convergence,
        )

    @application.get(
        "/optimize/{scenario_id}/compare",
        response_model=CompareResponse,
        tags=["optimize"],
    )
    def compare(
        scenario_id: str,
        store: StoreDep,
        seed: int | None = None,
        iterations: int | None = None,
        population: int | None = None,
    ) -> CompareResponse:
        """Run every solver on one scenario and compare them.

        Cost matrices are identical across all solvers and all are scored by the
        same fitness function, so a difference between two rows is a difference
        in search quality rather than in accounting. Brute force is skipped for
        instances beyond its exact limit, and reports why.
        """
        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        iterations = iterations or DEFAULT_ITERATIONS
        population = population or DEFAULT_POPULATION
        n_deliveries = record.scenario.n_deliveries

        results: list[SolverResultOut] = []
        solutions: dict[str, tuple] = {}
        for spec in SOLVERS:
            if spec.is_exact and n_deliveries > (spec.exact_limit or 0):
                results.append(
                    SolverResultOut(
                        solver=spec.key,
                        solver_name=spec.name,
                        skipped=(
                            f"exact solver limited to {spec.exact_limit} deliveries, "
                            f"scenario has {n_deliveries}"
                        ),
                    )
                )
                continue

            start = time.perf_counter()
            solution, cost, _ = spec(
                record.cost_matrix,
                record.scenario,
                seed=seed,
                population=population,
                iterations=iterations,
            )
            runtime_ms = (time.perf_counter() - start) * 1000.0
            evaluation = evaluate(solution, record.scenario, record.cost_matrix)
            solutions[spec.key] = (solution, evaluation)
            results.append(
                SolverResultOut(
                    solver=spec.key,
                    solver_name=spec.name,
                    cost=cost,
                    travel_cost=evaluation.travel_cost,
                    feasible=evaluation.feasible,
                    runtime_ms=runtime_ms,
                )
            )

        ran = [
            result
            for result in results
            if result.skipped is None and result.travel_cost is not None
        ]
        best_known = min((result.travel_cost for result in ran), default=None)
        optimal = (
            solutions["brute_force"][1].travel_cost if "brute_force" in solutions else None
        )

        def gap(value: float | None, reference: float | None) -> float | None:
            if value is None or reference is None or reference <= 0.0:
                return None
            return 100.0 * (value - reference) / reference

        for result in results:
            if result.skipped is not None:
                continue
            result.gap_vs_optimal_pct = gap(result.travel_cost, optimal)
            result.gap_vs_best_pct = gap(result.travel_cost, best_known)

        return CompareResponse(
            scenario_id=record.scenario_id,
            optimal=optimal,
            best_known=best_known,
            results=results,
        )

    return application


app = create_app()
