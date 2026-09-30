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
``POST /scenarios/{scenario_id}/incident``
    Inject a live incident — a closed road, or one that is slow but passable —
    into a stored scenario. Rebuilds its cost matrix; does not re-optimize.
``DELETE /scenarios/{scenario_id}/incident/{incident_id}``
    Revert one incident, restoring the conditions it replaced.
``POST /scenarios/{scenario_id}/detect``
    Judge one newly observed travel time against that road's logged history and
    say whether it is anomalous. Flags; does not re-optimize, and writes nothing.
``POST /scenarios/{scenario_id}/watcher``
    Dispatch a simulated GPS fleet against a stored scenario: solve it, put its
    vehicles on the roads their routes run along, and start measuring.
``GET  /scenarios/{scenario_id}/watcher``
    What that fleet is doing, and where each of its vehicles is.
``POST /scenarios/{scenario_id}/watcher/tick``
    Advance the fleet one interval, synchronously.
``DELETE /scenarios/{scenario_id}/watcher``
    Stop it. The readings it took stay.
``POST /optimize/{scenario_id}``
    Run one solver on a stored scenario. Defaults to the registry's production
    default, **QPSO**.
``GET  /optimize/{scenario_id}/compare``
    Run every solver on the same cost matrix and return the comparison — the
    Phase 4 benchmark, scoped to one scenario. This is where QPSO's headline
    result sits next to the conventional solvers it was benchmarked against.
``GET  /graph/delhi``
    The road network as GeoJSON, so a map renders real street geometry rather
    than straight lines between stops. Scope it to a scenario to keep it small.
``GET  /traffic/log``
    Inspect the traffic-condition log — paginated. The log is being collected
    for a future travel-time model; nothing here predicts anything.

Three design points worth knowing
---------------------------------
**The graph is a dependency, not a global.** :func:`get_graph` is a FastAPI
dependency, so tests can substitute a synthetic graph through
``app.dependency_overrides`` and exercise the whole API without the cached Delhi
extract or the network. It is loaded lazily and cached, because
``load_delhi_graph`` is ~20 s on a cold cache and ~0.2 s warm. The traffic log
store is injected the same way, so a test run never writes to the developer's
collected data.

**Responses resolve indices back into ids.** Solvers return
:class:`~qgati.optimizer.models.Solution`, which holds *positions* in the
scenario's delivery list — meaningless to a client. Every response here maps them
back to delivery ids and road-graph nodes, and optionally to a road polyline.

**The scenario is named by the URL, never the body.** ``POST /optimize/{scenario_id}``
takes only solver parameters; anything that identifies *what* to solve is a path
parameter. That keeps the two kinds of input from drifting into one payload.

Simulated traffic conditions
----------------------------
``POST /scenarios`` accepts an optional ``conditions`` block — a timestamp,
accidents, closures. Those conditions price the scenario's cost matrix, are
recorded with it, and are echoed back in the response. They are **fixed for the
scenario's lifetime**: the same id always optimizes the same costs, so comparing
two solvers on it compares solvers rather than moments. Create two scenarios from
one seed with different timestamps to compare peak against moderate.

The one thing that moves those costs afterwards is an explicit operator report,
through the two ``/incident`` routes below. They are the deliberate exception to
the paragraph above, and they are narrow on purpose: an incident changes a
scenario's *costs* and nothing else, leaving any solution a client already holds
where it is until that client asks for a new one. Re-optimizing on an incident is
a separate concern with its own module.

Injecting one re-folds the scenario's creation-time conditions with its live
reports and re-prices the matrix from the result, so reverting is exact rather
than subtractive and two scenarios made from one seed stay independent. Each
change writes a single audit row — the incident edge, the scenario's own
timestamp, and the operator's word for what happened — to the traffic log.

Pricing also writes the roads it touched to the traffic log as a side effect, so
collection needs no separate step. Road routes are traced under the same weights
the matrix was built with — drawing a peak-priced solution on static weights
would render a road the optimizer never chose, and the same is true of a solution
priced before an incident against one priced after.

Anomaly detection
-----------------
``POST /scenarios/{scenario_id}/detect`` is the one route that reads the collected
log back. Each scenario carries a snapshot of per-road, per-condition history
taken when it was created, and a newly observed travel time is scored against it:
a z-score where there is enough history for a standard deviation, and the
simulator's own expectation with a fixed margin where there is not. The rules and
their constants live in :mod:`qgati.traffic.detection`.

It flags and stops. Whether a flag should trigger a re-optimization, and what
that re-optimization should be allowed to change, is :mod:`qgati.reopt` — a later
phase — so that this route cannot move a vehicle as a side effect of reading a
number.

Simulated fleet
---------------
Everything above that consumes fleet telemetry had nothing producing any. The
``/watcher`` routes are the stand-in: a background thread that places each of a
scenario's vehicles on the road it is currently driving, draws a plausible
travel time for it, and does the two things a real GPS ping would do with it —
judge it against that road's history, and apply it as a **measured** time.

Applying it is Tier 1 of the two-tier design in ``DESIGN_DECISIONS.md``: a
measurement **overwrites** the rule-based estimate for that road, which is what
retires the flat ``x2.9`` an incident applies. The incident's number is a
placeholder for a delay nobody has measured, so it prices a road only until
something does. A closure still outranks both, because passability is not an
estimate of speed.

Each tick re-prices the scenario, so a later ``POST /optimize`` searches under
what the fleet actually saw, and writes one log row per measured road — the
ingestion phase the ``/detect`` route deferred. The rows are stamped with the
*measured* seconds, and with the incident word when one was in force, which is
what keeps an incident-time reading out of a later baseline.

**It is a simulation, not fleet integration.** There is no GPS device and no
vehicle; the pings come from the same traffic model the rest of the app prices
with, plus lognormal noise. The noise is not decoration — without it every
reading of one road in one condition band is the same number, which is the state
the log is already in and the reason its standard deviation is zero.
"""

from __future__ import annotations

import functools
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Annotated, Mapping, Sequence

import networkx as nx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from qgati.api.schemas import (
    AnalyticsSummary,
    AvoidRoadRequest,
    CompareResponse,
    ConditionsIn,
    ConditionsOut,
    DeliveryOut,
    DepotOut,
    DetectionVerdict,
    EdgeOut,
    EtaSummary,
    ExplanationOut,
    FigureOut,
    IncidentCreateRequest,
    IncidentOut,
    IncidentResponse,
    MoveOut,
    ObservationRequest,
    OptimizeRequest,
    OptimizeResponse,
    ReoptimizeRequest,
    ReoptimizeResponse,
    RoadReportOut,
    RouteExplanationOut,
    RouteOut,
    RunEntry,
    RunPage,
    ScenarioCreateRequest,
    ScenarioResponse,
    ScenarioSummary,
    SolverResultOut,
    StatementOut,
    StopOut,
    TrafficLogEntry,
    TrafficLogPage,
    TriggerOut,
    VehicleOut,
    VehicleProgressOut,
    VehicleTrackOut,
    WatcherReadingOut,
    WatcherStartRequest,
    WatcherStartResponse,
    WatcherStatus,
    WatcherTick,
)
from qgati.api.store import (
    ScenarioChanged,
    ScenarioNotFound,
    ScenarioStore,
    StoredScenario,
)
from qgati.analytics import RunLogRow, RunLogStore, RunSummary
from qgati.fleet import (
    Reading,
    ScenarioWatcher,
    TickResult,
    WatcherExists,
    WatcherRegistry,
    WatcherState,
    cost_lookup,
    initial_tracks,
)
from qgati.graph import (
    DEFAULT_PADDING_M,
    bbox_around_nodes,
    graph_to_geojson,
    load_delhi_graph,
    nearest_node,
    parse_bbox,
    route_polyline,
)
from qgati.graph.cost_matrix import CostMatrix, build_cost_matrix, changed_entries
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
from qgati.optimizer.registry import SolverSpec
from qgati.reopt import (
    NothingToReplan,
    ReoptPlan,
    RoadDelay,
    SolverTooSmall,
    Trigger,
    before_view,
    detect_trigger,
    explain_replan,
    fleet_progress,
    override_trigger,
    reoptimize,
    restart_node,
)
from qgati.routing.dijkstra import WeightFn
from qgati.traffic import (
    CLEARED,
    CLOSURE,
    FALLBACK_FACTOR,
    Z_SCORE,
    Z_THRESHOLD,
    ActiveConditions,
    Baseline,
    Incident,
    TrafficLogRow,
    TrafficLogStore,
    TrafficState,
    build_baseline,
    congestion_state,
    detect,
    edge_of,
    price_scenario,
    simulated_travel_time,
    traffic_weight_function,
)

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

#: Page size for the traffic log, and the ceiling a client may ask for. The
#: default matches the store's, so an unfiltered inspection returns a screenful.
DEFAULT_LOG_PAGE = 100
MAX_LOG_PAGE = 1000

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


@functools.lru_cache(maxsize=1)
def _open_log_store() -> TrafficLogStore:
    """The process's traffic log, opened on first use.

    Cached rather than constructed per request: the store holds one SQLite
    connection behind a lock, and opening a fresh one for every scenario would
    be pure overhead. Lazy, like the graph, so importing the app touches no
    files.
    """
    return TrafficLogStore()


def get_log_store() -> TrafficLogStore:
    """The traffic log, as an overridable dependency."""
    return _open_log_store()


@functools.lru_cache(maxsize=1)
def _open_run_store() -> RunLogStore:
    """The process's run history, opened on first use.

    Lazy for the same reason the graph and the traffic log are: importing the app
    should touch no files. Cached for the same reason too — one SQLite connection
    behind a lock, not a fresh one per request.
    """
    return RunLogStore()


def get_run_store() -> RunLogStore:
    """The run history, as an overridable dependency.

    A test overrides this to point at an in-memory store. That is not tidiness:
    the solve endpoints write here unconditionally, so a suite without the override
    would fill the developer's real history with test rows — and the History tab
    would open onto them.
    """
    return _open_run_store()


#: Every simulated fleet this process is running, keyed by scenario id. Module
#: level for the same reason the scenario store is: FastAPI resolves a
#: dependency per request, so a registry built inside the dependency would hand
#: every request a fresh, empty one and no watcher would ever be found twice.
WATCHERS = WatcherRegistry()


def get_watchers() -> WatcherRegistry:
    """The fleet registry, as an overridable dependency.

    A test overrides this so a watcher it starts cannot tick against another
    test's store, and so no background thread outlives the suite.
    """
    return WATCHERS


GraphDep = Annotated[nx.Graph, Depends(get_graph)]
StoreDep = Annotated[ScenarioStore, Depends(get_store)]
LogStoreDep = Annotated[TrafficLogStore, Depends(get_log_store)]
RunStoreDep = Annotated[RunLogStore, Depends(get_run_store)]
WatchersDep = Annotated[WatcherRegistry, Depends(get_watchers)]


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


def _json_safe(value):
    """A copy of ``value`` with any non-finite float replaced by ``None``.

    Pydantic echoes the offending value back in a validation error, so an
    observation of ``1e400`` yields an error body containing ``{"input": inf}``.
    Starlette renders every response with ``json.dumps(..., allow_nan=False)``,
    which raises on an infinity — so the 422 raised *because* the value was
    refused could not itself be written, and surfaced as a 500 instead. A
    rejection has to be deliverable.

    ``null`` is what JavaScript's ``JSON.stringify`` does with a non-finite
    number, and the ``msg`` beside it ("Input should be a finite number") already
    says what was wrong, so nothing a client can act on is lost.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _not_found(scenario_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"no scenario with id {scenario_id!r}",
    )


def _edge_pairs(pairs) -> list[EdgeOut]:
    """Edge pairs as response models, in a stable order.

    Sorted by their text form because node ids may be ints or strings and the two
    are not orderable together. Stable output keeps responses diffable.
    """
    return [
        EdgeOut(u=u, v=v)
        for u, v in sorted(pairs, key=lambda pair: (str(pair[0]), str(pair[1])))
    ]


def _incident_out(incident: Incident) -> IncidentOut:
    """A stored incident as the response model."""
    return IncidentOut(
        incident_id=incident.incident_id,
        incident_type=incident.incident_type,
        edge=EdgeOut(u=incident.u, v=incident.v),
        created_at=datetime.fromisoformat(incident.created_at),
    )


def _conditions_out(record: StoredScenario) -> ConditionsOut:
    """A scenario's *current* conditions, derived fields included.

    Takes the record rather than a bare state because it has two things to say
    beyond the state itself: whether the conditions have been mutated, and the
    live reports behind any mutation, each with the id needed to revert it.

    The state rendered is the effective one — the conditions the scenario's
    costs are actually built under — not the creation-time state the record also
    carries. With no incidents the two are the same value.
    """
    state = record.effective_traffic_state()
    return ConditionsOut(
        timestamp=state.timestamp,
        peak_hour=state.peak_hour,
        traffic_condition=state.traffic_condition,
        accident_edges=_edge_pairs(state.conditions.accident_edges),
        closed_edges=_edge_pairs(state.conditions.closed_edges),
        mutated=record.conditions_mutated,
        incidents=[_incident_out(incident) for incident in record.incidents],
    )


def _changed_legs(before: CostMatrix, after: CostMatrix) -> int:
    """How many entries of the objective matrix moved between two pricings.

    A thin alias for :func:`~qgati.graph.cost_matrix.changed_entries`, which is
    where the comparison lives now that a fleet tick needs the same number for
    the same reason. See that function for why the objective is what is counted
    and why the comparison is ``isclose`` rather than ``==``.
    """
    return changed_entries(before, after)


def _log_incident(
    graph: nx.Graph,
    log_store: TrafficLogStore,
    state: TrafficState,
    incident: Incident,
    word: str,
) -> int:
    """Write the one audit row a change to an incident is worth.

    The roads a scenario touches were logged when it was priced, so re-logging
    the whole network on every incident would add a hundred near-duplicate rows
    to answer a question about one road. What is genuinely new is the incident
    edge itself, and it is logged under the scenario's own timestamp — making
    the row a valid observation of the conditions this scenario is priced under,
    and one that pairs with the rows already collected for it.

    The row's ``incident_type`` is the operator's word — ``"closure"``,
    ``"slow"``, or ``"road_clear"`` for a revert — rather than the effect name
    :meth:`TrafficState.incident_type` would report. That column is an audit
    trail of what was reported and by whom it was resolved, and a road reported
    slow is not an accident. The travel time on the row still comes from
    :func:`~qgati.traffic.simulator.simulated_travel_time`, so the seconds it
    claims are provably the seconds routing charges.

    Best-effort, exactly like :func:`~qgati.traffic.recorder.price_scenario`:
    collecting data must never fail the request that is serving the user now.
    """
    rows = [
        replace(row, incident_type=word)
        for row in TrafficLogRow.from_edges(graph, [incident.edge], state)
    ]
    try:
        return log_store.write(rows)
    except Exception:  # noqa: BLE001 - deliberate; see the docstring
        LOGGER.exception("failed to write the incident audit row; continuing")
        return 0


def _record_run(
    run_store: RunLogStore,
    *,
    kind: str,
    record: StoredScenario,
    solver: SolverSpec,
    evaluation: Evaluation,
    runtime_ms: float,
    seed: int | None = None,
    iterations: int | None = None,
    population: int | None = None,
    trigger: str | None = None,
    trigger_detail: str | None = None,
    affected_vehicle: str | None = None,
    old_eta_seconds: float | None = None,
    moved: int = 0,
) -> int:
    """Write the one history row a solve is worth, and never fail the solve for it.

    Every figure passed in is one the solve produced — the same evaluation, the
    same measured runtime — rather than anything recomputed here. That is the whole
    point of the table: a row that could disagree with what the caller was told
    would be worse than no row.

    For three of the four kinds what is passed is literally the response's own
    figures. ``avoid_road`` is the one exception, and it is deliberate. That
    response reports ``plan.evaluation``, which covers only the vehicle that
    reported, while the row is written from the fleet-sized ``after_evaluation``
    its ``after`` array sums to. Pricing the row over one vehicle would make its
    ``travel_time`` incomparable with its own ``old_eta_seconds``, which is the whole
    fleet's — a saving that was an artefact of scope. So that row is fleet-scoped
    throughout, and its ``cost``/``travel_cost``/``travel_time``/``distance_m``/
    ``fuel_litres`` are the ``after`` total rather than the response's top-level one.

    ``old_eta_seconds`` is the one field that is not always available, and it is
    honestly absent rather than defaulted. An ``optimize`` or a ``dispatch`` is the
    first answer for its scenario, so there is no earlier plan to measure against;
    writing ``0.0`` would say the previous plan was instantaneous.

    Best-effort, exactly like :func:`~qgati.traffic.recorder.price_scenario` and
    :func:`_log_incident`: record-keeping must never cost the user the plan they
    just asked for. A failure is logged and the return is ``0``, which no caller
    branches on.

    Returns the row's id, or ``0`` if the write failed.
    """
    row = RunLogRow(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        kind=kind,
        scenario_id=record.scenario_id,
        n_deliveries=record.scenario.n_deliveries,
        n_vehicles=len(record.scenario.vehicles),
        solver=solver.key,
        solver_name=solver.name,
        seed=seed,
        iterations=iterations if solver.is_stochastic else None,
        population=population if solver.is_stochastic else None,
        trigger=trigger,
        trigger_detail=trigger_detail,
        affected_vehicle=affected_vehicle,
        cost=evaluation.fitness,
        travel_cost=evaluation.travel_cost,
        travel_time=evaluation.travel_time,
        distance_m=evaluation.distance,
        fuel_litres=evaluation.fuel,
        feasible=evaluation.feasible,
        runtime_ms=runtime_ms,
        old_eta_seconds=old_eta_seconds,
        moved=moved,
    )
    try:
        return run_store.write(row)
    except Exception:  # noqa: BLE001 - deliberate; see the docstring
        LOGGER.exception("failed to write the run history row; continuing")
        return 0


def _summary_out(summary: RunSummary) -> AnalyticsSummary:
    """The store's aggregate, as the endpoint's response model.

    A translation rather than a passthrough because the store has no web
    dependency and the schema does; the two shapes are otherwise identical.
    """
    return AnalyticsSummary(
        total_runs=summary.total_runs,
        first_run_at=summary.first_run_at,
        last_run_at=summary.last_run_at,
        runs_by_kind=summary.runs_by_kind,
        runs_by_solver=summary.runs_by_solver,
        feasible_runs=summary.feasible_runs,
        infeasible_runs=summary.infeasible_runs,
        incident_triggered_runs=summary.incident_triggered_runs,
        avg_runtime_ms=summary.avg_runtime_ms,
        min_runtime_ms=summary.min_runtime_ms,
        max_runtime_ms=summary.max_runtime_ms,
        avg_cost=summary.avg_cost,
        avg_travel_cost=summary.avg_travel_cost,
        avg_travel_time_seconds=summary.avg_travel_time_seconds,
        eta=EtaSummary(
            runs=summary.eta_runs,
            avg_saved_seconds=summary.avg_saved_seconds,
            total_saved_seconds=summary.total_saved_seconds,
            improved_runs=summary.improved_runs,
            worsened_runs=summary.worsened_runs,
            unchanged_runs=summary.unchanged_runs,
        ),
    )


def _change_incident(
    *,
    graph: nx.Graph,
    store: ScenarioStore,
    log_store: TrafficLogStore,
    record: StoredScenario,
    incidents: tuple[Incident, ...],
    incident: Incident,
    word: str,
    applied: bool,
) -> IncidentResponse:
    """Apply or revert one incident, re-price, log the change, and store it.

    Shared by both routes because they differ in exactly three things: the
    incident tuple they end up with, whether the change is an application or a
    revert, and the word the audit row carries. Everything else — the re-price,
    the rollback when the instance is severed, the count of legs that moved, the
    log write and the compare-and-swap — is identical, and duplicating it would
    mean two places to get the rollback wrong.

    Nothing here runs a solver. An incident moves the *costs*; the routes a
    client already holds stay where they are until it asks for a new solution.
    """
    # Build the prospective record first, so the state is derived by the same
    # method the rest of the app uses rather than by a second copy of the fold.
    candidate = replace(record, incidents=incidents)
    state = candidate.effective_traffic_state()

    # Priced with logging suppressed: `_log_incident` writes the rows this
    # change is actually about, and the network's rows were collected already.
    try:
        priced = price_scenario(graph, record.scenario, state, log_store=None)
    except ValueError as error:
        # A closure can leave a stop unreachable. Refusing is the same answer
        # creation gives the same closure, and nothing is stored.
        detail = str(error)
        if state.conditions.closed_edges:
            detail += (
                f" This change was not applied: it would leave "
                f"{len(state.conditions.closed_edges)} closure(s) in place, "
                "which can sever a route that exists in the static network."
            )
        raise HTTPException(status_code=UNPROCESSABLE, detail=detail) from None

    changed_legs = _changed_legs(record.cost_matrix, priced.cost_matrix)
    rows_logged = _log_incident(graph, log_store, state, incident, word)

    updated = replace(
        candidate,
        cost_matrix=priced.cost_matrix,
        traffic_rows_logged=record.traffic_rows_logged + rows_logged,
    )
    try:
        store.replace(updated, previous=record)
    except ScenarioChanged:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"scenario {record.scenario_id!r} was modified while this change "
                "was being applied; re-read it and retry"
            ),
        ) from None

    return IncidentResponse(
        scenario_id=record.scenario_id,
        incident=_incident_out(incident),
        applied=applied,
        conditions=_conditions_out(updated),
        changed_legs=changed_legs,
        traffic_rows_logged=rows_logged,
    )


def _baseline(log_store: TrafficLogStore) -> Baseline:
    """Snapshot the log's per-road, per-condition history, once.

    Read in a single pass over the whole table rather than a query per road: the
    detector may be asked about any road a scenario touches, and asking SQLite for
    one road at a time would mean a query per candidate. At the scale this demo
    collects — thousands of rows — one pass with a ``GROUP BY`` in Python is both
    simpler and fast enough.

    Called once, when a scenario is created, and the result is stored with it.
    That is the documented simplification against a production system's daily
    batch refresh: the numbers behind a verdict are fixed for the scenario's
    lifetime, so two observations on one scenario are judged against one history.
    """
    rows, total = log_store.read(limit=MAX_LOG_PAGE, offset=0)
    # `read` pages, so a log larger than one page would silently contribute only
    # its newest rows. Walk the rest rather than pretending the cap is the whole
    # table — a baseline that quietly drops history would be worse than a slow one.
    collected = list(rows)
    while len(collected) < total:
        page, _ = log_store.read(limit=MAX_LOG_PAGE, offset=len(collected))
        if not page:
            break
        collected.extend(page)
    return build_baseline(collected)


def _node_filter(value: str | None) -> int | str | None:
    """Interpret a road-endpoint query parameter as a node id.

    Query strings arrive as text, but node ids are integers on the Delhi graph.
    SQLite compares across storage classes without coercing — integer always
    sorts before text, never equal to it — so a filter bound as ``"249782331"``
    would silently match no integer-keyed row. A parameter that reads as an
    integer is therefore bound as one, and anything else is passed through as the
    string id a differently-keyed graph would use.
    """
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _traffic_state(graph: nx.Graph, conditions: ConditionsIn | None) -> TrafficState:
    """Turn a request's conditions into a state, checking every named edge.

    An incident naming a road that is not in the graph is a client error rather
    than something to ignore: silently dropping it would price the scenario under
    conditions the caller did not ask for, and report success.

    Omitting the block means "now, no incidents" — so the plain request still
    produces a scenario with a real timestamp, a congestion state derived from
    it, and a log write.
    """
    if conditions is None:
        return TrafficState.now()

    absent = [
        (edge.u, edge.v)
        for edge in (*conditions.accident_edges, *conditions.closed_edges)
        if not graph.has_edge(edge.u, edge.v)
    ]
    if absent:
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail=(
                f"{len(absent)} incident edge(s) are not in the road graph: "
                f"{absent[:5]}" + (" ..." if len(absent) > 5 else "")
            ),
        )

    return TrafficState(
        timestamp=conditions.timestamp or datetime.now().astimezone(),
        conditions=ActiveConditions(
            accident_edges=frozenset((e.u, e.v) for e in conditions.accident_edges),
            closed_edges=frozenset((e.u, e.v) for e in conditions.closed_edges),
        ),
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
        conditions=_conditions_out(record),
        traffic_rows_logged=record.traffic_rows_logged,
    )


def _routes_out(
    record,
    solution: Solution,
    evaluation: Evaluation,
    graph: nx.Graph,
    include_geometry: bool = True,
    weight: str | WeightFn = "weight",
    start_nodes: Sequence[object] | None = None,
) -> list[RouteOut]:
    """Resolve a solution's delivery indices into ids, nodes, loads and costs.

    ``Solution`` holds *positions* in the scenario's delivery list, which mean
    nothing to a client; this is where they become delivery ids and graph nodes.

    With ``include_geometry``, each route is also traced along the road network.
    The depot is prepended and appended by this function because a
    :class:`Solution` route lists only its stops — the tour's return leg is
    implicit until something has to draw it.

    ``weight`` must be the weight the cost matrix was built with. For a scenario
    priced under traffic conditions that is the traffic weight function, and
    passing the static default instead would trace each leg along the *untraffic'd*
    shortest path — drawing a road the optimizer never chose, at a cost that does
    not match the one it minimised.

    ``start_nodes`` is how a **re-optimization** is drawn: one node per vehicle,
    in vehicle order, saying where that vehicle begins. It is ``None`` for every
    ordinary plan, where the answer is the depot for all of them and prepending it
    is the whole story. Where it is given, the depot is still the *end* of every
    route — a re-planned vehicle comes home even though it did not leave from
    there — and an entry that is ``None`` means the vehicle is not on a road at
    all, so its route is drawn without a line rather than from a position nobody
    knows.
    """
    scenario = record.scenario
    depot_node = scenario.depot.node
    routes: list[RouteOut] = []

    for position, route in enumerate(solution.routes):
        vehicle = scenario.vehicles[position]
        stop_nodes = [scenario.deliveries[index].node for index in route]

        # A route with no known start — a vehicle stopped behind a closure — is
        # reported like an unused one: the stops are real, the line is not.
        start = depot_node if start_nodes is None else start_nodes[position]
        drawable = include_geometry and stop_nodes and start is not None

        # An unused vehicle has no tour to draw; emitting the depot twice would
        # render as a one-point line rather than as nothing.
        geometry = (
            route_polyline(graph, [start, *stop_nodes, depot_node], weight=weight)
            if drawable
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
                travel_time=evaluation.route_times[position],
                distance_m=evaluation.route_distances[position],
                fuel_litres=evaluation.route_fuels[position],
                geometry=geometry,
            )
        )
    return routes


def _road_delays(
    record: StoredScenario, graph: nx.Graph, trigger: Trigger
) -> tuple[RoadDelay, ...]:
    """Every road the trigger names, priced before the report and after it.

    The "before" half has to be a network *without* this report, and the record
    keeps exactly that: :attr:`~qgati.api.store.StoredScenario.traffic_state` is
    the state the scenario was created with, while
    :meth:`~qgati.api.store.StoredScenario.effective_traffic_state` layers the live
    incidents and the fleet's measurements on top of it.

    Only the *conditions* are rolled back, and the current timestamp and
    measurements are kept. That is what makes the difference between the two
    numbers attributable to the incident alone: a road the fleet has also driven
    is priced with the same measurement in both readings, so the seconds that
    change are the seconds the report changed.

    A road that is closed reads ``None`` on the after side rather than a large
    number, which is what stops anything downstream from quoting a travel time
    for a road that has none. Failures are swallowed per road — a road missing
    from the graph is a reason to leave it out of the trace, not a reason to fail
    a re-plan that has already been solved.
    """
    if not trigger.edges:
        return ()

    current = record.effective_traffic_state()
    # Same instant, same measurements, creation-time conditions: the network as it
    # was before any of the live reports existed.
    before = replace(current, conditions=record.traffic_state.conditions)

    delays: list[RoadDelay] = []
    for u, v in trigger.edges:
        try:
            delays.append(
                RoadDelay(
                    u=u,
                    v=v,
                    before_seconds=simulated_travel_time(graph, (u, v), before),
                    after_seconds=simulated_travel_time(graph, (u, v), current),
                )
            )
        except (KeyError, ValueError):  # pragma: no cover - defensive
            continue
    return tuple(delays)


def _explanation_out(
    plan: ReoptPlan,
    before_evaluation: Evaluation,
    trigger: Trigger,
    delays: Sequence[RoadDelay],
    before_positions: Mapping[str, int] | None = None,
) -> ExplanationOut:
    """The decision trace for a re-plan, in the response's own vocabulary.

    A pure translation of :func:`~qgati.reopt.explain.explain_replan` — every
    sentence and every figure is built there, and this only renames the fields.
    Keeping the two apart is what lets the explanation be tested on a hand-built
    plan without an HTTP client, a store or a graph.

    ``before_positions`` is passed through for the one caller whose before half is
    priced over a wider fleet than the plan solved; see ``explain_replan``.
    """
    explanation = explain_replan(
        plan,
        before_evaluation,
        trigger,
        road_delays=delays,
        before_positions=before_positions,
    )
    return ExplanationOut(
        headline=explanation.headline,
        roads=[
            RoadReportOut(
                u=road.u,
                v=road.v,
                statements=[
                    StatementOut(
                        text=statement.text,
                        figures=[
                            FigureOut(
                                label=figure.label,
                                value=figure.value,
                                unit=figure.unit,
                            )
                            for figure in statement.figures
                        ],
                    )
                    for statement in road.statements
                ],
            )
            for road in explanation.roads
        ],
        routes=[
            RouteExplanationOut(
                vehicle_id=route.vehicle_id,
                headline=route.headline,
                statements=[
                    StatementOut(
                        text=statement.text,
                        figures=[
                            FigureOut(
                                label=figure.label,
                                value=figure.value,
                                unit=figure.unit,
                            )
                            for figure in statement.figures
                        ],
                    )
                    for statement in route.statements
                ],
            )
            for route in explanation.routes
        ],
    )


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def _pick_solver(record: StoredScenario, solver: str | None) -> SolverSpec:
    """Resolve a solver key for one scenario, or refuse it with a 422.

    Shared by ``POST /optimize`` and the fleet's start route, which are the two
    places a solver is chosen for a stored scenario and which must give the same
    answer to the same request: an unknown key, or an exact solver on an
    instance past its limit, is a client error rather than something to fall back
    from.
    """
    try:
        spec = get_solver(solver or DEFAULT_SOLVER_KEY)
    except KeyError as error:
        raise HTTPException(status_code=UNPROCESSABLE, detail=str(error)) from None

    if spec.is_exact and record.scenario.n_deliveries > (spec.exact_limit or 0):
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail=(
                f"{spec.name} is limited to {spec.exact_limit} deliveries; this "
                f"scenario has {record.scenario.n_deliveries}. Use a heuristic."
            ),
        )
    return spec


def _watcher_or_404(watchers: WatcherRegistry, scenario_id: str) -> ScenarioWatcher:
    """This scenario's running fleet, or a 404 naming the route that starts one.

    A 404 rather than a 200 with ``running: false``, because the watcher is the
    resource the URL names and there is none: the same answer
    ``DELETE /incident/{id}`` gives for an incident that is not there. A scenario
    that does not exist at all is indistinguishable from here, which is fine —
    either way there is no fleet to report on.
    """
    try:
        return watchers.get(scenario_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"scenario {scenario_id!r} has no fleet running; start one with "
                f"POST /scenarios/{scenario_id}/watcher"
            ),
        ) from None


def _status_out(state: WatcherState) -> WatcherStatus:
    """Internal watcher state -> the JSON shape ``GET`` and ``DELETE`` return."""
    return WatcherStatus(
        scenario_id=state.scenario_id,
        solver=state.solver,
        running=state.running,
        interval_seconds=state.interval_seconds,
        time_scale=state.time_scale,
        seed=state.seed,
        ticks=state.ticks,
        started_at=state.started_at,
        last_tick_at=state.last_tick_at,
        vehicles=[
            VehicleTrackOut(
                vehicle_id=vehicle.vehicle_id,
                edge=_edge_out(vehicle.edge),
                remaining_seconds=vehicle.remaining_seconds,
                progress=vehicle.progress,
                finished=vehicle.finished,
            )
            for vehicle in state.vehicles
        ],
    )


def _reading_out(reading: Reading) -> WatcherReadingOut:
    """One ping -> one row of a tick's response."""
    verdict = reading.verdict
    return WatcherReadingOut(
        vehicle_id=reading.vehicle_id,
        edge=_edge_out(reading.edge),
        modelled_travel_time=reading.modelled,
        observed_travel_time=reading.observed,
        flag=False if verdict is None else verdict.flagged,
        rule=None if verdict is None else verdict.rule,
        condition=None if verdict is None else verdict.condition,
        sample_count=0 if verdict is None else verdict.sample_count,
        z_score=None if verdict is None else verdict.z_score,
        reason=None if verdict is None else verdict.reason,
        note=reading.note,
    )


def _edge_out(edge: tuple | None) -> EdgeOut | None:
    """A bare ``(u, v)`` pair as the API's edge shape, or ``None``."""
    return None if edge is None else EdgeOut(u=edge[0], v=edge[1])


def _tick_out(result: TickResult) -> WatcherTick:
    """One tick -> the JSON shape the tick route returns.

    Deliberately no route, cost or solver field: see :class:`WatcherTick`.
    """
    return WatcherTick(
        scenario_id=result.scenario_id,
        tick=result.tick,
        at=result.at,
        readings=[_reading_out(reading) for reading in result.readings],
        changed_legs=result.changed_legs,
        rows_logged=result.rows_logged,
        conditions_mutated=result.conditions_mutated,
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Stop every fleet on the way out.

        A watcher is a daemon thread holding a reference to the log store: left
        running it would outlive the process's own shutdown and, in a test, keep
        ticking against a store the fixture has already closed. Daemon threads
        are killed at interpreter exit, which is not the same thing as being
        stopped cleanly, so this is where that happens.

        The override is honoured so that a test's own registry is the one
        stopped. ``dependency_overrides`` is the only place the app knows about
        it — a lifespan runs outside any request and cannot ask FastAPI to
        resolve a dependency.
        """
        yield
        registry = application.dependency_overrides.get(get_watchers, get_watchers)()
        stopped = registry.stop_all()
        if stopped:
            LOGGER.info("stopped %d fleet watcher(s) on shutdown", stopped)

    application = FastAPI(
        title="Q-Gati API",
        version="0.1.0",
        lifespan=lifespan,
        description=(
            "Multi-algorithm vehicle routing for Delhi. Five solvers behind one "
            "contract; QPSO is the production default, the problem statement's "
            "focus algorithm, benchmarked against four conventional "
            "metaheuristics and exact ground truth."
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

    @application.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """FastAPI's own 422 body, with any non-finite input made writable.

        Identical to FastAPI's default handler except for the ``_json_safe``
        pass. The default echoes the offending value back — and a value that
        failed a *finite number* check is by definition one ``json.dumps`` with
        ``allow_nan=False`` refuses to write, so the response fails to render and
        the client gets a 500 where the validation produced a 422. ``1e400`` is
        the case that reaches this: a valid JSON number literal that any
        conforming client may send, which overflows to infinity on parse.
        """
        return JSONResponse(
            status_code=UNPROCESSABLE,
            content=_json_safe({"detail": jsonable_encoder(exc.errors())}),
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
        request: ScenarioCreateRequest,
        graph: GraphDep,
        store: StoreDep,
        log_store: LogStoreDep,
    ) -> ScenarioResponse:
        """Create a scenario, generated at random or described explicitly.

        The instance is priced under the requested traffic conditions (or the
        system clock, if none were given), and the roads that pricing touched are
        appended to the traffic log. Both happen here so that collecting the
        dataset is a side effect of ordinary use rather than a separate step
        somebody has to remember to run.
        """
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

        state = _traffic_state(graph, request.conditions)

        # Snapshot the log's history *before* pricing this scenario, because
        # pricing appends a row per road it touched. Read afterwards, a scenario
        # would contribute to the very history it is later judged against — and
        # it would do so with the reading it is supposed to be an exception to.
        baseline = _baseline(log_store)

        try:
            priced = price_scenario(graph, scenario, state, log_store)
        except ValueError as error:
            # Either a node missing from the graph, or a pair unreachable *under
            # these conditions*. The builder refuses to substitute inf, because
            # that would let an optimizer return a confidently wrong answer; the
            # message tells the client to use the servable subgraph.
            detail = str(error)
            if state.conditions.closed_edges:
                detail += (
                    f" This scenario was priced with "
                    f"{len(state.conditions.closed_edges)} closure(s), which can "
                    "sever a route that exists in the static network."
                )
            raise HTTPException(
                status_code=UNPROCESSABLE, detail=detail
            ) from None

        return _scenario_response(
            store.add(
                scenario,
                priced.cost_matrix,
                state,
                priced.rows_logged,
                baseline=baseline,
            )
        )

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

    @application.post(
        "/scenarios/{scenario_id}/incident",
        response_model=IncidentResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["scenarios"],
    )
    def create_incident(
        scenario_id: str,
        payload: IncidentCreateRequest,
        graph: GraphDep,
        store: StoreDep,
        log_store: LogStoreDep,
    ) -> IncidentResponse:
        """Inject a live incident into a stored scenario.

        The road is re-priced into the scenario's cost matrix and the change is
        written to the traffic log, but **nothing is re-optimized**. Routes a
        client already holds are not moved by an incident — only the costs the
        next ``POST /optimize/{scenario_id}`` searches under. That separation is
        the point: an operator's report should not silently dispatch vehicles.

        Conditions are mutable per scenario, so the incident applies to this
        ``scenario_id`` alone and disappears with it. Creating two scenarios from
        one seed and injecting into one leaves the other untouched.

        A closure that severs the instance — leaving a stop unreachable — is
        refused with 422 and changes nothing, which is the same answer creation
        gives the same closure.
        """
        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        edge = payload.edge
        # Checked against the graph rather than the scenario's own nodes, because
        # an incident names a *road*: it may sit anywhere on a route the scenario
        # could take, including one no current cheapest path uses. A road that is
        # not in the graph at all is a client error, not something to ignore.
        if not graph.has_edge(edge.u, edge.v):
            raise HTTPException(
                status_code=UNPROCESSABLE,
                detail=(
                    f"incident edge: no road from {edge.u!r} to {edge.v!r} in the "
                    "road graph"
                ),
            )

        incident = Incident(
            incident_id=uuid.uuid4().hex,
            incident_type=payload.incident_type,
            u=edge.u,
            v=edge.v,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        return _change_incident(
            graph=graph,
            store=store,
            log_store=log_store,
            record=record,
            incidents=(*record.incidents, incident),
            incident=incident,
            word=incident.incident_type,
            applied=True,
        )

    @application.delete(
        "/scenarios/{scenario_id}/incident/{incident_id}",
        response_model=IncidentResponse,
        tags=["scenarios"],
    )
    def delete_incident(
        scenario_id: str,
        incident_id: str,
        graph: GraphDep,
        store: StoreDep,
        log_store: LogStoreDep,
    ) -> IncidentResponse:
        """Revert one live incident, restoring the conditions it replaced.

        Reverting is exact rather than subtractive. The incident is dropped from
        the scenario's reports and the conditions are re-folded from what the
        scenario was created with, so an accident or closure it was born with
        survives the round trip, and so does every other live incident. The cost
        matrix is rebuilt from those restored conditions.

        Like its counterpart this re-prices and stops — it does not re-optimize.
        """
        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        incident = next(
            (item for item in record.incidents if item.incident_id == incident_id),
            None,
        )
        if incident is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"scenario {scenario_id!r} has no live incident "
                    f"{incident_id!r}; it may already have been reverted"
                ),
            )

        return _change_incident(
            graph=graph,
            store=store,
            log_store=log_store,
            record=record,
            incidents=tuple(
                item for item in record.incidents if item.incident_id != incident_id
            ),
            incident=incident,
            word=CLEARED,
            applied=False,
        )

    @application.post(
        "/scenarios/{scenario_id}/detect",
        response_model=DetectionVerdict,
        tags=["traffic"],
    )
    def detect_anomaly(
        scenario_id: str,
        payload: ObservationRequest,
        graph: GraphDep,
        store: StoreDep,
    ) -> DetectionVerdict:
        """Judge one observed travel time against this scenario's history.

        Plain statistics, no model: the road's logged mean and standard deviation
        for the congestion band the reading falls in, and a z-score against them.
        Above ``Z_THRESHOLD`` standard deviations the reading is flagged. With
        fewer than ``MIN_SAMPLES`` samples behind the road there is no standard
        deviation worth dividing by, so the score is skipped entirely and the
        simulator's own expectation with a ``FALLBACK_FACTOR`` margin decides
        instead. All three are constants in :mod:`qgati.traffic.detection`, and the
        verdict says which rule ran.

        **Nothing is re-optimized.** A flag is a signal for a later phase to act
        on, not an action — the same boundary the incident routes draw. And
        nothing is written: the observation is judged and discarded. Recording it
        here would feed the anomaly back into the baseline that is supposed to
        catch it, so ingestion belongs to whatever phase actually receives fleet
        telemetry.

        The expectation is the road's cost under the clock alone. A live incident
        does not raise it, deliberately: if an operator's report could explain the
        slowness away, the fallback would be blind to the change it exists to
        catch, and the two rules would disagree about what "normal" means.
        """
        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        edge = payload.edge
        if not graph.has_edge(edge.u, edge.v):
            raise HTTPException(
                status_code=UNPROCESSABLE,
                detail=(
                    f"observation edge: no road from {edge.u!r} to {edge.v!r} in "
                    "the road graph"
                ),
            )

        # The band the reading is judged in. Defaults to the scenario's own
        # timestamp so an observation is compared against the conditions its costs
        # were built under, unless the caller says it saw the road at another hour.
        observed_at = payload.timestamp or record.traffic_state.timestamp
        condition = congestion_state(observed_at)

        # Conditions cleared — see the docstring. `edge_of` reads the road's real
        # class, which is what decides how much congestion it takes.
        expected = simulated_travel_time(
            graph,
            edge_of(graph, edge.u, edge.v),
            TrafficState(timestamp=observed_at, conditions=ActiveConditions()),
        )

        stats = (
            None
            if record.baseline is None
            else record.baseline.for_edge(edge.u, edge.v, condition)
        )
        verdict = detect(
            payload.travel_time, stats=stats, expected=expected, condition=condition
        )

        return DetectionVerdict(
            scenario_id=scenario_id,
            edge=EdgeOut(u=edge.u, v=edge.v),
            observed_travel_time=verdict.observed,
            condition=verdict.condition,
            rule=verdict.rule,
            flagged=verdict.flagged,
            expected_travel_time=verdict.expected,
            sample_count=verdict.sample_count,
            mean=verdict.mean,
            std_dev=verdict.std_dev,
            z_score=verdict.z_score,
            threshold=(
                Z_THRESHOLD if verdict.rule == Z_SCORE else FALLBACK_FACTOR
            ),
            reason=verdict.reason,
        )

    # -- fleet --------------------------------------------------------------- #
    @application.post(
        "/scenarios/{scenario_id}/watcher",
        response_model=WatcherStartResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["fleet"],
    )
    def start_watcher(
        scenario_id: str,
        graph: GraphDep,
        store: StoreDep,
        log_store: LogStoreDep,
        run_store: RunStoreDep,
        watchers: WatchersDep,
        payload: WatcherStartRequest | None = None,
    ) -> WatcherStartResponse:
        """Dispatch a simulated GPS fleet against a stored scenario.

        A scenario has no routes until something solves it, and this is what
        does: the named solver — the production default unless one is named —
        runs once, and the vehicles it puts on the road become the fleet the
        watcher drives.

        From then on the watcher measures. Every tick places each vehicle on the
        road it is currently on, draws a plausible travel time for it, judges it
        against that road's history, and folds it in as a **measured** time — the
        Tier-1 override that retires the flat ``x2.9`` placeholder an incident
        applies. The scenario's cost matrix is rebuilt from the result, so a
        later ``POST /optimize`` searches under what the fleet actually saw.

        The first tick runs inline, so this response already carries readings.
        The rest happen on a daemon thread at ``interval_seconds``, and are
        stopped by ``DELETE`` or by the app shutting down.

        A watcher already running on this scenario is a 409 rather than a second
        fleet: two would fight over the same record's compare-and-swap, and the
        loser's readings would be discarded tick after tick.
        """
        request = payload or WatcherStartRequest()

        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        # Checked twice on purpose. Here, so the common case fails before a
        # solver run is spent on it; and again at `watchers.add` below, which is
        # the atomic claim — two requests can both pass this line.
        if scenario_id in watchers:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"scenario {scenario_id!r} already has a fleet running; stop "
                    "it with DELETE on the same URL first"
                ),
            )

        spec = _pick_solver(record, request.solver)

        start = time.perf_counter()
        solution, _cost, convergence = spec(
            record.cost_matrix,
            record.scenario,
            seed=request.seed,
            population=request.population or DEFAULT_POPULATION,
            iterations=request.iterations or DEFAULT_ITERATIONS,
        )
        # Timed here rather than nowhere: dispatching a fleet *is* a solve, and it
        # is the one the dashboard's first plan comes from. Leaving it untimed
        # would put a zero in the history for the run a user is most likely to be
        # looking at.
        runtime_ms = (time.perf_counter() - start) * 1000.0
        evaluation = evaluate(solution, record.scenario, record.cost_matrix)

        # The corridor each vehicle drives is traced under the state that prices
        # the scenario *now* — the effective one, incidents included — so the
        # fleet starts spread along the roads it would really take, not along the
        # free-flow ones it would have taken without the incident.
        weight = traffic_weight_function(graph, record.effective_traffic_state())
        watcher = ScenarioWatcher(
            scenario_id=scenario_id,
            solver=spec.key,
            store=store,
            graph=graph,
            log_store=log_store,
            tracks=initial_tracks(record.scenario, solution, graph, weight),
            # The plan this fleet is driving. Handed over because a
            # re-optimization has to show what it changed *from*, and this route
            # is the only place the solution exists — `POST /optimize` discards
            # its own, and nothing else keeps the fleet's routes.
            solution=solution,
            interval_seconds=request.interval_seconds,
            time_scale=request.time_scale,
            seed=request.seed,
        )

        try:
            # Registered before the first tick, not after: `add` is the atomic
            # claim on the scenario, and two concurrent starts must not both get
            # as far as measuring.
            watchers.add(watcher)
        except WatcherExists:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"scenario {scenario_id!r} already has a fleet running; stop "
                    "it with DELETE on the same URL first"
                ),
            ) from None

        try:
            first = watcher.tick()
        except Exception:
            # Nothing was started and nothing should be left registered.
            watchers.remove(scenario_id)
            raise

        watcher.start()

        _record_run(
            run_store,
            kind="dispatch",
            record=record,
            solver=spec,
            evaluation=evaluation,
            runtime_ms=runtime_ms,
            seed=request.seed,
            iterations=request.iterations or DEFAULT_ITERATIONS,
            population=request.population or DEFAULT_POPULATION,
        )

        return WatcherStartResponse(
            **_status_out(watcher.status()).model_dump(),
            routes=_routes_out(
                record,
                solution,
                evaluation,
                graph,
                include_geometry=request.include_geometry,
                weight=weight,
            ),
            first_tick=_tick_out(first),
            convergence=list(convergence),
        )

    @application.get(
        "/scenarios/{scenario_id}/watcher",
        response_model=WatcherStatus,
        tags=["fleet"],
    )
    def watcher_status(scenario_id: str, watchers: WatchersDep) -> WatcherStatus:
        """What this scenario's fleet is doing, and where each vehicle is.

        Positions are resolved under the scenario's current costs, so a vehicle
        on a road that has just been measured as slow is reported further from
        the end of it than the model alone would have put it. Nothing here ticks:
        reading the fleet does not move it.
        """
        return _status_out(_watcher_or_404(watchers, scenario_id).status())

    @application.post(
        "/scenarios/{scenario_id}/watcher/tick",
        response_model=WatcherTick,
        tags=["fleet"],
    )
    def tick_watcher(scenario_id: str, watchers: WatchersDep) -> WatcherTick:
        """Advance this scenario's fleet by one interval, synchronously.

        The same function the timer thread calls, exposed so a client can step
        the fleet and read the result instead of waiting an interval to see what
        happened. Useful for a demo and for a test, and the reason no test in the
        suite has to sleep.

        It reads, measures, prices and writes — and returns no route, because
        acting on a fleet's readings is ``reopt``'s job, not this route's.
        """
        watcher = _watcher_or_404(watchers, scenario_id)
        try:
            return _tick_out(watcher.tick())
        except ScenarioNotFound:
            # The scenario was dropped between the lookup above and the tick.
            watchers.remove(scenario_id)
            raise _not_found(scenario_id) from None

    @application.delete(
        "/scenarios/{scenario_id}/watcher",
        response_model=WatcherStatus,
        tags=["fleet"],
    )
    def stop_watcher(scenario_id: str, watchers: WatchersDep) -> WatcherStatus:
        """Stop this scenario's fleet, returning its final state.

        Stops rather than destroys: the readings it took are already in the
        scenario's cost matrix and in the traffic log, and both survive it. Only
        the ticking stops, and a vehicle that was mid-route is reported where it
        stood rather than as finished.
        """
        watcher = _watcher_or_404(watchers, scenario_id)
        # Read the state before stopping, so the vehicles carry their real
        # positions; only `running` is then corrected.
        state = watcher.status()
        watchers.remove(scenario_id)
        return _status_out(replace(state, running=False))

    # -- re-optimization ---------------------------------------------------- #
    @application.post(
        "/scenarios/{scenario_id}/reoptimize",
        response_model=ReoptimizeResponse,
        tags=["reopt"],
    )
    def reoptimize_scenario(
        scenario_id: str,
        graph: GraphDep,
        store: StoreDep,
        run_store: RunStoreDep,
        watchers: WatchersDep,
        payload: ReoptimizeRequest | None = None,
    ) -> ReoptimizeResponse:
        """Re-plan only the unserved deliveries of a scenario whose fleet is out.

        Everything this needs is derived rather than declared. Where each vehicle
        is, which stops it has already served and how much capacity it has left
        are read from the **running fleet** — there is no body-declared fleet
        state, so a caller cannot assert a state the fleet is not in. Whether a
        re-optimization is justified at all is derived too: a live incident, or a
        reading the detector flagged on the fleet's most recent tick. Neither
        holding is a 409.

        Completed stops are safe because they are not in the instance that is
        solved, not because something removes them from an answer. The new plan
        is produced by the same QPSO, through the same objective and the same
        window handling, as an ordinary ``/optimize`` — the only difference is the
        scenario it is handed, which holds the remaining stops, a reduced capacity
        per vehicle, and where each vehicle starts from.

        **This is a read.** It writes nothing: the stored scenario is untouched and
        the fleet keeps driving the routes it was already on. Applying a plan —
        rebuilding tracks and handing the new routes to the vehicles — is a
        separate step and deliberately not this route's.
        """
        request = payload or ReoptimizeRequest()

        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        watcher = _watcher_or_404(watchers, scenario_id)

        trigger = detect_trigger(record, watcher)
        if trigger is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"nothing has happened to scenario {scenario_id!r} that "
                    "warrants re-planning: there is no live incident, and the "
                    "fleet's most recent tick flagged no anomalous reading. "
                    "Inject one with POST /scenarios/{id}/incident, or step the "
                    "fleet with POST /scenarios/{id}/watcher/tick."
                ),
            )

        # Resolved here rather than through `_pick_solver`, which measures the
        # **stored** scenario against an exact solver's limit. That is the wrong
        # count for a re-optimization: a ten-delivery scenario with three stops
        # left is a three-delivery instance, and brute force can close it. The key
        # is validated here and the limit inside `reoptimize`, against what
        # actually remains.
        try:
            spec = get_solver(request.solver or DEFAULT_SOLVER_KEY)
        except KeyError as error:
            raise HTTPException(status_code=UNPROCESSABLE, detail=str(error)) from None

        # The scenario's own live pricing, so a vehicle's position is resolved
        # under the same costs the optimizer is charged — a vehicle on a road an
        # incident has just slowed is placed further from the end of it.
        weight = traffic_weight_function(graph, record.effective_traffic_state())
        lookup = cost_lookup(graph, weight)

        # Read under the tick lock: a tick advances every track as it works, and
        # catching the fleet mid-step would report a vehicle between two roads.
        # The lock is not held across the solve below.
        progress = watcher.read(
            lambda tracks: fleet_progress(record.scenario, tracks, lookup)
        )

        try:
            plan = reoptimize(
                scenario=record.scenario,
                progress=progress,
                graph=graph,
                weight=weight,
                solver_key=spec.key,
                seed=request.seed,
                population=request.population or DEFAULT_POPULATION,
                iterations=request.iterations or DEFAULT_ITERATIONS,
            )
        except NothingToReplan as error:
            # Not a failure: a fleet that has delivered everything has nothing to
            # re-plan, which is an answer rather than an error.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(error)
            ) from None
        except SolverTooSmall as error:
            raise HTTPException(
                status_code=UNPROCESSABLE, detail=str(error)
            ) from None
        except ValueError as error:
            # `build_cost_matrix` refusing an unreachable pair. Under a live
            # closure that is a real answer — a vehicle on one side of a severed
            # network and a delivery on the other — and the positions are what
            # makes it actionable, since the builder's own message names node
            # pairs and an operator thinks in vehicles.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"the remaining deliveries cannot be re-planned under the "
                    f"current conditions: {error}. Vehicles are at "
                    + ", ".join(
                        f"{item.vehicle_id} -> {item.node!r}"
                        for item in progress
                        if item.node is not None
                    )
                    + "."
                ),
            ) from None

        # The *before* half: the rest of the plan the fleet is already driving, in
        # its original order and priced from each vehicle's real position — the
        # same footing the new plan is measured on, so the two are comparable as
        # plans rather than as two different pricings of the same work. Scoring a
        # partial solution reports a coverage penalty, which is why `feasible` on
        # the response comes from the re-optimization's own evaluation instead;
        # here the per-route loads, costs and times are what is wanted, and
        # `evaluate` computes them per route regardless of what is missing.
        before = Solution(routes=tuple(item.remaining for item in progress))
        before_scenario, before_matrix = before_view(
            record.scenario, progress, graph, weight=weight
        )
        # Hoisted out of the `_routes_out` call below because the explanation
        # needs the same evaluation the response reports — the per-route costs and
        # window penalties every "why this route" statement is built from. Scoring
        # it twice would be two chances to disagree.
        before_evaluation = evaluate(before, before_scenario, before_matrix)
        before_out = _routes_out(
            replace(record, scenario=before_scenario, cost_matrix=before_matrix),
            before,
            before_evaluation,
            graph,
            include_geometry=request.include_geometry,
            weight=weight,
            start_nodes=[
                before_scenario.depot.node if item.node is None else item.node
                for item in progress
            ],
        )

        # The derived instance, dressed as a record so `_routes_out` resolves the
        # new plan's indices through it. Its deliveries carry their original ids,
        # so an "after" stop is the same delivery a "before" stop was, and its
        # vehicles carry reduced capacities, which is what those routes are
        # actually bounded by.
        derived = replace(
            record, scenario=plan.instance, cost_matrix=plan.matrix
        )
        after_out = _routes_out(
            derived,
            plan.solution,
            plan.evaluation,
            graph,
            include_geometry=request.include_geometry,
            weight=weight,
            start_nodes=[item.node for item in plan.active],
        )

        ids = record.scenario.delivery_ids()
        completed = tuple(
            ids[stop] for item in progress for stop in item.completed
        )

        # The ETA pair is the one comparison this route computes and would
        # otherwise throw away: the same remaining stops, priced on the routes the
        # fleet is driving and on the routes it has just been offered. Both are
        # fleet-sized and measured from each vehicle's real position, which is what
        # makes them comparable rather than two different pricings of the same work.
        _record_run(
            run_store,
            kind="reoptimize",
            record=record,
            solver=spec,
            evaluation=plan.evaluation,
            runtime_ms=plan.runtime_ms,
            seed=request.seed,
            iterations=request.iterations,
            population=request.population,
            trigger=trigger.primary,
            trigger_detail=trigger.detail,
            old_eta_seconds=before_evaluation.travel_time,
            moved=len(plan.moved),
        )

        return ReoptimizeResponse(
            scenario_id=record.scenario_id,
            solver=spec.key,
            solver_name=spec.name,
            seed=request.seed,
            iterations=request.iterations if spec.is_stochastic else None,
            population=request.population if spec.is_stochastic else None,
            trigger=TriggerOut(
                kinds=list(trigger.kinds),
                primary=trigger.primary,
                detail=trigger.detail,
                edges=_edge_pairs(trigger.edges),
                reasons=list(trigger.reasons),
            ),
            vehicles=[
                VehicleProgressOut(
                    vehicle_id=item.vehicle_id,
                    node=item.node,
                    edge=_edge_out(item.edge),
                    completed=[ids[stop] for stop in item.completed],
                    remaining=[ids[stop] for stop in item.remaining],
                    elapsed_seconds=item.elapsed,
                    remaining_capacity=item.remaining_capacity,
                    stuck=item.stuck,
                    finished=item.finished,
                    available=item.available,
                )
                for item in progress
            ],
            # Every vehicle that could still work — which is every vehicle the
            # new plan was solved for, and so every route in `after`.
            replanned_vehicles=[item.vehicle_id for item in plan.active],
            before=before_out,
            after=after_out,
            completed=list(completed),
            replanned=[ids[stop] for stop in plan.pool],
            moved=[
                MoveOut(
                    delivery_id=ids[move.delivery],
                    from_vehicle=move.from_vehicle,
                    to_vehicle=move.to_vehicle,
                )
                for move in plan.moved
            ],
            cost=plan.evaluation.fitness,
            travel_cost=plan.evaluation.travel_cost,
            travel_time=plan.evaluation.travel_time,
            distance_m=plan.evaluation.distance,
            fuel_litres=plan.evaluation.fuel,
            feasible=plan.evaluation.feasible,
            runtime_ms=plan.runtime_ms,
            convergence=plan.convergence,
            explanation=_explanation_out(
                plan,
                before_evaluation,
                trigger,
                _road_delays(record, graph, trigger),
            ),
        )

    @application.post(
        "/scenarios/{scenario_id}/vehicles/{vehicle_id}/avoid-road",
        response_model=ReoptimizeResponse,
        tags=["reopt"],
    )
    def avoid_road(
        scenario_id: str,
        vehicle_id: str,
        payload: AvoidRoadRequest,
        graph: GraphDep,
        store: StoreDep,
        run_store: RunStoreDep,
        watchers: WatchersDep,
        log_store: LogStoreDep,
    ) -> ReoptimizeResponse:
        """A driver says a road is worse than the model thinks, and gets a new route.

        The manual override, alongside the two triggers ``POST /reoptimize``
        derives. An incident's flat ``x2.9`` is a *placeholder* — a guess at how
        much worse a slow road is — and the party who knows better is sitting in
        it. Without this, their only route to being re-planned is for the fleet to
        drive the road, be measured, and have the detector agree the trip was
        unusual, which at ``NOISE_SIGMA = 0.06`` is a lot to ask of a road that is
        simply much worse than the model's number.

        **The report is filed as a real incident**, through the same code
        ``POST /incident`` uses: the road is re-priced for the whole scenario, an
        audit row is written, and it lands in the scenario's own incident list —
        so ``DELETE /scenarios/{id}/incident/{incident_id}`` reverts it and a
        later fleet-wide ``POST /reoptimize`` sees it. That is what keeps
        ``/reoptimize``'s rule intact rather than cutting a hole in it: the driver
        has changed the network, not asserted a justification, and a re-plan that
        would have been refused a moment ago is now justified by a live incident
        like any other. This route **writes**; ``/reoptimize`` does not.

        What the override adds on top is **scope**. Exactly one vehicle is in the
        instance that is solved — the one that reported — so the pool is *its*
        remaining stops and the fleet is *it*. No other vehicle's route can move,
        because no other vehicle is in the problem. ``before`` and ``after``
        nonetheless cover the whole fleet, so the untouched routes are visible as
        unchanged rather than merely asserted to be.

        Where the new route begins is the one thing the report changes that is not
        about cost. A vehicle is planned from the far end of the road it is on —
        the next intersection it reaches. But a plan cannot begin on the far side
        of a road the vehicle cannot drive, so a **closure** on that road moves the
        start back to the near end: the driver turns around. A ``slow`` report
        leaves the road drivable, so the ordinary rule stands. The same exception
        covers a vehicle already stopped behind some other closure, which is why
        this can route a vehicle that ``/reoptimize`` has nothing to offer.

        The summary figures (``cost``, ``travel_*``, ``distance_m``,
        ``fuel_litres``, ``feasible``) describe **the plan that was solved** — the
        reporting vehicle's remaining stops — exactly as they do on
        ``/reoptimize``. The per-route figures on ``before`` and ``after`` carry
        the rest of the fleet, and every route that was not re-planned has the
        same numbers on both sides.
        """
        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        watcher = _watcher_or_404(watchers, scenario_id)

        edge = (payload.edge.u, payload.edge.v)
        # Checked against the graph rather than the scenario's nodes, the same way
        # and for the same reason `create_incident` checks it: a road is a road
        # wherever it is, and one that is not in the graph at all is a client
        # error rather than something to quietly ignore.
        if not graph.has_edge(edge[0], edge[1]):
            raise HTTPException(
                status_code=UNPROCESSABLE,
                detail=(
                    f"avoid-road edge: no road from {edge[0]!r} to {edge[1]!r} in "
                    "the road graph"
                ),
            )

        index = next(
            (
                position
                for position, vehicle in enumerate(record.scenario.vehicles)
                if vehicle.id == vehicle_id
            ),
            None,
        )
        if index is None:
            known = ", ".join(vehicle.id for vehicle in record.scenario.vehicles)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"scenario {scenario_id!r} has no vehicle {vehicle_id!r}; its "
                    f"fleet is: {known}."
                ),
            )

        # Read **before** the report is applied. Where a vehicle is is a fact about
        # the vehicle, not about the prices: under the closure the driver is about
        # to file, their own road is unpriceable and `fleet_progress` would report
        # them as stuck with no position at all — which is exactly the case this
        # endpoint exists to answer, so it must not be the case that loses the
        # answer.
        weight_before = traffic_weight_function(graph, record.effective_traffic_state())
        progress = watcher.read(
            lambda tracks: fleet_progress(
                record.scenario, tracks, cost_lookup(graph, weight_before)
            )
        )

        target = progress[index]
        if not target.remaining:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"vehicle {vehicle_id!r} has no unserved stops"
                    + (
                        " — it is stopped behind a closure and carrying nothing"
                        if target.stuck
                        else "; it has delivered everything it was dispatched with"
                    )
                    + ", so there is no route of its own to re-plan."
                ),
            )

        # `restart_node` decides, and it is passed the road only when the treatment
        # makes it impassable: a road reported *slow* is still drivable, so the
        # vehicle reaches its far end and the ordinary rule stands.
        start = restart_node(target, edge if payload.treatment == CLOSURE else None)
        if start is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"vehicle {vehicle_id!r} is not on a road, so there is nowhere "
                    "to begin a new route from."
                ),
            )

        # Resolved before the report is filed, so a request that is going to be
        # refused for its arguments is refused *without* having changed the
        # network. The report is a write; a malformed request must not be.
        try:
            spec = get_solver(payload.solver or DEFAULT_SOLVER_KEY)
        except KeyError as error:
            raise HTTPException(status_code=UNPROCESSABLE, detail=str(error)) from None

        # The report, filed with the machinery the incident routes use — the fold,
        # the re-price, the rollback when a closure severs the instance, the audit
        # row and the compare-and-swap are literally the same code.
        incident = Incident(
            incident_id=uuid.uuid4().hex,
            incident_type=payload.treatment,
            u=edge[0],
            v=edge[1],
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _change_incident(
            graph=graph,
            store=store,
            log_store=log_store,
            record=record,
            incidents=(*record.incidents, incident),
            incident=incident,
            word=incident.incident_type,
            applied=True,
        )
        # Re-read: the plan below must be priced under the conditions the driver
        # just created, not the ones that prompted the report.
        record = store.get(scenario_id)

        weight = traffic_weight_function(graph, record.effective_traffic_state())

        # The scoping, in one line: a one-element progress sequence *is* a
        # one-vehicle fleet, and `reoptimize` builds its pool, its capacities and
        # its starts from whatever it is handed. Every other vehicle is absent from
        # the instance, so none of them can be moved by it.
        #
        # `stuck` is cleared rather than carried. A vehicle holding a route that
        # does not use the road it cannot drive is no longer stuck *for planning*,
        # and reporting it as stuck while handing it a new route would contradict
        # itself. The road is not hidden — it is on `trigger.edges` and on the
        # incident the response carries.
        restarted = tuple(
            (
                replace(target, node=start, stuck=False)
                if position == index
                else item
            )
            for position, item in enumerate(progress)
        )

        try:
            plan = reoptimize(
                scenario=record.scenario,
                progress=(restarted[index],),
                graph=graph,
                weight=weight,
                solver_key=spec.key,
                seed=payload.seed,
                population=payload.population or DEFAULT_POPULATION,
                iterations=payload.iterations or DEFAULT_ITERATIONS,
            )
        except NothingToReplan as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(error)
            ) from None
        except SolverTooSmall as error:
            raise HTTPException(
                status_code=UNPROCESSABLE, detail=str(error)
            ) from None
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"vehicle {vehicle_id!r} cannot be re-planned around "
                    f"{edge[0]!r} -> {edge[1]!r}: {error}."
                ),
            ) from None

        # One scenario and one matrix for both halves, so an untouched route is
        # identical on either side by construction — same stops, same start, same
        # weights — rather than by a comparison someone remembered to write.
        before_scenario, before_matrix = before_view(
            record.scenario, restarted, graph, weight=weight
        )
        fleet_record = replace(
            record, scenario=before_scenario, cost_matrix=before_matrix
        )
        start_nodes = [
            before_scenario.depot.node if item.node is None else item.node
            for item in restarted
        ]

        before = Solution(routes=tuple(item.remaining for item in restarted))
        # The new route comes back holding positions in `plan.instance.deliveries`,
        # which are the reporting vehicle's remaining stops re-indexed; `plan.pool`
        # maps each back to the position it has in the stored scenario, so `after`
        # is in one index space and resolves through one scenario.
        after_routes = list(before.routes)
        after_routes[index] = tuple(
            plan.pool[stop] for stop in plan.solution.routes[0]
        )
        after = Solution(routes=tuple(after_routes))

        # Hoisted so the explanation compares against the same pricing the
        # response reports; this is the fleet-sized evaluation, indexed by
        # `restarted` order, which is what `before_positions` below refers to.
        before_evaluation = evaluate(before, before_scenario, before_matrix)

        before_out = _routes_out(
            fleet_record,
            before,
            before_evaluation,
            graph,
            include_geometry=payload.include_geometry,
            weight=weight,
            start_nodes=start_nodes,
        )
        # Hoisted rather than inlined into `_routes_out`, which is where it used to
        # live, because the run history needs the same number the response reports.
        # It is also the *only* fleet-sized "after" available here — `plan.evaluation`
        # covers the single reporting vehicle, since that is all the re-optimization
        # solved — so scoring it twice would be two chances for the row and the
        # response to disagree, and picking `plan.evaluation` instead would compare a
        # fleet's remaining travel time against one vehicle's.
        after_evaluation = evaluate(after, before_scenario, before_matrix)
        after_out = _routes_out(
            fleet_record,
            after,
            after_evaluation,
            graph,
            include_geometry=payload.include_geometry,
            weight=weight,
            start_nodes=start_nodes,
        )

        ids = record.scenario.delivery_ids()
        trigger = override_trigger(vehicle_id, edge, payload.treatment)

        _record_run(
            run_store,
            kind="avoid_road",
            record=record,
            solver=spec,
            evaluation=after_evaluation,
            runtime_ms=plan.runtime_ms,
            seed=payload.seed,
            iterations=payload.iterations,
            population=payload.population,
            trigger=trigger.primary,
            trigger_detail=trigger.detail,
            affected_vehicle=vehicle_id,
            old_eta_seconds=before_evaluation.travel_time,
            moved=len(plan.moved),
        )

        return ReoptimizeResponse(
            scenario_id=record.scenario_id,
            solver=spec.key,
            solver_name=spec.name,
            seed=payload.seed,
            iterations=payload.iterations if spec.is_stochastic else None,
            population=payload.population if spec.is_stochastic else None,
            trigger=TriggerOut(
                kinds=list(trigger.kinds),
                primary=trigger.primary,
                detail=trigger.detail,
                edges=_edge_pairs(trigger.edges),
                reasons=list(trigger.reasons),
            ),
            vehicles=[
                VehicleProgressOut(
                    vehicle_id=item.vehicle_id,
                    node=item.node,
                    edge=_edge_out(item.edge),
                    completed=[ids[stop] for stop in item.completed],
                    remaining=[ids[stop] for stop in item.remaining],
                    elapsed_seconds=item.elapsed,
                    remaining_capacity=item.remaining_capacity,
                    stuck=item.stuck,
                    finished=item.finished,
                    available=item.available,
                )
                for item in restarted
            ],
            replanned_vehicles=[vehicle_id],
            incident=_incident_out(incident),
            before=before_out,
            after=after_out,
            completed=[ids[stop] for item in restarted for stop in item.completed],
            replanned=[ids[stop] for stop in plan.pool],
            # Empty by construction, not by omission: one vehicle was solved, so
            # no delivery can change hands. What changed is the order, which is
            # what `before` and `after` are for.
            moved=[
                MoveOut(
                    delivery_id=ids[move.delivery],
                    from_vehicle=move.from_vehicle,
                    to_vehicle=move.to_vehicle,
                )
                for move in plan.moved
            ],
            cost=plan.evaluation.fitness,
            travel_cost=plan.evaluation.travel_cost,
            travel_time=plan.evaluation.travel_time,
            distance_m=plan.evaluation.distance,
            fuel_litres=plan.evaluation.fuel,
            feasible=plan.evaluation.feasible,
            runtime_ms=plan.runtime_ms,
            convergence=plan.convergence,
            # The before half is priced over the **whole fleet** here while the
            # re-plan solved a one-vehicle instance, so the default
            # "aligned to plan.progress" assumption does not hold and the
            # vehicle's position in the fleet-sized array is stated explicitly.
            # Without it the cost comparison would be against another vehicle's
            # route — a wrong number that would read as a plausible one.
            explanation=_explanation_out(
                plan,
                before_evaluation,
                trigger,
                _road_delays(record, graph, trigger),
                before_positions={vehicle_id: index},
            ),
        )

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
        run_store: RunStoreDep,
        payload: OptimizeRequest | None = None,
    ) -> OptimizeResponse:
        """Run one solver on a stored scenario.

        Defaults to the production default solver (QPSO) at the benchmark's
        default search effort, so an API result is comparable with a benchmark
        result. The body is optional: ``POST /optimize/{id}`` with no payload at
        all runs the default.

        One row is written to the run history for the solve. See
        :func:`_record_run`.
        """
        request = payload or OptimizeRequest()

        try:
            record = store.get(scenario_id)
        except ScenarioNotFound:
            raise _not_found(scenario_id) from None

        spec = _pick_solver(record, request.solver)

        iterations = request.iterations or DEFAULT_ITERATIONS
        population = request.population or DEFAULT_POPULATION

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

        # Trace the routes under the same weights that priced them, so the drawn
        # line is the road the optimizer actually chose. The *effective* state,
        # not the creation-time one: once an incident is live the two differ, and
        # drawing the pre-incident road would show a route nobody was charged for.
        weight = traffic_weight_function(graph, record.effective_traffic_state())

        _record_run(
            run_store,
            kind="optimize",
            record=record,
            solver=spec,
            evaluation=evaluation,
            runtime_ms=runtime_ms,
            seed=request.seed,
            iterations=iterations,
            population=population,
        )

        return OptimizeResponse(
            scenario_id=record.scenario_id,
            solver=spec.key,
            solver_name=spec.name,
            seed=request.seed,
            iterations=iterations if spec.is_stochastic else None,
            population=population if spec.is_stochastic else None,
            cost=cost,
            travel_cost=evaluation.travel_cost,
            travel_time=evaluation.travel_time,
            distance_m=evaluation.distance,
            fuel_litres=evaluation.fuel,
            feasible=evaluation.feasible,
            runtime_ms=runtime_ms,
            routes=_routes_out(
                record,
                solution,
                evaluation,
                graph,
                include_geometry=request.include_geometry,
                weight=weight,
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
                    travel_time=evaluation.travel_time,
                    distance_m=evaluation.distance,
                    fuel_litres=evaluation.fuel,
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

    # -- traffic ------------------------------------------------------------ #
    @application.get(
        "/traffic/log", response_model=TrafficLogPage, tags=["traffic"]
    )
    def traffic_log(
        log_store: LogStoreDep,
        limit: int = Query(default=DEFAULT_LOG_PAGE, ge=1, le=MAX_LOG_PAGE),
        offset: int = Query(default=0, ge=0),
        traffic_condition: str | None = Query(
            default=None,
            description="Filter to 'normal', 'moderate' or 'peak'.",
        ),
        incident_type: str | None = Query(
            default=None, description="Filter to 'accident' or 'road_closure'."
        ),
        road_u: str | None = Query(default=None, description="Filter by edge start."),
        road_v: str | None = Query(default=None, description="Filter by edge end."),
    ) -> TrafficLogPage:
        """Inspect what the simulator has been logging, newest first.

        Rows are written as a side effect of creating scenarios — one per road a
        scenario's pricing touched, plus any road an incident named. This
        endpoint only reads them; there is no prediction here. The table exists so
        a later phase has a ``(road, time, condition, incident) -> travel_time``
        history to learn from.

        ``total`` counts every row matching the filters, not just this page, so a
        client can page through a filter without a separate count request. Ordered
        by insertion rather than by ``timestamp``: timestamps are ISO strings, and
        string order is only chronological while every row shares one UTC offset.
        """
        rows, total = log_store.read(
            limit=limit,
            offset=offset,
            road_u=_node_filter(road_u),
            road_v=_node_filter(road_v),
            traffic_condition=traffic_condition,
            incident_type=incident_type,
        )
        return TrafficLogPage(
            items=[TrafficLogEntry(**row.to_dict()) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(rows) < total,
        )

    # -- analytics ---------------------------------------------------------- #
    @application.get("/analytics/runs", response_model=RunPage, tags=["analytics"])
    def analytics_runs(
        run_store: RunStoreDep,
        limit: int = Query(default=DEFAULT_LOG_PAGE, ge=1, le=MAX_LOG_PAGE),
        offset: int = Query(default=0, ge=0),
        kind: str | None = Query(
            default=None,
            description=(
                "Filter to 'dispatch', 'optimize', 'reoptimize' or 'avoid_road'."
            ),
        ),
        solver: str | None = Query(default=None, description="Filter by solver key."),
        scenario_id: str | None = Query(
            default=None, description="Filter to one scenario."
        ),
    ) -> RunPage:
        """Inspect every solve this process has run, newest first.

        One row per solve, written by ``POST /optimize``, ``POST /reoptimize``,
        ``POST .../avoid-road`` and ``POST .../watcher``. The last of those is a
        solve too — dispatching a fleet is how a scenario gets its first plan —
        which is why it appears here and not only in the fleet endpoints.

        Nothing is written for a **refused** request. A re-optimization that
        answers 409 decided there was nothing to re-plan and returned before
        reaching the store, so this table is a record of runs rather than of
        attempts.

        ``total`` counts every row matching the filters, not just this page, so a
        client can page through a filter without a separate count request. Ordered
        by insertion rather than by ``timestamp``: timestamps are ISO strings, and
        string order is only chronological while every row shares one UTC offset.
        """
        rows, total = run_store.read(
            limit=limit, offset=offset, kind=kind, solver=solver, scenario_id=scenario_id
        )
        return RunPage(
            items=[RunEntry(**row.to_dict()) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(rows) < total,
        )

    @application.get(
        "/analytics/summary", response_model=AnalyticsSummary, tags=["analytics"]
    )
    def analytics_summary(run_store: RunStoreDep) -> AnalyticsSummary:
        """Aggregate statistics over the whole run history.

        Averages are reported as ``null`` rather than ``0`` when there is nothing
        to average. On a fresh install every one of them is ``null``, and that is
        the honest answer — an average runtime of zero milliseconds would be a
        claim about performance where no measurement exists.

        The ETA block is restricted to runs that had a baseline, which is to say
        the re-plans. A dispatch is never averaged into it, because averaging its
        absent baseline in as a zero saving would understate every real one.
        """
        return _summary_out(run_store.summary())

    return application


app = create_app()
