"""API tests.

The API is exercised through ``TestClient``, i.e. real HTTP requests against the
real application, rather than by calling route functions directly. That is worth
the extra indirection: validation, status codes, and response serialisation are
where an API actually breaks, and none of them are covered by calling a handler.

The road graph is injected through ``app.dependency_overrides``, so the whole
surface is tested against a synthetic graph with no cached Delhi extract, no
network, and no ~20 s cold load. That override point is the reason ``get_graph``
is a dependency rather than a module global.

What is asserted about *solvers* here is deliberately thin — that the default is
ACO, that a named solver is honoured, that errors map to the right status codes.
Search quality belongs to the optimizer's own tests; these are about the HTTP
contract.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qgati.api.main import create_app, get_graph, get_store
from qgati.api.store import ScenarioStore
from qgati.graph import build_synthetic_graph
from qgati.optimizer import servable_nodes

#: Big enough that its largest strongly-connected subgraph can host the
#: instances these tests generate.
GRAPH_NODES = 60
GRAPH_EDGE_PROB = 0.25
GRAPH_SEED = 7


#: The synthetic layout is a 0..1000 square in arbitrary units, which is not a
#: valid latitude or longitude. These map it into a small Delhi-ish box so the
#: coordinate-snapping paths can be exercised with real degrees. Routing uses
#: edge weights, never coordinates, so rescaling changes no cost.
LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05


@pytest.fixture(scope="module")
def graph():
    """A synthetic road graph tagged with x/y, as OSMnx graphs are.

    ``build_synthetic_graph`` stores its layout under ``pos``; OSMnx's coordinate
    snapping and the API's coordinate lookup both read ``x``/``y`` (longitude and
    latitude respectively), so both are added here.
    """
    road_graph = build_synthetic_graph(
        n_nodes=GRAPH_NODES, edge_prob=GRAPH_EDGE_PROB, seed=GRAPH_SEED
    )
    for _, data in road_graph.nodes(data=True):
        x, y = data["pos"]
        data["x"] = LON_ORIGIN + (x / 1000.0) * DEGREE_SPAN
        data["y"] = LAT_ORIGIN + (y / 1000.0) * DEGREE_SPAN
    return road_graph


@pytest.fixture
def client(graph):
    """A fresh app and an empty store per test, so tests cannot leak state.

    The store override must be a *callable returning one instance*, not the class
    itself: ``get_store`` is a dependency, so passing ``ScenarioStore`` would have
    FastAPI construct a new empty store for every request and nothing would ever
    be found by id.
    """
    application = create_app()
    store = ScenarioStore()
    application.dependency_overrides[get_graph] = lambda: graph
    application.dependency_overrides[get_store] = lambda: store
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


def make_scenario(client, **overrides) -> dict:
    """Create a generated scenario, returning the parsed response."""
    payload = {"kind": "generate", "n_deliveries": 8, "n_vehicles": 3, "seed": 303}
    payload.update(overrides)
    response = client.post("/scenarios", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_solvers_endpoint_lists_every_solver_and_the_default(client) -> None:
    """The registry is the single source of truth, and ACO is the default.

    The production-default assertion is the important one: it is the API-level
    expression of the Phase 4 decision, so if it ever silently reverts to QPSO
    this test fails.
    """
    response = client.get("/solvers")
    assert response.status_code == 200
    payload = response.json()

    assert payload["default"] == "aco"
    keys = [spec["key"] for spec in payload["solvers"]]
    assert keys == [
        "brute_force",
        "savings",
        "aco",
        "genetic_algorithm",
        "classical_pso",
        "qpso",
    ]

    by_key = {spec["key"]: spec for spec in payload["solvers"]}
    assert by_key["brute_force"]["is_exact"] is True
    assert by_key["savings"]["is_stochastic"] is False
    assert by_key["aco"]["is_stochastic"] is True


def test_servable_node_count(client) -> None:
    response = client.get("/graph/servable")
    assert response.status_code == 200
    assert response.json()["servable_nodes"] > 0


# --------------------------------------------------------------------------- #
# Scenario creation
# --------------------------------------------------------------------------- #
def test_generate_scenario(client) -> None:
    body = make_scenario(client)
    assert body["n_deliveries"] == 8
    assert body["n_vehicles"] == 3
    assert len(body["deliveries"]) == 8
    assert len(body["vehicles"]) == 3
    assert body["total_demand"] > 0
    assert body["total_demand"] <= body["total_capacity"]
    assert body["exactly_solvable"] is True  # n=8 is within brute-force reach

    # The depot always carries coordinates, whichever graph layout produced them.
    assert isinstance(body["depot"]["lat"], float)
    assert isinstance(body["depot"]["lon"], float)


def test_generated_scenario_is_reproducible_for_a_seed(client) -> None:
    first = make_scenario(client, seed=5)
    second = make_scenario(client, seed=5)
    assert [d["node"] for d in first["deliveries"]] == [
        d["node"] for d in second["deliveries"]
    ]
    # Different seed, different instance.
    third = make_scenario(client, seed=6)
    assert [d["node"] for d in first["deliveries"]] != [
        d["node"] for d in third["deliveries"]
    ]


def test_generate_rejects_a_graph_too_small_for_the_request(client) -> None:
    response = client.post(
        "/scenarios",
        json={"kind": "generate", "n_deliveries": 500, "n_vehicles": 5},
    )
    assert response.status_code == 422


def test_explicit_scenario_by_node_id(client, graph) -> None:
    nodes = servable_nodes(graph)
    depot_node, *stop_nodes = nodes[:4]
    response = client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": depot_node},
            "deliveries": [
                {"id": f"D{i}", "node": node, "demand": 2}
                for i, node in enumerate(stop_nodes)
            ],
            "vehicles": [{"id": "V0", "capacity": 100}],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["n_deliveries"] == 3
    assert [d["id"] for d in body["deliveries"]] == ["D0", "D1", "D2"]
    assert body["total_demand"] == 6


def test_explicit_scenario_by_coordinates_is_snapped(client) -> None:
    """A client that speaks in coordinates gets a node back, not an error."""
    mid_lat = LAT_ORIGIN + DEGREE_SPAN / 2
    mid_lon = LON_ORIGIN + DEGREE_SPAN / 2
    response = client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"lat": mid_lat, "lon": mid_lon},
            "deliveries": [
                {"id": "D0", "lat": LAT_ORIGIN + 0.01, "lon": LON_ORIGIN + 0.01, "demand": 1},
                {"id": "D1", "lat": LAT_ORIGIN + 0.04, "lon": LON_ORIGIN + 0.04, "demand": 1},
            ],
            "vehicles": [{"id": "V0", "capacity": 50}],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["n_deliveries"] == 2
    # Snapped nodes come back as graph nodes, and the depot's stored coordinates
    # are the snapped node's, not the ones the client sent.
    assert body["depot"]["node"] is not None
    assert body["deliveries"][0]["node"] is not None
    assert LAT_ORIGIN <= body["depot"]["lat"] <= LAT_ORIGIN + DEGREE_SPAN
    assert LON_ORIGIN <= body["depot"]["lon"] <= LON_ORIGIN + DEGREE_SPAN


def test_explicit_scenario_rejects_unknown_node(client) -> None:
    response = client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": "definitely-not-a-node"},
            "deliveries": [{"id": "D0", "node": 1, "demand": 1}],
            "vehicles": [{"id": "V0", "capacity": 10}],
        },
    )
    assert response.status_code == 422
    assert "not in the road graph" in response.text


def test_explicit_scenario_rejects_infeasible_demand(client, graph) -> None:
    """Demand beyond the fleet is a client error, reported as one."""
    nodes = servable_nodes(graph)
    response = client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": nodes[0]},
            "deliveries": [
                {"id": "D0", "node": nodes[1], "demand": 50},
                {"id": "D1", "node": nodes[2], "demand": 50},
            ],
            "vehicles": [{"id": "V0", "capacity": 10}],
        },
    )
    assert response.status_code == 422
    assert "infeasible" in response.text.lower()


def test_scenario_schema_rejects_partial_coordinates(client) -> None:
    response = client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"lat": 28.6},  # no lon
            "deliveries": [{"id": "D0", "node": 1, "demand": 1}],
            "vehicles": [{"id": "V0", "capacity": 10}],
        },
    )
    assert response.status_code == 422


def test_list_and_fetch_scenarios(client) -> None:
    created = make_scenario(client)
    scenario_id = created["scenario_id"]

    listing = client.get("/scenarios")
    assert listing.status_code == 200
    assert [item["scenario_id"] for item in listing.json()] == [scenario_id]

    fetched = client.get(f"/scenarios/{scenario_id}")
    assert fetched.status_code == 200
    assert fetched.json()["scenario_id"] == scenario_id

    assert client.get("/scenarios/nope").status_code == 404


# --------------------------------------------------------------------------- #
# Optimize
# --------------------------------------------------------------------------- #
def test_optimize_defaults_to_aco(client) -> None:
    """The production default, asserted at the HTTP boundary."""
    scenario_id = make_scenario(client)["scenario_id"]
    response = client.post(f"/optimize/{scenario_id}", json={"seed": 0})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["solver"] == "aco"
    assert body["solver_name"] == "ACO"
    assert body["travel_cost"] > 0
    assert body["cost"] >= body["travel_cost"]
    assert body["feasible"] is True
    assert body["runtime_ms"] > 0


def test_optimize_with_no_body_at_all(client) -> None:
    """The scenario is in the URL, so the body is optional — the whole point of
    naming it in the path."""
    scenario_id = make_scenario(client)["scenario_id"]
    response = client.post(f"/optimize/{scenario_id}")
    assert response.status_code == 200, response.text
    assert response.json()["solver"] == "aco"


def test_optimize_honours_a_named_solver(client) -> None:
    scenario_id = make_scenario(client)["scenario_id"]
    for key in ("qpso", "genetic_algorithm", "classical_pso", "savings"):
        response = client.post(
            f"/optimize/{scenario_id}", json={"solver": key, "seed": 0}
        )
        assert response.status_code == 200, response.text
        assert response.json()["solver"] == key


def test_optimize_resolves_deliveries_into_a_valid_fleet(client) -> None:
    """Every delivery served exactly once, one route per vehicle, within capacity."""
    created = make_scenario(client)
    scenario_id = created["scenario_id"]
    response = client.post(f"/optimize/{scenario_id}", json={"seed": 0})
    body = response.json()

    assert len(body["routes"]) == created["n_vehicles"]

    served = [
        stop["delivery_id"] for route in body["routes"] for stop in route["stops"]
    ]
    expected = sorted(d["id"] for d in created["deliveries"])
    assert sorted(served) == expected, "every delivery served exactly once"

    expected_nodes = {d["id"]: d["node"] for d in created["deliveries"]}
    for route in body["routes"]:
        assert route["load"] <= route["capacity"]
        for stop in route["stops"]:
            assert stop["node"] == expected_nodes[stop["delivery_id"]]


def test_optimize_is_reproducible_for_a_seed(client) -> None:
    scenario_id = make_scenario(client)["scenario_id"]
    first = client.post(f"/optimize/{scenario_id}", json={"seed": 11}).json()
    second = client.post(f"/optimize/{scenario_id}", json={"seed": 11}).json()
    assert first["cost"] == second["cost"]
    assert first["routes"] == second["routes"]


def test_optimize_returns_a_convergence_history_for_stochastic_solvers(client) -> None:
    scenario_id = make_scenario(client)["scenario_id"]
    body = client.post(
        f"/optimize/{scenario_id}",
        json={"solver": "aco", "iterations": 25, "seed": 0},
    ).json()
    assert len(body["convergence"]) == 25
    assert all(
        later <= earlier + 1e-9
        for earlier, later in zip(body["convergence"], body["convergence"][1:])
    )
    # The final history entry is the best-so-far cost the response reports.
    assert body["convergence"][-1] == pytest.approx(body["cost"])


def test_optimize_reports_no_history_for_deterministic_solvers(client) -> None:
    scenario_id = make_scenario(client)["scenario_id"]
    body = client.post(f"/optimize/{scenario_id}", json={"solver": "savings"}).json()
    assert body["convergence"] == []
    assert body["iterations"] is None
    assert body["population"] is None


def test_optimize_unknown_scenario_is_404(client) -> None:
    assert client.post("/optimize/missing").status_code == 404


def test_optimize_unknown_solver_lists_the_valid_ones(client) -> None:
    scenario_id = make_scenario(client)["scenario_id"]
    response = client.post(f"/optimize/{scenario_id}", json={"solver": "annealing"})
    assert response.status_code == 422
    assert "unknown solver" in response.text
    assert "aco" in response.text


def test_optimize_rejects_brute_force_beyond_its_exact_limit(client) -> None:
    scenario_id = make_scenario(client, n_deliveries=15, n_vehicles=3)["scenario_id"]
    response = client.post(f"/optimize/{scenario_id}", json={"solver": "brute_force"})
    assert response.status_code == 422
    assert "limited to" in response.text


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
def test_compare_runs_every_solver_with_gaps_against_the_optimum(client) -> None:
    """n=8 is within exact reach, so the optimum and both gap columns are real."""
    scenario_id = make_scenario(client)["scenario_id"]
    response = client.get(f"/optimize/{scenario_id}/compare", params={"seed": 0})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["optimal"] is not None
    assert body["best_known"] == pytest.approx(body["optimal"], rel=1e-6)

    results = {row["solver"]: row for row in body["results"]}
    assert set(results) == {
        "brute_force",
        "savings",
        "aco",
        "genetic_algorithm",
        "classical_pso",
        "qpso",
    }
    # Every solver ran; none was skipped.
    assert all(row["skipped"] is None for row in body["results"])

    # Brute force is the optimum by definition, so its gap is exactly zero.
    assert results["brute_force"]["gap_vs_optimal_pct"] == pytest.approx(0.0)
    # Savings is a heuristic and should not beat the optimum.
    assert results["savings"]["gap_vs_optimal_pct"] > 0.0
    # And no solver may report a cost below the optimum.
    for row in body["results"]:
        assert row["travel_cost"] >= body["optimal"] - 1e-6


def test_compare_skips_brute_force_beyond_its_exact_limit(client) -> None:
    scenario_id = make_scenario(client, n_deliveries=15, n_vehicles=3)["scenario_id"]
    response = client.get(f"/optimize/{scenario_id}/compare", params={"seed": 0})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["optimal"] is None  # no exact answer to compare against

    results = {row["solver"]: row for row in body["results"]}
    assert results["brute_force"]["skipped"] is not None
    assert "exact solver limited to" in results["brute_force"]["skipped"]
    # A solver that did not run reports no result at all, rather than a zero-cost
    # infeasible one — the distinction matters to a client rendering the table.
    for field in ("cost", "travel_cost", "feasible", "runtime_ms"):
        assert results["brute_force"][field] is None, field
    assert results["brute_force"]["gap_vs_optimal_pct"] is None

    # The rest still ran, and the best of them defines best_known.
    assert body["best_known"] == pytest.approx(
        min(row["travel_cost"] for row in body["results"] if row["skipped"] is None)
    )
    assert results["aco"]["gap_vs_best_pct"] is not None


def test_compare_unknown_scenario_is_404(client) -> None:
    assert client.get("/optimize/missing/compare").status_code == 404


# --------------------------------------------------------------------------- #
# Road geometry
# --------------------------------------------------------------------------- #
def _rounded_positions(graph) -> dict[tuple[float, float], int]:
    """Map each node's rounded ``(lon, lat)`` back to its node id.

    Geometry is serialised rounded to 6 decimals, so a drawn coordinate can only
    be matched back to a node through the same rounding.
    """
    return {
        (round(data["x"], 6), round(data["y"], 6)): node
        for node, data in graph.nodes(data=True)
    }


def test_route_geometry_follows_real_roads(client, graph) -> None:
    """The drawn route is a walk over actual graph edges — not straight lines.

    This is the assertion that matters for the map: a polyline connecting stops
    directly would look plausible in JSON and be wrong on screen. Every step of
    the returned line must be a real edge of the road network.
    """
    created = make_scenario(client)
    scenario_id = created["scenario_id"]
    body = client.post(f"/optimize/{scenario_id}", json={"seed": 0}).json()

    adjacency = {(u, v) for u, v in graph.edges()}
    at = _rounded_positions(graph)
    depot_point = next(
        point
        for point, node in at.items()
        if node == created["depot"]["node"]
    )

    served = 0
    for route in body["routes"]:
        geometry = route["geometry"]
        if not route["stops"]:
            assert geometry == [], "an unused vehicle has no tour to draw"
            continue

        served += 1
        # Depot at both ends, so the tour is closed.
        assert tuple(geometry[0]) == depot_point
        assert tuple(geometry[-1]) == depot_point

        # At least one point per leg, plus the closing point.
        assert len(geometry) >= len(route["stops"]) + 1

        for start, end in zip(geometry, geometry[1:]):
            if start == end:
                continue
            u, v = at[tuple(start)], at[tuple(end)]
            assert (u, v) in adjacency, (
                f"geometry step {u} -> {v} is not a road; "
                "the route was drawn as a straight line"
            )

    assert served > 0, "the instance must exercise at least one used vehicle"


def test_route_geometry_passes_through_every_stop(client, graph) -> None:
    """Each stop's own node appears in the drawn line, in visiting order.

    A client gets a stop's coordinates by joining its node id against
    ``/graph/delhi``'s Point features, which carry ``id`` — this test performs
    that same join, so it covers the integration the frontend actually does.
    """
    created = make_scenario(client)
    body = client.post(f"/optimize/{created['scenario_id']}", json={"seed": 0}).json()

    points_by_node = {
        feature["properties"]["id"]: tuple(feature["geometry"]["coordinates"])
        for feature in client.get("/graph/delhi").json()["features"]
        if feature["geometry"]["type"] == "Point"
    }

    checked = 0
    for route in body["routes"]:
        drawn = [tuple(point) for point in route["geometry"]]
        visit_order = []
        for stop in route["stops"]:
            position = points_by_node[stop["node"]]
            assert position in drawn, "a stop is missing from its own route"
            visit_order.append(drawn.index(position))
            checked += 1
        # Stops are visited in order, so their positions appear in that order.
        assert visit_order == sorted(visit_order)

    assert checked > 0


def test_include_geometry_false_skips_the_tracing(client) -> None:
    """The escape hatch for large instances: ordering without the road tracing."""
    scenario_id = make_scenario(client)["scenario_id"]
    body = client.post(
        f"/optimize/{scenario_id}", json={"seed": 0, "include_geometry": False}
    ).json()
    assert all(route["geometry"] == [] for route in body["routes"])
    # The routing is skipped, not the optimisation.
    assert body["travel_cost"] > 0


# --------------------------------------------------------------------------- #
# Graph GeoJSON
# --------------------------------------------------------------------------- #
def test_graph_geojson_is_a_feature_collection_of_roads(client) -> None:
    response = client.get("/graph/delhi")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["type"] == "FeatureCollection"
    assert body["features"]

    line_strings = [
        f for f in body["features"] if f["geometry"]["type"] == "LineString"
    ]
    points = [f for f in body["features"] if f["geometry"]["type"] == "Point"]
    assert line_strings and points

    for feature in line_strings:
        coordinates = feature["geometry"]["coordinates"]
        assert len(coordinates) >= 2
        for lon, lat in coordinates:
            # The synthetic layout is rescaled into a small Delhi-ish box, so
            # this is a genuine lon/lat ordering check, not a tautology.
            assert LON_ORIGIN <= lon <= LON_ORIGIN + DEGREE_SPAN
            assert LAT_ORIGIN <= lat <= LAT_ORIGIN + DEGREE_SPAN
        assert "u" in feature["properties"] and "v" in feature["properties"]


def test_graph_geojson_scoped_to_a_scenario_is_smaller(client) -> None:
    """Scoping is the whole point: a delivery round needs its own streets only."""
    scenario_id = make_scenario(client)["scenario_id"]
    whole = client.get("/graph/delhi").json()
    scoped = client.get("/graph/delhi", params={"scenario_id": scenario_id}).json()

    assert 0 < len(scoped["features"]) < len(whole["features"])


def test_graph_geojson_scenario_padding_widens_the_clip(client) -> None:
    """``padding_m`` is how much context around the stops the map gets."""
    scenario_id = make_scenario(client)["scenario_id"]
    tight = client.get(
        "/graph/delhi", params={"scenario_id": scenario_id, "padding_m": 50}
    ).json()
    loose = client.get(
        "/graph/delhi", params={"scenario_id": scenario_id, "padding_m": 1500}
    ).json()

    assert 0 < len(tight["features"]) < len(loose["features"])


def test_graph_geojson_explicit_bbox_clips_the_network(client) -> None:
    whole = client.get("/graph/delhi").json()

    # Anchored on a real node: the synthetic layout is sparse enough that an
    # arbitrary box can land on empty space between nodes.
    depot = make_scenario(client)["depot"]
    lon, lat = depot["lon"], depot["lat"]
    box = f"{lon - 0.006},{lat - 0.006},{lon + 0.006},{lat + 0.006}"

    clipped = client.get("/graph/delhi", params={"bbox": box}).json()
    assert 0 < len(clipped["features"]) < len(whole["features"])

    # include_nodes=False is the basemap-only fetch: roads, no junctions.
    roads_only = client.get(
        "/graph/delhi", params={"bbox": box, "include_nodes": False}
    ).json()
    assert roads_only["features"]
    assert all(
        feature["geometry"]["type"] == "LineString"
        for feature in roads_only["features"]
    )
    assert len(roads_only["features"]) < len(clipped["features"])


def test_graph_geojson_rejects_a_malformed_bbox(client) -> None:
    for bad in ("1,2,3", "a,b,c,d", "10,10,0,0"):
        response = client.get("/graph/delhi", params={"bbox": bad})
        assert response.status_code == 422, bad
        assert "bbox" in response.text


def test_graph_geojson_rejects_bbox_and_scenario_together(client) -> None:
    scenario_id = make_scenario(client)["scenario_id"]
    response = client.get(
        "/graph/delhi",
        params={"scenario_id": scenario_id, "bbox": "77.2,28.6,77.25,28.65"},
    )
    assert response.status_code == 422
    assert "not both" in response.text


def test_graph_geojson_unknown_scenario_is_404(client) -> None:
    assert client.get("/graph/delhi", params={"scenario_id": "nope"}).status_code == 404


# --------------------------------------------------------------------------- #
# CORS and error handling
# --------------------------------------------------------------------------- #
def test_cors_allows_the_nextjs_dev_server(client) -> None:
    """The frontend is a different origin, so the browser needs this preflight."""
    response = client.options(
        "/scenarios",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_does_not_allow_an_unlisted_origin(client) -> None:
    response = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_unhandled_errors_return_json_without_a_traceback(graph) -> None:
    """A crash is a 500 with a parseable body and none of the server's internals.

    Starlette's default is a bare ``Internal Server Error`` *text* body, which
    breaks a client that parses every response as JSON — on the one response it
    most needs to handle. The traceback belongs in the server log.
    """
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: graph

    def exploding_store():
        raise RuntimeError("deliberate failure at C:/secret/internal/path.py")

    application.dependency_overrides[get_store] = exploding_store
    try:
        with TestClient(application, raise_server_exceptions=False) as test_client:
            response = test_client.get("/scenarios")
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "secret" not in response.text
    assert "Traceback" not in response.text


# --------------------------------------------------------------------------- #
# OpenAPI
# --------------------------------------------------------------------------- #
def test_openapi_schema_is_generated(client) -> None:
    """The documented contract exists and includes every named endpoint."""
    schema = client.get("/openapi.json").json()
    assert "/optimize/{scenario_id}" in schema["paths"]
    assert "/optimize/{scenario_id}/compare" in schema["paths"]
    assert "/scenarios" in schema["paths"]
    assert "/graph/delhi" in schema["paths"]
