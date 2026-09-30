"""The driver's own override: *avoid this road, give me a new route*.

Two claims this file exists to defend, and they are different in kind.

The first is about **scope**. ``POST /scenarios/{id}/vehicles/{id}/avoid-road``
re-plans one vehicle, and "one vehicle" is not a claim about intent — it is a
claim about the instance that was solved. Every other vehicle is absent from it, so
its route cannot move. The test here checks that the *implementation* realises
that, by comparing every other vehicle's route on both sides of the call, stop for
stop and rupee for rupee, rather than by reading the prose that says so.

The second is about **where a new route may begin**. A vehicle is planned from the
far end of the road it is on — the next intersection it reaches — unless that end
lies beyond a road it cannot drive, in which case the near end. That rule is what
lets this endpoint answer a driver whose own road has just shut, which is the one
case ``POST /reoptimize`` has nothing to offer: its fleet excludes a stopped
vehicle, and here the stopped vehicle is the only one that matters. The four cases
are pinned as unit tests, and the two that a caller can actually reach through the
API are pinned again end to end.

The graphs
----------
Two, for two different jobs. The **tour graph** is the same two-corridor one the
other fleet tests use, and it is what the focused and error cases run on: it is
small enough that a route can be reasoned about by hand.

The **grid** is here because a driver-scoped re-plan needs a route with enough
*distinct* stops in it for the other vehicles to still be mid-route when the call
arrives. On a three-node graph a route is two legs long, every delivery sits at one
of two intersections, and a vehicle two thirds of the way along its own corridor
has almost always finished — which would leave the untouched-vehicle comparison
passing on two empty routes. On a grid every leg is a real road and a route is a
chain, so a vehicle a third of the way along still has most of its work ahead of
it.
"""

from __future__ import annotations

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from qgati.analytics import RunLogStore
from qgati.api.main import (
    create_app,
    get_graph,
    get_log_store,
    get_run_store,
    get_store,
    get_watchers,
)
from qgati.api.store import ScenarioStore
from qgati.fleet import WatcherRegistry
from qgati.optimizer import MAX_EXACT_DELIVERIES
from qgati.reopt import VehicleProgress, restart_node
from qgati.traffic import CLOSURE, SLOW, TrafficLogStore

ARTERIAL_S = 10.0
SIDE_STREET_S = 11.0

#: 02:00 prices as ``normal``, so every modelled road costs its own base time.
NORMAL_TIME = "2026-09-21T02:00:00+05:30"

LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05
TOY_SPEED_KPH = 30.0

#: A long interval against a tiny scale: a tick advances the fleet by
#: ``interval_seconds * time_scale`` seconds while leaving the background thread
#: effectively asleep, so only the explicit ticks in a test move anything.
IDLE_INTERVAL = 300.0
ONE_SECOND = {"interval_seconds": IDLE_INTERVAL, "time_scale": 1.0 / IDLE_INTERVAL}

#: Deterministic, for the same reason the re-optimization tests use it: these are
#: tests about where a vehicle is and what it may be given, not about the search.
DETERMINISTIC = {"solver": "savings"}

#: A fixed seed, so the fleet's noise is the draw already known not to flag itself.
START_DEFAULTS = {"seed": 3}


def _length_of(travel_time: float) -> float:
    return travel_time * (TOY_SPEED_KPH / 3.6)


def build_tour_graph() -> nx.DiGraph:
    """Two disjoint corridors through three nodes, plus a dead-end spur.

    The same graph ``test_reopt.py`` and the other fleet tests build; every
    intersection a scenario may use has a road back to the depot, which is what
    lets a closure be placed without severing anything.
    """
    graph = nx.DiGraph()
    positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (2.0, 2.0)}
    for node, (x, y) in positions.items():
        graph.add_node(
            node, x=LON_ORIGIN + x * DEGREE_SPAN, y=LAT_ORIGIN + y * DEGREE_SPAN
        )

    for u, v in ((0, 1), (1, 2), (2, 0)):
        graph.add_edge(
            u, v, highway="secondary", travel_time=ARTERIAL_S, weight=ARTERIAL_S,
            length=_length_of(ARTERIAL_S),
        )
    for u, v in ((0, 2), (2, 1), (1, 0)):
        graph.add_edge(
            u, v, highway="residential", travel_time=SIDE_STREET_S,
            weight=SIDE_STREET_S, length=_length_of(SIDE_STREET_S),
        )
    graph.add_edge(
        0, 3, highway="residential", travel_time=500.0, weight=500.0,
        length=_length_of(500.0),
    )
    return graph


#: A square grid, so a route is a chain of distinct intersections rather than a
#: shuttle between two of them.
GRID = 5
GRID_STEP_S = 10.0
GRID_SCALE = 0.25


def grid_node(x: int, y: int) -> int:
    return y * GRID + x


def build_grid_graph() -> nx.DiGraph:
    """A ``GRID`` x ``GRID`` lattice of two-way residential streets."""
    graph = nx.DiGraph()
    for y in range(GRID):
        for x in range(GRID):
            graph.add_node(
                grid_node(x, y),
                x=LON_ORIGIN + x * DEGREE_SPAN * GRID_SCALE,
                y=LAT_ORIGIN + y * DEGREE_SPAN * GRID_SCALE,
            )

    for y in range(GRID):
        for x in range(GRID):
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx >= GRID or y + dy >= GRID:
                    continue
                u, v = grid_node(x, y), grid_node(x + dx, y + dy)
                for a, b in ((u, v), (v, u)):
                    graph.add_edge(
                        a, b, highway="residential", travel_time=GRID_STEP_S,
                        weight=GRID_STEP_S, length=_length_of(GRID_STEP_S),
                    )
    return graph


#: Every node but the depot: the grid scenario's delivery points, so that no two
#: deliveries share an intersection and no leg of any route is zero-length.
GRID_STOPS = [grid_node(x, y) for y in range(GRID) for x in range(GRID)][1:]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tour_graph() -> nx.DiGraph:
    return build_tour_graph()


@pytest.fixture(scope="module")
def grid_graph() -> nx.DiGraph:
    return build_grid_graph()


@pytest.fixture
def log_store() -> TrafficLogStore:
    return TrafficLogStore(":memory:")


@pytest.fixture
def store() -> ScenarioStore:
    return ScenarioStore()


@pytest.fixture
def watchers() -> WatcherRegistry:
    return WatcherRegistry()


def _client_for(graph, store, log_store, watchers) -> TestClient:
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: graph
    application.dependency_overrides[get_store] = lambda: store
    application.dependency_overrides[get_log_store] = lambda: log_store
    # In-memory, like the traffic log. Reporting a road *is* a re-plan, so every
    # avoid-road call here would otherwise append a row to the developer's real
    # ``backend/data/run_history.db``.
    application.dependency_overrides[get_run_store] = lambda: RunLogStore(":memory:")
    application.dependency_overrides[get_watchers] = lambda: watchers
    return application


@pytest.fixture
def client(tour_graph, store, log_store, watchers):
    """An app over the tour graph, with a fleet registry that lives for the test."""
    application = _client_for(tour_graph, store, log_store, watchers)
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


@pytest.fixture
def grid_client(grid_graph, store, log_store, watchers):
    """The same app over the grid, for the multi-vehicle case."""
    application = _client_for(grid_graph, store, log_store, watchers)
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
SMALL_DELIVERIES = [
    {"id": "D0", "node": 1, "demand": 1},
    {"id": "D1", "node": 2, "demand": 1},
    {"id": "D2", "node": 1, "demand": 1},
]


def create_scenario(client, *, depot=0, deliveries, vehicles, timestamp=NORMAL_TIME):
    response = client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": depot},
            "deliveries": deliveries,
            "vehicles": vehicles,
            "conditions": {"timestamp": timestamp},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_small(client) -> dict:
    """One vehicle, three stops over two intersections, all of it still ahead.

    The route is long enough that the vehicle sits on a real road halfway along
    it, which is what the start-node cases need — and it is deliberately small
    enough that the whole thing can be reasoned about by hand.
    """
    return create_scenario(
        client,
        deliveries=SMALL_DELIVERIES,
        vehicles=[{"id": "V0", "capacity": 3}],
    )


def create_grid_fleet(client) -> dict:
    """Two vehicles over the grid's whole set of intersections, half each."""
    half = len(GRID_STOPS) // 2
    return create_scenario(
        client,
        deliveries=[
            {"id": f"D{index:02d}", "node": node, "demand": 1}
            for index, node in enumerate(GRID_STOPS)
        ],
        vehicles=[
            {"id": "V0", "capacity": half},
            {"id": "V1", "capacity": half},
        ],
    )


def start(client, scenario_id: str, **body) -> dict:
    response = client.post(
        f"/scenarios/{scenario_id}/watcher",
        json={**DETERMINISTIC, **START_DEFAULTS, **ONE_SECOND, **body},
    )
    assert response.status_code == 201, response.text
    return response.json()


def status(client, scenario_id: str) -> dict:
    response = client.get(f"/scenarios/{scenario_id}/watcher")
    assert response.status_code == 200, response.text
    return response.json()


def avoid(client, scenario_id: str, vehicle_id: str, edge, treatment=SLOW, **body):
    """Call the route and hand back the raw response, refusal or not.

    Deliberately not asserting a status: a 409 is an answer this file tests for,
    and one that has to be readable rather than raised.
    """
    return client.post(
        f"/scenarios/{scenario_id}/vehicles/{vehicle_id}/avoid-road",
        json={
            "edge": {"u": edge[0], "v": edge[1]},
            "treatment": treatment,
            "include_geometry": False,
            **body,
        },
    )


def road_of(client, scenario_id: str, vehicle_id: str) -> tuple:
    """The road the fleet says this vehicle is on."""
    for vehicle in status(client, scenario_id)["vehicles"]:
        if vehicle["vehicle_id"] == vehicle_id:
            edge = vehicle["edge"]
            assert edge is not None, "a dispatched vehicle starts on a road"
            return edge["u"], edge["v"]
    raise AssertionError(f"no vehicle {vehicle_id!r} in the fleet")


def furthest_from_finishing(client, scenario_id: str) -> str:
    """The vehicle with the most of its own route still ahead of it.

    The fleet spreads itself at ``(i + 1) / (n + 1)`` of each corridor, so this is
    the one ``initial_tracks`` placed first — and picking it by name would be
    asserting a fact about the placement rather than about the endpoint.
    """
    vehicles = status(client, scenario_id)["vehicles"]
    return min(vehicles, key=lambda vehicle: vehicle["progress"])["vehicle_id"]


def vehicle_json(plan: dict, vehicle_id: str) -> dict:
    for vehicle in plan["vehicles"]:
        if vehicle["vehicle_id"] == vehicle_id:
            return vehicle
    raise AssertionError(f"no vehicle {vehicle_id!r} in the response")


def route_json(plan: dict, side: str) -> dict:
    return {
        route["vehicle_id"]: route for route in plan[side]
    }


def ordered_ids(route: dict) -> list[str]:
    return [stop["delivery_id"] for stop in route["stops"]]


def served_ids(plan: dict) -> set[str]:
    return {
        stop["delivery_id"] for route in plan["after"] for stop in route["stops"]
    }


# --------------------------------------------------------------------------- #
# Unit: where a vehicle may begin a new plan
# --------------------------------------------------------------------------- #
def standing(
    node=None, edge=None, stuck=False, remaining=(0,), vehicle_id="V0"
) -> VehicleProgress:
    return VehicleProgress(
        vehicle_id=vehicle_id,
        vehicle_index=0,
        node=node,
        completed=(),
        remaining=remaining,
        elapsed=0.0,
        remaining_capacity=1.0,
        stuck=stuck,
        edge=edge,
    )


def test_a_rolling_vehicle_starts_from_the_far_end_of_its_road() -> None:
    """The ordinary rule, and the one Prompt 8 already uses.

    Starting from the *next* intersection rather than the last one passed is what
    keeps a re-planned route from opening with a leg the vehicle has driven.
    """
    assert restart_node(standing(node="b", edge=("a", "b"))) == "b"


def test_avoiding_your_own_road_starts_from_the_near_end() -> None:
    """A road given up cannot be planned through, so the far end is unreachable."""
    assert restart_node(standing(node="b", edge=("a", "b")), ("a", "b")) == "a"


def test_avoiding_somebody_elses_road_changes_nothing() -> None:
    """Only the road ahead matters; a closure elsewhere is a cost, not a wall."""
    assert restart_node(standing(node="b", edge=("a", "b")), ("c", "d")) == "b"


def test_a_stopped_vehicle_restarts_from_the_near_end_of_what_blocked_it() -> None:
    """The case ``reoptimize`` cannot answer, and the reason ``edge`` is carried.

    A stuck vehicle is one ``None`` from ``position``, and without the road it is
    stopped behind there would be no intersection to begin from at all.
    """
    stopped = standing(node=None, edge=("a", "b"), stuck=True)
    assert restart_node(stopped) == "a"
    assert restart_node(stopped, ("c", "d")) == "a"


def test_a_vehicle_on_no_road_has_nowhere_to_begin() -> None:
    """Never dispatched, or finished. Both are refusals rather than plans."""
    assert restart_node(standing(node=None, edge=None, remaining=())) is None


# --------------------------------------------------------------------------- #
# API: the start node follows reachability
# --------------------------------------------------------------------------- #
def test_a_slow_report_leaves_the_vehicle_planning_on_from_its_road(
    client, store, watchers
) -> None:
    """A road left drivable is still driven, so the ordinary rule stands."""
    scenario = create_small(client)
    start(client, scenario["scenario_id"])
    edge = road_of(client, scenario["scenario_id"], "V0")

    response = avoid(client, scenario["scenario_id"], "V0", edge, treatment=SLOW)
    assert response.status_code == 200, response.text
    plan = response.json()

    assert plan["trigger"]["primary"] == "override"
    assert plan["replanned_vehicles"] == ["V0"]
    assert vehicle_json(plan, "V0")["node"] == edge[1]


def test_a_closure_on_your_own_road_turns_you_around(client, store, watchers) -> None:
    """The same road, closed, moves the start to the intersection behind it.

    This is the whole of the reachability rule, and the one thing a driver's
    report changes that is not about cost: the vehicle cannot be sent through a
    road that is shut, so its new route begins where it last *was* rather than
    where it was going.
    """
    scenario = create_small(client)
    start(client, scenario["scenario_id"])
    edge = road_of(client, scenario["scenario_id"], "V0")

    response = avoid(client, scenario["scenario_id"], "V0", edge, treatment=CLOSURE)
    assert response.status_code == 200, response.text
    plan = response.json()

    assert vehicle_json(plan, "V0")["node"] == edge[0]
    assert vehicle_json(plan, "V0")["edge"] == {"u": edge[0], "v": edge[1]}


def test_the_two_treatments_disagree_about_where_the_new_route_begins(
    client, store, watchers
) -> None:
    """One road, two reports, two different starts — on identical scenarios.

    Built twice rather than called twice on one scenario, so the second report is
    not answering a network the first one already changed.
    """
    slow = create_small(client)
    start(client, slow["scenario_id"])
    closed = create_small(client)
    start(client, closed["scenario_id"])

    slow_edge = road_of(client, slow["scenario_id"], "V0")
    closed_edge = road_of(client, closed["scenario_id"], "V0")
    assert slow_edge == closed_edge, "the same scenario must dispatch the same route"

    as_slow = avoid(client, slow["scenario_id"], "V0", slow_edge, treatment=SLOW)
    as_closed = avoid(client, closed["scenario_id"], "V0", closed_edge, treatment=CLOSURE)
    assert as_slow.status_code == as_closed.status_code == 200

    assert vehicle_json(as_slow.json(), "V0")["node"] == slow_edge[1]
    assert vehicle_json(as_closed.json(), "V0")["node"] == closed_edge[0]


def test_a_stopped_vehicle_can_be_restarted_even_though_reoptimize_refuses_it(
    client, store, watchers
) -> None:
    """The case the manual override exists for.

    A closure is injected on the road a vehicle is on. The fleet now reports it as
    stopped, ``POST /reoptimize`` has no vehicle left to move its load to and
    refuses — and the driver, who knows exactly which road is the problem, is
    routed from the intersection behind it.
    """
    scenario = create_scenario(
        client,
        deliveries=[
            {"id": "D0", "node": 1, "demand": 1},
            {"id": "D1", "node": 2, "demand": 1},
            {"id": "D2", "node": 1, "demand": 1},
        ],
        vehicles=[{"id": "V0", "capacity": 3}],
    )
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)
    blocked = road_of(client, scenario_id, "V0")

    injected = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": CLOSURE, "edge": {"u": blocked[0], "v": blocked[1]}},
    )
    assert injected.status_code == 201, injected.text

    # The fence. A fleet-wide re-plan has nowhere to put this vehicle's load.
    fleet_wide = client.post(f"/scenarios/{scenario_id}/reoptimize", json=None)
    assert fleet_wide.status_code == 409, fleet_wide.text

    # And the driver's own report, which can. The road avoided here is the
    # *reverse* of the one that stopped them — a different road, so the start
    # moves back because they are stopped, not because they named it.
    reverse = (blocked[1], blocked[0])
    response = avoid(client, scenario_id, "V0", reverse, treatment=SLOW)
    assert response.status_code == 200, response.text
    plan = response.json()

    assert vehicle_json(plan, "V0")["node"] == blocked[0]
    assert vehicle_json(plan, "V0")["stuck"] is False
    assert vehicle_json(plan, "V0")["remaining"], "it still had its load on board"


# --------------------------------------------------------------------------- #
# API: only one vehicle is in the problem
# --------------------------------------------------------------------------- #
def test_no_other_vehicle_is_touched(grid_client, store, watchers) -> None:
    """The claim this endpoint makes, checked against the routes themselves.

    The scoping is structural — one vehicle is in the instance that was solved —
    but that is an argument about the code, and this compares the *output*: every
    vehicle the plan did not solve for has the same stops in the same order at the
    same cost on both sides of the call. A splice in the wrong index, or a fleet
    handed to the solver instead of a vehicle, would show up here.
    """
    scenario = create_grid_fleet(grid_client)
    scenario_id = scenario["scenario_id"]
    start(grid_client, scenario_id)

    target = furthest_from_finishing(grid_client, scenario_id)
    other = "V1" if target == "V0" else "V0"

    response = avoid(grid_client, scenario_id, target, road_of(grid_client, scenario_id, target))
    assert response.status_code == 200, response.text
    plan = response.json()

    assert plan["replanned_vehicles"] == [target]
    assert vehicle_json(plan, target)["remaining"], (
        "the scenario should leave the reporting vehicle with work still to do"
    )

    before, after = route_json(plan, "before"), route_json(plan, "after")
    assert set(before) == set(after) == {"V0", "V1"}

    # The comparison below is only about something if there is a route on the
    # other side of it. A vehicle that delivered everything it was carrying has
    # an empty route on both sides, and the invariance would hold vacuously.
    assert vehicle_json(plan, other)["remaining"], (
        f"{other} has finished its route, so it has nothing left for this test to "
        f"find unchanged"
    )

    unchanged = before[other]
    assert ordered_ids(after[other]) == ordered_ids(unchanged)
    assert after[other]["travel_cost"] == pytest.approx(unchanged["travel_cost"])
    assert after[other]["travel_time"] == pytest.approx(unchanged["travel_time"])
    assert after[other]["load"] == pytest.approx(unchanged["load"])

    # The reporting vehicle keeps the same work and may only reorder it: nothing
    # can change hands, because there is nobody for it to change hands to.
    assert sorted(ordered_ids(after[target])) == sorted(ordered_ids(before[target]))
    assert plan["moved"] == []


def test_the_whole_fleet_is_in_the_before_and_after(grid_client, store, watchers) -> None:
    """Both halves cover every vehicle, so the diff reads as a fleet plan."""
    scenario = create_grid_fleet(grid_client)
    scenario_id = scenario["scenario_id"]
    start(grid_client, scenario_id)

    target = furthest_from_finishing(grid_client, scenario_id)
    response = avoid(
        grid_client, scenario_id, target, road_of(grid_client, scenario_id, target)
    )
    assert response.status_code == 200, response.text
    plan = response.json()

    assert [route["vehicle_id"] for route in plan["before"]] == ["V0", "V1"]
    assert [route["vehicle_id"] for route in plan["after"]] == ["V0", "V1"]

    # And both halves cover the same deliveries — the plan the fleet was driving
    # and the plan it is handed serve one set of work, not two amounts of it.
    before_stops = {
        stop["delivery_id"] for route in plan["before"] for stop in route["stops"]
    }
    assert served_ids(plan) == before_stops

    # `replanned` is the *instance*, not the fleet: the reporting vehicle's own
    # remaining stops, which is the whole of what was solved.
    reporting = next(
        route for route in plan["before"] if route["vehicle_id"] == target
    )
    assert set(plan["replanned"]) == set(ordered_ids(reporting))
    assert len(plan["replanned"]) < len(before_stops), (
        "the fleet's work is bigger than the one vehicle's share of it"
    )

    # Which is only worth anything if the other vehicle is carrying some of that
    # work. A vehicle that has already finished is empty on both sides of the
    # call, and `before_stops` would then be the reporting vehicle's own stops.
    other = "V1" if target == "V0" else "V0"
    assert vehicle_json(plan, other)["remaining"], (
        f"the fleet's remaining work is the reporting vehicle's alone — {other} "
        f"has finished — so the two halves of this plan are the same one route"
    )


def test_a_completed_stop_is_not_in_the_new_plan(grid_client, store, watchers) -> None:
    """The invariant carries over from the fleet-wide re-optimization.

    Completed deliveries are not in the derived instance either, so the same
    structural argument holds: a solver cannot name one.
    """
    scenario = create_grid_fleet(grid_client)
    scenario_id = scenario["scenario_id"]
    start(grid_client, scenario_id)
    target = furthest_from_finishing(grid_client, scenario_id)

    response = avoid(
        grid_client, scenario_id, target, road_of(grid_client, scenario_id, target)
    )
    assert response.status_code == 200, response.text
    plan = response.json()

    completed = set(plan["completed"])
    assert completed, "the fleet should have delivered something by now"
    assert not (completed & served_ids(plan))
    for vehicle in plan["vehicles"]:
        assert not (completed & set(vehicle["remaining"]))


# --------------------------------------------------------------------------- #
# API: the report is a real incident, and it is revertable
# --------------------------------------------------------------------------- #
def test_the_report_lands_on_the_scenario_and_can_be_reverted(client, store) -> None:
    """``avoid-road`` writes. That is the point, and it has to be undoable.

    Filing the report as a real incident is what keeps ``/reoptimize``'s trigger
    rule intact rather than carving a hole in it: the driver changed the network,
    and the next fleet-wide call is justified by a live incident like any other.
    """
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)
    edge = road_of(client, scenario_id, "V0")

    response = avoid(client, scenario_id, "V0", edge, treatment=CLOSURE)
    assert response.status_code == 200, response.text
    plan = response.json()

    filed = plan["incident"]
    assert filed is not None and filed["incident_type"] == CLOSURE
    assert filed["edge"] == {"u": edge[0], "v": edge[1]}

    stored = client.get(f"/scenarios/{scenario_id}").json()
    assert [item["incident_id"] for item in stored["conditions"]["incidents"]] == [
        filed["incident_id"]
    ]
    assert stored["conditions"]["mutated"] is True

    reverted = client.delete(
        f"/scenarios/{scenario_id}/incident/{filed['incident_id']}"
    )
    assert reverted.status_code == 200, reverted.text
    assert client.get(f"/scenarios/{scenario_id}").json()["conditions"]["mutated"] is False


def test_a_driver_report_justifies_the_fleet_wide_reoptimize_too(client, store, watchers) -> None:
    """The two endpoints agreeing, which is the whole reason the report is real.

    Nothing is forced and nothing is bypassed: the driver's own road report is a
    live incident, so a call that answered 409 a moment ago now has a reason.
    """
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)

    refused = client.post(f"/scenarios/{scenario_id}/reoptimize", json=None)
    assert refused.status_code == 409, refused.text

    edge = road_of(client, scenario_id, "V0")
    assert avoid(client, scenario_id, "V0", edge).status_code == 200

    allowed = client.post(
        f"/scenarios/{scenario_id}/reoptimize", json={"include_geometry": False}
    )
    assert allowed.status_code == 200, allowed.text
    body = allowed.json()
    assert "incident" in body["trigger"]["kinds"]
    assert body["replanned_vehicles"], "a fleet-wide re-plan solves for its fleet"


def test_the_report_is_attributed_to_the_vehicle_that_made_it(
    client, store, watchers
) -> None:
    """The trigger names the vehicle, the road and the word it was given.

    None of the three is recoverable from the scenario afterwards — the incident
    records the road and the treatment, but not who asked — so this is the only
    place a reader can find out whose report it was.
    """
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)
    edge = road_of(client, scenario_id, "V0")

    plan = avoid(client, scenario_id, "V0", edge, treatment=SLOW).json()
    trigger = plan["trigger"]

    assert trigger["primary"] == "override"
    assert trigger["kinds"] == ["override"], "and never doubled up with the incident"
    assert "V0" in trigger["detail"]
    assert "slow" in trigger["detail"]
    assert trigger["edges"] == [{"u": edge[0], "v": edge[1]}]


# --------------------------------------------------------------------------- #
# API: what the route refuses
# --------------------------------------------------------------------------- #
def test_an_unknown_vehicle_is_a_404(client, store) -> None:
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)

    response = avoid(client, scenario_id, "V9", (0, 1))
    assert response.status_code == 404, response.text
    assert "V0" in response.json()["detail"], "the refusal should name the real fleet"


def test_a_road_that_is_not_in_the_graph_is_a_422(client, store) -> None:
    """The same check, and the same answer, ``POST /incident`` gives.

    ``(3, 0)`` is the spar's missing half: the tour graph has a road *out* to the
    dead end and none coming back, so it is a real pair of nodes and not a road —
    which is exactly the distinction the check is about.
    """
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)

    response = avoid(client, scenario_id, "V0", (3, 0))
    assert response.status_code == 422, response.text
    assert "no road" in response.json()["detail"]


def test_a_vehicle_with_nothing_left_is_a_409(client, store, watchers) -> None:
    """Nothing to re-plan is an answer, not a failure — and it writes nothing.

    The step is well past the longest corridor this graph can produce for the
    scenario, so the vehicle has served everything it was dispatched with rather
    than merely being near the end of it.
    """
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id, time_scale=60.0 / IDLE_INTERVAL)

    response = avoid(client, scenario_id, "V0", (0, 1))
    assert response.status_code == 409, response.text
    assert "unserved stops" in response.json()["detail"]
    assert client.get(f"/scenarios/{scenario_id}").json()["conditions"]["mutated"] is False


def test_a_scenario_with_no_fleet_is_a_404(client, store) -> None:
    """No fleet, no positions — the same 404 ``/reoptimize`` gives."""
    scenario = create_small(client)
    response = avoid(client, scenario["scenario_id"], "V0", (0, 1))
    assert response.status_code == 404, response.text


def test_no_scenario_at_all_is_a_404(client) -> None:
    response = avoid(client, "nope", "V0", (0, 1))
    assert response.status_code == 404, response.text


def test_an_unknown_solver_is_a_422_and_writes_nothing(client, store) -> None:
    """The solver is resolved *before* the report is filed, so a request refused
    for its arguments has not already changed the network underneath it."""
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)

    response = avoid(
        client, scenario_id, "V0", road_of(client, scenario_id, "V0"), solver="nope"
    )
    assert response.status_code == 422, response.text
    assert client.get(f"/scenarios/{scenario_id}").json()["conditions"]["mutated"] is False


def test_an_exact_solver_measures_the_scoped_instance_not_the_scenario(
    client, store, watchers
) -> None:
    """A driver with a handful of stops left can be solved exactly inside a big job.

    ``/reoptimize`` already makes this distinction — the limit is checked against
    what remains rather than against what was stored — and a driver-scoped re-plan
    is smaller still, so it inherits it with room to spare. Twelve deliveries is
    past brute force's limit; six unserved stops is not.
    """
    scenario = create_scenario(
        client,
        deliveries=[
            {"id": f"D{index:02d}", "node": 1 + index % 2, "demand": 1}
            for index in range(12)
        ],
        vehicles=[{"id": "V0", "capacity": 12}],
    )
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)
    assert scenario["n_deliveries"] > MAX_EXACT_DELIVERIES

    response = avoid(
        client,
        scenario_id,
        "V0",
        road_of(client, scenario_id, "V0"),
        solver="brute_force",
    )
    assert response.status_code == 200, response.text
    plan = response.json()

    assert plan["solver"] == "brute_force"
    assert 0 < len(plan["replanned"]) <= MAX_EXACT_DELIVERIES


def test_an_ordinary_plan_is_still_refused_when_nothing_has_happened(
    client, store, watchers
) -> None:
    """The rule this endpoint sits beside, unchanged by sitting beside it."""
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)

    response = client.post(f"/scenarios/{scenario_id}/reoptimize", json=None)
    assert response.status_code == 409, response.text
    assert "no live incident" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Unit: the scenario fixtures are doing what the tests above assume
# --------------------------------------------------------------------------- #
def test_the_grid_gives_every_delivery_its_own_intersection() -> None:
    """No two deliveries share a node, so no leg of any route is zero-length.

    That is the property the untouched-vehicle comparison leans on: it is what
    keeps a vehicle a third of the way along its corridor from having finished.
    """
    assert len(GRID_STOPS) == GRID * GRID - 1
    assert len(set(GRID_STOPS)) == len(GRID_STOPS)
    assert 0 not in GRID_STOPS


def test_the_grid_leaves_both_vehicles_with_work_left(
    grid_client, store, watchers
) -> None:
    """The premise of the untouched-vehicle comparisons, asserted not assumed.

    ``initial_tracks`` spreads two vehicles at a third and two thirds of their
    corridors. Being a third of the way along is only *partway through the work*
    if the delivery legs outweigh the leg home — otherwise the vehicle two thirds
    along has delivered everything and is driving an empty route back, and the
    invariance checks would be comparing two empty routes. Every intersection but
    the depot being a delivery is what makes that hold here; this pins it down.

    Asked of the same response the other two tests read, rather than of the
    watcher's status view — those report the same fleet, and ``remaining`` here is
    the exact field they assert on.
    """
    scenario = create_grid_fleet(grid_client)
    scenario_id = scenario["scenario_id"]
    start(grid_client, scenario_id)

    target = furthest_from_finishing(grid_client, scenario_id)
    response = avoid(
        grid_client, scenario_id, target, road_of(grid_client, scenario_id, target)
    )
    assert response.status_code == 200, response.text

    for vehicle in response.json()["vehicles"]:
        assert vehicle["remaining"], (
            f"{vehicle['vehicle_id']} has nothing left to deliver, so the "
            f"untouched-vehicle comparisons have no route of its to compare"
        )


def test_the_small_scenario_leaves_its_vehicle_with_work_to_do(
    client, store, watchers
) -> None:
    """The premise of the start-node cases, asserted rather than assumed."""
    scenario = create_small(client)
    scenario_id = scenario["scenario_id"]
    start(client, scenario_id)
    edge = road_of(client, scenario_id, "V0")

    response = avoid(client, scenario_id, "V0", edge)
    assert response.status_code == 200, response.text
    vehicle = vehicle_json(response.json(), "V0")
    assert vehicle["completed"], "halfway along, it should have served something"
    assert vehicle["remaining"], "and should still have something left"

