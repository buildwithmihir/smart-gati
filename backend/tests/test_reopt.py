"""Adaptive re-optimization: what moves, and — mostly — what cannot.

The claim this file exists to defend is the one the brief leads with:
**a completed stop can never be reassigned.** It is defended structurally rather
than by inspection. Completed deliveries are not in the instance that is solved,
so the sharpest assertion available is the direct one — no delivery id reported as
completed appears in any ``after`` route, for any solver, at any budget — and it is
asserted that way rather than by checking that some filter ran.

Two kinds of test, deliberately separated
------------------------------------------
The **unit** tests build the fleet's state by hand. That is the only way to be
exact about a re-optimization: where a simulated fleet has driven to after N ticks
is a fact about the tour graph and the solver's route, and a test that recomputed
it would be asserting its own arithmetic rather than the code's. Handing
``reoptimize`` a known position, a known set of completed stops and a known
remaining capacity makes every claim in it checkable, and makes the *swap* — two
vehicles in each other's territory, where the optimum is provably for each to take
the other's stop — something that can be set up on purpose instead of hoped for.

The **API** tests drive the real thing end to end, and assert only properties that
hold wherever the fleet happens to be: the pool is the union of what remains, the
after routes cover exactly the pool, completed stops are in none of them, and
nothing was written. What they give up in specificity they pay back in being true
of the actual endpoint.

The tour graph here is the same two-corridor one the other fleet tests use.
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
from qgati.fleet.pings import CostOf, VehicleTrack
from qgati.optimizer import Delivery, Depot, Scenario, Vehicle
from qgati.reopt import (
    NothingToReplan,
    VehicleProgress,
    before_view,
    fleet_progress,
    reoptimize,
)
from qgati.traffic import SLOW, TrafficLogStore

ARTERIAL_S = 10.0
SIDE_STREET_S = 11.0

#: 02:00 prices as ``normal``, so every modelled road costs its own base time.
NORMAL_TIME = "2026-09-21T02:00:00+05:30"

LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05
TOY_SPEED_KPH = 30.0
SPUR_S = 500.0

#: A long interval paired with a tiny scale: a tick advances the fleet by
#: ``interval_seconds * time_scale`` seconds while leaving the *background thread*
#: effectively asleep, so only the explicit ticks in a test move anything. Same
#: device the fleet tests use, and for the same reason.
IDLE_INTERVAL = 300.0

#: Seconds of driving per tick in the API tests. Small enough that the fleet is
#: still on its first legs several ticks in; see the tour graph's geometry below.
SCRIPTED_STEP_SECONDS = 1.0
ONE_SECOND = {
    "interval_seconds": IDLE_INTERVAL,
    "time_scale": SCRIPTED_STEP_SECONDS / IDLE_INTERVAL,
}

#: A larger step, chosen so the two-stop scenario's single vehicle has served its
#: first stop by the time the start tick returns. The corridor is depot -> 1 -> 2
#: -> depot at 10 + 10 + 10 seconds, and ``initial_tracks`` starts that vehicle
#: half way along it, at 15 s. Four more seconds lands it inside the second leg,
#: which is past the first stop and short of the second — the one moment this test
#: file needs. See :func:`test_the_fleet_is_where_the_test_thinks_it_is`, which
#: asserts that rather than trusting it.
MID_ROUTE_SECONDS = 4.0
MID_ROUTE = {
    "interval_seconds": IDLE_INTERVAL,
    "time_scale": MID_ROUTE_SECONDS / IDLE_INTERVAL,
}

#: A deterministic solver: these tests are about the fleet and the re-plan, not
#: about the search, and QPSO's routes depend on a seed.
DETERMINISTIC = {"solver": "savings"}

#: A fixed seed, for the same reason the fleet tests take one: at
#: ``NOISE_SIGMA = 0.06`` the detector's fallback trips on ordinary noise roughly
#: once in a thousand readings, so an unseeded run of these tests would fail on a
#: draw about that often. A seed makes that a steady pass or an immediate,
#: reproducible failure. 3 is the seed ``test_an_ordinary_reading_is_not_flagged``
#: pins, so the fleet here is the one already known not to flag itself.
START_DEFAULTS = {"seed": 3}


def _length_of(travel_time: float) -> float:
    return travel_time * (TOY_SPEED_KPH / 3.6)


def build_tour_graph() -> nx.DiGraph:
    """Two disjoint corridors through three nodes, plus a dead-end spur."""
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
        0, 3, highway="residential", travel_time=SPUR_S, weight=SPUR_S,
        length=_length_of(SPUR_S),
    )
    return graph


def build_swap_graph() -> nx.DiGraph:
    """Three nodes, every pair directly connected, and deliberately asymmetric.

    Full connectivity is what makes the costs *exact* rather than shortest paths
    through something else: with an edge for every ordered pair, every leg is the
    direct one and the cost matrix is the numbers written here. Asymmetry is what
    makes the swap case below a real decision — with symmetric costs, two vehicles
    in each other's territory would be equally happy to stay put.
    """
    graph = nx.DiGraph()
    for node in (0, 1, 2):
        graph.add_node(node, x=LON_ORIGIN + node * DEGREE_SPAN, y=LAT_ORIGIN)

    for u, v, seconds in (
        (0, 1, 10.0), (1, 0, 11.0),
        (0, 2, 11.0), (2, 0, 10.0),
        (1, 2, 10.0), (2, 1, 11.0),
    ):
        graph.add_edge(
            u, v, highway="secondary", travel_time=seconds, weight=seconds,
            length=_length_of(seconds),
        )
    return graph


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tour_graph() -> nx.DiGraph:
    return build_tour_graph()


@pytest.fixture(scope="module")
def swap_graph() -> nx.DiGraph:
    return build_swap_graph()


@pytest.fixture
def log_store() -> TrafficLogStore:
    return TrafficLogStore(":memory:")


@pytest.fixture
def store() -> ScenarioStore:
    return ScenarioStore()


@pytest.fixture
def watchers() -> WatcherRegistry:
    """A registry of its own, so one test's fleet cannot tick against another's."""
    return WatcherRegistry()


@pytest.fixture
def client(tour_graph, store, log_store, watchers):
    """An app over the tour graph, with a fleet registry that lives for the test.

    The run history is in-memory like the traffic log: every solve here — the
    dispatch, the re-plan — writes a row, and a test run must not append to the
    developer's real ``backend/data/run_history.db``.
    """
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: tour_graph
    application.dependency_overrides[get_store] = lambda: store
    application.dependency_overrides[get_log_store] = lambda: log_store
    application.dependency_overrides[get_run_store] = lambda: RunLogStore(":memory:")
    application.dependency_overrides[get_watchers] = lambda: watchers
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def create_scenario(client, *, vehicles=None, deliveries=None, timestamp=NORMAL_TIME):
    """Create a scenario through the API.

    Two stops at nodes 1 and 2 by default, with one vehicle that can carry both.
    ``vehicles`` is how the multi-vehicle case is set up — two one-unit vehicles
    cannot share a route, so the solver must dispatch both.
    """
    response = client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": 0},
            "deliveries": deliveries
            or [
                {"id": "D0", "node": 1, "demand": 1},
                {"id": "D1", "node": 2, "demand": 1},
            ],
            "vehicles": vehicles or [{"id": "V0", "capacity": 2}],
            "conditions": {"timestamp": timestamp},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def start(client, scenario_id: str, **body) -> dict:
    response = client.post(
        f"/scenarios/{scenario_id}/watcher",
        json={**DETERMINISTIC, **START_DEFAULTS, **ONE_SECOND, **body},
    )
    assert response.status_code == 201, response.text
    return response.json()


def tick(client, scenario_id: str) -> dict:
    response = client.post(f"/scenarios/{scenario_id}/watcher/tick")
    assert response.status_code == 200, response.text
    return response.json()


def reoptimize_call(client, scenario_id: str, **body):
    return client.post(f"/scenarios/{scenario_id}/reoptimize", json=body or None)


def edge_json(edge) -> dict:
    return {"u": edge[0], "v": edge[1]}


def vehicle_edge(client, scenario_id: str, index: int = 0):
    """The road the ``index``-th vehicle is currently on, as the API reports it."""
    vehicles = client.get(f"/scenarios/{scenario_id}/watcher").json()["vehicles"]
    edge = vehicles[index]["edge"]
    assert edge is not None, "a dispatched vehicle starts on a road"
    return edge["u"], edge["v"]


def report_slow(client, scenario_id: str, edge) -> dict:
    response = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": SLOW, "edge": edge_json(edge)},
    )
    assert response.status_code == 201, response.text
    return response.json()


def stop_ids(route: dict) -> set[str]:
    return {stop["delivery_id"] for stop in route["stops"]}


def all_stop_ids(routes: list[dict]) -> set[str]:
    return {stop_id for route in routes for stop_id in stop_ids(route)}


# --------------------------------------------------------------------------- #
# Unit: how far along its route a vehicle is
# --------------------------------------------------------------------------- #
#: A corridor depot -> 1 -> 2 -> depot, one edge per leg, ten seconds each.
CORRIDOR_EDGES = ((0, 1), (1, 2), (2, 0))
CORRIDOR_COSTS: dict = {edge: 10.0 for edge in CORRIDOR_EDGES}


def corridor_track(travelled: float, **overrides) -> VehicleTrack:
    """A vehicle two stops into a three-leg corridor, at ``travelled`` seconds.

    ``leg_ends`` is what :func:`~qgati.fleet.pings.initial_tracks` would record:
    the corridor index at which each leg finishes, so ``(1, 2, 3)`` for three
    one-edge legs. The last is the return to the depot rather than a stop, which
    is exactly the entry :func:`~qgati.fleet.pings._stops_behind` drops.
    """
    fields = {
        "vehicle_id": "V0",
        "edges": CORRIDOR_EDGES,
        "travelled": travelled,
        "delivery_indices": (0, 1),
        "leg_ends": (1, 2, 3),
    }
    fields.update(overrides)
    return VehicleTrack(**fields)


def costs(overrides: dict | None = None) -> CostOf:
    """A cost lookup over the corridor, with roads overridden to ``None``.

    A mapping rather than ``**kwargs``, because an edge is a tuple of nodes and a
    keyword argument's name has to be a string.
    """
    table = {**CORRIDOR_COSTS, **(overrides or {})}
    return lambda edge: table.get(edge)


@pytest.mark.parametrize(
    ("travelled", "completed"),
    [
        (0.0, 0),    # at the depot, nothing served
        (5.0, 0),    # on the way to the first stop
        (10.0, 1),   # exactly at the first stop
        (15.0, 1),   # between the two
        (20.0, 2),   # at the second
        (29.9, 2),   # on the way home
        (30.0, 2),   # home: every stop served
        (300.0, 2),  # long past the end
    ],
)
def test_completed_stops_follows_the_corridor(travelled, completed) -> None:
    """Progress is read off the corridor, at every position that matters.

    The boundaries are the point: a vehicle exactly *at* a stop has served it, and
    a vehicle at the depot has served both and must not claim a third — which is
    what dropping the final leg boundary prevents.
    """
    assert completed_stops_of(corridor_track(travelled)) == completed


def completed_stops_of(track: VehicleTrack, cost_of: CostOf | None = None) -> int:
    from qgati.fleet.pings import completed_stops

    return completed_stops(track, cost_of or costs())


def test_a_vehicle_stopped_by_a_closure_keeps_only_what_it_delivered() -> None:
    """The distinction the whole derivation turns on, and the bug it nearly had.

    A vehicle behind a closure is in the same ``None`` from
    :func:`~qgati.fleet.pings.position` as one that has finished its route, and
    the two mean opposite things for a re-optimization: a finished vehicle has
    nothing left to hand anyone, and a stopped one is carrying exactly the load
    that needs reassigning. Reporting a stopped vehicle as finished — which
    "count every stop, the walk ran out" would do — makes its undelivered load
    vanish silently, which is the worst available failure.
    """
    track = corridor_track(15.0)
    blocked = costs({(1, 2): None})

    assert completed_stops_of(track, blocked) == 1


def test_a_track_with_no_leg_boundaries_reports_nothing_completed() -> None:
    """Defensive, and the right way round: unknown progress is not full progress."""
    assert completed_stops_of(corridor_track(15.0, leg_ends=())) == 0


# --------------------------------------------------------------------------- #
# Unit: reading the fleet
# --------------------------------------------------------------------------- #
def two_stop_scenario(**overrides) -> Scenario:
    fields = {
        "depot": Depot(node=0, lat=0.0, lon=0.0),
        "deliveries": (
            Delivery(id="D0", node=1, demand=1),
            Delivery(id="D1", node=2, demand=1),
        ),
        "vehicles": (Vehicle(id="V0", capacity=2), Vehicle(id="V1", capacity=2)),
    }
    fields.update(overrides)
    return Scenario(**fields)


def test_progress_partitions_the_dispatched_route() -> None:
    """Completed and remaining are the two halves of one route, and they add up.

    The partition is the whole contract: what is in ``completed`` may never be
    touched, and ``remaining`` is the entire extent of what may move. A gap or an
    overlap between them would be a delivery either lost or delivered twice.
    """
    scenario = two_stop_scenario(
        vehicles=(Vehicle(id="V0", capacity=2),)
    )
    tracks = (corridor_track(15.0, vehicle_id="V0"),)

    (progress,) = fleet_progress(scenario, tracks, costs())

    assert progress.completed == (0,)
    assert progress.remaining == (1,)
    assert tuple(progress.completed) + tuple(progress.remaining) == (0, 1)
    assert progress.elapsed == 15.0
    assert progress.node == 2, "the head of the edge it is on is where a plan starts"


def test_remaining_capacity_frees_up_as_a_vehicle_unloads() -> None:
    """What bounds anything a re-optimization adds to a vehicle.

    Quoting the capacity it set out with would let a re-plan hand a half-empty
    vehicle the load it has already dropped.
    """
    scenario = two_stop_scenario(vehicles=(Vehicle(id="V0", capacity=5),))
    tracks = (corridor_track(15.0, vehicle_id="V0"),)

    (progress,) = fleet_progress(scenario, tracks, costs())

    assert progress.remaining_capacity == 4.0, "one unit of two delivered"


def test_a_finished_vehicle_is_finished_rather_than_empty() -> None:
    """Distinct states, because they lead to different places.

    A finished vehicle is not in the new fleet, and the reason is ``remaining``:
    it has nothing left to serve. It keeps room it will never use — a capacity of
    three against two units delivered leaves one to spare — precisely because
    capacity is *not* what excludes it. A rule that admitted any vehicle with room
    and work would be sending a vehicle that has already gone home back out.

    A stuck vehicle is the opposite case: load on board and nowhere to put it,
    which is what the pool is for.
    """
    scenario = two_stop_scenario(vehicles=(Vehicle(id="V0", capacity=3),))
    track = corridor_track(40.0, vehicle_id="V0")

    (progress,) = fleet_progress(scenario, tracks=(track,), cost_of=costs())

    assert progress.finished is True
    assert progress.available is False
    assert progress.remaining == ()
    assert progress.completed == (0, 1)
    assert progress.remaining_capacity == pytest.approx(1.0), (
        "three units of capacity less the two it delivered"
    )


def test_a_vehicle_with_no_track_is_reported_rather_than_omitted() -> None:
    """One entry per vehicle, always, so the result lines up with the fleet.

    ``initial_tracks`` skips a vehicle whose route is empty, and an index-shifted
    answer would silently attribute one vehicle's progress to another.
    """
    scenario = two_stop_scenario()
    tracks = (corridor_track(15.0, vehicle_id="V0"),)

    first, second = fleet_progress(scenario, tracks, costs())

    assert first.vehicle_id == "V0" and first.vehicle_index == 0
    assert second.vehicle_id == "V1" and second.vehicle_index == 1
    assert second.completed == () and second.remaining == ()
    assert second.node is None
    assert second.finished is True, "a vehicle that never left has nothing outstanding"


# --------------------------------------------------------------------------- #
# Unit: the re-optimization itself
# --------------------------------------------------------------------------- #
def swap_progress() -> tuple[VehicleProgress, ...]:
    """Two vehicles, each carrying the other's stop, each able to carry only one.

    Vehicle 0 is at node 2 holding the delivery that belongs at node 1; vehicle 1
    is at node 1 holding the one that belongs at node 2. Neither has delivered
    anything, so nothing is pinned and the whole assignment is in play.
    """
    return (
        VehicleProgress(
            vehicle_id="V0", vehicle_index=0, node=2, completed=(), remaining=(0,),
            elapsed=0.0, remaining_capacity=1.0,
        ),
        VehicleProgress(
            vehicle_id="V1", vehicle_index=1, node=1, completed=(), remaining=(1,),
            elapsed=0.0, remaining_capacity=1.0,
        ),
    )


def test_a_re_plan_can_hand_a_stop_to_the_vehicle_that_is_next_to_it(
    swap_graph: nx.DiGraph,
) -> None:
    """The point of re-optimizing at all, made provable rather than likely.

    Each vehicle is sitting on the other's delivery. Serving what it holds means
    driving across the network and back; swapping means one of them serves a stop
    without moving and the other drives home from where it already is. Brute force
    settles which, so this is an optimal answer rather than a plausible one — and
    both deliveries change hands, which is what makes the before/after in the demo
    a real re-plan rather than the same plan solved twice.
    """
    scenario = two_stop_scenario(
        vehicles=(Vehicle(id="V0", capacity=1), Vehicle(id="V1", capacity=1))
    )

    plan = reoptimize(
        scenario=scenario,
        progress=swap_progress(),
        graph=swap_graph,
        solver_key="brute_force",
    )

    assert {(move.delivery, move.from_vehicle, move.to_vehicle) for move in plan.moved} == {
        (0, "V0", "V1"),
        (1, "V1", "V0"),
    }
    # Vehicle 0 stays at node 2 and serves the stop that is there; vehicle 1, at
    # node 1, does the same. Derived index 0 is D0 (node 1) and index 1 is D1
    # (node 2), so the routes are the mirror of the assignments they replace.
    assert plan.solution.routes == ((1,), (0,))


def test_completed_deliveries_are_absent_from_the_instance_entirely(
    swap_graph: nx.DiGraph,
) -> None:
    """Not filtered out of the answer — never in the question.

    The distinction is the whole design. A plan that removed completed stops
    after solving would be one solver change away from putting them back; this
    asserts the stronger thing, that the instance the solver was handed does not
    contain them, which is why no solver can reassign one.
    """
    scenario = two_stop_scenario(
        vehicles=(Vehicle(id="V0", capacity=1), Vehicle(id="V1", capacity=1))
    )
    progress = (
        VehicleProgress(
            vehicle_id="V0", vehicle_index=0, node=2, completed=(0,), remaining=(),
            elapsed=0.0, remaining_capacity=1.0,
        ),
        VehicleProgress(
            vehicle_id="V1", vehicle_index=1, node=1, completed=(), remaining=(1,),
            elapsed=0.0, remaining_capacity=1.0,
        ),
    )

    plan = reoptimize(
        scenario=scenario, progress=progress, graph=swap_graph, solver_key="brute_force"
    )

    assert plan.pool == (1,)
    assert plan.instance.delivery_ids() == ("D1",)
    assert len(plan.instance.vehicles) == 1
    assert plan.instance.vehicles[0].id == "V1"


def test_the_reduced_fleet_holds_the_reduced_capacity(swap_graph: nx.DiGraph) -> None:
    """A vehicle that has delivered is not offered the room it has already spent."""
    scenario = two_stop_scenario(
        vehicles=(Vehicle(id="V0", capacity=4), Vehicle(id="V1", capacity=4))
    )
    progress = (
        VehicleProgress(
            vehicle_id="V0", vehicle_index=0, node=2, completed=(), remaining=(0,),
            elapsed=0.0, remaining_capacity=1.5,
        ),
        VehicleProgress(
            vehicle_id="V1", vehicle_index=1, node=1, completed=(), remaining=(1,),
            elapsed=0.0, remaining_capacity=3.0,
        ),
    )

    plan = reoptimize(
        scenario=scenario, progress=progress, graph=swap_graph, solver_key="brute_force"
    )

    assert plan.instance.capacities == (1.5, 3.0)
    assert plan.instance.starts == (2, 1), "each begins where it currently is"


def test_an_exhausted_fleet_is_refused_rather_than_solved(swap_graph: nx.DiGraph) -> None:
    """A re-optimization with nothing to optimize is an answer, not an error.

    Raising is what lets the route turn it into a 409 with a reason rather than
    returning an empty plan that looks like success.
    """
    scenario = two_stop_scenario(vehicles=(Vehicle(id="V0", capacity=2),))
    progress = (
        VehicleProgress(
            vehicle_id="V0", vehicle_index=0, node=None, completed=(0, 1),
            remaining=(), elapsed=90.0, remaining_capacity=2.0,
        ),
    )

    with pytest.raises(NothingToReplan, match="every delivery has been served"):
        reoptimize(
            scenario=scenario, progress=progress, graph=swap_graph,
            solver_key="brute_force",
        )


def test_load_no_one_can_carry_is_refused_with_the_arithmetic(
    swap_graph: nx.DiGraph,
) -> None:
    """The stuck-vehicle case, which is the one place the pool can overflow.

    The pool is every vehicle's undelivered load, but the capacity is only the
    *available* vehicles'. A closure stranding a loaded vehicle therefore leaves
    more demand in play than the rest of the fleet can carry — which is a real
    answer, and one that left to ``Scenario.__post_init__`` would surface as a
    message about total demand: true, and useless for working out why a scenario
    that was feasible a moment ago is not.
    """
    scenario = two_stop_scenario(
        deliveries=(
            Delivery(id="D0", node=1, demand=3),
            Delivery(id="D1", node=2, demand=1),
        ),
        vehicles=(Vehicle(id="V0", capacity=4), Vehicle(id="V1", capacity=1)),
    )
    progress = (
        # Stranded with three units on board that nobody else has room for.
        VehicleProgress(
            vehicle_id="V0", vehicle_index=0, node=None, completed=(), remaining=(0,),
            elapsed=30.0, remaining_capacity=4.0, stuck=True,
        ),
        VehicleProgress(
            vehicle_id="V1", vehicle_index=1, node=1, completed=(), remaining=(1,),
            elapsed=30.0, remaining_capacity=1.0,
        ),
    )

    with pytest.raises(NothingToReplan, match="no one else has room for"):
        reoptimize(
            scenario=scenario, progress=progress, graph=swap_graph,
            solver_key="brute_force",
        )


def test_an_exact_solver_is_judged_on_what_remains(swap_graph: nx.DiGraph) -> None:
    """The limit is measured on the re-optimization, not on the stored scenario.

    A scenario past brute force's limit can be one with three stops left, and
    refusing from the original size would be refusing work the exact solver could
    have closed — which is why the route resolves the solver key itself rather
    than going through the helper that measures the stored instance. Both halves
    are asserted here: allowed against what remains, refused against what does not.
    """
    from qgati.optimizer.brute_force import MAX_EXACT_DELIVERIES
    from qgati.reopt import SolverTooSmall

    n = MAX_EXACT_DELIVERIES + 2
    scenario = Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=tuple(
            Delivery(id=f"D{index}", node=1 + (index % 2), demand=1) for index in range(n)
        ),
        vehicles=(Vehicle(id="V0", capacity=n), Vehicle(id="V1", capacity=n)),
    )

    def progress_with(remaining: tuple[int, ...]) -> tuple[VehicleProgress, ...]:
        """Vehicle 0 holds ``remaining``; vehicle 1 has finished and drops out."""
        delivered = tuple(index for index in range(n) if index not in remaining)
        return (
            VehicleProgress(
                vehicle_id="V0", vehicle_index=0, node=2, completed=delivered,
                remaining=remaining, elapsed=0.0,
                remaining_capacity=float(n - len(delivered)),
            ),
            VehicleProgress(
                vehicle_id="V1", vehicle_index=1, node=1, completed=(), remaining=(),
                elapsed=0.0, remaining_capacity=float(n),
            ),
        )

    # One stop left of twelve: an exact solver closes it, and must be allowed to.
    plan = reoptimize(
        scenario=scenario,
        progress=progress_with((n - 1,)),
        graph=swap_graph,
        solver_key="brute_force",
    )

    assert plan.pool == (n - 1,)
    assert plan.solver == "brute_force"

    # The whole twelve left: the same solver, on an instance it cannot close.
    with pytest.raises(SolverTooSmall, match="limited to"):
        reoptimize(
            scenario=scenario,
            progress=progress_with(tuple(range(n))),
            graph=swap_graph,
            solver_key="brute_force",
        )


# --------------------------------------------------------------------------- #
# Unit: the before half
# --------------------------------------------------------------------------- #
def test_the_before_view_prices_each_vehicle_where_it_stands(
    swap_graph: nx.DiGraph,
) -> None:
    """Comparability, which the before/after table depends on entirely.

    Pricing the old routes from the depot would compare a new plan measured from
    each vehicle's real position against an old one measured from a place none of
    them is — and the depot is 10-11 seconds from both stops here, so the two
    answers differ by enough to notice.
    """
    scenario = two_stop_scenario(
        vehicles=(Vehicle(id="V0", capacity=1), Vehicle(id="V1", capacity=1))
    )

    priced, matrix = before_view(
        scenario, swap_progress(), swap_graph, weight="weight"
    )

    assert priced.starts == (2, 1)
    assert matrix.start_index(0) == matrix.node_index(2)
    assert matrix.start_index(1) == matrix.node_index(1)
    # Vehicle 0 is *at* node 2 and its stop is at node 1, so the outbound leg is
    # the 11-second road rather than the 11 seconds from the depot by coincidence
    # — 2 -> 1 is 11 and 0 -> 1 is 10, which is what makes this worth asserting.
    assert matrix.cost(matrix.start_index(0), matrix.node_index(1)) == pytest.approx(11.0)
    assert matrix.cost(matrix.depot_index, matrix.node_index(1)) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Unit: the windows
# --------------------------------------------------------------------------- #
def test_windows_are_rebased_onto_the_vehicle_clock(swap_graph: nx.DiGraph) -> None:
    """Seconds from departure, shifted by the least-elapsed participating vehicle.

    The shift cannot be exact — a window belongs to the delivery and elapsed time
    belongs to whichever vehicle serves it, which after re-optimization may be a
    different one. Shifting by the *smallest* elapsed is the optimistic choice, and
    it is the one that guarantees the shift alone never turns a reachable window
    into an unreachable one.
    """
    deliveries = (
        Delivery(id="D0", node=1, demand=1, earliest_arrival=100.0, latest_arrival=200.0),
        Delivery(id="D1", node=2, demand=1),
    )
    scenario = two_stop_scenario(
        deliveries=deliveries,
        vehicles=(Vehicle(id="V0", capacity=1), Vehicle(id="V1", capacity=1)),
    )
    progress = (
        VehicleProgress(
            vehicle_id="V0", vehicle_index=0, node=2, completed=(), remaining=(0,),
            elapsed=60.0, remaining_capacity=1.0,
        ),
        VehicleProgress(
            vehicle_id="V1", vehicle_index=1, node=1, completed=(), remaining=(1,),
            elapsed=90.0, remaining_capacity=1.0,
        ),
    )

    plan = reoptimize(
        scenario=scenario, progress=progress, graph=swap_graph, solver_key="brute_force"
    )

    rebased = plan.instance.deliveries[0]
    assert rebased.earliest_arrival == pytest.approx(40.0)
    assert rebased.latest_arrival == pytest.approx(140.0)
    assert plan.instance.has_time_windows is True


def test_a_window_that_closed_while_driving_becomes_immediate(
    swap_graph: nx.DiGraph,
) -> None:
    """Clamped at zero rather than going negative, and never inverted.

    A window that expired in transit is one the re-plan is already late for, and
    the objective should price that lateness rather than the model refusing to
    represent it. Order survives the subtraction, so the result is an empty-looking
    but *valid* window — earliest equal to latest, never past it, which is what
    ``Delivery`` rejects.
    """
    deliveries = (
        Delivery(id="D0", node=1, demand=1, earliest_arrival=5.0, latest_arrival=9.0),
        Delivery(id="D1", node=2, demand=1),
    )
    scenario = two_stop_scenario(
        deliveries=deliveries,
        vehicles=(Vehicle(id="V0", capacity=1), Vehicle(id="V1", capacity=1)),
    )
    progress = (
        VehicleProgress(
            vehicle_id="V0", vehicle_index=0, node=2, completed=(), remaining=(0,),
            elapsed=60.0, remaining_capacity=1.0,
        ),
        VehicleProgress(
            vehicle_id="V1", vehicle_index=1, node=1, completed=(), remaining=(1,),
            elapsed=60.0, remaining_capacity=1.0,
        ),
    )

    plan = reoptimize(
        scenario=scenario, progress=progress, graph=swap_graph, solver_key="brute_force"
    )

    rebased = plan.instance.deliveries[0]
    assert rebased.earliest_arrival == 0.0
    assert rebased.latest_arrival == 0.0


# --------------------------------------------------------------------------- #
# API: what it refuses, and how
# --------------------------------------------------------------------------- #
def test_reoptimizing_an_unknown_scenario_is_404(client) -> None:
    assert reoptimize_call(client, "nope").status_code == 404


def test_reoptimizing_without_a_fleet_is_404(client) -> None:
    """The user's decision, enforced: state comes from the running fleet.

    There is no body-declared fleet state, so a scenario with no fleet has no
    positions, no completed stops and no remaining capacity — and the detail names
    the route that starts one rather than leaving a caller to guess.
    """
    created = create_scenario(client)

    response = reoptimize_call(client, created["scenario_id"])

    assert response.status_code == 404
    assert "no fleet running" in response.json()["detail"]


def test_nothing_having_happened_is_a_conflict(client) -> None:
    """The structural guarantee that this is not a re-solve button.

    A fleet is running and there is work left, so the request is well-formed —
    and it still fails, because neither an incident nor a flagged reading
    justifies it. Without this the endpoint would be indistinguishable from
    ``/optimize`` with extra steps, and "adaptive" would be a claim in a demo
    script rather than a property of the API.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    response = reoptimize_call(client, scenario_id)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "no live incident" in detail
    assert "flagged no anomalous reading" in detail


def test_an_unknown_solver_is_refused(client) -> None:
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    report_slow(client, scenario_id, vehicle_edge(client, scenario_id))

    response = reoptimize_call(client, scenario_id, solver="nope")

    assert response.status_code == 422
    assert "nope" in response.json()["detail"]


def test_an_incident_is_enough_on_its_own(client) -> None:
    """A report needs no corroboration: an operator closing a road is reason enough.

    Reported on a road *ahead* of the fleet so the re-plan runs on the ordinary
    case — the readings on the road under the vehicle are a separate test below,
    and mixing them here would make this one assert two things.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    report_slow(client, scenario_id, (2, 0))

    response = reoptimize_call(client, scenario_id, solver="savings")

    assert response.status_code == 200, response.text
    trigger = response.json()["trigger"]
    assert trigger["kinds"] == ["incident"]
    assert trigger["primary"] == "incident"
    assert trigger["edges"] == [{"u": 2, "v": 0}]


def test_a_flagged_reading_is_a_trigger_and_carries_its_reason(client) -> None:
    """The anomaly half, and the evidence is the detector's own, not the script's.

    Reported slow on the road the vehicle is *on*, so the fleet measures a road
    under an incident: the reading is drawn around the flat placeholder and lands
    far past the detector's margin, which flags it. Both triggers then hold, and
    both are reported — "we were told" and "we measured" are different strengths
    of evidence, and collapsing them would throw one away.

    The reason string is passed through verbatim rather than summarised, because
    it names the numbers the detector judged.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    edge = vehicle_edge(client, scenario_id)
    report_slow(client, scenario_id, edge)

    body = tick(client, scenario_id)
    flagged = [reading for reading in body["readings"] if reading["flag"]]
    assert flagged, f"the road under an incident should be flagged: {body['readings']}"

    response = reoptimize_call(client, scenario_id, solver="savings")

    assert response.status_code == 200, response.text
    trigger = response.json()["trigger"]
    assert set(trigger["kinds"]) == {"incident", "anomaly"}
    assert trigger["reasons"] == [reading["reason"] for reading in flagged]


def test_a_fleet_that_has_finished_is_a_conflict_not_a_plan(client) -> None:
    """Nothing left to serve is an answer, and not one that looks like success."""
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id, time_scale=400.0)
    report_slow(client, scenario_id, (2, 0))
    tick(client, scenario_id)

    response = reoptimize_call(client, scenario_id, solver="savings")

    assert response.status_code == 409
    assert "nothing left to re-plan" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# API: the invariant, end to end
# --------------------------------------------------------------------------- #
@pytest.fixture
def mid_route(client) -> str:
    """A fleet whose vehicle has served its first stop and not its second.

    Built from the API's own answers rather than from arithmetic about the tour
    graph: :func:`test_the_fleet_is_where_the_test_thinks_it_is` asserts the
    premise on its own, so a change to ``initial_tracks`` fails one obvious test
    rather than five confusing ones.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id, **MID_ROUTE)
    return scenario_id


def test_the_fleet_is_where_the_test_thinks_it_is(client, mid_route) -> None:
    """The premise every test below rests on, asserted rather than assumed.

    One vehicle, two stops, halfway round its corridor: neither at the depot nor
    finished, which is the only kind of moment at which a re-optimization has both
    something to pin and something to move. If this stops holding, the tests that
    follow are not wrong so much as vacuous, and a vacuous test is worse than a
    failing one.
    """
    scenario_id = mid_route

    (vehicle,) = client.get(f"/scenarios/{scenario_id}/watcher").json()["vehicles"]

    assert 0.0 < vehicle["progress"] < 1.0
    assert vehicle["finished"] is False
    assert vehicle["edge"] is not None


def test_the_completed_stops_are_named_and_absent_from_the_new_plan(
    client, mid_route
) -> None:
    """The headline claim, on the real endpoint.

    A delivery the fleet has handed over appears in ``completed`` and in **no**
    ``after`` route — asserted over every vehicle rather than the one it came
    from, because "it stayed with its own vehicle" is not the guarantee. The
    guarantee is that it is nowhere, which is what building the instance without
    it buys.
    """
    scenario_id = mid_route
    edge = vehicle_edge(client, scenario_id)
    report_slow(client, scenario_id, edge)

    response = reoptimize_call(client, scenario_id, solver="savings")

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["completed"], (
        "the fixture's whole premise is a fleet that has delivered something; "
        f"got {body['vehicles']}"
    )
    assert not set(body["completed"]) & all_stop_ids(body["after"]), (
        "a completed delivery was handed to somebody in the new plan"
    )
    assert set(body["completed"]).isdisjoint(all_stop_ids(body["before"]))


def test_the_two_halves_cover_the_same_work(client, mid_route) -> None:
    """Before and after are two answers to one question, not two questions.

    ``before`` is the rest of the plan the fleet is already driving and ``after``
    is the new assignment of exactly those stops, so the ids must match set for
    set. A before that included a completed stop, or an after that dropped one,
    would make the comparison between them meaningless.
    """
    scenario_id = mid_route
    report_slow(client, scenario_id, vehicle_edge(client, scenario_id))

    body = reoptimize_call(client, scenario_id, solver="savings").json()

    pool = set(body["replanned"])
    assert pool == all_stop_ids(body["before"])
    assert pool == all_stop_ids(body["after"])
    assert pool == {
        delivery_id
        for vehicle in body["vehicles"]
        for delivery_id in vehicle["remaining"]
    }
    assert set(body["completed"]) | pool == {"D0", "D1"}


def test_the_answer_says_where_every_vehicle_is(client, mid_route) -> None:
    """The state the plan was built from, reported rather than implied.

    Every vehicle appears, including one with nothing left — and ``finished`` and
    ``available`` are what tell the two kinds of idle vehicle apart, so a reader
    is not left working out why the new plan has fewer vehicles than the old one.
    """
    scenario_id = mid_route
    report_slow(client, scenario_id, vehicle_edge(client, scenario_id))

    body = reoptimize_call(client, scenario_id, solver="savings").json()

    assert [vehicle["vehicle_id"] for vehicle in body["vehicles"]] == ["V0"]
    (vehicle,) = body["vehicles"]
    assert vehicle["elapsed_seconds"] > 0.0
    assert vehicle["remaining_capacity"] >= 0.0
    assert vehicle["finished"] is not vehicle["available"]
    assert vehicle["node"] is not None


def test_the_new_plan_starts_from_where_the_vehicle_is(client, mid_route) -> None:
    """Not from the depot — the ``starts`` plumbing, visible in the answer.

    A one-vehicle fleet with one stop left makes this checkable to the second.
    The vehicle is at node 2, which is where its remaining stop is, so its new
    route is that stop and then the 10-second road home: **10 seconds**. A route
    priced from the depot would run depot -> node 2 -> depot, which on this graph
    is 11 + 10 = 21. The gap between the two is the whole of what per-vehicle
    starts buy, and it is asserted as the number rather than as an inequality
    because an inequality here would pass on a coincidence.
    """
    scenario_id = mid_route
    edge = vehicle_edge(client, scenario_id)
    report_slow(client, scenario_id, edge)

    body = reoptimize_call(client, scenario_id, solver="savings").json()

    (after,) = body["after"]
    assert [stop["delivery_id"] for stop in after["stops"]] == ["D1"]
    assert after["travel_time"] == pytest.approx(10.0), (
        "the leg home from where it stands, not the depot's 11 + 10"
    )


# --------------------------------------------------------------------------- #
# API: the boundary
# --------------------------------------------------------------------------- #
def test_re_optimizing_writes_nothing(client, store, mid_route) -> None:
    """A read, and this is what makes it one.

    Rewriting the stored scenario would make a second call behave differently
    from the first for no reason a caller asked for, and re-dispatching the fleet
    means resetting how far each vehicle has travelled — a simulation decision
    rather than an optimizer one. Both are absent, so the record is untouched and
    the fleet has not moved.
    """
    scenario_id = mid_route
    report_slow(client, scenario_id, vehicle_edge(client, scenario_id))
    before = store.get(scenario_id)

    assert reoptimize_call(client, scenario_id, solver="savings").status_code == 200

    after = store.get(scenario_id)
    assert after.traffic_rows_logged == before.traffic_rows_logged
    assert after.observations == before.observations
    assert after.cost_matrix is before.cost_matrix
    assert len(after.incidents) == len(before.incidents)


def test_the_fleet_keeps_ticking_afterwards(client, mid_route) -> None:
    """Re-optimizing reads the tracks, so it must leave them exactly as it found them."""
    scenario_id = mid_route
    report_slow(client, scenario_id, vehicle_edge(client, scenario_id))
    before = client.get(f"/scenarios/{scenario_id}/watcher").json()["ticks"]

    reoptimize_call(client, scenario_id, solver="savings")
    body = tick(client, scenario_id)

    assert body["tick"] == before + 1
    assert client.get(f"/scenarios/{scenario_id}/watcher").json()["running"] is True


def test_the_same_fleet_gives_the_same_plan_twice(client, mid_route) -> None:
    """Determinism, because re-optimizing is a read of an unchanged fleet.

    Two calls, nothing in between, must agree on every route — otherwise the
    endpoint is reporting something other than what it was asked about, and a
    client cannot tell a real change from noise.
    """
    scenario_id = mid_route
    report_slow(client, scenario_id, vehicle_edge(client, scenario_id))

    first = reoptimize_call(client, scenario_id, solver="savings").json()
    second = reoptimize_call(client, scenario_id, solver="savings").json()

    assert [route["stops"] for route in first["after"]] == [
        route["stops"] for route in second["after"]
    ]
    assert first["replanned"] == second["replanned"]
    assert first["completed"] == second["completed"]
