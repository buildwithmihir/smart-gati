"""Traffic simulation and log tests.

Two hand-built graphs carry most of the weight here, because the properties
being tested are about *which route wins*, and a random graph cannot promise
that a particular comparison flips. Each graph isolates one mechanism:

``build_reroute_graph``
    One leg with two ways to drive it, arranged so peak hour moves a single
    shortest path from the arterial onto side streets.
``build_tour_graph``
    Two disjoint corridors between the same three nodes, arranged so peak hour
    reverses the better order to visit two stops in — the tour-level flip, which
    is what a demo actually shows.

Both are directed graphs with integer nodes and ``x``/``y`` coordinates, so they
also exercise the API's coordinate and geometry paths without a cached Delhi
extract.

Times are built with a fixed ``UTC+05:30`` offset rather than
``ZoneInfo("Asia/Kolkata")``: Windows ships no system zone database, so
``zoneinfo`` would need the ``tzdata`` package and these tests would fail on a
machine that had not installed it.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from qgati.analytics import RunLogStore
from qgati.api.main import create_app, get_graph, get_log_store, get_run_store, get_store
from qgati.api.store import ScenarioStore
from qgati.optimizer import Delivery, Depot, Scenario, Vehicle, solve_brute_force
from qgati.traffic import (
    MODERATE,
    MODERATE_FACTOR,
    NORMAL,
    PEAK,
    PEAK_FACTOR,
    ActiveConditions,
    Edge,
    TrafficLogRow,
    TrafficLogStore,
    TrafficState,
    congestion_state,
    edge_of,
    get_traffic_multiplier,
    is_peak_hour,
    price_scenario,
    road_class_of,
    sensitivity_of,
    simulated_travel_time,
    traffic_weight_function,
)

#: Delhi's offset. Fixed, deliberately — see the module docstring.
IST = timezone(timedelta(hours=5, minutes=30))

#: One timestamp per congestion state, named for the state rather than the clock
#: position. 02:00 is outside the daytime band, 14:00 is inside it but between
#: peak windows, and 09:00 is in the morning peak.
NORMAL_TIME = datetime(2026, 9, 21, 2, 0, tzinfo=IST)
MODERATE_TIME = datetime(2026, 9, 21, 14, 0, tzinfo=IST)
PEAK_TIME = datetime(2026, 9, 21, 9, 0, tzinfo=IST)

#: The synthetic coordinate box, mirroring tests/test_api.py.
LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05

#: What a side street's multiplier works out to in each state, from
#: ``1 + (FACTOR - 1) * 0.30``.
LOCAL_MODERATE = 1.0 + (MODERATE_FACTOR - 1.0) * 0.30
LOCAL_PEAK = 1.0 + (PEAK_FACTOR - 1.0) * 0.30


def _place(graph: nx.Graph, positions: dict[int, tuple[float, float]]) -> None:
    """Give nodes the ``x``/``y`` degrees OSMnx graphs carry."""
    for node, (x, y) in positions.items():
        graph.add_node(
            node,
            x=LON_ORIGIN + x * DEGREE_SPAN,
            y=LAT_ORIGIN + y * DEGREE_SPAN,
        )


#: Speed the hand-built fixtures below are assumed to be driven at. Their numbers
#: are travel times, and the objective prices distance and fuel as well, so every
#: edge needs a length to price. A constant speed is the one case where the fuel
#: curve contributes nothing beyond a fixed multiple of distance, which keeps
#: these tests about the traffic layer rather than about the fuel model.
#:
#: This matters for what the tests below assert. Congestion scales *times* and
#: leaves lengths alone, so a congested leg has a lower implied average speed and
#: a higher fuel-per-kilometre — which is the effect the fuel term exists to
#: capture, and it is why the flip these tests pin still happens on the combined
#: objective rather than only on time.
TOY_SPEED_KPH = 30.0


def _length_of(travel_time: float) -> float:
    """Metres covered in ``travel_time`` seconds at the fixture's speed."""
    return travel_time * (TOY_SPEED_KPH / 3.6)


def build_reroute_graph() -> nx.DiGraph:
    """One leg, two ways to drive it, favouring different roads by time of day.

    ``0 -> 1`` is a single arterial hop costing 10s. ``0 -> 2 -> 1`` is two side
    streets at 6s each, so 12s. In the normal band the arterial wins on both
    counts; from moderate on it loses, because congestion is weighted by road
    class — x1.6 against side-street x1.18 makes it 16s against 14.16s, and at
    peak the gap only widens (29s against 18.84s). The shortest path moves onto
    the side streets and stays there.
    """
    graph = nx.DiGraph()
    _place(graph, {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.4, 0.6)})

    graph.add_edge(
        0, 1, highway="secondary", travel_time=10.0, weight=10.0,
        length=_length_of(10.0),
    )
    graph.add_edge(
        1, 0, highway="secondary", travel_time=10.0, weight=10.0,
        length=_length_of(10.0),
    )
    for u, v in ((0, 2), (2, 1), (1, 2), (2, 0)):
        graph.add_edge(
            u, v, highway="residential", travel_time=6.0, weight=6.0,
            length=_length_of(6.0),
        )
    return graph


def build_tour_graph() -> nx.DiGraph:
    """Two disjoint corridors between three nodes, for a tour-level flip.

    Arterials form the cycle ``0 -> 1 -> 2 -> 0`` at 10s a hop; side streets form
    the reverse cycle at 11s a hop. In the normal band the arterial cycle is
    cheaper — 30 against 33 — so a two-stop tour visits 1 then 2. From moderate
    on it is not: the arterial cycle costs 48 against 38.94 at moderate, and 87
    against 51.81 at peak, so the better tour is the reverse. Same stops, same
    fleet, different route.
    """
    graph = nx.DiGraph()
    _place(graph, {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)})

    for u, v in ((0, 1), (1, 2), (2, 0)):
        graph.add_edge(
            u, v, highway="secondary", travel_time=10.0, weight=10.0,
            length=_length_of(10.0),
        )
    for u, v in ((0, 2), (2, 1), (1, 0)):
        graph.add_edge(
            u, v, highway="residential", travel_time=11.0, weight=11.0,
            length=_length_of(11.0),
        )
    return graph


def two_stop_scenario() -> Scenario:
    """Depot at 0, deliveries at 1 and 2, one vehicle carrying both."""
    return Scenario(
        depot=Depot(node=0, lat=LAT_ORIGIN, lon=LON_ORIGIN),
        deliveries=(
            Delivery(id="D0", node=1, demand=1),
            Delivery(id="D1", node=2, demand=1),
        ),
        vehicles=(Vehicle(id="V0", capacity=2),),
    )


@pytest.fixture
def tour_graph() -> nx.DiGraph:
    return build_tour_graph()


@pytest.fixture
def tour_client(tour_graph):
    """An app over the tour graph, with stores that live for the test.

    Both overrides must be *callables returning one instance*, not the classes
    themselves: a dependency override is invoked per request, so passing the
    class would hand every request a brand-new empty store — and for the log,
    a brand-new empty in-memory database.

    Overriding the log store matters beyond isolation: without it every scenario
    created by a test would append to the developer's real
    ``backend/data/traffic_log.db``. The run history is overridden for the same
    reason — this file calls ``/optimize``, which writes a row to it.
    """
    application = create_app()
    store = ScenarioStore()
    log_store = TrafficLogStore(":memory:")
    application.dependency_overrides[get_graph] = lambda: tour_graph
    application.dependency_overrides[get_store] = lambda: store
    application.dependency_overrides[get_log_store] = lambda: log_store
    application.dependency_overrides[get_run_store] = lambda: RunLogStore(":memory:")
    with TestClient(application) as client:
        yield client
    application.dependency_overrides.clear()


def create_tour_scenario(client, **conditions):
    """Create the two-stop scenario through the API, returning the response."""
    payload = {
        "kind": "explicit",
        "depot": {"node": 0},
        "deliveries": [
            {"id": "D0", "node": 1, "demand": 1},
            {"id": "D1", "node": 2, "demand": 1},
        ],
        "vehicles": [{"id": "V0", "capacity": 2}],
    }
    if conditions:
        payload["conditions"] = conditions
    response = client.post("/scenarios", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def route_nodes(client, scenario_id, solver="brute_force"):
    """The solved stop order, as scenario node ids."""
    response = client.post(
        f"/optimize/{scenario_id}",
        json={"solver": solver, "include_geometry": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    return [
        [stop["node"] for stop in route["stops"]] for route in body["routes"]
    ]


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def test_the_normal_state_costs_nothing_extra() -> None:
    """Outside the daytime band nothing is congested, so no road class is marked
    up — the excess over 1.0 that sensitivity scales is zero."""
    for road_class in ("secondary", "residential", ""):
        assert get_traffic_multiplier(Edge(1, 2, road_class), NORMAL_TIME) == 1.0


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (7, 59, False),  # just before the morning peak
        (8, 0, True),  # the window is half-open: 08:00 is in
        (9, 59, True),
        (10, 0, False),  # ...and 10:00 is out
        (16, 59, False),
        (17, 0, True),  # evening peak
        (19, 59, True),
        (20, 0, False),
        (2, 0, False),
    ],
)
def test_peak_window_boundaries(hour, minute, expected) -> None:
    assert is_peak_hour(datetime(2026, 9, 21, hour, minute, tzinfo=IST)) is expected


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (2, NORMAL),
        (5, NORMAL),
        (6, MODERATE),  # the daytime band is half-open: 06:00 is in
        (7, MODERATE),
        (8, PEAK),  # peak outranks the daytime band it sits inside...
        (10, MODERATE),  # ...and is over again by 10:00
        (16, MODERATE),
        (17, PEAK),
        (20, MODERATE),
        (21, MODERATE),
        (22, NORMAL),  # the daytime band ends at 22:00
        (23, NORMAL),
    ],
)
def test_the_clock_alone_selects_the_congestion_state(hour, expected) -> None:
    """Nothing in this table is operator input.

    The state has to follow from the clock, because a demo left running
    unattended never gets the chance to set one.
    """
    assert congestion_state(datetime(2026, 9, 21, hour, 0, tzinfo=IST)) == expected


def test_through_roads_take_the_full_peak_factor() -> None:
    assert get_traffic_multiplier(Edge(1, 2, "secondary"), PEAK_TIME) == pytest.approx(
        PEAK_FACTOR
    )
    assert get_traffic_multiplier(Edge(1, 2, "tertiary"), PEAK_TIME) == pytest.approx(
        PEAK_FACTOR
    )
    assert get_traffic_multiplier(Edge(1, 2, "primary_link"), PEAK_TIME) == pytest.approx(
        PEAK_FACTOR
    )


def test_side_streets_congest_less_than_through_roads() -> None:
    """The whole reason congestion can reroute anything.

    A uniform factor scales every tour equally and cannot change which one wins,
    so the gap between these two numbers is what makes the simulation useful
    rather than decorative.
    """
    through = get_traffic_multiplier(Edge(1, 2, "secondary"), PEAK_TIME)
    local = get_traffic_multiplier(Edge(1, 2, "residential"), PEAK_TIME)

    assert local == pytest.approx(LOCAL_PEAK)
    assert local < through


def test_moderate_is_weighted_by_road_class_too() -> None:
    """Only the normal state is flat, and only because its excess is zero. Both
    congested states scale, so either can disagree about which road is quicker."""
    assert get_traffic_multiplier(
        Edge(1, 2, "secondary"), MODERATE_TIME
    ) == pytest.approx(MODERATE_FACTOR)
    assert get_traffic_multiplier(
        Edge(1, 2, "residential"), MODERATE_TIME
    ) == pytest.approx(LOCAL_MODERATE)


def test_an_untagged_road_is_treated_as_a_quiet_one() -> None:
    """Unknown classes default to the local sensitivity — understating congestion
    is the safer direction than overstating it."""
    assert sensitivity_of("") == 0.30
    assert get_traffic_multiplier(Edge(1, 2), PEAK_TIME) == pytest.approx(LOCAL_PEAK)


def test_an_accident_only_slows_the_roads_it_names() -> None:
    conditions = ActiveConditions(accident_edges=[(1, 2)])
    assert get_traffic_multiplier(
        Edge(1, 2, "secondary"), NORMAL_TIME, conditions
    ) == pytest.approx(PEAK_FACTOR)
    assert get_traffic_multiplier(Edge(1, 3, "secondary"), NORMAL_TIME, conditions) == 1.0


def test_an_accident_is_flat_across_road_classes() -> None:
    """x2.9 on a side street exactly as on an arterial.

    Congestion is scaled by road class; this deliberately is not. It stands in
    for a delay nobody has measured yet, and erring high is the safe direction.
    """
    conditions = ActiveConditions(accident_edges=[(1, 2)])
    for road_class in ("secondary", "residential", ""):
        assert get_traffic_multiplier(
            Edge(1, 2, road_class), NORMAL_TIME, conditions
        ) == pytest.approx(PEAK_FACTOR)


def test_an_accident_replaces_the_congestion_multiplier_rather_than_compounding() -> None:
    """A crash at 5pm is not ``2.9 x 2.9``.

    The peak factor is already the conservative placeholder for an unmeasured
    delay, so stacking the two would charge the same congestion twice.
    """
    conditions = ActiveConditions(accident_edges=[(1, 2)])

    assert get_traffic_multiplier(
        Edge(1, 2, "secondary"), NORMAL_TIME, conditions
    ) == pytest.approx(PEAK_FACTOR)
    assert get_traffic_multiplier(
        Edge(1, 2, "secondary"), PEAK_TIME, conditions
    ) == pytest.approx(PEAK_FACTOR)


def test_a_closed_road_is_impassable() -> None:
    conditions = ActiveConditions(closed_edges=[(1, 2)])
    assert math.isinf(
        get_traffic_multiplier(Edge(1, 2, "secondary"), NORMAL_TIME, conditions)
    )


def test_a_closed_road_logs_no_travel_time() -> None:
    graph = build_reroute_graph()
    state = TrafficState(
        timestamp=NORMAL_TIME, conditions=ActiveConditions(closed_edges=[(0, 1)])
    )
    assert simulated_travel_time(graph, Edge(0, 1), state) is None


def test_a_closure_outranks_an_accident_on_the_same_road() -> None:
    state = TrafficState(
        timestamp=NORMAL_TIME,
        conditions=ActiveConditions(
            accident_edges=[(1, 2)], closed_edges=[(1, 2)]
        ),
    )
    assert state.incident_type(Edge(1, 2)) == "road_closure"


def test_incident_type_is_none_for_an_ordinary_road() -> None:
    state = TrafficState(timestamp=NORMAL_TIME)
    assert state.incident_type(Edge(1, 2)) is None


def test_a_naive_timestamp_is_read_as_local_time() -> None:
    """So logged rows always carry an offset, and never a bare wall clock."""
    state = TrafficState(timestamp=datetime(2026, 9, 21, 9, 0))
    assert state.timestamp.tzinfo is not None
    assert state.peak_hour is True


def test_conditions_are_read_off_the_edge_the_graph_holds() -> None:
    graph = build_tour_graph()
    assert edge_of(graph, 0, 1).road_class == "secondary"
    assert edge_of(graph, 0, 2).road_class == "residential"
    assert road_class_of({"highway": ["tertiary", "residential"]}) == "tertiary"

    with pytest.raises(KeyError):
        edge_of(graph, 0, 99)


# --------------------------------------------------------------------------- #
# Routing under conditions
# --------------------------------------------------------------------------- #
def test_congestion_reroutes_a_leg_onto_side_streets() -> None:
    """The leg-level claim, stated as a shortest path rather than a cost.

    The flip lands on the *moderate* band, not peak: x1.6 is already enough to
    make the 10s arterial (16s) lose to two 6s side streets (14.16s). Peak then
    widens a gap that moderate opened.
    """
    graph = build_reroute_graph()

    def shortest(timestamp):
        weight = traffic_weight_function(graph, TrafficState(timestamp=timestamp))
        return nx.dijkstra_path(graph, 0, 1, weight=weight)

    assert shortest(NORMAL_TIME) == [0, 1]  # the arterial, direct
    assert shortest(MODERATE_TIME) == [0, 2, 1]  # around it, via two side streets
    assert shortest(PEAK_TIME) == [0, 2, 1]


def test_a_closure_is_routed_around_rather_than_removed() -> None:
    graph = build_reroute_graph()
    state = TrafficState(
        timestamp=NORMAL_TIME, conditions=ActiveConditions(closed_edges=[(0, 1)])
    )
    path = nx.dijkstra_path(graph, 0, 1, weight=traffic_weight_function(graph, state))

    assert path == [0, 2, 1]
    # The graph itself is untouched: routing avoided the road, nothing deleted it.
    assert graph.has_edge(0, 1)


def test_the_simulated_time_equals_the_weight_routing_used() -> None:
    """The guarantee the log depends on: what gets recorded is what got charged."""
    graph = build_reroute_graph()
    state = TrafficState(timestamp=PEAK_TIME)
    weight = traffic_weight_function(graph, state)

    for u, v in ((0, 1), (0, 2), (2, 1)):
        data = graph.adj[u][v]
        assert simulated_travel_time(graph, Edge(u, v), state) == pytest.approx(
            weight(u, v, data)
        )


def test_parallel_edges_resolve_to_the_cheapest_after_multipliers() -> None:
    """A side street running alongside an arterial can win under congestion and
    lose without it, which is only true if the multipliers are applied per edge
    rather than to the cheapest base edge."""
    graph = nx.MultiDiGraph()
    _place(graph, {0: (0.0, 0.0), 1: (1.0, 0.0)})
    graph.add_edge(0, 1, key=0, highway="secondary", travel_time=10.0, weight=10.0)
    graph.add_edge(0, 1, key=1, highway="residential", travel_time=11.0, weight=11.0)

    normal = TrafficState(timestamp=NORMAL_TIME)
    peak = TrafficState(timestamp=PEAK_TIME)

    assert simulated_travel_time(graph, Edge(0, 1), normal) == pytest.approx(10.0)
    assert simulated_travel_time(graph, Edge(0, 1), peak) == pytest.approx(
        11.0 * LOCAL_PEAK
    )


# --------------------------------------------------------------------------- #
# Pricing a scenario
# --------------------------------------------------------------------------- #
def test_the_normal_and_peak_states_price_the_same_instance_differently() -> None:
    graph = build_tour_graph()
    scenario = two_stop_scenario()

    normal = price_scenario(graph, scenario, TrafficState(timestamp=NORMAL_TIME))
    peak = price_scenario(graph, scenario, TrafficState(timestamp=PEAK_TIME))

    assert not (normal.cost_matrix.matrix == peak.cost_matrix.matrix).all()


def test_peak_reverses_the_better_tour() -> None:
    """Same stops, same fleet, different route — the demonstration this whole
    layer exists to make possible."""
    graph = build_tour_graph()
    scenario = two_stop_scenario()

    normal = price_scenario(graph, scenario, TrafficState(timestamp=NORMAL_TIME))
    peak = price_scenario(graph, scenario, TrafficState(timestamp=PEAK_TIME))

    normal_route = solve_brute_force(scenario, normal.cost_matrix).routes[0]
    peak_route = solve_brute_force(scenario, peak.cost_matrix).routes[0]

    # Delivery indices: 0 is the stop at node 1, 1 is the stop at node 2.
    assert normal_route == (0, 1)
    assert peak_route == (1, 0)


def test_pricing_reports_the_roads_it_used() -> None:
    graph = build_tour_graph()
    priced = price_scenario(
        graph, two_stop_scenario(), TrafficState(timestamp=NORMAL_TIME)
    )
    assert priced.used_edges
    for u, v in priced.used_edges:
        assert graph.has_edge(u, v)


def test_pricing_logs_roads_the_scenario_touched(tmp_path) -> None:
    store = TrafficLogStore(tmp_path / "log.db")
    graph = build_tour_graph()

    priced = price_scenario(
        graph, two_stop_scenario(), TrafficState(timestamp=NORMAL_TIME), store
    )

    assert priced.rows_logged == len(priced.used_edges)
    assert store.count() == priced.rows_logged


def test_a_closed_road_is_logged_even_though_no_route_can_use_it(tmp_path) -> None:
    """The reason incident edges are logged separately from traversed ones.

    A closed edge carries infinite weight, so no cheapest path will ever include
    it. Log only the traversed roads and ``road_closure`` would never once appear
    in the table — the one observation a later model could not reconstruct.
    """
    store = TrafficLogStore(tmp_path / "log.db")
    graph = build_tour_graph()
    state = TrafficState(
        timestamp=NORMAL_TIME, conditions=ActiveConditions(closed_edges=[(0, 1)])
    )

    priced = price_scenario(graph, two_stop_scenario(), state, store)

    assert (0, 1) not in priced.used_edges  # nothing drives through it
    rows, _ = store.read(limit=1000, incident_type="road_closure")
    assert [row.road_id for row in rows] == ["0->1"]
    assert rows[0].travel_time is None
    assert rows[0].road_u == 0 and isinstance(rows[0].road_u, int)


def test_an_accident_road_is_logged_with_its_penalised_time(tmp_path) -> None:
    store = TrafficLogStore(tmp_path / "log.db")
    graph = build_tour_graph()
    state = TrafficState(
        timestamp=NORMAL_TIME, conditions=ActiveConditions(accident_edges=[(1, 2)])
    )

    price_scenario(graph, two_stop_scenario(), state, store)

    rows, _ = store.read(limit=1000, incident_type="accident")
    assert [row.road_id for row in rows] == ["1->2"]
    assert rows[0].travel_time == pytest.approx(10.0 * PEAK_FACTOR)


def test_a_failed_log_write_does_not_fail_pricing() -> None:
    """Collection is best-effort. A dataset write must never break the request
    that is serving a user right now."""

    class BrokenStore:
        def write(self, rows):
            raise RuntimeError("disk on fire")

    graph = build_tour_graph()
    priced = price_scenario(
        graph, two_stop_scenario(), TrafficState(timestamp=NORMAL_TIME), BrokenStore()
    )

    assert priced.rows_logged == 0
    assert priced.cost_matrix is not None
    assert priced.used_edges


def test_pricing_without_a_log_store_records_nothing() -> None:
    priced = price_scenario(
        build_tour_graph(), two_stop_scenario(), TrafficState(timestamp=NORMAL_TIME)
    )
    assert priced.rows_logged == 0


def test_a_closure_that_severs_the_network_is_an_error() -> None:
    """Reported, not papered over with an inf cost."""
    graph = build_tour_graph()
    scenario = two_stop_scenario()
    # Cutting both ways out of the depot leaves the instance unservable.
    state = TrafficState(
        timestamp=NORMAL_TIME,
        conditions=ActiveConditions(closed_edges=[(0, 1), (0, 2)]),
    )
    with pytest.raises(ValueError):
        price_scenario(graph, scenario, state)


# --------------------------------------------------------------------------- #
# The log store
# --------------------------------------------------------------------------- #
def test_log_round_trips_every_column(tmp_path) -> None:
    store = TrafficLogStore(tmp_path / "log.db")
    graph = build_tour_graph()
    state = TrafficState(
        timestamp=PEAK_TIME, conditions=ActiveConditions(accident_edges=[(1, 2)])
    )
    # Built through the same factory the recorder uses.
    store.write(TrafficLogRow.from_edges(graph, [(1, 2)], state))

    rows, total = store.read()
    assert total == 1
    row = rows[0]
    assert row.road_id == "1->2"
    assert row.road_u == 1 and row.road_v == 2
    assert row.day_of_week == "Monday"
    assert row.time_of_day == "09:00"
    assert row.traffic_condition == "peak"
    assert row.incident_type == "accident"
    # The accident's flat x2.9, not peak's x2.9 stacked on top of it.
    assert row.travel_time == pytest.approx(10.0 * PEAK_FACTOR)


def test_log_pagination_is_newest_first_and_totals_the_filtered_set(tmp_path) -> None:
    store = TrafficLogStore(tmp_path / "log.db")
    graph = build_tour_graph()

    for hour, minute in ((9, 0), (12, 0), (18, 0)):
        state = TrafficState(timestamp=datetime(2026, 9, 21, hour, minute, tzinfo=IST))
        store.write(TrafficLogRow.from_edges(graph, [(0, 1)], state))

    assert store.count() == 3

    page, total = store.read(limit=2, offset=0)
    assert total == 3
    assert [row.time_of_day for row in page] == ["18:00", "12:00"]  # newest first

    page, _ = store.read(limit=2, offset=2)
    assert [row.time_of_day for row in page] == ["09:00"]


def test_log_filters_narrow_the_total(tmp_path) -> None:
    store = TrafficLogStore(tmp_path / "log.db")
    graph = build_tour_graph()
    for hour in (9, 14):
        state = TrafficState(timestamp=datetime(2026, 9, 21, hour, 0, tzinfo=IST))
        store.write(TrafficLogRow.from_edges(graph, [(0, 1)], state))

    _, peak_only = store.read(traffic_condition="peak")
    _, moderate_only = store.read(traffic_condition="moderate")
    _, by_road = store.read(road_u=0, road_v=1)
    _, other_road = store.read(road_u=9, road_v=9)

    assert (peak_only, moderate_only, by_road, other_road) == (1, 1, 2, 0)


def test_log_is_empty_before_anything_is_written(tmp_path) -> None:
    store = TrafficLogStore(tmp_path / "log.db")
    rows, total = store.read()
    assert rows == [] and total == 0
    assert len(store) == 0


def test_writing_no_rows_is_a_no_op(tmp_path) -> None:
    store = TrafficLogStore(tmp_path / "log.db")
    assert store.write([]) == 0
    assert store.count() == 0


# --------------------------------------------------------------------------- #
# Through the HTTP API
# --------------------------------------------------------------------------- #
def test_a_scenario_echoes_the_conditions_it_was_priced_under(tour_client) -> None:
    body = create_tour_scenario(
        tour_client,
        timestamp="2026-09-21T09:00:00+05:30",
        accident_edges=[{"u": 0, "v": 1}],
    )
    conditions = body["conditions"]

    assert conditions["peak_hour"] is True
    assert conditions["traffic_condition"] == "peak"
    assert conditions["accident_edges"] == [{"u": 0, "v": 1}]
    assert conditions["closed_edges"] == []
    assert body["traffic_rows_logged"] > 0


def test_a_scenario_without_conditions_uses_the_clock(tour_client) -> None:
    """The plain request still produces a state, and still logs.

    Congestion is not something a caller can leave out: it comes off the
    timestamp, so even this bare request is priced under a real band.
    """
    body = create_tour_scenario(tour_client)

    assert body["conditions"]["timestamp"] is not None
    assert body["conditions"]["traffic_condition"] in {"normal", "moderate", "peak"}
    assert body["traffic_rows_logged"] > 0


def test_the_normal_and_peak_scenarios_take_different_routes(tour_client) -> None:
    """The demonstration, over HTTP: one instance, two times of day.

    Compared against the *normal* band rather than the moderate one, because
    moderate already reverses this instance's tour — see the pricing tests
    above. Normal against peak is therefore the pairing that shows a flip.
    """
    normal = create_tour_scenario(
        tour_client, timestamp="2026-09-21T02:00:00+05:30"
    )
    peak = create_tour_scenario(tour_client, timestamp="2026-09-21T09:00:00+05:30")

    normal_route = route_nodes(tour_client, normal["scenario_id"])
    peak_route = route_nodes(tour_client, peak["scenario_id"])

    assert normal_route != peak_route
    assert normal_route == [[1, 2]]  # normal: arterials, 1 then 2
    assert peak_route == [[2, 1]]  # peak: side streets, 2 then 1


def test_cost_rises_with_the_congestion_state(tour_client) -> None:
    """One instance, three times of day, three strictly increasing costs.

    The *routes* flip as well, so this is not just a uniform markup — normal
    runs the arterials at 30s, while moderate and peak both take the side
    streets, at 38.94s and 51.81s.
    """

    def cost(timestamp):
        created = create_tour_scenario(tour_client, timestamp=timestamp)
        response = tour_client.post(
            f"/optimize/{created['scenario_id']}",
            json={"solver": "brute_force", "include_geometry": False},
        )
        return response.json()["travel_cost"]

    normal = cost("2026-09-21T02:00:00+05:30")
    moderate = cost("2026-09-21T14:00:00+05:30")
    peak = cost("2026-09-21T09:00:00+05:30")

    assert normal < moderate < peak


def test_incident_edges_must_exist_in_the_road_graph(tour_client) -> None:
    response = tour_client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": 0},
            "deliveries": [{"id": "D0", "node": 1, "demand": 1}],
            "vehicles": [{"id": "V0", "capacity": 1}],
            "conditions": {"closed_edges": [{"u": 0, "v": 99}]},
        },
    )
    assert response.status_code == 422
    assert "not in the road graph" in response.json()["detail"]


def test_a_closure_that_severs_the_instance_is_reported(tour_client) -> None:
    response = tour_client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": 0},
            "deliveries": [{"id": "D0", "node": 1, "demand": 1}],
            "vehicles": [{"id": "V0", "capacity": 1}],
            "conditions": {
                "closed_edges": [{"u": 0, "v": 1}, {"u": 0, "v": 2}]
            },
        },
    )
    assert response.status_code == 422
    assert "closure" in response.json()["detail"]


def test_the_traffic_log_endpoint_returns_what_was_collected(tour_client) -> None:
    body = create_tour_scenario(tour_client, timestamp="2026-09-21T09:00:00+05:30")

    response = tour_client.get("/traffic/log", params={"limit": 5})
    assert response.status_code == 200, response.text
    page = response.json()

    assert page["total"] == body["traffic_rows_logged"]
    assert page["limit"] == 5
    assert page["offset"] == 0
    assert page["has_more"] is True

    entry = page["items"][0]
    assert entry["traffic_condition"] == "peak"
    assert entry["day_of_week"] == "Monday"
    assert entry["time_of_day"] == "09:00"
    assert entry["incident_type"] is None
    assert entry["travel_time"] is not None
    assert entry["road_id"] == f"{entry['road_u']}->{entry['road_v']}"


def test_the_traffic_log_pages_without_repeating(tour_client) -> None:
    create_tour_scenario(tour_client, timestamp="2026-09-21T09:00:00+05:30")

    first = tour_client.get("/traffic/log", params={"limit": 2}).json()
    second = tour_client.get("/traffic/log", params={"limit": 2, "offset": 2}).json()

    assert len(first["items"]) == 2
    assert {row["road_id"] for row in first["items"]} & {
        row["road_id"] for row in second["items"]
    } == set()


def test_the_traffic_log_filters_by_condition_and_incident(tour_client) -> None:
    create_tour_scenario(
        tour_client,
        timestamp="2026-09-21T09:00:00+05:30",
        closed_edges=[{"u": 0, "v": 1}],
    )

    closures = tour_client.get(
        "/traffic/log", params={"incident_type": "road_closure"}
    ).json()
    moderate = tour_client.get(
        "/traffic/log", params={"traffic_condition": "moderate"}
    ).json()

    assert closures["total"] == 1
    assert closures["items"][0]["road_id"] == "0->1"
    assert closures["items"][0]["travel_time"] is None
    assert moderate["total"] == 0


def test_the_traffic_log_rejects_an_oversized_page(tour_client) -> None:
    response = tour_client.get("/traffic/log", params={"limit": 10_000})
    assert response.status_code == 422


@pytest.fixture
def reroute_client():
    """An app over the leg-reroute graph."""
    application = create_app()
    graph = build_reroute_graph()
    store = ScenarioStore()
    log_store = TrafficLogStore(":memory:")
    application.dependency_overrides[get_graph] = lambda: graph
    application.dependency_overrides[get_store] = lambda: store
    application.dependency_overrides[get_log_store] = lambda: log_store
    application.dependency_overrides[get_run_store] = lambda: RunLogStore(":memory:")
    with TestClient(application) as client:
        yield client
    application.dependency_overrides.clear()


def one_stop_scenario_payload(**conditions):
    payload = {
        "kind": "explicit",
        "depot": {"node": 0},
        "deliveries": [{"id": "D0", "node": 1, "demand": 1}],
        "vehicles": [{"id": "V0", "capacity": 1}],
    }
    if conditions:
        payload["conditions"] = conditions
    return payload


def test_route_geometry_follows_the_route_that_was_priced(reroute_client) -> None:
    """A traced route must use the weights its matrix was built with.

    In the normal band the priced path is the direct arterial, so the drawn line
    has three points (depot, stop, depot). Under peak the priced path detours
    through node 2 in both directions, so the same stops draw five. Tracing a
    peak-priced solution on the static weights would draw the normal-band road
    instead — a road the optimizer never chose.
    """
    pinned = {}
    for label, timestamp in (
        ("normal", "2026-09-21T02:00:00+05:30"),
        ("peak", "2026-09-21T09:00:00+05:30"),
    ):
        created = reroute_client.post(
            "/scenarios", json=one_stop_scenario_payload(timestamp=timestamp)
        )
        assert created.status_code == 201, created.text
        scenario_id = created.json()["scenario_id"]

        optimized = reroute_client.post(
            f"/optimize/{scenario_id}", json={"solver": "brute_force"}
        )
        assert optimized.status_code == 200, optimized.text
        geometry = optimized.json()["routes"][0]["geometry"]
        pinned[label] = len(geometry)

        # The drawn line passes through every stop and returns to the depot.
        assert geometry[0] == geometry[-1]

    assert pinned == {"normal": 3, "peak": 5}


