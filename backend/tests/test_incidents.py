"""Live incidents: mutable per-scenario conditions, and their audit trail.

A scenario's traffic conditions are fixed when it is created, and that stays the
default. What these tests cover is the deliberate exception: an operator's report
arriving afterwards, through ``POST``/``DELETE /scenarios/{id}/incident``.

Two properties of the fixture matter to what can be asserted on it.

**It is a tour graph with two disjoint corridors.** Depot 0, stops 1 and 2;
arterials run ``0 -> 1 -> 2 -> 0`` at 10s a hop, side streets run
``0 -> 2 -> 1 -> 0`` at 11s. Every leg therefore has an alternative, which is
what lets an incident be shown moving a *cost matrix* — the thing the prompt asks
for — rather than only an edge. A fourth node carries a dead-end spur, so that
the case where an incident legitimately changes nothing can also be shown.

**Every edge is driven at one constant speed** (see ``_length_of``), so distance
and fuel are fixed multiples of travel time and the rupee objective stays
proportional to seconds. That keeps these tests about incidents rather than about
the fuel curve, and it is why the tour comparisons below can be reasoned about in
seconds.

The instance at 02:00 (``normal``, every multiplier 1.0):

    leg 0->1   10s direct, 22s via 0->2->1
    leg 0->2   11s direct, 20s via 0->1->2
    tour (1, 2)   10 + 10 + 10 = 30s   <- optimal
    tour (2, 1)   11 + 11 + 11 = 33s

A *slow* report on ``0->1`` puts that edge at 10 x 2.9 = 29s, so the leg moves to
its 22s detour and tour (1, 2) becomes 22 + 10 + 10 = 42s. The tour flips to
(2, 1) at 33s — a different route, caused by a cost change and not by a solver.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from qgati.analytics import RunLogStore
from qgati.api.main import create_app, get_graph, get_log_store, get_run_store, get_store
from qgati.api.store import ScenarioStore
from qgati.traffic import (
    CLEARED,
    CLOSURE,
    INCIDENT_MULTIPLIERS,
    SLOW,
    ActiveConditions,
    Edge,
    Incident,
    TrafficLogStore,
    TrafficState,
    conditions_for,
    get_traffic_multiplier,
)
from qgati.traffic.simulator import PEAK_FACTOR

#: Delhi's offset, as the rest of the suite uses.
IST = timezone(timedelta(hours=5, minutes=30))

#: 02:00 is outside the daytime band, so every road class prices at x1.0 and the
#: fixture's base times are the priced times. Incidents are demonstrated here
#: rather than at peak for exactly that reason — see the peak test below.
NORMAL_TIME = "2026-09-21T02:00:00+05:30"
PEAK_TIME = "2026-09-21T09:00:00+05:30"

LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05
TOY_SPEED_KPH = 30.0

#: The arterial hop, and the side-street one, in seconds.
ARTERIAL_S = 10.0
SIDE_STREET_S = 11.0

#: What a slow report does to the arterial: 10 x 2.9, flat.
SLOWED_ARTERIAL_S = ARTERIAL_S * PEAK_FACTOR

#: The same, on the side street.
SLOWED_SIDE_STREET_S = SIDE_STREET_S * PEAK_FACTOR

#: The leg 0->1 becomes its detour, 0->2->1, once the direct road is slowed.
DETOURED_LEG_S = 2 * SIDE_STREET_S

#: The leg 2->1 becomes the arterial way round, 2->0->1, once its own direct
#: side street is the slowed road.
ARTERIAL_DETOUR_S = 2 * ARTERIAL_S

SLOW_ON_0_1 = {"incident_type": SLOW, "edge": {"u": 0, "v": 1}}
SLOW_ON_2_1 = {"incident_type": SLOW, "edge": {"u": 2, "v": 1}}
CLOSURE_ON_0_1 = {"incident_type": CLOSURE, "edge": {"u": 0, "v": 1}}
CLOSURE_ON_2_1 = {"incident_type": CLOSURE, "edge": {"u": 2, "v": 1}}
#: A dead-end spur, in the graph but on nobody's cheapest path.
SPUR_ON_0_3 = {"incident_type": SLOW, "edge": {"u": 0, "v": 3}}

#: The spur's travel time, in seconds. Large and unreachable-from: node 3 has no
#: outgoing edges, so no path between the scenario's own nodes can route over it.
SPUR_S = 500.0


def _length_of(travel_time: float) -> float:
    """Metres covered in ``travel_time`` seconds at the fixture's speed."""
    return travel_time * (TOY_SPEED_KPH / 3.6)


def build_tour_graph() -> nx.DiGraph:
    """Two disjoint corridors between three nodes, for a tour-level flip.

    Plus one dead-end spur, ``0 -> 3``. Every road in the three-node core is the
    direct route for its own leg, so slowing any of them moves the matrix and
    there would be no way to show that an incident *can* legitimately change
    nothing. The spur is that case: a real road, reported on, that no route here
    has any reason to take.
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
            u, v, highway="residential", travel_time=SIDE_STREET_S, weight=SIDE_STREET_S,
            length=_length_of(SIDE_STREET_S),
        )
    graph.add_edge(
        0, 3, highway="residential", travel_time=SPUR_S, weight=SPUR_S,
        length=_length_of(SPUR_S),
    )
    return graph


@pytest.fixture(scope="module")
def tour_graph() -> nx.DiGraph:
    return build_tour_graph()


@pytest.fixture
def client(tour_graph):
    """An app over the tour graph, with stores that live for the test.

    Both overrides must be callables returning one instance: a dependency
    override is invoked per request, so passing the class would hand every
    request a brand-new empty store. Overriding the log store and the run history
    also keeps the suite from appending to the developer's real databases — this
    file calls ``/optimize``, which writes a row to the latter.
    """
    application = create_app()
    store = ScenarioStore()
    log_store = TrafficLogStore(":memory:")
    application.dependency_overrides[get_graph] = lambda: tour_graph
    application.dependency_overrides[get_store] = lambda: store
    application.dependency_overrides[get_log_store] = lambda: log_store
    application.dependency_overrides[get_run_store] = lambda: RunLogStore(":memory:")
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def create_scenario(client, *, timestamp: str = NORMAL_TIME, **conditions) -> dict:
    """Create the two-stop tour scenario through the API."""
    payload = {
        "kind": "explicit",
        "depot": {"node": 0},
        "deliveries": [
            {"id": "D0", "node": 1, "demand": 1},
            {"id": "D1", "node": 2, "demand": 1},
        ],
        "vehicles": [{"id": "V0", "capacity": 2}],
        "conditions": {"timestamp": timestamp, **conditions},
    }
    response = client.post("/scenarios", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def inject(client, scenario_id: str, payload: dict) -> dict:
    """POST an incident and return the body, asserting it was accepted."""
    response = client.post(f"/scenarios/{scenario_id}/incident", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def revert(client, scenario_id: str, incident_id: str) -> dict:
    """DELETE an incident and return the body, asserting it was accepted."""
    response = client.delete(
        f"/scenarios/{scenario_id}/incident/{incident_id}"
    )
    assert response.status_code == 200, response.text
    return response.json()


def conditions_of(client, scenario_id: str) -> dict:
    """The scenario's current conditions, read back over HTTP."""
    response = client.get(f"/scenarios/{scenario_id}")
    assert response.status_code == 200, response.text
    return response.json()["conditions"]


def leg_seconds(client, scenario_id: str) -> list[list[float]]:
    """The stored cost matrix in seconds.

    Read straight off the record rather than through an endpoint, because no
    route exposes the matrix — ``changed_legs`` on the incident response is the
    HTTP-level view of the same thing, and the alternative here is inference
    from a solve.
    """
    record = client.app.dependency_overrides[get_store]().get(scenario_id)
    return record.cost_matrix.matrix.tolist()


def solve(client, scenario_id: str) -> dict:
    """Run the exact solver, with geometry off."""
    response = client.post(
        f"/optimize/{scenario_id}",
        json={"solver": "brute_force", "include_geometry": False},
    )
    assert response.status_code == 200, response.text
    return response.json()


def tour(client, scenario_id: str) -> list[list[int]]:
    """The solved stop order, as scenario node ids."""
    return [
        [stop["node"] for stop in route["stops"]]
        for route in solve(client, scenario_id)["routes"]
    ]


def log_rows(client, **filters) -> list[dict]:
    """Traffic-log rows matching a filter, newest first."""
    response = client.get("/traffic/log", params={"limit": 1000, **filters})
    assert response.status_code == 200, response.text
    return response.json()["items"]


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def test_a_closure_closes_an_edge_and_a_slow_report_slows_it() -> None:
    """The two reports fold onto the two effect buckets, and nothing else moves."""
    base = ActiveConditions(closed_edges=frozenset({(1, 0)}))

    with_slow = conditions_for(
        base, [Incident("i1", SLOW, 0, 1, "2026-09-21T15:20:00+00:00")]
    )
    assert with_slow.accident_edges == frozenset({(0, 1)})
    assert with_slow.closed_edges == frozenset({(1, 0)})  # the base survives

    with_closure = conditions_for(
        base, [Incident("i2", CLOSURE, 0, 1, "2026-09-21T15:20:00+00:00")]
    )
    assert with_closure.closed_edges == frozenset({(1, 0), (0, 1)})
    assert with_closure.accident_edges == frozenset()


def test_dropping_an_incident_and_refolding_restores_the_base() -> None:
    """Reverting is exact, which is why the base is kept rather than subtracted.

    Both incidents here name the *same* edge as a base closure. Subtracting the
    live one from an effective set would take the base closure with it; folding
    from the base cannot.
    """
    base = ActiveConditions(closed_edges=frozenset({(0, 1)}))
    incident = Incident("i1", SLOW, 0, 1, "2026-09-21T15:20:00+00:00")

    assert conditions_for(base, [incident]).accident_edges == frozenset({(0, 1)})
    assert conditions_for(base, []) is not base
    assert conditions_for(base, []) == base


def test_the_locked_multipliers_are_the_ones_the_design_records() -> None:
    """Pins Prompt 2's set: closure is impassable, slow is a flat x2.9.

    The multiple is asserted as the *peak factor* rather than the literal 2.9,
    so this cannot drift from the simulator's one definition of it.
    """
    assert INCIDENT_MULTIPLIERS[SLOW] == PEAK_FACTOR
    assert PEAK_FACTOR == 2.9
    assert INCIDENT_MULTIPLIERS[CLOSURE] == float("inf")


def test_the_multipliers_are_what_routing_actually_charges() -> None:
    """The fold reaches pricing, not just the dataclass."""
    now = datetime(2026, 9, 21, 2, 0, tzinfo=IST)
    slow = TrafficState(
        timestamp=now,
        conditions=conditions_for(
            ActiveConditions(), [Incident("i", SLOW, 0, 1, "x")]
        ),
    )
    closed = TrafficState(
        timestamp=now,
        conditions=conditions_for(
            ActiveConditions(), [Incident("i", CLOSURE, 0, 1, "x")]
        ),
    )

    assert get_traffic_multiplier(Edge(0, 1), now, slow.conditions) == PEAK_FACTOR
    assert get_traffic_multiplier(Edge(0, 1), now, closed.conditions) == float("inf")
    # An incident is per edge: the other corridor is untouched.
    assert get_traffic_multiplier(Edge(0, 2), now, slow.conditions) == 1.0


# --------------------------------------------------------------------------- #
# Injecting
# --------------------------------------------------------------------------- #
def test_a_new_scenario_is_not_mutated(client) -> None:
    created = create_scenario(client)
    conditions = conditions_of(client, created["scenario_id"])

    assert conditions["mutated"] is False
    assert conditions["incidents"] == []
    assert conditions["accident_edges"] == []
    assert conditions["closed_edges"] == []


def test_a_slow_incident_moves_the_cost_matrix(client) -> None:
    """The demonstration: an edge weight genuinely changes in the matrix.

    The leg moves to 22.0s, not to the edge's own 29.0s, and that difference is
    the point of routing over a matrix rather than over edges — 29s down a road
    with a 22s way round it *is* a 22s leg. The edge's own 29s is what the log
    records; see the audit-trail test.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    before = leg_seconds(client, scenario_id)[0][1]
    assert before == ARTERIAL_S

    body = inject(client, scenario_id, SLOW_ON_0_1)

    assert body["applied"] is True
    assert body["changed_legs"] == 1
    assert leg_seconds(client, scenario_id)[0][1] == DETOURED_LEG_S
    assert leg_seconds(client, scenario_id)[0][2] == SIDE_STREET_S  # untouched


def test_a_closure_makes_the_leg_take_its_only_other_way_round(client) -> None:
    """The detour is the same 22s; what differs is that the road is shut."""
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    body = inject(client, scenario_id, CLOSURE_ON_0_1)

    assert body["changed_legs"] == 1
    assert leg_seconds(client, scenario_id)[0][1] == DETOURED_LEG_S
    assert body["conditions"]["closed_edges"] == [{"u": 0, "v": 1}]
    assert body["conditions"]["mutated"] is True


def test_a_slow_incident_on_an_unused_road_changes_nothing(client) -> None:
    """``changed_legs`` can legitimately be zero, and says so.

    The spur is a real road in the graph that no cheapest path between this
    scenario's nodes runs along — node 3 is a dead end. Slowing it moves no entry
    of the matrix, and reporting that is more useful than forcing a change,
    refusing the report, or quietly claiming success.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    before = leg_seconds(client, scenario_id)

    body = inject(client, scenario_id, SPUR_ON_0_3)

    assert body["changed_legs"] == 0
    assert leg_seconds(client, scenario_id) == before
    # It is still a live incident, and still reported as one.
    assert body["conditions"]["mutated"] is True
    assert len(log_rows(client, incident_type=SLOW)) == 1


def test_a_slow_incident_at_peak_is_a_no_op_on_a_through_road(client) -> None:
    """An incident *replaces* the congestion multiplier rather than compounding.

    At 09:00 the arterial is already 10 x 2.9 = 29s, and a slow report is also a
    flat x2.9, so the road is exactly as slow as it was. This is DESIGN_DECISIONS'
    rule showing up as arithmetic, and it is why the demo above runs at 02:00.
    """
    created = create_scenario(client, timestamp=PEAK_TIME)
    scenario_id = created["scenario_id"]
    before = leg_seconds(client, scenario_id)

    body = inject(client, scenario_id, SLOW_ON_0_1)

    assert body["changed_legs"] == 0
    assert leg_seconds(client, scenario_id) == before


# --------------------------------------------------------------------------- #
# The audit trail
# --------------------------------------------------------------------------- #
def test_injecting_an_incident_logs_a_row_for_that_road(client) -> None:
    """Edge, timestamp, type, day and time — the whole point of the trail.

    ``travel_time`` is the *edge's* 29s, computed by
    :func:`~qgati.traffic.simulator.simulated_travel_time` — the same call
    routing makes — rather than the 22s the matrix leg became.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    inject(client, scenario_id, SLOW_ON_0_1)

    rows = log_rows(client, incident_type=SLOW)
    assert len(rows) == 1
    row = rows[0]
    assert row["road_id"] == "0->1"
    assert row["road_u"] == 0 and row["road_v"] == 1
    assert row["incident_type"] == SLOW
    assert row["travel_time"] == SLOWED_ARTERIAL_S
    assert row["traffic_condition"] == "normal"
    assert row["timestamp"] == "2026-09-21T02:00:00+05:30"
    assert row["day_of_week"] == "Monday"
    assert row["time_of_day"] == "02:00"


def test_a_closed_road_logs_no_travel_time(client) -> None:
    """Impassable, so there is no travel time to record; the type says why."""
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    inject(client, scenario_id, CLOSURE_ON_0_1)

    rows = log_rows(client, incident_type=CLOSURE)
    assert len(rows) == 1
    assert rows[0]["road_id"] == "0->1"
    assert rows[0]["travel_time"] is None


def test_an_incident_logs_one_row_and_not_the_whole_network(client) -> None:
    """Re-pricing is not re-collection: the scenario's roads were logged at birth.

    Counting the roads instead of the rows would pass on a graph this small, so
    the assertion is on the total row count moving by exactly one.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    before = client.get("/traffic/log", params={"limit": 1}).json()["total"]

    body = inject(client, scenario_id, SLOW_ON_0_1)

    after = client.get("/traffic/log", params={"limit": 1}).json()["total"]
    assert after - before == 1
    assert body["traffic_rows_logged"] == 1


# --------------------------------------------------------------------------- #
# Reverting
# --------------------------------------------------------------------------- #
def test_reverting_restores_the_matrix_exactly(client) -> None:
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    before = leg_seconds(client, scenario_id)

    incident_id = inject(client, scenario_id, SLOW_ON_0_1)["incident"]["incident_id"]
    assert leg_seconds(client, scenario_id) != before

    body = revert(client, scenario_id, incident_id)

    assert body["applied"] is False
    assert body["changed_legs"] == 1
    assert leg_seconds(client, scenario_id) == before
    assert body["conditions"]["mutated"] is False
    assert body["conditions"]["incidents"] == []


def test_reverting_logs_the_road_coming_back(client) -> None:
    """A revert is a change too, and the trail says when the road reopened."""
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    incident_id = inject(client, scenario_id, SLOW_ON_0_1)["incident"]["incident_id"]

    revert(client, scenario_id, incident_id)

    rows = log_rows(client, incident_type=CLEARED)
    assert len(rows) == 1
    assert rows[0]["road_id"] == "0->1"
    assert rows[0]["travel_time"] == ARTERIAL_S


def test_reverting_one_incident_leaves_the_others_standing(client) -> None:
    """Two live reports on different roads; reverting one leaves the other applied.

    The survivor is checked in the matrix, not only in the conditions, so the
    test cannot pass on an incident that is still listed but no longer pricing.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    first = inject(client, scenario_id, SLOW_ON_0_1)["incident"]["incident_id"]
    inject(client, scenario_id, SLOW_ON_2_1)

    # With the side-street way to stop 1 slowed as well, 0->1's 22s detour is no
    # longer a way out, so that leg is the slowed road itself.
    assert leg_seconds(client, scenario_id)[0][1] == SLOWED_ARTERIAL_S
    assert leg_seconds(client, scenario_id)[2][1] == SLOWED_SIDE_STREET_S

    body = revert(client, scenario_id, first)

    assert leg_seconds(client, scenario_id)[0][1] == ARTERIAL_S  # reverted
    # The survivor still prices: 2->1 would be its direct 11s with nothing live,
    # and is instead the 20s arterial way round that the slow report forces.
    assert leg_seconds(client, scenario_id)[2][1] == ARTERIAL_DETOUR_S

    incidents = body["conditions"]["incidents"]
    assert [item["incident_type"] for item in incidents] == [SLOW]
    assert incidents[0]["edge"] == {"u": 2, "v": 1}
    assert body["conditions"]["mutated"] is True


def test_a_scenario_keeps_the_conditions_it_was_created_with(client) -> None:
    """The base is not clobbered by an inject/revert cycle."""
    created = create_scenario(client, accident_edges=[{"u": 0, "v": 2}])
    scenario_id = created["scenario_id"]
    assert conditions_of(client, scenario_id)["accident_edges"] == [{"u": 0, "v": 2}]

    incident_id = inject(client, scenario_id, SLOW_ON_0_1)["incident"]["incident_id"]
    revert(client, scenario_id, incident_id)

    conditions = conditions_of(client, scenario_id)
    assert conditions["accident_edges"] == [{"u": 0, "v": 2}]
    assert conditions["incidents"] == []
    assert conditions["mutated"] is False


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def test_an_unknown_scenario_is_404_on_both_routes(client) -> None:
    created = client.post("/scenarios/nope/incident", json=SLOW_ON_0_1)
    deleted = client.delete("/scenarios/nope/incident/whatever")

    assert created.status_code == 404
    assert deleted.status_code == 404


def test_an_incident_edge_must_exist_in_the_road_graph(client) -> None:
    """A road that is not in the graph is a client error, not something to drop."""
    created = create_scenario(client)

    response = client.post(
        f"/scenarios/{created['scenario_id']}/incident",
        json={"incident_type": SLOW, "edge": {"u": 0, "v": 99}},
    )

    assert response.status_code == 422
    assert "no road from 0 to 99" in response.json()["detail"]


def test_an_incident_type_must_be_one_of_the_two(client) -> None:
    created = create_scenario(client)

    response = client.post(
        f"/scenarios/{created['scenario_id']}/incident",
        json={"incident_type": "earthquake", "edge": {"u": 0, "v": 1}},
    )

    assert response.status_code == 422


def test_reverting_an_unknown_incident_is_404(client) -> None:
    created = create_scenario(client)

    response = client.delete(
        f"/scenarios/{created['scenario_id']}/incident/not-an-id"
    )

    assert response.status_code == 404
    assert "already have been reverted" in response.json()["detail"]


def test_an_incident_cannot_be_reverted_twice(client) -> None:
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    incident_id = inject(client, scenario_id, SLOW_ON_0_1)["incident"]["incident_id"]

    assert revert(client, scenario_id, incident_id)["applied"] is False
    second = client.delete(f"/scenarios/{scenario_id}/incident/{incident_id}")

    assert second.status_code == 404


def test_a_closure_that_severs_the_instance_is_refused_and_changes_nothing(client) -> None:
    """Closing 0->1 leaves the side street; closing 2->1 as well severs stop 1.

    The second report is refused with 422 — the same answer creation gives the
    same closure — and, crucially, the *first* is left standing. A refusal that
    rolled back a previously accepted incident would be worse than the refusal.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    inject(client, scenario_id, CLOSURE_ON_0_1)
    after_first = leg_seconds(client, scenario_id)

    response = client.post(
        f"/scenarios/{scenario_id}/incident", json=CLOSURE_ON_2_1
    )

    assert response.status_code == 422
    assert "not applied" in response.json()["detail"]
    assert leg_seconds(client, scenario_id) == after_first
    assert len(conditions_of(client, scenario_id)["incidents"]) == 1


# --------------------------------------------------------------------------- #
# The boundary with re-optimization
# --------------------------------------------------------------------------- #
def test_the_incident_routes_do_not_re_optimize(client) -> None:
    """A change report is not a solution.

    Re-optimizing on an incident is the reopt module's job, in a later phase.
    The evidence that this route leaves it alone is structural: the response has
    no field a route, a cost or a solver could be returned in.
    """
    created = create_scenario(client)
    body = inject(client, created["scenario_id"], SLOW_ON_0_1)

    assert set(body) == {
        "scenario_id",
        "incident",
        "applied",
        "conditions",
        "changed_legs",
        "traffic_rows_logged",
    }


def test_an_incident_changes_the_next_solve_without_running_one(client) -> None:
    """One instance, one injection, a different route — and no solver in between.

    The tour flips because the *costs* moved, not because a search was re-run by
    the incident route. Between the two solves nothing is called but the incident
    endpoint and the explicit ``/optimize`` below.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    assert tour(client, scenario_id) == [[1, 2]]
    assert solve(client, scenario_id)["travel_time"] == pytest.approx(30.0)

    inject(client, scenario_id, SLOW_ON_0_1)

    assert tour(client, scenario_id) == [[2, 1]]
    assert solve(client, scenario_id)["travel_time"] == pytest.approx(33.0)


def test_the_incident_is_scoped_to_one_scenario(client) -> None:
    """Two scenarios from one seed, one injected into, one untouched.

    Per-scenario was the requirement, and this is what would break first if the
    conditions were folded into anything shared — the graph, or a module-level
    state.
    """
    first = create_scenario(client)
    second = create_scenario(client)

    inject(client, first["scenario_id"], SLOW_ON_0_1)

    assert leg_seconds(client, first["scenario_id"])[0][1] == DETOURED_LEG_S
    assert leg_seconds(client, second["scenario_id"])[0][1] == ARTERIAL_S
    assert conditions_of(client, second["scenario_id"])["mutated"] is False
