"""Decision traces: that every sentence is made of numbers the plan computed.

The claim this file defends is the one the feature is built on — **nothing in an
explanation is invented**. It is defended structurally rather than by reading the
generators: a statement carries the figures it was built from, so the sharpest
assertion available is the direct one — every ``Figure.value`` equals the field of
the ``Evaluation`` it claims to describe — and it is asserted that way rather
than by checking that some formatting ran.

Two consequences of that rule are tested as behaviour rather than as wording:

* **A closure is never given a travel time.** A closed road is ``math.inf`` in the
  matrix, so there is no after-time to quote. The temptation to print "delay
  increased to N minutes" for a closed road is the exact failure the feature is
  meant to avoid, so it gets a test of its own with the slow case beside it as the
  control.
* **A vehicle that handed work away is not reported as a saving.** A re-plan moves
  stops *between* vehicles, so a lower cost is usually less work rather than a
  better route. Comparing the two as though they were the same job is the most
  plausible wrong sentence this module could write.

The plans here are built by calling the real ``reoptimize`` and then replacing the
solution — and its evaluation — with a consistent alternative. So every number
asserted on was computed by ``evaluate``, not written into the test.
"""

from __future__ import annotations

from dataclasses import replace

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
from qgati.optimizer import Delivery, Depot, Scenario, Solution, Vehicle
from qgati.optimizer.fitness import evaluate
from qgati.reopt import (
    MAX_STATEMENTS,
    Move,
    RoadDelay,
    Trigger,
    VehicleProgress,
    before_view,
    explain_replan,
    explain_road,
    reoptimize,
)
from qgati.traffic import TrafficLogStore

LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05
TOY_SPEED_KPH = 30.0


def _length_of(travel_time: float) -> float:
    return travel_time * (TOY_SPEED_KPH / 3.6)


def build_graph() -> nx.DiGraph:
    """Three nodes, every ordered pair connected, and deliberately asymmetric.

    Full connectivity is what makes the costs exact rather than shortest paths
    through something else: with an edge for every pair, every leg is the direct
    one and a route's cost is the sum of the numbers written here. Asymmetry is
    what makes an ordering decision real — with symmetric costs the two orders of
    two stops would price identically and there would be no delta to explain.
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


@pytest.fixture(scope="module")
def graph() -> nx.DiGraph:
    return build_graph()


def build_scenario(*, vehicles: int = 1, stops: int = 2) -> Scenario:
    """A scenario whose deliveries sit on nodes 1 and 2.

    ``stops`` beyond two puts the extra deliveries on node 1 again, which is a
    legitimate scenario — two deliveries at one address — and gives a third stop
    to move around without a fourth node.
    """
    nodes = [1, 2, 1, 2]
    return Scenario(
        depot=Depot(node=0, lat=LAT_ORIGIN, lon=LON_ORIGIN),
        deliveries=tuple(
            Delivery(id=f"D{index}", node=nodes[index], demand=1.0)
            for index in range(stops)
        ),
        vehicles=tuple(
            Vehicle(id=f"V{index}", capacity=float(stops))
            for index in range(vehicles)
        ),
    )


def build_progress(
    scenario: Scenario, remaining: list[tuple[int, ...]] | None = None, **overrides
) -> tuple[VehicleProgress, ...]:
    """Every vehicle at the depot, with the work each has still ahead of it.

    The default — vehicle zero carrying everything, the rest carrying nothing —
    is the ordinary case at the start of a shift and the one that gives the
    explanation the most to compare: nothing completed, so the whole route is in
    play and the before half is a real plan rather than an empty one.

    ``remaining`` overrides that per vehicle, which a test needs whenever it wants
    two vehicles *both* still working. It is not cosmetic: a vehicle with an empty
    ``remaining`` fails :attr:`VehicleProgress.available` and drops out of the
    solved instance entirely, so a test that intends to move work *to* it has to
    hand it something.
    """
    items = []
    for index, vehicle in enumerate(scenario.vehicles):
        if remaining is None:
            stops = tuple(range(scenario.n_deliveries)) if index == 0 else ()
        else:
            stops = remaining[index]
        fields = {
            "vehicle_id": vehicle.id,
            "vehicle_index": index,
            "node": 0,
            "completed": (),
            "remaining": stops,
            "elapsed": 0.0,
            "remaining_capacity": vehicle.capacity,
        }
        fields.update(overrides)
        items.append(VehicleProgress(**fields))
    return tuple(items)


def run(graph: nx.DiGraph, scenario: Scenario, progress=None):
    """A real re-optimization, plus the *before* evaluation the route reports.

    ``savings`` rather than QPSO: these tests are about what is said about a plan,
    not about how it was found, and a deterministic solver means a failure here is
    never a seed.
    """
    progress = progress if progress is not None else build_progress(scenario)
    plan = reoptimize(
        scenario=scenario, progress=progress, graph=graph, solver_key="savings"
    )
    before_scenario, before_matrix = before_view(scenario, progress, graph)
    before = Solution(routes=tuple(item.remaining for item in progress))
    return plan, evaluate(before, before_scenario, before_matrix)


def with_routes(plan, routes):
    """The same plan with a different assignment, and its matching evaluation.

    The point of the helper: a test that wants a vehicle to have handed work away
    has to build a plan where that is true, and building it by calling
    ``evaluate`` means the costs the explanation reads are the costs the objective
    computed for exactly those routes — not numbers chosen to make a test pass.
    """
    solution = Solution(routes=tuple(routes))
    return replace(
        plan,
        solution=solution,
        evaluation=evaluate(solution, plan.instance, plan.matrix),
    )


def worse_ordering(plan) -> Solution:
    """The plan's own route, driven the other way round.

    This graph is asymmetric edge by edge, so the two orders of the same two stops
    never price alike — which is what makes the cost delta a real number rather
    than one that happens to be zero. Deriving the alternative from the plan's own
    route instead of writing one into the test is what stops the delta depending
    on which order the solver happened to pick: a hard-coded ``(1, 0)`` here would
    silently become the no-op case on the day savings returned that order itself,
    and both cost-delta tests would pass while asserting nothing.
    """
    route = plan.solution.routes[0]
    assert len(route) == 2, "this helper is for the two-stop instance"
    return Solution(routes=(tuple(reversed(route)),))


def trigger(**overrides) -> Trigger:
    fields = {"kinds": ("incident",), "detail": "1 live incident(s) on the network (closure)"}
    fields.update(overrides)
    return Trigger(**fields)


def all_statements(explanation):
    """Every statement in a trace, roads and routes together."""
    return [
        statement
        for road in explanation.roads
        for statement in road.statements
    ] + [
        statement
        for route in explanation.routes
        for statement in route.statements
    ]


# --------------------------------------------------------------------------- #
# The rule: every figure is one the plan computed
# --------------------------------------------------------------------------- #
def test_every_figure_is_a_number_the_evaluation_computed(graph) -> None:
    """Each figure equals the field it claims to describe.

    Asserted by label rather than by position, because positions are exactly what
    an off-by-one in the alignment would move — and a cost read from the wrong
    vehicle is a plausible number, not an error.
    """
    scenario = build_scenario()
    plan, _ = run(graph, scenario)
    # The same two stops in the opposite order, so the cost statement fires with a
    # real delta and an unchanged stop count.
    before = evaluate(worse_ordering(plan), plan.instance, plan.matrix)

    explanation = explain_replan(plan, before, trigger())
    figures = {
        figure.label: figure.value
        for statement in all_statements(explanation)
        for figure in statement.figures
    }

    assert "cost before" in figures and "cost after" in figures
    assert figures["cost before"] == pytest.approx(before.route_costs[0])
    assert figures["cost after"] == pytest.approx(plan.evaluation.route_costs[0])
    assert figures["change"] == pytest.approx(
        plan.evaluation.route_costs[0] - before.route_costs[0]
    )
    # The percentage is the same ratio, stated twice; if they disagree the
    # sentence is quoting a number that is not the one beside it.
    assert figures["change percent"] == pytest.approx(
        (plan.evaluation.route_costs[0] - before.route_costs[0])
        / before.route_costs[0]
        * 100.0
    )


def test_no_statement_is_emitted_without_its_numbers(graph) -> None:
    """The structural half of "never fabricated": a sentence always has receipts."""
    scenario = build_scenario(vehicles=2, stops=3)
    plan, before = run(graph, scenario)
    explanation = explain_replan(
        plan,
        before,
        trigger(),
        road_delays=[RoadDelay(u=0, v=1, before_seconds=10.0, after_seconds=29.0)],
    )

    assert all_statements(explanation), "a re-plan of this size has something to say"
    for statement in all_statements(explanation):
        assert statement.text.strip()
        assert statement.figures, f"statement carries no figures: {statement.text!r}"

    for route in explanation.routes:
        assert 1 <= len(route.statements) <= MAX_STATEMENTS
        assert route.headline.strip()
        assert route.vehicle_id


def test_the_headline_is_the_trigger_s_own_sentence(graph) -> None:
    """Passed through verbatim, so the trace and the trigger cannot disagree."""
    scenario = build_scenario()
    plan, before = run(graph, scenario)
    stated = trigger(detail="a very specific sentence about a very specific road")

    explanation = explain_replan(plan, before, stated)

    assert explanation.headline == stated.detail


def test_vehicles_are_explained_only_when_they_are_in_both_halves(graph) -> None:
    """The join is on ``vehicle_id``, and a vehicle in one half only is skipped.

    A finished vehicle has no after-route and a freshly added one has no
    before-route; either would need a sentence about work that is not there.
    """
    scenario = build_scenario(vehicles=2)
    plan, before = run(graph, scenario)
    explanation = explain_replan(plan, before, trigger())

    explained = {route.vehicle_id for route in explanation.routes}
    assert explained == {item.vehicle_id for item in plan.active}
    assert explained <= {item.vehicle_id for item in plan.progress}


# --------------------------------------------------------------------------- #
# A closure has no travel time
# --------------------------------------------------------------------------- #
def test_a_closure_is_never_described_with_a_travel_time() -> None:
    """The negative case the design is built around, with the slow as control.

    A closed road is impassable — ``math.inf`` in every matrix — so there is no
    after-time. A statement that quoted one would be inventing it, which is the
    single thing this feature must not do.
    """
    closed = explain_road(RoadDelay(u=10, v=11, before_seconds=41.0, after_seconds=None))
    statement = closed.statements[0]

    assert "closed" in statement.text.lower()
    assert "impassable" in statement.text.lower()
    # No figure may describe the road's state *now* as a duration.
    assert not any(
        figure.unit == "seconds" and figure.label == "after"
        for figure in statement.figures
    )
    # The before-time is still quoted, because it is known and it is the number
    # that makes the closure mean anything.
    assert any(
        figure.label == "before" and figure.value == pytest.approx(41.0)
        for figure in statement.figures
    )


def test_a_slow_quotes_the_real_before_and_after(graph) -> None:
    """The control: where an after-time exists, it is stated and it is exact."""
    slowed = explain_road(
        RoadDelay(u=10, v=11, before_seconds=41.0, after_seconds=118.9)
    )
    statement = slowed.statements[0]
    figures = {figure.label: figure.value for figure in statement.figures}

    assert figures["before"] == pytest.approx(41.0)
    assert figures["after"] == pytest.approx(118.9)
    assert figures["delay"] == pytest.approx(77.9)
    assert figures["slower by"] == pytest.approx(118.9 / 41.0)
    assert "min" in statement.text


def test_a_road_that_was_impassable_is_not_reported_as_a_negative_delay() -> None:
    """Reachable when an incident is reverted over a creation-time condition."""
    reopened = explain_road(
        RoadDelay(u=10, v=11, before_seconds=None, after_seconds=41.0)
    )
    statement = reopened.statements[0]

    assert "open" in statement.text.lower()
    assert "{:.0f} s".format(41.0) in statement.text


# --------------------------------------------------------------------------- #
# Work changing hands is not a saving
# --------------------------------------------------------------------------- #
def test_a_vehicle_that_kept_its_work_is_compared_like_for_like(graph) -> None:
    """Same stop count, so the two costs are prices for the same job."""
    scenario = build_scenario()
    plan, _ = run(graph, scenario)
    before = evaluate(worse_ordering(plan), plan.instance, plan.matrix)
    assert before.route_costs[0] != plan.evaluation.route_costs[0], (
        "the reversed ordering has to price differently, or this test is "
        "asserting the no-op case it means to be the opposite of"
    )

    explanation = explain_replan(plan, before, trigger())
    text = " ".join(
        statement.text for statement in explanation.routes[0].statements
    )

    assert "cost" in text
    # The like-for-like branch is the one that may quote a percentage, because
    # only there are the two numbers prices for the same work.
    assert "%" in text


def test_a_vehicle_that_handed_work_away_is_not_reported_as_a_saving(graph) -> None:
    """The most plausible wrong sentence, and the test that prevents it.

    Two vehicles, three stops, both still out. The first hands one away, so its
    after-cost is lower — for the obvious reason that it has less to do. Reported
    as a saving it would read as the optimizer having found it a better route,
    which is a different and false claim.

    The second vehicle is given a stop of its own rather than left idle, and that
    is load-bearing: a vehicle with nothing remaining is not ``available``, is not
    in the instance that gets solved, and so could not be the destination of a
    move — the plan would be a one-vehicle plan wearing a two-vehicle scenario.
    """
    scenario = build_scenario(vehicles=2, stops=3)
    progress = build_progress(scenario, remaining=[(1, 2), (0,)])
    plan, _ = run(graph, scenario, progress)
    # The first vehicle keeps one stop and hands the other to the second, which
    # already had one of its own.
    plan = with_routes(plan, [(1,), (0, 2)])
    plan = replace(plan, moved=(Move(delivery=2, from_vehicle="V0", to_vehicle="V1"),))

    before_scenario, before_matrix = before_view(scenario, progress, graph)
    before = evaluate(
        Solution(routes=(progress[0].remaining, progress[1].remaining)),
        before_scenario,
        before_matrix,
    )

    explanation = explain_replan(plan, before, trigger())
    first = next(route for route in explanation.routes if route.vehicle_id == "V0")
    text = " ".join(statement.text for statement in first.statements)

    assert "share" in text.lower()
    # No percentage: there is no like-for-like comparison to make one from.
    assert "%" not in text
    # And the move itself is reported.
    assert any(
        figure.label == "stops moved out" for statement in first.statements
        for figure in statement.figures
    )


def test_a_replan_that_changed_nothing_says_so(graph) -> None:
    """No moves, no delta, no lateness — and one honest sentence rather than none.

    This is the ``route changed: no`` case the panel already words as a result
    rather than a failure. An empty panel would suggest something was missing.
    """
    scenario = build_scenario()
    plan, _ = run(graph, scenario)
    # The plan's own routes, priced identically: nothing changed, by construction.
    before = evaluate(
        Solution(routes=plan.solution.routes), plan.instance, plan.matrix
    )

    explanation = explain_replan(plan, before, trigger())
    statements = explanation.routes[0].statements

    assert len(statements) == 1
    assert "kept its assignment" in statements[0].text.lower()
    assert statements[0].figures


# --------------------------------------------------------------------------- #
# API: the endpoint reports a trace the UI can draw
# --------------------------------------------------------------------------- #
def build_tour_graph() -> nx.DiGraph:
    """A corridor depot -> 1 -> 2 -> depot, one edge per leg, ten seconds each.

    The graph the fleet runs on. Separate from :func:`build_graph` because those
    tests are about an *ordering* decision — which needs asymmetric costs — while
    these only need a fleet that can be started, ticked and re-planned.
    """
    graph = nx.DiGraph()
    for node in (0, 1, 2):
        graph.add_node(node, x=LON_ORIGIN + node * DEGREE_SPAN, y=LAT_ORIGIN)
    for u, v in ((0, 1), (1, 2), (2, 0)):
        graph.add_edge(
            u, v, highway="secondary", travel_time=10.0, weight=10.0,
            length=_length_of(10.0),
        )
    return graph


@pytest.fixture
def tour_client() -> TestClient:
    """An app over the tour graph, with a fleet registry of its own."""
    application = create_app()
    graph = build_tour_graph()
    application.dependency_overrides[get_graph] = lambda: graph
    application.dependency_overrides[get_store] = lambda: ScenarioStore()
    application.dependency_overrides[get_log_store] = lambda: TrafficLogStore(":memory:")
    # In-memory like the traffic log: this endpoint's setup dispatches a fleet and
    # re-plans it, and both write a run history row.
    application.dependency_overrides[get_run_store] = lambda: RunLogStore(":memory:")
    application.dependency_overrides[get_watchers] = lambda: WatcherRegistry()
    with TestClient(application) as client:
        yield client
    application.dependency_overrides.clear()


def test_the_response_carries_a_trace_over_the_vehicles_it_replanned(
    tour_client: TestClient,
) -> None:
    """End to end: the endpoint's explanation covers exactly the ``after`` fleet.

    Nothing is asserted about *what* the sentences say — where the fleet is on a
    real tick is not something a test can pin, and a test that did would be
    asserting its own arithmetic. What holds wherever it happens to be is the
    shape: one route explained per route returned, a road named for the incident
    that triggered it, and a convergence history to draw.
    """
    created = tour_client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": 0},
            "deliveries": [
                {"id": "D0", "node": 1, "demand": 1},
                {"id": "D1", "node": 2, "demand": 1},
            ],
            "vehicles": [{"id": "V0", "capacity": 2}],
            "conditions": {"timestamp": "2026-09-21T02:00:00+05:30"},
        },
    )
    assert created.status_code == 201, created.text
    scenario_id = created.json()["scenario_id"]

    started = tour_client.post(
        f"/scenarios/{scenario_id}/watcher",
        json={
            # QPSO rather than the deterministic `savings`, and not for its answer:
            # `savings` is a construction with no search behind it and returns an
            # empty history, so it cannot witness the thing this test is about.
            # Small budget because the shape of the series is the claim, not its
            # length — and a step of one simulated second leaves the vehicle on
            # the first leg of its corridor, which is what puts an edge under it.
            "solver": "qpso",
            "seed": 3,
            "population": 8,
            "iterations": 12,
            "interval_seconds": 300.0,
            "time_scale": 1.0 / 300.0,
        },
    )
    assert started.status_code == 201, started.text
    # The dispatch carries the solve's history, which is what the chart draws
    # before anything has happened to the scenario. This is the field the watcher
    # handler used to discard, so it is asserted rather than assumed.
    dispatch_convergence = started.json()["convergence"]
    assert dispatch_convergence, "a dispatch reports its own convergence"
    assert all(isinstance(value, (int, float)) for value in dispatch_convergence)

    edge = tour_client.get(f"/scenarios/{scenario_id}/watcher").json()["vehicles"][0][
        "edge"
    ]
    assert edge is not None, "a dispatched vehicle starts on a road"
    reported = tour_client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": "slow", "edge": {"u": edge["u"], "v": edge["v"]}},
    )
    assert reported.status_code == 201, reported.text

    response = tour_client.post(f"/scenarios/{scenario_id}/reoptimize", json={})
    assert response.status_code == 200, response.text
    body = response.json()

    explanation = body["explanation"]
    assert explanation is not None
    assert explanation["headline"] == body["trigger"]["detail"]
    assert explanation["roads"], "the incident's road is reported"

    # The slow is real: the road has a before-time and an after-time, and the
    # after is the larger. This is the user-facing claim the panel makes.
    slowed = explanation["roads"][0]["statements"][0]
    figures = {figure["label"]: figure["value"] for figure in slowed["figures"]}
    assert figures["after"] > figures["before"]

    explained = [route["vehicle_id"] for route in explanation["routes"]]
    after = [route["vehicle_id"] for route in body["after"]]
    assert explained == after

    for route in explanation["routes"]:
        assert route["statements"], "every explained vehicle has something to say"
        for statement in route["statements"]:
            assert statement["figures"]

    assert body["convergence"], "the chart has a history to draw"
    assert all(isinstance(value, (int, float)) for value in body["convergence"])
