"""VRP core tests: models, cost matrix, brute force, and Savings.

The central claim under test is that Clarke-Wright Savings never beats the exact
solver — if it did, either the heuristic is broken or brute force is not actually
optimal. Both are checked separately: Savings against brute force, and brute
force against an independent naive enumeration on a case small enough to
exhaust.
"""

from __future__ import annotations

import itertools
import math

import pytest

from qgati.graph import build_cost_matrix, build_synthetic_graph
from qgati.graph.graph_builder import largest_strongly_connected_subgraph
from qgati.optimizer import (
    Delivery,
    Depot,
    Scenario,
    Solution,
    Vehicle,
    build_random_scenario,
    clarke_wright_savings,
    evaluate,
    route_metrics,
    route_travel_cost,
    route_travel_time,
    servable_nodes,
    solve_brute_force,
)

TOLERANCE = 1e-6

# (deliveries, vehicles, seed, label) — the tiny instances from the brief.
SCENARIO_SPECS = [
    (4, 2, 101, "n=4 k=2"),
    (6, 2, 202, "n=6 k=2"),
    (8, 3, 303, "n=8 k=3"),
]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def road_graph():
    """A dense synthetic road graph — dense so its largest SCC is large enough."""
    return build_synthetic_graph(n_nodes=30, edge_prob=0.25, seed=7)


@pytest.fixture(scope="module")
def tiny_handmade():
    """A 4-delivery, 2-vehicle instance with hand-written costs.

    Costs are deliberately asymmetric and non-metric so that a solver assuming
    symmetry or the triangle inequality would produce a visibly wrong answer.
    """
    scenario = Scenario(
        depot=Depot(node="depot", lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node="a", demand=3),
            Delivery(id="B", node="b", demand=4),
            Delivery(id="C", node="c", demand=2),
            Delivery(id="D", node="d", demand=5),
        ),
        vehicles=(Vehicle(id="V0", capacity=8), Vehicle(id="V1", capacity=8)),
    )
    matrix = [
        #    depot   a     b     c     d
        [0.0, 10.0, 12.0, 11.0, 13.0],  # depot
        [9.0, 0.0, 3.0, 7.0, 8.0],  # a
        [12.0, 4.0, 0.0, 6.0, 5.0],  # b
        [11.0, 7.0, 6.0, 0.0, 9.0],  # c
        [13.0, 6.0, 5.0, 9.0, 0.0],  # d
    ]
    return scenario, _matrix_from_lists(matrix)


def _matrix_from_lists(rows: list[list[float]]):
    """Wrap a hand-written nested list as a CostMatrix with depot at index 0."""
    import numpy as np

    from qgati.graph.cost_matrix import CostMatrix

    times = np.array(rows, dtype=float)
    return CostMatrix(
        matrix=times,
        nodes=("depot", "a", "b", "c", "d"),
        delivery_node_index=(1, 2, 3, 4),
        distance_matrix=_distance_from_times(times),
    )


#: Speed the hand-written toy networks are assumed to be driven at. Their numbers
#: are travel times, and the objective needs a distance to price, so the two are
#: related by a stated constant.
#:
#: A constant speed is the one case where the fuel curve contributes nothing
#: beyond a fixed multiple of distance, which is deliberate here: these tests are
#: about routing structure and constraint handling, and a speed-varying fuel term
#: would make every expected value depend on the fuel model as well. The
#: congestion behaviour that the fuel term exists for is tested on its own in
#: ``test_objective.py``.
TOY_SPEED_KPH = 30.0


def _distance_from_times(times, speed_kph: float = TOY_SPEED_KPH):
    """Metres covered by the given travel seconds at a constant speed."""
    import numpy as np

    return np.asarray(times, dtype=float) * (speed_kph / 3.6)


def naive_optimal_cost(scenario: Scenario, cost_matrix) -> float:
    """Independent exact solver: enumerate every assignment and every ordering.

    Exponential and useless in general, but it shares no code with the Held-Karp
    implementation in ``brute_force``, so agreement between the two is real
    evidence rather than a tautology.
    """
    n, k = scenario.n_deliveries, scenario.n_vehicles
    demands, capacities = scenario.demands, scenario.capacities
    best = math.inf

    for assignment in itertools.product(range(k), repeat=n):
        loads = [0.0] * k
        for delivery, vehicle in enumerate(assignment):
            loads[vehicle] += demands[delivery]
        if any(
            loads[vehicle] > capacities[vehicle] + TOLERANCE for vehicle in range(k)
        ):
            continue

        total = 0.0
        for vehicle in range(k):
            stops = [d for d in range(n) if assignment[d] == vehicle]
            if not stops:
                continue
            total += min(
                route_travel_cost(order, scenario, cost_matrix)
                for order in itertools.permutations(stops)
            )
        best = min(best, total)

    return best


def assert_solution_is_valid(solution: Solution, scenario: Scenario, cost_matrix) -> None:
    """Every structural requirement a valid VRP answer must satisfy.

    Deliberately independent of :func:`evaluate` — this walks the raw solution so
    a bug in the fitness function cannot hide behind it.
    """
    assert len(solution.routes) == scenario.n_vehicles

    served = sorted(solution.served())
    assert served == list(range(scenario.n_deliveries)), (
        f"expected each delivery exactly once, got {served}"
    )

    for vehicle, route in enumerate(solution.routes):
        load = sum(scenario.demands[d] for d in route)
        assert load <= scenario.capacities[vehicle] + TOLERANCE, (
            f"vehicle {vehicle} carries {load} > {scenario.capacities[vehicle]}"
        )

    # Depot legs are counted: the reported cost is the depot-inclusive sum of
    # the same routes, so routes genuinely start and end at the depot.
    manual = sum(
        route_travel_cost(route, scenario, cost_matrix) for route in solution.routes
    )
    assert manual >= 0.0


# --------------------------------------------------------------------------- #
# The headline test: Savings vs exact
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n_deliveries,n_vehicles,seed,label", SCENARIO_SPECS
)
def test_savings_never_beats_brute_force(
    road_graph, n_deliveries, n_vehicles, seed, label, capsys
) -> None:
    scenario = build_random_scenario(
        road_graph, n_deliveries, n_vehicles, seed=seed
    )
    cost_matrix = build_cost_matrix(road_graph, scenario)

    optimal = solve_brute_force(scenario, cost_matrix)
    heuristic = clarke_wright_savings(scenario, cost_matrix)

    optimal_eval = evaluate(optimal, scenario, cost_matrix)
    heuristic_eval = evaluate(heuristic, scenario, cost_matrix)

    gap = (
        (heuristic_eval.travel_cost - optimal_eval.travel_cost)
        / optimal_eval.travel_cost
        * 100.0
        if optimal_eval.travel_cost
        else 0.0
    )
    with capsys.disabled():
        print(
            f"\n[{label} seed={seed}] "
            f"brute=Rs{optimal_eval.travel_cost:.2f}  "
            f"savings=Rs{heuristic_eval.travel_cost:.2f}  "
            f"gap=+{gap:.2f}%"
        )
        print(f"    optimal : {optimal.describe(scenario)}")
        print(f"    savings : {heuristic.describe(scenario)}")

    # Both must be genuine, feasible answers...
    for name, solution, evaluation in (
        ("brute_force", optimal, optimal_eval),
        ("savings", heuristic, heuristic_eval),
    ):
        assert evaluation.feasible, f"{name} produced an infeasible solution"
        assert_solution_is_valid(solution, scenario, cost_matrix)

    # ...and the heuristic must never undercut the true optimum.
    assert heuristic_eval.travel_cost >= optimal_eval.travel_cost - TOLERANCE, (
        f"savings ({heuristic_eval.travel_cost}) beat the optimum "
        f"({optimal_eval.travel_cost}) — the heuristic or brute force is wrong"
    )


# --------------------------------------------------------------------------- #
# Brute force really is optimal
# --------------------------------------------------------------------------- #
def test_brute_force_matches_naive_enumeration(tiny_handmade) -> None:
    scenario, cost_matrix = tiny_handmade

    optimal = solve_brute_force(scenario, cost_matrix)
    optimal_cost = evaluate(optimal, scenario, cost_matrix).travel_cost
    naive_cost = naive_optimal_cost(scenario, cost_matrix)

    assert optimal_cost == pytest.approx(naive_cost, abs=TOLERANCE)


def test_savings_on_handmade_instance(tiny_handmade, capsys) -> None:
    scenario, cost_matrix = tiny_handmade

    optimal = solve_brute_force(scenario, cost_matrix)
    heuristic = clarke_wright_savings(scenario, cost_matrix)
    optimal_cost = evaluate(optimal, scenario, cost_matrix).travel_cost
    heuristic_cost = evaluate(heuristic, scenario, cost_matrix).travel_cost

    with capsys.disabled():
        print(
            f"\n[handmade n=4 k=2] brute={optimal_cost:.4f}  "
            f"savings={heuristic_cost:.4f}"
        )
        print(f"    optimal : {optimal.describe(scenario)}")
        print(f"    savings : {heuristic.describe(scenario)}")

    assert heuristic_cost >= optimal_cost - TOLERANCE
    assert evaluate(heuristic, scenario, cost_matrix).feasible


def test_brute_force_rejects_oversized_instances() -> None:
    from qgati.optimizer.brute_force import MAX_EXACT_DELIVERIES

    big = Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=tuple(
            Delivery(id=f"D{i}", node=i + 1, demand=1)
            for i in range(MAX_EXACT_DELIVERIES + 1)
        ),
        vehicles=(Vehicle(id="V0", capacity=100),),
    )
    import numpy as np

    from qgati.graph.cost_matrix import CostMatrix

    size = MAX_EXACT_DELIVERIES + 2
    ones = np.ones((size, size)) - np.eye(size)
    matrix = CostMatrix(
        matrix=ones,
        nodes=tuple(range(size)),
        delivery_node_index=tuple(range(1, size)),
        distance_matrix=_distance_from_times(ones),
    )
    with pytest.raises(ValueError, match="limited to"):
        solve_brute_force(big, matrix)


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def test_fitness_flags_missing_delivery(tiny_handmade) -> None:
    scenario, cost_matrix = tiny_handmade
    # Delivery D (index 3) is never served.
    solution = Solution(routes=((0, 1), (2,)))

    evaluation = evaluate(solution, scenario, cost_matrix)
    assert not evaluation.feasible
    assert evaluation.coverage_penalty > 0.0
    assert evaluation.capacity_penalty == 0.0


def test_fitness_flags_duplicate_delivery(tiny_handmade) -> None:
    scenario, cost_matrix = tiny_handmade
    # Delivery 0 is served by both vehicles.
    solution = Solution(routes=((0, 1, 3), (0, 2)))

    evaluation = evaluate(solution, scenario, cost_matrix)
    assert not evaluation.feasible
    assert evaluation.coverage_penalty > 0.0


def test_fitness_flags_capacity_violation(tiny_handmade) -> None:
    scenario, cost_matrix = tiny_handmade
    # Everything on one vehicle: load 14 against capacity 8.
    solution = Solution(routes=((0, 1, 2, 3), ()))

    evaluation = evaluate(solution, scenario, cost_matrix)
    assert not evaluation.feasible
    assert evaluation.capacity_penalty > 0.0
    assert evaluation.route_loads[0] == pytest.approx(14.0)


def test_fitness_flags_wrong_route_count(tiny_handmade) -> None:
    scenario, cost_matrix = tiny_handmade
    solution = Solution(routes=((0, 1, 2, 3),))

    evaluation = evaluate(solution, scenario, cost_matrix)
    assert not evaluation.feasible
    assert evaluation.shape_penalty > 0.0


def test_fitness_penalty_dominates_any_travel_saving(tiny_handmade) -> None:
    """An infeasible solution must never outscore a feasible one."""
    scenario, cost_matrix = tiny_handmade
    feasible = Solution(routes=((0, 1), (2, 3)))
    dropping_a_delivery = Solution(routes=((0, 1), (2,)))  # cheaper travel

    feasible_eval = evaluate(feasible, scenario, cost_matrix)
    cheating_eval = evaluate(dropping_a_delivery, scenario, cost_matrix)

    assert feasible_eval.feasible
    assert not cheating_eval.feasible
    assert cheating_eval.travel_cost < feasible_eval.travel_cost  # it did save
    assert cheating_eval.fitness > feasible_eval.fitness  # but still loses


def test_route_travel_cost_includes_both_depot_legs(tiny_handmade) -> None:
    """The objective on a route whose arithmetic can be done by hand.

    depot -> a (10s / 10*8.333m) -> b (3s / 3*8.333m) -> depot (12s) is 25
    seconds and 208.33 metres. The three prices are spelled out rather than read
    from the defaults on purpose: this test is the pin on the objective's
    definition, so a silent change to any of them should fail it.
    """
    scenario, cost_matrix = tiny_handmade
    metrics = route_metrics((0, 1), scenario, cost_matrix)

    seconds = 25.0
    metres = seconds * (30.0 / 3.6)  # the toy network's constant speed
    litres = metres / 1000.0 * (211.25 / 30.0 + 1.0 + 0.05 * 30.0) / 100.0

    assert metrics.time == pytest.approx(seconds)
    assert metrics.distance == pytest.approx(metres)
    assert metrics.fuel == pytest.approx(litres)
    # No windows on this instance, so nothing waits and the elapsed time is the
    # driving time — the equality that keeps windowless instances exactly as they
    # were before windows existed.
    assert metrics.waiting == 0.0
    assert metrics.driving_time == pytest.approx(seconds)

    expected = (
        seconds * (150.0 / 3600.0)   # time, Rs150/hour
        + metres * (5.0 / 1000.0)    # distance, Rs5/km
        + litres * 90.0              # fuel, Rs90/litre
    )
    assert metrics.cost == pytest.approx(expected)
    assert route_travel_cost((0, 1), scenario, cost_matrix) == pytest.approx(expected)
    assert route_travel_time((0, 1), scenario, cost_matrix) == pytest.approx(seconds)
    assert route_travel_cost((), scenario, cost_matrix) == 0.0
    assert route_travel_time((), scenario, cost_matrix) == 0.0


# --------------------------------------------------------------------------- #
# Scenario generation — the SCC guarantee
# --------------------------------------------------------------------------- #
def test_generated_scenarios_use_only_servable_nodes(road_graph) -> None:
    servable = set(servable_nodes(road_graph))
    for n_deliveries, n_vehicles, seed, _ in SCENARIO_SPECS:
        scenario = build_random_scenario(
            road_graph, n_deliveries, n_vehicles, seed=seed
        )
        assert scenario.depot.node in servable
        for delivery in scenario.deliveries:
            assert delivery.node in servable


def test_generated_scenarios_are_internally_connected(road_graph) -> None:
    """Every generated instance must have a feasible tour by construction."""
    component = largest_strongly_connected_subgraph(road_graph)
    for n_deliveries, n_vehicles, seed, _ in SCENARIO_SPECS:
        scenario = build_random_scenario(
            road_graph, n_deliveries, n_vehicles, seed=seed
        )
        nodes = [scenario.depot.node, *(d.node for d in scenario.deliveries)]
        for source in nodes:
            for target in nodes:
                assert source in component and target in component
        # And brute force must actually find a solution.
        cost_matrix = build_cost_matrix(road_graph, scenario)
        assert evaluate(
            solve_brute_force(scenario, cost_matrix), scenario, cost_matrix
        ).feasible


def test_generated_scenario_is_deterministic(road_graph) -> None:
    first = build_random_scenario(road_graph, 5, 2, seed=99)
    second = build_random_scenario(road_graph, 5, 2, seed=99)
    assert first == second
    assert first.delivery_ids() == ("D0", "D1", "D2", "D3", "D4")


def test_scenario_generator_rejects_impossible_requests(road_graph) -> None:
    with pytest.raises(ValueError, match="n_deliveries"):
        build_random_scenario(road_graph, 0, 2, seed=1)
    with pytest.raises(ValueError, match="n_vehicles"):
        build_random_scenario(road_graph, 3, 0, seed=1)
    with pytest.raises(ValueError, match="capacity_slack"):
        build_random_scenario(road_graph, 3, 2, seed=1, capacity_slack=0.5)
    with pytest.raises(ValueError, match="mutually-reachable"):
        build_random_scenario(road_graph, 500, 2, seed=1)


# --------------------------------------------------------------------------- #
# Scenario model validation
# --------------------------------------------------------------------------- #
def test_scenario_rejects_demand_exceeding_capacity() -> None:
    with pytest.raises(ValueError, match="infeasible instance"):
        Scenario(
            depot=Depot(node=0, lat=0.0, lon=0.0),
            deliveries=(Delivery(id="A", node=1, demand=10),),
            vehicles=(Vehicle(id="V0", capacity=5),),
        )


def test_scenario_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate delivery ids"):
        Scenario(
            depot=Depot(node=0, lat=0.0, lon=0.0),
            deliveries=(Delivery(id="A", node=1, demand=1), Delivery(id="A", node=2, demand=1)),
            vehicles=(Vehicle(id="V0", capacity=5),),
        )
    with pytest.raises(ValueError, match="duplicate vehicle ids"):
        Scenario(
            depot=Depot(node=0, lat=0.0, lon=0.0),
            deliveries=(Delivery(id="A", node=1, demand=1),),
            vehicles=(Vehicle(id="V0", capacity=5), Vehicle(id="V0", capacity=5)),
        )


def test_scenario_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="no deliveries"):
        Scenario(depot=Depot(node=0, lat=0.0, lon=0.0), vehicles=(Vehicle(id="V", capacity=1),))
    with pytest.raises(ValueError, match="no vehicles"):
        Scenario(
            depot=Depot(node=0, lat=0.0, lon=0.0),
            deliveries=(Delivery(id="A", node=1, demand=1),),
        )


def test_scenario_accepts_two_deliveries_at_one_node(road_graph) -> None:
    """Two drops at the same building are legal and map to one matrix index."""
    scenario = Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node=1, demand=2),
            Delivery(id="B", node=1, demand=3),  # same node as A
        ),
        vehicles=(Vehicle(id="V0", capacity=5),),
    )
    assert scenario.n_deliveries == 2


# --------------------------------------------------------------------------- #
# Cost matrix
# --------------------------------------------------------------------------- #
def test_cost_matrix_layout(road_graph) -> None:
    scenario = build_random_scenario(road_graph, 6, 2, seed=42)
    cost_matrix = build_cost_matrix(road_graph, scenario)

    # Depot first, then one column per distinct delivery node.
    assert cost_matrix.depot_index == 0
    assert cost_matrix.nodes[0] == scenario.depot.node
    assert len(cost_matrix) == 7
    assert cost_matrix.matrix.shape == (7, 7)

    # Diagonal is zero and off-diagonal entries are strictly positive.
    import numpy as np

    assert np.allclose(np.diag(cost_matrix.matrix), 0.0)
    off_diagonal = cost_matrix.matrix[~np.eye(7, dtype=bool)]
    assert (off_diagonal > 0).all()

    # Every delivery resolves to a real index.
    for delivery in scenario.deliveries:
        assert 0 <= cost_matrix.node_index(delivery.node) < len(cost_matrix)


def test_cost_matrix_maps_duplicate_nodes_to_one_index(road_graph) -> None:
    servable = servable_nodes(road_graph)
    shared = servable[1]
    scenario = Scenario(
        depot=Depot(node=servable[0], lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node=shared, demand=1),
            Delivery(id="B", node=shared, demand=1),
            Delivery(id="C", node=servable[2], demand=1),
        ),
        vehicles=(Vehicle(id="V0", capacity=5),),
    )
    cost_matrix = build_cost_matrix(road_graph, scenario)

    assert len(cost_matrix) == 3  # depot + 2 distinct delivery nodes
    assert cost_matrix.delivery_node_index[0] == cost_matrix.delivery_node_index[1]
    assert cost_matrix.delivery_node_index[2] != cost_matrix.delivery_node_index[0]


def test_cost_matrix_rejects_unknown_nodes(road_graph) -> None:
    scenario = Scenario(
        depot=Depot(node="not-a-node", lat=0.0, lon=0.0),
        deliveries=(Delivery(id="A", node=1, demand=1),),
        vehicles=(Vehicle(id="V0", capacity=5),),
    )
    with pytest.raises(ValueError, match="not in the graph"):
        build_cost_matrix(road_graph, scenario)


def test_cost_matrix_raises_on_unreachable_pair() -> None:
    """An unreachable pair must fail loudly, not silently become inf."""
    graph = build_synthetic_graph(n_nodes=10, edge_prob=0.05, seed=3)
    # Force a node that cannot reach anything.
    graph.add_node("island", pos=(0.0, 0.0))
    scenario = Scenario(
        depot=Depot(node="island", lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node=next(iter(graph.nodes)), demand=1),
        ),
        vehicles=(Vehicle(id="V0", capacity=5),),
    )
    with pytest.raises(ValueError, match="unreachable"):
        build_cost_matrix(graph, scenario)


# --------------------------------------------------------------------------- #
# Per-vehicle start nodes — and the guarantee that they change nothing by default
# --------------------------------------------------------------------------- #
# The extension exists so a re-optimization can start each vehicle from where it
# actually is. Every branch it added is gated on ``Scenario.has_custom_starts``,
# and the whole point of that gating is that a scenario naming no starts is priced,
# split and assembled *exactly* as it was before the field existed. That is not a
# claim a comment can make; these are the tests that hold it.
#
# The benchmark is the end-to-end version of the same claim — ``run_comparison.py``
# must not move — but a benchmark that has drifted tells you a number changed, not
# which of three branches did it. These say which.


def _start_scenario(**overrides) -> Scenario:
    """The ``tiny_handmade`` instance, with ``starts`` named where a test wants them."""
    fields = {
        "depot": Depot(node="depot", lat=0.0, lon=0.0),
        "deliveries": (
            Delivery(id="A", node="a", demand=3),
            Delivery(id="B", node="b", demand=4),
            Delivery(id="C", node="c", demand=2),
            Delivery(id="D", node="d", demand=5),
        ),
        "vehicles": (Vehicle(id="V0", capacity=8), Vehicle(id="V1", capacity=8)),
    }
    fields.update(overrides)
    return Scenario(**fields)


def test_a_scenario_without_starts_leaves_the_matrix_untouched(tiny_handmade) -> None:
    """``vehicle_start_index`` stays **empty**, not a tuple of zeros.

    The distinction is the whole reason the field defaults the way it does. Filled
    with zeros the matrix would behave identically and be a *different object*, and
    the watcher compares two matrices with ``is`` while ``changed_entries``
    compares them element-wise — so "benignly different" is not a thing a cached
    matrix should be.
    """
    scenario, matrix = tiny_handmade

    assert matrix.vehicle_start_index == ()
    assert scenario.has_custom_starts is False
    # And every vehicle is answered the same way, without the caller branching.
    assert {matrix.start_index(v) for v in range(scenario.n_vehicles)} == {0}
    assert scenario.start_node(0) == scenario.depot.node


def test_a_start_node_is_added_to_the_matrix_after_the_depot(road_graph) -> None:
    """The depot stays at index 0 however many starts a scenario names.

    ``DEPOT_INDEX`` is hard-coded in the decoder, the fitness function and every
    saved matrix. Inserting starts ahead of the depot would move it and silently
    invalidate all three.
    """
    servable = servable_nodes(road_graph)
    depot, first, second = servable[0], servable[1], servable[2]
    scenario = Scenario(
        depot=Depot(node=depot, lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node=first, demand=1),
            Delivery(id="B", node=second, demand=1),
        ),
        vehicles=(Vehicle(id="V0", capacity=2), Vehicle(id="V1", capacity=2)),
        starts=(second, first),
    )
    matrix = build_cost_matrix(road_graph, scenario)

    assert matrix.depot_index == 0
    assert matrix.nodes[0] == depot
    assert len(matrix.vehicle_start_index) == 2
    assert matrix.start_index(0) == matrix.node_index(second)
    assert matrix.start_index(1) == matrix.node_index(first)


def test_only_the_outbound_leg_moves_to_a_vehicle_start(tiny_handmade) -> None:
    """A re-planned vehicle leaves from where it is and still comes home.

    The return leg is the depot for every vehicle on every scenario. Only the
    outbound leg is a vehicle's own, which is exactly the asymmetry
    ``CostMatrix`` documents — one index cannot describe a different start per
    vehicle, so the starts are a separate mapping and the depot stays where it is.
    """
    _scenario, matrix = tiny_handmade
    route = (0,)  # the single delivery "A", at node "a"

    # Depot -> a -> depot, the ordinary case.
    assert route_metrics(route, _scenario, matrix, 0).time == pytest.approx(10.0 + 9.0)

    from qgati.graph.cost_matrix import CostMatrix

    started = _start_scenario(
        # The whole instance's demand, not the one stop this route serves:
        # ``Scenario`` bounds total demand by total capacity, and the fixture
        # carries four deliveries weighing 3, 4, 2 and 5.
        vehicles=(Vehicle(id="V0", capacity=14),), starts=("a",)
    )
    from_the_stop = CostMatrix(
        matrix=matrix.matrix,
        nodes=matrix.nodes,
        delivery_node_index=matrix.delivery_node_index,
        distance_matrix=matrix.distance_matrix,
        # Vehicle 0 now begins at node "a" — which is where its only stop is.
        vehicle_start_index=(matrix.node_index("a"),),
    )

    assert route_metrics(route, started, from_the_stop, 0).time == pytest.approx(9.0)


def test_per_vehicle_capacities_only_bind_once_starts_are_named(tiny_handmade) -> None:
    """A scenario without starts is bounded by the fleet's largest capacity.

    Stated as the difference it makes rather than as which split the DP prefers,
    because which split is cheapest is the DP's business and asserting it here
    would be asserting an accident of these particular numbers. What is *forced*
    is that the two rules are not equally permissive: bounding a route by its own
    vehicle is strictly tighter than bounding every route by the biggest one, so
    there are instances the old rule accepts and the new one refuses.

    This is one. Two vehicles of 2 and 5 units; three deliveries weighing 1, 3 and
    3 — seven units in total, so the instance is feasible for a fleet of seven and
    :class:`Scenario` accepts it. Under the largest-capacity rule the 6-unit tail
    is a route of its own with a 5-unit ceiling nowhere in sight, because the
    ceiling being applied is the 5 *and* the 2 together seen as one number: 5. The
    first two deliveries weigh 4, which fits, and the last weighs 3, which fits.
    Under per-vehicle capacities the 2-unit vehicle cannot take the 4-unit pair,
    and the 5-unit one cannot take the 6-unit tail, so nothing is feasible at all.

    Which is why the branch is gated rather than simply adopted. It is arguably
    the more correct rule, and it is a *different problem* — and the Phase 4
    benchmark was run under the old one. A benchmark that moves because a
    constraint got tighter compares two things that are not the same.
    """
    from qgati.optimizer.decoding import optimal_split

    _scenario, matrix = tiny_handmade
    three = Scenario(
        depot=Depot(node="depot", lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node="a", demand=1),
            Delivery(id="B", node="b", demand=3),
            Delivery(id="C", node="c", demand=3),
        ),
        vehicles=(Vehicle(id="V0", capacity=2), Vehicle(id="V1", capacity=5)),
    )
    permutation = (0, 1, 2)

    assert sum(three.demands) == sum(three.capacities), "feasible as a fleet"
    assert optimal_split(permutation, three, matrix) is not None

    apart = Scenario(
        depot=three.depot,
        deliveries=three.deliveries,
        vehicles=three.vehicles,
        starts=(three.depot.node, three.depot.node),
    )
    assert optimal_split(permutation, apart, matrix) is None


def test_the_default_decode_is_the_one_the_six_solvers_already_got(
    tiny_handmade,
) -> None:
    """The regression the benchmark would catch, caught one layer down.

    ``decode_permutation`` is the single entry point all six solvers share, so a
    leak in any gated branch would move every solver at once. Asserted by asking
    for the same permutation twice — once the ordinary way, and once with the
    pre-extension defaults spelled out — so a difference means a branch fired that
    should not have. ``tiny_handmade`` is asymmetric and non-metric on purpose, so
    a decoder that got a depot wrong would land on a different answer rather than
    a coincidentally equal one.
    """
    from qgati.optimizer.decoding import decode_permutation, optimal_split

    scenario, matrix = tiny_handmade
    permutation = (1, 0, 2, 3)

    assert optimal_split(permutation, scenario, matrix) == optimal_split(
        permutation,
        scenario,
        matrix,
        capacity=max(scenario.capacities),
        max_routes=scenario.n_vehicles,
    )

    solution = decode_permutation(permutation, scenario, matrix)
    assert solution.is_structurally_valid(scenario)
    assert sorted(solution.served()) == [0, 1, 2, 3]
    assert set(matrix.vehicle_start_index) == set(), "no starts were invented"


def test_cost_matrix_real_delhi_is_asymmetric() -> None:
    """One-way streets make real travel times direction-dependent."""
    import os

    if not os.environ.get("SMART_GATI_RUN_SLOW"):
        pytest.skip("set SMART_GATI_RUN_SLOW=1 to run against the real Delhi graph")

    from qgati.graph import is_delhi_graph_cached, load_delhi_graph

    if not is_delhi_graph_cached():
        pytest.skip("no cached Delhi graph")

    graph = load_delhi_graph()
    scenario = build_random_scenario(graph, 6, 2, seed=5)
    cost_matrix = build_cost_matrix(graph, scenario)

    assert not cost_matrix.is_symmetric()
    assert len(cost_matrix) == 7

    optimal = solve_brute_force(scenario, cost_matrix)
    heuristic = clarke_wright_savings(scenario, cost_matrix)
    optimal_cost = evaluate(optimal, scenario, cost_matrix).travel_cost
    heuristic_cost = evaluate(heuristic, scenario, cost_matrix).travel_cost

    print(
        f"\n[real Delhi n=6 k=2] brute=Rs{optimal_cost:.1f} "
        f"savings=Rs{heuristic_cost:.1f}"
    )
    assert heuristic_cost >= optimal_cost - TOLERANCE
