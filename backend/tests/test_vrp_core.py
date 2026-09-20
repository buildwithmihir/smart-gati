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
    route_travel_cost,
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

    return CostMatrix(
        matrix=np.array(rows, dtype=float),
        nodes=("depot", "a", "b", "c", "d"),
        delivery_node_index=(1, 2, 3, 4),
    )


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
            f"brute={optimal_eval.travel_cost:.4f}s  "
            f"savings={heuristic_eval.travel_cost:.4f}s  "
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
    matrix = CostMatrix(
        matrix=np.ones((size, size)) - np.eye(size),
        nodes=tuple(range(size)),
        delivery_node_index=tuple(range(1, size)),
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
    scenario, cost_matrix = tiny_handmade
    # depot -> a (10) -> b (3) -> depot (12) = 25
    assert route_travel_cost((0, 1), scenario, cost_matrix) == pytest.approx(25.0)
    assert route_travel_cost((), scenario, cost_matrix) == 0.0


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


def test_cost_matrix_real_delhi_is_asymmetric() -> None:
    """One-way streets make real travel times direction-dependent."""
    import os

    if not os.environ.get("QGATI_RUN_SLOW"):
        pytest.skip("set QGATI_RUN_SLOW=1 to run against the real Delhi graph")

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
        f"\n[real Delhi n=6 k=2] brute={optimal_cost:.1f}s "
        f"savings={heuristic_cost:.1f}s"
    )
    assert heuristic_cost >= optimal_cost - TOLERANCE
