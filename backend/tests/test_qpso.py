"""QPSO tests.

Two claims are under test:

1. **Correctness** — QPSO must find the exact optimum, or come close, where an
   exact answer is computable. Brute force from Phase 2 is the arbiter.
2. **Value** — QPSO must beat the Clarke-Wright Savings baseline on instances
   too large to solve exactly, which is the whole reason it is here.

QPSO is stochastic, so every assertion is over a set of seeds rather than a
single run. Where a threshold is used, the comment records the measured
distribution it was chosen from.
"""

from __future__ import annotations

import pytest

from qgati.graph import build_cost_matrix, build_synthetic_graph
from qgati.graph.cost_matrix import CostMatrix
from qgati.optimizer import (
    Delivery,
    Depot,
    Scenario,
    Vehicle,
    build_random_scenario,
    clarke_wright_savings,
    decode_position,
    evaluate,
    optimal_split,
    run_qpso,
    solve_brute_force,
)
from qgati.optimizer.qpso import (
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_PARTICLES,
)

TOLERANCE = 1e-6
#: The brief's tolerance: a stochastic run may miss the optimum, but not by much.
ALLOWED_GAP = 0.05

# (deliveries, vehicles, scenario seed, label)
TINY_SPECS = [
    (4, 2, 101, "n=4 k=2"),
    (6, 2, 202, "n=6 k=2"),
    (8, 3, 303, "n=8 k=3"),
]
LARGE_SPECS = [
    (15, 3, 1, "n=15 k=3"),
    (20, 4, 2, "n=20 k=4"),
]

SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_graph():
    """Dense enough that its largest SCC comfortably fits the tiny instances."""
    return build_synthetic_graph(n_nodes=30, edge_prob=0.25, seed=7)


@pytest.fixture(scope="module")
def large_graph():
    """A harder matrix: sparser edges mean high asymmetry between directions."""
    return build_synthetic_graph(n_nodes=45, edge_prob=0.22, seed=11)


# --------------------------------------------------------------------------- #
# Correctness against the exact optimum
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_deliveries,n_vehicles,seed,label", TINY_SPECS)
def test_qpso_within_tolerance_of_optimum(
    small_graph, n_deliveries, n_vehicles, seed, label, capsys
) -> None:
    """Every run must land within 5% of the exact optimum, at default settings.

    Measured at 30 particles / 100 iterations over 10 seeds: worst gap 2.3%
    (n=8), 2.0% (n=6), 0.0% (n=4).
    """
    scenario = build_random_scenario(small_graph, n_deliveries, n_vehicles, seed=seed)
    cost_matrix = build_cost_matrix(small_graph, scenario)

    optimum = evaluate(solve_brute_force(scenario, cost_matrix), scenario, cost_matrix)
    runs = []
    for run_seed in SEEDS:
        solution, cost, _ = run_qpso(cost_matrix, scenario, seed=run_seed)
        evaluation = evaluate(solution, scenario, cost_matrix)
        assert evaluation.feasible, f"QPSO returned an infeasible solution (seed {run_seed})"
        runs.append(evaluation.travel_cost)

    best = min(runs)
    worst_gap = (max(runs) - optimum.travel_cost) / optimum.travel_cost

    with capsys.disabled():
        print(
            f"\n[{label}] brute={optimum.travel_cost:.2f}  "
            f"qpso best={best:.2f} mean={sum(runs)/len(runs):.2f} worst={max(runs):.2f}  "
            f"worst gap={worst_gap*100:.2f}%  exact={sum(1 for r in runs if r <= optimum.travel_cost + 1e-6)}/{len(runs)}"
        )

    assert worst_gap <= ALLOWED_GAP, (
        f"worst QPSO run was {worst_gap*100:.1f}% above optimum, tolerance is {ALLOWED_GAP*100:.0f}%"
    )
    # And it must be capable of finding the optimum exactly, not merely nearing it.
    assert best <= optimum.travel_cost + TOLERANCE


@pytest.mark.parametrize(
    "n_deliveries,n_vehicles,scenario_seed,label",
    [(6, 2, 202, "n=6 k=2"), (8, 3, 303, "n=8 k=3")],
)
def test_qpso_hits_optimum_in_most_runs(
    small_graph, n_deliveries, n_vehicles, scenario_seed, label, capsys
) -> None:
    """With a larger budget, most runs should land exactly on the optimum.

    Run at 300 iterations rather than the 100 default, because convergence on
    n=8 is still incomplete at 100 — measured exact-hit rates over 10 seeds were
    5/10 at 100 iterations, 6/10 at 200, and 9/10 at 300. Asserting "most runs"
    at the default settings would be asserting something not yet true.
    """
    scenario = build_random_scenario(
        small_graph, n_deliveries, n_vehicles, seed=scenario_seed
    )
    cost_matrix = build_cost_matrix(small_graph, scenario)
    optimum = evaluate(
        solve_brute_force(scenario, cost_matrix), scenario, cost_matrix
    ).travel_cost

    hits = 0
    for run_seed in range(8):
        solution, _, _ = run_qpso(cost_matrix, scenario, num_iterations=300, seed=run_seed)
        if evaluate(solution, scenario, cost_matrix).travel_cost <= optimum + 1e-6:
            hits += 1

    with capsys.disabled():
        print(f"\n[{label} @300 iters] exact optimum hit in {hits}/8 runs")

    assert hits >= 6, f"only {hits}/8 runs found the exact optimum"


def test_brute_force_arbitrates_qpso_and_savings(large_graph, capsys) -> None:
    """Where an exact answer exists, check both heuristics against it.

    On this graph the matrix is ~28% asymmetric, which is what makes Savings
    struggle; brute force confirms the gap is real rather than an artifact of
    either implementation.
    """
    print()
    for n_deliveries, n_vehicles, scenario_seed in ((9, 3, 2), (10, 3, 3)):
        scenario = build_random_scenario(
            large_graph, n_deliveries, n_vehicles, seed=scenario_seed
        )
        cost_matrix = build_cost_matrix(large_graph, scenario)

        optimum = evaluate(
            solve_brute_force(scenario, cost_matrix), scenario, cost_matrix
        ).travel_cost
        savings = evaluate(
            clarke_wright_savings(scenario, cost_matrix), scenario, cost_matrix
        ).travel_cost
        qpso_best = min(
            evaluate(run_qpso(cost_matrix, scenario, seed=s)[0], scenario, cost_matrix).travel_cost
            for s in SEEDS
        )

        with capsys.disabled():
            print(
                f"[n={n_deliveries} k={n_vehicles}] brute={optimum:.1f}  "
                f"savings={savings:.1f} (+{100*(savings-optimum)/optimum:.1f}%)  "
                f"qpso={qpso_best:.1f} (+{100*(qpso_best-optimum)/optimum:.1f}%)"
            )

        assert qpso_best <= optimum * (1 + ALLOWED_GAP)
        # Neither heuristic may beat the exact answer.
        assert savings >= optimum - TOLERANCE


# --------------------------------------------------------------------------- #
# Value over the Savings baseline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_deliveries,n_vehicles,seed,label", LARGE_SPECS)
def test_qpso_beats_savings(
    large_graph, n_deliveries, n_vehicles, seed, label, capsys
) -> None:
    """QPSO should beat Clarke-Wright on instances too large to solve exactly.

    Individual seeds may lose — QPSO is stochastic and Savings is a strong
    constructive heuristic — so this asserts on aggregate behaviour and prints
    any losing seed rather than failing on one.
    """
    scenario = build_random_scenario(large_graph, n_deliveries, n_vehicles, seed=seed)
    cost_matrix = build_cost_matrix(large_graph, scenario)

    savings_eval = evaluate(clarke_wright_savings(scenario, cost_matrix), scenario, cost_matrix)
    assert savings_eval.feasible

    runs = []
    for run_seed in SEEDS:
        solution, _, _ = run_qpso(cost_matrix, scenario, seed=run_seed)
        evaluation = evaluate(solution, scenario, cost_matrix)
        assert evaluation.feasible
        runs.append(evaluation.travel_cost)

    wins = sum(1 for cost in runs if cost < savings_eval.travel_cost - TOLERANCE)
    losses = [
        (seed, cost)
        for seed, cost in enumerate(runs)
        if cost > savings_eval.travel_cost + TOLERANCE
    ]
    best = min(runs)

    with capsys.disabled():
        print(
            f"\n[{label}] savings={savings_eval.travel_cost:.1f}  "
            f"qpso best={best:.1f} mean={sum(runs)/len(runs):.1f} worst={max(runs):.1f}  "
            f"wins={wins}/{len(runs)}  best vs savings={100*(best-savings_eval.travel_cost)/savings_eval.travel_cost:+.1f}%"
        )
        if losses:
            print(f"    seeds that lost to savings: {losses}")

    # The best run must clearly beat Savings, and most runs should too.
    assert best < savings_eval.travel_cost, (
        f"QPSO never beat Savings on {label}"
    )
    assert wins >= 3, f"QPSO only beat Savings on {wins}/{len(runs)} seeds"


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_qpso_is_reproducible(small_graph) -> None:
    scenario = build_random_scenario(small_graph, 6, 2, seed=202)
    cost_matrix = build_cost_matrix(small_graph, scenario)

    first = run_qpso(cost_matrix, scenario, seed=123)
    second = run_qpso(cost_matrix, scenario, seed=123)

    assert first[1] == second[1]
    assert first[0].routes == second[0].routes
    assert first[2] == second[2]

    # A different seed should explore differently.
    assert run_qpso(cost_matrix, scenario, seed=124)[2] != first[2]


def test_convergence_history_shape_and_monotonicity(small_graph) -> None:
    scenario = build_random_scenario(small_graph, 8, 3, seed=303)
    cost_matrix = build_cost_matrix(small_graph, scenario)

    _, best_cost, history = run_qpso(cost_matrix, scenario, num_iterations=40, seed=7)

    assert len(history) == 40
    assert all(later <= earlier + TOLERANCE for earlier, later in zip(history, history[1:])), (
        "best-so-far cost must never increase"
    )
    assert history[-1] == pytest.approx(best_cost)
    assert history[0] >= history[-1]


def test_qpso_respects_iteration_budget(small_graph) -> None:
    scenario = build_random_scenario(small_graph, 5, 2, seed=5)
    cost_matrix = build_cost_matrix(small_graph, scenario)
    assert len(run_qpso(cost_matrix, scenario, num_iterations=1, seed=1)[2]) == 1


def test_qpso_rejects_bad_hyperparameters(small_graph) -> None:
    scenario = build_random_scenario(small_graph, 4, 2, seed=1)
    cost_matrix = build_cost_matrix(small_graph, scenario)
    with pytest.raises(ValueError, match="num_particles"):
        run_qpso(cost_matrix, scenario, num_particles=0)
    with pytest.raises(ValueError, match="num_iterations"):
        run_qpso(cost_matrix, scenario, num_iterations=0)


def test_qpso_scales_with_swarm_size(small_graph) -> None:
    """A larger swarm should not do worse on average."""
    scenario = build_random_scenario(small_graph, 8, 3, seed=303)
    cost_matrix = build_cost_matrix(small_graph, scenario)
    small = [run_qpso(cost_matrix, scenario, num_particles=5, seed=s)[1] for s in SEEDS]
    large = [
        run_qpso(cost_matrix, scenario, num_particles=DEFAULT_NUM_PARTICLES, seed=s)[1]
        for s in SEEDS
    ]
    assert min(large) <= min(small) + TOLERANCE


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def test_decode_produces_feasible_fleet_shaped_solutions(small_graph) -> None:
    import numpy as np

    scenario = build_random_scenario(small_graph, 8, 3, seed=303)
    cost_matrix = build_cost_matrix(small_graph, scenario)
    rng = np.random.default_rng(0)

    for _ in range(50):
        solution = decode_position(rng.random(scenario.n_deliveries), scenario, cost_matrix)
        evaluation = evaluate(solution, scenario, cost_matrix)
        assert evaluation.feasible
        assert len(solution.routes) == scenario.n_vehicles
        assert sorted(solution.served()) == list(range(scenario.n_deliveries))
        for vehicle, route in enumerate(solution.routes):
            load = sum(scenario.demands[d] for d in route)
            assert load <= scenario.capacities[vehicle] + TOLERANCE


def test_optimal_split_prefers_the_cheaper_partition() -> None:
    """A hand-checkable split: two tight clusters should not be interleaved."""
    import numpy as np

    # depot at 0, one cluster near 10, another near 100
    points = [0.0, 10.0, 10.5, 100.0, 100.5]
    matrix = np.abs(np.subtract.outer(points, points)).astype(float)
    cost_matrix = CostMatrix(
        matrix=matrix,
        nodes=tuple(range(5)),
        delivery_node_index=(1, 2, 3, 4),
        # These numbers are travel seconds; the objective needs metres too, so
        # the toy network is taken to run at a constant 30 km/h.
        distance_matrix=matrix * (30.0 / 3.6),
    )
    scenario = Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=tuple(
            Delivery(id=f"D{i}", node=i + 1, demand=1) for i in range(4)
        ),
        vehicles=tuple(Vehicle(id=f"V{i}", capacity=2) for i in range(2)),
    )

    # Clusters are contiguous in this permutation, so the split must cut between
    # them: 222 seconds of driving against 401 for the interleaved alternative.
    # At this toy network's constant 30 km/h the combined objective is a fixed
    # multiple of the time — every leg prices at Rs0.1549/s — so the same split
    # wins either way, at Rs34.4 against Rs62.1.
    routes = optimal_split([0, 1, 2, 3], scenario, cost_matrix, capacity=2, max_routes=2)
    assert routes is not None
    groups = sorted(sorted(route) for route in routes)
    assert groups == [[0, 1], [2, 3]], f"expected clusters split apart, got {routes}"


def test_optimal_split_returns_none_when_infeasible() -> None:
    """The ``None`` path guards bin-packing infeasibility specifically.

    Aggregate infeasibility (total demand above fleet capacity) can never reach
    here — ``Scenario.__post_init__`` rejects it at construction. What remains is
    a single delivery no vehicle can carry, which passes that check because the
    fleet total is still sufficient.
    """
    import numpy as np

    cost_matrix = CostMatrix(
        matrix=np.zeros((3, 3)),
        nodes=(0, 1, 2),
        delivery_node_index=(1, 2),
        distance_matrix=np.zeros((3, 3)),
    )
    scenario = Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node=1, demand=12),  # heavier than any vehicle
            Delivery(id="B", node=2, demand=1),
        ),
        vehicles=(Vehicle(id="V0", capacity=10), Vehicle(id="V1", capacity=10)),
    )
    assert scenario.total_demand <= scenario.total_capacity  # passes validation

    assert optimal_split([0, 1], scenario, cost_matrix, capacity=10, max_routes=2) is None


# --------------------------------------------------------------------------- #
# Real Delhi graph — opt-in, needs a populated cache
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_qpso_on_real_delhi_graph(capsys) -> None:
    """The headline claim, on real travel times rather than synthetic ones.

    Measured: QPSO matches the exact optimum on n=6/8/10 and beats Savings by
    ~9-10% on n=15-20, where no exact answer is available. Costs are the combined
    rupee objective, not seconds.
    """
    import os

    from qgati.graph import is_delhi_graph_cached, load_delhi_graph

    if not os.environ.get("SMART_GATI_RUN_SLOW"):
        pytest.skip("set SMART_GATI_RUN_SLOW=1 to run against the real Delhi graph")
    if not is_delhi_graph_cached():
        pytest.skip("no cached Delhi graph; run load_delhi_graph() once")

    graph = load_delhi_graph()
    print()

    # Small enough to check against the exact optimum.
    for n_deliveries, n_vehicles, scenario_seed in ((6, 2, 11), (8, 3, 12)):
        scenario = build_random_scenario(graph, n_deliveries, n_vehicles, seed=scenario_seed)
        assert not build_cost_matrix(graph, scenario).is_symmetric()

        cost_matrix = build_cost_matrix(graph, scenario)
        optimum = evaluate(
            solve_brute_force(scenario, cost_matrix), scenario, cost_matrix
        ).travel_cost
        savings = evaluate(
            clarke_wright_savings(scenario, cost_matrix), scenario, cost_matrix
        ).travel_cost
        qpso_best = min(
            evaluate(run_qpso(cost_matrix, scenario, seed=s)[0], scenario, cost_matrix).travel_cost
            for s in SEEDS
        )

        with capsys.disabled():
            print(
                f"[Delhi n={n_deliveries} k={n_vehicles}] brute=Rs{optimum:.1f}  "
                f"savings=Rs{savings:.1f} (+{100*(savings-optimum)/optimum:.1f}%)  "
                f"qpso=Rs{qpso_best:.1f} (+{100*(qpso_best-optimum)/optimum:.1f}%)"
            )

        assert qpso_best <= optimum * (1 + ALLOWED_GAP)
        assert qpso_best < savings  # must beat the baseline on real data too

    # Beyond exact reach: only the Savings comparison is available.
    scenario = build_random_scenario(graph, 15, 3, seed=21)
    cost_matrix = build_cost_matrix(graph, scenario)
    savings = evaluate(
        clarke_wright_savings(scenario, cost_matrix), scenario, cost_matrix
    ).travel_cost
    runs = []
    for run_seed in SEEDS:
        solution, _, _ = run_qpso(cost_matrix, scenario, seed=run_seed)
        evaluation = evaluate(solution, scenario, cost_matrix)
        assert evaluation.feasible
        runs.append(evaluation.travel_cost)

    with capsys.disabled():
        print(
            f"[Delhi n=15 k=3] savings=Rs{savings:.1f}  qpso best=Rs{min(runs):.1f}  "
            f"({100*(min(runs)-savings)/savings:+.1f}%)  "
            f"wins={sum(1 for r in runs if r < savings)}/{len(runs)}"
        )

    assert min(runs) < savings
