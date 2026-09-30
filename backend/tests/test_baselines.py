"""Baseline solver tests: genetic algorithm and classical PSO.

Three claims are under test:

1. **Contract** — every Phase 4 solver answers in the same shape as QPSO: a
   feasible, fleet-shaped solution, a cost that agrees with that solution, and a
   convergence history of exactly the requested length. This is what lets
   ``benchmarks/run_comparison.py`` dispatch every solver interchangeably, so it
   is asserted against the three metaheuristics rather than QPSO alone.
2. **Quality** — all of them reach the exact optimum on instances small enough
   for brute force, and beat Clarke-Wright Savings on instances too large to
   solve exactly.
3. **Comparability** — classical PSO is only meaningful as a controlled
   experiment against QPSO, so its budget is asserted to be the same *values*,
   not merely a similar-looking pair of constants.

All three are stochastic, so every assertion is over a set of seeds. Thresholds
record the measured distribution they were chosen from; where a solver's
*typical* run would fail a threshold, the assertion is on its best run instead —
asserting more than the measurements support would make the suite flaky, which
is worse than asserting less.
"""

from __future__ import annotations

import numpy as np
import pytest

from qgati.graph import build_cost_matrix, build_synthetic_graph
from qgati.optimizer import (
    build_random_scenario,
    clarke_wright_savings,
    evaluate,
    run_classical_pso,
    run_genetic_algorithm,
    run_qpso,
    solve_brute_force,
)
from qgati.optimizer.classical_pso import DEFAULT_NUM_ITERATIONS as CPSO_ITERATIONS
from qgati.optimizer.classical_pso import DEFAULT_NUM_PARTICLES as CPSO_PARTICLES
from qgati.optimizer.genetic_algorithm import order_crossover, swap_mutation
from qgati.optimizer.qpso import DEFAULT_NUM_ITERATIONS as QPSO_ITERATIONS
from qgati.optimizer.qpso import DEFAULT_NUM_PARTICLES as QPSO_PARTICLES

TOLERANCE = 1e-6
#: A solver's best run may miss the optimum by this much.
ALLOWED_GAP = 0.05

SEEDS = (0, 1, 2, 3, 4)

#: (label, callable, name of that solver's generation-budget argument)
SOLVERS = (
    ("QPSO", run_qpso, "num_iterations"),
    ("Classical PSO", run_classical_pso, "num_iterations"),
    ("Genetic Algorithm", run_genetic_algorithm, "num_generations"),
)

#: Explicit test ids, so a failure names the solver rather than an address.
SOLVER_IDS = [label for label, _, _ in SOLVERS]

TINY_SPECS = [
    (4, 2, 101, "n=4 k=2"),
    (6, 2, 202, "n=6 k=2"),
    (8, 3, 303, "n=8 k=3"),
]


@pytest.fixture(scope="module")
def small_graph():
    return build_synthetic_graph(n_nodes=30, edge_prob=0.25, seed=7)


@pytest.fixture(scope="module")
def large_graph():
    """A harder matrix: sparser edges mean high asymmetry between directions."""
    return build_synthetic_graph(n_nodes=45, edge_prob=0.22, seed=11)


def _run(name, solver, budget_kwarg, cost_matrix, scenario, seed, budget=None):
    kwargs = {budget_kwarg: budget} if budget is not None else {}
    return solver(cost_matrix, scenario, seed=seed, **kwargs)


# --------------------------------------------------------------------------- #
# The shared contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,solver,budget_kwarg", SOLVERS, ids=SOLVER_IDS)
def test_solvers_return_feasible_fleet_shaped_solutions(
    small_graph, label, solver, budget_kwarg
) -> None:
    """The output contract every solver must satisfy, asserted on all three."""
    scenario = build_random_scenario(small_graph, 8, 3, seed=303)
    cost_matrix = build_cost_matrix(small_graph, scenario)

    for seed in SEEDS:
        solution, cost, _ = _run(label, solver, budget_kwarg, cost_matrix, scenario, seed)
        evaluation = evaluate(solution, scenario, cost_matrix)

        assert evaluation.feasible, f"{label} returned an infeasible solution at seed {seed}"
        assert len(solution.routes) == scenario.n_vehicles
        assert sorted(solution.served()) == list(range(scenario.n_deliveries))
        for vehicle, route in enumerate(solution.routes):
            load = sum(scenario.demands[d] for d in route)
            assert load <= scenario.capacities[vehicle] + TOLERANCE
        # The reported cost must be the fitness of the returned solution.
        assert cost == pytest.approx(evaluation.fitness)


@pytest.mark.parametrize("label,solver,budget_kwarg", SOLVERS, ids=SOLVER_IDS)
def test_convergence_history_shape_and_monotonicity(
    small_graph, label, solver, budget_kwarg
) -> None:
    scenario = build_random_scenario(small_graph, 8, 3, seed=303)
    cost_matrix = build_cost_matrix(small_graph, scenario)

    _, best_cost, history = _run(
        label, solver, budget_kwarg, cost_matrix, scenario, 7, budget=40
    )

    assert len(history) == 40, f"{label} ignored its generation budget"
    assert all(
        later <= earlier + TOLERANCE for earlier, later in zip(history, history[1:])
    ), f"{label} best-so-far cost increased"
    assert history[-1] == pytest.approx(best_cost)


@pytest.mark.parametrize("label,solver,budget_kwarg", SOLVERS, ids=SOLVER_IDS)
def test_solvers_are_reproducible(small_graph, label, solver, budget_kwarg) -> None:
    scenario = build_random_scenario(small_graph, 6, 2, seed=202)
    cost_matrix = build_cost_matrix(small_graph, scenario)

    first = _run(label, solver, budget_kwarg, cost_matrix, scenario, 123)
    second = _run(label, solver, budget_kwarg, cost_matrix, scenario, 123)

    assert first[1] == second[1]
    assert first[0].routes == second[0].routes
    assert first[2] == second[2]
    # A different seed must actually explore differently.
    assert _run(label, solver, budget_kwarg, cost_matrix, scenario, 124)[2] != first[2]


# --------------------------------------------------------------------------- #
# Quality against the exact optimum
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,solver,budget_kwarg", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("n_deliveries,n_vehicles,scenario_seed,spec", TINY_SPECS)
def test_solvers_reach_the_optimum_on_tiny_instances(
    small_graph, label, solver, budget_kwarg, n_deliveries, n_vehicles, scenario_seed, spec, capsys
) -> None:
    """Every solver's best run must land within 5% of the exact optimum.

    Measured at default settings over 5 seeds, worst best-run gap per solver:
    QPSO 0.0%, classical PSO 0.0%, GA 0.0%. The 5% bound therefore holds with
    margin for all three, but asserting a *per-run* bound would not: classical
    PSO's single worst run on n=8 k=3 was 7.3% above optimum.
    """
    scenario = build_random_scenario(
        small_graph, n_deliveries, n_vehicles, seed=scenario_seed
    )
    cost_matrix = build_cost_matrix(small_graph, scenario)
    optimum = evaluate(
        solve_brute_force(scenario, cost_matrix), scenario, cost_matrix
    ).travel_cost

    runs = []
    for seed in SEEDS:
        solution, _, _ = _run(label, solver, budget_kwarg, cost_matrix, scenario, seed)
        evaluation = evaluate(solution, scenario, cost_matrix)
        assert evaluation.feasible
        runs.append(evaluation.travel_cost)

    best_gap = (min(runs) - optimum) / optimum

    with capsys.disabled():
        print(
            f"\n[{spec} {label}] brute={optimum:.1f}  best={min(runs):.1f} "
            f"(+{best_gap*100:.2f}%)  worst=+{100*(max(runs)-optimum)/optimum:.2f}%  "
            f"exact={sum(1 for r in runs if r <= optimum + TOLERANCE)}/{len(runs)}"
        )

    assert best_gap <= ALLOWED_GAP, (
        f"{label} best run was {best_gap*100:.1f}% above optimum on {spec}, "
        f"tolerance is {ALLOWED_GAP*100:.0f}%"
    )


@pytest.mark.parametrize("label,solver,budget_kwarg", SOLVERS, ids=SOLVER_IDS)
def test_solvers_beat_savings_beyond_exact_reach(
    large_graph, label, solver, budget_kwarg, capsys
) -> None:
    """The whole point of a metaheuristic: beat the constructive baseline.

    Individual seeds may lose — all three are stochastic and Savings is a strong
    heuristic — so this asserts on aggregate behaviour. Measured on n=15 k=3
    under the earlier time-only objective: every solver beat Savings on 5/5
    seeds, by 33-44%. The assertion below is deliberately weaker (3/5) so it
    does not pin a margin that the combined objective may move.
    """
    scenario = build_random_scenario(large_graph, 15, 3, seed=1)
    cost_matrix = build_cost_matrix(large_graph, scenario)

    savings = evaluate(
        clarke_wright_savings(scenario, cost_matrix), scenario, cost_matrix
    )
    assert savings.feasible

    runs = []
    for seed in SEEDS:
        solution, _, _ = _run(label, solver, budget_kwarg, cost_matrix, scenario, seed)
        evaluation = evaluate(solution, scenario, cost_matrix)
        assert evaluation.feasible
        runs.append(evaluation.travel_cost)

    wins = sum(1 for cost in runs if cost < savings.travel_cost - TOLERANCE)

    with capsys.disabled():
        print(
            f"\n[n=15 k=3 {label}] savings={savings.travel_cost:.1f}  "
            f"best={min(runs):.1f} ({100*(min(runs)-savings.travel_cost)/savings.travel_cost:+.1f}%)  "
            f"mean={sum(runs)/len(runs):.1f}  wins={wins}/{len(runs)}"
        )

    assert min(runs) < savings.travel_cost, f"{label} never beat Savings"
    assert wins >= 3, f"{label} only beat Savings on {wins}/{len(runs)} seeds"


# --------------------------------------------------------------------------- #
# Comparability: classical PSO is a controlled experiment against QPSO
# --------------------------------------------------------------------------- #
def test_classical_pso_shares_qpso_budget() -> None:
    """The two swarms must run at the same budget, or the comparison lies.

    Classical PSO imports these rather than restating them, so that tuning one
    solver cannot silently stop the benchmark from comparing like with like.
    """
    assert CPSO_PARTICLES == QPSO_PARTICLES
    assert CPSO_ITERATIONS == QPSO_ITERATIONS


def test_classical_pso_differs_from_qpso_only_in_the_update(
    small_graph, capsys
) -> None:
    """Both swarms must decode identically, and so explore comparably.

    They share a representation and a decoder, so on an instance they both solve
    exactly, both must find the optimum. If the two ever diverged in decoding,
    neither could claim the other as a fair control.
    """
    scenario = build_random_scenario(small_graph, 6, 2, seed=202)
    cost_matrix = build_cost_matrix(small_graph, scenario)
    optimum = evaluate(
        solve_brute_force(scenario, cost_matrix), scenario, cost_matrix
    ).travel_cost

    qpso = [
        evaluate(run_qpso(cost_matrix, scenario, seed=s)[0], scenario, cost_matrix).travel_cost
        for s in SEEDS
    ]
    classical = [
        evaluate(
            run_classical_pso(cost_matrix, scenario, seed=s)[0], scenario, cost_matrix
        ).travel_cost
        for s in SEEDS
    ]

    with capsys.disabled():
        print(
            f"\n[n=6 k=2] optimum={optimum:.1f}  qpso best={min(qpso):.1f}  "
            f"classical best={min(classical):.1f}"
        )

    assert min(qpso) <= optimum + TOLERANCE
    assert min(classical) <= optimum + TOLERANCE


# --------------------------------------------------------------------------- #
# GA operators
# --------------------------------------------------------------------------- #
def test_order_crossover_produces_a_valid_permutation() -> None:
    rng = np.random.default_rng(0)
    for size in (2, 5, 8, 25):
        for _ in range(50):
            parent_a = list(rng.permutation(size))
            parent_b = list(rng.permutation(size))
            child = order_crossover(parent_a, parent_b, rng)
            assert sorted(child) == list(range(size)), "OX must return a permutation"


def test_order_crossover_inherits_a_contiguous_block_from_the_first_parent() -> None:
    """The defining property of OX1: a block of the first parent survives intact.

    With ``parent_a`` the identity permutation, an inherited block shows up as an
    adjacent pair of consecutive integers. The segment bounds are random, so this
    checks the property rather than one expected child.
    """
    rng = np.random.default_rng(1)
    parent_a = list(range(8))
    parent_b = [7, 6, 5, 4, 3, 2, 1, 0]
    for _ in range(50):
        child = order_crossover(parent_a, parent_b, rng)
        assert any(
            child[i + 1] == child[i] + 1 for i in range(len(child) - 1)
        ), f"no block of the first parent survived: {child}"


def test_swap_mutation_stays_a_permutation_and_is_rate_sensitive() -> None:
    rng = np.random.default_rng(2)
    original = list(range(20))

    untouched = list(original)
    swap_mutation(untouched, rng, 0.0)
    assert untouched == original, "rate 0 must not mutate"

    for _ in range(20):
        mutated = list(original)
        swap_mutation(mutated, rng, 0.5)
        assert sorted(mutated) == original


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,solver,budget_kwarg", SOLVERS, ids=SOLVER_IDS)
def test_solvers_reject_bad_hyperparameters(
    small_graph, label, solver, budget_kwarg
) -> None:
    scenario = build_random_scenario(small_graph, 4, 2, seed=1)
    cost_matrix = build_cost_matrix(small_graph, scenario)

    with pytest.raises(ValueError, match=budget_kwarg):
        solver(cost_matrix, scenario, **{budget_kwarg: 0})


def test_solvers_reject_bad_solver_specific_arguments(small_graph) -> None:
    scenario = build_random_scenario(small_graph, 4, 2, seed=1)
    cost_matrix = build_cost_matrix(small_graph, scenario)

    with pytest.raises(ValueError, match="population_size"):
        run_genetic_algorithm(cost_matrix, scenario, population_size=0)
    with pytest.raises(ValueError, match="elites"):
        run_genetic_algorithm(cost_matrix, scenario, elites=30)
    with pytest.raises(ValueError, match="mutation_rate"):
        run_genetic_algorithm(cost_matrix, scenario, mutation_rate=2.0)
    with pytest.raises(ValueError, match="crossover_rate"):
        run_genetic_algorithm(cost_matrix, scenario, crossover_rate=-0.1)
    with pytest.raises(ValueError, match="tournament_size"):
        run_genetic_algorithm(cost_matrix, scenario, tournament_size=0)
    # Would otherwise surface as numpy's "sample larger than population".
    with pytest.raises(ValueError, match="tournament_size"):
        run_genetic_algorithm(cost_matrix, scenario, population_size=4, tournament_size=5)

    with pytest.raises(ValueError, match="max_velocity"):
        run_classical_pso(cost_matrix, scenario, max_velocity=0.0)
