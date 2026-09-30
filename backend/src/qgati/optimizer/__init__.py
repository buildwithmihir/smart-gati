"""VRP solvers and the shared problem contract.

The data contract lives in :mod:`qgati.optimizer.models`; the prices that turn
time, distance and fuel into one objective live in
:mod:`qgati.optimizer.objective`; the cost function every solver is scored by
lives in :mod:`qgati.optimizer.fitness`; the encoding that turns a search's
internal representation into routes lives in :mod:`qgati.optimizer.decoding`.
All four are deliberately free of any road-graph dependency — solvers see a cost
matrix, not a street network.

Five solvers, one contract:

===========================  ==========================================
:mod:`~qgati.optimizer.brute_force`  exact (Held-Karp) — ground truth
:mod:`~qgati.optimizer.savings`      Clarke-Wright — constructive baseline
:mod:`~qgati.optimizer.qpso`         quantum-behaved PSO — headline solver
:mod:`~qgati.optimizer.classical_pso` velocity-driven PSO — the control
:mod:`~qgati.optimizer.genetic_algorithm` GA — evolutionary baseline
===========================  ==========================================

A sixth, ACO, was removed; see "Removed: ACO" in ``DESIGN_DECISIONS.md`` for why
that is a deletion worth recording rather than a tidying-up.

The three metaheuristics all take ``(cost_matrix, scenario, ...)``, return
``(solution, cost, convergence_history)``, and decode through the same module, so
they are interchangeable — which is what lets
``benchmarks/run_comparison.py`` compare them on identical instances.

    >>> from qgati.optimizer import build_random_scenario, solve_brute_force
    >>> from qgati.graph import build_cost_matrix
    >>> scenario = build_random_scenario(graph, 6, 2, seed=1)
    >>> matrix = build_cost_matrix(graph, scenario)
    >>> solution = solve_brute_force(scenario, matrix)
"""

from qgati.optimizer.brute_force import MAX_EXACT_DELIVERIES, solve_brute_force
from qgati.optimizer.classical_pso import run_classical_pso
from qgati.optimizer.decoding import (
    assemble_routes,
    decode_permutation,
    decode_position,
    greedy_split,
    optimal_split,
)
from qgati.optimizer.fitness import (
    Evaluation,
    PenaltyConfig,
    RouteMetrics,
    evaluate,
    route_metrics,
    route_total_cost,
    route_travel_cost,
    route_travel_time,
)
from qgati.optimizer.genetic_algorithm import run_genetic_algorithm
from qgati.optimizer.models import Delivery, Depot, Scenario, Solution, Vehicle
from qgati.optimizer.objective import (
    DEFAULT_FUEL_MODEL,
    DEFAULT_LATE_PER_HOUR,
    DEFAULT_WEIGHTS,
    CostWeights,
    FuelModel,
)
from qgati.optimizer.qpso import (
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_PARTICLES,
    run_qpso,
)
from qgati.optimizer.registry import (
    DEFAULT_SOLVER_KEY,
    SOLVERS,
    SolverSpec,
    default_solver,
    get_solver,
    solver_keys,
)
from qgati.optimizer.savings import clarke_wright_savings
from qgati.optimizer.scenarios import build_random_scenario, servable_nodes

__all__ = [
    "DEFAULT_BETA_END",
    "DEFAULT_BETA_START",
    "DEFAULT_FUEL_MODEL",
    "DEFAULT_LATE_PER_HOUR",
    "DEFAULT_NUM_ITERATIONS",
    "DEFAULT_NUM_PARTICLES",
    "DEFAULT_SOLVER_KEY",
    "DEFAULT_WEIGHTS",
    "SOLVERS",
    "CostWeights",
    "Delivery",
    "Depot",
    "Evaluation",
    "FuelModel",
    "MAX_EXACT_DELIVERIES",
    "PenaltyConfig",
    "RouteMetrics",
    "Scenario",
    "Solution",
    "SolverSpec",
    "Vehicle",
    "assemble_routes",
    "build_random_scenario",
    "clarke_wright_savings",
    "decode_permutation",
    "decode_position",
    "default_solver",
    "evaluate",
    "get_solver",
    "greedy_split",
    "optimal_split",
    "route_metrics",
    "route_total_cost",
    "route_travel_cost",
    "route_travel_time",
    "run_classical_pso",
    "run_genetic_algorithm",
    "run_qpso",
    "servable_nodes",
    "solver_keys",
    "solve_brute_force",
]
