"""VRP solvers and the shared problem contract.

The data contract lives in :mod:`qgati.optimizer.models`; the cost function that
every solver is scored by lives in :mod:`qgati.optimizer.fitness`; the encoding
that turns a search's internal representation into routes lives in
:mod:`qgati.optimizer.decoding`. All three are deliberately free of any
road-graph dependency — solvers see a cost matrix, not a street network.

Six solvers, one contract:

===========================  ==========================================
:mod:`~qgati.optimizer.brute_force`  exact (Held-Karp) — ground truth
:mod:`~qgati.optimizer.savings`      Clarke-Wright — constructive baseline
:mod:`~qgati.optimizer.qpso`         quantum-behaved PSO — headline solver
:mod:`~qgati.optimizer.classical_pso` velocity-driven PSO — the control
:mod:`~qgati.optimizer.genetic_algorithm` GA — evolutionary baseline
:mod:`~qgati.optimizer.aco`          ant colony — pheromone baseline
===========================  ==========================================

The four metaheuristics all take ``(cost_matrix, scenario, ...)``, return
``(solution, cost, convergence_history)``, and decode through the same module, so
they are interchangeable — which is what lets
``benchmarks/run_comparison.py`` compare them on identical instances.

    >>> from qgati.optimizer import build_random_scenario, solve_brute_force
    >>> from qgati.graph import build_cost_matrix
    >>> scenario = build_random_scenario(graph, 6, 2, seed=1)
    >>> matrix = build_cost_matrix(graph, scenario)
    >>> solution = solve_brute_force(scenario, matrix)
"""

from qgati.optimizer.aco import run_aco
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
    evaluate,
    route_travel_cost,
)
from qgati.optimizer.genetic_algorithm import run_genetic_algorithm
from qgati.optimizer.models import Delivery, Depot, Scenario, Solution, Vehicle
from qgati.optimizer.qpso import (
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_PARTICLES,
    run_qpso,
)
from qgati.optimizer.savings import clarke_wright_savings
from qgati.optimizer.scenarios import build_random_scenario, servable_nodes

__all__ = [
    "DEFAULT_BETA_END",
    "DEFAULT_BETA_START",
    "DEFAULT_NUM_ITERATIONS",
    "DEFAULT_NUM_PARTICLES",
    "Delivery",
    "Depot",
    "Evaluation",
    "MAX_EXACT_DELIVERIES",
    "PenaltyConfig",
    "Scenario",
    "Solution",
    "Vehicle",
    "assemble_routes",
    "build_random_scenario",
    "clarke_wright_savings",
    "decode_permutation",
    "decode_position",
    "evaluate",
    "greedy_split",
    "optimal_split",
    "route_travel_cost",
    "run_aco",
    "run_classical_pso",
    "run_genetic_algorithm",
    "run_qpso",
    "servable_nodes",
    "solve_brute_force",
]
