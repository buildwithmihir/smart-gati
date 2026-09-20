"""VRP solvers and the shared problem contract.

The data contract lives in :mod:`qgati.optimizer.models`; the cost function that
every solver is scored by lives in :mod:`qgati.optimizer.fitness`. Both are
deliberately free of any road-graph dependency — solvers see a cost matrix, not
a street network.

    >>> from qgati.optimizer import build_random_scenario, solve_brute_force
    >>> from qgati.graph import build_cost_matrix
    >>> scenario = build_random_scenario(graph, 6, 2, seed=1)
    >>> matrix = build_cost_matrix(graph, scenario)
    >>> solution = solve_brute_force(scenario, matrix)
"""

from qgati.optimizer.brute_force import MAX_EXACT_DELIVERIES, solve_brute_force
from qgati.optimizer.fitness import (
    Evaluation,
    PenaltyConfig,
    evaluate,
    route_travel_cost,
)
from qgati.optimizer.models import Delivery, Depot, Scenario, Solution, Vehicle
from qgati.optimizer.qpso import (
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_PARTICLES,
    decode_position,
    optimal_split,
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
    "build_random_scenario",
    "clarke_wright_savings",
    "decode_position",
    "evaluate",
    "optimal_split",
    "route_travel_cost",
    "run_qpso",
    "servable_nodes",
    "solve_brute_force",
]
