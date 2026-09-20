"""Shared solution evaluation — the one cost function every optimizer reuses.

Brute force, Savings, and later QPSO/GA/PSO/ACO all score candidate solutions
through :func:`evaluate`. Sharing it is what makes their results comparable: if
each solver had its own notion of "cost", a benchmark between them would be
measuring the accounting rather than the search.

The model is a penalty function. Travel cost is minimised, and constraint
violations add penalties large enough that an infeasible solution can never
outscore a feasible one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qgati.optimizer.models import CAPACITY_EPSILON, Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = ["Evaluation", "PenaltyConfig", "evaluate", "route_travel_cost"]

#: Multiplier applied to the matrix scale to build default penalties.
#: Ten times the longest leg comfortably exceeds the most a solver could ever
#: save by dropping a delivery (~2 legs) or overloading a vehicle (~1 leg).
_PENALTY_MULTIPLIER = 10.0


@dataclass(frozen=True, slots=True)
class PenaltyConfig:
    """Penalty weights. Defaults scale off the cost matrix, so no hand-tuning."""

    capacity_per_unit: float
    missing_delivery: float
    duplicate_delivery: float
    wrong_route_count: float

    @classmethod
    def for_matrix(cls, matrix: np.ndarray) -> PenaltyConfig:
        """Derive penalties from the magnitude of the costs being minimised.

        Anchoring to the data keeps the function usable on both toy matrices
        (costs ~10) and real Delhi travel times (costs ~10^3 seconds) without
        the caller picking constants per scenario.
        """
        finite = matrix[np.isfinite(matrix)]
        scale = float(finite.max()) if finite.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return cls(
            capacity_per_unit=_PENALTY_MULTIPLIER * scale,
            missing_delivery=_PENALTY_MULTIPLIER * scale,
            duplicate_delivery=_PENALTY_MULTIPLIER * scale,
            wrong_route_count=_PENALTY_MULTIPLIER * scale,
        )


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The full breakdown of a solution's score. Lower ``fitness`` is better."""

    travel_cost: float
    capacity_penalty: float
    coverage_penalty: float
    shape_penalty: float
    route_costs: tuple[float, ...]
    route_loads: tuple[float, ...]

    @property
    def penalty(self) -> float:
        return self.capacity_penalty + self.coverage_penalty + self.shape_penalty

    @property
    def fitness(self) -> float:
        """Objective the optimizers minimise: travel cost plus all penalties."""
        return self.travel_cost + self.penalty

    @property
    def feasible(self) -> bool:
        """True when no constraint is violated at all."""
        return self.penalty <= CAPACITY_EPSILON

    def summary(self) -> str:
        state = "feasible" if self.feasible else f"INFEASIBLE (penalty {self.penalty:g})"
        return f"cost={self.travel_cost:.4f} {state}"


def route_travel_cost(
    route: tuple[int, ...] | list[int],
    scenario: Scenario,
    cost_matrix: CostMatrix,
) -> float:
    """Travel time for one route, depot -> stops in order -> depot.

    An empty route costs nothing: the vehicle simply never leaves.
    """
    if not route:
        return 0.0

    index = cost_matrix.delivery_node_index
    matrix = cost_matrix.matrix
    depot = cost_matrix.depot_index

    total = matrix[depot, index[route[0]]]
    for previous, following in zip(route, route[1:]):
        total += matrix[index[previous], index[following]]
    total += matrix[index[route[-1]], depot]
    return float(total)


def evaluate(
    solution: Solution,
    scenario: Scenario,
    cost_matrix: CostMatrix,
    penalties: PenaltyConfig | None = None,
) -> Evaluation:
    """Score a candidate solution.

    Checks three things, and reports each separately so a caller can tell *why*
    a solution was rejected rather than only that it was:

    * **capacity** — no route's demand load exceeds its vehicle's capacity
    * **coverage** — every delivery served exactly once (none missed, none repeated)
    * **shape** — one route per vehicle, matching the fleet
    """
    if penalties is None:
        penalties = PenaltyConfig.for_matrix(cost_matrix.matrix)

    demands = scenario.demands
    capacities = scenario.capacities
    n_deliveries = scenario.n_deliveries
    n_vehicles = scenario.n_vehicles

    route_costs: list[float] = []
    route_loads: list[float] = []
    capacity_penalty = 0.0

    for position, route in enumerate(solution.routes):
        load = float(sum(demands[d] for d in route if 0 <= d < n_deliveries))
        route_loads.append(load)
        route_costs.append(route_travel_cost(route, scenario, cost_matrix))

        if position < n_vehicles:
            excess = load - capacities[position]
            if excess > CAPACITY_EPSILON:
                capacity_penalty += penalties.capacity_per_unit * excess

    # Coverage: count how many times each delivery index is visited.
    counts = Counter(solution.served())
    coverage_penalty = 0.0
    for delivery in range(n_deliveries):
        visits = counts.get(delivery, 0)
        if visits == 0:
            coverage_penalty += penalties.missing_delivery
        elif visits > 1:
            coverage_penalty += penalties.duplicate_delivery * (visits - 1)
    for delivery, visits in counts.items():
        if not 0 <= delivery < n_deliveries:
            coverage_penalty += penalties.duplicate_delivery * visits

    shape_penalty = (
        0.0
        if len(solution.routes) == n_vehicles
        else penalties.wrong_route_count
    )

    return Evaluation(
        travel_cost=float(sum(route_costs)),
        capacity_penalty=float(capacity_penalty),
        coverage_penalty=float(coverage_penalty),
        shape_penalty=float(shape_penalty),
        route_costs=tuple(route_costs),
        route_loads=tuple(route_loads),
    )
