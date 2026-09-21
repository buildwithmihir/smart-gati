"""Ant Colony Optimization for the capacitated VRP.

Ants build routes stop by stop. Each ant walks a route until its vehicle is full
or nothing left fits, then opens the next vehicle; the next stop is drawn at
random with probability proportional to

    tau(i, j)^alpha · (1 / c(i, j))^beta

where ``tau`` is the learned pheromone on the leg and ``c(i, j)`` its travel
time. After every ant has built a solution the pheromone evaporates and the ants
reinforce what they used, so short routes accumulate trail and long ones fade.
That loop — construct, evaporate, reinforce — is the classical Ant System of
Dorigo et al.; the departure here is only that the deposit is weighted by
``1 / fitness`` rather than being applied to every ant equally.

Why capacity is respected by construction
-----------------------------------------
The candidate set at each step is filtered to deliveries that still fit the
current vehicle, so an ant can never overload one. The fleet is the only thing
that can genuinely run out: if the ants fill all ``k`` vehicles and deliveries
remain, those are appended to the vehicle with the largest residual capacity.
That is the outcome which *minimises* the total overflow, and it is deliberately
a solution the shared fitness function will reject on its capacity penalty rather
than a malformed one — the same rule
:func:`~qgati.optimizer.decoding.greedy_split` follows. Because ``1 / fitness``
then makes such an ant deposit almost no pheromone, the colony is pushed away
from overflowing constructions on its own.

Unlike the swarm solvers and the GA, ACO's routes are built directly rather than
decoded from a permutation, so it is the one baseline that exercises the
constraint handling rather than inheriting it. It is decoded through the same
:func:`~qgati.optimizer.decoding.assemble_routes` as everything else so the
resulting :class:`~qgati.optimizer.models.Solution` has the identical fleet shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qgati.optimizer.decoding import assemble_routes
from qgati.optimizer.fitness import evaluate
from qgati.optimizer.models import CAPACITY_EPSILON, Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEFAULT_DEPOSIT",
    "DEFAULT_EVAPORATION",
    "DEFAULT_NUM_ANTS",
    "DEFAULT_NUM_ITERATIONS",
    "run_aco",
]

#: Matches QPSO's 30 particles / 100 iterations — see the note in
#: :mod:`qgati.optimizer.genetic_algorithm`. Each ant is one candidate solution
#: per iteration, so this is 30 solutions per iteration in both cases.
DEFAULT_NUM_ANTS = 30
DEFAULT_NUM_ITERATIONS = 100

#: Pheromone weight. 1.0 is the classical Ant System value.
DEFAULT_ALPHA = 1.0

#: Heuristic (inverse travel time) weight. 2.0 is the classical value; it makes
#: an ant prefer nearby stops strongly enough to build sensible routes before any
#: pheromone has accumulated. Not to be confused with QPSO's contraction-
#: expansion coefficient, which shares the name and means something unrelated.
DEFAULT_BETA = 2.0

#: Evaporation rate per iteration. 0.1 leaves a long trail memory, which suits
#: instances this small — at 0.5 the colony forgets faster than it learns.
DEFAULT_EVAPORATION = 0.1

#: Pheromone laid by a solution of cost 1. Scaled by ``1 / fitness``, so this is
#: a unit and not a magnitude: the pheromone scale is set by
#: :func:`_initial_pheromone` to match whatever costs the instance produces.
DEFAULT_DEPOSIT = 1.0

#: Floor for ``1 / leg_cost``. Two deliveries can share a road node, making a leg
#: cost zero; without this the heuristic would be infinite.
_MIN_LEG = 1e-6

#: Pheromone never reaches exactly zero — a leg with no trail would be
#: unreachable forever, which would freeze the search's exploration.
_TAU_FLOOR = 1e-12


def _initial_pheromone(cost_matrix: CostMatrix) -> float:
    """Starting trail: the magnitude one ant's deposit would have.

    Anchoring to the data rather than picking a constant keeps the algorithm
    behaving the same on toy matrices (costs ~10) and on real Delhi travel times
    (costs ~10^3 seconds) — the same reasoning behind
    :meth:`~qgati.optimizer.fitness.PenaltyConfig.for_matrix`. Starting from an
    arbitrary 1.0 instead would make the heuristic term dominate for the first
    dozens of iterations and quietly turn ACO into a greedy nearest-neighbour.
    """
    matrix = cost_matrix.matrix
    size = matrix.shape[0]
    if size < 2:  # pragma: no cover - a cost matrix always holds depot + stops
        return 1.0
    off_diagonal = matrix[~np.eye(size, dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    scale = float(finite.mean()) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return 1.0 / (size * scale)


def _construct_routes(
    pheromone: np.ndarray,
    cost_matrix: CostMatrix,
    scenario: Scenario,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
) -> list[list[int]]:
    """Walk one ant: fill vehicle after vehicle until every delivery is placed.

    ``pheromone`` is indexed by *cost-matrix* index (depot at 0), not by delivery
    index; ``cost_matrix.delivery_node_index`` is the translation.
    """
    demands = scenario.demands
    capacities = scenario.capacities
    matrix = cost_matrix.matrix
    index = cost_matrix.delivery_node_index
    depot = cost_matrix.depot_index

    unvisited = set(range(scenario.n_deliveries))
    routes: list[list[int]] = []
    loads: list[float] = []

    for vehicle in range(scenario.n_vehicles):
        if not unvisited:
            break

        route: list[int] = []
        load = 0.0
        current = depot  # a cost-matrix index, as the pheromone matrix uses
        while True:
            # Sorted so the draw is reproducible: iterating the set directly
            # would make the outcome depend on set ordering.
            candidates = sorted(
                delivery
                for delivery in unvisited
                if load + demands[delivery] <= capacities[vehicle] + CAPACITY_EPSILON
            )
            if not candidates:
                break

            weights = np.empty(len(candidates))
            for position, delivery in enumerate(candidates):
                node = index[delivery]
                pheromone_term = float(pheromone[current, node]) ** alpha
                heuristic_term = (1.0 / max(float(matrix[current, node]), _MIN_LEG)) ** beta
                weights[position] = pheromone_term * heuristic_term

            total = float(weights.sum())
            if total > 0.0 and np.isfinite(total):
                draw = rng.random() * total
                pick = int(np.searchsorted(np.cumsum(weights), draw))
                pick = min(pick, len(candidates) - 1)
            else:  # pragma: no cover - weights are never all zero or non-finite
                pick = int(rng.integers(0, len(candidates)))

            delivery = candidates[pick]
            route.append(delivery)
            load += demands[delivery]
            unvisited.discard(delivery)
            current = index[delivery]

        routes.append(route)
        loads.append(load)

    if unvisited:
        # The fleet ran out before the ant ran out of deliveries. Putting the
        # remainder on the vehicle with the most room left minimises the total
        # overflow, and `evaluate` reports it as a capacity penalty.
        target = max(range(len(routes)), key=lambda v: capacities[v] - loads[v])
        routes[target].extend(sorted(unvisited))

    return routes


def _reinforce(
    pheromone: np.ndarray,
    solution: Solution,
    cost_matrix: CostMatrix,
    amount: float,
) -> None:
    """Add pheromone along every leg of a solution, depot to depot."""
    index = cost_matrix.delivery_node_index
    depot = cost_matrix.depot_index

    for route in solution.routes:
        if not route:
            continue
        previous = depot
        for delivery in route:
            node = index[delivery]
            pheromone[previous, node] += amount
            previous = node
        pheromone[previous, depot] += amount


def run_aco(
    cost_matrix: CostMatrix,
    scenario: Scenario,
    num_ants: int = DEFAULT_NUM_ANTS,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    seed: int | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    evaporation: float = DEFAULT_EVAPORATION,
    deposit: float = DEFAULT_DEPOSIT,
) -> tuple[Solution, float, list[float]]:
    """Optimize a VRP instance with Ant Colony Optimization.

    Parameters
    ----------
    cost_matrix, scenario
        A scenario and its precomputed costs, in the same order every other
        metaheuristic here takes them (see :func:`~qgati.optimizer.qpso.run_qpso`).
    num_ants, num_iterations
        Colony size and iterations. One iteration is one solution per ant
        followed by a single evaporation-and-reinforce pass.
    seed
        Fixes the run. The same seed reproduces the same result exactly.
    alpha, beta
        Pheromone and heuristic weights. ``beta`` here is the usual ACO
        heuristic weight, unrelated to QPSO's contraction-expansion coefficient.
    evaporation
        Fraction of pheromone lost each iteration, in ``[0, 1)``.
    deposit
        Reinforcement per unit of inverse fitness.

    Returns
    -------
    (best_solution, best_cost, convergence_history)
        Same shape as :func:`~qgati.optimizer.qpso.run_qpso`, so the benchmark
        runner can dispatch every metaheuristic uniformly.
        ``convergence_history[i]`` is the best-so-far cost after iteration ``i``,
        is non-increasing, and has exactly ``num_iterations`` entries.
    """
    if num_ants < 1:
        raise ValueError("num_ants must be at least 1")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")
    if alpha < 0.0:
        raise ValueError("alpha must not be negative")
    if beta < 0.0:
        raise ValueError("beta must not be negative")
    if not 0.0 <= evaporation < 1.0:
        raise ValueError("evaporation must be in [0, 1)")

    rng = np.random.default_rng(seed)

    size = len(cost_matrix)
    pheromone = np.full((size, size), _initial_pheromone(cost_matrix), dtype=float)

    best_solution: Solution | None = None
    best_cost = float("inf")
    convergence: list[float] = []

    for _ in range(num_iterations):
        colony: list[tuple[Solution, float]] = []
        for _ant in range(num_ants):
            solution = assemble_routes(
                _construct_routes(pheromone, cost_matrix, scenario, rng, alpha, beta),
                scenario,
            )
            cost = evaluate(solution, scenario, cost_matrix).fitness
            colony.append((solution, cost))
            if cost < best_cost:
                best_cost = cost
                best_solution = solution

        # Evaporate, then reinforce. Every ant deposits — weighted by
        # 1 / fitness, so an ant that overloaded a vehicle or dropped a delivery
        # (and therefore carries a penalty of ~10x the matrix scale) lays
        # essentially nothing, and the colony drifts away from it on its own.
        pheromone *= 1.0 - evaporation
        for solution, cost in colony:
            _reinforce(pheromone, solution, cost_matrix, deposit / max(cost, _MIN_LEG))
        np.maximum(pheromone, _TAU_FLOOR, out=pheromone)

        convergence.append(best_cost)

    if best_solution is None:  # pragma: no cover - num_iterations >= 1 and num_ants >= 1
        raise RuntimeError("no solution was constructed")

    return best_solution, best_cost, convergence
