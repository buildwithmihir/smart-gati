"""Quantum-behaved Particle Swarm Optimization for the VRP.

This is the project's headline solver. It is the genuine quantum-inspired
algorithm (Sun et al., 2004) — particles are attracted to a quantum potential
well and their positions sampled from a probability distribution — not classical
velocity-driven PSO wearing a different name.

Encoding, and why
-----------------
QPSO is a *continuous* optimizer: its position update moves real-valued
coordinates. The VRP is combinatorial. Something has to bridge that gap, and the
choice matters more than any hyperparameter.

**Positions are random keys.** A particle is a vector of ``n`` reals; sorting it
ascending yields a delivery permutation. This keeps positions continuous (so the
quantum update applies unchanged) while making every real vector decode to a
valid permutation. Crucially it is *order-preserving*: two nearby positions
decode to similar routes, so the search landscape has the locality QPSO's
gradient-free guidance relies on. A direct integer encoding would destroy that —
one unit of movement could reorder the whole tour.

**Route boundaries come from an optimal split, not from encoded split points.**
The brief suggested "permutation + split points per vehicle". Fixing the split
points into the position would put capacity feasibility at the mercy of the
search, leaving QPSO to spend its budget repairing overloaded vehicles instead of
shortening routes. Instead the split is *decoded optimally*: given the
permutation, a short dynamic program (Prins' split) finds the cheapest way to cut
it into at most ``k`` capacity-feasible routes. Capacity is then satisfied by
construction, every particle is a feasible solution, and the search optimises
pure travel cost.

The trade-off is that the split is no longer searched — but it is solved exactly
rather than heuristically, so nothing is lost: the optimal split dominates any
particular choice of split points the swarm might have encoded.

Cost of the extra dynamic program is ``O(k · n²)`` per decode, small next to the
``n`` evaluations of the cost function the route construction already needs.

Reference
---------
Sun, Feng, Xu (2004), *Particle swarm optimization with particles having quantum
behavior*, IEEE CEC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qgati.optimizer.fitness import evaluate
from qgati.optimizer.models import CAPACITY_EPSILON, Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = [
    "DEFAULT_BETA_END",
    "DEFAULT_BETA_START",
    "DEFAULT_NUM_ITERATIONS",
    "DEFAULT_NUM_PARTICLES",
    "decode_position",
    "optimal_split",
    "run_qpso",
]

DEFAULT_NUM_PARTICLES = 30
DEFAULT_NUM_ITERATIONS = 100

#: Contraction-expansion coefficient schedule. Standard practice is to decay
#: beta linearly from 1.0 to 0.5: large early displacement to explore, small
#: late displacement to converge. Beta controls the width of the quantum
#: potential well, and therefore how far a particle may tunnel from its
#: attractor.
DEFAULT_BETA_START = 1.0
DEFAULT_BETA_END = 0.5

#: Floor for the uniform sample feeding ``ln(1/u)``. u must stay strictly
#: positive or the logarithm diverges.
_U_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# Decoding: random keys -> permutation -> optimally split routes
# --------------------------------------------------------------------------- #
def optimal_split(
    permutation,
    scenario: Scenario,
    cost_matrix: CostMatrix,
    capacity: float | None = None,
    max_routes: int | None = None,
) -> list[list[int]] | None:
    """Cut a delivery permutation into the cheapest capacity-feasible routes.

    Dynamic program over prefixes: ``prev[j]`` is the cheapest way to serve the
    first ``j`` deliveries of the permutation using at most ``r`` routes. For
    each ``r`` a final route is appended covering ``perm[i:j]``, giving
    ``O(k · n²)`` overall.

    The inner loop exploits a rank-1 structure in the route costs. For a fixed
    permutation, the cost of serving ``perm[i:j]`` as one route decomposes as
    ``A[i] + B[j-1]``, so no per-candidate summation is needed.

    Returns ``None`` when no capacity-feasible split exists (only possible if a
    single demand exceeds ``capacity``, or the fleet is too small outright).
    """
    n = len(permutation)
    if n == 0:
        return []

    if capacity is None:
        capacity = max(scenario.capacities)
    if max_routes is None:
        max_routes = scenario.n_vehicles

    demands = scenario.demands
    index = cost_matrix.delivery_node_index
    matrix = cost_matrix.matrix
    depot = cost_matrix.depot_index

    # A[i] + B[j] == cost of the single route depot -> perm[i..j] -> depot.
    outbound = [float(matrix[depot, index[d]]) for d in permutation]
    inbound = [float(matrix[index[d], depot]) for d in permutation]
    leg_cumulative = [0.0] * n
    for t in range(1, n):
        leg_cumulative[t] = leg_cumulative[t - 1] + float(
            matrix[index[permutation[t - 1]], index[permutation[t]]]
        )
    combined = [outbound[i] - leg_cumulative[i] for i in range(n)]
    trailing = [leg_cumulative[j] + inbound[j] for j in range(n)]

    load_cumulative = [0.0] * (n + 1)
    for t, delivery in enumerate(permutation):
        load_cumulative[t + 1] = load_cumulative[t] + demands[delivery]

    infinity = float("inf")
    previous = [infinity] * (n + 1)
    previous[0] = 0.0
    # choice[r][j] = split point i used by route r at prefix j, or -1 if route r
    # went unused there.
    choice = [[-1] * (n + 1) for _ in range(max_routes + 1)]

    for r in range(1, max_routes + 1):
        current = previous[:]  # option: use fewer than r routes
        for j in range(1, n + 1):
            best = current[j]
            best_split = -1
            trailing_j = trailing[j - 1]
            load_j = load_cumulative[j]
            for i in range(j):
                if load_j - load_cumulative[i] > capacity + CAPACITY_EPSILON:
                    continue
                prefix_cost = previous[i]
                if prefix_cost == infinity:
                    continue
                candidate = prefix_cost + combined[i] + trailing_j
                if candidate < best:
                    best = candidate
                    best_split = i
            current[j] = best
            choice[r][j] = best_split
        previous = current

    if previous[n] == infinity:
        return None

    routes: list[list[int]] = []
    r, j = max_routes, n
    while j > 0:
        if r == 0:  # pragma: no cover - unreachable while previous[n] is finite
            return None
        split = choice[r][j]
        if split < 0:
            r -= 1  # this route carries nothing; step back a level
        else:
            routes.append(list(permutation[split:j]))
            j = split
            r -= 1
    routes.reverse()
    return routes


def _greedy_split(
    permutation, scenario: Scenario, capacity: float, max_routes: int
) -> list[list[int]]:
    """Fallback fill-to-capacity split. Only reached if the exact split fails.

    Overflow is pushed onto the final route rather than opening a route beyond
    the fleet, so the result stays fleet-shaped and the fitness function's
    capacity penalty — not a malformed solution — is what reports the problem.
    """
    demands = scenario.demands
    routes: list[list[int]] = []
    current: list[int] = []
    load = 0.0

    for delivery in permutation:
        if current and load + demands[delivery] > capacity + CAPACITY_EPSILON:
            routes.append(current)
            current, load = [], 0.0
        current.append(delivery)
        load += demands[delivery]
    if current:
        routes.append(current)

    while len(routes) > max_routes and max_routes > 0:
        routes[-2].extend(routes.pop())
    return routes


def _assemble(routes: list[list[int]], scenario: Scenario) -> Solution:
    """Pad routes to the fleet and hand the heaviest routes to the largest vehicles."""
    fleet = scenario.n_vehicles
    capacities = scenario.capacities
    demands = scenario.demands

    ordered = sorted(routes, key=lambda route: -sum(demands[d] for d in route))
    vehicle_order = sorted(range(fleet), key=lambda v: -capacities[v])

    assignment: list[tuple[int, ...]] = [()] * fleet
    for position, route in enumerate(ordered[:fleet]):
        assignment[vehicle_order[position]] = tuple(route)
    return Solution(routes=tuple(assignment))


def decode_position(
    position: np.ndarray, scenario: Scenario, cost_matrix: CostMatrix
) -> Solution:
    """Turn a continuous particle position into a VRP solution.

    Exposed because the explainability phase needs to show what a position
    means, not just what it costs.
    """
    permutation = list(np.argsort(position, kind="stable"))
    capacity = max(scenario.capacities)
    routes = optimal_split(permutation, scenario, cost_matrix, capacity)
    if routes is None:
        routes = _greedy_split(permutation, scenario, capacity, scenario.n_vehicles)
    return _assemble(routes, scenario)


# --------------------------------------------------------------------------- #
# QPSO
# --------------------------------------------------------------------------- #
def run_qpso(
    cost_matrix: CostMatrix,
    scenario: Scenario,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    seed: int | None = None,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
) -> tuple[Solution, float, list[float]]:
    """Optimize a VRP instance with Quantum-behaved Particle Swarm Optimization.

    The update, per particle per dimension, is the quantum one. With ``mbest``
    the mean of all personal bests and ``p`` an attractor drawn between the
    particle's own best and the swarm's best::

        p = phi * pbest + (1 - phi) * gbest          phi ~ U(0,1)
        x = p ± beta * |mbest - x| * ln(1/u)         u   ~ U(0,1)

    The sign is chosen at random and ``ln(1/u)`` is the quantum sampling term
    that lets a particle appear anywhere in the well rather than travelling
    there — the behaviour that distinguishes QPSO from classical PSO. ``beta``
    is the contraction-expansion coefficient, decayed linearly from
    ``beta_start`` to ``beta_end`` over the run.

    Parameters
    ----------
    cost_matrix, scenario
        A scenario and its precomputed costs. Note this argument order is the
        reverse of :func:`~qgati.optimizer.brute_force.solve_brute_force` and
        :func:`~qgati.optimizer.savings.clarke_wright_savings`, which take
        ``(scenario, cost_matrix)``.
    num_particles, num_iterations
        Swarm size and generations.
    seed
        Fixes the run. The same seed reproduces the same result exactly.
    beta_start, beta_end
        Contraction-expansion schedule.

    Returns
    -------
    (best_solution, best_cost, convergence_history)
        ``best_cost`` is the fitness actually minimised — travel cost plus any
        constraint penalties — and equals the travel cost whenever the solution
        is feasible, which the optimal-split decoder makes the normal case.
        ``convergence_history[i]`` is the best-so-far cost after generation
        ``i``; it is non-increasing and has exactly ``num_iterations`` entries.
    """
    if num_particles < 1:
        raise ValueError("num_particles must be at least 1")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    n = scenario.n_deliveries
    rng = np.random.default_rng(seed)

    # Random keys, uniform in [0, 1). Clipped back into that range after every
    # move: only the ordering of the keys carries meaning, so letting them drift
    # unboundedly would add nothing but numerical risk.
    positions = rng.random((num_particles, n))
    personal_best = positions.copy()
    personal_best_cost = np.full(num_particles, np.inf)

    def cost_of(position: np.ndarray) -> float:
        return evaluate(
            decode_position(position, scenario, cost_matrix), scenario, cost_matrix
        ).fitness

    global_best = positions[0].copy()
    global_best_cost = np.inf
    for i in range(num_particles):
        candidate = cost_of(positions[i])
        personal_best_cost[i] = candidate
        if candidate < global_best_cost:
            global_best_cost = candidate
            global_best = positions[i].copy()

    convergence: list[float] = [global_best_cost]

    for iteration in range(1, num_iterations):
        beta = beta_start + (beta_end - beta_start) * (iteration / (num_iterations - 1))

        # Mean of all personal bests — the centre of the quantum potential well
        # the whole swarm is attracted to.
        mean_best = personal_best.mean(axis=0)

        # Vectorised across the entire swarm at once. The three independent
        # draws per (particle, dimension) are sampled as whole arrays so the
        # random stream stays reproducible for a given seed.
        phi = rng.random((num_particles, n))
        u = 1.0 - rng.random((num_particles, n))  # (0, 1], so ln(1/u) >= 0
        positive = rng.random((num_particles, n)) < 0.5

        attractor = phi * personal_best + (1.0 - phi) * global_best
        displacement = (
            beta * np.abs(mean_best - positions) * np.log(1.0 / np.maximum(u, _U_FLOOR))
        )
        positions = np.where(positive, attractor + displacement, attractor - displacement)
        np.clip(positions, 0.0, 1.0, out=positions)

        for i in range(num_particles):
            candidate = cost_of(positions[i])
            if candidate < personal_best_cost[i]:
                personal_best_cost[i] = candidate
                personal_best[i] = positions[i].copy()
                if candidate < global_best_cost:
                    global_best_cost = candidate
                    global_best = positions[i].copy()

        convergence.append(global_best_cost)

    # Re-derive the returned solution from the winning position so the solution
    # and the reported cost can never disagree.
    best_solution = decode_position(global_best, scenario, cost_matrix)
    best_cost = evaluate(best_solution, scenario, cost_matrix).fitness
    return best_solution, best_cost, convergence
