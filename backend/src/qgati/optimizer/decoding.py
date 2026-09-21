"""Shared decoding: what an optimizer's internal representation *means* as routes.

Two of this project's four metaheuristics (QPSO, classical PSO) are *continuous*
optimizers. The other two (GA, ACO) work on permutations directly. The VRP is in
every case combinatorial, so something has to bridge the representational gap —
and that choice matters more than any hyperparameter. It is made once, here, and
shared, for two reasons:

* **Behaviour.** The encoding below is what lets a continuous search work at all.
* **Comparability.** The Phase 4 benchmark pits four metaheuristics against each
  other. If each carried its own decoder, the table would be measuring route
  construction and capacity handling rather than search quality. Every solver
  decodes through :func:`decode_permutation`, so capacity holds by construction
  for all of them, and a difference between two rows is a difference in search.

The encoding, and why
---------------------
**Positions are random keys.** A particle is a vector of ``n`` reals; sorting it
ascending yields a delivery permutation. This keeps positions continuous (so a
continuous update rule applies unchanged) while making every real vector decode
to a valid permutation. Crucially it is *order-preserving*: two nearby positions
decode to similar routes, so the search landscape has the locality that
gradient-free guidance relies on. A direct integer encoding would destroy that —
one unit of movement could reorder the whole tour.

**Route boundaries come from an optimal split, not from encoded split points.**
The brief suggested "permutation + split points per vehicle". Fixing the split
points into the position would put capacity feasibility at the mercy of the
search, leaving the optimizer to spend its budget repairing overloaded vehicles
instead of shortening routes. Instead the split is *decoded optimally*: given the
permutation, a short dynamic program (Prins' split) finds the cheapest way to cut
it into at most ``k`` capacity-feasible routes. Capacity is then satisfied by
construction, every candidate is a feasible solution, and the search optimises
pure travel cost.

The trade-off is that the split is no longer searched — but it is solved exactly
rather than heuristically, so nothing is lost: the optimal split dominates any
particular choice of split points a solver might have encoded.

Cost of the extra dynamic program is ``O(k · n²)`` per decode, small next to the
``n`` evaluations of the cost function the route construction already needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from qgati.optimizer.models import CAPACITY_EPSILON, Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = [
    "assemble_routes",
    "decode_permutation",
    "decode_position",
    "greedy_split",
    "optimal_split",
]


# --------------------------------------------------------------------------- #
# Optimal split: permutation -> capacity-feasible routes
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


def greedy_split(
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


def assemble_routes(routes: list[list[int]], scenario: Scenario) -> Solution:
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


# --------------------------------------------------------------------------- #
# Public decode entry points
# --------------------------------------------------------------------------- #
def decode_permutation(
    permutation: Sequence[int], scenario: Scenario, cost_matrix: CostMatrix
) -> Solution:
    """Turn a delivery permutation into a fleet-shaped, capacity-feasible solution.

    The single decode every permutation-based solver shares. Delivery indices are
    coerced to plain :class:`int` so a caller working in numpy — as the swarm
    solvers do — cannot leak ``np.int64`` into a :class:`Solution`.
    """
    sequence = [int(d) for d in permutation]
    capacity = max(scenario.capacities)
    routes = optimal_split(sequence, scenario, cost_matrix, capacity)
    if routes is None:
        routes = greedy_split(sequence, scenario, capacity, scenario.n_vehicles)
    return assemble_routes(routes, scenario)


def decode_position(
    position: np.ndarray, scenario: Scenario, cost_matrix: CostMatrix
) -> Solution:
    """Turn a continuous particle position into a VRP solution.

    Random keys to permutation (the argsort) and then
    :func:`decode_permutation`. Exposed because the explainability phase needs to
    show what a position means, not just what it costs.
    """
    return decode_permutation(
        np.argsort(position, kind="stable"), scenario, cost_matrix
    )
