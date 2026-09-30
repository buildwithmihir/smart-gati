"""Shared decoding: what an optimizer's internal representation *means* as routes.

Two of this project's three metaheuristics (QPSO, classical PSO) are *continuous*
optimizers. The other one (GA) works on permutations directly. The VRP is in
every case combinatorial, so something has to bridge the representational gap —
and that choice matters more than any hyperparameter. It is made once, here, and
shared, for two reasons:

* **Behaviour.** The encoding below is what lets a continuous search work at all.
* **Comparability.** The Phase 4 benchmark pits three metaheuristics against each
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
the weighted objective alone — "cheapest" throughout meaning
:attr:`~qgati.graph.cost_matrix.CostMatrix.objective_matrix`, the priced
combination of time, distance and fuel.

The trade-off is that the split is no longer searched — but it is solved exactly
rather than heuristically, so nothing is lost: the optimal split dominates any
particular choice of split points a solver might have encoded.

Cost of the extra dynamic program is ``O(k · n²)`` per decode, small next to the
``n`` evaluations of the cost function the route construction already needs.

When vehicles start somewhere other than the depot
--------------------------------------------------
A re-optimization (see :mod:`qgati.reopt`) hands this decoder a scenario whose
vehicles begin at scattered points, each with its own remaining capacity. Two
things follow, and they are the only two:

* **Which vehicle a route would belong to is decided by the split, not after
  it.** Route ``r`` of the split is vehicle ``r-1``'s, so its first leg is
  priced from that vehicle's start and its load is bounded by that vehicle's
  capacity. The rank-1 decomposition above survives intact — the outbound term
  simply becomes one row per vehicle instead of one vector — so the dynamic
  program keeps its shape and its ``O(k · n²)`` cost.
* **Route order can no longer be reshuffled.** :func:`assemble_routes` normally
  sorts routes by load and hands the heaviest to the largest vehicle. That would
  hand a route to a vehicle that did not start where the route was priced from,
  which is the whole thing the scenario asked for. Order is preserved instead.

Every branch of this is gated on :attr:`~qgati.optimizer.models.Scenario.has_custom_starts`.
A scenario that names no starts takes exactly the code it took before the
mechanism existed, down to the arithmetic, because the solver benchmark's
reproducibility rests on that being true.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from qgati.optimizer.fitness import route_total_cost
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

    ``capacity`` is the bound a route gets when every vehicle leaves the depot.
    When the scenario gives vehicles their own starts it is ignored in favour of
    each vehicle's own capacity, because route ``r`` here *is* vehicle ``r-1`` —
    see the module docstring.
    """
    n = len(permutation)
    if n == 0:
        return []

    if capacity is None:
        capacity = max(scenario.capacities)
    if max_routes is None:
        max_routes = scenario.n_vehicles

    each_vehicle_starts_apart = scenario.has_custom_starts
    if each_vehicle_starts_apart:
        # A route past the end of the fleet has no vehicle whose start it could
        # depart from, so there is nothing for one to mean here. `assemble_routes`
        # truncates to the fleet in any case; doing it now keeps every per-vehicle
        # lookup below in range without a guard at each one.
        max_routes = min(max_routes, scenario.n_vehicles)
    per_vehicle_capacity = scenario.capacities if each_vehicle_starts_apart else None

    demands = scenario.demands
    index = cost_matrix.delivery_node_index
    depot = cost_matrix.depot_index

    load_cumulative = [0.0] * (n + 1)
    for t, delivery in enumerate(permutation):
        load_cumulative[t + 1] = load_cumulative[t] + demands[delivery]

    if scenario.has_time_windows:
        # A window makes a route's cost depend on how long the route took to
        # reach each stop, so it is not a sum over its legs and the rank-1
        # decomposition below does not exist. Falling back to asking the shared
        # route cost directly costs an O(n) evaluation per candidate split —
        # ``O(k·n³)`` per decode rather than ``O(k·n²)`` — which is why it is
        # only paid when a window is actually present.
        #
        # The figure asked for is the same one the solvers minimise, lateness
        # included: a split that ignored windows would hand the search a set of
        # routes it never asked for, and the two would disagree about cheapest.
        #
        # The vehicle is passed through because the clock starts where that
        # vehicle is, and on a re-optimization the first leg is a different
        # length for each of them.
        def leg_cost(i: int, j: int, vehicle: int) -> float:
            return route_total_cost(
                permutation[i:j], scenario, cost_matrix, vehicle
            )

    else:
        # The priced objective, not raw travel time: the split is part of the
        # solution, so cutting the permutation into routes has to minimise the
        # same quantity the search does, or the two would disagree about what
        # "cheapest" means and the search would be handed a split it did not ask
        # for.
        matrix = cost_matrix.objective_matrix

        # A[i] + B[j] == cost of the single route <start> -> perm[i..j] -> depot.
        #
        # Only `outbound` depends on where the vehicle began, so only it has to
        # be rebuilt per vehicle; the inbound leg and the between-stops
        # cumulative are properties of the permutation alone.
        inbound = [float(matrix[index[d], depot]) for d in permutation]
        leg_cumulative = [0.0] * n
        for t in range(1, n):
            leg_cumulative[t] = leg_cumulative[t - 1] + float(
                matrix[index[permutation[t - 1]], index[permutation[t]]]
            )
        trailing = [leg_cumulative[j] + inbound[j] for j in range(n)]

        if each_vehicle_starts_apart:
            combined = [
                [
                    float(matrix[cost_matrix.start_index(v), index[d]])
                    - leg_cumulative[i]
                    for i, d in enumerate(permutation)
                ]
                for v in range(max_routes)
            ]

            def leg_cost(i: int, j: int, vehicle: int) -> float:
                return combined[vehicle][i] + trailing[j - 1]

        else:
            # One vector, indexed by nothing, exactly as before this mechanism
            # existed: every vehicle leaves the depot, so every row of the
            # matrix above would have been the same row.
            outbound = [float(matrix[depot, index[d]]) for d in permutation]
            combined = [outbound[i] - leg_cumulative[i] for i in range(n)]

            def leg_cost(i: int, j: int, vehicle: int) -> float:
                return combined[i] + trailing[j - 1]

    infinity = float("inf")
    previous = [infinity] * (n + 1)
    previous[0] = 0.0
    # choice[r][j] = split point i used by route r at prefix j, or -1 if route r
    # went unused there.
    choice = [[-1] * (n + 1) for _ in range(max_routes + 1)]

    for r in range(1, max_routes + 1):
        # Route r is vehicle r-1's, which is what makes a per-vehicle start and a
        # per-vehicle capacity expressible at all: the outer loop already counts
        # routes, so the vehicle is known one level up from where it is needed.
        vehicle = r - 1
        bound = (
            capacity
            if per_vehicle_capacity is None
            else per_vehicle_capacity[vehicle]
        )
        current = previous[:]  # option: use fewer than r routes
        for j in range(1, n + 1):
            best = current[j]
            best_split = -1
            load_j = load_cumulative[j]
            for i in range(j):
                if load_j - load_cumulative[i] > bound + CAPACITY_EPSILON:
                    continue
                prefix_cost = previous[i]
                if prefix_cost == infinity:
                    continue
                candidate = prefix_cost + leg_cost(i, j, vehicle)
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

    ``capacity`` is the bound per route when every vehicle leaves the depot; on a
    scenario with its own starts each route is filled to its own vehicle's
    capacity instead, for the same reason the exact split is: route ``r`` will be
    driven by vehicle ``r``.
    """
    demands = scenario.demands
    per_vehicle = scenario.capacities if scenario.has_custom_starts else None
    routes: list[list[int]] = []
    current: list[int] = []
    load = 0.0

    for delivery in permutation:
        # The route being filled will become route index len(routes).
        bound = capacity
        if per_vehicle is not None and len(routes) < len(per_vehicle):
            bound = per_vehicle[len(routes)]
        if current and load + demands[delivery] > bound + CAPACITY_EPSILON:
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
    """Pad routes to the fleet and hand the heaviest routes to the largest vehicles.

    That load-sorted assignment is skipped on a scenario where vehicles start
    apart, and the skip is not an optimisation — it is the difference between
    doing what the scenario asked and quietly not doing it. A route there was
    priced from vehicle ``r``'s own start and bounded by vehicle ``r``'s own
    capacity, so handing it to a different vehicle would charge it for a journey
    that vehicle is not making. Where every vehicle leaves the depot the
    assignment is free to be rearranged, because any vehicle can drive any of
    these routes at the same cost.

    Derived from the scenario rather than taken as an argument: the caller that
    would have to pass the flag is the decoder, and a flag is a thing a future
    caller can forget.
    """
    fleet = scenario.n_vehicles
    capacities = scenario.capacities
    demands = scenario.demands

    if scenario.has_custom_starts:
        assignment: list[tuple[int, ...]] = [tuple(route) for route in routes[:fleet]]
        assignment += [()] * (fleet - len(assignment))
        return Solution(routes=tuple(assignment))

    ordered = sorted(routes, key=lambda route: -sum(demands[d] for d in route))
    vehicle_order = sorted(range(fleet), key=lambda v: -capacities[v])

    assignment = [()] * fleet
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

    No argument for where vehicles start, because the scenario carries it: the
    five solvers hand a scenario in and never mention starts, and they must not
    have to. The ``capacity`` computed here is the single bound used when every
    vehicle leaves the depot; a scenario with its own starts supersedes it
    per route inside :func:`optimal_split`.
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
