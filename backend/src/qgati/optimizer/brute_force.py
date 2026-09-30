"""Exact VRP solver for small instances — the ground truth to check others against.

Correctness beats speed here: this exists so the heuristics (Savings now, QPSO
and friends later) can be measured against a provably optimal answer rather than
against each other.

Method
------
1. **Held-Karp** over every subset of deliveries computes the optimal depot ->
   subset -> depot tour for all ``2^n`` subsets at once, in ``O(2^n · n^2)``.
   The heuristics' per-route ordering problem is therefore solved exactly and
   once, not re-derived per candidate.
2. **Partition search** enumerates ways to assign deliveries to vehicles, pruned
   by capacity, and minimises the sum of the precomputed subset tours.

Step 1 is what keeps this fast: without it, every one of the ``k^n`` assignments
would need its own travelling-salesman solve.

Starts
------
A scenario can name a different starting node per vehicle — a re-optimized fleet
is already out on the road, so its vehicles begin wherever they are. Step 1 is
therefore run **once per distinct start node**, and step 2 minimises against each
vehicle's own table. Only the outbound leg moves; every route still returns to
the depot. With no starts named there is one table, which is the depot, and the
whole of this collapses to the ordinary framing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from qgati.optimizer.models import CAPACITY_EPSILON, Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix
    from qgati.optimizer.objective import CostWeights

__all__ = ["MAX_EXACT_DELIVERIES", "solve_brute_force"]

#: Above this, ``2^n · n^2`` work and ``k^n`` assignments stop being instant.
MAX_EXACT_DELIVERIES = 10


def solve_brute_force(scenario: Scenario, cost_matrix: CostMatrix) -> Solution:
    """Return a provably optimal solution, or raise if none exists.

    Raises
    ------
    ValueError
        If the instance is too large for exact search, or no capacity-feasible
        assignment of deliveries to vehicles exists at all.
    """
    n = scenario.n_deliveries
    k = scenario.n_vehicles
    infinity = np.inf

    if n > MAX_EXACT_DELIVERIES:
        raise ValueError(
            f"brute force is limited to {MAX_EXACT_DELIVERIES} deliveries, got {n}; "
            "use a heuristic for larger instances"
        )
    if n == 0:  # pragma: no cover - Scenario forbids this
        return Solution(routes=tuple(() for _ in range(k)))

    # One Held-Karp pass per *distinct* start node, not per vehicle: the tournament
    # below is indexed by where a vehicle begins, and two vehicles that begin in
    # the same place share the whole of it. On a scenario that names no starts
    # every vehicle begins at the depot and this is a single pass — exactly what
    # it was before per-vehicle starts existed.
    starts = [cost_matrix.start_index(vehicle) for vehicle in range(k)]
    tours: dict[int, tuple[np.ndarray, "Callable[[int], tuple[int, ...]]"]] = {}
    for start in dict.fromkeys(starts):
        local, local_time = _local_arrays(cost_matrix, n, start)
        if scenario.has_time_windows:
            tours[start] = _windowed_tours(
                scenario, cost_matrix, n, local, local_time
            )
        else:
            tours[start] = _additive_tours(n, local)

    # --- Step 2: assign deliveries to vehicles ---------------------------- #
    demands = scenario.demands
    capacities = scenario.capacities
    # Identical capacity is not enough to relabel vehicles freely any more: two
    # vehicles that start in different places are not the same vehicle, and
    # collapsing their permutations would drop every assignment where the nearer
    # one takes the nearer stop.
    interchangeable = len(set(capacities)) == 1 and len(set(starts)) == 1

    masks = [0] * k
    loads = [0.0] * k
    assigned = [0] * k  # deliveries per vehicle, for symmetry breaking
    best_total = infinity
    best_masks: list[int] | None = None

    def recurse(delivery: int) -> None:
        nonlocal best_total, best_masks
        if delivery == n:
            total = sum(tours[starts[v]][0][masks[v]] for v in range(k))
            if total < best_total:
                best_total = total
                best_masks = list(masks)
            return

        demand = demands[delivery]
        # Identical vehicles are interchangeable, so only ever open the next
        # unused one. This collapses the k! relabellings of every partition.
        limit = k
        if interchangeable:
            limit = min(k, sum(1 for count in assigned if count > 0) + 1)

        for vehicle in range(limit):
            if loads[vehicle] + demand > capacities[vehicle] + CAPACITY_EPSILON:
                continue
            loads[vehicle] += demand
            assigned[vehicle] += 1
            masks[vehicle] |= 1 << delivery
            recurse(delivery + 1)
            masks[vehicle] ^= 1 << delivery
            assigned[vehicle] -= 1
            loads[vehicle] -= demand

    recurse(0)

    if best_masks is None or not np.isfinite(best_total):
        raise ValueError(
            "no capacity-feasible assignment of deliveries to vehicles exists for "
            f"this scenario (demand {scenario.total_demand:g} across "
            f"{k} vehicle(s))"
        )

    return Solution(
        routes=tuple(
            tours[starts[vehicle]][1](mask) for vehicle, mask in enumerate(best_masks)
        )
    )


# --------------------------------------------------------------------------- #
# Step 1, in two regimes: additive cost, or cost-plus-a-clock
# --------------------------------------------------------------------------- #
def _local_arrays(
    cost_matrix: CostMatrix, n: int, start: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The objective and the clock as ``(n+1, n+1)`` start-first arrays.

    Position 0 is the node the vehicle *begins* at and ``i + 1`` is delivery
    ``i``. Working in this compact frame keeps the DP loops off the full matrix,
    whose extra rows only ever repeat the scenario's own nodes.

    ``start`` is that vehicle's own starting node, and only the **outbound** leg
    uses it: the return leg is the depot for every vehicle on every scenario. A
    re-planned vehicle leaves from where it is and still comes home, which is
    exactly the asymmetry ``CostMatrix`` documents — one index cannot describe a
    different start per vehicle, so the starts are a separate mapping and the
    depot row stays where it is. ``None`` means the depot, which is the ordinary
    framing and what a scenario naming no starts wants.

    Two arrays rather than one because the objective is what is minimised and
    the clock is what windows are checked against, and neither can be recovered
    from the other: a leg's rupee cost depends on its distance and implied speed
    as well as its duration, and a duration says nothing about when the vehicle
    set out.

    Both regimes below minimise the shared weighted objective, so "optimal" here
    means optimal for the same rupee cost every heuristic is scored on — which is
    what makes the gap between them meaningful.
    """
    index = cost_matrix.delivery_node_index
    matrix = cost_matrix.objective_matrix
    times = cost_matrix.matrix
    depot = cost_matrix.depot_index

    if start is None:
        start = depot

    local = np.zeros((n + 1, n + 1), dtype=float)
    local_time = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n):
        local[0, i + 1] = matrix[start, index[i]]
        local[i + 1, 0] = matrix[index[i], depot]
        local_time[0, i + 1] = times[start, index[i]]
        local_time[i + 1, 0] = times[index[i], depot]
        for j in range(n):
            local[i + 1, j + 1] = matrix[index[i], index[j]]
            local_time[i + 1, j + 1] = times[index[i], index[j]]
    return local, local_time


def _additive_tours(
    n: int, local: np.ndarray
) -> tuple[np.ndarray, "Callable[[int], tuple[int, ...]]"]:
    """Held-Karp over every subset, for an objective that is a sum over legs.

    ``best[mask, last]`` is the cheapest way to leave the vehicle's start, visit
    exactly ``mask`` and finish at delivery ``last``. One number per state
    suffices because with no windows the only thing that matters about a partial
    tour is what it has cost so far — how long it took is not part of the
    objective, and nothing downstream can depend on it.

    Returns the cost of the best start-to-depot tour for every subset, and a way
    to recover its visiting order.
    """
    subset_count = 1 << n
    infinity = np.inf
    best = np.full((subset_count, n), infinity, dtype=float)
    predecessor = np.full((subset_count, n), -1, dtype=np.int64)
    for i in range(n):
        best[1 << i, i] = local[0, i + 1]

    for mask in range(subset_count):
        for last in range(n):
            if not (mask >> last) & 1:
                continue
            current = best[mask, last]
            if current == infinity:
                continue
            row = local[last + 1]
            for following in range(n):
                if (mask >> following) & 1:
                    continue
                next_mask = mask | (1 << following)
                candidate = current + row[following + 1]
                if candidate < best[next_mask, following]:
                    best[next_mask, following] = candidate
                    predecessor[next_mask, following] = last

    # Close each subset's tour back to the depot.
    tour_cost = np.full(subset_count, infinity, dtype=float)
    tour_end = np.full(subset_count, -1, dtype=np.int64)
    tour_cost[0] = 0.0  # an empty route costs nothing
    for mask in range(1, subset_count):
        for last in range(n):
            if not (mask >> last) & 1:
                continue
            if best[mask, last] == infinity:
                continue
            total = best[mask, last] + local[last + 1, 0]
            if total < tour_cost[mask]:
                tour_cost[mask] = total
                tour_end[mask] = last

    def order_for(mask: int) -> tuple[int, ...]:
        """Walk the predecessor chain to recover visit order for ``mask``."""
        if mask == 0:
            return ()
        sequence: list[int] = []
        current, last = mask, int(tour_end[mask])
        while last != -1:
            sequence.append(last)
            previous = int(predecessor[current, last])
            current ^= 1 << last
            last = previous
        sequence.reverse()
        return tuple(sequence)

    return tour_cost, order_for


class _Label:
    """One partial route on a ``(subset, last)`` state's Pareto frontier.

    ``cost`` is everything the partial route has cost so far — weighted travel,
    **plus** any waiting, **plus** any lateness already incurred — and ``ready``
    is the clock time at which its last stop was served. ``parent`` is the label
    this one extends, which is how the visit order is recovered at the end.

    The parent is a reference and not an index into its state's list, because
    :func:`_pareto` reorders and drops labels after their children have been
    built; an index would silently come to mean a different label.
    """

    __slots__ = ("cost", "ready", "last", "parent")

    def __init__(
        self,
        cost: float,
        ready: float,
        last: int,
        parent: "_Label | None",
    ) -> None:
        self.cost = cost
        self.ready = ready
        self.last = last
        self.parent = parent


def _serve(
    cost: float,
    arrival: float,
    window: tuple[float | None, float | None],
    weights: CostWeights,
) -> tuple[float, float]:
    """Charge one stop's window against an arrival: ``(cost, ready time)``.

    The single place a wait or a lateness is turned into rupees, shared by the
    seeded first stop and by every later transition. It has to be shared: the
    opening stop of a route is the one arrival with no predecessor to charge it
    against, which makes it the easiest window in the whole DP to drop by
    accident — and dropping it is invisible on any instance whose first stop
    happens to be served on time.

    A wait moves the clock to the window's opening, so it is returned in
    ``ready`` rather than only priced: everything downstream is delayed by it.
    """
    earliest, latest = window
    ready = arrival
    wait = 0.0
    if earliest is not None and arrival < earliest:
        wait = earliest - arrival
        ready = earliest
    late = 0.0
    if latest is not None and ready > latest:
        late = ready - latest
    return (
        cost + wait * weights.time_per_second + late * weights.late_per_second,
        ready,
    )


def _pareto(labels: list[_Label]) -> list[_Label]:
    """Drop every label another beats on **both** cost and ready time.

    The objection worth answering is that arriving early is not obviously good
    when the vehicle then has to sit and wait, and waiting is charged. It is
    safe anyway, and the reason is that the clock is priced: a route's time
    component is exactly ``time_per_second * clock``, so a label's cost already
    contains the price of every second it has spent, waiting included.

    Take labels X and Y with X no dearer and no later than Y, and extend both by
    the same remaining stops. X's clock stays at or behind Y's throughout —
    service begins at ``max(arrival, earliest)``, which is monotone — so X is
    never late where Y is punctual, and the two cover the same roads, so their
    distance and fuel are identical. Whatever X spends, it spends by a clock no
    later than Y's, and time is priced per second, so X's total cannot be higher.
    Y can be dropped without losing the optimum.
    """
    kept: list[_Label] = []
    for label in labels:
        if any(
            other.cost <= label.cost and other.ready <= label.ready for other in kept
        ):
            continue
        kept = [
            other
            for other in kept
            if not (label.cost <= other.cost and label.ready <= other.ready)
        ]
        kept.append(label)
    return kept


def _windowed_tours(
    scenario: Scenario,
    cost_matrix: CostMatrix,
    n: int,
    local: np.ndarray,
    local_time: np.ndarray,
) -> tuple[np.ndarray, "Callable[[int], tuple[int, ...]]"]:
    """The same guarantee when deliveries carry service windows.

    The scalar state above is no longer enough, and the reason is worth stating
    because it is the whole difficulty of time windows. Arrival time depends on
    the entire prefix of a route, and a stop reached early is waited out — which
    pushes every arrival after it later. So "cheapest way to visit this subset"
    is not a complete description of a partial tour: two ways of visiting the
    same subset can differ in both what they cost and when they finish, and the
    cheaper one may be the one that misses the next window.

    So each ``(subset, last)`` carries a **Pareto set** of ``(cost, ready time)``
    labels instead of one number, and a label is dropped only when another beats
    it on both. Pruning on both axes is safe even though a vehicle that arrives
    early must wait and waiting is charged — :func:`_pareto` gives the argument.

    Still exact, and still exponential in the same way — the frontiers stay small
    because the objective is dominated by travel, so only a handful of genuinely
    different trade-offs survive per state.
    """
    weights = cost_matrix.weights
    windows = scenario.windows

    subset_count = 1 << n
    states: list[list[list[_Label]]] = [
        [[] for _ in range(n)] for _ in range(subset_count)
    ]
    for i in range(n):
        # The first stop's own window is charged here, through the same helper
        # every later transition uses. It is not a special case to skip: a route
        # that opens at a stop it cannot reach before that stop's window opens
        # waits just as surely as one that reaches it mid-route.
        cost, ready = _serve(
            float(local[0, i + 1]), float(local_time[0, i + 1]), windows[i], weights
        )
        states[1 << i][i] = [_Label(cost=cost, ready=ready, last=i, parent=None)]

    for mask in range(subset_count):
        for last in range(n):
            entries = states[mask][last]
            if not entries:
                continue
            for following in range(n):
                if (mask >> following) & 1:
                    continue
                leg_cost = float(local[last + 1, following + 1])
                if not np.isfinite(leg_cost):
                    continue
                leg_time = float(local_time[last + 1, following + 1])

                bucket = states[mask | (1 << following)][following]
                for entry in entries:
                    cost, ready = _serve(
                        entry.cost + leg_cost,
                        entry.ready + leg_time,
                        windows[following],
                        weights,
                    )
                    bucket.append(
                        _Label(cost=cost, ready=ready, last=following, parent=entry)
                    )
                states[mask | (1 << following)][following] = _pareto(bucket)

    # Close each subset back to the depot. The depot has no window and the round
    # ends there, so nothing downstream depends on the time any more and the
    # frontier collapses to its cheapest label — this is where the Pareto set
    # becomes the single number the partition search needs.
    tour_cost = np.full(subset_count, np.inf, dtype=float)
    tour_best: list[_Label | None] = [None] * subset_count
    tour_cost[0] = 0.0
    for mask in range(1, subset_count):
        for last in range(n):
            back = float(local[last + 1, 0])
            if not np.isfinite(back):
                continue
            for entry in states[mask][last]:
                total = entry.cost + back
                if total < tour_cost[mask]:
                    tour_cost[mask] = total
                    tour_best[mask] = entry

    def order_for(mask: int) -> tuple[int, ...]:
        """Walk the parent chain to recover visit order for ``mask``."""
        entry = tour_best[mask] if mask else None
        if entry is None:  # pragma: no cover - only an unreachable subset
            return ()
        sequence: list[int] = []
        while entry is not None:
            sequence.append(entry.last)
            entry = entry.parent
        sequence.reverse()
        return tuple(sequence)

    return tour_cost, order_for
