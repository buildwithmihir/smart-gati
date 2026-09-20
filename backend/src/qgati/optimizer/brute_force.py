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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qgati.optimizer.models import CAPACITY_EPSILON, Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

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

    if n > MAX_EXACT_DELIVERIES:
        raise ValueError(
            f"brute force is limited to {MAX_EXACT_DELIVERIES} deliveries, got {n}; "
            "use a heuristic for larger instances"
        )
    if n == 0:  # pragma: no cover - Scenario forbids this
        return Solution(routes=tuple(() for _ in range(k)))

    index = cost_matrix.delivery_node_index
    matrix = cost_matrix.matrix
    depot = cost_matrix.depot_index

    # Local (n+1)x(n+1) cost array: position 0 is the depot, i+1 is delivery i.
    # Working in this compact frame keeps the DP loops off the full matrix.
    local = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n):
        local[0, i + 1] = matrix[depot, index[i]]
        local[i + 1, 0] = matrix[index[i], depot]
        for j in range(n):
            local[i + 1, j + 1] = matrix[index[i], index[j]]

    # --- Step 1: Held-Karp over all subsets ------------------------------- #
    subset_count = 1 << n
    infinity = np.inf
    # best[mask, last] = cheapest way to leave the depot, visit exactly `mask`,
    # and finish at delivery `last`.
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

    # --- Step 2: assign deliveries to vehicles ---------------------------- #
    demands = scenario.demands
    capacities = scenario.capacities
    interchangeable = len(set(capacities)) == 1

    masks = [0] * k
    loads = [0.0] * k
    assigned = [0] * k  # deliveries per vehicle, for symmetry breaking
    best_total = infinity
    best_masks: list[int] | None = None

    def recurse(delivery: int) -> None:
        nonlocal best_total, best_masks
        if delivery == n:
            total = sum(tour_cost[mask] for mask in masks)
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

    return Solution(routes=tuple(order_for(mask) for mask in best_masks))
