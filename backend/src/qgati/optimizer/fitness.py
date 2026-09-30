"""Shared solution evaluation — the one cost function every optimizer reuses.

Brute force, Savings, and the three metaheuristics all score candidate solutions
through :func:`evaluate`. Sharing it is what makes their results comparable: if
each solver had its own notion of "cost", a benchmark between them would be
measuring the accounting rather than the search.

The objective is the one recorded in ``DESIGN_DECISIONS.md``: a single
rupee-equivalent figure combining **time**, **distance** and **fuel**, with
constraint violations added as penalties large enough that an infeasible solution
can never outscore a feasible one.

The three goals are priced by :mod:`qgati.optimizer.objective` and married to the
road network in :class:`~qgati.graph.cost_matrix.CostMatrix`, which is where the
weighted sum is actually formed. This module does not re-derive it — it reads
``cost_matrix.objective_matrix`` and sums routes. That indirection is the point:
there is exactly one place in the codebase where a second, a metre and a litre
become a rupee, and every solver reaches it through the same array.

The raw components are carried alongside the total rather than discarded. A cost
nobody can decompose is a cost nobody can check, and "the optimizer saved 12%"
is a much weaker claim than "it saved 12% by trading 3 minutes for 800 metres".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qgati.optimizer.models import CAPACITY_EPSILON, Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = [
    "Evaluation",
    "PenaltyConfig",
    "RouteMetrics",
    "evaluate",
    "route_metrics",
    "route_total_cost",
    "route_travel_cost",
    "route_travel_time",
]

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
        (costs ~10) and real Delhi costs (rupees in the hundreds per leg) without
        the caller picking constants per scenario. It also keeps the penalties
        commensurate with the objective they must dominate: penalties scaled to
        travel *seconds* against a rupee objective would be no penalty at all.

        Callers pass :attr:`~qgati.graph.cost_matrix.CostMatrix.objective_matrix`.
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
class RouteMetrics:
    """One route's raw quantities and what they cost together.

    The physical measurements are kept separately from the price on purpose:
    ``cost`` depends on the weights the matrix was built with, while ``time``,
    ``distance`` and ``fuel`` are properties of the route itself and stay
    meaningful whichever policy priced them.

    ``time`` is **elapsed** time — driving plus any waiting at a stop whose
    window had not opened — because the vehicle and its driver are committed for
    the whole span, not only while the wheels turn, and the time price is quoted
    per hour of exactly that. ``driving_time`` is the part actually spent moving.
    With no windows the two are equal, so an instance that does not use windows
    sees precisely the numbers it saw before.
    """

    time: float
    """Seconds from leaving the depot to returning, waiting included."""

    driving_time: float
    """The part of ``time`` spent moving; ``time - waiting``."""

    distance: float
    """Metres driven."""

    fuel: float
    """Litres burned, estimated from distance and average speed."""

    cost: float
    """Rupees: ``cost = w1*time + w2*distance + w3*fuel``. Excludes lateness."""

    waiting: float
    """Seconds spent stationary because a window had not yet opened."""

    lateness: float
    """Seconds late summed over stops — see :attr:`window_penalty`."""

    window_penalty: float
    """Rupees owed for ``lateness``, at ``weights.late_per_hour``."""

    arrival_times: tuple[float, ...]
    """Elapsed seconds on reaching each stop, in visit order, after waiting."""

    @property
    def total_cost(self) -> float:
        """``cost`` plus the lateness it incurred — the whole per-route figure.

        What a solver should minimise for a single route: routing cost and
        window penalty are two ways of paying for the same route, and a solver
        that saw only one of them would trade punctuality for distance whenever
        the trade looked free.
        """
        return self.cost + self.window_penalty


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The full breakdown of a solution's score. Lower ``fitness`` is better.

    ``travel_cost`` is the weighted objective in **rupees**, not seconds — the
    problem statement's three goals combined. ``travel_time`` and the rest are
    the raw quantities behind it, reported so the total can be audited.
    ``window_penalty`` is kept out of ``travel_cost`` and added into ``fitness``
    instead: it is a constraint violation, and burying it inside the routing
    cost would make "how much of this is lateness?" unanswerable.
    """

    travel_cost: float
    travel_time: float
    distance: float
    fuel: float
    capacity_penalty: float
    coverage_penalty: float
    shape_penalty: float
    window_penalty: float
    waiting_time: float
    lateness: float
    route_costs: tuple[float, ...]
    route_loads: tuple[float, ...]
    route_times: tuple[float, ...]
    route_distances: tuple[float, ...]
    route_fuels: tuple[float, ...]
    route_window_penalties: tuple[float, ...]

    @property
    def penalty(self) -> float:
        return (
            self.capacity_penalty
            + self.coverage_penalty
            + self.shape_penalty
            + self.window_penalty
        )

    @property
    def fitness(self) -> float:
        """Objective the optimizers minimise: travel cost plus all penalties."""
        return self.travel_cost + self.penalty

    @property
    def feasible(self) -> bool:
        """True when no constraint is violated at all.

        A late arrival is a violation like any other, so a solution that misses a
        window is *not* feasible — but it is still scored rather than discarded,
        which is what lets one infeasible solution rank above another.
        """
        return self.penalty <= CAPACITY_EPSILON

    def summary(self) -> str:
        state = "feasible" if self.feasible else f"INFEASIBLE (penalty {self.penalty:g})"
        late = f" late={self.lateness:.0f}s" if self.lateness > 0.0 else ""
        return (
            f"cost=Rs{self.travel_cost:.2f} "
            f"[time={self.travel_time:.1f}s distance={self.distance:.0f}m "
            f"fuel={self.fuel:.3f}L{late}] {state}"
        )


def route_metrics(
    route: tuple[int, ...] | list[int],
    scenario: Scenario,
    cost_matrix: CostMatrix,
    vehicle: int = 0,
) -> RouteMetrics:
    """Time, distance, fuel, lateness and weighted cost for one route.

    An empty route costs nothing: the vehicle simply never leaves.

    ``vehicle`` says whose route this is, and it decides one thing: where the
    route *begins*. Every route ends at the depot, but a vehicle re-optimized
    mid-shift starts wherever it already is, and the scenario records that per
    vehicle rather than per scenario. Defaults to vehicle zero, whose start is
    the depot on any scenario that names no starts — so a caller that never
    thinks about this gets exactly the cost it got before the parameter
    existed.

    Legs are read straight out of the matrix rather than re-walked through the
    graph. The distance of a leg is the length of its fastest path, which is what
    the matrix builder recorded — re-deriving it here would risk the two
    disagreeing.

    The clock starts at zero when the vehicle leaves its start node, and is
    advanced by each leg's travel time. Two things then happen at each stop:

    * **Arriving before the window opens**, the vehicle waits. Waiting is what a
      driver actually does — the goods cannot be handed over early — so it is not
      penalised as a violation, but it is *charged*, because the vehicle and its
      driver are committed for that hour exactly as much as for an hour of
      driving. The clock moves to the window's opening, so the wait carries into
      every later arrival.
    * **Arriving after it closes**, the stop is late, and the lateness is priced
      at ``weights.late_per_hour``.

    Waiting is what makes a window bite even when nothing is late: a stop reached
    too early still pushes every downstream arrival later, which can be what
    makes the next one miss.

    This is a route whose cost is no longer a sum over its legs — a wait depends
    on the prefix that produced it — which is exactly why it cannot be folded
    into the cost matrix the way time, distance and fuel were.
    """
    if not route:
        return RouteMetrics(
            time=0.0,
            driving_time=0.0,
            distance=0.0,
            fuel=0.0,
            cost=0.0,
            waiting=0.0,
            lateness=0.0,
            window_penalty=0.0,
            arrival_times=(),
        )

    index = cost_matrix.delivery_node_index
    depot = cost_matrix.depot_index
    # Where this vehicle is now, which is the depot for every vehicle on a
    # scenario that names no starts. Named separately from `depot` because the
    # two are the same index in that case and different nodes in the other, and
    # a route that returned to its own start would never come home.
    origin = cost_matrix.start_index(vehicle)
    time_matrix = cost_matrix.matrix
    distance_matrix = cost_matrix.distance_matrix
    fuel_matrix = cost_matrix.fuel_matrix
    windows = scenario.windows
    weights = cost_matrix.weights

    clock = 0.0
    metres = 0.0
    litres = 0.0
    waiting = 0.0
    lateness = 0.0
    window_penalty = 0.0
    arrivals: list[float] = []

    legs = [(origin, index[route[0]])]
    legs += [
        (index[before], index[after]) for before, after in zip(route, route[1:])
    ]
    legs.append((index[route[-1]], depot))

    # Every leg but the last is followed by a stop that can have a window; the
    # return to the depot is not, which is why the window handling is driven off
    # the stop rather than off the leg.
    for position, (start, finish) in enumerate(legs):
        clock += float(time_matrix[start, finish])
        metres += float(distance_matrix[start, finish])
        litres += float(fuel_matrix[start, finish])

        if position == len(legs) - 1:
            break  # back at the depot; it has no window

        earliest, latest = windows[route[position]]
        if earliest is not None and clock < earliest:
            wait = earliest - clock
            waiting += wait
            clock = earliest
        arrivals.append(clock)
        if latest is not None and clock > latest:
            late = clock - latest
            lateness += late
            window_penalty += late * weights.late_per_second

    # Waiting advances the clock without moving the vehicle, so the clock is the
    # elapsed time and the driving time is what is left after removing it.
    driving_time = clock - waiting
    return RouteMetrics(
        time=clock,
        driving_time=driving_time,
        distance=metres,
        fuel=litres,
        cost=weights.cost(clock, metres, litres),
        waiting=waiting,
        lateness=lateness,
        window_penalty=window_penalty,
        arrival_times=tuple(arrivals),
    )


def route_travel_cost(
    route: tuple[int, ...] | list[int],
    scenario: Scenario,
    cost_matrix: CostMatrix,
    vehicle: int = 0,
) -> float:
    """Weighted cost of one route, in rupees.

    The objective, not the clock — the name is about cost and the unit is money.
    Use :func:`route_travel_time` when it is seconds you want.

    Excludes the lateness penalty; :func:`route_total_cost` is this plus it.
    """
    return route_metrics(route, scenario, cost_matrix, vehicle).cost


def route_total_cost(
    route: tuple[int, ...] | list[int],
    scenario: Scenario,
    cost_matrix: CostMatrix,
    vehicle: int = 0,
) -> float:
    """Everything one route costs: routing plus its lateness penalty.

    The figure a solver should minimise when choosing between orderings of the
    same stops. Route cost and window penalty are both prices for the same
    route, so comparing routes on either alone is comparing half a decision —
    and a solver that saw only the routing half would happily break a window to
    save a rupee.

    ``vehicle`` matters here more than it looks: this is what
    :func:`~qgati.optimizer.decoding.optimal_split` calls to price a candidate
    split when a window is present, and on a re-optimization the first leg's
    length depends on which vehicle would drive it.
    """
    return route_metrics(route, scenario, cost_matrix, vehicle).total_cost


def route_travel_time(
    route: tuple[int, ...] | list[int],
    scenario: Scenario,
    cost_matrix: CostMatrix,
    vehicle: int = 0,
) -> float:
    """:func:`route_travel_cost`'s other half: the same route in seconds."""
    return route_metrics(route, scenario, cost_matrix, vehicle).time


def evaluate(
    solution: Solution,
    scenario: Scenario,
    cost_matrix: CostMatrix,
    penalties: PenaltyConfig | None = None,
) -> Evaluation:
    """Score a candidate solution.

    Checks four things, and reports each separately so a caller can tell *why*
    a solution was rejected rather than only that it was:

    * **capacity** — no route's demand load exceeds its vehicle's capacity
    * **coverage** — every delivery served exactly once (none missed, none repeated)
    * **shape** — one route per vehicle, matching the fleet
    * **time windows** — no stop is reached after its ``latest_arrival``

    Penalties default to the objective's own scale, so they dominate any saving
    a solver could make by violating a constraint rather than by routing better.

    The window penalty is the exception to that scaling, and deliberately so: it
    is proportional to lateness rather than a flat amount, because "how late" is
    meaningful information that a fixed penalty would throw away. It is priced by
    ``weights.late_per_hour``, set high enough that a solver will lengthen a
    route to protect a window — see :data:`~qgati.optimizer.objective.DEFAULT_LATE_PER_HOUR`.

    Note that a window is checked against the arrival the route actually makes,
    waiting included. A stop reached early is waited out rather than penalised,
    but that wait moves every later arrival, so a window can fail a route without
    anything in it being late at the point it was reached.
    """
    if penalties is None:
        penalties = PenaltyConfig.for_matrix(cost_matrix.objective_matrix)

    demands = scenario.demands
    capacities = scenario.capacities
    n_deliveries = scenario.n_deliveries
    n_vehicles = scenario.n_vehicles

    route_costs: list[float] = []
    route_times: list[float] = []
    route_distances: list[float] = []
    route_fuels: list[float] = []
    route_window_penalties: list[float] = []
    route_loads: list[float] = []
    capacity_penalty = 0.0
    total_waiting = 0.0
    total_lateness = 0.0

    for position, route in enumerate(solution.routes):
        load = float(sum(demands[d] for d in route if 0 <= d < n_deliveries))
        route_loads.append(load)

        metrics = route_metrics(route, scenario, cost_matrix, position)
        route_costs.append(metrics.cost)
        route_times.append(metrics.time)
        route_distances.append(metrics.distance)
        route_fuels.append(metrics.fuel)
        route_window_penalties.append(metrics.window_penalty)
        total_waiting += metrics.waiting
        total_lateness += metrics.lateness

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

    # Waiting is not tracked per route on the Evaluation: it is already inside
    # `route_times`, and a breakdown that reported it twice would not add up.
    return Evaluation(
        travel_cost=float(sum(route_costs)),
        travel_time=float(sum(route_times)),
        distance=float(sum(route_distances)),
        fuel=float(sum(route_fuels)),
        capacity_penalty=float(capacity_penalty),
        coverage_penalty=float(coverage_penalty),
        shape_penalty=float(shape_penalty),
        window_penalty=float(sum(route_window_penalties)),
        waiting_time=float(total_waiting),
        lateness=float(total_lateness),
        route_costs=tuple(route_costs),
        route_loads=tuple(route_loads),
        route_times=tuple(route_times),
        route_distances=tuple(route_distances),
        route_fuels=tuple(route_fuels),
        route_window_penalties=tuple(route_window_penalties),
    )
