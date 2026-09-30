"""The shared VRP data contract.

Every optimizer — brute force, Savings, GA, classical PSO and QPSO — consumes
these types and nothing else. They are deliberately free of any dependency on
the road graph or on numpy: a :class:`Scenario` describes *what* has to be
delivered, and the cost matrix describes *how expensive* the road network makes
it. Keeping the two apart is what lets a solver be tested in milliseconds on
hand-written costs and then run unchanged on real Delhi travel times.

Plain stdlib dataclasses rather than pydantic: these are internal contracts, and
pydantic v2 can validate and serialise stdlib dataclasses directly when they
need to cross the API boundary in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

__all__ = ["Delivery", "Depot", "Scenario", "Solution", "Vehicle"]

NodeId = Hashable

#: Slack allowed on capacity comparisons, so floating-point demand sums do not
#: make an exactly-full vehicle look overloaded.
CAPACITY_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class Depot:
    """Where every route starts and ends."""

    node: NodeId
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class Delivery:
    """One stop: a demand to be dropped at a road-graph node.

    ``demand`` is in whatever unit capacities are quoted in — kilograms, parcels,
    it does not matter as long as it is consistent.

    ``earliest_arrival`` and ``latest_arrival`` are an optional **service
    window**, in seconds from the moment the vehicle leaves the depot. Both are
    nullable, and a scenario where neither is set anywhere behaves exactly as it
    did before windows existed.

    Seconds from departure rather than a clock time, deliberately. This layer —
    like the cost matrix it consumes — is free of any notion of wall-clock time;
    absolute time belongs to the traffic layer, which prices the road network
    under a timestamp and then hands the optimizer an array of seconds. Putting a
    clock in here would mean the router and the optimizer disagreed about what
    "09:00" means once congestion was involved. An API that accepts ISO times is
    a conversion at the boundary, not a change of unit in the core.
    """

    id: str
    node: NodeId
    demand: float
    earliest_arrival: float | None = None
    latest_arrival: float | None = None

    def __post_init__(self) -> None:
        if self.demand < 0:
            raise ValueError(
                f"delivery {self.id!r} has negative demand {self.demand}"
            )
        for name in ("earliest_arrival", "latest_arrival"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(
                    f"delivery {self.id!r} has negative {name} {value}; windows "
                    "are measured forward from departure"
                )
        if (
            self.earliest_arrival is not None
            and self.latest_arrival is not None
            and self.earliest_arrival > self.latest_arrival
        ):
            raise ValueError(
                f"delivery {self.id!r} has an empty window: earliest "
                f"{self.earliest_arrival} is after latest {self.latest_arrival}"
            )

    @property
    def has_window(self) -> bool:
        """True when either end of the window is set."""
        return self.earliest_arrival is not None or self.latest_arrival is not None


@dataclass(frozen=True, slots=True)
class Vehicle:
    """A vehicle that may serve one route. ``capacity`` is a hard limit."""

    id: str
    capacity: float

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(
                f"vehicle {self.id!r} has non-positive capacity {self.capacity}"
            )


@dataclass(frozen=True, slots=True)
class Scenario:
    """A complete VRP instance: one depot, some deliveries, a fixed fleet.

    Raises on construction if the instance cannot possibly be served — total
    demand exceeding total fleet capacity, duplicate ids, and so on. Failing
    here rather than inside a solver means an infeasible instance can never
    masquerade as an optimizer that simply performed badly.

    ``starts`` is how a vehicle that is **already out** is described: one node
    per vehicle, in vehicle order, saying where that vehicle begins. Empty means
    what it always meant — every vehicle leaves the depot, which is the only
    thing a plan solved from scratch can say.

    It exists because a route's first leg is not a special case of its others.
    A vehicle re-optimized halfway through a shift leaves from wherever it is,
    and modelling that as "a depot that happens to be elsewhere" would need one
    depot per vehicle. See :meth:`has_custom_starts` for what a solver does
    differently, which is very little: the search space is untouched, only the
    cost of a candidate route changes.
    """

    depot: Depot
    deliveries: tuple[Delivery, ...] = field(default_factory=tuple)
    vehicles: tuple[Vehicle, ...] = field(default_factory=tuple)
    starts: tuple[NodeId, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Accept lists/tuples from callers but store tuples, so instances are
        # immutable and hashable.
        object.__setattr__(self, "deliveries", tuple(self.deliveries))
        object.__setattr__(self, "vehicles", tuple(self.vehicles))
        object.__setattr__(self, "starts", tuple(self.starts))

        if not self.deliveries:
            raise ValueError("scenario has no deliveries")
        if not self.vehicles:
            raise ValueError("scenario has no vehicles")
        if self.starts and len(self.starts) != len(self.vehicles):
            raise ValueError(
                f"{len(self.starts)} start node(s) for {len(self.vehicles)} "
                "vehicle(s); starts is one node per vehicle, in vehicle order"
            )

        duplicate_deliveries = _duplicates(d.id for d in self.deliveries)
        if duplicate_deliveries:
            raise ValueError(f"duplicate delivery ids: {duplicate_deliveries}")
        duplicate_vehicles = _duplicates(v.id for v in self.vehicles)
        if duplicate_vehicles:
            raise ValueError(f"duplicate vehicle ids: {duplicate_vehicles}")

        if self.total_demand > self.total_capacity + CAPACITY_EPSILON:
            raise ValueError(
                f"infeasible instance: total demand {self.total_demand} exceeds "
                f"total fleet capacity {self.total_capacity}"
            )

    # -- convenience ------------------------------------------------------- #
    @property
    def n_deliveries(self) -> int:
        return len(self.deliveries)

    @property
    def n_vehicles(self) -> int:
        return len(self.vehicles)

    @property
    def total_demand(self) -> float:
        return float(sum(d.demand for d in self.deliveries))

    @property
    def total_capacity(self) -> float:
        return float(sum(v.capacity for v in self.vehicles))

    @property
    def demands(self) -> tuple[float, ...]:
        """Demands in delivery-index order, for solvers that work on indices."""
        return tuple(float(d.demand) for d in self.deliveries)

    @property
    def capacities(self) -> tuple[float, ...]:
        """Capacities in vehicle-index order."""
        return tuple(float(v.capacity) for v in self.vehicles)

    @property
    def has_time_windows(self) -> bool:
        """True when at least one delivery constrains its arrival time.

        The solvers branch on this: with no windows the route cost is additive
        and the old, faster code paths are still exact, so an instance that does
        not use windows pays nothing for the feature existing.
        """
        return any(delivery.has_window for delivery in self.deliveries)

    @property
    def has_custom_starts(self) -> bool:
        """True when at least one vehicle begins somewhere other than the depot.

        The decoder branches on this rather than on the error it would otherwise
        make. Two of its decisions — which capacity bounds a route, and whether
        route order may be reshuffled across vehicles — are the same for every
        vehicle when they all leave the depot, and are not when they do not.

        It is deliberately about the *scenario* rather than the caller: the five
        solvers hand a scenario to a decoder and never mention starts, so the
        branch has to be visible from where the decoder sits.
        """
        return bool(self.starts)

    def start_node(self, vehicle_index: int) -> NodeId:
        """Where vehicle ``vehicle_index`` begins: its own start, or the depot."""
        if not self.starts:
            return self.depot.node
        return self.starts[vehicle_index]

    @property
    def windows(self) -> tuple[tuple[float | None, float | None], ...]:
        """Per-delivery ``(earliest, latest)`` in delivery-index order.

        Materialised once per evaluation rather than reached through
        ``scenario.deliveries`` in the inner loop, which is where arrival times
        are propagated.
        """
        return tuple(
            (delivery.earliest_arrival, delivery.latest_arrival)
            for delivery in self.deliveries
        )

    def delivery_ids(self) -> tuple[str, ...]:
        return tuple(d.id for d in self.deliveries)

    def summary(self) -> str:
        return (
            f"Scenario({self.n_deliveries} deliveries, {self.n_vehicles} vehicles, "
            f"demand {self.total_demand:g}/{self.total_capacity:g})"
        )


@dataclass(frozen=True, slots=True)
class Solution:
    """An assignment of deliveries to vehicles, and the order within each.

    ``routes`` always holds **exactly one entry per vehicle**, in scenario order;
    an unused vehicle gets an empty tuple. That fixed shape is deliberate — it
    makes "used more vehicles than are in the fleet" unrepresentable rather than
    something every solver has to remember to check.

    Each entry lists **delivery indices** (positions in ``scenario.deliveries``),
    not cost-matrix indices. Indices rather than nodes keep the type independent
    of the road graph; delivery positions rather than node ids keep two
    deliveries at the same node distinguishable.

    The depot is implicit: it is the end of every route and is never listed. So
    is the start, which is the depot unless the scenario names a different one
    per vehicle in :attr:`Scenario.starts` — a re-optimized route begins where
    its vehicle already is, and that node is a property of the vehicle rather
    than a stop on the route.
    """

    routes: tuple[tuple[int, ...], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "routes", tuple(tuple(route) for route in self.routes)
        )

    @property
    def n_vehicles(self) -> int:
        return len(self.routes)

    def served(self) -> tuple[int, ...]:
        """Every delivery index appearing in any route, in visit order."""
        return tuple(d for route in self.routes for d in route)

    def is_structurally_valid(self, scenario: Scenario) -> bool:
        """True when the shape matches the fleet and no delivery repeats.

        Does not check capacity — that needs demands and belongs to
        :func:`qgati.optimizer.fitness.evaluate`.
        """
        if len(self.routes) != scenario.n_vehicles:
            return False
        served = self.served()
        if len(served) != len(set(served)):
            return False
        return all(0 <= d < scenario.n_deliveries for d in served)

    def describe(self, scenario: Scenario) -> str:
        """Human-readable route listing, using delivery ids."""
        ids = scenario.delivery_ids()
        parts = [
            "depot -> " + " -> ".join(ids[d] for d in route) + " -> depot"
            if route
            else "(unused)"
            for route in self.routes
        ]
        return "; ".join(
            f"{vehicle.id}: {part}"
            for vehicle, part in zip(scenario.vehicles, parts, strict=True)
        )


def _duplicates(values) -> list:
    """Return the values that occur more than once, preserving first-seen order."""
    seen, repeats = set(), []
    for value in values:
        if value in seen and value not in repeats:
            repeats.append(value)
        seen.add(value)
    return repeats
