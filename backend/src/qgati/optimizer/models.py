"""The shared VRP data contract.

Every optimizer — brute force, Savings, and later QPSO/GA/PSO/ACO — consumes
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
    """

    id: str
    node: NodeId
    demand: float

    def __post_init__(self) -> None:
        if self.demand < 0:
            raise ValueError(
                f"delivery {self.id!r} has negative demand {self.demand}"
            )


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
    """

    depot: Depot
    deliveries: tuple[Delivery, ...] = field(default_factory=tuple)
    vehicles: tuple[Vehicle, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Accept lists/tuples from callers but store tuples, so instances are
        # immutable and hashable.
        object.__setattr__(self, "deliveries", tuple(self.deliveries))
        object.__setattr__(self, "vehicles", tuple(self.vehicles))

        if not self.deliveries:
            raise ValueError("scenario has no deliveries")
        if not self.vehicles:
            raise ValueError("scenario has no vehicles")

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

    The depot is implicit: it is the start and end of every route and is never
    listed.
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
