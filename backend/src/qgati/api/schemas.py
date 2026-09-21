"""Request and response models for the Q-Gati HTTP API.

Pydantic here, stdlib dataclasses in :mod:`qgati.optimizer.models`. That split is
deliberate and was planned from the start: the optimizer's contract stays free of
any web dependency, and this module is the boundary where it crosses into JSON.
The internal types are never sent raw — a ``Solution`` holds *delivery indices*,
which mean nothing to a client, so responses resolve them back into delivery ids
and road-graph nodes.

Node ids are ``int | str`` throughout. Real Delhi nodes are OSM ids (large
integers) and synthetic test graphs use small ones, but nothing guarantees an
integer, so the API does not assume it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "CompareResponse",
    "DeliveryOut",
    "DepotIn",
    "DepotOut",
    "OptimizeRequest",
    "OptimizeResponse",
    "RouteOut",
    "ScenarioCreateRequest",
    "ScenarioResponse",
    "ScenarioSummary",
    "SolverResultOut",
    "StopIn",
    "StopOut",
    "VehicleIn",
    "VehicleOut",
]

NodeId = Annotated[int | str, Field(description="Road-graph node id")]

#: Demand is in whatever unit capacities are quoted in — kilograms, parcels. It
#: only has to be consistent between deliveries and vehicles.
Demand = Annotated[float, Field(gt=0, description="Non-negative demand")]


# --------------------------------------------------------------------------- #
# Scenario input
# --------------------------------------------------------------------------- #
class DepotIn(BaseModel):
    """Where every route starts and ends.

    Give either ``node`` (a road-graph node id) or ``lat``/``lon`` to snap to the
    nearest one.
    """

    node: NodeId | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def _require_a_location(self) -> DepotIn:
        if self.node is None and (self.lat is None or self.lon is None):
            raise ValueError("give either 'node' or both 'lat' and 'lon'")
        if (self.lat is None) != (self.lon is None):
            raise ValueError("'lat' and 'lon' must be given together")
        return self


class StopIn(BaseModel):
    """One delivery stop. Same location rule as :class:`DepotIn`."""

    id: str = Field(min_length=1)
    demand: Demand
    node: NodeId | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def _require_a_location(self) -> StopIn:
        if self.node is None and (self.lat is None or self.lon is None):
            raise ValueError(
                f"stop {self.id!r}: give either 'node' or both 'lat' and 'lon'"
            )
        if (self.lat is None) != (self.lon is None):
            raise ValueError(f"stop {self.id!r}: 'lat' and 'lon' must be given together")
        return self


class VehicleIn(BaseModel):
    id: str = Field(min_length=1)
    capacity: float = Field(gt=0)


class ScenarioCreateRequest(BaseModel):
    """Create a scenario either by generating one or by describing it exactly.

    ``kind="generate"`` draws a random instance from the road graph's largest
    strongly-connected subgraph — the quickest way to get something to optimize,
    and what the demo and tests use. ``kind="explicit"`` is for a real instance.
    """

    kind: Literal["generate", "explicit"] = "generate"

    # kind="generate"
    n_deliveries: int | None = Field(default=None, ge=1, le=500)
    n_vehicles: int | None = Field(default=None, ge=1, le=100)
    seed: int = Field(default=0, description="Fixes the generated instance")

    # kind="explicit"
    depot: DepotIn | None = None
    deliveries: list[StopIn] | None = None
    vehicles: list[VehicleIn] | None = None

    @model_validator(mode="after")
    def _require_the_right_fields(self) -> ScenarioCreateRequest:
        if self.kind == "generate":
            if self.n_deliveries is None or self.n_vehicles is None:
                raise ValueError(
                    "kind='generate' needs 'n_deliveries' and 'n_vehicles'"
                )
        else:
            missing = [
                name
                for name, value in (
                    ("depot", self.depot),
                    ("deliveries", self.deliveries),
                    ("vehicles", self.vehicles),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"kind='explicit' needs {', '.join(repr(m) for m in missing)}"
                )
        return self


# --------------------------------------------------------------------------- #
# Scenario output
# --------------------------------------------------------------------------- #
class DepotOut(BaseModel):
    node: NodeId
    lat: float
    lon: float


class DeliveryOut(BaseModel):
    id: str
    node: NodeId
    demand: float


class VehicleOut(BaseModel):
    id: str
    capacity: float


class ScenarioResponse(BaseModel):
    """A stored scenario, echoing back what was created."""

    scenario_id: str
    depot: DepotOut
    deliveries: list[DeliveryOut]
    vehicles: list[VehicleOut]
    total_demand: float
    total_capacity: float
    n_deliveries: int
    n_vehicles: int
    #: True when the instance is small enough for brute force to solve exactly.
    exactly_solvable: bool


class ScenarioSummary(BaseModel):
    """Compact form for listings — enough to identify, not enough to optimize."""

    scenario_id: str
    n_deliveries: int
    n_vehicles: int
    total_demand: float


# --------------------------------------------------------------------------- #
# Optimize
# --------------------------------------------------------------------------- #
class OptimizeRequest(BaseModel):
    """Run one solver on a stored scenario.

    The scenario is named by the URL, not the body — ``POST /optimize/{scenario_id}``
    — so the body carries only *how* to solve, never *what* to solve.

    ``solver`` defaults to the registry's production default (ACO). The search
    parameters default to the same values the Phase 4 benchmark used, so an
    API result is comparable with a benchmark result.
    """

    solver: str | None = Field(
        default=None, description="Solver key; defaults to the production default"
    )
    seed: int | None = Field(default=None, description="Fixes a stochastic run")
    iterations: int | None = Field(default=None, ge=1, le=100_000)
    population: int | None = Field(default=None, ge=1, le=1000)
    include_geometry: bool = Field(
        default=True,
        description=(
            "Trace each route along the actual road network. Costs one shortest-"
            "path search per stop; disable for large instances when only the "
            "ordering matters."
        ),
    )


class StopOut(BaseModel):
    delivery_id: str
    node: NodeId


class RouteOut(BaseModel):
    """One vehicle's route, depot implicit at both ends."""

    vehicle_id: str
    stops: list[StopOut]
    load: float
    capacity: float
    travel_cost: float
    #: The route drawn on the real road network, as ``[[lon, lat], ...]`` — note
    #: GeoJSON axis order, longitude first. Empty when ``include_geometry`` is
    #: false, or when the graph carries no coordinates for these nodes.
    geometry: list[list[float]] = Field(default_factory=list)


class OptimizeResponse(BaseModel):
    """The solution, resolved back into ids a client can act on."""

    scenario_id: str
    solver: str
    solver_name: str
    seed: int | None
    iterations: int | None
    population: int | None
    #: Fitness actually minimised: travel cost plus any constraint penalties.
    cost: float
    travel_cost: float
    feasible: bool
    runtime_ms: float
    routes: list[RouteOut]
    #: Best-so-far cost per generation. Empty for the deterministic solvers.
    convergence: list[float] = Field(default_factory=list)

    @property
    def total_stops(self) -> int:
        return sum(len(route.stops) for route in self.routes)


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
class SolverResultOut(BaseModel):
    """One solver's outcome within a comparison.

    Every result field is ``None`` when :attr:`skipped` is set, so a solver that
    did not run is unambiguously *absent* rather than reported as a zero-cost
    infeasible answer.
    """

    solver: str
    solver_name: str
    cost: float | None = None
    travel_cost: float | None = None
    feasible: bool | None = None
    runtime_ms: float | None = None
    #: Percentage above the exact optimum; ``None`` when it is not known.
    gap_vs_optimal_pct: float | None = None
    #: Percentage above the best cost any solver in this comparison found.
    gap_vs_best_pct: float | None = None
    #: Why this solver did not run, if it did not (e.g. brute force on n > 10).
    skipped: str | None = None


class CompareResponse(BaseModel):
    """Every solver's result on one scenario's cost matrix.

    This is where QPSO's research value is visible next to the production
    default — the same six-solver comparison the Phase 4 benchmark presents,
    scoped to a single stored scenario.
    """

    scenario_id: str
    optimal: float | None = None
    best_known: float | None = None
    results: list[SolverResultOut]
