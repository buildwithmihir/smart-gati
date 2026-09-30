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

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from qgati.fleet.pings import DEFAULT_INTERVAL_SECONDS, DEFAULT_TIME_SCALE

__all__ = [
    "AnalyticsSummary",
    "AvoidRoadRequest",
    "CompareResponse",
    "ConditionsIn",
    "ConditionsOut",
    "DeliveryOut",
    "DepotIn",
    "DepotOut",
    "DetectionVerdict",
    "EdgeIn",
    "EdgeOut",
    "EtaSummary",
    "IncidentCreateRequest",
    "IncidentOut",
    "IncidentResponse",
    "MoveOut",
    "ObservationRequest",
    "OptimizeRequest",
    "OptimizeResponse",
    "ReoptimizeRequest",
    "ReoptimizeResponse",
    "RouteOut",
    "RunEntry",
    "RunPage",
    "ScenarioCreateRequest",
    "ScenarioResponse",
    "ScenarioSummary",
    "SolverResultOut",
    "StopIn",
    "StopOut",
    "TrafficLogEntry",
    "TrafficLogPage",
    "TriggerOut",
    "VehicleIn",
    "VehicleOut",
    "VehicleProgressOut",
    "VehicleTrackOut",
    "WatcherReadingOut",
    "WatcherStartRequest",
    "WatcherStartResponse",
    "WatcherStatus",
    "WatcherTick",
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


class EdgeIn(BaseModel):
    """One directed road segment, named by its endpoints.

    Direction matters: closing a one-way street is a realistic incident, so
    ``{u: 1, v: 2}`` and ``{u: 2, v: 1}`` are different roads.
    """

    u: NodeId
    v: NodeId


class ConditionsIn(BaseModel):
    """Simulated traffic conditions to price a scenario under.

    Omit the whole block for the plain case: the system clock decides the
    congestion state and nothing manual is applied.

    ``timestamp`` is *when* the scenario is priced. Leave it out for "now", or
    set it to compare the same instance at 09:00 and at 14:00. A naive timestamp
    is read as local time.

    Congestion itself is not settable. It follows from ``timestamp``, so a
    scenario always prices under a real state — there is no "moderate" flag to
    forget to send.
    """

    timestamp: datetime | None = Field(
        default=None,
        description="When to price; defaults to the system clock at creation.",
    )
    accident_edges: list[EdgeIn] = Field(
        default_factory=list,
        description=(
            "Roads carrying an accident, x2.9. Each must exist in the road "
            "graph. Applied flat, and in place of that road's congestion "
            "multiplier rather than compounded with it — x2.9 is already a "
            "worst-case placeholder for an unmeasured delay."
        ),
    )
    closed_edges: list[EdgeIn] = Field(
        default_factory=list,
        description=(
            "Impassable roads. Each must exist in the road graph. A closure can "
            "make a scenario infeasible, which is reported as 422."
        ),
    )


class IncidentCreateRequest(BaseModel):
    """Report one road as closed or slow, against a stored scenario.

    The scenario is named by the URL, not the body, for the same reason
    ``POST /optimize/{scenario_id}`` is: the body carries only *what* is being
    reported, never which instance it applies to.

    There is no ``timestamp``. An incident is injected now, and it changes the
    conditions a scenario is already priced under rather than re-timing it — the
    scenario's own timestamp is what its costs, and the resulting log row, are
    stamped with.
    """

    incident_type: Literal["closure", "slow"] = Field(
        description=(
            "'closure' makes the road impassable; 'slow' leaves it passable at "
            "2.9x its base travel time. Both are applied flat, replacing that "
            "road's congestion multiplier rather than compounding with it."
        )
    )
    edge: EdgeIn = Field(
        description=(
            "The directed road to apply it to. It must exist in the road graph. "
            "Direction matters: closing a one-way street is a realistic "
            "incident, so {u: 1, v: 2} and {u: 2, v: 1} are different roads."
        )
    )


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

    #: Traffic conditions to price the instance under. Fixed for the scenario's
    #: lifetime once created — see :class:`ConditionsOut`.
    conditions: ConditionsIn | None = None

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
    #: The traffic conditions this scenario's costs were built under.
    conditions: ConditionsOut
    #: Rows written to the traffic log when this scenario was priced. Reported so
    #: it is visible that collection happens as a side effect of normal use, with
    #: no separate step. Zero means the log write failed — the request still
    #: succeeded, and the server log says why.
    traffic_rows_logged: int


class ScenarioSummary(BaseModel):
    """Compact form for listings — enough to identify, not enough to optimize."""

    scenario_id: str
    n_deliveries: int
    n_vehicles: int
    total_demand: float


class EdgeOut(BaseModel):
    u: NodeId
    v: NodeId


class IncidentOut(BaseModel):
    """One live incident, as the scenario carrying it reports it.

    :attr:`created_at` is when the operator made the change, which is *not* the
    timestamp the scenario is priced under — that stays fixed for the scenario's
    lifetime. Keeping the wall-clock moment here means the audit trail can answer
    "when was this reported?" without the ``traffic_log`` row having to claim a
    congestion band the scenario was never priced under.
    """

    incident_id: str = Field(description="Pass this to DELETE to revert it")
    incident_type: Literal["closure", "slow"]
    edge: EdgeOut
    created_at: datetime


class ConditionsOut(BaseModel):
    """The conditions a stored scenario is priced under, with what they imply.

    ``peak_hour`` and ``traffic_condition`` are derived from ``timestamp`` alone
    — congestion needs no operator input, so a client does not have to
    re-implement the bands to know which regime it is looking at. The three
    ``traffic_condition`` values are ``"normal"``, ``"moderate"`` and ``"peak"``.

    ``accident_edges`` and ``closed_edges`` are the **resulting** effect sets,
    which is why they can be non-empty while ``incidents`` is empty: a scenario
    created with conditions carries them from birth, and only live reports are
    revertible. When ``mutated`` is true the two views overlap — an edge named by
    an incident also appears in whichever set that incident's effect puts it in —
    and ``incidents`` is what says which entries came from a report, and with
    which id to revert it.
    """

    timestamp: datetime
    peak_hour: bool
    traffic_condition: str
    accident_edges: list[EdgeOut]
    closed_edges: list[EdgeOut]
    #: True while at least one live incident is applied to this scenario.
    mutated: bool = False
    #: The live incidents currently applied, oldest first.
    incidents: list[IncidentOut] = Field(default_factory=list)


class IncidentResponse(BaseModel):
    """The outcome of applying or reverting one incident.

    Deliberately not a solution: injecting an incident rebuilds the cost matrix
    and stops. Re-optimizing is a separate, explicit call to
    ``POST /optimize/{scenario_id}``, which is what keeps an operator's report
    from silently moving vehicles.

    ``changed_legs`` is the evidence that it took effect — how many entries of
    the scenario's objective matrix moved. It can legitimately be **zero**: an
    incident on a road no cheapest path uses changes nothing about this
    instance, and reporting that is more useful than pretending otherwise.
    """

    scenario_id: str
    incident: IncidentOut
    #: True when the incident was applied, false when it was reverted.
    applied: bool
    #: The scenario's conditions *after* the change, mutation flag included.
    conditions: ConditionsOut
    #: Objective-matrix entries whose cost moved. Zero is a valid answer.
    changed_legs: int
    #: Audit rows written to the traffic log by this call. Zero means the write
    #: failed — the change still took effect, and the server log says why.
    traffic_rows_logged: int


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #
class ObservationRequest(BaseModel):
    """One newly observed travel time for one road, to be judged.

    The scenario is named by the URL, like every other route that acts on a
    stored scenario: the body carries only *what was seen*, never which instance
    it is being judged against. The baseline it is compared with is that
    scenario's, taken when it was created.
    """

    edge: EdgeIn = Field(
        description=(
            "The directed road the reading is for. It must exist in the road "
            "graph. Direction matters, as everywhere else: a queue on one "
            "carriageway says nothing about the other."
        )
    )
    travel_time: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="The observed travel time for that road, in seconds",
    )
    timestamp: datetime | None = Field(
        default=None,
        description=(
            "When the reading was taken; defaults to the scenario's own "
            "timestamp. This is what decides which congestion band's history the "
            "reading is judged against, so a 09:00 observation is compared with "
            "that road's 09:00 history rather than with a pooled figure."
        ),
    )


class DetectionVerdict(BaseModel):
    """Whether one observed travel time is anomalous, and what decided it.

    Not a solution. Detecting an anomaly flags it for a later phase to act on;
    nothing here re-routes, and re-optimizing on a flag is the ``reopt`` module's
    job. The structural evidence is that this model has no field a route, a cost
    or a solver could be returned in.

    ``mean``, ``std_dev`` and ``z_score`` are ``None`` when the fallback rule
    decided, because no z-score was taken. ``sample_count`` is always reported,
    so a client can see how much history was behind the answer rather than having
    to ask whether the verdict was well founded.

    ``z_score`` is also ``None`` when the score is not finite — a history with no
    spread at all, which is the ordinary state of this log — because JSON cannot
    carry an infinity. ``std_dev`` of ``0`` is what distinguishes that case from
    the fallback, and ``reason`` spells it out in words.
    """

    scenario_id: str
    edge: EdgeOut
    observed_travel_time: float
    #: The congestion band the reading was judged in.
    condition: str
    #: ``"z_score"``, or ``"insufficient_samples"`` when there was too little
    #: history for a standard deviation and the rule-based margin decided instead.
    rule: Literal["z_score", "insufficient_samples"]
    flagged: bool
    #: The road's modelled cost under the clock alone, with manual conditions
    #: cleared. ``None`` when the road is impassable under those conditions.
    expected_travel_time: float | None = None
    sample_count: int
    mean: float | None = None
    std_dev: float | None = None
    z_score: float | None = None
    #: Whichever threshold decided this verdict — 2 standard deviations on the
    #: z-score path, 1.2x the expectation on the fallback path.
    threshold: float
    #: One sentence naming the number that decided it.
    reason: str


# --------------------------------------------------------------------------- #
# Optimize
# --------------------------------------------------------------------------- #
class OptimizeRequest(BaseModel):
    """Run one solver on a stored scenario.

    The scenario is named by the URL, not the body — ``POST /optimize/{scenario_id}``
    — so the body carries only *how* to solve, never *what* to solve.

    ``solver`` defaults to the registry's production default (QPSO). The search
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
    """One vehicle's route, depot implicit at both ends.

    The four cost fields are the objective and its parts. ``travel_cost`` is the
    weighted figure the optimizer minimised, in **rupees** — not seconds, which
    is what it used to be before distance and fuel joined the objective. The
    other three are the raw quantities it was computed from, reported so the
    number can be checked rather than taken on trust.
    """

    vehicle_id: str
    stops: list[StopOut]
    load: float
    capacity: float
    travel_cost: float = Field(description="Weighted objective for this route, in rupees")
    travel_time: float = Field(description="Seconds on the road, depot to depot")
    distance_m: float = Field(description="Metres driven")
    fuel_litres: float = Field(description="Litres burned, estimated from distance and speed")
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
    #: Fitness actually minimised: the weighted travel cost plus any constraint
    #: penalties. Equals ``travel_cost`` whenever the solution is feasible.
    cost: float
    #: The weighted objective in **rupees**: time, distance and fuel combined
    #: under the prices in ``qgati.optimizer.objective``. Not seconds.
    travel_cost: float
    travel_time: float = Field(description="Total seconds on the road")
    distance_m: float = Field(description="Total metres driven")
    fuel_litres: float = Field(description="Total litres burned")
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

    ``travel_cost`` is the weighted objective in rupees; ``travel_time``,
    ``distance_m`` and ``fuel_litres`` are the raw quantities behind it. Carrying
    all four is what lets the table show a solver that bought seconds with extra
    kilometres, rather than only that it scored better.
    """

    solver: str
    solver_name: str
    cost: float | None = None
    travel_cost: float | None = None
    travel_time: float | None = None
    distance_m: float | None = None
    fuel_litres: float | None = None
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
    default — the same five-solver comparison the Phase 4 benchmark presents,
    scoped to a single stored scenario.
    """

    scenario_id: str
    optimal: float | None = None
    best_known: float | None = None
    results: list[SolverResultOut]


# --------------------------------------------------------------------------- #
# Traffic log
# --------------------------------------------------------------------------- #
class TrafficLogEntry(BaseModel):
    """One logged road-condition observation.

    ``travel_time`` is ``None`` for a closed road — it is impassable, so there is
    no travel time to report, and ``incident_type`` says why.

    This table is being collected for a future travel-time model. Nothing reads
    it back except this endpoint; there is no prediction here.
    """

    road_id: str = Field(description='The edge, as "u->v"')
    road_u: NodeId
    road_v: NodeId
    timestamp: str
    day_of_week: str
    time_of_day: str
    traffic_condition: str
    incident_type: str | None = None
    travel_time: float | None = None


class TrafficLogPage(BaseModel):
    """One page of the traffic log, newest first."""

    items: list[TrafficLogEntry]
    #: Rows matching the filters across the whole table, not just this page.
    total: int
    limit: int
    offset: int
    has_more: bool


# --------------------------------------------------------------------------- #
# Run history
# --------------------------------------------------------------------------- #
class RunEntry(BaseModel):
    """One recorded solve.

    Every figure is the one the solve's own response reported, captured at the
    point it was computed rather than recomputed here — the runtime is the same
    measured span, and the cost and totals come from the same evaluation.

    ``old_eta_seconds`` is present only for a re-plan, and only a re-plan *has* a
    baseline: an ``optimize`` or a ``dispatch`` is the first answer for its
    scenario, so there is nothing for it to have been measured against. An
    ``avoid_road`` re-plan is included — it is a re-plan, scoped to one vehicle.

    The ETA pair is the **remaining unserved work**, in seconds: what it would have
    taken on the routes the fleet was already driving, against what the re-planned
    routes take. ``eta_saved_seconds`` is positive when the re-plan came out ahead,
    and is **negative when it did not** — a re-solve on a small instance can land on
    a worse arrangement than the one in hand, and reporting that as a zero would
    hide the one case an operator most wants to see.

    **``cost``, ``travel_cost``, ``travel_time``, ``distance_m`` and
    ``fuel_litres`` cover different work from one ``kind`` to the next**, and
    comparing them across kinds is the one way this table can mislead. For
    ``optimize`` and ``dispatch`` they are the whole scenario. For ``reoptimize``
    they are the re-solved remainder — the unserved stops, from where each vehicle
    actually is — which is necessarily less. For ``avoid_road`` they are the
    *fleet's* unserved work after the reporting vehicle's route was re-solved: the
    same total that response's ``after`` array sums to, and deliberately **not** the
    figure at the top of it, which covers the reporting vehicle alone (see
    :func:`~qgati.api.main._record_run`). ``n_deliveries`` is always the
    *scenario's* size, so it says which instance the run was against, never how much
    work the run covered.
    """

    id: int | None = None
    timestamp: str = Field(description="ISO-8601 UTC, seconds precision")
    kind: Literal["dispatch", "optimize", "reoptimize", "avoid_road"] = Field(
        description=(
            "Which solve ran. `dispatch` is `POST /scenarios/{id}/watcher`, which "
            "solves the scenario on its way to putting a fleet on the road."
        )
    )
    scenario_id: str
    n_deliveries: int
    n_vehicles: int
    solver: str
    solver_name: str
    seed: int | None = None
    iterations: int | None = None
    population: int | None = None
    trigger: str | None = Field(
        default=None,
        description="Why a re-plan was allowed: 'incident', 'anomaly' or 'override'.",
    )
    trigger_detail: str | None = Field(
        default=None, description="The trigger's own sentence, as shown to the user"
    )
    affected_vehicle: str | None = Field(
        default=None, description="The vehicle that reported, for an 'avoid_road' run"
    )
    cost: float = Field(description="The fitness actually minimised, penalties included")
    travel_cost: float = Field(description="The weighted objective in rupees")
    travel_time: float = Field(description="Total seconds on the road")
    distance_m: float
    fuel_litres: float
    feasible: bool
    runtime_ms: float
    old_eta_seconds: float | None = None
    new_eta_seconds: float | None = None
    eta_saved_seconds: float | None = None
    moved: int = Field(
        default=0, description="Deliveries that changed hands in this re-plan"
    )


class RunPage(BaseModel):
    """One page of the run history, newest first.

    The same shape as :class:`TrafficLogPage`, deliberately: one pagination
    contract for the whole API rather than two that a client has to tell apart.
    """

    items: list[RunEntry]
    #: Rows matching the filters across the whole table, not just this page.
    total: int
    limit: int
    offset: int
    has_more: bool


class EtaSummary(BaseModel):
    """What re-planning has saved, over the re-plans that had a baseline.

    Every figure is ``None`` or ``0`` on an empty history rather than a plausible
    zero: ``avg_saved_seconds`` is ``None`` when nothing has been re-planned, and
    ``runs`` says how many rows it was computed over so a reader can tell "no
    saving" from "no data".
    """

    runs: int = Field(description="Re-plans that had a before and an after")
    avg_saved_seconds: float | None = None
    total_saved_seconds: float | None = None
    improved_runs: int = 0
    worsened_runs: int = 0
    unchanged_runs: int = 0


class AnalyticsSummary(BaseModel):
    """Aggregate statistics over the whole run history.

    The averages are optional because they are genuinely unknown on an empty
    table — SQL's ``AVG`` over no rows is ``NULL`` — and the API passes that through
    rather than substituting ``0.0``. The same principle the dashboard applies to
    ``feasible`` on a dispatched plan, which is ``null`` rather than guessed at.
    """

    total_runs: int
    first_run_at: str | None = None
    last_run_at: str | None = None
    runs_by_kind: dict[str, int] = Field(default_factory=dict)
    runs_by_solver: dict[str, int] = Field(default_factory=dict)
    feasible_runs: int = 0
    infeasible_runs: int = 0
    incident_triggered_runs: int = Field(
        default=0,
        description="Runs a live incident or a flagged reading caused, of any kind",
    )
    avg_runtime_ms: float | None = None
    min_runtime_ms: float | None = None
    max_runtime_ms: float | None = None
    avg_cost: float | None = None
    avg_travel_cost: float | None = None
    avg_travel_time_seconds: float | None = None
    eta: EtaSummary


# --------------------------------------------------------------------------- #
# Simulated fleet watcher
# --------------------------------------------------------------------------- #
class WatcherStartRequest(BaseModel):
    """Start a simulated GPS fleet against a stored scenario.

    The scenario is named by the URL, like every other route that acts on one,
    and the body carries only *how* to run the fleet.

    Starting a watcher **solves the scenario**: a fleet needs routes to drive and
    a stored scenario has none until something solves it. The solver is chosen
    here — the production default unless named — and the routes it returns are
    reported in the response, so the caller sees what the fleet was dispatched
    on.
    """

    solver: str | None = Field(
        default=None, description="Solver key; defaults to the production default"
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Fixes both the solve and the fleet's noise, so one request produces "
            "the same routes and the same readings twice."
        ),
    )
    interval_seconds: float = Field(
        default=DEFAULT_INTERVAL_SECONDS,
        ge=1,
        le=300,
        description=(
            "Seconds of wall clock between ticks. The default is the middle of "
            "the 15-20 s a real fleet would ping at; a demo can drop it to 1."
        ),
    )
    time_scale: float = Field(
        default=DEFAULT_TIME_SCALE,
        gt=0,
        le=600,
        description=(
            "How many seconds of driving each tick represents. 1.0 is real time; "
            "raise it to move vehicles along their routes faster than the clock."
        ),
    )
    population: int | None = Field(default=None, ge=1, le=1000)
    iterations: int | None = Field(default=None, ge=1, le=100_000)
    include_geometry: bool = Field(
        default=True,
        description=(
            "Trace each dispatched route along the road network, so the map can "
            "draw what the fleet was sent down. Costs one shortest-path search "
            "per stop; disable when only the ordering matters."
        ),
    )


class WatcherReadingOut(BaseModel):
    """One vehicle's ping, and what the detector made of it.

    ``edge`` is ``None`` when the vehicle had nothing to report — its route is
    finished, or the road ahead is impassable — and ``note`` says which. That is
    a legitimate state rather than an error, so it is reported in the same shape
    as a real reading rather than as a failure.

    ``modelled_travel_time`` is the road's time under the scenario's *own*
    conditions, incidents included: what the vehicle is driving through. The
    measurement is drawn around it, so the two sit together and their ratio is
    the size of the deviation.
    """

    vehicle_id: str
    edge: EdgeOut | None = None
    modelled_travel_time: float | None = None
    observed_travel_time: float | None = None
    flag: bool = False
    rule: Literal["z_score", "insufficient_samples"] | None = None
    condition: str | None = None
    sample_count: int = 0
    z_score: float | None = None
    reason: str | None = None
    note: str | None = None


class VehicleTrackOut(BaseModel):
    """Where one vehicle is, as of the last tick."""

    vehicle_id: str
    edge: EdgeOut | None = None
    remaining_seconds: float | None = None
    progress: float = Field(
        description="Fraction of the route's cost consumed, under current costs"
    )
    finished: bool = False


class WatcherStatus(BaseModel):
    """A running fleet's state. Returned by ``GET`` and ``DELETE``."""

    scenario_id: str
    solver: str
    running: bool
    interval_seconds: float
    time_scale: float
    seed: int | None = None
    ticks: int
    started_at: str
    last_tick_at: str | None = None
    vehicles: list[VehicleTrackOut] = Field(default_factory=list)


class WatcherTick(BaseModel):
    """What one tick did: the readings, and the two counts that prove it landed.

    Not a solution. A tick measures roads and re-prices the scenario; whether
    anything should be *re-routed* on what it found is the ``reopt`` module's
    job, and re-optimizing is left to the client. The structural evidence, as
    with :class:`DetectionVerdict`, is that this model has no field a route, a
    cost or a solver could be returned in.

    ``changed_legs`` is the same number an incident reports: entries of the
    objective matrix that moved. A tick on roads nobody is driving changes
    nothing, and one on a road under an incident moves it by the difference
    between the placeholder and the measurement.
    """

    scenario_id: str
    tick: int
    at: str
    readings: list[WatcherReadingOut] = Field(default_factory=list)
    changed_legs: int = 0
    rows_logged: int = 0
    #: True while a live incident prices the scenario — the tick's readings then
    #: carry the effect word, and are excluded from a later baseline.
    conditions_mutated: bool = False


class WatcherStartResponse(WatcherStatus):
    """The fleet that was just started: its routes, and its first tick.

    Carries the routes because starting a watcher is what solves the scenario,
    and they are otherwise only reachable through a second ``/optimize`` call
    that would solve the same instance again. The first tick is run inline for
    the same reason — a caller should not have to wait an interval to see that
    anything is happening.
    """

    routes: list[RouteOut] = Field(default_factory=list)
    first_tick: WatcherTick
    #: The solve that produced :attr:`routes`, best-so-far cost per iteration.
    #:
    #: Carried here because starting a watcher *is* how the dashboard gets its
    #: first plan — the routes above are a solve's output, and the history of how
    #: that solve converged is the same kind of fact as the routes themselves.
    #: It used to be discarded, which left the convergence chart empty until the
    #: first re-plan; nothing else about the solve was withheld.
    convergence: list[float] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Re-optimization
# --------------------------------------------------------------------------- #
class ReoptimizeRequest(BaseModel):
    """Re-plan the unserved deliveries of a scenario whose fleet is already out.

    The scenario is named by the URL, like every other route that acts on one.
    Note what is **not** here: anywhere to declare where the vehicles are. The
    fleet's positions, its completed stops and its remaining capacity are read
    from the running watcher, so a caller cannot assert a fleet state the fleet
    is not in.

    Nor is there a ``force`` or ``trigger`` field. Whether a re-optimization is
    justified is derived from the network and the fleet's own readings — an
    incident, or a flagged measurement — and a request that arrives when neither
    holds is refused with a 409. That is deliberate: without it, "adaptive
    routing" would be an endpoint indistinguishable from a re-solve button.

    ``solver`` defaults to the registry's production default (QPSO), and the
    search parameters to the same values the Phase 4 benchmark used, so a
    re-planned result is comparable with a benchmark result and with an
    ``/optimize`` result.
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
            "Trace each route along the road network, from where the vehicle "
            "actually is rather than from the depot. Costs one shortest-path "
            "search per stop; disable for large instances."
        ),
    )


class AvoidRoadRequest(BaseModel):
    """A driver's own report, about a road they are on or heading for.

    The manual override alongside the two derived triggers. An incident's flat
    ``x2.9`` is a placeholder — a *guess* at how much worse a slow road is — and a
    driver sitting in the real thing knows better than the model does. This is
    where they say so, and the only way to say it without waiting for the fleet to
    drive the road, be measured, and have the detector agree the trip was unusual.

    Both the scenario and the vehicle are named by the URL: the body carries only
    what is being reported and how it should be re-planned. Where the vehicle is,
    what it has already delivered and how much room it has left are read from the
    running fleet, exactly as ``POST /reoptimize`` reads them — there is nowhere
    here to assert a fleet state the fleet is not in.
    """

    edge: EdgeIn = Field(
        description=(
            "The directed road to avoid. It must exist in the road graph, and it "
            "may be the road the vehicle is currently on — that is the case this "
            "route exists for, and one an operator's incident cannot express."
        )
    )
    treatment: Literal["slow", "closure"] = Field(
        default="slow",
        description=(
            "The same two words ``POST /scenarios/{id}/incident`` takes. 'slow' "
            "leaves the road passable at 2.9x its base travel time and the vehicle "
            "is re-planned from the far end of it; 'closure' makes it impassable "
            "and the vehicle is re-planned from the near end, because it cannot be "
            "sent through a road that is shut. Either way the report is filed as a "
            "real incident — priced, logged, and revertable."
        ),
    )

    solver: str | None = Field(
        default=None, description="Solver key; defaults to the production default"
    )
    seed: int | None = Field(default=None, description="Fixes a stochastic run")
    iterations: int | None = Field(default=None, ge=1, le=100_000)
    population: int | None = Field(default=None, ge=1, le=1000)
    include_geometry: bool = Field(
        default=True,
        description=(
            "Trace each route along the road network, from where the vehicle "
            "actually is rather than from the depot. Costs one shortest-path "
            "search per stop; disable for large instances."
        ),
    )


class TriggerOut(BaseModel):
    """Why this re-optimization was allowed to run, and on what evidence.

    ``kinds`` holds one or more of ``incident``, ``anomaly`` and ``override``,
    most authoritative first, and they are different strengths of evidence that a
    reader deciding whether to trust the result should be able to tell apart. An
    **incident** is a report — somebody said a road is closed or slow, and the
    scenario is priced as though it is; it needs no corroboration. An **anomaly**
    is a measurement — the fleet drove a road and the detector found the trip
    unlike that road's history.

    They are commonly both set: an incident is injected, a vehicle drives it, and
    the detector independently flags the trip. ``reasons`` carries the detector's
    own explanations verbatim, because they name the numbers it judged and a
    summary would throw that away.

    The third kind, ``override``, is only ever reported on its own. It is a
    **driver** saying their own road is worse than the model thinks — the only
    trigger that arrives with the vehicle it is about, and the only one a caller
    supplies rather than derives. It is filed as a real incident first, so it does
    not bypass the rule above; listing that incident here as well would report one
    fact twice. ``primary`` is what the re-plan is chiefly reacting to.
    """

    kinds: list[Literal["incident", "anomaly", "override"]] = Field(default_factory=list)
    primary: Literal["incident", "anomaly", "override"]
    detail: str
    edges: list[EdgeOut] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class VehicleProgressOut(BaseModel):
    """One vehicle as the re-optimization found it.

    ``completed`` and ``remaining`` partition the route the vehicle was
    dispatched on, and between them they are the whole reason this model exists:
    the first is what a re-optimization may never touch, and the second is
    everything it is allowed to move.

    Both are delivery **ids**, not indices, so they can be matched against the
    routes. ``node`` is where the vehicle would begin a new plan — the next
    intersection it reaches, since a cost matrix is indexed by intersections and
    a vehicle halfway along a road is not at one. It is ``None`` for a vehicle
    that is not on a road: a finished one, or one stopped behind a closure, and
    ``finished`` and ``stuck`` say which.

    ``edge`` is the road all of that is about, and it is the field that explains
    ``node``. A vehicle on road ``(u, v)`` starts from ``v``; one stopped behind
    that road — or told to avoid it — starts from ``u`` instead, which is the last
    intersection it can be said to have reached.
    """

    vehicle_id: str
    node: NodeId | None = None
    edge: EdgeOut | None = Field(
        default=None,
        description=(
            "The directed road it is on, or the road ahead it cannot drive when "
            "it is stopped. None when it is on no road at all."
        ),
    )
    #: Delivery ids already served. These appear in no ``after`` route, at any
    #: search budget, for any solver — they are not in the instance being solved.
    completed: list[str] = Field(default_factory=list)
    remaining: list[str] = Field(default_factory=list)
    elapsed_seconds: float = Field(
        description="Seconds since this vehicle left the depot"
    )
    remaining_capacity: float = Field(
        description="Capacity less the demand it has already dropped"
    )
    stuck: bool = False
    finished: bool = False
    available: bool = Field(
        description="Something left to serve, and moving — so it is in the new fleet"
    )


class MoveOut(BaseModel):
    """One delivery that changed hands.

    ``to_vehicle`` is ``None`` when the new plan does not serve the delivery at
    all, which is a legitimate answer rather than a failure — an infeasible plan,
    and one ``feasible`` reports on the response.

    For a **driver-scoped** re-plan this list is empty, by construction rather
    than by omission: exactly one vehicle is in the instance that was solved, so
    no delivery can change hands however differently the new route orders them.
    What changed there is the *order*, which is what ``before`` and ``after``
    show.
    """

    delivery_id: str
    from_vehicle: str
    to_vehicle: str | None = None


class FigureOut(BaseModel):
    """One number a statement was built from, and what it is.

    ``unit`` is a bare word rather than a formatted string, so a client renders
    it and a reader can check it. Every figure here appears in the ``text`` of
    the statement it belongs to — that is what the pairing is for.
    """

    label: str
    value: float
    unit: str


class StatementOut(BaseModel):
    """One sentence of a decision trace, and its receipts."""

    text: str
    figures: list[FigureOut] = Field(default_factory=list)


class RouteExplanationOut(BaseModel):
    """Why one vehicle's route looks the way it does after a re-plan."""

    vehicle_id: str
    headline: str
    statements: list[StatementOut] = Field(default_factory=list)


class RoadReportOut(BaseModel):
    """What one reported road did to the network, before the report and after."""

    u: str | int
    v: str | int
    statements: list[StatementOut] = Field(default_factory=list)


class ExplanationOut(BaseModel):
    """The whole decision trace for a re-optimization.

    ``headline`` is the trigger's own sentence, verbatim — the same string as
    ``trigger.detail``. ``roads`` is plan-level because a road's delay is a fact
    about the road and not about any one vehicle; ``routes`` is the per-vehicle
    half, and holds only vehicles that appear in both the before and after
    fleets, since nothing can be said about a vehicle that is in one and not the
    other.
    """

    headline: str
    roads: list[RoadReportOut] = Field(default_factory=list)
    routes: list[RouteExplanationOut] = Field(default_factory=list)


class ReoptimizeResponse(BaseModel):
    """A re-optimized plan, and the before/after that shows what changed.

    ``before`` and ``after`` cover **the same deliveries**: the ones still
    unserved when this ran. ``before`` is the rest of the plan the fleet was
    already driving, in its original order and priced under current conditions;
    ``after`` is the new assignment of exactly those stops. Comparing them is
    therefore a comparison of two ways to serve one set of work, not of two
    different amounts of work — which is what makes the reassignment meaningful.

    Completed deliveries are not in either list. They are reported once, on
    ``vehicles``, and there is deliberately no field anywhere in this model that
    could carry a completed stop in an ``after`` route: the guarantee is that
    they were never in the instance that was solved, not that something removed
    them afterwards.

    **Two endpoints return this model, and they differ in scope.**
    ``POST /reoptimize`` solves for every vehicle that can still work, so its
    ``after`` holds only those vehicles' routes. ``POST .../vehicles/{id}/avoid-road``
    solves for the one vehicle that reported a road, and its ``after`` holds the
    whole fleet — that vehicle's new route, and every other vehicle's exactly as
    it already was. ``replanned_vehicles`` is the field that says which vehicles
    were actually *solved*, and therefore which routes a reader should expect to
    have moved. The two `before`/`after` lists hold one route per vehicle either
    way, in vehicle order, so they can be compared entry by entry.
    """

    scenario_id: str
    solver: str
    solver_name: str
    seed: int | None = None
    iterations: int | None = None
    population: int | None = None

    trigger: TriggerOut
    vehicles: list[VehicleProgressOut] = Field(default_factory=list)
    replanned_vehicles: list[str] = Field(
        default_factory=list,
        description=(
            "The vehicles that were solved for. Every other vehicle was not in "
            "the instance, so its route is unchanged in `after`."
        ),
    )
    incident: IncidentOut | None = Field(
        default=None,
        description=(
            "The report this request filed, when it filed one. The revert handle "
            "is `incident_id`. None for a re-optimization that was derived from "
            "the network rather than requested by a person, which writes nothing."
        ),
    )

    before: list[RouteOut] = Field(default_factory=list)
    after: list[RouteOut] = Field(default_factory=list)

    completed: list[str] = Field(
        default_factory=list,
        description="Delivery ids served already, and so absent from the instance solved",
    )
    replanned: list[str] = Field(
        default_factory=list, description="Delivery ids that were in play"
    )
    moved: list[MoveOut] = Field(
        default_factory=list, description="Deliveries that changed vehicle"
    )

    cost: float = Field(description="Weighted objective for the new plan, in rupees")
    travel_cost: float
    travel_time: float
    distance_m: float
    fuel_litres: float
    feasible: bool
    runtime_ms: float
    convergence: list[float] = Field(default_factory=list)
    explanation: ExplanationOut | None = Field(
        default=None,
        description=(
            "Why the plan looks the way it does: the trigger's sentence, what "
            "each reported road costs now against what it cost before the "
            "report, and 1-3 decision statements per re-planned vehicle. Every "
            "figure quoted is one the re-plan computed — see "
            "`qgati.reopt.explain`."
        ),
    )
