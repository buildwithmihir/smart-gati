"""Rule-based traffic simulation.

**This is a simulation for demonstrating dynamic routing, not real-world traffic
data.** The congestion factors are anchored to the TomTom Traffic Index 2025
figure for New Delhi — an average congestion level of 60.2%, and 192% at the
evening peak — but the road-class split below is a modelling choice, not a
measurement. Nothing in this project should be read as a claim about actual road
conditions.

What it does
------------
:func:`traffic_weight_function` returns a callable suitable as the ``weight``
argument of anything in :mod:`qgati.routing` or
:func:`~qgati.graph.cost_matrix.build_cost_matrix`. Handing that callable to the
cost-matrix builder is the *entire* integration: past that point an optimizer
sees an ordinarily-priced cost matrix, built under the simulated conditions. The
simulation changes travel *time* only; a leg's distance is a property of the road
it runs along and does not move with traffic, so congestion shows up in the
objective through the time term and through a lower average speed, which is what
raises the fuel term.

The graph is never mutated. A road closure cannot be expressed by deleting an
edge, because the Delhi graph is a cached, shared, read-only object — deleting
from it would leak the closure into every later request. Instead a closed edge
is given infinite weight, which Dijkstra discards for the same reason it
discards any unaffordable edge.

The states
----------
Time of day alone selects the congestion state. There is no operator toggle, so
a demo left running unattended always prices under something sensible::

    normal      x1.0    outside the daytime band
    moderate    x1.6    06:00-22:00, outside the peak windows
    peak        x2.9    08:00-10:00 and 17:00-20:00

The finding that shaped the model
---------------------------------
A multiplier applied **uniformly** to every road cannot change which route is
optimal. A tour is a sum of legs: scale every edge by 2.9 and every tour scales
by 2.9, leaving the argmin untouched. So a flat congestion factor produces the
*same* routes at a higher price, which is useless for demonstrating dynamic
routing.

Congestion is therefore weighted by road class, which is what the OSM
``highway`` tag on every edge already tells us::

    congestion_multiplier(edge) = 1 + (FACTOR - 1) * sensitivity(edge)

    through roads (secondary, tertiary, primary, trunk, motorway, + _link)  sensitivity 1.00
    everything else (residential, living_street, service, ...)              sensitivity 0.30

so an arterial takes the full x2.9 at peak against x1.57 on a side street, and
x1.6 against x1.18 at moderate. On this extract the split is load-bearing:
arterials run at 48.6 kph against 35.0 kph on residential streets, so in the
normal band an arterial is 1.39x faster. Under moderate the two nearly converge
(30.4 against 29.7 kph) and at peak the side street is plainly the quicker road
(22.3 against 16.8 kph), so routing moves off the arterial.

Incidents
---------
An accident on a named edge **overrides** the congestion multiplier with a flat
x2.9 — the peak factor, reused as a conservative placeholder because
overestimating a crash delay is safer than underestimating it. It deliberately
does not compound: x2.9 is already a worst case, not a base to stack further
multipliers on. A fully blocked road is a closure instead, and takes infinite
weight.

Rain used to take a flat x1.4 here. It has no factor any more: by the invariance
above a flat factor reroutes nothing, so it was a price rise wearing a
condition's clothes.

Observations
------------
A fleet vehicle's own measured time is **Tier 1** and outranks everything above;
see :mod:`qgati.traffic.observations`. It rides on :class:`TrafficState` and is
applied in :func:`_simulated_cost`, so the cost matrix, the geometry tracer and
the logged rows all pick it up without being told about it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable, Hashable, Iterable, Mapping

import networkx as nx

from qgati.traffic.observations import Observation, observation_for

__all__ = [
    "ACCIDENT",
    "ActiveConditions",
    "CONGESTION_FACTORS",
    "Edge",
    "LOCAL_ROAD_SENSITIVITY",
    "MODERATE",
    "MODERATE_FACTOR",
    "MODERATE_WINDOWS",
    "NORMAL",
    "NORMAL_FACTOR",
    "PEAK",
    "PEAK_FACTOR",
    "PEAK_WINDOWS",
    "ROAD_CLOSURE",
    "THROUGH_ROAD_CLASSES",
    "THROUGH_ROAD_SENSITIVITY",
    "TrafficState",
    "WeightFn",
    "base_travel_time",
    "congestion_multiplier",
    "congestion_state",
    "edge_of",
    "get_traffic_multiplier",
    "is_peak_hour",
    "road_class_of",
    "sensitivity_of",
    "simulated_travel_time",
    "traffic_weight_function",
]

Node = Hashable

#: ``(u, v, edge_data) -> float``, the signature :mod:`qgati.routing.dijkstra`
#: and :func:`~qgati.graph.cost_matrix.build_cost_matrix` accept for ``weight``.
WeightFn = Callable[[Node, Node, dict], float]


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
#: The three congestion states, and the ``traffic_condition`` column values.
NORMAL = "normal"
MODERATE = "moderate"
PEAK = "peak"

#: Free-flowing. The baseline every other factor is a multiple of.
NORMAL_FACTOR = 1.0

#: Delhi's *average* congestion is 60.2% (TomTom Traffic Index 2025), which is a
#: travel time of 1.602x free-flow — 1.6 here. This is the ordinary daytime
#: state, not an incident.
MODERATE_FACTOR = 1.6

#: The 6pm peak reaches 192% (TomTom Traffic Index 2025), a travel time of 2.92x
#: free-flow — 2.9 here. This is *the* congestion number in the model: every
#: other state's figure is this one scaled by a road's sensitivity, and an
#: accident reuses it outright.
PEAK_FACTOR = 2.9

#: Peak windows as half-open ``[start, end)`` wall-clock ranges, evaluated in the
#: timestamp's own timezone. 08:00-10:00 and 17:00-20:00, so 10:00 is already
#: out of peak and 08:00 already is.
#:
#: Every day of the week is treated alike. Real Delhi peaks are heavier on
#: weekdays and start later at weekends; the log records ``day_of_week`` so a
#: later phase can learn that split from data rather than have it hard-coded.
PEAK_WINDOWS: tuple[tuple[time, time], ...] = (
    (time(8, 0), time(10, 0)),
    (time(17, 0), time(20, 0)),
)

#: The daytime band, in the same half-open form. Outside it the roads are
#: treated as free-flowing, which is what gives the model its ``normal`` state.
#:
#: This boundary is **not** TomTom-sourced the way the factors above are — the
#: index publishes congestion by hour, not a definition of "daytime". 06:00-22:00
#: is a modelling choice, and the first number to revisit if a demo ever prices
#: oddly at an edge-of-day hour.
MODERATE_WINDOWS: tuple[tuple[time, time], ...] = ((time(6, 0), time(22, 0)),)

#: State -> multiplier, for a through road. A side street takes a fraction of
#: the excess above 1.0; see :func:`congestion_multiplier`.
CONGESTION_FACTORS: Mapping[str, float] = {
    NORMAL: NORMAL_FACTOR,
    MODERATE: MODERATE_FACTOR,
    PEAK: PEAK_FACTOR,
}

#: Roads that carry through traffic — where congestion actually accumulates.
THROUGH_ROAD_CLASSES = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    }
)

#: A through road takes the full peak factor.
THROUGH_ROAD_SENSITIVITY = 1.0

#: Residential streets, service roads and anything unrecognised take 30% of it.
#: Unrecognised defaults to this — assuming an unknown road is a quiet one is the
#: conservative direction, since it under- rather than over-states congestion.
LOCAL_ROAD_SENSITIVITY = 0.30

#: Values of the ``incident_type`` column.
ACCIDENT = "accident"
ROAD_CLOSURE = "road_closure"

#: Used when an edge carries neither ``travel_time`` nor ``weight``. The Delhi
#: graph always has ``travel_time`` and the synthetic test graphs have ``weight``,
#: so this only guards a degenerate hand-built graph.
DEFAULT_BASE_TRAVEL_TIME = 1.0


def _within(timestamp: datetime, windows: tuple[tuple[time, time], ...]) -> bool:
    """True when the timestamp's wall clock falls inside any of ``windows``.

    Compares wall-clock time in the timestamp's own timezone, so an aware
    timestamp taken in Delhi is judged by Delhi's clock.
    """
    moment = timestamp.time()
    return any(start <= moment < end for start, end in windows)


def is_peak_hour(timestamp: datetime) -> bool:
    """True when ``timestamp`` falls inside a peak window."""
    return _within(timestamp, PEAK_WINDOWS)


def congestion_state(timestamp: datetime) -> str:
    """The congestion state at ``timestamp`` — ``"normal"``, ``"moderate"`` or
    ``"peak"``.

    Derived from the clock alone, so a scenario prices under a real state
    whether or not anyone remembered to set one. Peak is tested first because
    the peak windows sit *inside* the daytime band; that order is what stops
    09:00 reading as merely moderate.
    """
    if _within(timestamp, PEAK_WINDOWS):
        return PEAK
    if _within(timestamp, MODERATE_WINDOWS):
        return MODERATE
    return NORMAL


def road_class_of(attributes: Mapping) -> str:
    """The OSM ``highway`` class of an edge, or ``""`` when untagged.

    OSM permits a tag to repeat, which OSMnx surfaces as a list; the first value
    is taken, matching how the rest of the project reads repeated tags.
    """
    value = attributes.get("highway")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return "" if value is None else str(value)


def sensitivity_of(road_class: str) -> float:
    """How much of a congestion factor's excess a road class takes, ``[0, 1]``."""
    return (
        THROUGH_ROAD_SENSITIVITY
        if road_class in THROUGH_ROAD_CLASSES
        else LOCAL_ROAD_SENSITIVITY
    )


def congestion_multiplier(road_class: str, state: str) -> float:
    """The multiplier a road class takes in a congestion state.

    The state's factor is the figure a *through* road takes; every other road
    takes the same fraction of the excess over 1.0 that :func:`sensitivity_of`
    reports::

        multiplier = 1 + (FACTOR - 1) * sensitivity

    ``"normal"`` works out to exactly 1.0 for every class, since the excess is
    zero — no special case needed.
    """
    factor = CONGESTION_FACTORS[state]
    return 1.0 + (factor - 1.0) * sensitivity_of(road_class)


# --------------------------------------------------------------------------- #
# Edges and conditions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Edge:
    """A directed road segment, identified by its endpoints.

    ``road_class`` is carried rather than the whole attribute mapping so that an
    ``Edge`` stays hashable and cheap: incidents are held in sets, and a mapping
    field would make the type unhashable.

    Construct through :func:`edge_of` to be sure the class came from the graph.
    """

    u: Node
    v: Node
    road_class: str = ""


def edge_of(graph: nx.Graph, u: Node, v: Node) -> Edge:
    """The edge from ``u`` to ``v``, as it exists in ``graph``.

    Parallel edges are collapsed the way routing collapses them — the *cheapest*
    one wins, so the class reported is the class of the road a driver would
    actually take.

    Raises
    ------
    KeyError
        If there is no such edge.
    """
    if u not in graph.adj or v not in graph.adj[u]:
        raise KeyError(f"no edge from {u!r} to {v!r} in this graph")
    return Edge(u, v, road_class_of(_cheapest_attributes(graph, u, v)))


def _cheapest_attributes(graph: nx.Graph, u: Node, v: Node) -> Mapping:
    """Attributes of the cheapest parallel edge between ``u`` and ``v``."""
    data = graph.adj[u][v]
    if graph.is_multigraph():
        return min(data.values(), key=base_travel_time)
    return data


def base_travel_time(attributes: Mapping) -> float:
    """The untraffic'd travel time of an edge, in seconds.

    Reads ``travel_time``, falling back to ``weight`` — the two are equal on the
    Delhi graph, and synthetic graphs carry only ``weight``.
    """
    for key in ("travel_time", "weight"):
        value = attributes.get(key)
        if value is not None:
            return float(value)
    return DEFAULT_BASE_TRAVEL_TIME


def _normalise_edges(edges: Iterable) -> frozenset[tuple[Node, Node]]:
    """Coerce whatever a caller supplied into a set of ``(u, v)`` pairs."""
    normalised: set[tuple[Node, Node]] = set()
    for edge in edges:
        if isinstance(edge, Edge):
            normalised.add((edge.u, edge.v))
        else:
            u, v = edge
            normalised.add((u, v))
    return frozenset(normalised)


@dataclass(frozen=True, slots=True)
class ActiveConditions:
    """The manually-reported conditions, layered on top of the clock.

    Congestion is deliberately *not* here: it is derived from the timestamp, so
    there is no way to represent "peak hour is off at 9am" — and no way for a
    demo left running to forget to set it. Incidents are explicit because
    nothing in the data could imply them.
    """

    #: Directed edges carrying an accident — a flat x2.9 on those roads only.
    accident_edges: frozenset[tuple[Node, Node]] = frozenset()
    #: Directed edges that are impassable.
    closed_edges: frozenset[tuple[Node, Node]] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "accident_edges", _normalise_edges(self.accident_edges))
        object.__setattr__(self, "closed_edges", _normalise_edges(self.closed_edges))

    @property
    def is_clear(self) -> bool:
        """True when nothing manual is applied — the baseline case."""
        return not (self.accident_edges or self.closed_edges)


@dataclass(frozen=True, slots=True)
class TrafficState:
    """Everything needed to price one road at one moment.

    A naive ``timestamp`` is interpreted as local time and given the correct UTC
    offset for that instant, so logged rows always carry an unambiguous time.

    Conditions are fixed when the state is built and never re-evaluated: a
    scenario priced under a state keeps those costs for its lifetime, which is
    what makes two solvers compared on it comparable.

    ``observations`` are the fleet's own measurements — Tier 1, see
    :mod:`qgati.traffic.observations`. They belong on the state rather than being
    passed separately to :func:`traffic_weight_function` because *every* pricing
    path already threads a state through: the cost matrix, the geometry tracer,
    :func:`simulated_travel_time` and the log rows all read it. A measurement
    handed round alongside the state instead would reach whichever callers
    remembered it and silently miss the rest.
    """

    timestamp: datetime
    conditions: ActiveConditions = field(default_factory=ActiveConditions)

    #: Measured travel times, keyed by directed road. Empty is the ordinary case:
    #: nothing has driven these roads yet, and the model prices them alone.
    #:
    #: Held as a mapping for the lookup, which happens once per edge relaxation
    #: inside Dijkstra, and excluded from :meth:`__hash__` because that lookup is
    #: what it is for — the dataclass stays hashable and two states are still
    #: equal only if they carry the same measurements. Treat it as read-only and
    #: build a new state rather than editing this one; editing it would change
    #: the costs behind a request already being served.
    observations: Mapping[Edge, Observation] = field(
        default_factory=dict, hash=False
    )

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            # astimezone() on a naive datetime assumes local time and attaches
            # the offset in force *at that instant*, DST included.
            object.__setattr__(self, "timestamp", self.timestamp.astimezone())

    @classmethod
    def now(cls, conditions: ActiveConditions | None = None) -> TrafficState:
        """The state right now, by the system clock."""
        return cls(
            timestamp=datetime.now().astimezone(),
            conditions=conditions or ActiveConditions(),
        )

    @property
    def peak_hour(self) -> bool:
        return is_peak_hour(self.timestamp)

    @property
    def congestion(self) -> str:
        """The congestion state: ``"normal"``, ``"moderate"`` or ``"peak"``."""
        return congestion_state(self.timestamp)

    @property
    def traffic_condition(self) -> str:
        """The ``traffic_condition`` column value — :attr:`congestion`."""
        return self.congestion

    def multiplier(self, edge: Edge | tuple[Node, Node]) -> float:
        """The simulated multiplier for one edge. See
        :func:`get_traffic_multiplier`.

        A multiplier is the wrong shape for a measurement, so this reports only
        what the model and the reported incidents say; an observed road's *cost*
        comes from :func:`simulated_travel_time`.
        """
        return get_traffic_multiplier(edge, self.timestamp, self.conditions)

    def observation(self, u: Node, v: Node) -> Observation | None:
        """The fleet's measurement of the road from ``u`` to ``v``, if any."""
        return self.observations.get((u, v))

    def incident_type(self, edge: Edge | tuple[Node, Node]) -> str | None:
        """``"road_closure"``, ``"accident"``, or ``None`` for a normal road.

        A closure outranks an accident: a shut road is shut regardless of what
        else was reported on it.
        """
        u, v, _ = _unpack(edge)
        if (u, v) in self.conditions.closed_edges:
            return ROAD_CLOSURE
        if (u, v) in self.conditions.accident_edges:
            return ACCIDENT
        return None


def _unpack(edge: Edge | tuple[Node, Node]) -> tuple[Node, Node, str]:
    """Accept an :class:`Edge` or a bare ``(u, v)`` pair.

    A bare pair carries no class, so it is priced as an untagged road. Callers
    that have the graph should prefer :func:`edge_of`, which reads the real one.
    """
    if isinstance(edge, Edge):
        return edge.u, edge.v, edge.road_class
    u, v = edge
    return u, v, ""


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def get_traffic_multiplier(
    edge: Edge | tuple[Node, Node],
    timestamp: datetime,
    active_conditions: ActiveConditions | None = None,
) -> float:
    """How much slower ``edge`` is under these conditions, as a factor.

    Returns ``math.inf`` for a closed edge. That is the encoding of "impassable"
    throughout this module: routing multiplies the base travel time by this
    factor, and an infinite cost is one Dijkstra will never settle.

    Congestion and incidents do **not** compound. An accident *replaces* the
    edge's congestion multiplier with a flat x2.9 — see the module docstring for
    why a peak-hour crash is not ``2.9 x 2.9``. A closure outranks an accident,
    since a shut road is shut whatever else was reported on it.

    Parameters
    ----------
    edge
        An :class:`Edge`, or a bare ``(u, v)`` pair (priced as untagged).
    timestamp
        When. The congestion state is decided from this alone.
    active_conditions
        Manual conditions; ``None`` means none.
    """
    u, v, road_class = _unpack(edge)
    conditions = active_conditions or ActiveConditions()

    if (u, v) in conditions.closed_edges:
        return math.inf
    if (u, v) in conditions.accident_edges:
        # Flat, and deliberately not scaled by road class: this stands in for a
        # delay nobody has measured yet, and erring high is the safe direction.
        return PEAK_FACTOR
    return congestion_multiplier(road_class, congestion_state(timestamp))


def _simulated_cost(
    u: Node, v: Node, attributes: Mapping, state: TrafficState
) -> float:
    """Simulated cost of travelling one specific edge, attributes included.

    This is the one place the three possible answers about a road are ranked, and
    the ranking is the whole of the tiering described in
    :mod:`qgati.traffic.observations`: a closure cannot be driven at all, a
    measurement beats every estimate, and the model's own number — with any
    reported incident folded into it — is what is left.
    """
    edge = Edge(u, v, road_class_of(attributes))
    multiplier = state.multiplier(edge)
    if math.isinf(multiplier):
        return math.inf

    observed = state.observation(u, v)
    if observed is not None:
        # Tier 1. A measured time replaces the estimate outright, and is *not*
        # scaled by anything: it is already the seconds the road took. Applying
        # the congestion factor or an incident's flat x2.9 on top of it would
        # charge the same delay twice, and would let the placeholder this exists
        # to retire survive underneath the measurement that retired it.
        return observed.travel_time

    return base_travel_time(attributes) * multiplier


def _cheapest_traversal(
    graph: nx.Graph, u: Node, v: Node, state: TrafficState
) -> float:
    """The cost routing would charge to get from ``u`` straight to ``v``.

    With parallel edges this is the minimum over all of them *after* the
    multipliers are applied — not the cheapest base edge scaled, since a quiet
    side street alongside a congested arterial can win under peak and lose
    off it.
    """
    data = graph.adj[u][v]
    if graph.is_multigraph():
        return min(
            _simulated_cost(u, v, attributes, state) for attributes in data.values()
        )
    return _simulated_cost(u, v, data, state)


def simulated_travel_time(
    graph: nx.Graph, edge: Edge | tuple[Node, Node], state: TrafficState
) -> float | None:
    """The travel time routing would charge for ``edge``, in seconds.

    ``None`` for a closed road: it is impassable, so there is no travel time to
    report. That is the value the log records, with ``incident_type`` carrying
    the reason.

    A road the fleet has measured returns the **measurement**, not the model's
    estimate of it — see :mod:`qgati.traffic.observations`. A row logged from
    here therefore carries the seconds a vehicle actually took, on a road that
    has been driven, and the model's own seconds on one that has not.

    This is the same computation :func:`traffic_weight_function` performs, routed
    through the same helpers, so a number logged from here is provably the number
    the optimizer was charged.
    """
    u, v, _ = _unpack(edge)
    cost = _cheapest_traversal(graph, u, v, state)
    return None if math.isinf(cost) else cost


def traffic_weight_function(graph: nx.Graph, state: TrafficState) -> WeightFn:
    """A ``weight`` callable for routing under ``state``.

    Hand it to any of :func:`~qgati.routing.dijkstra.dijkstra`,
    :func:`~qgati.routing.dijkstra.dijkstra_all_pairs`,
    :func:`~qgati.graph.cost_matrix.build_cost_matrix` or
    :func:`~qgati.graph.geometry.route_polyline`, and that call sees the
    simulated road network instead of the static one.

    The closure encoding lives here: a closed edge is given infinite weight, so
    the search routes around it. The graph itself is only read.
    """

    def weight(u: Node, v: Node, data: dict) -> float:
        if graph.is_multigraph():
            return min(
                _simulated_cost(u, v, attributes, state)
                for attributes in data.values()
            )
        return _simulated_cost(u, v, data, state)

    return weight
