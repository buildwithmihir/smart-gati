"""Rule-based traffic simulation.

**This is a simulation for demonstrating dynamic routing, not real-world traffic
data.** The factors below are plausible round numbers chosen so that a route
computed at 9am differs from one computed at 3pm — they are not calibrated
against measured Delhi traffic, and nothing in this project should be read as a
claim about actual road conditions.

What it does
------------
:func:`traffic_weight_function` returns a callable suitable as the ``weight``
argument of anything in :mod:`qgati.routing` or
:func:`~qgati.graph.cost_matrix.build_cost_matrix`. Handing that callable to the
cost-matrix builder is the *entire* integration: past that point an optimizer
sees an ordinary travel-time matrix, priced under the simulated conditions.

The graph is never mutated. A road closure cannot be expressed by deleting an
edge, because the Delhi graph is a cached, shared, read-only object — deleting
from it would leak the closure into every later request. Instead a closed edge
is given infinite weight, which Dijkstra discards for the same reason it
discards any unaffordable edge.

The finding that shaped the model
---------------------------------
A multiplier applied **uniformly** to every road cannot change which route is
optimal. A tour is a sum of legs: scale every edge by 1.7 and every tour scales
by 1.7, leaving the argmin untouched. So a flat peak factor produces the *same*
routes at a higher price, which is useless for demonstrating dynamic routing.

Congestion is therefore weighted by road class, which is what the OSM
``highway`` tag on every edge already tells us::

    peak_multiplier(edge) = 1 + (PEAK_FACTOR - 1) * sensitivity(edge)

    through roads (secondary, tertiary, primary, trunk, motorway, + _link)  x1.70
    everything else (residential, living_street, service, ...)             x1.24

``PEAK_FACTOR`` is the through-road figure, so the headline ``x1.7`` is exactly
what an arterial takes. On this extract the split is load-bearing: arterials run
at 48.6 kph against 35.0 kph on residential streets, so off-peak the arterial is
1.39x faster, but under peak their effective speeds converge (28.6 against 28.9
kph) and routing moves onto side streets.

Rain, by contrast, is applied flat at ``x1.4``. By the invariance above, rain on
its own raises every route by 40% and reroutes nothing — it bites only in
combination with a factor that varies by road. That is a property of the model
rather than a defect, and the README states it as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable, Hashable, Iterable, Mapping

import networkx as nx

__all__ = [
    "ACCIDENT",
    "ACCIDENT_FACTOR",
    "ActiveConditions",
    "Edge",
    "LOCAL_ROAD_SENSITIVITY",
    "PEAK_FACTOR",
    "PEAK_WINDOWS",
    "RAIN_FACTOR",
    "ROAD_CLOSURE",
    "THROUGH_ROAD_CLASSES",
    "THROUGH_ROAD_SENSITIVITY",
    "TrafficState",
    "WeightFn",
    "base_travel_time",
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
#: Peak-hour congestion factor for a through road. This is *the* number in the
#: model — every other peak figure is this one scaled by a road's sensitivity.
PEAK_FACTOR = 1.7

#: Rain slows every road by the same 40%. See the module docstring: flat factors
#: cannot reroute, so this matters only alongside a road-dependent factor.
RAIN_FACTOR = 1.4

#: An accident is a localised ×3. Applied to named roads only, never network-wide
#: — a global ×3 is uniform, so it would cost more without changing any route.
ACCIDENT_FACTOR = 3.0

#: Peak windows as half-open ``[start, end)`` wall-clock ranges, evaluated in the
#: timestamp's own timezone. 08:00-10:00 and 17:00-20:00, so 10:00 is already
#: off-peak and 08:00 already is.
#:
#: Every day of the week is treated alike. Real Delhi peaks are heavier on
#: weekdays and start later at weekends; the log records ``day_of_week`` so a
#: later phase can learn that split from data rather than have it hard-coded.
PEAK_WINDOWS: tuple[tuple[time, time], ...] = (
    (time(8, 0), time(10, 0)),
    (time(17, 0), time(20, 0)),
)

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


def is_peak_hour(timestamp: datetime) -> bool:
    """True when ``timestamp`` falls inside a peak window.

    Compares wall-clock time in the timestamp's own timezone, so an aware
    timestamp taken in Delhi is judged by Delhi's clock.
    """
    moment = timestamp.time()
    return any(start <= moment < end for start, end in PEAK_WINDOWS)


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
    """How much of the peak factor a road class takes, in ``[0, 1]``."""
    return (
        THROUGH_ROAD_SENSITIVITY
        if road_class in THROUGH_ROAD_CLASSES
        else LOCAL_ROAD_SENSITIVITY
    )


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
    """The manually-toggled conditions, layered on top of the clock.

    Peak hour is deliberately *not* here: it is derived from the timestamp, so
    there is no way to represent "peak hour is off at 9am". Rain and incidents
    are explicit because nothing in the data could imply them.
    """

    rain: bool = False
    #: Directed edges carrying an accident — a localised ×3, not network-wide.
    accident_edges: frozenset[tuple[Node, Node]] = frozenset()
    #: Directed edges that are impassable.
    closed_edges: frozenset[tuple[Node, Node]] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "accident_edges", _normalise_edges(self.accident_edges))
        object.__setattr__(self, "closed_edges", _normalise_edges(self.closed_edges))

    @property
    def is_clear(self) -> bool:
        """True when nothing manual is applied — the baseline case."""
        return not (self.rain or self.accident_edges or self.closed_edges)


@dataclass(frozen=True, slots=True)
class TrafficState:
    """Everything needed to price one road at one moment.

    A naive ``timestamp`` is interpreted as local time and given the correct UTC
    offset for that instant, so logged rows always carry an unambiguous time.

    Conditions are fixed when the state is built and never re-evaluated: a
    scenario priced under a state keeps those costs for its lifetime, which is
    what makes two solvers compared on it comparable.
    """

    timestamp: datetime
    conditions: ActiveConditions = field(default_factory=ActiveConditions)

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
    def weather(self) -> str:
        """The ``weather_condition`` column value."""
        return "rain" if self.conditions.rain else "clear"

    @property
    def traffic_condition(self) -> str:
        """The ``traffic_condition`` column value."""
        return "peak" if self.peak_hour else "off_peak"

    def multiplier(self, edge: Edge | tuple[Node, Node]) -> float:
        """The simulated multiplier for one edge. See
        :func:`get_traffic_multiplier`."""
        return get_traffic_multiplier(edge, self.timestamp, self.conditions)

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

    The factors compose **multiplicatively**, so peak rain on an accident road is
    ``1.7 × 1.4 × 3.0``. Compounding rather than taking the worst is the choice
    that matches how the conditions actually interact — a crash in the rain at
    5pm is worse than any one of the three.

    Parameters
    ----------
    edge
        An :class:`Edge`, or a bare ``(u, v)`` pair (priced as untagged).
    timestamp
        When. Peak hour is decided from this alone.
    active_conditions
        Manual conditions; ``None`` means none.
    """
    u, v, road_class = _unpack(edge)
    conditions = active_conditions or ActiveConditions()

    if (u, v) in conditions.closed_edges:
        return math.inf

    multiplier = 1.0
    if is_peak_hour(timestamp):
        multiplier *= 1.0 + (PEAK_FACTOR - 1.0) * sensitivity_of(road_class)
    if conditions.rain:
        multiplier *= RAIN_FACTOR
    if (u, v) in conditions.accident_edges:
        multiplier *= ACCIDENT_FACTOR
    return multiplier


def _simulated_cost(
    u: Node, v: Node, attributes: Mapping, state: TrafficState
) -> float:
    """Simulated cost of travelling one specific edge, attributes included."""
    edge = Edge(u, v, road_class_of(attributes))
    multiplier = state.multiplier(edge)
    if math.isinf(multiplier):
        return math.inf
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
