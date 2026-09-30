"""Plausible fleet readings: where each vehicle is, and how long its road took.

This is the half of the simulated fleet that has no threads in it. Given a
scenario, a solution and a road network, it can say which road each vehicle is on
— and, given a random source, how long that road took *this* time. Everything
here is a pure function of its arguments, so the interesting questions (does the
noise vary? does a vehicle stay on a road that slows down?) are answerable
without starting anything.

Why a measured time is noisy at all
-----------------------------------
A rule-based reading is ``base_travel_time x multiplier``, both deterministic, so
every sample of one road in one condition band is the *same number* and its
standard deviation is exactly zero. That is the state the log is in before this
module exists, and it is why :mod:`qgati.traffic.detection` has a rule for a
zero-spread history at all.

A fleet changes that: two drivers on the same road at the same hour do not take
the same time, and the spread between them is the signal a detector needs. The
noise here is therefore not decoration — it is the entire reason a z-score
becomes meaningful. It is drawn from a **lognormal**, which is the conventional
shape for a travel time: strictly positive, right-skewed (a road can be much
slower than usual far more easily than much faster), and centred so the mean
multiplier is exactly 1.0, meaning the fleet is right about the road on average
and only ever disagrees about the individual trip.

How wide that spread is drawn is not a free choice either; see
:data:`NOISE_SIGMA`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Sequence

import networkx as nx

from qgati.traffic.simulator import WeightFn

__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "DEFAULT_SEED",
    "DEFAULT_TIME_SCALE",
    "NOISE_SIGMA",
    "CostOf",
    "TrackPosition",
    "VehicleTrack",
    "blocked_edge",
    "completed_stops",
    "corridor",
    "corridor_legs",
    "cost_lookup",
    "initial_tracks",
    "noisy_reading",
    "position",
    "route_nodes",
]

Node = Hashable
Edge = tuple[Node, Node]

#: How much a single trip's time moves around the modelled one, as the shape
#: parameter of the lognormal.
#:
#: **This is not a free parameter.** The detector's fallback rule flags a reading
#: above ``FALLBACK_FACTOR`` (1.2) times the road's modelled time, and an ordinary
#: reading must not trip it — a detector that flags one drive in ten is a
#: detector nobody reads. For a lognormal multiplier, 1.2x sits
#: ``(ln 1.2 + sigma^2 / 2) / sigma`` standard deviations above the mean, so the
#: noise amplitude decides the false-positive rate on its own:
#:
#:     sigma 0.06   ~3.0 sd    ~1 reading in 1000
#:     sigma 0.15   ~1.3 sd    ~1 reading in 10
#:
#: 0.06 is chosen for the first line. It is a modest spread — the middle half of
#: readings lands within about 4% of the model — but modest is what the locked
#: 1.2x margin has room for, and it is real spread rather than none, which is what
#: a standard deviation needs. An incident's flat x2.9 sits at
#: ``ln(2.9) / sigma`` ~ 18 standard deviations, so raising this would not make
#: the true positives any more visible; it would only bury them in false ones.
NOISE_SIGMA = 0.06

#: Seconds between ticks. The prompt asks for "~15-20 seconds"; the middle of
#: that range, and configurable per watcher through the start request.
DEFAULT_INTERVAL_SECONDS = 18.0

#: How fast the simulated fleet's clock runs against the wall clock. 1.0 is
#: real time; a demo raising it covers more of a route per tick.
DEFAULT_TIME_SCALE = 1.0

#: Fixes the noise, so a demo prints the same numbers twice. Not a security
#: device; it exists so output can be diffed.
DEFAULT_SEED = 0

#: ``(u, v) -> seconds``, or ``None`` for a road that cannot be driven. The
#: watcher passes the scenario's own live pricing, so a vehicle's position is
#: resolved under the same costs the optimizer is charged.
CostOf = Callable[[Edge], "float | None"]


def cost_lookup(graph: nx.Graph, weight: WeightFn) -> CostOf:
    """Adapt a networkx ``weight`` callable into a ``(u, v) -> seconds`` lookup.

    Two differences from ``weight`` itself, both deliberate.

    It reads the edge's real attributes rather than taking them as an argument,
    so a caller placing a vehicle does not have to carry the graph's internals
    around. And it collapses "impassable" to ``None``: ``weight`` reports a
    closed road as ``inf``, which is the right encoding for a search that is
    minimising, but not for a question about where a vehicle *is*. A vehicle is
    never an infinite distance into a road it cannot enter.
    """

    def lookup(edge: Edge) -> float | None:
        u, v = edge
        data = graph.get_edge_data(u, v)
        if data is None:
            return None
        cost = float(weight(u, v, data))
        return cost if math.isfinite(cost) else None

    return lookup


def noisy_reading(modelled: float, rng: random.Random) -> float:
    """One trip's time on a road the model says takes ``modelled`` seconds.

    The multiplier is lognormal with mean exactly 1.0 — ``exp(mu + sigma^2/2)``
    with ``mu = -sigma^2/2`` — so the fleet is unbiased about the road while
    every individual reading differs from the last. Strictly positive, so a
    reading can never be a zero or negative time no matter how far it lands from
    the mean; drawing a normal multiplier would eventually produce one.
    """
    return modelled * rng.lognormvariate(-(NOISE_SIGMA**2) / 2.0, NOISE_SIGMA)


def route_nodes(scenario, route: Iterable[int]) -> list[Node]:
    """The graph nodes a solution route visits, depot first and depot last.

    A :class:`~qgati.optimizer.models.Solution` route lists only its stops, so
    the depot's two appearances — the departure and the return leg — are added
    here. Without them the final leg back to the depot would not exist and a
    vehicle would stop reporting once it reached its last customer.
    """
    depot = scenario.depot.node
    return [depot, *(scenario.deliveries[index].node for index in route), depot]


def corridor_legs(
    graph: nx.Graph, nodes: Sequence[Node], weight
) -> tuple[tuple[Edge, ...], ...]:
    """The ordered road segments of each leg, one tuple per leg.

    The same walk :func:`corridor` does, kept in its legs rather than flattened.
    A route is a sequence of stops, so "which stop is this vehicle heading for"
    is a question about *which leg it is on* — and the flattened form has thrown
    that away by construction. :func:`completed_stops` is the caller that needs
    it back.

    Raises
    ------
    networkx.NetworkXNoPath
        If a leg cannot be driven at all under ``weight``. Left to the caller:
        a closure that severs a route is a real answer, and the watcher decides
        what to do about it rather than this function guessing.
    """
    return tuple(
        tuple(zip(path, path[1:]))
        for path in (
            nx.shortest_path(graph, origin, target, weight=weight)
            for origin, target in zip(nodes, nodes[1:])
        )
    )


def corridor(graph: nx.Graph, nodes: Sequence[Node], weight) -> tuple[Edge, ...]:
    """The ordered road segments a route runs along, under ``weight``.

    Each leg is the *cheapest path* between two consecutive stops, which is the
    road the vehicle would actually drive — the same traversal the cost matrix is
    built from, so the fleet reports on roads the optimizer priced rather than on
    a straight line between customers.

    Raises
    ------
    networkx.NetworkXNoPath
        If a leg cannot be driven at all under ``weight``. Left to the caller:
        a closure that severs a route is a real answer, and the watcher decides
        what to do about it rather than this function guessing.
    """
    return tuple(edge for leg in corridor_legs(graph, nodes, weight) for edge in leg)


@dataclass(slots=True)
class VehicleTrack:
    """One vehicle's progress along its own corridor.

    ``travelled`` is in **seconds of travel**, not distance and not edges: a
    vehicle advances by how long the clock ran, and where that puts it depends on
    how long each road takes. That is what makes a road slowing down keep the
    vehicle on it longer, which is the behaviour being simulated.

    ``delivery_indices`` and ``leg_ends`` exist so a track can still answer
    *which of its stops has it served* — the question a re-optimization opens
    with. They are recorded here because this is the only moment they exist: the
    corridor is assembled leg by leg and then flattened, and the solution it came
    from is kept nowhere. Given only ``edges`` and ``travelled`` a later reader
    can say what road the vehicle is on, but not what that road was taking it to.

    Mutable, unlike almost everything else in this project. A track is not shared
    — one watcher owns its tracks for its own scenario and holds a lock across
    every tick — so there is nothing for immutability to protect.
    """

    vehicle_id: str
    edges: tuple[Edge, ...]
    travelled: float = 0.0
    #: Delivery indices in visit order, matching the corridor's legs.
    delivery_indices: tuple[int, ...] = ()
    #: How many corridor edges have been covered once each stop is reached:
    #: ``leg_ends[k]`` is the corridor index a vehicle sits at having arrived at
    #: stop ``k``. One entry per leg, so the last one is the whole corridor and
    #: is the return to the depot rather than a stop.
    leg_ends: tuple[int, ...] = ()

    @property
    def dispatched(self) -> bool:
        """False for a vehicle with no stops, which has no corridor to drive."""
        return bool(self.edges)

    def advance(self, seconds: float) -> None:
        self.travelled += seconds


@dataclass(frozen=True, slots=True)
class TrackPosition:
    """Where a vehicle is: which road, and how far into it."""

    index: int
    edge: Edge
    #: Seconds already spent on :attr:`edge`.
    offset: float
    #: The road's cost under the costs it was placed with.
    cost: float

    @property
    def remaining(self) -> float:
        """Seconds left before the vehicle reaches the next intersection."""
        return max(self.cost - self.offset, 0.0)


def position(track: VehicleTrack, cost_of: CostOf) -> TrackPosition | None:
    """The road a vehicle is on, walking its corridor under the current costs.

    ``None`` means it is not on one: either it has finished its route, or the
    road ahead cannot be driven. Both are ordinary states rather than errors — a
    finished vehicle is simply no longer active, and a blocked one is the thing a
    closure is for — so the caller distinguishes them from the track itself
    rather than this function raising. :func:`blocked_edge` tells the two apart.

    One limitation, stated because it is a modelling choice rather than a bug: a
    road that closes *behind* a moving vehicle is not modelled. The walk has no
    way to know a road has already been driven without knowing how long it took,
    which is the thing being computed, so such a vehicle is reported as stopped
    at the closure instead of continuing past it. Routes are cheapest paths, and
    a cheapest path never includes a closed road, so this only arises from a
    closure injected onto a corridor a fleet is already driving.
    """
    remaining = track.travelled
    for index, edge in enumerate(track.edges):
        cost = cost_of(edge)
        if cost is None:
            return None
        if remaining < cost:
            return TrackPosition(index=index, edge=edge, offset=remaining, cost=cost)
        remaining -= cost
    return None


def blocked_edge(track: VehicleTrack, cost_of: CostOf) -> Edge | None:
    """The first road on a corridor that cannot be driven, if there is one.

    The companion to :func:`position`, answering the other half of the question
    it returns ``None`` to. A vehicle that is not on a road has either finished
    its route or come up against a closure, and a report that could not tell
    those apart would leave a reader guessing which.
    """
    for edge in track.edges:
        if cost_of(edge) is None:
            return edge
    return None


def completed_stops(track: VehicleTrack, cost_of: CostOf) -> int:
    """How many of a route's stops this vehicle has already reached.

    The stops it has served are the ones behind it on its own corridor, so this
    is the corridor index it currently sits at, measured against the leg
    boundaries recorded at dispatch. Nothing else has to be tracked per tick: the
    progress is already carried by ``travelled``, which the watcher advances.

    A vehicle that is not on a road has either finished — every stop reached — or
    is stopped behind a closure. :func:`blocked_edge` tells the two apart, the
    same distinction :func:`position` leaves to its caller, and a stuck vehicle
    reports what it genuinely delivered rather than either extreme.
    """
    if not track.leg_ends:
        return 0

    here = position(track, cost_of)
    if here is not None:
        return _stops_behind(track, here.index)

    blocked = blocked_edge(track, cost_of)
    if blocked is not None:
        # `position` only reaches the road it cannot price by consuming every
        # road before it, so a blocked vehicle sits *at* that road rather than
        # somewhere indeterminate behind it — and the stops behind that index are
        # ones it genuinely served.
        return _stops_behind(track, track.edges.index(blocked))
    return len(track.delivery_indices)


def _stops_behind(track: VehicleTrack, index: int) -> int:
    """Stops whose leg ends at or before corridor position ``index``.

    The last leg boundary is dropped because it is the return to the depot rather
    than an arrival at a stop — counting it would let a vehicle that has served
    every customer claim one delivery more than its route contains.
    """
    return sum(
        1 for end in track.leg_ends[: len(track.delivery_indices)] if end <= index
    )


def initial_tracks(
    scenario, solution, graph: nx.Graph, weight: WeightFn
) -> list[VehicleTrack]:
    """Place every dispatched vehicle on its corridor, already under way.

    Vehicles start **spread along their own routes** rather than all at the
    depot. A watcher attaches to a fleet that is already out — three vehicles
    leaving the same depot at the same instant would report on the same first
    road, which is both unrealistic and a poor demonstration. Vehicle ``i`` of
    ``n`` therefore starts ``(i + 1) / (n + 1)`` of the way along its own
    corridor, so a three-vehicle fleet is at the first, middle and last thirds.

    A vehicle whose solution route is empty is skipped: it has no corridor and
    nothing to report, and it would only ever be a row of blanks.

    ``weight`` must be the one the cost matrix was built with. The corridor is
    the road the vehicle would actually drive *under those costs*, and the
    starting offsets are measured in the seconds those costs imply — so a fleet
    dispatched under peak conditions starts spread along its peak routes, not
    along the free-flow ones it would have taken at 3am.
    """
    lookup = cost_lookup(graph, weight)
    tracks: list[VehicleTrack] = []
    routed = [
        (index, route) for index, route in enumerate(solution.routes) if route
    ]
    for order, (index, route) in enumerate(routed):
        nodes = route_nodes(scenario, route)
        try:
            legs = corridor_legs(graph, nodes, weight)
        except nx.NetworkXNoPath:
            # A stop is unreachable under these costs. Nothing about the fleet's
            # placement should be the thing that raises: an unroutable vehicle is
            # simply not dispatched, and the caller reports how many were.
            continue
        edges = tuple(edge for leg in legs for edge in leg)
        if not edges:
            continue
        total = sum(cost for cost in map(lookup, edges) if cost is not None)
        fraction = (order + 1) / (len(routed) + 1)
        tracks.append(
            VehicleTrack(
                vehicle_id=scenario.vehicles[index].id,
                edges=edges,
                travelled=fraction * total,
                delivery_indices=tuple(route),
                leg_ends=_leg_ends(legs),
            )
        )
    return tracks


def _leg_ends(legs: tuple[tuple[Edge, ...], ...]) -> tuple[int, ...]:
    """Run a corridor's per-leg edge counts up into cumulative boundaries.

    A leg carrying no edges — two stops sharing a road node, or a stop at the
    depot — leaves the running total where it was, which is right: the vehicle
    arrives at the second of them the moment it arrives at the first, and
    :func:`_stops_behind` counts both.
    """
    running = 0
    ends: list[int] = []
    for leg in legs:
        running += len(leg)
        ends.append(running)
    return tuple(ends)
