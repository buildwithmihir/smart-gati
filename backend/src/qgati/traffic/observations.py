"""Tier 1: a measured travel time, and how it displaces the modelled estimate.

Where an incident is a *report* — someone says a road is shut or crawling, and
the model turns that into a number — an observation is a **measurement**. A
vehicle drove the road and took this long. The two are different kinds of
statement and are kept in different modules for that reason.

The two tiers
-------------
Asking what one road costs has three possible answers, and they rank:

1. **A measurement.** A fleet vehicle traversed the road and reported the time.
   This module holds those, and :func:`update_edge_from_observation` is where one
   arrives.
2. **A report.** An incident's flat ``x2.9`` — the conservative placeholder that
   stands in for a delay *nobody has measured yet*. This is what Tier 1 exists to
   retire: the placeholder applies only until real data arrives, and then the
   measurement wins.
3. **The model.** ``base_travel_time x congestion_multiplier`` from the clock.

Tier 2 and Tier 3 both live in :mod:`~qgati.traffic.simulator`; Tier 1 lives
here, and the simulator reads it. A closure is the exception to the ranking, and
is discussed below.

An observation is not a multiplier
----------------------------------
Everywhere else in the traffic layer a road's cost is ``base x factor``. An
observation has no factor: it *is* the time. Storing one as a multiplier of the
road's base would tie a measurement to the model it is supposed to correct —
recompute the base and the measurement silently changes with it — so
:class:`Observation` holds seconds and the simulator returns them as they are.

Why a closure outranks a measurement
------------------------------------
Tier 1 replaces an *estimate of speed*. A closure is not an estimate of anything:
it is a statement that the road cannot be driven. A measurement taken on a road
that is also reported shut is therefore contradictory rather than more
authoritative, and it does not reopen the road — the closed set is tested first,
in :func:`~qgati.traffic.simulator.get_traffic_multiplier`, and stays first.

In practice this only arises from a closure arriving on a road a vehicle is
already on. The fleet's own routes are cheapest paths, and a cheapest path never
includes an edge of infinite weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

__all__ = [
    "GPS",
    "Observation",
    "observation_for",
    "update_edge_from_observation",
]

Node = Hashable

#: Where an observation came from. One value today — a vehicle's own GPS — and a
#: field rather than a constant so a later phase can record a driver's radio call
#: or a road-side sensor without the schema changing meaning.
GPS = "gps"

Edge = tuple[Node, Node]


@dataclass(frozen=True, slots=True)
class Observation:
    """One road, measured, at one moment.

    ``travel_time`` is in seconds and is the whole point: it is a time, not a
    multiplier of anything.

    ``observed_at`` is an ISO string in the same form
    :class:`~qgati.api.store.StoredScenario` uses elsewhere. It defaults, in
    practice, to the scenario's own pricing timestamp rather than the wall-clock
    moment of the reading — the same choice the incident audit row makes, and for
    the same reason: a scenario is priced under one set of conditions, so an
    observation stamped any other way would claim a congestion band the costs
    around it were never built under.
    """

    u: Node
    v: Node
    travel_time: float
    observed_at: str
    source: str = GPS

    def __post_init__(self) -> None:
        # The API refuses these at the boundary, but the weight function is also
        # reachable from the watcher and from a caller building a state by hand,
        # and a non-finite cost here would propagate into a cost matrix and then
        # into a response body that cannot be rendered. Cheaper to refuse it at
        # the one place an observation is constructed.
        if not math.isfinite(self.travel_time) or self.travel_time <= 0.0:
            raise ValueError(
                f"an observation must be a finite, positive number of seconds, "
                f"got {self.travel_time!r}"
            )

    @property
    def edge(self) -> Edge:
        """The directed road this measurement is of."""
        return self.u, self.v

    def to_dict(self) -> dict:
        """JSON-ready form, matching :class:`~qgati.api.schemas.ObservationOut`."""
        return {
            "edge": {"u": self.u, "v": self.v},
            "travel_time": self.travel_time,
            "observed_at": self.observed_at,
            "source": self.source,
        }


def update_edge_from_observation(
    observations: Mapping[Edge, Observation],
    *,
    u: Node,
    v: Node,
    travel_time: float,
    observed_at: str,
    source: str = GPS,
) -> dict[Edge, Observation]:
    """Tier 1: record a measured time for one road, replacing any earlier one.

    Returns a **new** mapping rather than mutating the one handed in, because the
    mapping it is given belongs to a scenario record and to the immutable traffic
    state built from it. Editing either in place would change the costs behind a
    request that is already being served.

    One entry per road, not a history: the newest measurement is the one that
    prices the road, and a log of every reading a vehicle ever took belongs in
    ``traffic_log`` — which is where the caller puts it — rather than in the live
    overlay that decides what a route costs.
    """
    updated = dict(observations)
    updated[(u, v)] = Observation(
        u=u, v=v, travel_time=travel_time, observed_at=observed_at, source=source
    )
    return updated


def observation_for(
    observations: Iterable[Observation] | Mapping[Edge, Observation] | None,
    u: Node,
    v: Node,
) -> Observation | None:
    """The measurement of ``(u, v)``, or ``None`` if the road has not been measured.

    Accepts either shape: a mapping, which is what a traffic state holds, or a
    plain sequence of observations, which is what a caller assembling one by hand
    is likely to have. The mapping is the fast path — it is consulted once per
    edge relaxation inside Dijkstra, so the linear scan is only for the rare
    sequence form.
    """
    if observations is None:
        return None
    if isinstance(observations, Mapping):
        return observations.get((u, v))
    for item in observations:
        if item.edge == (u, v):
            return item
    return None
