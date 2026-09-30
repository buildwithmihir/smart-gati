"""What justifies re-planning a fleet that is already out.

A re-optimization is not free. It invalidates a plan drivers may already be
working to, it costs a solve, and — the reason this module exists — without a
rule about when it may run, "adaptive routing" is just an endpoint nobody can
tell apart from a manual re-solve. So the rule is: something must have actually
happened, and the endpoint refuses when nothing has.

Two things count, and they are different kinds of claim:

**An incident** is a *report*. Somebody said a road is closed or slow, and the
scenario is priced as though it is. It needs no corroboration — an operator
closing a road is reason enough to re-plan around it, and waiting for a vehicle
to drive into it first would be a strange way to use the information.

**An anomaly** is an *observation*. The fleet drove a road and the detector found
the trip unlike what that road's history says it should be — a measurement, not a
claim, and the only signal here that comes from the network itself rather than
from someone describing it.

Both can hold at once, and the common case is that they do: an incident is
injected, a vehicle drives it, and the detector independently flags the trip. That
is reported as both rather than collapsed into one, because "we were told" and "we
measured" are different strengths of evidence and a reader deciding whether to
trust a re-plan should be able to see which they have.

An anomaly is read from the fleet's **most recent tick only**. A flag says a road
is misbehaving now; one from twenty ticks ago describes a moment nobody is
re-planning for, and a trigger that never expires would make every later request
look justified. The cost is stated where it is incurred — see
:attr:`~qgati.fleet.watcher.ScenarioWatcher.last_flags`.

The third kind is not derived at all
------------------------------------
:data:`OVERRIDE` is **a driver saying so**. It is the only trigger that arrives
with the vehicle it is about, and the only one a caller asks for by name, so it is
also the only one with an endpoint of its own —
``POST /scenarios/{id}/vehicles/{vehicle_id}/avoid-road``.

It is not a hole in the rule above, and the reason is worth being exact about. An
override is filed as a **real incident** before anything is re-planned: the road
is priced, the report lands in the scenario's own incident list, and it can be
reverted like any other. So the driver has changed the network rather than
asserted a justification, and :func:`detect_trigger` — which is untouched and
knows nothing about overrides — would reach the same verdict on the next call
through the incident branch it already has. What the override *adds* is scope: it
re-plans one vehicle rather than the fleet, because that is what the driver asked
about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol, Sequence

from qgati.api.store import StoredScenario
from qgati.fleet.watcher import Reading

__all__ = [
    "ANOMALY",
    "INCIDENT",
    "OVERRIDE",
    "FlagSource",
    "Trigger",
    "detect_trigger",
    "incident_edges",
    "override_trigger",
]

#: A reported incident is live on the network.
INCIDENT = "incident"
#: The fleet's own measurement was judged anomalous by the detector.
ANOMALY = "anomaly"
#: A driver asked to be routed around a road they named. The only trigger a
#: caller supplies rather than derives, and the only one scoped to one vehicle.
OVERRIDE = "override"

Edge = tuple[Hashable, Hashable]


class FlagSource(Protocol):
    """Anything that can say what the fleet last saw flagged.

    A protocol rather than the watcher itself, so the trigger rule can be tested
    against a flag without a fleet, a store, a graph or a tick — which matters
    because manufacturing a real anomaly on demand is not something this project
    can do: at ``NOISE_SIGMA = 0.06`` the fleet's own noise trips the detector's
    fallback about once in a thousand readings, and that is deliberate.
    """

    @property
    def last_flags(self) -> tuple[Reading, ...]: ...


@dataclass(frozen=True, slots=True)
class Trigger:
    """Why a re-optimization is allowed to run, and on what evidence.

    ``kinds`` is ordered most-authoritative first and holds one or more of
    :data:`INCIDENT`, :data:`ANOMALY` and :data:`OVERRIDE`. It is a tuple rather
    than a single value because both of the derived kinds holding is the ordinary
    case, not an edge case, and reporting only the incident would throw away the
    measurement that confirms it.

    :data:`OVERRIDE` is never reported alongside them. A driver's report is one
    fact — somebody, about one road, said so — and listing the incident it also
    filed would count the same fact twice. The incident is on the response's own
    ``incident`` field, where it can also be reverted from.

    ``edges`` is the union of what each kind points at — the reported roads and
    the measured ones — which is what a caller highlights on a map to show where
    the trouble is.
    """

    kinds: tuple[str, ...]
    detail: str
    edges: tuple[Edge, ...] = ()
    #: The detector's own explanations, when an anomaly is among the kinds. Passed
    #: through verbatim rather than summarised: it names the numbers it judged.
    reasons: tuple[str, ...] = ()

    @property
    def primary(self) -> str:
        """The leading justification — what the re-plan is chiefly reacting to."""
        return self.kinds[0]

    def to_dict(self) -> dict:
        return {
            "kinds": list(self.kinds),
            "primary": self.primary,
            "detail": self.detail,
            "edges": [{"u": u, "v": v} for u, v in self.edges],
            "reasons": list(self.reasons),
        }


def override_trigger(vehicle_id: str, edge: Edge, treatment: str) -> Trigger:
    """The trigger for a driver's own report, about their own road.

    Built rather than detected, and deliberately kept here next to
    :func:`detect_trigger` so the two read as the two halves of one rule: that
    function answers "has something happened?", and this one records what a driver
    said when the answer would otherwise have been no.

    The detail names the vehicle, the road and the treatment, because all three are
    things the caller supplied and none of them is recoverable from the scenario
    afterwards — the incident records the road and the word, but not who asked.
    """
    return Trigger(
        kinds=(OVERRIDE,),
        detail=(
            f"vehicle {vehicle_id} reported the road {edge[0]!r} -> {edge[1]!r} as "
            f"{treatment}, and asked to be routed around it"
        ),
        edges=(edge,),
    )


def detect_trigger(
    record: StoredScenario, flags: FlagSource
) -> Trigger | None:
    """What has happened to this scenario that warrants re-planning, if anything.

    ``None`` is the answer that matters: it is what the route turns into a 409,
    and it is the only thing standing between this endpoint and a re-solve button.
    """
    kinds: list[str] = []
    edges: list[Edge] = []
    parts: list[str] = []

    if record.incidents:
        reported = tuple((incident.u, incident.v) for incident in record.incidents)
        kinds.append(INCIDENT)
        edges.extend(reported)
        named = sorted({incident.incident_type for incident in record.incidents})
        parts.append(
            f"{len(record.incidents)} live incident(s) on the network "
            f"({', '.join(named)})"
        )

    flagged = flags.last_flags
    reasons: tuple[str, ...] = ()
    if flagged:
        measured = tuple(
            reading.edge for reading in flagged if reading.edge is not None
        )
        reasons = tuple(
            reading.verdict.reason
            for reading in flagged
            if reading.verdict is not None
        )
        kinds.append(ANOMALY)
        edges.extend(measured)
        parts.append(
            f"the fleet's most recent tick flagged {len(flagged)} reading(s) as "
            "anomalous"
        )

    if not kinds:
        return None

    # Deduplicated in order: a road can be both reported and measured, and
    # listing it twice would suggest two roads.
    unique_edges = tuple(dict.fromkeys(edges))
    return Trigger(
        kinds=tuple(kinds),
        detail="; ".join(parts),
        edges=unique_edges,
        reasons=reasons,
    )


def incident_edges(record: StoredScenario) -> Sequence[Edge]:
    """Just the reported roads, for a caller that needs them on their own."""
    return tuple((incident.u, incident.v) for incident in record.incidents)
