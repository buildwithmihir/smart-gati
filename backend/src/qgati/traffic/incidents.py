"""Live incidents: an operator's report, applied to one road of one scenario.

A scenario's congestion state is derived from its timestamp and cannot be set,
because nothing in the data could imply an incident and nothing else *should* be
settable — see :mod:`~qgati.traffic.simulator`. An incident is the one thing that
genuinely arrives from outside the model: a road shuts, or crawls, some time
after the scenario was priced. This module is where such a report is named,
carried, and folded onto the conditions a scenario already has.

The vocabulary, and why it is not the simulator's
-------------------------------------------------
An incident is named by what the operator **reported**: ``"closure"`` or
``"slow"``. Conditions are stored by their **effect**: an edge is in
``closed_edges`` (impassable) or in the flat-x2.9 bucket. The two vocabularies
meet in exactly one place, :func:`conditions_for`, and nowhere else.

They are kept apart deliberately. The effect names in
:mod:`~qgati.traffic.simulator` (``ROAD_CLOSURE``, ``ACCIDENT``) predate this
module and are already written into the ``traffic_log`` table, which is an
accumulating dataset for a later phase. Renaming them would split that history
across two spellings of the same state. Adding the report's own word alongside
costs nothing and keeps the audit trail truthful: a road reported *slow* is
recorded as slow, not as an accident it may never have had.

The multipliers are not redefined here. :data:`INCIDENT_MULTIPLIERS` reads them
off the simulator, so the locked set in ``DESIGN_DECISIONS.md`` has one home.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

from qgati.traffic.simulator import PEAK_FACTOR, ActiveConditions

__all__ = [
    "CLEARED",
    "CLOSURE",
    "INCIDENT_MULTIPLIERS",
    "INCIDENT_TYPES",
    "SLOW",
    "Incident",
    "conditions_for",
]

Node = Hashable

#: The two things an operator can report about one road, and the values
#: ``POST /scenarios/{id}/incident`` accepts.
#:
#: ``DESIGN_DECISIONS.md`` calls these "blocked" and "slow". "Closure" is the
#: same first report under the name the API uses; nothing else differs.
CLOSURE = "closure"
SLOW = "slow"

#: What a *revert* of an incident writes to the log's ``incident_type`` column.
#: Not an incident type — it cannot be posted — because it reports a road going
#: back to normal, which is the absence of one. It exists so an audit trail can
#: say when an override was lifted rather than only that some row was written.
CLEARED = "road_clear"

#: Reportable incident types, in the order the API documents them.
INCIDENT_TYPES: tuple[str, ...] = (CLOSURE, SLOW)

#: Report -> the multiplier it applies to a road's base travel time.
#:
#: Read off the simulator rather than restated, so the locked set lives in one
#: place. ``inf`` is the module-wide encoding of "impassable" and is what routing
#: multiplies a closed road's time by.
INCIDENT_MULTIPLIERS: Mapping[str, float] = {
    CLOSURE: math.inf,
    SLOW: PEAK_FACTOR,
}


@dataclass(frozen=True, slots=True)
class Incident:
    """One operator report against one directed road, for one scenario.

    Frozen, so an incident can be stored in a scenario's tuple and compared by
    value without any risk of it being edited under a reader. Reverting one
    produces a new tuple with it removed, which is also what makes the store's
    compare-and-swap meaningful.

    ``created_at`` is when the operator made the change, as an ISO string in the
    same form :class:`~qgati.api.store.StoredScenario` uses for its own
    creation time. It is deliberately *not* the scenario's pricing timestamp:
    the two answer different questions, and the log row takes the pricing
    timestamp while this keeps the wall-clock moment.
    """

    incident_id: str
    incident_type: str
    u: Node
    v: Node
    created_at: str

    def __post_init__(self) -> None:
        if self.incident_type not in INCIDENT_TYPES:
            raise ValueError(
                f"incident_type must be one of {INCIDENT_TYPES}, "
                f"got {self.incident_type!r}"
            )

    @property
    def edge(self) -> tuple[Node, Node]:
        """The directed road this incident applies to."""
        return self.u, self.v

    @property
    def multiplier(self) -> float:
        """The factor this incident applies to its road's base travel time."""
        return INCIDENT_MULTIPLIERS[self.incident_type]

    def to_dict(self) -> dict:
        """JSON-ready form, matching :class:`~qgati.api.schemas.IncidentOut`."""
        return {
            "incident_id": self.incident_id,
            "incident_type": self.incident_type,
            "edge": {"u": self.u, "v": self.v},
            "created_at": self.created_at,
        }


def conditions_for(
    base: ActiveConditions, incidents: Iterable[Incident]
) -> ActiveConditions:
    """Fold live reports onto a scenario's creation-time conditions.

    ``base`` is what the scenario was created with and is never mutated, which is
    what lets a revert be exact: dropping an incident and re-folding reproduces
    the previous conditions precisely, including any accident or closure the
    scenario was created with. Storing only the *effective* set and subtracting
    from it would lose that — an edge that is both a creation-time closure and a
    live closure would keep one of them on revert, and nothing could tell which
    was meant.

    A closure outranks a slow report on the same edge, but only because
    :func:`~qgati.traffic.simulator.get_traffic_multiplier` tests the closed set
    first. Both buckets keep the edge, so reverting the closure reveals the
    still-standing slow report underneath rather than silently clearing it.
    """
    closed = set(base.closed_edges)
    slow = set(base.accident_edges)

    for incident in incidents:
        if incident.incident_type == CLOSURE:
            closed.add(incident.edge)
            continue
        # SLOW is the only other type: `Incident` validates against
        # INCIDENT_TYPES, so there is no third case to fall through to.
        slow.add(incident.edge)

    return ActiveConditions(
        accident_edges=frozenset(slow),
        closed_edges=frozenset(closed),
    )
