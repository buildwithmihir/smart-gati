"""Adaptive partial re-optimization when traffic or orders change mid-plan.

Every plan this project produces is **static**. ``POST /optimize`` solves a
scenario once and hands back routes; the fleet watcher then drives those routes
while traffic moves underneath it — roads close, measurements overwrite
estimates — and the routes never change. A vehicle stays committed to a plan
chosen under costs that may no longer exist. This package is where the fleet gets
to react.

Given a scenario, where each vehicle currently is, which stops it has already
served, and what is actually happening on the network, it re-solves **only the
remaining unserved deliveries** and returns new assignments — each affected
vehicle starting from where it is rather than from the depot.

Four modules, split the way :mod:`qgati.traffic` and :mod:`qgati.fleet` split:

:mod:`~qgati.reopt.state`
    Where each vehicle is and what it has already delivered, derived from the
    running fleet's own corridors. No I/O and no solver, so the whole of it is
    testable on a hand-built scenario and a hand-built track.
:mod:`~qgati.reopt.triggers`
    What justifies re-planning at all — an incident, a flagged reading from the
    fleet's last tick, or a driver reporting their own road. ``None`` when nothing
    has happened, which is what the route turns into a 409.
:mod:`~qgati.reopt.plan`
    Builds the remaining-stops-only instance and solves it with the same QPSO,
    the same objective and the same windows every other plan in this project goes
    through.
:mod:`~qgati.reopt.explain`
    Why the result looks the way it does, in sentences built only from numbers the
    plan already computed. Pure arithmetic over a :class:`ReoptPlan` and the
    evaluation of the plan it replaced — no graph, no solver, no I/O — so the
    claim that nothing in it is invented is enforced by what it is able to see
    rather than by review.

The invariant that matters most
-------------------------------
**A completed stop can never be reassigned.** Not because something filters
completed deliveries out of an answer, but because they are not in the instance
that is solved — so no solver can name one however badly it searches. The
guarantee is structural, so it holds for all five solvers rather than for the ones
somebody remembered to check.

The boundary this package does not cross
----------------------------------------
Re-optimizing is a **read**. It computes a plan, returns it, and writes nothing —
it does not rewrite the stored scenario and it does not re-dispatch the fleet.
Rewriting the stored scenario would make a second call behave differently from
the first for no reason a caller asked for, and re-dispatching means rebuilding
tracks and resetting how far each vehicle has travelled, which is a simulation
decision rather than an optimizer one. Applying a plan — handing the new routes
to the fleet — is the obvious next step and it is named as the boundary rather
than left ambiguous.

That stays true of this package, and it is worth being precise about where the one
exception lives. ``POST /scenarios/{id}/vehicles/{vehicle_id}/avoid-road`` **does**
write — it files the driver's report as a real incident, through the same code the
incident routes use. The write is the report, not the re-plan, and the route owns
it the way ``POST /incident`` owns its own; nothing in here gained an I/O
dependency to make that work.

    >>> from qgati.reopt import fleet_progress, detect_trigger, reoptimize
    >>> progress = watcher.read(lambda t: fleet_progress(scenario, t, lookup))
    >>> trigger = detect_trigger(record, watcher)
    >>> if trigger is not None:
    ...     plan = reoptimize(scenario=scenario, progress=progress,
    ...                       graph=graph, weight=weight)
"""

from qgati.reopt.explain import (
    MAX_STATEMENTS,
    Figure,
    ReplanExplanation,
    RoadDelay,
    RoadReport,
    RouteExplanation,
    Statement,
    explain_replan,
    explain_road,
)
from qgati.reopt.plan import (
    Move,
    NothingToReplan,
    ReoptPlan,
    ReoptRefused,
    SolverTooSmall,
    before_view,
    reoptimize,
)
from qgati.reopt.state import VehicleProgress, fleet_progress, restart_node
from qgati.reopt.triggers import (
    ANOMALY,
    INCIDENT,
    OVERRIDE,
    FlagSource,
    Trigger,
    detect_trigger,
    incident_edges,
    override_trigger,
)

__all__ = [
    "ANOMALY",
    "INCIDENT",
    "MAX_STATEMENTS",
    "OVERRIDE",
    "Figure",
    "FlagSource",
    "Move",
    "NothingToReplan",
    "ReoptPlan",
    "ReoptRefused",
    "ReplanExplanation",
    "RoadDelay",
    "RoadReport",
    "RouteExplanation",
    "SolverTooSmall",
    "Statement",
    "Trigger",
    "VehicleProgress",
    "before_view",
    "detect_trigger",
    "explain_replan",
    "explain_road",
    "fleet_progress",
    "incident_edges",
    "override_trigger",
    "reoptimize",
    "restart_node",
]
