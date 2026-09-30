"""Why a re-plan looks the way it does, in sentences built from its own numbers.

A re-optimization returns two plans and a pile of figures, and none of it says
*why* the fleet moved. An operator can see that Vehicle 2 lost two of its stops
and cannot see what that did to its cost, whether it is now late somewhere, or
what the road closure actually cost in seconds. This module writes that down.

The rule it is built on
-----------------------
**Every number in a statement is one the caller computed.** Not a template with
a plausible figure slotted in — the figure and the sentence are produced in the
same branch, and the figure is carried out alongside the sentence in
:class:`Figure` so a reader can check the arithmetic without reading this file.
That is what makes "never fabricated" a property of the structure rather than a
promise in a docstring: there is no code path here that can emit a number it was
not handed.

The corollary is that a statement is **omitted** when its numbers are missing,
rather than emitted with a placeholder. A closure has no travel time — it is
``math.inf`` in the matrix — so no statement here says "the delay on the closed
road is N minutes". It says the road is impassable, because that is what is true,
and quotes the seconds it cost *before* the report, which is a number the caller
supplies. A zero cost delta produces no sentence at all, because "cost changed by
nothing" is not a reason for anything.

The claim this module is most careful about
-------------------------------------------
A re-plan moves work *between* vehicles, so a vehicle's cost before and its cost
after are usually the cost of **two different jobs**. Reporting "V2's cost fell
₹412" when V2 also handed away two stops would be the classic re-optimization
lie: the fall is less work, not a better route. So the two are compared only when
the stop count is unchanged, and the wording says which comparison it is making.
Everything else is reported as a change of *share* — stops and cost together,
with no claim that one plan beat the other.

Where the vehicles come from
----------------------------
``before`` and ``after`` are two different fleets: the before half covers every
vehicle the fleet reports, the after half only those that could still work. They
are joined on ``vehicle_id``, which is the same string on both sides, and a
vehicle that appears in only one of them is simply not explained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from qgati.optimizer.fitness import Evaluation, route_metrics
from qgati.reopt.plan import ReoptPlan
from qgati.reopt.triggers import Trigger

__all__ = [
    "MAX_STATEMENTS",
    "Figure",
    "ReplanExplanation",
    "RoadDelay",
    "RoadReport",
    "RouteExplanation",
    "Statement",
    "explain_replan",
]

#: The most statements any one vehicle gets. Three is what a person reads; a
#: fourth starts being a report nobody finishes.
MAX_STATEMENTS = 3

#: Below this, two figures are the same figure. Rupees and seconds both arrive
#: through floating-point arithmetic on the way here, and a delta of 1e-9 rupees
#: is not a change worth a sentence.
_EPSILON = 1e-6

#: Minutes per second, for the two places a duration reads better in minutes.
_PER_MINUTE = 60.0


@dataclass(frozen=True, slots=True)
class Figure:
    """One number a statement was built from, and what it is.

    ``unit`` is a bare word — ``"rupees"``, ``"seconds"``, ``"stops"`` — not a
    formatted string. Formatting belongs to whoever is rendering, and a figure
    that arrived pre-formatted could not be plotted, sorted or asserted against.
    """

    label: str
    value: float
    unit: str


@dataclass(frozen=True, slots=True)
class Statement:
    """One sentence, and the numbers it is made of.

    :attr:`figures` is not decoration. It is the receipt: a test can assert that
    every figure quoted in :attr:`text` appears here, and a reader can check the
    arithmetic without trusting the prose.
    """

    text: str
    figures: tuple[Figure, ...]


@dataclass(frozen=True, slots=True)
class RouteExplanation:
    """One vehicle's part in a re-plan."""

    vehicle_id: str
    #: One line, always present, saying what this vehicle's position in the new
    #: plan is — so a panel can show a summary without expanding anything.
    headline: str
    statements: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class RoadDelay:
    """A reported road, priced before the report and after it.

    Supplied by the caller rather than measured here, because measuring it needs
    the graph, the traffic state and the incident list — none of which belong in
    a module that is otherwise pure arithmetic.

    ``after_seconds`` is ``None`` when the road is now **closed**. That is not a
    missing value to be worked around: a closed road is impassable, routing
    charges it infinity, and there is no travel time for it to have.
    """

    u: Hashable
    v: Hashable
    before_seconds: float | None
    after_seconds: float | None


@dataclass(frozen=True, slots=True)
class RoadReport:
    """What one reported road did to the network."""

    u: Hashable
    v: Hashable
    statements: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class ReplanExplanation:
    """The whole decision trace: what happened, road by road, then vehicle by vehicle."""

    #: The trigger's own sentence, passed through verbatim.
    headline: str
    roads: tuple[RoadReport, ...]
    routes: tuple[RouteExplanation, ...]


def explain_road(delay: RoadDelay) -> RoadReport:
    """What one reported road costs now, against what it cost before the report.

    Three cases, and the wording of each is decided by what is actually known:

    * **closed** — no after-time exists. The statement says impassable and quotes
      the before-time, and never invents the delay a closed road is not having.
    * **priced both ways** — the real before/after seconds and their ratio. This
      is the only case that can honestly say "delay increased to Z minutes".
    * **was impassable, now is not** — reachable when an incident is reverted and
      a creation-time condition underneath it is revealed. Reported as what it
      is rather than as a negative delay.
    """
    edge = f"{delay.u!r} → {delay.v!r}"

    if delay.after_seconds is None:
        # No figure is invented for the "closed" state itself. A closure is an
        # absence of a travel time, and a receipt saying so would be a number
        # standing in for a thing that has none. What is quoted is the seconds the
        # road cost *before* the report, which is known and is what makes the
        # closure mean anything.
        if delay.before_seconds is None:
            text = (
                f"Road {edge} is closed — impassable, so it has no travel time at "
                "all. It was already impassable before the report."
            )
            figures: tuple[Figure, ...] = ()
        else:
            text = (
                f"Road {edge} took {delay.before_seconds:.0f} s before the report "
                f"and is closed now — impassable, so it has no travel time at all."
            )
            figures = (Figure("before", delay.before_seconds, "seconds"),)
        return RoadReport(
            u=delay.u, v=delay.v, statements=(Statement(text=text, figures=figures),)
        )

    if delay.before_seconds is None:
        return RoadReport(
            u=delay.u,
            v=delay.v,
            statements=(
                Statement(
                    text=(
                        f"Road {edge} was impassable and is now open, at "
                        f"{delay.after_seconds:.0f} s."
                    ),
                    figures=(Figure("after", delay.after_seconds, "seconds"),),
                ),
            ),
        )

    delta = delay.after_seconds - delay.before_seconds
    figures = [
        Figure("before", delay.before_seconds, "seconds"),
        Figure("after", delay.after_seconds, "seconds"),
        Figure("delay", delta, "seconds"),
    ]
    if delay.before_seconds > _EPSILON:
        ratio = delay.after_seconds / delay.before_seconds
        figures.append(Figure("slower by", ratio, "multiple"))
        text = (
            f"Road {edge} took {delay.before_seconds:.0f} s before the report and "
            f"takes {delay.after_seconds:.0f} s now — a delay of "
            f"{delta / _PER_MINUTE:.1f} min, {ratio:.1f}× the original."
        )
    else:
        # A road with no before-time is one the router never charged anything
        # for; quoting a ratio against zero would be a division dressed up as a
        # measurement.
        text = (
            f"Road {edge} takes {delay.after_seconds:.0f} s now, up "
            f"{delta:.0f} s on the report's arrival."
        )
    return RoadReport(
        u=delay.u, v=delay.v, statements=(Statement(text=text, figures=tuple(figures)),)
    )


def explain_replan(
    plan: ReoptPlan,
    before: Evaluation,
    trigger: Trigger,
    road_delays: Sequence[RoadDelay] = (),
    before_positions: Mapping[str, int] | None = None,
) -> ReplanExplanation:
    """The decision trace for one re-optimization.

    ``before`` must be the evaluation of the plan the fleet was already driving —
    the same object ``POST /reoptimize`` prices to build its ``before`` half —
    because every comparison here is against it.

    By default it is assumed to be aligned to ``plan.progress``: position ``i`` of
    its per-route arrays is ``plan.progress[i]``'s vehicle. That holds wherever the
    before half was priced over the same fleet the plan was built for. It does
    **not** hold for ``POST .../vehicles/{id}/avoid-road``, which solves a
    one-vehicle instance while pricing its before half over the whole fleet; there
    the caller passes ``before_positions`` to say where each re-planned vehicle
    sits in the array it is handing over. Getting this wrong would compare one
    vehicle's cost against another's, which is the kind of error that reads as a
    plausible number rather than as a failure, so the override exists rather than
    a second evaluation being built to avoid needing it.

    ``plan.evaluation`` is aligned to ``plan.active`` instead, and the two halves
    are joined on ``vehicle_id``.

    Vehicles that appear in only one of them are skipped rather than guessed at: a
    vehicle with nothing left to do has no after-route to explain, and a vehicle
    that has just joined the fleet has no before-route to compare against.
    """
    # vehicle_id -> its position in the *before* fleet's route array. Built from
    # `progress` rather than `active`, because `before` covers every vehicle the
    # fleet reported, not only the ones that could still work.
    positions = (
        dict(before_positions)
        if before_positions is not None
        else {item.vehicle_id: position for position, item in enumerate(plan.progress)}
    )
    # Looked up by id rather than by position: `active` is a filtered subset of
    # `progress`, so an index into one is not an index into the other as soon as a
    # single vehicle drops out. Reading the wrong vehicle's stop count here would
    # produce a sentence about work that is not this vehicle's.
    by_id = {item.vehicle_id: item for item in plan.progress}

    routes: list[RouteExplanation] = []
    for position, item in enumerate(plan.active):
        before_position = positions.get(item.vehicle_id)
        if before_position is None:  # pragma: no cover - active is drawn from progress
            continue
        if before_position >= len(before.route_costs):  # pragma: no cover - same fleet
            continue
        routes.append(
            _explain_route(
                plan,
                before,
                before_position=before_position,
                after_position=position,
                vehicle_id=item.vehicle_id,
                before_stops=by_id[item.vehicle_id].remaining,
            )
        )

    return ReplanExplanation(
        headline=trigger.detail,
        roads=tuple(explain_road(delay) for delay in road_delays),
        routes=tuple(routes),
    )


def _explain_route(
    plan: ReoptPlan,
    before: Evaluation,
    *,
    before_position: int,
    after_position: int,
    vehicle_id: str,
    before_stops: Sequence[int],
) -> RouteExplanation:
    """One vehicle's statements, in the order a reader wants them.

    Built by trying each generator in turn and keeping what fires, capped at
    :data:`MAX_STATEMENTS`. Order is by how much the statement explains: what
    changed hands, then what that did to the cost, then anything now owed for
    lateness.
    """
    route_after = (
        plan.solution.routes[after_position]
        if after_position < len(plan.solution.routes)
        else ()
    )
    statements: list[Statement] = []

    moved = _moved_statement(plan, vehicle_id, len(before_stops))
    if moved is not None:
        statements.append(moved)

    share = _share_statement(
        before,
        plan.evaluation,
        before_position=before_position,
        after_position=after_position,
        before_stops=len(before_stops),
        after_stops=len(route_after),
    )
    if share is not None:
        statements.append(share)

    late = _lateness_statement(plan, route_after, after_position)
    if late is not None:
        statements.append(late)

    statements = statements[:MAX_STATEMENTS]
    if not statements:
        # Nothing changed and nothing is owed. That is the re-plan's answer — the
        # fleet was handed its real positions and the current assignment came back
        # cheapest — and it is worth one sentence rather than an empty panel.
        statements = [
            Statement(
                text=(
                    "Kept its assignment: re-solved from this vehicle's real "
                    "position, the current order of its remaining stops is still "
                    "the cheapest one found."
                ),
                figures=(Figure("stops kept", float(len(before_stops)), "stops"),),
            )
        ]

    return RouteExplanation(
        vehicle_id=vehicle_id,
        headline=_headline(route_after, before_stops=len(before_stops)),
        statements=tuple(statements),
    )


def _headline(route_after: Sequence[int], *, before_stops: int) -> str:
    """The one-line summary, true whatever the statements below it turned out to be."""
    if len(route_after) == before_stops:
        return f"{len(route_after)} unserved stop(s), re-ordered"
    if len(route_after) > before_stops:
        return (
            f"{len(route_after)} unserved stop(s), {len(route_after) - before_stops} "
            "taken on"
        )
    return (
        f"{len(route_after)} unserved stop(s), {before_stops - len(route_after)} "
        "handed over"
    )


def _moved_statement(
    plan: ReoptPlan, vehicle_id: str, before_stops: int
) -> Statement | None:
    """What changed hands, from the moves the re-plan recorded.

    Read off ``plan.moved`` rather than inferred from the two routes, because that
    tuple *is* the record of the comparison — recomputing it here would be a
    second opinion that could disagree with the one the API reports.
    """
    out = [move for move in plan.moved if move.from_vehicle == vehicle_id]
    into = [move for move in plan.moved if move.to_vehicle == vehicle_id]
    if not out and not into:
        return None

    parts: list[str] = []
    figures: list[Figure] = []
    if out:
        targets = sorted({move.to_vehicle or "nowhere" for move in out})
        parts.append(
            f"{len(out)} of its {before_stops} unserved stop(s) went to "
            + _join(targets)
        )
        figures.append(Figure("stops moved out", float(len(out)), "stops"))
        figures.append(Figure("stops before", float(before_stops), "stops"))
    if into:
        sources = sorted({move.from_vehicle for move in into})
        parts.append(f"it took on {len(into)} from " + _join(sources))
        figures.append(Figure("stops moved in", float(len(into)), "stops"))

    return Statement(
        text="Re-assigned: " + "; ".join(parts) + ".", figures=tuple(figures)
    )


def _share_statement(
    before: Evaluation,
    after: Evaluation,
    *,
    before_position: int,
    after_position: int,
    before_stops: int,
    after_stops: int,
) -> Statement | None:
    """What this vehicle's part of the work costs now, against what it cost then.

    The comparison is only called *like for like* when the vehicle is carrying the
    same number of stops, because that is the only case where the two costs are
    prices for the same job. A vehicle that handed work away is cheaper for the
    obvious reason, and saying so as though it had found a better route would be
    the single most misleading sentence this module could write.
    """
    if before_position >= len(before.route_costs):
        return None
    if after_position >= len(after.route_costs):
        return None

    cost_before = before.route_costs[before_position]
    cost_after = after.route_costs[after_position]
    delta = cost_after - cost_before
    if abs(delta) <= _EPSILON:
        return None

    # A noun, because both sentences below put it after "a ... of ₹N". A verb
    # reads as "₹412 fell", which is not a sentence about the money.
    direction = "fall" if delta < 0 else "rise"
    figures = [
        Figure("cost before", cost_before, "rupees"),
        Figure("cost after", cost_after, "rupees"),
        Figure("change", delta, "rupees"),
    ]

    if before_stops == after_stops:
        text = (
            f"Its {after_stops} remaining stop(s) cost ₹{cost_before:.0f} on the "
            f"plan it was driving and ₹{cost_after:.0f} re-planned — a {direction} "
            f"of ₹{abs(delta):.0f}"
        )
        if cost_before > _EPSILON:
            percent = delta / cost_before * 100.0
            figures.append(Figure("change percent", percent, "percent"))
            text += f" ({abs(percent):.0f}%)"
        return Statement(text=text + ".", figures=tuple(figures))

    # Different jobs, so the sentence reports a change of share and says outright
    # that it is not a comparison — because a reader who took the two totals as
    # comparable would read a smaller workload as a better route.
    return Statement(
        text=(
            f"Its share of the remaining work went from {before_stops} stop(s) at "
            f"₹{cost_before:.0f} to {after_stops} at ₹{cost_after:.0f} — a "
            f"{direction} of ₹{abs(delta):.0f}. The two totals cover different "
            "amounts of work, so this is a change of share, not a saving."
        ),
        figures=tuple(figures),
    )


def _lateness_statement(
    plan: ReoptPlan, route_after: Sequence[int], after_position: int
) -> Statement | None:
    """Any window the new route misses, and what missing it costs.

    Re-metrics the route rather than reading the evaluation's window penalty
    alone, because the penalty is a rupee total and the useful fact is *which*
    stop is late and by how much. ``route_metrics`` is the same function the
    evaluation priced the route with, so the penalty quoted here is the penalty
    the solver was charged — not a second opinion about it.
    """
    if not route_after:
        return None
    if after_position >= len(plan.evaluation.route_window_penalties):
        return None
    penalty = plan.evaluation.route_window_penalties[after_position]
    if penalty <= _EPSILON:
        return None

    metrics = route_metrics(route_after, plan.instance, plan.matrix, after_position)
    windows = plan.instance.windows
    ids = plan.instance.delivery_ids()

    worst: tuple[str, float] | None = None
    for position, stop in enumerate(route_after):
        # `windows` and `ids` are indexed by *delivery* index while
        # `arrival_times` is indexed by *position in the route*, which is why the
        # stop is looked up rather than the position. The guard is on the stop,
        # for the same reason.
        if position >= len(metrics.arrival_times) or stop >= len(windows):
            break
        latest = windows[stop][1]
        if latest is None:
            continue
        late = metrics.arrival_times[position] - latest
        if late <= _EPSILON:
            continue
        if worst is None or late > worst[1]:
            worst = (ids[stop], late)

    figures = [Figure("window penalty", penalty, "rupees")]
    if worst is None:  # pragma: no cover - a penalty implies a late arrival
        text = (
            f"Its new route misses a delivery window, which the objective prices "
            f"at ₹{penalty:.0f}."
        )
    else:
        delivery_id, late = worst
        figures.insert(0, Figure("late by", late, "seconds"))
        text = (
            f"Its new route reaches {delivery_id} {late / _PER_MINUTE:.1f} min after "
            f"its window closes — priced at ₹{penalty:.0f}."
        )
    return Statement(text=text, figures=tuple(figures))


def _join(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` — an Oxford-comma-free list, as prose."""
    items = list(names)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"
