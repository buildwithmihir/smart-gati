"""Where each vehicle is, and what it has already delivered.

The input half of a re-optimization, and the half that has to be *derived*
rather than declared. Nothing in this project records a delivery being made: the
scenario describes what was ordered, not what has been handed over, and the fleet
watcher tracks vehicles by the seconds they have travelled rather than by the
stops they have cleared. So the question "what is left to do?" is answered by
reading each vehicle's own corridor against how far along it has got — see
:func:`~qgati.fleet.pings.completed_stops`, which is where that lives.

Two things this module deliberately does not do. It does not touch the store or
the solver, so the whole of it is testable on a hand-built scenario and a
hand-built track. And it does not decide what happens to a vehicle that is stuck:
it reports one, and :mod:`qgati.reopt.plan` decides that its undelivered stops go
into the pool for someone else.

The distinction that carries the most weight here is between a vehicle that has
**finished** and one that is **stuck**, because the two are one ``None`` from
:func:`~qgati.fleet.pings.position`. A finished vehicle has nothing left to
deliver and is simply done. A stuck one is stopped behind a closed road with its
remaining load still on board, and those deliveries are exactly the ones a
re-optimization exists to hand to somebody else. Conflating them would either
strand a load or invent work for a vehicle that has gone home.

Both are also answered by :func:`restart_node`, which is the one place that
decides where a vehicle can begin a *new* plan — and, for a stuck one, the only
place that can, since a vehicle behind a closure is otherwise known to be stuck
and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from qgati.fleet.pings import (
    CostOf,
    VehicleTrack,
    blocked_edge,
    completed_stops,
    position,
)
from qgati.optimizer.models import Scenario

__all__ = ["VehicleProgress", "fleet_progress", "restart_node"]

Node = Hashable
Edge = tuple[Node, Node]


@dataclass(frozen=True, slots=True)
class VehicleProgress:
    """One vehicle, as of the last tick: where it is and what it has done.

    ``completed`` and ``remaining`` partition the route it was dispatched on, and
    between them they are the reason this type exists — the first half is what a
    re-optimization may never touch, and the second is the whole of what it is
    allowed to move.

    ``node`` is where the vehicle would begin a new plan: the far end of the road
    it is currently on, because a cost matrix is indexed by intersections and a
    vehicle halfway along a road is not at one. Choosing the *next* intersection
    rather than the last one passed keeps a re-planned route from starting with a
    leg the vehicle has already driven. :func:`restart_node` is that rule, and
    the one exception to it.

    ``edge`` is the road that claim is about. Without it a stuck vehicle is only
    known to be stuck — ``node`` is ``None`` and there is no record of *where*,
    which is precisely what a driver asking to be routed around their own road
    needs to be answered with.
    """

    vehicle_id: str
    vehicle_index: int
    node: Node | None
    completed: tuple[int, ...]
    remaining: tuple[int, ...]
    #: Seconds since it left the depot, which is the clock its windows are quoted
    #: against.
    elapsed: float
    #: ``capacity`` less the demand it has already dropped. This is what actually
    #: frees up as a vehicle works through its route, and what bounds anything a
    #: re-optimization adds to it.
    remaining_capacity: float
    stuck: bool = False
    #: The directed road it is on, or — when it is stuck — the road ahead it
    #: cannot drive. ``None`` when it is on no road at all: never dispatched, or
    #: finished.
    edge: Edge | None = None

    @property
    def finished(self) -> bool:
        """True when it has delivered everything it was dispatched with."""
        return not self.remaining and not self.stuck

    @property
    def available(self) -> bool:
        """True when it can be given more work: something left to serve, and moving.

        A stuck vehicle fails this and its load is redistributed, which is the
        point of noticing it at all.
        """
        return bool(self.remaining) and not self.stuck


def fleet_progress(
    scenario: Scenario,
    tracks: tuple[VehicleTrack, ...],
    cost_of: CostOf,
) -> tuple[VehicleProgress, ...]:
    """Read every vehicle's position and progress off its corridor.

    One entry per vehicle **in scenario order**, including vehicles that were
    never dispatched, so the result lines up index-for-index with
    ``scenario.vehicles`` and with an :class:`~qgati.optimizer.models.Solution`.
    A vehicle with no track has no corridor to be on and nothing delivered, which
    is a fact about it rather than a gap in the answer.

    ``tracks`` rather than the watcher, so this is callable without one and
    testable without starting a thread. The caller is responsible for reading
    them consistently — :meth:`~qgati.fleet.watcher.ScenarioWatcher.read` is what
    holds the lock while this runs.
    """
    by_vehicle = {track.vehicle_id: track for track in tracks}
    progress: list[VehicleProgress] = []

    for index, vehicle in enumerate(scenario.vehicles):
        track = by_vehicle.get(vehicle.id)
        if track is None:
            progress.append(
                VehicleProgress(
                    vehicle_id=vehicle.id,
                    vehicle_index=index,
                    node=None,
                    completed=(),
                    remaining=(),
                    elapsed=0.0,
                    remaining_capacity=vehicle.capacity,
                )
            )
            continue

        served = completed_stops(track, cost_of)
        route = track.delivery_indices
        completed, remaining = tuple(route[:served]), tuple(route[served:])

        delivered = sum(scenario.deliveries[stop].demand for stop in completed)
        here = position(track, cost_of)
        blocked = blocked_edge(track, cost_of) if here is None else None

        progress.append(
            VehicleProgress(
                vehicle_id=vehicle.id,
                vehicle_index=index,
                # A stuck vehicle stays where it is; a finished one has no next
                # intersection to start from, and neither is in the new fleet.
                node=None if here is None else here.edge[1],
                completed=completed,
                remaining=remaining,
                elapsed=track.travelled,
                remaining_capacity=vehicle.capacity - delivered,
                stuck=blocked is not None,
                # The road it is on, or the one it cannot get past — which is the
                # only record of where a stuck vehicle actually is.
                edge=here.edge if here is not None else blocked,
            )
        )

    return tuple(progress)


def restart_node(progress: VehicleProgress, avoided: Edge | None = None) -> Node | None:
    """The intersection this vehicle begins a new plan from.

    Normally the far end of the road it is on — the next intersection it reaches,
    which is the same choice :attr:`VehicleProgress.node` already records, and the
    one that keeps a re-planned route from starting with a leg already driven.

    The exception is reachability, and it is the same exception in both of the
    cases it covers. A route is a sequence of roads, so a plan cannot begin on the
    far side of one the vehicle cannot drive: it would be dispatched through a
    road that is shut. Where that holds, the answer is the **near** end — the last
    intersection the vehicle can be said to have reached.

    ``avoided`` is the road being given up, and the caller passes it only when the
    treatment closes it. Reporting a road as *slow* leaves it drivable, so the
    vehicle will still get to the far end of it and the ordinary rule stands; it is
    a closure — or a road already closed under the vehicle — that moves the start
    back. Keeping that decision at the call site is what stops this function from
    being a statement about incident vocabulary rather than about reachability.

    ``None`` means there is nowhere to begin: a vehicle that was never dispatched,
    or one that has finished. Both are refusals rather than plans.
    """
    if progress.edge is None:
        return progress.node
    if progress.node is None or progress.edge == avoided:
        return progress.edge[0]
    return progress.node
