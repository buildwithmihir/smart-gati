"""A simulated GPS fleet, ticking in the background against one scenario.

There is no real fleet in this project, and a demo of dynamic routing needs one:
without vehicles reporting on roads, the traffic log only ever contains what the
model already knew — ``base_travel_time x multiplier``, both deterministic, and
therefore a table with no spread for a detector to detect from. This module
stands in for the fleet. Every tick it places each of a running scenario's
vehicles on the road it is currently on, draws a plausible travel time for that
road, and does the two things a real GPS ping would do with it:

1. **Judges it** against the road's logged history —
   :func:`~qgati.traffic.detection.detect` — so an anomaly is flagged.
2. **Applies it** through
   :func:`~qgati.traffic.observations.update_edge_from_observation`, which is
   Tier 1: a measured time overwrites the rule-based estimate for that road. The
   flat ``x2.9`` an incident applies is a placeholder for a delay nobody has
   measured yet, and it stops applying the moment somebody has.

What a tick does, in order
--------------------------
1. Read the scenario and its *effective* state — the creation-time conditions
   with any live incidents and any earlier readings folded in.
2. Advance every vehicle by one interval, then ask where it is under the current
   costs. A road that has slowed keeps a vehicle on it longer, which is the whole
   reason position is resolved in seconds rather than in edges.
3. Read the road's modelled time and draw a noisy measurement from it.
4. Judge the measurement, and fold it into the scenario's observations.
5. Re-price the scenario under the result, and write one log row per reading.
6. Write the new record back with a compare-and-swap.

Why a tick is a plain function
------------------------------
:meth:`ScenarioWatcher.tick` is called two ways — by the timer thread, and
directly by ``POST /scenarios/{id}/watcher/tick``. That is what makes the whole
feature testable without a single ``sleep``: every claim about what the fleet
does is asserted by calling ``tick()`` and reading the answer.

The three states a tick builds, and why not one
-----------------------------------------------
The measurement and the expectation must come from **observation-free** states,
or the fleet would be scored against its own last reading and would drift upward
by a random step every tick. Within that, there are two model numbers and they
answer different questions:

``live``
    The scenario's own conditions, incidents included and measurements excluded.
    This is what the vehicle is actually driving through, so it is what the
    measurement is drawn around. Measured from the incident-free model instead,
    a road reported slow would never read as slow.
``clean``
    Incidents cleared as well. This is what the detector compares against, and it
    is the same choice ``POST /scenarios/{id}/detect`` makes: if an operator's
    report could explain the slowness away, the fallback rule would be blind to
    the change it exists to catch.

Concurrency
-----------
One daemon thread per watched scenario, and one lock per watcher held across
every tick. The lock is what makes ``stop()`` mean something: it cannot return
while a tick is writing. The store's compare-and-swap — built for exactly this
shape of mutation, one that spends time re-pricing before it writes — is what
keeps a tick from discarding an incident that landed while it was working.
Losing that race is not an error: the readings were still taken, their log rows
were still written, and the next tick simply re-reads and tries again.
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Hashable, Iterable

import networkx as nx

from qgati.api.store import ScenarioChanged, ScenarioNotFound, ScenarioStore
from qgati.fleet.pings import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_SEED,
    DEFAULT_TIME_SCALE,
    VehicleTrack,
    blocked_edge,
    cost_lookup,
    noisy_reading,
    position,
)
from qgati.graph.cost_matrix import changed_entries
from qgati.optimizer.models import Solution
from qgati.traffic import (
    ActiveConditions,
    Detection,
    TrafficLogRow,
    TrafficLogStore,
    TrafficState,
    detect,
    edge_of,
    price_scenario,
    simulated_travel_time,
    traffic_weight_function,
    update_edge_from_observation,
)

__all__ = [
    "Reading",
    "ScenarioWatcher",
    "TickResult",
    "VehicleState",
    "WatcherExists",
    "WatcherRegistry",
    "WatcherState",
]

LOGGER = logging.getLogger(__name__)

Node = Hashable
Edge = tuple[Node, Node]


def _now() -> str:
    """The wall-clock moment, in the form every other ISO string here uses."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Reading:
    """One vehicle's ping, and everything that was made of it.

    A reading with ``edge`` of ``None`` is not a failure: it says the vehicle is
    not on a road it can report about, and ``note`` says which of the two reasons
    applies. A finished route and a road closed under the vehicle are both
    ordinary states, and neither is worth inventing a travel time for.
    """

    vehicle_id: str
    edge: Edge | None = None
    #: The road's time under the scenario's own conditions — what the fleet is
    #: driving through, and what the measurement was drawn around.
    modelled: float | None = None
    observed: float | None = None
    verdict: Detection | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class TickResult:
    """What one tick did."""

    scenario_id: str
    tick: int
    at: str
    readings: tuple[Reading, ...]
    #: Entries of the objective matrix that moved, counted the same way an
    #: incident counts them.
    changed_legs: int
    rows_logged: int
    conditions_mutated: bool

    @property
    def observed(self) -> tuple[Reading, ...]:
        """Just the readings that measured a road."""
        return tuple(reading for reading in self.readings if reading.observed is not None)

    @property
    def flagged(self) -> tuple[Reading, ...]:
        """Just the readings the detector flagged as anomalous."""
        return tuple(
            reading
            for reading in self.observed
            if reading.verdict is not None and reading.verdict.flagged
        )


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Where one vehicle is, as of the last tick."""

    vehicle_id: str
    edge: Edge | None
    #: Seconds left before it reaches the next intersection.
    remaining_seconds: float | None
    #: Fraction of the route's total cost consumed, under current costs.
    progress: float
    finished: bool


@dataclass(frozen=True, slots=True)
class WatcherState:
    """A watcher's outward-facing state — everything ``GET`` reports."""

    scenario_id: str
    solver: str
    running: bool
    interval_seconds: float
    time_scale: float
    seed: int | None
    ticks: int
    started_at: str
    last_tick_at: str | None
    vehicles: tuple[VehicleState, ...]


class WatcherExists(RuntimeError):
    """Raised when a scenario already has a watcher running."""


class ScenarioWatcher:
    """One scenario's simulated fleet.

    Constructed by the API's start route, which also solves the scenario: this
    object is handed routes, not a scenario to solve, so it does nothing on
    construction and only ever acts when ticked.
    """

    def __init__(
        self,
        *,
        scenario_id: str,
        solver: str,
        store: ScenarioStore,
        graph: nx.Graph,
        log_store: TrafficLogStore | None,
        tracks: Iterable[VehicleTrack],
        solution: Solution | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        time_scale: float = DEFAULT_TIME_SCALE,
        seed: int | None = DEFAULT_SEED,
    ) -> None:
        self.scenario_id = scenario_id
        self.solver = solver
        self.interval_seconds = float(interval_seconds)
        self.time_scale = float(time_scale)
        self.seed = seed

        self._store = store
        self._graph = graph
        self._log_store = log_store
        self._tracks: list[VehicleTrack] = list(tracks)
        # The plan this fleet is driving. Kept because a re-optimization has to
        # show what it changed *from*, and `/optimize` throws its solution away —
        # so without this there is no "before" to compare a new plan against.
        self._solution = solution
        # The most recent tick's outcome, for the same reason one step removed:
        # a flagged reading is the trigger for a re-optimization and it exists
        # nowhere else. See `last_flags`.
        self._last_result: TickResult | None = None
        # Fixed once, and drawn from for the life of the watcher, so a demo with
        # a seed prints the same numbers twice.
        self._rng = random.Random(seed)

        # Held across a whole tick. Two ticks interleaving would advance the same
        # vehicle twice from the same reading, and stop() could return while one
        # was still writing.
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks = 0
        self._started_at = _now()
        self._last_tick_at: str | None = None

    # -- one tick ---------------------------------------------------------- #
    def tick(self) -> TickResult:
        """Advance the fleet one interval, measure, judge, apply, store.

        Raises
        ------
        ScenarioNotFound
            If the scenario has been dropped from the store. The timer thread
            treats this as its signal to stop; a request surfaces it as a 404.
        """
        with self._lock:
            record = self._store.get(self.scenario_id)
            state = record.effective_traffic_state()
            weight = traffic_weight_function(self._graph, state)
            lookup = cost_lookup(self._graph, weight)

            # See the module docstring: observation-free, on purpose, and two of
            # them because the measurement and the expectation answer different
            # questions.
            live = TrafficState(timestamp=state.timestamp, conditions=state.conditions)
            clean = TrafficState(
                timestamp=state.timestamp, conditions=ActiveConditions()
            )

            step = self.interval_seconds * self.time_scale
            readings: list[Reading] = []
            observations = record.observations
            condition = state.congestion

            for track in self._tracks:
                track.advance(step)
                placed = position(track, lookup)
                if placed is None:
                    # Two different answers behind one None: a finished route, or
                    # a road the vehicle cannot enter. `blocked_edge` says which,
                    # so a reader is not left guessing at a blank row.
                    blocked = blocked_edge(track, lookup)
                    readings.append(
                        Reading(
                            vehicle_id=track.vehicle_id,
                            edge=blocked,
                            note=(
                                "the road is impassable under these conditions"
                                if blocked is not None
                                else "not on a road: the route is finished"
                            ),
                        )
                    )
                    continue

                u, v = placed.edge
                edge = edge_of(self._graph, u, v)
                modelled = simulated_travel_time(self._graph, edge, live)
                if modelled is None:
                    # Unreachable in practice — `position` resolves a closed road
                    # to None before we get here — but a fleet must not invent a
                    # time for a road it cannot be on.
                    readings.append(
                        Reading(
                            vehicle_id=track.vehicle_id,
                            edge=(u, v),
                            note="the road is impassable under these conditions",
                        )
                    )
                    continue

                observed = noisy_reading(modelled, self._rng)
                stats = (
                    None
                    if record.baseline is None
                    else record.baseline.for_edge(u, v, condition)
                )
                expected = simulated_travel_time(self._graph, edge, clean)

                readings.append(
                    Reading(
                        vehicle_id=track.vehicle_id,
                        edge=(u, v),
                        modelled=modelled,
                        observed=observed,
                        verdict=detect(
                            observed,
                            stats=stats,
                            expected=expected,
                            condition=condition,
                        ),
                    )
                )
                observations = update_edge_from_observation(
                    observations,
                    u=u,
                    v=v,
                    travel_time=observed,
                    # The scenario's own pricing timestamp, not the wall clock —
                    # the same choice an incident's audit row makes. A reading
                    # stamped any other way would claim a congestion band the
                    # costs around it were never built under.
                    observed_at=state.timestamp.isoformat(),
                )

            updated = replace(record, observations=observations)
            new_state = updated.effective_traffic_state()

            changed = 0
            try:
                priced = price_scenario(
                    self._graph, record.scenario, new_state, log_store=None
                )
            except ValueError:
                # A measurement is finite and positive, so it cannot sever a route
                # that was already servable — this is defensive. If it ever does
                # fire, nothing is stored: a cost matrix that disagrees with the
                # state that produced it is worse than a tick that did nothing.
                LOGGER.exception(
                    "re-pricing scenario %s after a fleet tick failed; the readings "
                    "were not applied",
                    self.scenario_id,
                )
                self._ticks += 1
                self._last_tick_at = _now()
                return self._remember(
                    TickResult(
                        scenario_id=self.scenario_id,
                        tick=self._ticks,
                        at=self._last_tick_at,
                        readings=tuple(readings),
                        changed_legs=0,
                        rows_logged=0,
                        conditions_mutated=record.conditions_mutated,
                    )
                )

            changed = changed_entries(record.cost_matrix, priced.cost_matrix)
            updated = replace(updated, cost_matrix=priced.cost_matrix)

            # One row per measured road, stamped with the *new* state — which now
            # carries the readings, so `simulated_travel_time` returns the
            # measured seconds and the row records what a vehicle actually took.
            # A row written against the old state would record the model's own
            # estimate, which is the number the measurement just replaced.
            rows_logged = self._write_rows(
                [reading.edge for reading in readings if reading.observed is not None],
                new_state,
            )
            updated = replace(
                updated, traffic_rows_logged=record.traffic_rows_logged + rows_logged
            )

            try:
                self._store.replace(updated, previous=record)
            except ScenarioChanged:
                # Something else — an incident, most likely — wrote while this
                # tick was re-pricing. The rows stay: the log is global history,
                # and a reading that was taken did happen. Only the live overlay
                # is lost, and the next tick re-reads and measures again.
                LOGGER.info(
                    "scenario %s changed during a fleet tick; the readings were "
                    "logged but not applied",
                    self.scenario_id,
                )

            self._ticks += 1
            self._last_tick_at = _now()
            return self._remember(
                TickResult(
                    scenario_id=self.scenario_id,
                    tick=self._ticks,
                    at=self._last_tick_at,
                    readings=tuple(readings),
                    changed_legs=changed,
                    rows_logged=rows_logged,
                    conditions_mutated=record.conditions_mutated,
                )
            )

    def _remember(self, result: TickResult) -> TickResult:
        """Keep a tick's outcome, and hand it back to the caller.

        Two fields of a :class:`TickResult` are read after the tick that produced
        them and nowhere else: a flagged reading is the trigger a re-optimization
        looks for. Everything else about a tick is either in the response, in the
        log, or on the store.
        """
        self._last_result = result
        return result

    def _write_rows(self, edges: list[Edge], state: TrafficState) -> int:
        """Record the roads this tick measured, best-effort.

        Best-effort exactly like :func:`~qgati.traffic.recorder.price_scenario`
        and the incident audit row: collecting data for a later phase must never
        fail the tick that is keeping the demo's picture current.
        """
        if not edges or self._log_store is None:
            return 0
        rows = TrafficLogRow.from_edges(self._graph, edges, state)
        try:
            return self._log_store.write(rows)
        except Exception:  # noqa: BLE001 - deliberate; see the docstring
            LOGGER.exception(
                "failed to write %d fleet observation row(s); continuing", len(rows)
            )
            return 0

    # -- lifecycle --------------------------------------------------------- #
    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # -- what a re-optimization reads --------------------------------------- #
    @property
    def solution(self) -> Solution | None:
        """The plan this fleet was dispatched on, if the caller supplied it."""
        return self._solution

    @property
    def last_flags(self) -> tuple[Reading, ...]:
        """Readings the detector flagged in the most recent tick.

        The only place an anomaly outlives its tick. :func:`detect` returns a
        verdict, the tick reports it in the response, and nothing kept it — so
        before this existed a re-optimization could not be triggered by an
        anomaly at all, only by an incident.

        Deliberately just the latest tick and not a history: a flag says the road
        is behaving abnormally *now*, and one from twenty ticks ago describes a
        moment nobody is re-planning for. The cost of that choice is that a flag
        is lost the moment the next tick does not repeat it — stated here because
        it is a real limit, not an oversight.
        """
        with self._lock:
            result = self._last_result
        return () if result is None else result.flagged

    def read(self, derive):
        """Run ``derive`` over the fleet's tracks, under the tick lock.

        A tick advances every track's ``travelled`` as it works, so anything
        reading track state from outside this class has to be serialised against
        a tick in flight or it can catch a fleet mid-step. The lock is not
        exposed and ``derive`` is handed the tracks instead, because the one
        thing a caller must not do with it is hold it across a solve.
        """
        with self._lock:
            return derive(tuple(self._tracks))

    def start(self) -> None:
        """Begin ticking on a background thread.

        The first tick is *not* run here — the start route runs it inline, so the
        response that starts a watcher already carries readings rather than
        making the caller wait an interval for the first one.
        """
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"qgati-watcher-{self.scenario_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the thread to finish, and wait for any tick in flight.

        The event makes the sleep return immediately rather than after the whole
        interval; the join is what guarantees no tick is still holding the lock
        (and writing to the log) when this returns.
        """
        self._stop.set()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout)
        self._thread = None

    def _loop(self) -> None:
        """Sleep, tick, repeat — until stopped or the scenario disappears."""
        while not self._stop.wait(self.interval_seconds):
            try:
                self.tick()
            except ScenarioNotFound:
                # Nothing to log an error about and nothing to recover: the
                # scenario is gone, so the fleet it described is gone too.
                LOGGER.info(
                    "scenario %s no longer exists; stopping its watcher",
                    self.scenario_id,
                )
                return
            except Exception:  # noqa: BLE001 - a demo's background thread must not die
                LOGGER.exception(
                    "the watcher for scenario %s failed a tick; continuing",
                    self.scenario_id,
                )

    # -- reporting --------------------------------------------------------- #
    def status(self) -> WatcherState:
        """The watcher's state, and where every vehicle currently is."""
        with self._lock:
            return WatcherState(
                scenario_id=self.scenario_id,
                solver=self.solver,
                running=self.running,
                interval_seconds=self.interval_seconds,
                time_scale=self.time_scale,
                seed=self.seed,
                ticks=self._ticks,
                started_at=self._started_at,
                last_tick_at=self._last_tick_at,
                vehicles=self._vehicle_states(),
            )

    def _vehicle_states(self) -> tuple[VehicleState, ...]:
        """Place every vehicle under the scenario's current costs.

        Returns nothing at all if the scenario has gone, which is what makes
        ``status()`` safe to call from a shutdown hook and from a route that is
        racing a deletion.
        """
        try:
            record = self._store.get(self.scenario_id)
        except ScenarioNotFound:
            return ()

        state = record.effective_traffic_state()
        lookup = cost_lookup(self._graph, traffic_weight_function(self._graph, state))

        placed: list[VehicleState] = []
        for track in self._tracks:
            costs = [lookup(edge) for edge in track.edges]
            total = sum(cost for cost in costs if cost is not None)
            here = position(track, lookup)
            placed.append(
                VehicleState(
                    vehicle_id=track.vehicle_id,
                    edge=None if here is None else here.edge,
                    remaining_seconds=None if here is None else here.remaining,
                    progress=(
                        1.0 if total <= 0.0 else min(track.travelled / total, 1.0)
                    ),
                    finished=here is None,
                )
            )
        return tuple(placed)


class WatcherRegistry:
    """Every watcher this process is running, keyed by scenario id.

    The API holds one of these as an overridable dependency, so a test gets a
    registry of its own and a watcher started by one test cannot tick against
    another test's store. :meth:`stop_all` is wired to the app's shutdown, so no
    daemon thread outlives the log store it writes to.
    """

    def __init__(self) -> None:
        self._watchers: dict[str, ScenarioWatcher] = {}
        self._lock = threading.Lock()

    def add(self, watcher: ScenarioWatcher) -> ScenarioWatcher:
        """Register ``watcher``, or raise :class:`WatcherExists`."""
        with self._lock:
            if watcher.scenario_id in self._watchers:
                raise WatcherExists(watcher.scenario_id)
            self._watchers[watcher.scenario_id] = watcher
        return watcher

    def get(self, scenario_id: str) -> ScenarioWatcher:
        """The watcher for ``scenario_id``, or raise ``KeyError``."""
        with self._lock:
            watcher = self._watchers.get(scenario_id)
        if watcher is None:
            raise KeyError(scenario_id)
        return watcher

    def __contains__(self, scenario_id: object) -> bool:
        with self._lock:
            return scenario_id in self._watchers

    def remove(self, scenario_id: str) -> ScenarioWatcher | None:
        """Take a watcher out of the registry, stopped. ``None`` if there was none."""
        with self._lock:
            watcher = self._watchers.pop(scenario_id, None)
        if watcher is not None:
            watcher.stop()
        return watcher

    def stop_all(self) -> int:
        """Stop everything, returning how many were stopped."""
        with self._lock:
            watchers = list(self._watchers.values())
            self._watchers.clear()
        for watcher in watchers:
            watcher.stop()
        return len(watchers)

    def __len__(self) -> int:
        with self._lock:
            return len(self._watchers)
