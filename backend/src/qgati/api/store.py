"""In-memory scenario store.

Scenarios are created through the API and referenced by id afterwards, so
something has to hold them between requests. This is that something, and it is
deliberately the simplest thing that works: a dict behind a lock.

Why the cost matrix is cached alongside
---------------------------------------
Turning a scenario into a cost matrix runs an all-pairs Dijkstra over its nodes —
about 160 ms for a 21-stop instance (see the backend README). Every optimizer call
and every comparison needs it, so rebuilding it per request would make the
routing dominate the search. Storing it with the scenario also guarantees that
two calls against one ``scenario_id`` are optimizing *identical* costs, which is
what makes a comparison between solvers meaningful.

Persistence is out of scope for this phase. Restarting the process drops every
scenario; a real deployment would put this behind a database, and the store's
interface is small enough that swapping the implementation would not touch the
routes.

Why the traffic state is cached alongside
-----------------------------------------
A scenario's cost matrix is priced under a set of simulated traffic conditions
decided when it is created. Storing the state with the matrix is what keeps the
two consistent: the same ``scenario_id`` answers with the same costs, and it can
also say *which* conditions those costs describe. Re-pricing a stored scenario
on every request would mean the same id returned different costs at 09:00 and at
14:00, and a solver comparison on it would be comparing solvers across two
different problems.

That default is unchanged, and it is what makes a comparison meaningful. What
this store now also holds is the operator's explicit override of it: the live
incidents in :attr:`StoredScenario.incidents`. Those do change a scenario's
costs, but only when someone asks — never because time passed — and the record
still keeps the creation-time state underneath, so a revert lands exactly where
it started.

The detection baseline is cached here for the same reason, one step removed: it
is the log's answer at the moment the scenario was created, and re-reading it per
request would mean two observations against one scenario were judged against two
different histories. Building it *before* the scenario is priced is what keeps a
scenario's own rows out of the history it is later judged against.

Why the fleet's measurements live here too
------------------------------------------
The same argument, applied to the newest thing that changes a scenario's costs: a
live GPS reading. Those are held on the record rather than in a process-wide
registry, because a registry would make one scenario's costs depend on what a
*neighbouring* scenario's vehicles happened to drive, and the store's whole
purpose is that one ``scenario_id`` answers with one set of costs.

The split that falls out of this: **costs are per scenario, history is global**.
An observation prices the roads of the scenario whose fleet took it. The row it
also writes to ``traffic_log`` is not scoped to anything — a later scenario's
baseline reads it back, so the detector still learns from the fleet as a whole.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from qgati.graph.cost_matrix import CostMatrix
from qgati.optimizer.models import Scenario
from qgati.traffic.detection import Baseline
from qgati.traffic.incidents import Incident, conditions_for
from qgati.traffic.observations import Observation
from qgati.traffic.simulator import TrafficState

__all__ = [
    "ScenarioChanged",
    "ScenarioNotFound",
    "ScenarioStore",
    "StoredScenario",
]


class ScenarioNotFound(KeyError):
    """Raised when a scenario id is not in the store."""


class ScenarioChanged(RuntimeError):
    """Raised when a stored scenario is no longer the one a caller read.

    A mutation reads a record, spends ~160 ms re-pricing its cost matrix, and
    writes it back. Two incidents on one scenario can interleave inside that
    window, and the loser would silently discard the winner's incident — so the
    write is refused instead, and the caller retries against fresh state.
    """


@dataclass(frozen=True)
class StoredScenario:
    """A scenario, the id it is known by, its costs, and what priced them."""

    scenario_id: str
    scenario: Scenario
    cost_matrix: CostMatrix
    created_at: str
    traffic_state: TrafficState
    traffic_rows_logged: int = 0
    #: Live incidents applied on top of :attr:`traffic_state`, oldest first.
    #:
    #: Kept as a tuple of reports rather than baked into the state, so a revert
    #: is exact: dropping one and re-folding reproduces the previous conditions
    #: including whatever the scenario was created with. See
    #: :func:`~qgati.traffic.incidents.conditions_for`.
    incidents: tuple[Incident, ...] = ()
    #: Per-road history as it stood when this scenario was created, for
    #: :func:`~qgati.traffic.detection.detect` to judge observations against.
    #:
    #: ``None`` on a record built by a caller that had no log to hand. A snapshot
    #: rather than a live query, which is the deliberate simplification recorded in
    #: ``DESIGN_DECISIONS.md``: a production system would refresh these on a daily
    #: batch, and this one takes one reading per scenario and reuses it.
    baseline: Baseline | None = None
    #: The fleet's own measured travel times, keyed by directed road.
    #:
    #: **Tier 1** — a measurement outranks every rule and every report; see
    #: :mod:`qgati.traffic.observations`. These belong to the scenario rather than
    #: to the process, which is what keeps the store's promise that one
    #: ``scenario_id`` answers with one set of costs. The rows a measurement also
    #: writes to ``traffic_log`` are global instead, so a *later* scenario's
    #: baseline learns from the fleet even though this scenario's matrix does not.
    observations: Mapping[tuple, Observation] = field(default_factory=dict)

    @property
    def conditions_mutated(self) -> bool:
        """True while any live incident is applied.

        Derived rather than stored, so it cannot disagree with the incidents it
        describes, and so reverting the last one returns it to false by itself.
        """
        return bool(self.incidents)

    def effective_traffic_state(self) -> TrafficState:
        """The state this scenario is *currently* priced under.

        :attr:`traffic_state` keeps its original meaning — the timestamp and
        conditions the scenario was created with — and this layers the live
        incidents and the fleet's measurements on top. One of them has to be
        derived; deriving this one means the creation-time record is never lost,
        which is what makes a revert exact.

        Both overlays are derived here rather than stored, so neither can
        disagree with the reports and readings they describe, and so reverting an
        incident cannot disturb a measurement.
        """
        return TrafficState(
            timestamp=self.traffic_state.timestamp,
            conditions=conditions_for(self.traffic_state.conditions, self.incidents),
            observations=self.observations,
        )


class ScenarioStore:
    """Thread-safe dict of scenario id -> :class:`StoredScenario`.

    Locked because FastAPI runs synchronous endpoints on a worker threadpool, so
    two requests can touch the store at once.
    """

    def __init__(self) -> None:
        self._items: dict[str, StoredScenario] = {}
        self._lock = threading.Lock()

    def add(
        self,
        scenario: Scenario,
        cost_matrix: CostMatrix,
        traffic_state: TrafficState,
        traffic_rows_logged: int = 0,
        baseline: Baseline | None = None,
    ) -> StoredScenario:
        """Store a scenario, its costs and the conditions that priced them."""
        record = StoredScenario(
            scenario_id=uuid.uuid4().hex,
            scenario=scenario,
            cost_matrix=cost_matrix,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            traffic_state=traffic_state,
            traffic_rows_logged=traffic_rows_logged,
            baseline=baseline,
        )
        with self._lock:
            self._items[record.scenario_id] = record
        return record

    def get(self, scenario_id: str) -> StoredScenario:
        """Look a scenario up, or raise :class:`ScenarioNotFound`."""
        with self._lock:
            record = self._items.get(scenario_id)
        if record is None:
            raise ScenarioNotFound(scenario_id)
        return record

    def replace(
        self, record: StoredScenario, *, previous: StoredScenario
    ) -> StoredScenario:
        """Write ``record`` back, but only if ``previous`` is still stored.

        A compare-and-swap, because the caller cannot hold the lock across the
        re-pricing that produces ``record``: that is an all-pairs Dijkstra, and
        holding the store's lock for it would serialise every other request
        behind one incident.

        ``previous`` is compared by **identity**, never by equality.
        :class:`StoredScenario` carries NumPy arrays, so ``==`` returns an array
        rather than a bool and ``if current != previous`` would raise instead of
        deciding. Identity is also exactly the right question here — it asks
        "is this still the record I read?", which is true of no other record
        even if one happened to compare equal.

        Raises
        ------
        ScenarioNotFound
            If the scenario has since been dropped from the store.
        ScenarioChanged
            If it has been replaced by someone else's write.
        """
        with self._lock:
            current = self._items.get(record.scenario_id)
            if current is None:
                raise ScenarioNotFound(record.scenario_id)
            if current is not previous:
                raise ScenarioChanged(record.scenario_id)
            self._items[record.scenario_id] = record
        return record

    def list(self) -> list[StoredScenario]:
        """Every stored scenario, oldest first."""
        with self._lock:
            return sorted(self._items.values(), key=lambda item: item.created_at)

    def clear(self) -> None:
        """Drop everything. Used by tests to isolate one from the next."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
