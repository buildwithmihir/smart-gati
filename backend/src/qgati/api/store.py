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
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from qgati.graph.cost_matrix import CostMatrix
from qgati.optimizer.models import Scenario

__all__ = ["ScenarioNotFound", "ScenarioStore", "StoredScenario"]


class ScenarioNotFound(KeyError):
    """Raised when a scenario id is not in the store."""


@dataclass(frozen=True)
class StoredScenario:
    """A scenario, the id it is known by, and its precomputed costs."""

    scenario_id: str
    scenario: Scenario
    cost_matrix: CostMatrix
    created_at: str


class ScenarioStore:
    """Thread-safe dict of scenario id -> :class:`StoredScenario`.

    Locked because FastAPI runs synchronous endpoints on a worker threadpool, so
    two requests can touch the store at once.
    """

    def __init__(self) -> None:
        self._items: dict[str, StoredScenario] = {}
        self._lock = threading.Lock()

    def add(self, scenario: Scenario, cost_matrix: CostMatrix) -> StoredScenario:
        """Store a scenario and its costs, returning it with a fresh id."""
        record = StoredScenario(
            scenario_id=uuid.uuid4().hex,
            scenario=scenario,
            cost_matrix=cost_matrix,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
