"""The run history — a SQLite table of every solve this process has performed.

Where :mod:`~qgati.traffic.log_store` records what the *network* was doing, this
records what the *optimizer* did. One row per solve: when it ran, on which
scenario, with which solver, at what cost, how long it took, and — when the solve
was a re-plan — what the remaining work would have taken before it.

Why this exists
---------------
Every solve endpoint already computes a row's worth of numbers and then throws
them away. ``POST /reoptimize`` computes the most valuable of them: the remaining
work priced on the routes the fleet is already driving, against the same work
priced on the re-planned routes. That comparison is the whole point of re-planning,
and until now it existed only inside one HTTP response and was gone.

Nothing reads the table back except the two endpoints in :mod:`qgati.api.main`.
There is no model and no prediction here; the rows are a record, and the two
readers are an inspection list and an aggregate.

Why SQLite
----------
Same reasoning as the traffic log: one file, no server, no migration tooling, and
a write is a single row inside one transaction. The schema is flat so it can be
lifted into a dataframe or a warehouse later without reshaping.

What a row is, and what it is not
---------------------------------
A row is a **solve that happened**, not a request that arrived. A refused
re-optimization — 409, nothing warrants one — returns before the write, so the
table holds runs rather than attempts. That is deliberate: "how many times did we
fail to re-plan" is a question about logs, and this is not a log.

Every figure is one the solve produced. ``runtime_ms`` is the same measured span,
``cost``/``travel_cost``/``travel_time``/``distance_m``/``fuel_litres`` come from
the same :class:`~qgati.optimizer.models.Evaluation` object, and the ETA pair is
the same two evaluations the before/after panel is built from. Nothing here is
recomputed.

**The totals cover different work depending on ``kind``.** For ``optimize`` and
``dispatch`` they are the whole scenario; for ``reoptimize`` they are the unserved
remainder; for ``avoid_road`` they are the whole fleet's remainder after the
reporting vehicle was re-solved, which is the response's ``after`` total rather
than the figure at the top of it — that one covers the reporting vehicle alone, and
a row priced that way could not be compared with its own ``old_eta_seconds``. That
is not a defect to be normalised away — it is what those runs are — but it does
mean the cost and travel-time columns are not comparable across kinds, and
``n_deliveries`` is the *scenario's* size rather than the run's.

One derived field, on purpose
-----------------------------
``new_eta_seconds`` is not a column. For a re-plan it *is* ``travel_time`` — the
seconds the re-planned routes take — and storing the same number twice is two
fields that can drift apart. :meth:`RunLogRow.to_dict` derives it, and derives
``eta_saved_seconds`` from the pair, so the wire shape reads as a before/after
without the table holding a redundant copy.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from qgati.graph.graph_builder import DEFAULT_DATA_DIR

__all__ = [
    "DEFAULT_RUN_DB_PATH",
    "KINDS",
    "RunLogRow",
    "RunLogStore",
    "RunSummary",
]


#: The four things that solve. Named here rather than inline at the call sites so
#: the set is one list a reader can see whole — and so a test can assert a row's
#: kind came from it.
KINDS = ("dispatch", "optimize", "reoptimize", "avoid_road")

#: The history lives beside the graph cache in ``backend/data/``. Override with
#: ``SMART_GATI_RUN_DB`` — the test suite points it at an in-memory store so a test run
#: never writes into the developer's real history.
DEFAULT_RUN_DB_PATH = Path(
    os.environ.get("SMART_GATI_RUN_DB", DEFAULT_DATA_DIR / "run_history.db")
)

#: SQLite's in-memory path, accepted so a caller can keep a whole history in RAM.
MEMORY_PATH = ":memory:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    kind             TEXT    NOT NULL,
    scenario_id      TEXT    NOT NULL,
    n_deliveries     INTEGER NOT NULL,
    n_vehicles       INTEGER NOT NULL,
    solver           TEXT    NOT NULL,
    solver_name      TEXT    NOT NULL,
    seed             INTEGER,
    iterations       INTEGER,
    population       INTEGER,
    trigger          TEXT,
    trigger_detail   TEXT,
    affected_vehicle TEXT,
    cost             REAL    NOT NULL,
    travel_cost      REAL    NOT NULL,
    travel_time      REAL    NOT NULL,
    distance_m       REAL    NOT NULL,
    fuel_litres      REAL    NOT NULL,
    feasible         INTEGER NOT NULL,
    runtime_ms       REAL    NOT NULL,
    old_eta_seconds  REAL,
    moved            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_run_history_timestamp ON run_history (timestamp);
CREATE INDEX IF NOT EXISTS idx_run_history_scenario  ON run_history (scenario_id);
CREATE INDEX IF NOT EXISTS idx_run_history_kind      ON run_history (kind);
CREATE INDEX IF NOT EXISTS idx_run_history_solver    ON run_history (solver);
"""

_COLUMNS = (
    "timestamp",
    "kind",
    "scenario_id",
    "n_deliveries",
    "n_vehicles",
    "solver",
    "solver_name",
    "seed",
    "iterations",
    "population",
    "trigger",
    "trigger_detail",
    "affected_vehicle",
    "cost",
    "travel_cost",
    "travel_time",
    "distance_m",
    "fuel_litres",
    "feasible",
    "runtime_ms",
    "old_eta_seconds",
    "moved",
)

_INSERT = (
    f"INSERT INTO run_history ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(_COLUMNS))})"
)

_SELECT = f"SELECT id, {', '.join(_COLUMNS)} FROM run_history"


@dataclass(frozen=True, slots=True)
class RunLogRow:
    """One recorded solve.

    Mirrors the table, including the surrogate ``id`` — which the reader returns
    because a client paging a list benefits from a stable key, unlike
    :class:`~qgati.traffic.log_store.TrafficLogRow` where the id belongs to the
    database and not to the observation. Here the row *is* a database record of an
    event rather than a measurement of the world, so the two coincide.

    ``old_eta_seconds`` is the only optional figure, and it is ``None`` for
    everything that is not a re-plan: an ``optimize`` or a ``dispatch`` has no
    earlier plan to be measured against, and a zero would read as "the previous
    plan was instant" rather than "there was no previous plan".
    """

    timestamp: str
    kind: str
    scenario_id: str
    n_deliveries: int
    n_vehicles: int
    solver: str
    solver_name: str
    cost: float
    travel_cost: float
    travel_time: float
    distance_m: float
    fuel_litres: float
    feasible: bool
    runtime_ms: float
    seed: int | None = None
    iterations: int | None = None
    population: int | None = None
    trigger: str | None = None
    trigger_detail: str | None = None
    affected_vehicle: str | None = None
    old_eta_seconds: float | None = None
    moved: int = 0
    id: int | None = None

    @property
    def new_eta_seconds(self) -> float | None:
        """What the re-planned routes take, or ``None`` when there was no re-plan.

        Derived rather than stored: for a re-plan this is exactly
        :attr:`travel_time`, and holding the same number in two columns is an
        invitation for them to disagree.
        """
        return None if self.old_eta_seconds is None else self.travel_time

    @property
    def eta_saved_seconds(self) -> float | None:
        """Seconds the re-plan took off the remaining work; negative if it added.

        Positive means the re-planed routes are quicker than the routes the fleet
        was already driving. A negative value is possible — a re-solve on a small
        instance can land on a worse arrangement than the one in hand — and is
        reported rather than clamped.
        """
        if self.old_eta_seconds is None:
            return None
        return self.old_eta_seconds - self.travel_time

    def to_dict(self) -> dict:
        """JSON-ready form, with the derived ETA pair filled in."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "scenario_id": self.scenario_id,
            "n_deliveries": self.n_deliveries,
            "n_vehicles": self.n_vehicles,
            "solver": self.solver,
            "solver_name": self.solver_name,
            "seed": self.seed,
            "iterations": self.iterations,
            "population": self.population,
            "trigger": self.trigger,
            "trigger_detail": self.trigger_detail,
            "affected_vehicle": self.affected_vehicle,
            "cost": self.cost,
            "travel_cost": self.travel_cost,
            "travel_time": self.travel_time,
            "distance_m": self.distance_m,
            "fuel_litres": self.fuel_litres,
            "feasible": self.feasible,
            "runtime_ms": self.runtime_ms,
            "old_eta_seconds": self.old_eta_seconds,
            "new_eta_seconds": self.new_eta_seconds,
            "eta_saved_seconds": self.eta_saved_seconds,
            "moved": self.moved,
        }

    def _as_parameters(self) -> tuple:
        return (
            self.timestamp,
            self.kind,
            self.scenario_id,
            self.n_deliveries,
            self.n_vehicles,
            self.solver,
            self.solver_name,
            self.seed,
            self.iterations,
            self.population,
            self.trigger,
            self.trigger_detail,
            self.affected_vehicle,
            self.cost,
            self.travel_cost,
            self.travel_time,
            self.distance_m,
            self.fuel_litres,
            int(self.feasible),
            self.runtime_ms,
            self.old_eta_seconds,
            self.moved,
        )


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregate statistics over the whole run history.

    Every average is ``None`` when there is nothing to average, which is why they
    are typed as optional. SQL's ``AVG`` over no rows is ``NULL`` and that is kept
    all the way out to the API: an average over zero runs rendered as ``0.0`` would
    be a claim about performance where no measurement exists. The dashboard already
    draws this distinction elsewhere — ``DashboardPlan.feasible`` is ``null`` for a
    dispatched plan rather than guessed at.
    """

    total_runs: int
    first_run_at: str | None
    last_run_at: str | None
    runs_by_kind: dict[str, int]
    runs_by_solver: dict[str, int]
    feasible_runs: int
    infeasible_runs: int
    incident_triggered_runs: int
    avg_runtime_ms: float | None
    min_runtime_ms: float | None
    max_runtime_ms: float | None
    avg_cost: float | None
    avg_travel_cost: float | None
    avg_travel_time_seconds: float | None
    #: Re-plans that had a baseline to be compared against.
    eta_runs: int
    avg_saved_seconds: float | None
    total_saved_seconds: float | None
    improved_runs: int
    worsened_runs: int
    unchanged_runs: int


class RunLogStore:
    """Append-mostly SQLite table of solves.

    Thread-safe, because FastAPI runs synchronous endpoints on a worker threadpool
    and two requests can solve at once. One connection guarded by a lock is the
    simplest correct thing at this size; SQLite serialises writers anyway.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path) if path is not None else str(DEFAULT_RUN_DB_PATH)
        if self.path != MEMORY_PATH:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            # WAL lets the History tab read the table while a solve is writing to
            # it, which is the normal state of affairs for an inspection endpoint.
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()

    # -- writing ----------------------------------------------------------- #
    def write(self, row: RunLogRow) -> int:
        """Insert one row, returning its id.

        Raises whatever SQLite raised. Whether a failed history write should fail
        the solve that triggered it is a policy decision and lives one level up, in
        :func:`qgati.api.main._record_run` — which swallows it, because a user's
        plan must not be lost to a record-keeping failure.
        """
        with self._lock, self._connection:
            cursor = self._connection.execute(_INSERT, row._as_parameters())
            return int(cursor.lastrowid or 0)

    # -- reading ----------------------------------------------------------- #
    def read(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        kind: str | None = None,
        solver: str | None = None,
        scenario_id: str | None = None,
    ) -> tuple[list[RunLogRow], int]:
        """One page of runs, newest first, plus the total matching the filters.

        The total covers the *filtered* set rather than the whole table, so a client
        can page through a filter without asking for a count separately. A filter
        left at ``None`` is simply not applied.

        Ordered by the autoincrement ``id``, never by ``timestamp``. Timestamps are
        ISO strings, and lexicographic order is only chronological while every row
        shares one UTC offset; ``id`` is monotonic per insert and is therefore both
        a correct "newest first" and a stable cursor.
        """
        filters = (
            ("kind", kind),
            ("solver", solver),
            ("scenario_id", scenario_id),
        )
        applied = [(column, value) for column, value in filters if value is not None]
        clauses = [f"{column} = ?" for column, _ in applied]
        parameters: list[object] = [value for _, value in applied]
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            total = self._connection.execute(
                f"SELECT COUNT(*) FROM run_history{where}", parameters
            ).fetchone()[0]
            cursor = self._connection.execute(
                f"{_SELECT}{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            )
            rows = [_row_from_sqlite(row) for row in cursor.fetchall()]

        return rows, int(total)

    def summary(self) -> RunSummary:
        """Aggregate statistics over every row in the table.

        Read in one lock so the totals cannot be taken across a write: a summary
        assembled from two snapshots of a table that changed between them would
        report counts and averages that never coexisted.
        """
        with self._lock:
            totals = self._connection.execute(
                """
                SELECT COUNT(*)                                        AS runs,
                       MIN(timestamp)                                  AS first_at,
                       MAX(timestamp)                                  AS last_at,
                       SUM(feasible)                                   AS feasible_runs,
                       -- 'incident' and 'anomaly' only. An 'override' is a driver
                       -- reporting a road themselves — the fleet did not flag it and
                       -- no incident caused it — so counting it here would overstate
                       -- how much of the history was the system reacting to the world.
                       SUM(trigger IN ('incident', 'anomaly'))         AS triggered_runs,
                       AVG(runtime_ms)                                 AS avg_runtime,
                       MIN(runtime_ms)                                 AS min_runtime,
                       MAX(runtime_ms)                                 AS max_runtime,
                       AVG(cost)                                       AS avg_cost,
                       AVG(travel_cost)                                AS avg_travel_cost,
                       AVG(travel_time)                                AS avg_travel_time
                FROM run_history
                """
            ).fetchone()

            by_kind = {
                row["kind"]: int(row["n"])
                for row in self._connection.execute(
                    "SELECT kind, COUNT(*) AS n FROM run_history GROUP BY kind"
                )
            }
            by_solver = {
                row["solver"]: int(row["n"])
                for row in self._connection.execute(
                    "SELECT solver, COUNT(*) AS n FROM run_history GROUP BY solver"
                )
            }
            # The ETA pass is restricted to rows that *have* a baseline, so a
            # dispatch's NULL cannot be averaged in as a zero saving.
            eta = self._connection.execute(
                """
                SELECT COUNT(*)                                          AS runs,
                       AVG(old_eta_seconds - travel_time)                AS avg_saved,
                       SUM(old_eta_seconds - travel_time)                AS total_saved,
                       SUM(travel_time <  old_eta_seconds)               AS improved,
                       SUM(travel_time >  old_eta_seconds)               AS worsened,
                       SUM(travel_time =  old_eta_seconds)               AS unchanged
                FROM run_history
                WHERE old_eta_seconds IS NOT NULL
                """
            ).fetchone()

        runs = int(totals["runs"])
        feasible = int(totals["feasible_runs"] or 0)

        return RunSummary(
            total_runs=runs,
            first_run_at=totals["first_at"],
            last_run_at=totals["last_at"],
            runs_by_kind=by_kind,
            runs_by_solver=by_solver,
            feasible_runs=feasible,
            infeasible_runs=runs - feasible,
            incident_triggered_runs=int(totals["triggered_runs"] or 0),
            avg_runtime_ms=totals["avg_runtime"],
            min_runtime_ms=totals["min_runtime"],
            max_runtime_ms=totals["max_runtime"],
            avg_cost=totals["avg_cost"],
            avg_travel_cost=totals["avg_travel_cost"],
            avg_travel_time_seconds=totals["avg_travel_time"],
            eta_runs=int(eta["runs"]),
            avg_saved_seconds=eta["avg_saved"],
            total_saved_seconds=eta["total_saved"],
            improved_runs=int(eta["improved"] or 0),
            worsened_runs=int(eta["worsened"] or 0),
            unchanged_runs=int(eta["unchanged"] or 0),
        )

    def count(self) -> int:
        """Every row in the table, filters aside."""
        with self._lock:
            return int(
                self._connection.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]
            )

    def __len__(self) -> int:
        return self.count()

    def clear(self) -> None:
        """Drop every row. Used by tests to isolate one from the next."""
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM run_history")

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _row_from_sqlite(row: sqlite3.Row) -> RunLogRow:
    return RunLogRow(
        id=row["id"],
        timestamp=row["timestamp"],
        kind=row["kind"],
        scenario_id=row["scenario_id"],
        n_deliveries=row["n_deliveries"],
        n_vehicles=row["n_vehicles"],
        solver=row["solver"],
        solver_name=row["solver_name"],
        seed=row["seed"],
        iterations=row["iterations"],
        population=row["population"],
        trigger=row["trigger"],
        trigger_detail=row["trigger_detail"],
        affected_vehicle=row["affected_vehicle"],
        cost=row["cost"],
        travel_cost=row["travel_cost"],
        travel_time=row["travel_time"],
        distance_m=row["distance_m"],
        fuel_litres=row["fuel_litres"],
        # SQLite has no boolean type; the column holds 0 or 1 and this is the one
        # place it becomes a bool again.
        feasible=bool(row["feasible"]),
        runtime_ms=row["runtime_ms"],
        old_eta_seconds=row["old_eta_seconds"],
        moved=row["moved"],
    )
