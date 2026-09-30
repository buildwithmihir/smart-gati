"""The traffic log — a SQLite table of simulated road conditions.

This exists for a future phase, not this one. There is no model here and nothing
reads these rows back except the API's inspection endpoint. The point is to start
collecting **now**, so that when a travel-time predictor is finally built it has
a history of ``(road, time, condition, incident) -> travel time`` to learn from
rather than an empty table.

Why SQLite
----------
The brief asks for SQLite and not Postgres, and that is the right call at this
size: one file, no server, no migration tooling, and the whole write is a single
``executemany`` inside one transaction. The schema is deliberately flat so it can
be lifted into a columnar store or a dataframe later without reshaping.

What a row means
----------------
One row is one road, under one set of conditions, at one moment. Rows come in two
kinds and both are wanted:

* **Traversed roads** — every edge the scenario's cheapest paths run along. An
  off-peak run and a peak run of the same scenario touch largely the same roads,
  which is what makes the table *paired*: the same key appears under several
  conditions, which is exactly the shape a supervised model needs.
* **Incident roads** — the roads an accident or closure names, whether or not any
  route used them. This half is not optional: a closed edge has infinite weight,
  so no cheapest path can ever include it, and without logging incident edges
  explicitly the ``road_closure`` value would never once appear in the table.

``travel_time`` is ``NULL`` for a closed road. It is impassable, so there is no
travel time to record, and ``incident_type`` carries the reason.

Two column notes
----------------
``road_u``/``road_v`` are declared with **no SQLite type**, which gives them BLOB
affinity and therefore stores exactly what is bound: an integer node id comes back
as an int and a string id as a str. Declaring them ``TEXT`` (or ``NUMERIC``, which
tries to coerce numeric-looking text) would silently rewrite OSM's large integer
node ids, and the frontend matches these against ids from the graph.

Pagination orders by the autoincrement ``id``, never by ``timestamp``. Timestamps
are ISO strings, and lexicographic order is only chronological while every row
shares one UTC offset — which a client-supplied timestamp is free to violate.
``id`` is monotonic per insert, so it is both a correct "newest first" and a
stable cursor.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

import networkx as nx

from qgati.graph.graph_builder import DEFAULT_DATA_DIR
from qgati.traffic.simulator import Edge, TrafficState, simulated_travel_time

__all__ = [
    "DEFAULT_LOG_DB_PATH",
    "TrafficLogRow",
    "TrafficLogStore",
    "road_id_of",
]

Node = Hashable

#: The log lives beside the graph cache in ``backend/data/``. Override with
#: ``SMART_GATI_TRAFFIC_DB`` — the test suite points it at a tmp path so a test run
#: never touches the developer's collected data.
DEFAULT_LOG_DB_PATH = Path(
    os.environ.get("SMART_GATI_TRAFFIC_DB", DEFAULT_DATA_DIR / "traffic_log.db")
)

#: SQLite's in-memory path, accepted so a caller can keep a whole log in RAM.
MEMORY_PATH = ":memory:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traffic_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    road_id           TEXT    NOT NULL,
    road_u                    NOT NULL,
    road_v                    NOT NULL,
    timestamp         TEXT    NOT NULL,
    day_of_week       TEXT    NOT NULL,
    time_of_day       TEXT    NOT NULL,
    traffic_condition TEXT    NOT NULL,
    incident_type     TEXT,
    travel_time       REAL
);
CREATE INDEX IF NOT EXISTS idx_traffic_log_timestamp ON traffic_log (timestamp);
CREATE INDEX IF NOT EXISTS idx_traffic_log_road      ON traffic_log (road_u, road_v);
CREATE INDEX IF NOT EXISTS idx_traffic_log_condition ON traffic_log (traffic_condition);
"""

_COLUMNS = (
    "road_id",
    "road_u",
    "road_v",
    "timestamp",
    "day_of_week",
    "time_of_day",
    "traffic_condition",
    "incident_type",
    "travel_time",
)

#: Columns an earlier version of this schema wrote and this one does not. Rain
#: stopped being a modelled condition, so ``weather_condition`` would be a
#: constant ``"clear"`` on every row — a feature that can never vary is worse
#: than no feature at all. Dropped on open; see
#: :meth:`TrafficLogStore._drop_retired_columns`.
RETIRED_COLUMNS = ("weather_condition",)

_INSERT = (
    f"INSERT INTO traffic_log ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(_COLUMNS))})"
)


def road_id_of(u: Node, v: Node) -> str:
    """The canonical id of a directed road segment.

    An edge is its endpoints, so the id is built from them. Kept in one place
    because both the writer and the reader derive it.
    """
    return f"{u}->{v}"


def _sort_key(edge: tuple[Node, Node]) -> tuple[str, str]:
    """Order edges deterministically for insertion.

    Node ids may be ints or strings and the two are not orderable together, so
    the key compares their text. Without this a frozenset of edges would write
    in an order that varies between runs, making the table awkward to diff.
    """
    u, v = edge
    return str(u), str(v)


@dataclass(frozen=True, slots=True)
class TrafficLogRow:
    """One logged road-condition observation.

    Mirrors the table exactly, minus the surrogate ``id`` — which belongs to the
    database, not to the observation. :attr:`road_id` is derived rather than
    stored on the instance so the format has a single definition; the column
    itself is written from this property.
    """

    road_u: Node
    road_v: Node
    timestamp: str
    day_of_week: str
    time_of_day: str
    traffic_condition: str
    incident_type: str | None = None
    travel_time: float | None = None

    @property
    def road_id(self) -> str:
        return road_id_of(self.road_u, self.road_v)

    def to_dict(self) -> dict:
        """JSON-ready form for the inspection endpoint."""
        return {
            "road_id": self.road_id,
            "road_u": self.road_u,
            "road_v": self.road_v,
            "timestamp": self.timestamp,
            "day_of_week": self.day_of_week,
            "time_of_day": self.time_of_day,
            "traffic_condition": self.traffic_condition,
            "incident_type": self.incident_type,
            "travel_time": self.travel_time,
        }

    def _as_parameters(self) -> tuple:
        return (
            self.road_id,
            self.road_u,
            self.road_v,
            self.timestamp,
            self.day_of_week,
            self.time_of_day,
            self.traffic_condition,
            self.incident_type,
            self.travel_time,
        )

    @classmethod
    def from_edges(
        cls, graph: nx.Graph, edges: Iterable[tuple[Node, Node]], state: TrafficState
    ) -> list[TrafficLogRow]:
        """Build one row per edge, all stamped with ``state``.

        Travel times come from :func:`~qgati.traffic.simulator.simulated_travel_time`
        — the same call routing makes — so what lands in the table is the cost the
        optimizer was actually charged, not a recomputation that might drift from
        it.
        """
        timestamp = state.timestamp
        rows: list[TrafficLogRow] = []
        for u, v in sorted(edges, key=_sort_key):
            edge = Edge(u, v)
            travel_time = simulated_travel_time(graph, edge, state)
            rows.append(
                cls(
                    road_u=u,
                    road_v=v,
                    timestamp=timestamp.isoformat(),
                    day_of_week=timestamp.strftime("%A"),
                    time_of_day=timestamp.strftime("%H:%M"),
                    traffic_condition=state.traffic_condition,
                    incident_type=state.incident_type(edge),
                    travel_time=None if travel_time is None else round(travel_time, 3),
                )
            )
        return rows


class TrafficLogStore:
    """Append-mostly SQLite table of road conditions.

    Thread-safe, because FastAPI runs synchronous endpoints on a worker
    threadpool and two requests can log at once. One connection guarded by a lock
    is the simplest thing that is actually correct here; SQLite serialises
    writers anyway, so a pool would buy nothing but complexity.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path) if path is not None else str(DEFAULT_LOG_DB_PATH)
        if self.path != MEMORY_PATH:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            # WAL lets a reader inspect the log while a request is writing to it,
            # which is the normal state of affairs for the inspection endpoint.
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.executescript(_SCHEMA)
            self._drop_retired_columns()
            self._connection.commit()

    def _drop_retired_columns(self) -> None:
        """Drop columns an older schema wrote, from a table that already exists.

        ``CREATE TABLE IF NOT EXISTS`` leaves a pre-existing table untouched, so
        a database written before rain stopped being modelled would keep its
        ``weather_condition`` column — declared ``NOT NULL`` with no default, and
        therefore fatal to every insert this version makes. The column is the
        thing that is obsolete, so the column goes and the rows stay: this log is
        an accumulating dataset, and losing a day of it to a schema tidy-up would
        be the wrong trade.

        A fresh database, which is the common case, has nothing to drop. A
        SQLite older than 3.35 has no ``ALTER TABLE ... DROP COLUMN`` and raises;
        that is left to surface rather than papered over, since silently keeping
        a broken column would be worse than a loud error.
        """
        present = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(traffic_log)")
        }
        for column in RETIRED_COLUMNS:
            if column in present:
                self._connection.execute(
                    f"ALTER TABLE traffic_log DROP COLUMN {column}"
                )

    # -- writing ----------------------------------------------------------- #
    def write(self, rows: Iterable[TrafficLogRow]) -> int:
        """Insert rows in one transaction, returning how many were written.

        Raises whatever SQLite raised. Whether a failed log should fail the
        request that triggered it is a policy decision and lives one level up, in
        :mod:`qgati.traffic.recorder`.
        """
        parameters = [row._as_parameters() for row in rows]
        if not parameters:
            return 0
        with self._lock, self._connection:
            cursor = self._connection.executemany(_INSERT, parameters)
            return cursor.rowcount

    # -- reading ----------------------------------------------------------- #
    def read(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        road_u: Node | None = None,
        road_v: Node | None = None,
        traffic_condition: str | None = None,
        incident_type: str | None = None,
    ) -> tuple[list[TrafficLogRow], int]:
        """One page of rows, newest first, plus the total matching the filters.

        The total covers the *filtered* set rather than the whole table, so a
        client can page through a filter without asking for a count separately.
        A filter left at ``None`` is simply not applied.
        """
        filters = (
            ("road_u", road_u),
            ("road_v", road_v),
            ("traffic_condition", traffic_condition),
            ("incident_type", incident_type),
        )
        applied = [(column, value) for column, value in filters if value is not None]
        clauses = [f"{column} = ?" for column, _ in applied]
        parameters: list[object] = [value for _, value in applied]
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            total = self._connection.execute(
                f"SELECT COUNT(*) FROM traffic_log{where}", parameters
            ).fetchone()[0]
            cursor = self._connection.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM traffic_log{where} "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            )
            rows = [_row_from_sqlite(row) for row in cursor.fetchall()]

        return rows, int(total)

    def count(self) -> int:
        """Every row in the table, filters aside."""
        with self._lock:
            return int(
                self._connection.execute("SELECT COUNT(*) FROM traffic_log").fetchone()[0]
            )

    def __len__(self) -> int:
        return self.count()

    def clear(self) -> None:
        """Drop every row. Used by tests to isolate one from the next."""
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM traffic_log")

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _row_from_sqlite(row: sqlite3.Row) -> TrafficLogRow:
    return TrafficLogRow(
        road_u=row["road_u"],
        road_v=row["road_v"],
        timestamp=row["timestamp"],
        day_of_week=row["day_of_week"],
        time_of_day=row["time_of_day"],
        traffic_condition=row["traffic_condition"],
        incident_type=row["incident_type"],
        travel_time=row["travel_time"],
    )
