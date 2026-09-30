"""Run history: what the optimizer did, kept so it can be looked at later.

One module, deliberately:

:mod:`~qgati.analytics.run_log`
    A SQLite table of solves — ``(timestamp, scenario, solver, cost, runtime, and
    the ETA before and after a re-plan)`` — and the aggregate statistics read from
    it.

**This is a record, not a log and not a model.** Nothing here predicts anything,
and nothing here is written for a refused request: a re-optimization that answers
409 did not run, so it leaves no row. The two readers are
``GET /analytics/runs`` and ``GET /analytics/summary``.

    >>> from qgati.analytics import RunLogRow, RunLogStore
    >>> store = RunLogStore(":memory:")
    >>> store.write(RunLogRow(...))                 # one solve
    >>> rows, total = store.read(limit=25)          # newest first
    >>> store.summary().avg_runtime_ms              # None when nothing has run
"""

from qgati.analytics.run_log import (
    DEFAULT_RUN_DB_PATH,
    KINDS,
    RunLogRow,
    RunLogStore,
    RunSummary,
)

__all__ = [
    "DEFAULT_RUN_DB_PATH",
    "KINDS",
    "RunLogRow",
    "RunLogStore",
    "RunSummary",
]
