"""Pricing a scenario under traffic conditions, and recording what was applied.

This is the seam between three otherwise independent things: the traffic rules
(:mod:`~qgati.traffic.simulator`), the cost-matrix builder
(:mod:`~qgati.graph.cost_matrix`) and the log (:mod:`~qgati.traffic.log_store`).
Keeping it here rather than in the API means the whole behaviour — price under a
state, work out which roads that touched, write rows — is testable without going
through HTTP.

The one function that matters is :func:`price_scenario`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Hashable

import networkx as nx

from qgati.graph.cost_matrix import CostMatrix, build_cost_matrix_detailed
from qgati.traffic.log_store import TrafficLogRow, TrafficLogStore
from qgati.traffic.simulator import TrafficState, traffic_weight_function

__all__ = ["TrafficPricing", "price_scenario"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrafficPricing:
    """The outcome of pricing one scenario under one traffic state."""

    cost_matrix: CostMatrix
    """Travel times under the state's conditions — what the optimizer sees."""

    used_edges: frozenset[tuple[Hashable, Hashable]]
    """Roads the scenario's cheapest paths run along."""

    rows_logged: int
    """Log rows written. Zero if the write failed; see :func:`price_scenario`."""


def price_scenario(
    graph: nx.Graph,
    scenario,
    state: TrafficState,
    log_store: TrafficLogStore | None = None,
) -> TrafficPricing:
    """Price ``scenario`` under ``state`` and log the roads it affected.

    The cost matrix is built once, with the traffic weight function, and the
    edges it routed along fall out of that same search — no second traversal.

    Which roads get logged is :attr:`CostMatrixBuild.used_edges` **plus** every
    edge named as an accident or closure. Both halves are needed. A closed edge
    carries infinite weight, so no cheapest path can ever include it; log only
    the traversed roads and the ``road_closure`` value would never appear in the
    dataset at all. The incident roads are precisely the observations a future
    model cannot reconstruct from the traversed ones.

    Logging is best-effort. A failure is reported through the logger and
    :attr:`TrafficPricing.rows_logged` comes back as ``0``, but the pricing
    result is still returned: collecting data for a later phase must never fail
    the request that is serving the user now.

    Parameters
    ----------
    graph
        Road graph. Never mutated.
    scenario
        The instance to price. Its nodes must exist in ``graph``.
    state
        The timestamp and conditions to price under.
    log_store
        Where to record the affected roads. ``None`` skips logging entirely,
        which is what a caller that only wants the costs should pass.

    Raises
    ------
    ValueError
        If the scenario's nodes are missing from the graph, or a pair of them is
        unreachable under these conditions — a closure can sever a route that
        exists in the static network, and that is a real answer rather than a
        bug to paper over.
    """
    build = build_cost_matrix_detailed(
        graph, scenario, weight=traffic_weight_function(graph, state)
    )

    affected = (
        build.used_edges
        | state.conditions.accident_edges
        | state.conditions.closed_edges
    )

    rows_logged = 0
    if log_store is not None:
        rows = TrafficLogRow.from_edges(graph, affected, state)
        try:
            rows_logged = log_store.write(rows)
        except Exception:  # noqa: BLE001 - deliberate; see the docstring
            LOGGER.exception(
                "failed to write %d traffic log rows; continuing without them",
                len(rows),
            )

    return TrafficPricing(
        cost_matrix=build.matrix,
        used_edges=build.used_edges,
        rows_logged=rows_logged,
    )
