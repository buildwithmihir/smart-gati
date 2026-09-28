"""Collapse the road graph into a dense cost matrix over a scenario's nodes.

This is the boundary between the routing world and the optimization world. After
:func:`build_cost_matrix` runs, no VRP solver should ever touch the road graph
again — they see an ``(n, n)`` array of travel times and nothing else. That
separation is what makes a solver runnable on real Delhi travel times and on
hand-written toy costs without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable

import networkx as nx
import numpy as np

from qgati.routing.dijkstra import WeightFn, dijkstra_all_pairs

if TYPE_CHECKING:  # avoids an optimizer <-> graph import cycle at runtime
    from qgati.optimizer.models import Scenario

__all__ = ["CostMatrix", "CostMatrixBuild", "build_cost_matrix",
           "build_cost_matrix_detailed"]


@dataclass(frozen=True)
class CostMatrix:
    """Travel times between a scenario's depot and delivery nodes.

    ``matrix[i, j]`` is the travel time in seconds from ``nodes[i]`` to
    ``nodes[j]``. Index ``0`` is always the depot, so solvers can hard-code that
    every route starts and ends at row/column zero.

    ``delivery_node_index[k]`` maps delivery ``k`` — a position in
    ``scenario.deliveries`` — to its row/column here. Two deliveries sharing a
    road node therefore map to the same index, which is correct: they are the
    same physical stop, counted separately for demand.
    """

    matrix: np.ndarray
    nodes: tuple[Hashable, ...]
    delivery_node_index: tuple[int, ...]

    #: The depot always sits at index 0.
    DEPOT_INDEX = 0

    def __post_init__(self) -> None:
        size = len(self.nodes)
        if self.matrix.shape != (size, size):
            raise ValueError(
                f"matrix is {self.matrix.shape} but there are {size} nodes"
            )

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def depot_index(self) -> int:
        return self.DEPOT_INDEX

    def node_index(self, node: Hashable) -> int:
        """Row/column of a road-graph node, or raise ``KeyError``."""
        try:
            return self.nodes.index(node)
        except ValueError:
            raise KeyError(f"node {node!r} is not in this cost matrix") from None

    def cost(self, i: int, j: int) -> float:
        """Travel time between two matrix indices."""
        return float(self.matrix[i, j])

    def is_symmetric(self, tolerance: float = 1e-6) -> bool:
        """True when every leg costs the same in both directions.

        Worth checking on real data: one-way streets make Delhi's network
        asymmetric, so a solver assuming symmetry would be quietly wrong.
        """
        return bool(
            np.allclose(self.matrix, self.matrix.T, atol=tolerance)
        )


@dataclass(frozen=True)
class CostMatrixBuild:
    """A cost matrix together with the road edges its cheapest paths traverse.

    ``used_edges`` is the set of ``(u, v)`` segments any leg's shortest path
    runs along — the road network this instance actually places load on. The
    traffic layer logs exactly these, so that every row it writes describes a
    road some route genuinely weighed, rather than an arbitrary slice of the
    graph.
    """

    matrix: CostMatrix
    used_edges: frozenset[tuple[Hashable, Hashable]]


def build_cost_matrix(
    graph: nx.Graph,
    scenario: Scenario,
    weight: str | WeightFn = "weight",
    allow_unreachable: bool = False,
) -> CostMatrix:
    """Compute travel-time costs between the depot and every delivery node.

    Uses all-pairs Dijkstra over just this small node subset — one search per
    distinct source, with early termination — rather than anything graph-wide.

    Parameters
    ----------
    graph
        Road graph from Phase 1, with travel-time ``weight``.
    scenario
        The instance to price. Depot and delivery nodes must exist in ``graph``.
    weight
        Edge attribute to minimise, or a callable ``(u, v, data) -> float``.
        The callable form is how the traffic layer prices a network under
        simulated conditions: see
        :func:`~qgati.traffic.simulator.traffic_weight_function`.
    allow_unreachable
        When false (the default), an unreachable pair raises. That is the
        intended behaviour: an unreachable pair means no feasible tour exists,
        and silently substituting ``inf`` would let an optimizer return a
        confidently wrong answer.

    Raises
    ------
    ValueError
        If any node is missing from the graph, or if a pair is unreachable and
        ``allow_unreachable`` is false.
    """
    return build_cost_matrix_detailed(
        graph, scenario, weight=weight, allow_unreachable=allow_unreachable
    ).matrix


def build_cost_matrix_detailed(
    graph: nx.Graph,
    scenario: Scenario,
    weight: str | WeightFn = "weight",
    allow_unreachable: bool = False,
) -> CostMatrixBuild:
    """:func:`build_cost_matrix`, plus the road edges it routed along.

    The extra product is free: the all-pairs search returns the path for every
    leg, and the matrix builder has always discarded them. Collecting them costs
    one pass over the paths already in hand, which is why this returns the edges
    rather than leaving a second, duplicate search to whoever needs them.

    Same arguments and the same ``ValueError`` conditions as
    :func:`build_cost_matrix`.
    """
    nodes = _scenario_nodes(graph, scenario)
    table = dijkstra_all_pairs(graph, nodes, weight=weight)

    matrix, unreachable = _matrix_from_paths(nodes, table)
    if unreachable and not allow_unreachable:
        raise ValueError(
            f"{len(unreachable)} node pair(s) are unreachable in this graph, so no "
            f"feasible tour exists; first pair: {unreachable[0]}. Build the scenario "
            "from largest_strongly_connected_subgraph(graph) so every stop is "
            "reachable from every other."
        )

    position = {node: index for index, node in enumerate(nodes)}
    return CostMatrixBuild(
        matrix=CostMatrix(
            matrix=matrix,
            nodes=tuple(nodes),
            delivery_node_index=tuple(
                position[delivery.node] for delivery in scenario.deliveries
            ),
        ),
        used_edges=_edges_along(table),
    )


def _scenario_nodes(graph: nx.Graph, scenario: Scenario) -> list[Hashable]:
    """The depot and every distinct delivery node, depot first.

    Raises
    ------
    ValueError
        If any of them is absent from the graph.
    """
    nodes: list[Hashable] = [scenario.depot.node]
    for delivery in scenario.deliveries:
        if delivery.node not in nodes:
            nodes.append(delivery.node)

    missing = [node for node in nodes if node not in graph]
    if missing:
        raise ValueError(
            f"{len(missing)} scenario node(s) are not in the graph: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    return nodes


def _matrix_from_paths(
    nodes: list[Hashable], table: dict
) -> tuple[np.ndarray, list[tuple[Hashable, Hashable]]]:
    """Dense the all-pairs result into an array, noting unreachable pairs."""
    size = len(nodes)
    matrix = np.full((size, size), np.inf, dtype=float)
    unreachable: list[tuple[Hashable, Hashable]] = []

    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            _, cost = table[(source, target)]
            matrix[i, j] = cost
            if np.isinf(cost) and i != j:
                unreachable.append((source, target))

    return matrix, unreachable


def _edges_along(table: dict) -> frozenset[tuple[Hashable, Hashable]]:
    """Every road edge traversed by any of the table's paths.

    An unreachable pair contributes nothing, since its path is empty.
    """
    edges: set[tuple[Hashable, Hashable]] = set()
    for path, _cost in table.values():
        for u, v in zip(path, path[1:]):
            edges.add((u, v))
    return frozenset(edges)
