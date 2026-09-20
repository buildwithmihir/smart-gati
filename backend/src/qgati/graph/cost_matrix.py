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

from qgati.routing.dijkstra import dijkstra_all_pairs

if TYPE_CHECKING:  # avoids an optimizer <-> graph import cycle at runtime
    from qgati.optimizer.models import Scenario

__all__ = ["CostMatrix", "build_cost_matrix"]


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


def build_cost_matrix(
    graph: nx.Graph,
    scenario: Scenario,
    weight: str = "weight",
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
        Edge attribute to minimise.
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

    table = dijkstra_all_pairs(graph, nodes, weight=weight)
    size = len(nodes)
    matrix = np.full((size, size), np.inf, dtype=float)
    unreachable: list[tuple[Hashable, Hashable]] = []

    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            _, cost = table[(source, target)]
            matrix[i, j] = cost
            if np.isinf(cost) and i != j:
                unreachable.append((source, target))

    if unreachable and not allow_unreachable:
        raise ValueError(
            f"{len(unreachable)} node pair(s) are unreachable in this graph, so no "
            f"feasible tour exists; first pair: {unreachable[0]}. Build the scenario "
            "from largest_strongly_connected_subgraph(graph) so every stop is "
            "reachable from every other."
        )

    position = {node: index for index, node in enumerate(nodes)}
    return CostMatrix(
        matrix=matrix,
        nodes=tuple(nodes),
        delivery_node_index=tuple(
            position[delivery.node] for delivery in scenario.deliveries
        ),
    )
