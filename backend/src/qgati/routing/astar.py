"""A* shortest-path search with an admissible straight-line heuristic.

Also implemented from scratch, and deliberately so: correctness here is checked
against ``networkx.astar_path`` on identical graphs with identical heuristics.

Heuristic convention follows networkx — ``heuristic(u, v)`` estimates the cost
from ``u`` to the goal ``v``. A* returns the optimal path as long as the
heuristic never overestimates the true remaining cost.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Callable, Hashable

import networkx as nx

from qgati.routing.dijkstra import (
    Node,
    WeightFn,
    _reconstruct_path,
    _successors,
    weight_function,
)

__all__ = [
    "astar",
    "astar_path_length",
    "haversine_m",
    "make_euclidean_heuristic",
    "make_haversine_heuristic",
    "zero_heuristic",
]

Heuristic = Callable[[Node, Node], float]

#: IUGG mean Earth radius, in metres.
EARTH_RADIUS_M = 6_371_008.8

#: Fallback speed ceiling when a graph carries no ``speed_kph`` attributes.
DEFAULT_MAX_SPEED_KPH = 120.0


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #
def zero_heuristic(u: Node, v: Node) -> float:
    """The trivial heuristic. A* with this degenerates to Dijkstra."""
    return 0.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def make_haversine_heuristic(
    graph: nx.Graph, max_speed_kph: float | None = None
) -> Heuristic:
    """Admissible heuristic for the real Delhi graph.

    Node ``(lat, lon)`` come from OSMnx's ``y``/``x`` attributes; edge weights are
    travel times in seconds. Dividing great-circle distance by the fastest speed
    anywhere in the network yields a lower bound on remaining travel time, since
    a road is never shorter than the straight line between its endpoints and no
    edge is faster than ``max_speed_kph``.

    ``max_speed_kph`` defaults to the fastest edge actually present, which keeps
    the bound tight without any hand-tuning.
    """
    positions = {
        node: (data["y"], data["x"]) for node, data in graph.nodes(data=True)
    }

    if max_speed_kph is None:
        speeds = [
            data["speed_kph"]
            for _, _, data in graph.edges(data=True)
            if data.get("speed_kph")
        ]
        max_speed_kph = max(speeds) if speeds else DEFAULT_MAX_SPEED_KPH

    max_speed_mps = max_speed_kph / 3.6

    def heuristic(u: Node, v: Node) -> float:
        lat_u, lon_u = positions[u]
        lat_v, lon_v = positions[v]
        return haversine_m(lat_u, lon_u, lat_v, lon_v) / max_speed_mps

    return heuristic


def make_euclidean_heuristic(
    graph: nx.Graph, pos_attr: str = "pos", scale: float = 1.0
) -> Heuristic:
    """Admissible heuristic for planar graphs (used by the synthetic test graph).

    ``scale`` multiplies the straight-line distance; keep it at or below the
    smallest edge ``weight / distance`` ratio or admissibility is lost. The
    synthetic generator guarantees every weight is at least the euclidean
    distance, so the default of ``1.0`` is safe there.
    """
    positions = nx.get_node_attributes(graph, pos_attr)
    if not positions:  # pragma: no cover - defensive
        raise ValueError(f"graph has no {pos_attr!r} node attribute")

    def heuristic(u: Node, v: Node) -> float:
        (x1, y1), (x2, y2) = positions[u], positions[v]
        return math.hypot(x1 - x2, y1 - y2) * scale

    return heuristic


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def astar(
    graph: nx.Graph,
    source: Node,
    target: Node,
    heuristic: Heuristic | None = None,
    weight: str | WeightFn = "weight",
) -> tuple[list[Node], float]:
    """Shortest path from ``source`` to ``target`` using A*.

    Returns ``(path, cost)`` including both endpoints. Raises
    :class:`networkx.NetworkXNoPath` when the target is unreachable and
    :class:`networkx.NodeNotFound` when an endpoint is missing.

    With an admissible heuristic the returned cost is identical to
    :func:`qgati.routing.dijkstra.dijkstra`; the heuristic only changes how much
    of the graph gets explored on the way there.
    """
    if source not in graph:
        raise nx.NodeNotFound(f"Source {source!r} is not in G")
    if target not in graph:
        raise nx.NodeNotFound(f"Target {target!r} is not in G")
    if source == target:
        return [source], 0.0

    if heuristic is None:
        heuristic = zero_heuristic

    weight_fn = weight_function(graph, weight)

    g_score: dict[Node, float] = {source: 0.0}
    predecessors: dict[Node, Node | None] = {source: None}
    closed: set[Node] = set()

    counter = itertools.count()
    # (f = g + h, g, tiebreak, node)
    queue: list[tuple[float, float, int, Node]] = [
        (heuristic(source, target), 0.0, next(counter), source)
    ]

    while queue:
        _, distance, _, u = heapq.heappop(queue)
        if u in closed:
            continue  # stale entry; a cheaper route to u was already expanded
        closed.add(u)

        if u == target:
            return _reconstruct_path(predecessors, source, target), g_score[u]

        for v, cost in _successors(graph, weight_fn, u):
            if v in closed:
                continue
            candidate = distance + cost
            if candidate < g_score.get(v, math.inf):
                g_score[v] = candidate
                predecessors[v] = u
                heapq.heappush(
                    queue,
                    (candidate + heuristic(v, target), candidate, next(counter), v),
                )

    raise nx.NetworkXNoPath(f"No path between {source!r} and {target!r}.")


def astar_path_length(
    graph: nx.Graph,
    source: Node,
    target: Node,
    heuristic: Heuristic | None = None,
    weight: str | WeightFn = "weight",
) -> float:
    """Cost of the A* path, without materialising the path itself."""
    return astar(graph, source, target, heuristic=heuristic, weight=weight)[1]
