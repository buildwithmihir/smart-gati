"""Shortest-path routing over the road graph (Dijkstra, A*).

Both algorithms are original implementations, kept independent of networkx's so
that the two can be differentially tested against each other.

    >>> from qgati.routing import dijkstra, astar, make_haversine_heuristic
    >>> path, cost = dijkstra(graph, source, target)
"""

from qgati.routing.astar import (
    astar,
    astar_path_length,
    haversine_m,
    make_euclidean_heuristic,
    make_haversine_heuristic,
    zero_heuristic,
)
from qgati.routing.dijkstra import (
    dijkstra,
    dijkstra_all_pairs,
    dijkstra_path_length,
)

__all__ = [
    "astar",
    "astar_path_length",
    "dijkstra",
    "dijkstra_all_pairs",
    "dijkstra_path_length",
    "haversine_m",
    "make_euclidean_heuristic",
    "make_haversine_heuristic",
    "zero_heuristic",
]
