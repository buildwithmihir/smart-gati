"""Road graph construction: build and cache the Delhi network from OSMnx/NetworkX.

    >>> from qgati.graph import load_delhi_graph, build_synthetic_graph
    >>> graph = load_delhi_graph()          # cached after the first fetch
    >>> toy = build_synthetic_graph(50, 0.1, seed=0)   # offline, for tests
    >>> costs = build_cost_matrix(graph, scenario)     # collapse to a dense array

:func:`build_cost_matrix` is the hand-off point to the optimizer: past it,
nothing touches the road graph again.
"""

from qgati.graph.cost_matrix import CostMatrix, build_cost_matrix
from qgati.graph.graph_builder import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DIST_M,
    DELHI_CENTER,
    build_synthetic_graph,
    delhi_graph_cache_path,
    is_delhi_graph_cached,
    largest_strongly_connected_subgraph,
    load_delhi_graph,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DIST_M",
    "DELHI_CENTER",
    "CostMatrix",
    "build_cost_matrix",
    "build_synthetic_graph",
    "delhi_graph_cache_path",
    "is_delhi_graph_cached",
    "largest_strongly_connected_subgraph",
    "load_delhi_graph",
]
