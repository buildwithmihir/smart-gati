"""Road graph construction: build and cache the Delhi network from OSMnx/NetworkX.

    >>> from qgati.graph import load_delhi_graph, build_synthetic_graph
    >>> graph = load_delhi_graph()          # cached after the first fetch
    >>> toy = build_synthetic_graph(50, 0.1, seed=0)   # offline, for tests
"""

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
    "build_synthetic_graph",
    "delhi_graph_cache_path",
    "is_delhi_graph_cached",
    "largest_strongly_connected_subgraph",
    "load_delhi_graph",
]
