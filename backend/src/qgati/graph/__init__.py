"""Road graph construction: build and cache the Delhi network from OSMnx/NetworkX.

    >>> from qgati.graph import load_delhi_graph, build_synthetic_graph
    >>> graph = load_delhi_graph()          # cached after the first fetch
    >>> toy = build_synthetic_graph(50, 0.1, seed=0)   # offline, for tests
    >>> costs = build_cost_matrix(graph, scenario)     # collapse to a dense array

:func:`build_cost_matrix` is the hand-off point to the optimizer: past it,
nothing touches the road graph again.
"""

from qgati.graph.cost_matrix import (
    CostMatrix,
    CostMatrixBuild,
    build_cost_matrix,
    build_cost_matrix_detailed,
)
from qgati.graph.geometry import (
    DEFAULT_PADDING_M,
    bbox_around_nodes,
    graph_to_geojson,
    parse_bbox,
    route_polyline,
)
from qgati.graph.graph_builder import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DIST_M,
    DELHI_CENTER,
    build_synthetic_graph,
    delhi_graph_cache_path,
    is_delhi_graph_cached,
    largest_strongly_connected_subgraph,
    load_delhi_graph,
    nearest_node,
    node_coordinates,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DIST_M",
    "DEFAULT_PADDING_M",
    "DELHI_CENTER",
    "CostMatrix",
    "CostMatrixBuild",
    "bbox_around_nodes",
    "build_cost_matrix",
    "build_cost_matrix_detailed",
    "build_synthetic_graph",
    "delhi_graph_cache_path",
    "graph_to_geojson",
    "is_delhi_graph_cached",
    "largest_strongly_connected_subgraph",
    "load_delhi_graph",
    "nearest_node",
    "node_coordinates",
    "parse_bbox",
    "route_polyline",
]
