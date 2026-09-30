"""Delhi road-network construction, travel-time weighting, and disk caching.

Two graph sources live here:

* :func:`load_delhi_graph` — the real Delhi street network from OpenStreetMap,
  fetched once through OSMnx and cached to ``backend/data/cache/*.graphml`` so
  later runs never touch the network.
* :func:`build_synthetic_graph` — a small random graph with no I/O at all, used
  by the test suite so tests stay fast and offline.

Both produce graphs whose edge ``weight`` is a routing cost. For the Delhi graph
that cost is **travel time in seconds** (not distance), so the optimizer reasons
about minutes on the road rather than metres.
"""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path

import networkx as nx
import osmnx as ox

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DELHI_CENTER",
    "DEFAULT_DIST_M",
    "DEFAULT_CACHE_DIR",
    "build_synthetic_graph",
    "delhi_graph_cache_path",
    "is_delhi_graph_cached",
    "largest_strongly_connected_subgraph",
    "load_delhi_graph",
    "nearest_node",
    "node_coordinates",
]

# Connaught Place — roughly the geographic centre of Delhi. We deliberately
# fetch a few km^2 around it rather than the whole city: a full Delhi drive
# graph is ~100k nodes, far too slow to iterate on.
DELHI_CENTER: tuple[float, float] = (28.6315, 77.2167)

# 2000 m => ~4 km x 4 km bounding box, which lands in the high-hundreds to
# low-thousands of nodes. Bump this once the pipeline is fast enough.
DEFAULT_DIST_M = 2000
DEFAULT_NETWORK_TYPE = "drive"

_CACHE_TEMPLATE = "delhi_drive_{dist}m.graphml"

# backend/data/ — src/qgati/graph/graph_builder.py -> backend/
_BACKEND_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = Path(os.environ.get("SMART_GATI_DATA_DIR", _BACKEND_DIR / "data"))
DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR / "cache"


# --------------------------------------------------------------------------- #
# Caching helpers
# --------------------------------------------------------------------------- #
def delhi_graph_cache_path(
    dist: int = DEFAULT_DIST_M, cache_dir: str | Path | None = None
) -> Path:
    """Return the GraphML path the Delhi graph is cached at for ``dist``."""
    directory = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return directory / _CACHE_TEMPLATE.format(dist=dist)


def is_delhi_graph_cached(
    dist: int = DEFAULT_DIST_M, cache_dir: str | Path | None = None
) -> bool:
    """True when a cached Delhi graph exists on disk (i.e. no fetch needed)."""
    return delhi_graph_cache_path(dist, cache_dir).exists()


# --------------------------------------------------------------------------- #
# Travel-time weighting
# --------------------------------------------------------------------------- #
#: Used only when OSM tagging is too sparse for osmnx to infer a speed.
_FALLBACK_KPH = 30.0


def _set_edge_attribute(graph: nx.Graph, source: str, target: str) -> None:
    """Copy an edge attribute across all edges (parallel edges included)."""
    if graph.is_multigraph():
        for _, _, _, data in graph.edges(keys=True, data=True):
            data[target] = data[source]
    else:
        for _, _, data in graph.edges(data=True):
            data[target] = data[source]


def _ensure_travel_times(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Guarantee every edge carries ``speed_kph``, ``travel_time`` and ``weight``.

    ``travel_time`` is seconds; ``weight`` mirrors it so that routing functions
    work with their networkx-compatible defaults. Idempotent, so it is safe to
    run after every load — including after a GraphML round-trip, which is the
    cheapest way to be sure derived columns are present.
    """
    try:
        graph = ox.routing.add_edge_speeds(graph)
        graph = ox.routing.add_edge_travel_times(graph)
    except Exception:  # pragma: no cover - depends on OSM tagging quality
        LOGGER.warning(
            "osmnx speed helpers failed; falling back to a flat %s km/h", _FALLBACK_KPH
        )
        for _, _, _, data in graph.edges(keys=True, data=True):
            data["speed_kph"] = _FALLBACK_KPH
            data["travel_time"] = data.get("length", 0.0) / (_FALLBACK_KPH / 3.6)

    _set_edge_attribute(graph, "travel_time", "weight")
    return graph


# --------------------------------------------------------------------------- #
# Real Delhi graph
# --------------------------------------------------------------------------- #
#: OSMnx caches raw Overpass responses in a folder of its own choosing. Its
#: default is the relative path ``./cache``, which resolves against whatever the
#: process working directory happens to be — scattering ~1.5 MB dumps around the
#: repo. We pin it alongside our own graph cache instead.
OSM_HTTP_CACHE_DIRNAME = "osm_http"


def _point_osmnx_cache_at(cache_dir: Path) -> None:
    """Redirect osmnx's HTTP response cache beneath ``cache_dir``."""
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(Path(cache_dir) / OSM_HTTP_CACHE_DIRNAME)


def load_delhi_graph(
    dist: int = DEFAULT_DIST_M,
    center: tuple[float, float] = DELHI_CENTER,
    network_type: str = DEFAULT_NETWORK_TYPE,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> nx.MultiDiGraph:
    """Load the Delhi drive network, fetching from OSM only if not already cached.

    Parameters
    ----------
    dist
        Half-width of the bounding box in metres around ``center``.
    center
        ``(lat, lon)`` of the area centre.
    network_type
        OSMnx network filter; ``"drive"`` gives the drivable road network.
    cache_dir
        Override the cache location (defaults to ``backend/data/cache``).
    force_refresh
        Ignore any cached file and re-fetch from OpenStreetMap.

    Returns
    -------
    networkx.MultiDiGraph
        Nodes carry ``x``/``y`` (lon/lat). Edges carry ``length`` (metres),
        ``speed_kph``, ``travel_time`` (seconds) and ``weight`` (== travel_time).
    """
    cache_path = delhi_graph_cache_path(dist, cache_dir)

    if cache_path.exists() and not force_refresh:
        LOGGER.info("Loading Delhi graph from cache: %s", cache_path)
        graph = ox.load_graphml(cache_path)
    else:
        LOGGER.info(
            "Fetching Delhi graph from OSM: dist=%sm around %s", dist, center
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _point_osmnx_cache_at(cache_path.parent)
        graph = ox.graph_from_point(
            center, dist=dist, network_type=network_type, simplify=True
        )
        ox.save_graphml(graph, cache_path)
        LOGGER.info("Cached Delhi graph to %s", cache_path)

    return _ensure_travel_times(graph)


def largest_strongly_connected_subgraph(graph: nx.Graph) -> nx.Graph:
    """Return the largest mutually-reachable component.

    OSM extracts routinely contain one-way stubs that can be entered but not
    left. Every node in the result can reach every other, which is what the VRP
    layer needs to guarantee a feasible tour exists.
    """
    component = max(nx.strongly_connected_components(graph), key=len)
    return graph.subgraph(component).copy()


def node_coordinates(data: dict) -> tuple[float, float] | None:
    """Read ``(lon, lat)`` out of a node's attribute dict, or ``None``.

    Two layouts are understood: OSMnx's ``x``/``y``, and the ``pos`` ``(x, y)``
    tuple that :func:`build_synthetic_graph` writes for tests.
    """
    if "x" in data and "y" in data:  # OSMnx
        return float(data["x"]), float(data["y"])
    if "pos" in data:  # synthetic: pos is (x, y)
        x, y = data["pos"]
        return float(x), float(y)
    return None


def nearest_node(graph: nx.Graph, lat: float, lon: float) -> object:
    """Snapshot the road-graph node nearest a coordinate.

    This is how a client that speaks in coordinates — a map pin, a delivery
    address — enters the graph-routing world, which speaks in node ids.

    Deliberately a plain linear scan rather than a spatial index. Scenarios have
    a handful of stops and the Delhi graph a couple of thousand nodes, so this
    costs microseconds, and it works on *any* graph carrying coordinates —
    including the synthetic test graphs, which declare no CRS for a spatial index
    to interpret. Distances are compared in an equirectangular approximation
    (longitude scaled by ``cos(lat)``), which is more than accurate enough to
    choose between nodes metres apart.

    Note the snapped node is not guaranteed to be in the largest strongly-
    connected subgraph: a coordinate can land closest to a one-way stub. Scenario
    builders should prefer :func:`~qgati.optimizer.scenarios.servable_nodes`.

    Raises
    ------
    ValueError
        If the graph has no nodes carrying coordinates.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("cannot snap a coordinate onto an empty graph")

    longitude_scale = math.cos(math.radians(lat))
    best_node: object | None = None
    best_distance = math.inf

    for node, data in graph.nodes(data=True):
        coordinates = node_coordinates(data)
        if coordinates is None:
            continue
        node_lon, node_lat = coordinates
        delta_lon = (node_lon - lon) * longitude_scale
        delta_lat = node_lat - lat
        distance = delta_lon * delta_lon + delta_lat * delta_lat
        if distance < best_distance:
            best_distance, best_node = distance, node

    if best_node is None:
        raise ValueError("no node in this graph carries coordinates to snap onto")
    return best_node


# --------------------------------------------------------------------------- #
# Synthetic graph (tests)
# --------------------------------------------------------------------------- #
def build_synthetic_graph(
    n_nodes: int = 60,
    edge_prob: float = 0.12,
    seed: int = 42,
    directed: bool = True,
    multigraph: bool = False,
    area: float = 1000.0,
    min_detour: float = 1.0,
    max_detour: float = 2.0,
) -> nx.Graph:
    """Build a small random graph for tests — no network, no OSM, deterministic.

    Nodes are scattered uniformly in a square of side ``area`` and tagged with a
    ``pos`` ``(x, y)`` attribute. An edge's ``weight`` is the euclidean distance
    between its endpoints multiplied by a detour factor in
    ``[min_detour, max_detour]``.

    Keeping ``min_detour >= 1.0`` means every weight is at least the straight-line
    distance, which makes the euclidean distance an *admissible* A* heuristic —
    the same property the real Delhi graph has (road length >= great-circle
    distance), so the two exercise the same code path.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if not 0.0 < edge_prob <= 1.0:
        raise ValueError("edge_prob must be in (0, 1]")
    if min_detour < 1.0:
        raise ValueError("min_detour must be >= 1.0 to keep the heuristic admissible")

    rng = random.Random(seed)

    if multigraph:
        graph: nx.Graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
    else:
        graph = nx.DiGraph() if directed else nx.Graph()

    for node in range(n_nodes):
        graph.add_node(node, pos=(rng.uniform(0.0, area), rng.uniform(0.0, area)))

    def add_edge(u: int, v: int, key: int = 0) -> None:
        length = math.dist(graph.nodes[u]["pos"], graph.nodes[v]["pos"])
        weight = length * rng.uniform(min_detour, max_detour)
        if multigraph:
            graph.add_edge(u, v, key=key, length=length, weight=weight)
        else:
            graph.add_edge(u, v, length=length, weight=weight)

    for u in range(n_nodes):
        for v in range(n_nodes):
            if u == v:
                continue
            if not directed and v < u:
                continue
            if rng.random() >= edge_prob:
                continue
            add_edge(u, v)
            # Occasionally add a parallel edge with an independent weight, so the
            # "collapse parallel edges to the cheapest" logic gets exercised.
            if multigraph and rng.random() < 0.25:
                add_edge(u, v, key=1)

    return graph
