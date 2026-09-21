"""GeoJSON for the road network, and real-road polylines for a solved route.

Why this exists
---------------
Everything up to here speaks in costs: a cost matrix is an ``(n, n)`` array of
travel times, and a :class:`~qgati.optimizer.models.Solution` is a list of
*delivery indices*. Neither can be drawn. A map needs coordinates, and a route
drawn as straight lines between consecutive stops is wrong in a way that is
obvious to anyone who knows the city — it cuts through blocks, ignores one-way
streets, and reports a shape that the optimizer never chose.

So this module closes the loop back to geometry. It is the one place that reads
the road graph *after* the cost matrix has been built, and it does so read-only:
it never feeds anything back into routing or search.

Two products
------------
:func:`graph_to_geojson`
    The street network as a GeoJSON ``FeatureCollection`` — the basemap. Scoped
    by bounding box or by scenario, because the full Delhi extract is ~1 MB and a
    map of one delivery round does not need the other 4750 edges.
:func:`route_polyline`
    One route's stop sequence turned into the actual road polyline, by stitching
    the Dijkstra legs between consecutive stops.

Both return ``[lon, lat]`` pairs, which is what GeoJSON specifies — note that
this is the *opposite* order to the ``(lat, lon)`` the rest of the project uses
for coordinates.

Geometry is what OSMnx says it is
---------------------------------
On the Delhi extract, 2339 of 4778 edges carry an explicit ``geometry``
``LineString`` (the ones that ``simplify=True`` collapsed from several OSM ways);
the rest are a straight segment between their endpoints. Both cases are handled,
and the presence of geometry is duck-typed rather than checked with an
``isinstance`` against shapely, so the module works on any graph whose edges
carry a ``.coords`` attribute — and on the synthetic test graphs, which carry
none at all.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Hashable, Iterable, Sequence

import networkx as nx

from qgati.graph.graph_builder import node_coordinates
from qgati.routing.dijkstra import dijkstra_all_pairs

if TYPE_CHECKING:  # avoids an optimizer <-> graph import cycle at runtime
    from qgati.optimizer.models import Scenario

__all__ = [
    "DEFAULT_PADDING_M",
    "METRES_PER_DEGREE",
    "bbox_around_nodes",
    "graph_to_geojson",
    "parse_bbox",
    "route_polyline",
]

Node = Hashable

#: Padding added around a scenario's stops when scoping the basemap to it, in
#: metres. Enough that the streets a route might use are all present, without
#: dragging in half the city.
DEFAULT_PADDING_M = 400.0

#: One degree of latitude, in metres — good enough at Delhi's scale (the
#: meridional degree varies by ~0.3% between the equator and 30 degrees).
METRES_PER_DEGREE = 111_320.0

#: Coordinates are rounded to 6 decimal places before serialising. That is ~11 cm
#: at Delhi's latitude — far finer than the road network's own accuracy, and it
#: keeps the payload from carrying 16 digits of float noise per coordinate.
COORDINATE_PRECISION = 6


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def _lonlat(graph: nx.Graph, node: Node) -> tuple[float, float] | None:
    """``(lon, lat)`` for a node, or ``None`` if it carries no coordinates."""
    return node_coordinates(graph.nodes[node])


def _round_coordinate(coordinate: tuple[float, float]) -> list[float]:
    lon, lat = coordinate
    return [round(lon, COORDINATE_PRECISION), round(lat, COORDINATE_PRECISION)]


def _edge_positions(
    graph: nx.Graph, source: Node, target: Node, data: dict
) -> list[list[float]]:
    """The polyline for one edge, using its own geometry when it has one.

    An edge that ``simplify=True`` collapsed out of several OSM ways carries a
    ``geometry`` attribute describing the real curve; the rest are straight
    between their endpoints. Duck-typed on ``.coords`` so this needs no shapely
    import and degrades to the straight segment for anything else.
    """
    geometry = data.get("geometry")
    coords = getattr(geometry, "coords", None)
    if coords is not None:
        positions = [_round_coordinate((float(x), float(y))) for x, y in coords]
        if len(positions) >= 2:
            return positions

    source_position = _lonlat(graph, source)
    target_position = _lonlat(graph, target)
    if source_position is None or target_position is None:
        return []
    return [_round_coordinate(source_position), _round_coordinate(target_position)]


# --------------------------------------------------------------------------- #
# Bounding boxes
# --------------------------------------------------------------------------- #
#: ``(min_lon, min_lat, max_lon, max_lat)``, the GeoJSON/``bbox`` convention.
BBox = tuple[float, float, float, float]


def parse_bbox(value: str) -> BBox:
    """Parse ``"min_lon,min_lat,max_lon,max_lat"`` from a query string.

    Raises
    ------
    ValueError
        If it is not four numbers, or the min/max pairs are inverted.
    """
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(
            "bbox must have four comma-separated numbers "
            "'min_lon,min_lat,max_lon,max_lat'"
        )
    try:
        min_lon, min_lat, max_lon, max_lat = (float(part) for part in parts)
    except ValueError:
        raise ValueError(f"bbox contains a non-numeric value: {value!r}") from None

    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError(
            "bbox min values must not exceed their max: "
            f"got {min_lon},{min_lat},{max_lon},{max_lat}"
        )
    return min_lon, min_lat, max_lon, max_lat


def bbox_around_nodes(
    graph: nx.Graph, nodes: Iterable[Node], padding_m: float = DEFAULT_PADDING_M
) -> BBox | None:
    """A bounding box covering ``nodes``, padded by ``padding_m`` metres.

    Returns ``None`` when none of the nodes carry coordinates, which lets a
    caller treat "no box" as "do not filter" rather than as an empty result.

    Longitude padding is divided by ``cos(latitude)`` because a degree of
    longitude shrinks towards the poles — without that, a padded box at Delhi's
    latitude would be ~13% too narrow.
    """
    positions = [
        position
        for node in nodes
        if (position := _lonlat(graph, node)) is not None
    ]
    if not positions:
        return None

    longitudes = [lon for lon, _ in positions]
    latitudes = [lat for _, lat in positions]
    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)

    latitude_padding = padding_m / METRES_PER_DEGREE
    # Guard the divisor: cos(90 deg) is 0, and a degenerate box is worse than a
    # slightly over-wide one.
    cosine = max(math.cos(math.radians((min_lat + max_lat) / 2.0)), 1e-6)
    longitude_padding = padding_m / (METRES_PER_DEGREE * cosine)

    return (
        min_lon - longitude_padding,
        min_lat - latitude_padding,
        max_lon + longitude_padding,
        max_lat + latitude_padding,
    )


def _within(position: tuple[float, float] | None, bbox: BBox | None) -> bool:
    """True when a coordinate is inside ``bbox``; unfiltered boxes pass all."""
    if bbox is None:
        return True
    if position is None:
        return False
    lon, lat = position
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


# --------------------------------------------------------------------------- #
# The network as GeoJSON
# --------------------------------------------------------------------------- #
def graph_to_geojson(
    graph: nx.Graph,
    bbox: BBox | None = None,
    include_nodes: bool = True,
) -> dict:
    """Render the road network as a GeoJSON ``FeatureCollection``.

    Parameters
    ----------
    graph
        Road graph whose nodes carry coordinates.
    bbox
        ``(min_lon, min_lat, max_lon, max_lat)`` to clip to. An edge is kept when
        *any* of its coordinates fall inside, so roads crossing the boundary are
        not truncated mid-block. ``None`` returns the whole network.
    include_nodes
        Emit the nodes as ``Point`` features as well as the edges. Useful for
        drawing junctions and for highlighting a stop's exact node; the map
        renders fine without them.

    Returns
    -------
    dict
        A GeoJSON ``FeatureCollection``. Edge features carry ``u``/``v``/``travel_time``
        properties — ``u``/``v`` so the frontend can highlight the edges a solved
        route actually used, ``travel_time`` for a future traffic overlay. Node
        features carry ``id``.

    Notes
    -----
    Every edge is emitted, including the reverse twin of a two-way street. Those
    are geometrically redundant but *topologically* meaningful: Delhi's network is
    asymmetric (the README measures ~12% relative asymmetry), so collapsing
    direction would misrepresent the thing this project routes on. Scope the
    request instead of deduplicating.
    """
    features: list[dict] = []

    for source, target, data in _edges_with_data(graph):
        positions = _edge_positions(graph, source, target, data)
        if not positions:
            continue
        if bbox is not None and not any(
            _within((lon, lat), bbox) for lon, lat in positions
        ):
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "u": source,
                    "v": target,
                    "travel_time": round(float(data.get("weight", 0.0)), 3),
                },
                "geometry": {"type": "LineString", "coordinates": positions},
            }
        )

    if include_nodes:
        for node, data in graph.nodes(data=True):
            position = _lonlat(graph, node)
            if position is None or not _within(position, bbox):
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {"id": node},
                    "geometry": {
                        "type": "Point",
                        "coordinates": _round_coordinate(position),
                    },
                }
            )

    return {"type": "FeatureCollection", "features": features}


def _edges_with_data(graph: nx.Graph) -> Iterable[tuple[Node, Node, dict]]:
    """Yield ``(u, v, data)`` for every edge, multigraph or not."""
    if graph.is_multigraph():
        for source, target, _key, data in graph.edges(keys=True, data=True):
            yield source, target, data
    else:
        for source, target, data in graph.edges(data=True):
            yield source, target, data


# --------------------------------------------------------------------------- #
# Route polylines
# --------------------------------------------------------------------------- #
def route_polyline(
    graph: nx.Graph,
    nodes: Sequence[Node],
    weight: str = "weight",
) -> list[list[float]]:
    """The road polyline for one route, as ``[[lon, lat], ...]``.

    ``nodes`` is the route in visit order, with the depot at *both* ends — the
    caller's job, because a :class:`~qgati.optimizer.models.Solution` route lists
    stops only and the depot is implicit.

    Consecutive stops are joined by the same shortest path the cost matrix was
    built from, so the drawn route is the route that was costed: the polyline
    length in travel time equals the leg cost the optimizer minimised. Joint
    stops are de-duplicated, so the result reads as one continuous line rather
    than a chain of overlapping legs.

    Returns an empty list when the route has no drawable nodes.
    """
    stops = list(nodes)
    if not stops:
        return []

    if len(stops) == 1:
        position = _lonlat(graph, stops[0])
        return [_round_coordinate(position)] if position is not None else []

    # One Dijkstra per distinct source, covering every target in this route —
    # rather than one per leg. Repeated nodes (two deliveries at one address) are
    # collapsed, which also avoids a self-leg.
    table = dijkstra_all_pairs(graph, dict.fromkeys(stops), weight=weight)

    polyline: list[list[float]] = []
    for source, target in zip(stops, stops[1:]):
        leg, _cost = table[(source, target)]
        for node in leg:
            position = _lonlat(graph, node)
            if position is None:
                continue
            point = _round_coordinate(position)
            if polyline and polyline[-1] == point:
                continue  # the joint between two legs, already drawn
            polyline.append(point)

    return polyline
