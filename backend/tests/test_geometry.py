"""Geometry tests: the GeoJSON renderer and the route-polyline tracer.

The API tests already cover these two through ``/graph/delhi`` and
``/optimize/{id}``. What is tested here is what an HTTP round-trip cannot reach:
the branch where an edge carries its own curve rather than a straight segment,
the degenerate routes (no stops, one stop, two stops at one address), and the
bounding-box arithmetic.

Everything runs on :func:`build_synthetic_graph` — no OSM, no network, no disk.
Where a test needs an edge with real OSM-style geometry it attaches a duck-typed
stand-in carrying ``.coords``, which is all the module reads; that keeps shapely
out of the test dependencies and pins the contract to the attribute rather than
to the class.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from qgati.graph.geometry import (
    DEFAULT_PADDING_M,
    METRES_PER_DEGREE,
    bbox_around_nodes,
    graph_to_geojson,
    parse_bbox,
    route_polyline,
)
from qgati.graph.graph_builder import build_synthetic_graph

#: The synthetic layout is a 0..1000 square in arbitrary units. These rescale it
#: into a small Delhi-ish box so the longitude-scaling arithmetic in
#: :func:`bbox_around_nodes` is exercised at a realistic latitude.
LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05


@pytest.fixture(scope="module")
def graph() -> nx.DiGraph:
    """A synthetic graph tagged with OSMnx-style ``x``/``y`` coordinates."""
    road_graph = build_synthetic_graph(n_nodes=60, edge_prob=0.25, seed=7)
    for _, data in road_graph.nodes(data=True):
        x, y = data["pos"]
        data["x"] = LON_ORIGIN + (x / 1000.0) * DEGREE_SPAN
        data["y"] = LAT_ORIGIN + (y / 1000.0) * DEGREE_SPAN
    return road_graph


def position_of(graph: nx.Graph, node) -> tuple[float, float]:
    """The rounded ``(lon, lat)`` a node serialises to."""
    data = graph.nodes[node]
    return round(data["x"], 6), round(data["y"], 6)


class FakeGeometry:
    """Stands in for a shapely ``LineString``: anything with ``.coords``."""

    def __init__(self, coords):
        self.coords = coords


def direct_edge(graph: nx.Graph) -> tuple:
    """An edge of the graph, for attaching geometry to."""
    return next(iter(graph.edges()))


# --------------------------------------------------------------------------- #
# Bounding-box parsing
# --------------------------------------------------------------------------- #
def test_parse_bbox_reads_four_floats() -> None:
    assert parse_bbox("77.2,28.6,77.25,28.65") == (77.2, 28.6, 77.25, 28.65)


@pytest.mark.parametrize(
    "value",
    [
        "77.2,28.6,77.25",  # too few
        "77.2,28.6,77.25,28.65,1",  # too many
        "77.2,28.6,seven,28.65",  # not a number
        "",  # empty
    ],
)
def test_parse_bbox_rejects_malformed_input(value: str) -> None:
    with pytest.raises(ValueError, match="bbox"):
        parse_bbox(value)


def test_parse_bbox_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        parse_bbox("77.30,28.60,77.20,28.65")
    with pytest.raises(ValueError, match="must not exceed"):
        parse_bbox("77.20,28.70,77.25,28.60")


# --------------------------------------------------------------------------- #
# bbox_around_nodes
# --------------------------------------------------------------------------- #
def test_bbox_around_nodes_pads_by_metres(graph) -> None:
    node = next(iter(graph.nodes))
    lon, lat = graph.nodes[node]["x"], graph.nodes[node]["y"]

    box = bbox_around_nodes(graph, [node], padding_m=1000.0)

    assert box is not None
    min_lon, min_lat, max_lon, max_lat = box
    # Latitude padding is a plain metres-per-degree conversion.
    assert min_lat == pytest.approx(lat - 1000.0 / METRES_PER_DEGREE)
    assert max_lat == pytest.approx(lat + 1000.0 / METRES_PER_DEGREE)
    # Longitude padding is wider, because a degree of longitude is shorter here.
    assert (max_lon - min_lon) > (max_lat - min_lat)
    assert min_lon < lon < max_lon


def test_bbox_around_nodes_is_symmetric_about_the_centre(graph) -> None:
    node = next(iter(graph.nodes))
    lon, lat = graph.nodes[node]["x"], graph.nodes[node]["y"]
    min_lon, min_lat, max_lon, max_lat = bbox_around_nodes(graph, [node], 500.0)

    assert (min_lon + max_lon) / 2 == pytest.approx(lon)
    assert (min_lat + max_lat) / 2 == pytest.approx(lat)


def test_bbox_around_nodes_longitude_scaling_matches_the_latitude(graph) -> None:
    """The cos(lat) divisor, checked against the latitude it was computed at.

    The module scales by the cosine of the box's *midpoint* latitude, which is
    the right reference for a box that may span a little latitude.
    """
    node = next(iter(graph.nodes))
    min_lon, min_lat, max_lon, max_lat = bbox_around_nodes(graph, [node], 500.0)

    mid_lat = (min_lat + max_lat) / 2
    expected = 2 * 500.0 / (METRES_PER_DEGREE * math.cos(math.radians(mid_lat)))
    assert (max_lon - min_lon) == pytest.approx(expected, rel=1e-9)


def test_bbox_around_nodes_covers_every_node_it_is_given(graph) -> None:
    nodes = list(graph.nodes)[:10]
    min_lon, min_lat, max_lon, max_lat = bbox_around_nodes(graph, nodes, 0.0)

    for node in nodes:
        lon, lat = graph.nodes[node]["x"], graph.nodes[node]["y"]
        assert min_lon <= lon <= max_lon
        assert min_lat <= lat <= max_lat


def test_bbox_around_nodes_returns_none_without_coordinates() -> None:
    """A coordinate-less graph yields no box, which the caller reads as "do not
    filter" — distinct from an empty result."""
    bare = nx.DiGraph()
    bare.add_edge("a", "b", weight=1.0)
    assert bbox_around_nodes(bare, ["a", "b"]) is None


def test_default_padding_is_a_few_blocks() -> None:
    assert 100.0 <= DEFAULT_PADDING_M <= 1000.0


# --------------------------------------------------------------------------- #
# graph_to_geojson
# --------------------------------------------------------------------------- #
def test_every_edge_becomes_a_line_string(graph) -> None:
    collection = graph_to_geojson(graph, include_nodes=False)
    features = collection["features"]

    assert collection["type"] == "FeatureCollection"
    assert len(features) == graph.number_of_edges()
    for feature in features:
        assert feature["geometry"]["type"] == "LineString"
        assert len(feature["geometry"]["coordinates"]) >= 2


def test_every_node_becomes_a_point(graph) -> None:
    collection = graph_to_geojson(graph)
    points = [
        f for f in collection["features"] if f["geometry"]["type"] == "Point"
    ]
    assert len(points) == graph.number_of_nodes()
    assert {f["properties"]["id"] for f in points} == set(graph.nodes)


def test_an_edge_without_geometry_is_a_straight_segment(graph) -> None:
    """The common case on this extract: no ``geometry``, so endpoints are used."""
    source, target = direct_edge(graph)
    collection = graph_to_geojson(graph, include_nodes=False)
    feature = next(
        f
        for f in collection["features"]
        if f["properties"]["u"] == source and f["properties"]["v"] == target
    )
    assert feature["geometry"]["coordinates"] == [
        list(position_of(graph, source)),
        list(position_of(graph, target)),
    ]


def test_an_edge_with_geometry_is_drawn_as_its_curve(graph) -> None:
    """Where OSMnx collapsed several ways into one edge, the curve is what draws.

    This is the branch that makes curved roads render as curves rather than as
    chords across the block.
    """
    source, target = direct_edge(graph)
    start = position_of(graph, source)
    end = position_of(graph, target)
    midpoint = ((start[0] + end[0]) / 2 + 0.0004, (start[1] + end[1]) / 2)
    graph[source][target]["geometry"] = FakeGeometry([start, midpoint, end])

    try:
        collection = graph_to_geojson(graph, include_nodes=False)
        feature = next(
            f
            for f in collection["features"]
            if f["properties"]["u"] == source and f["properties"]["v"] == target
        )
        coordinates = feature["geometry"]["coordinates"]
        assert len(coordinates) == 3
        assert tuple(coordinates[1]) == (round(midpoint[0], 6), round(midpoint[1], 6))
    finally:
        del graph[source][target]["geometry"]


def test_edge_features_carry_the_endpoints_and_travel_time(graph) -> None:
    """``u``/``v`` are what let the frontend highlight a solved route's roads."""
    collection = graph_to_geojson(graph, include_nodes=False)
    for feature in collection["features"][:20]:
        properties = feature["properties"]
        assert properties["u"] in graph
        assert properties["v"] in graph
        assert properties["travel_time"] >= 0


def test_geojson_clips_to_a_bbox_keeping_boundary_edges(graph) -> None:
    """An edge is kept when *any* of its coordinates fall inside — so a road
    crossing the boundary is not truncated mid-block."""
    node = next(iter(graph.nodes))
    lon, lat = graph.nodes[node]["x"], graph.nodes[node]["y"]
    box = (lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002)

    collection = graph_to_geojson(graph, bbox=box)
    assert collection["features"]

    for feature in collection["features"]:
        if feature["geometry"]["type"] != "Point":
            continue
        point_lon, point_lat = feature["geometry"]["coordinates"]
        assert box[0] <= point_lon <= box[2]
        assert box[1] <= point_lat <= box[3]


def test_geojson_clipping_never_adds_features(graph) -> None:
    whole = graph_to_geojson(graph)["features"]
    node = next(iter(graph.nodes))
    lon, lat = graph.nodes[node]["x"], graph.nodes[node]["y"]
    clipped = graph_to_geojson(
        graph, bbox=(lon - 0.003, lat - 0.003, lon + 0.003, lat + 0.003)
    )["features"]

    assert 0 < len(clipped) < len(whole)


def test_geojson_on_an_empty_box_returns_no_features(graph) -> None:
    # A box in the Atlantic: valid, and containing none of this network.
    collection = graph_to_geojson(graph, bbox=(-10.0, -10.0, -9.0, -9.0))
    assert collection["features"] == []


def test_geojson_skips_nodes_carrying_no_coordinates() -> None:
    bare = nx.DiGraph()
    bare.add_node(1, x=77.2, y=28.6)
    bare.add_node(2)  # no coordinates
    bare.add_edge(1, 2, weight=1.0)

    collection = graph_to_geojson(bare)
    # The edge has one drawable endpoint and no geometry, so it is dropped
    # rather than emitted as a half-line; only the coordinate-carrying node
    # survives.
    assert len(collection["features"]) == 1
    assert collection["features"][0]["properties"]["id"] == 1


# --------------------------------------------------------------------------- #
# route_polyline
# --------------------------------------------------------------------------- #
def test_polyline_walks_real_edges_between_stops(graph) -> None:
    """The point of the whole module: consecutive points are road-adjacent."""
    nodes = list(graph.nodes)
    stops = [node for node in nodes if node in graph][:3]
    adjacency = {(u, v) for u, v in graph.edges()}

    polyline = route_polyline(graph, [stops[0], stops[1], stops[2], stops[0]])
    at = {position_of(graph, node): node for node in graph.nodes}

    assert len(polyline) >= 4
    for start, end in zip(polyline, polyline[1:]):
        assert (at[tuple(start)], at[tuple(end)]) in adjacency


def test_polyline_starts_and_ends_at_the_depot(graph) -> None:
    nodes = list(graph.nodes)
    depot, stop = nodes[0], nodes[1]
    polyline = route_polyline(graph, [depot, stop, depot])

    assert tuple(polyline[0]) == position_of(graph, depot)
    assert tuple(polyline[-1]) == position_of(graph, depot)


def test_polyline_deduplicates_the_joint_between_two_legs(graph) -> None:
    """Each joint stop would otherwise appear twice — once ending a leg, once
    starting the next — and the line would carry a duplicate point."""
    nodes = list(graph.nodes)
    depot, stop = nodes[0], nodes[1]
    polyline = route_polyline(graph, [depot, stop, depot])

    for start, end in zip(polyline, polyline[1:]):
        assert start != end


def test_polyline_of_a_single_node_is_that_node(graph) -> None:
    node = next(iter(graph.nodes))
    assert route_polyline(graph, [node]) == [list(position_of(graph, node))]


def test_polyline_of_nothing_is_empty(graph) -> None:
    """An unused vehicle: no stops, so nothing to draw."""
    assert route_polyline(graph, []) == []


def test_polyline_of_one_stop_visited_twice_is_degenerate_but_valid(graph) -> None:
    """A route that only touches the depot has no distance to draw."""
    depot = next(iter(graph.nodes))
    assert route_polyline(graph, [depot, depot]) == [list(position_of(graph, depot))]


def test_polyline_handles_the_same_stop_twice(graph) -> None:
    """Two deliveries at one address: the second leg is a self-leg, and the
    polyline must not gain a spurious jump."""
    nodes = list(graph.nodes)
    depot, stop = nodes[0], nodes[1]
    polyline = route_polyline(graph, [depot, stop, stop, depot])

    assert tuple(polyline[0]) == position_of(graph, depot)
    assert tuple(polyline[-1]) == position_of(graph, depot)
    assert position_of(graph, stop) in [tuple(point) for point in polyline]


def test_polyline_cost_matches_the_leg_costs_it_was_built_from(graph) -> None:
    """Geometry and accounting agree: summing the drawn line's edge weights
    equals the route cost the optimizer minimised."""
    from qgati.routing.dijkstra import dijkstra_all_pairs

    nodes = list(graph.nodes)
    depot, first, second = nodes[0], nodes[1], nodes[2]
    route = [depot, first, second, depot]

    table = dijkstra_all_pairs(graph, route)
    expected = sum(table[(a, b)][1] for a, b in zip(route, route[1:]))

    polyline = route_polyline(graph, route)
    at = {position_of(graph, node): node for node in graph.nodes}
    weights = {
        (u, v): data["weight"] for u, v, data in graph.edges(data=True)
    }
    drawn = sum(
        weights[(at[tuple(start)], at[tuple(end)])]
        for start, end in zip(polyline, polyline[1:])
    )

    assert drawn == pytest.approx(expected)
