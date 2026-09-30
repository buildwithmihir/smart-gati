"""Routing correctness tests.

The heavy lifting is a differential comparison: our from-scratch Dijkstra and A*
are run against networkx's implementations on the same graphs with the same
weights and heuristics, and the costs must agree exactly (to 1e-9).

Everything here uses :func:`build_synthetic_graph` — no OSM, no network, no
disk. The single real-Delhi test at the bottom is opt-in via ``SMART_GATI_RUN_SLOW``.
"""

from __future__ import annotations

import math
import os
import random

import networkx as nx
import pytest

from qgati.graph.graph_builder import (
    build_synthetic_graph,
    is_delhi_graph_cached,
    largest_strongly_connected_subgraph,
    load_delhi_graph,
)
from qgati.routing.astar import (
    astar,
    astar_path_length,
    haversine_m,
    make_euclidean_heuristic,
    make_haversine_heuristic,
    zero_heuristic,
)
from qgati.routing.dijkstra import dijkstra, dijkstra_all_pairs, dijkstra_path_length

TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic() -> nx.DiGraph:
    """A 60-node random digraph with planar positions."""
    return build_synthetic_graph(n_nodes=60, edge_prob=0.12, seed=42)


@pytest.fixture(scope="module")
def synthetic_multigraph() -> nx.MultiDiGraph:
    """Same, but with parallel edges so the collapse-to-cheapest logic is used."""
    return build_synthetic_graph(
        n_nodes=60, edge_prob=0.12, seed=7, multigraph=True
    )


def reachable_pairs(
    graph: nx.Graph, count: int, seed: int
) -> list[tuple[int, int]]:
    """Sample ``count`` distinct ``(source, target)`` pairs that are connected.

    Unreachable pairs are dropped rather than asserted on, because a random
    digraph makes no connectivity promise; the unreachable case is covered
    separately by an explicit test.
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes)
    pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(pairs) < count and attempts < count * 200:
        attempts += 1
        source, target = rng.sample(nodes, 2)
        if nx.has_path(graph, source, target):
            pairs.append((source, target))
    assert pairs, "synthetic graph had no connected pairs — generator is broken"
    return pairs


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #
def test_synthetic_graph_shape(synthetic: nx.DiGraph) -> None:
    assert synthetic.number_of_nodes() == 60
    assert synthetic.number_of_edges() > 0
    assert all(
        synthetic.nodes[n]["pos"][0] is not None for n in synthetic.nodes
    )
    # Every weight must dominate the straight-line distance — that is what makes
    # the euclidean heuristic admissible.
    for u, v, data in synthetic.edges(data=True):
        straight = math.dist(
            synthetic.nodes[u]["pos"], synthetic.nodes[v]["pos"]
        )
        assert data["weight"] >= straight - 1e-12


def test_synthetic_graph_is_deterministic() -> None:
    first = build_synthetic_graph(n_nodes=30, seed=99)
    second = build_synthetic_graph(n_nodes=30, seed=99)
    assert nx.utils.graphs_equal(first, second)


def test_synthetic_graph_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        build_synthetic_graph(n_nodes=1)
    with pytest.raises(ValueError):
        build_synthetic_graph(edge_prob=0.0)
    with pytest.raises(ValueError):
        build_synthetic_graph(min_detour=0.5)


# --------------------------------------------------------------------------- #
# Dijkstra vs networkx
# --------------------------------------------------------------------------- #
def test_dijkstra_matches_networkx(synthetic: nx.DiGraph) -> None:
    pairs = reachable_pairs(synthetic, count=25, seed=1)
    for source, target in pairs:
        path, cost = dijkstra(synthetic, source, target)
        expected_cost = nx.dijkstra_path_length(synthetic, source, target, weight="weight")

        assert cost == pytest.approx(expected_cost, abs=TOLERANCE), (
            f"cost mismatch for {source}->{target}"
        )
        assert path[0] == source and path[-1] == target
        assert nx.dijkstra_path(synthetic, source, target, weight="weight") is not None

        # The path we return must itself cost what we claim.
        recomputed = sum(
            synthetic[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
        )
        assert recomputed == pytest.approx(cost, abs=TOLERANCE)


def test_dijkstra_matches_networkx_on_multigraph(
    synthetic_multigraph: nx.MultiDiGraph,
) -> None:
    """Parallel edges must be collapsed to their cheapest member."""
    pairs = reachable_pairs(synthetic_multigraph, count=25, seed=2)
    for source, target in pairs:
        _, cost = dijkstra(synthetic_multigraph, source, target)
        expected = nx.dijkstra_path_length(
            synthetic_multigraph, source, target, weight="weight"
        )
        assert cost == pytest.approx(expected, abs=TOLERANCE)


def test_dijkstra_matches_networkx_across_seeds() -> None:
    for seed in (3, 11, 2024):
        graph = build_synthetic_graph(n_nodes=40, edge_prob=0.15, seed=seed)
        for source, target in reachable_pairs(graph, count=8, seed=seed):
            _, cost = dijkstra(graph, source, target)
            expected = nx.dijkstra_path_length(graph, source, target, weight="weight")
            assert cost == pytest.approx(expected, abs=TOLERANCE)


def test_dijkstra_path_length_helper(synthetic: nx.DiGraph) -> None:
    source, target = reachable_pairs(synthetic, count=1, seed=5)[0]
    assert dijkstra_path_length(synthetic, source, target) == pytest.approx(
        dijkstra(synthetic, source, target)[1], abs=TOLERANCE
    )


def test_dijkstra_source_equals_target(synthetic: nx.DiGraph) -> None:
    node = next(iter(synthetic.nodes))
    assert dijkstra(synthetic, node, node) == ([node], 0.0)


def test_dijkstra_raises_when_unreachable() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_node("island")
    with pytest.raises(nx.NetworkXNoPath):
        dijkstra(graph, "a", "island")


def test_dijkstra_raises_on_missing_nodes(synthetic: nx.DiGraph) -> None:
    node = next(iter(synthetic.nodes))
    with pytest.raises(nx.NodeNotFound):
        dijkstra(synthetic, "nope", node)
    with pytest.raises(nx.NodeNotFound):
        dijkstra(synthetic, node, "nope")


def test_dijkstra_tolerates_zero_weight_edges() -> None:
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("a", "b", 0.0), ("b", "c", 2.0)])
    assert dijkstra(graph, "a", "c") == (["a", "b", "c"], 2.0)


# --------------------------------------------------------------------------- #
# A* vs networkx
# --------------------------------------------------------------------------- #
def test_astar_matches_networkx_with_heuristic(synthetic: nx.DiGraph) -> None:
    heuristic = make_euclidean_heuristic(synthetic, pos_attr="pos")
    pairs = reachable_pairs(synthetic, count=25, seed=6)

    for source, target in pairs:
        path, cost = astar(synthetic, source, target, heuristic=heuristic)
        expected = nx.astar_path_length(
            synthetic, source, target, heuristic=heuristic, weight="weight"
        )
        assert cost == pytest.approx(expected, abs=TOLERANCE), (
            f"A* cost mismatch for {source}->{target}"
        )
        assert path[0] == source and path[-1] == target


def test_astar_with_admissible_heuristic_equals_dijkstra(
    synthetic: nx.DiGraph,
) -> None:
    """The whole point of an admissible heuristic: same optimum, less search."""
    heuristic = make_euclidean_heuristic(synthetic, pos_attr="pos")
    for source, target in reachable_pairs(synthetic, count=15, seed=8):
        _, astar_cost = astar(synthetic, source, target, heuristic=heuristic)
        _, dijkstra_cost = dijkstra(synthetic, source, target)
        assert astar_cost == pytest.approx(dijkstra_cost, abs=TOLERANCE)


def test_astar_zero_heuristic_equals_dijkstra(synthetic: nx.DiGraph) -> None:
    for source, target in reachable_pairs(synthetic, count=10, seed=9):
        _, astar_cost = astar(
            synthetic, source, target, heuristic=zero_heuristic
        )
        _, dijkstra_cost = dijkstra(synthetic, source, target)
        assert astar_cost == pytest.approx(dijkstra_cost, abs=TOLERANCE)


def test_astar_defaults_to_dijkstra_behaviour(synthetic: nx.DiGraph) -> None:
    source, target = reachable_pairs(synthetic, count=1, seed=10)[0]
    assert astar_path_length(synthetic, source, target) == pytest.approx(
        dijkstra_path_length(synthetic, source, target), abs=TOLERANCE
    )


def test_astar_matches_networkx_on_multigraph(
    synthetic_multigraph: nx.MultiDiGraph,
) -> None:
    heuristic = make_euclidean_heuristic(synthetic_multigraph, pos_attr="pos")
    for source, target in reachable_pairs(synthetic_multigraph, count=15, seed=12):
        _, cost = astar(synthetic_multigraph, source, target, heuristic=heuristic)
        expected = nx.astar_path_length(
            synthetic_multigraph, source, target, heuristic=heuristic, weight="weight"
        )
        assert cost == pytest.approx(expected, abs=TOLERANCE)


def test_astar_raises_when_unreachable() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_node("island")
    with pytest.raises(nx.NetworkXNoPath):
        astar(graph, "a", "island")
    with pytest.raises(nx.NodeNotFound):
        astar(graph, "a", "nope")


def test_astar_source_equals_target(synthetic: nx.DiGraph) -> None:
    node = next(iter(synthetic.nodes))
    assert astar(synthetic, node, node) == ([node], 0.0)


def test_haversine_is_metric_and_sane() -> None:
    # Connaught Place -> India Gate is roughly 2.6 km in a straight line.
    distance = haversine_m(28.6315, 77.2167, 28.6129, 77.2295)
    assert 1_500 < distance < 3_500
    assert haversine_m(28.6, 77.2, 28.6, 77.2) == pytest.approx(0.0, abs=1e-6)
    # Symmetry.
    assert haversine_m(28.6, 77.2, 28.7, 77.3) == pytest.approx(
        haversine_m(28.7, 77.3, 28.6, 77.2), abs=1e-9
    )


def test_euclidean_heuristic_requires_positions() -> None:
    graph = nx.DiGraph()
    graph.add_edge(1, 2, weight=1.0)
    with pytest.raises(ValueError):
        make_euclidean_heuristic(graph, pos_attr="pos")


# --------------------------------------------------------------------------- #
# All-pairs
# --------------------------------------------------------------------------- #
def test_all_pairs_matches_individual_runs(synthetic: nx.DiGraph) -> None:
    nodes = list(synthetic.nodes)[:12]
    table = dijkstra_all_pairs(synthetic, nodes)

    assert len(table) == len(nodes) ** 2  # every ordered pair, self-pairs included
    for node in nodes:
        assert table[(node, node)] == ([node], 0.0)

    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            path, cost = table[(source, target)]
            if math.isinf(cost):
                assert not nx.has_path(synthetic, source, target)
                continue
            expected = nx.dijkstra_path_length(
                synthetic, source, target, weight="weight"
            )
            assert cost == pytest.approx(expected, abs=TOLERANCE)
            assert path[0] == source and path[-1] == target


def test_all_pairs_reports_unreachable_as_infinite() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_node("island")

    table = dijkstra_all_pairs(graph, ["a", "b", "island"])
    assert table[("a", "b")][1] == pytest.approx(1.0)
    assert table[("a", "island")] == ([], math.inf)
    assert table[("island", "a")] == ([], math.inf)


def test_all_pairs_handles_duplicate_and_single_nodes(synthetic: nx.DiGraph) -> None:
    node = next(iter(synthetic.nodes))
    assert dijkstra_all_pairs(synthetic, [node]) == {(node, node): ([node], 0.0)}

    # Duplicates must not produce duplicate keys or redundant search runs.
    table = dijkstra_all_pairs(synthetic, [node, node])
    assert table == {(node, node): ([node], 0.0)}


def test_all_pairs_raises_on_unknown_node(synthetic: nx.DiGraph) -> None:
    with pytest.raises(nx.NodeNotFound):
        dijkstra_all_pairs(synthetic, ["ghost"])


def test_all_pairs_scales_to_a_realistic_depot_problem(
    synthetic: nx.DiGraph,
) -> None:
    """Typical VRP shape: one depot plus a handful of stops."""
    nodes = list(synthetic.nodes)[:20]
    table = dijkstra_all_pairs(synthetic, nodes)
    finite = [cost for _, cost in table.values() if not math.isinf(cost)]
    assert finite, "expected at least some reachable pairs"
    assert all(cost >= 0.0 for cost in finite)


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def test_largest_strongly_connected_subgraph() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])  # a 3-cycle
    graph.add_edge(4, 5)  # a dangling pair that cannot reach anything else

    component = largest_strongly_connected_subgraph(graph)
    assert set(component.nodes) == {1, 2, 3}
    assert nx.is_strongly_connected(component)


# --------------------------------------------------------------------------- #
# Real Delhi graph — opt-in, needs a cached graph (or network on first fetch)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_routing_on_real_delhi_graph() -> None:
    if not os.environ.get("SMART_GATI_RUN_SLOW"):
        pytest.skip("set SMART_GATI_RUN_SLOW=1 to run tests against the real Delhi graph")
    if not is_delhi_graph_cached():
        pytest.skip(
            "no cached Delhi graph; run load_delhi_graph() once to populate "
            "backend/data/cache/"
        )

    graph = load_delhi_graph()
    assert graph.number_of_nodes() > 0

    # Every edge must carry a usable travel-time cost.
    for _, _, data in graph.edges(data=True):
        assert data["weight"] > 0.0
        assert data["weight"] == pytest.approx(data["travel_time"])

    component = largest_strongly_connected_subgraph(graph)
    nodes = list(component.nodes)
    assert len(nodes) >= 2

    # A short sample keeps this quick; correctness on real data is the point,
    # not coverage.
    source, target = nodes[0], nodes[-1]
    heuristic = make_haversine_heuristic(graph)

    path, cost = dijkstra(graph, source, target)
    expected = nx.dijkstra_path_length(graph, source, target, weight="weight")
    assert cost == pytest.approx(expected, abs=TOLERANCE)
    assert path[0] == source and path[-1] == target

    astar_path, astar_cost = astar(graph, source, target, heuristic=heuristic)
    assert astar_cost == pytest.approx(cost, abs=TOLERANCE)
    assert astar_path[0] == source and astar_path[-1] == target

    # Empirically confirm the heuristic never overestimates, which is the
    # property that makes the A* result above optimal rather than merely fast.
    for probe in nodes[:10]:
        straight_line_estimate = heuristic(probe, target)
        true_remaining = dijkstra_path_length(graph, probe, target)
        assert straight_line_estimate <= true_remaining + TOLERANCE, (
            f"heuristic overestimates {probe}->{target}: "
            f"{straight_line_estimate} > {true_remaining}"
        )

    # Coordinates must be real lat/lon so the haversine heuristic is meaningful.
    lat, lon = graph.nodes[source]["y"], graph.nodes[source]["x"]
    assert 28.0 < lat < 29.0 and 76.5 < lon < 78.0
