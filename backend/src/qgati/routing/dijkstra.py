"""Dijkstra's shortest-path algorithm, implemented from scratch.

This is a deliberate re-implementation rather than a call into
``networkx.dijkstra_path`` — the project's validation story is that our routing
layer is checked against networkx's, so the two must be independent code.

Costs must be non-negative (travel times always are). Parallel edges are
collapsed to their cheapest one, matching networkx's semantics for multigraphs.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Callable, Hashable, Iterable

import networkx as nx

__all__ = ["dijkstra", "dijkstra_all_pairs", "dijkstra_path_length", "weight_function"]

Node = Hashable
WeightFn = Callable[[Node, Node, dict], float]


def weight_function(graph: nx.Graph, weight: str | WeightFn = "weight") -> WeightFn:
    """Build ``f(u, v, data) -> float``, mirroring networkx's own resolution.

    For multigraphs ``data`` maps edge keys to attribute dicts, so the returned
    callable takes the minimum across parallel edges — the cheapest way to get
    from ``u`` to ``v`` is the only one routing should ever consider.
    """
    if callable(weight):
        return weight

    if graph.is_multigraph():

        def from_multigraph(u: Node, v: Node, data: dict) -> float:
            return min(attributes.get(weight, 1) for attributes in data.values())

        return from_multigraph

    def from_graph(u: Node, v: Node, data: dict) -> float:
        return data.get(weight, 1)

    return from_graph


def _successors(
    graph: nx.Graph, weight_fn: WeightFn, u: Node
) -> Iterable[tuple[Node, float]]:
    """Yield ``(neighbour, edge cost)`` for every successor of ``u``.

    ``data`` is whatever the graph type stores there: an attribute dict for a
    DiGraph, or a ``{key: attrs}`` mapping for a MultiDiGraph. Collapsing
    parallel edges is :func:`weight_function`'s job, so both cases take the same
    path here — the same split networkx uses.
    """
    for v, data in graph.adj[u].items():
        yield v, weight_fn(u, v, data)


def _reconstruct_path(
    predecessors: dict[Node, Node | None], source: Node, target: Node
) -> list[Node]:
    """Walk predecessor pointers back from ``target`` to ``source``."""
    path = [target]
    while path[-1] != source:
        parent = predecessors[path[-1]]
        if parent is None:  # pragma: no cover - defensive; pred chain is total
            return []
        path.append(parent)
    path.reverse()
    return path


def _dijkstra_core(
    graph: nx.Graph,
    source: Node,
    weight_fn: WeightFn,
    targets: set[Node] | None = None,
) -> tuple[dict[Node, float], dict[Node, Node | None]]:
    """Single-source Dijkstra, optionally stopping early.

    With ``targets`` given, the search halts as soon as every target has been
    finalised instead of exploring the whole component. That is what makes
    :func:`dijkstra_all_pairs` cheap: one run per *source* covering all its
    targets, rather than one run per ``(source, target)`` pair.

    Returns ``(distances, predecessors)``.
    """
    if source not in graph:
        raise nx.NodeNotFound(f"Source {source!r} is not in G")
    if targets is not None:
        unknown = set(targets) - set(graph)
        if unknown:
            raise nx.NodeNotFound(f"Targets {sorted(map(repr, unknown))} are not in G")

    distances: dict[Node, float] = {source: 0.0}
    predecessors: dict[Node, Node | None] = {source: None}
    settled: set[Node] = set()

    # The counter breaks ties between equal distances. Without it, heapq would
    # fall through to comparing the node objects, which need not be orderable.
    counter = itertools.count()
    queue: list[tuple[float, int, Node]] = [(0.0, next(counter), source)]
    remaining = set(targets) if targets is not None else None

    while queue:
        distance, _, u = heapq.heappop(queue)
        if u in settled:
            continue  # stale heap entry superseded by a shorter one
        settled.add(u)

        if remaining is not None:
            remaining.discard(u)
            if not remaining:
                break  # every target is final; anything left cannot be a target

        for v, cost in _successors(graph, weight_fn, u):
            if v in settled:
                continue
            candidate = distance + cost
            if candidate < distances.get(v, math.inf):
                distances[v] = candidate
                predecessors[v] = u
                heapq.heappush(queue, (candidate, next(counter), v))

    return distances, predecessors


def dijkstra(
    graph: nx.Graph, source: Node, target: Node, weight: str | WeightFn = "weight"
) -> tuple[list[Node], float]:
    """Shortest path from ``source`` to ``target``.

    Returns ``(path, cost)`` where ``path`` includes both endpoints. Raises
    :class:`networkx.NetworkXNoPath` when no route exists and
    :class:`networkx.NodeNotFound` when either endpoint is absent — the same
    contract networkx uses, so the two are drop-in comparable.
    """
    if source == target:
        if source not in graph:
            raise nx.NodeNotFound(f"Source {source!r} is not in G")
        return [source], 0.0

    distances, predecessors = _dijkstra_core(
        graph, source, weight_function(graph, weight), targets={target}
    )
    if target not in distances:
        raise nx.NetworkXNoPath(f"No path between {source!r} and {target!r}.")

    return _reconstruct_path(predecessors, source, target), distances[target]


def dijkstra_path_length(
    graph: nx.Graph, source: Node, target: Node, weight: str | WeightFn = "weight"
) -> float:
    """Cost of the shortest path, without materialising the path itself."""
    return dijkstra(graph, source, target, weight=weight)[1]


def dijkstra_all_pairs(
    graph: nx.Graph, nodes: Iterable[Node], weight: str | WeightFn = "weight"
) -> dict[tuple[Node, Node], tuple[list[Node], float]]:
    """Shortest paths for every ordered pair drawn from ``nodes``.

    Intended for cost-matrix construction, where the node set is a handful of
    stops rather than the whole graph. It runs **one** Dijkstra per distinct
    source — with early termination once all of that source's targets are
    settled — instead of one per pair, and short-circuits runs whose target set
    is empty.

    Unreachable pairs are reported as ``([], inf)`` rather than omitted, so a
    consumer can index the result without a membership check. Self-pairs are
    ``([node], 0.0)``.
    """
    unique_nodes = list(dict.fromkeys(nodes))
    unknown = set(unique_nodes) - set(graph)
    if unknown:
        raise nx.NodeNotFound(f"Nodes {sorted(map(repr, unknown))} are not in G")

    weight_fn = weight_function(graph, weight)
    results: dict[tuple[Node, Node], tuple[list[Node], float]] = {}

    for source in unique_nodes:
        results[(source, source)] = ([source], 0.0)

        targets = {node for node in unique_nodes if node != source}
        if not targets:
            continue

        distances, predecessors = _dijkstra_core(graph, source, weight_fn, targets)
        for target in targets:
            if target in distances:
                path = _reconstruct_path(predecessors, source, target)
                results[(source, target)] = (path, distances[target])
            else:
                results[(source, target)] = ([], math.inf)

    return results
