"""Collapse the road graph into a dense cost matrix over a scenario's nodes.

This is the boundary between the routing world and the optimization world. After
:func:`build_cost_matrix` runs, no VRP solver should ever touch the road graph
again — they see an ``(n, n)`` array of costs and nothing else. That separation
is what makes a solver runnable on real Delhi travel times and on hand-written
toy costs without modification.

Three quantities, one objective
-------------------------------
The problem statement asks for time, distance and fuel to be minimised together.
Only one of those is what routing minimises — the shortest path is the fastest
one — so the other two are *read off* the route that search returns: a leg's
distance is the length of the roads its fastest path runs along, and its fuel
follows from that distance and the leg's implied average speed. That ordering is
deliberate. A route driven is the fastest route; pricing it by the
shortest-by-distance path instead would score a journey nobody takes.

The three are then converted into a single rupee objective by
:func:`~qgati.optimizer.objective.price_legs`, and it is that objective — not
travel time — that :attr:`CostMatrix.objective_matrix` exposes and every solver
minimises. The raw components stay on the matrix, so a report can show where the
money went without re-deriving anything.

Doing the conversion here, once, at the boundary is what keeps the benchmark
honest: every solver is handed the same already-priced array and none of them
gets an opportunity to weight the three goals differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Hashable

import networkx as nx
import numpy as np

from qgati.optimizer.objective import DEFAULT_WEIGHTS, CostWeights, price_legs
from qgati.routing.dijkstra import WeightFn, dijkstra_all_pairs

if TYPE_CHECKING:  # avoids an optimizer <-> graph import cycle at runtime
    from qgati.optimizer.models import Scenario

__all__ = ["CostMatrix", "CostMatrixBuild", "build_cost_matrix",
           "build_cost_matrix_detailed", "changed_entries"]


@dataclass(frozen=True)
class CostMatrix:
    """What a scenario's road network costs, in time, distance, fuel and rupees.

    ``matrix[i, j]`` is the travel time in seconds from ``nodes[i]`` to
    ``nodes[j]``, and ``distance_matrix[i, j]`` is the length in metres of the
    roads that fastest path runs along. Index ``0`` is always the depot, so a
    route's *return* leg is always to row/column zero.

    A route's *departure* is row/column zero too, unless the scenario names a
    different start for some vehicle — see :attr:`vehicle_start_index`. That is
    the one asymmetry: a re-optimized vehicle begins wherever it already is and
    still ends at the depot, which is a different node per vehicle and so cannot
    be folded into a single index the way the depot can.

    ``delivery_node_index[k]`` maps delivery ``k`` — a position in
    ``scenario.deliveries`` — to its row/column here. Two deliveries sharing a
    road node therefore map to the same index, which is correct: they are the
    same physical stop, counted separately for demand.

    :attr:`objective_matrix` is the one number the optimizers minimise:
    ``weights`` applied to time, distance and the fuel derived from them. It is
    computed once here rather than per solver, so a benchmark compares search
    algorithms and not accounting policies. The three raw components are kept
    alongside it because a cost nobody can decompose is a cost nobody can check.

    An unreachable pair is ``inf`` in every array, so it can never be mistaken
    for a cheap leg.
    """

    matrix: np.ndarray
    nodes: tuple[Hashable, ...]
    delivery_node_index: tuple[int, ...]
    distance_matrix: np.ndarray
    weights: CostWeights = DEFAULT_WEIGHTS
    #: The row/column each vehicle departs from, in scenario vehicle order.
    #:
    #: Empty — the overwhelmingly common case — means every vehicle leaves the
    #: depot, and :meth:`start_index` answers :data:`DEPOT_INDEX` for all of
    #: them. It is empty rather than a tuple of zeros so that a matrix built the
    #: ordinary way is *identical* to one built before this field existed, which
    #: is what keeps the solver benchmark reproducible.
    vehicle_start_index: tuple[int, ...] = ()

    #: Derived from the three fields above; not part of the constructor because
    #: letting a caller set them would let the objective disagree with the time
    #: and distance it claims to be priced from.
    fuel_matrix: np.ndarray = field(init=False, repr=False, compare=False)
    objective_matrix: np.ndarray = field(init=False, repr=False, compare=False)

    #: The depot always sits at index 0.
    DEPOT_INDEX = 0

    def __post_init__(self) -> None:
        size = len(self.nodes)
        if self.matrix.shape != (size, size):
            raise ValueError(
                f"matrix is {self.matrix.shape} but there are {size} nodes"
            )
        if self.distance_matrix.shape != (size, size):
            raise ValueError(
                f"distance matrix is {self.distance_matrix.shape} but there are "
                f"{size} nodes"
            )
        out_of_range = [
            index for index in self.vehicle_start_index if not 0 <= index < size
        ]
        if out_of_range:
            raise ValueError(
                f"vehicle start index {out_of_range[0]} is outside a matrix of "
                f"{size} node(s)"
            )

        fuel, objective = price_legs(self.matrix, self.distance_matrix, self.weights)
        object.__setattr__(self, "fuel_matrix", fuel)
        object.__setattr__(self, "objective_matrix", objective)

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def depot_index(self) -> int:
        return self.DEPOT_INDEX

    def start_index(self, vehicle: int) -> int:
        """The row/column a route for ``vehicle`` departs from.

        A one-line method rather than a field read at each call site, because
        the empty case is the one that must never be got wrong: a caller that
        forgot to handle it would index an empty tuple on every ordinary
        scenario in the project.
        """
        if not self.vehicle_start_index:
            return self.DEPOT_INDEX
        return self.vehicle_start_index[vehicle]

    def node_index(self, node: Hashable) -> int:
        """Row/column of a road-graph node, or raise ``KeyError``."""
        try:
            return self.nodes.index(node)
        except ValueError:
            raise KeyError(f"node {node!r} is not in this cost matrix") from None

    def cost(self, i: int, j: int) -> float:
        """Travel time between two matrix indices, in seconds."""
        return float(self.matrix[i, j])

    def objective(self, i: int, j: int) -> float:
        """Weighted cost of the same leg, in rupees."""
        return float(self.objective_matrix[i, j])

    def is_symmetric(self, tolerance: float = 1e-6) -> bool:
        """True when every leg costs the same in both directions.

        Worth checking on real data: one-way streets make Delhi's network
        asymmetric, so a solver assuming symmetry would be quietly wrong.
        """
        return bool(
            np.allclose(self.matrix, self.matrix.T, atol=tolerance)
        )


@dataclass(frozen=True)
class CostMatrixBuild:
    """A cost matrix together with the road edges its cheapest paths traverse.

    ``used_edges`` is the set of ``(u, v)`` segments any leg's shortest path
    runs along — the road network this instance actually places load on. The
    traffic layer logs exactly these, so that every row it writes describes a
    road some route genuinely weighed, rather than an arbitrary slice of the
    graph.
    """

    matrix: CostMatrix
    used_edges: frozenset[tuple[Hashable, Hashable]]


def build_cost_matrix(
    graph: nx.Graph,
    scenario: Scenario,
    weight: str | WeightFn = "weight",
    allow_unreachable: bool = False,
    length_attr: str = "length",
    weights: CostWeights = DEFAULT_WEIGHTS,
) -> CostMatrix:
    """Compute the cost matrix between the depot and every delivery node.

    Uses all-pairs Dijkstra over just this small node subset — one search per
    distinct source, with early termination — rather than anything graph-wide.

    Parameters
    ----------
    graph
        Road graph from Phase 1, with travel-time ``weight``.
    scenario
        The instance to price. Depot and delivery nodes must exist in ``graph``.
    weight
        Edge attribute to minimise, or a callable ``(u, v, data) -> float``.
        The callable form is how the traffic layer prices a network under
        simulated conditions: see
        :func:`~qgati.traffic.simulator.traffic_weight_function`.
    allow_unreachable
        When false (the default), an unreachable pair raises. That is the
        intended behaviour: an unreachable pair means no feasible tour exists,
        and silently substituting ``inf`` would let an optimizer return a
        confidently wrong answer.
    length_attr
        Edge attribute holding edge length in metres, used for the distance and
        fuel terms. Read from the *chosen* path, so it is the length of the route
        actually driven. An edge lacking it raises rather than contributing a
        silent zero.
    weights
        Prices converting time, distance and fuel into the objective the solvers
        minimise. Defaults to the documented policy in
        :mod:`qgati.optimizer.objective`.

    Raises
    ------
    ValueError
        If any node is missing from the graph, if a pair is unreachable and
        ``allow_unreachable`` is false, or if a traversed edge carries no length.
    """
    return build_cost_matrix_detailed(
        graph,
        scenario,
        weight=weight,
        allow_unreachable=allow_unreachable,
        length_attr=length_attr,
        weights=weights,
    ).matrix


def build_cost_matrix_detailed(
    graph: nx.Graph,
    scenario: Scenario,
    weight: str | WeightFn = "weight",
    allow_unreachable: bool = False,
    length_attr: str = "length",
    weights: CostWeights = DEFAULT_WEIGHTS,
) -> CostMatrixBuild:
    """:func:`build_cost_matrix`, plus the road edges it routed along.

    The extra product is free: the all-pairs search returns the path for every
    leg, and the matrix builder has always discarded them. Collecting them costs
    one pass over the paths already in hand, which is why this returns the edges
    rather than leaving a second, duplicate search to whoever needs them. The
    same pass accumulates each path's length, which is where the distance and
    fuel terms come from.

    Same arguments and the same ``ValueError`` conditions as
    :func:`build_cost_matrix`.
    """
    nodes = _scenario_nodes(graph, scenario)
    table = dijkstra_all_pairs(graph, nodes, weight=weight)

    matrix, distances, unreachable = _matrix_from_paths(
        graph, nodes, table, length_attr
    )
    if unreachable and not allow_unreachable:
        raise ValueError(
            f"{len(unreachable)} node pair(s) are unreachable in this graph, so no "
            f"feasible tour exists; first pair: {unreachable[0]}. Build the scenario "
            "from largest_strongly_connected_subgraph(graph) so every stop is "
            "reachable from every other."
        )

    position = {node: index for index, node in enumerate(nodes)}
    return CostMatrixBuild(
        matrix=CostMatrix(
            matrix=matrix,
            nodes=tuple(nodes),
            delivery_node_index=tuple(
                position[delivery.node] for delivery in scenario.deliveries
            ),
            distance_matrix=distances,
            weights=weights,
            # Left empty for a scenario that starts every vehicle at the depot,
            # so the matrix is the one this function has always returned. Filling
            # it with zeros would behave identically and be a different object,
            # and "benignly different" is not a thing a cached matrix should be.
            vehicle_start_index=(
                tuple(
                    position[scenario.start_node(vehicle)]
                    for vehicle in range(scenario.n_vehicles)
                )
                if scenario.has_custom_starts
                else ()
            ),
        ),
        used_edges=_edges_along(table),
    )


def changed_entries(before: CostMatrix, after: CostMatrix) -> int:
    """How many entries of the objective matrix moved between two pricings.

    Measured on the objective rather than on travel time because the objective
    is what the solvers minimise, so it is the number a client can watch change
    in an optimize result.

    ``isclose`` rather than ``==`` for two reasons: the two matrices come from
    separate Dijkstra runs and can differ in the last bit without meaning
    anything, and it is the comparison that treats two infinities as equal while
    still calling an infinity and a finite cost different — which is exactly the
    question a closure asks.

    Lives here rather than beside any one caller because there are now two — an
    incident re-pricing a scenario, and a fleet observation doing the same — and
    they must report the same number for the same change. Both matrices must be
    over the same node list; the objective is compared element-wise, so two
    differently-shaped matrices would raise rather than answer.
    """
    moved = ~np.isclose(before.objective_matrix, after.objective_matrix)
    return int(np.count_nonzero(moved))


def _scenario_nodes(graph: nx.Graph, scenario: Scenario) -> list[Hashable]:
    """The depot, any custom start nodes, and every distinct delivery node.

    Depot first, so :data:`CostMatrix.DEPOT_INDEX` stays zero however many
    starts a scenario has. The order past that is fixed by the scenario rather
    than sorted, so two builds of the same scenario agree on every index —
    which is what lets :func:`changed_entries` compare two of them element-wise.

    Raises
    ------
    ValueError
        If any of them is absent from the graph.
    """
    nodes: list[Hashable] = [scenario.depot.node]
    for node in scenario.starts:
        if node not in nodes:
            nodes.append(node)
    for delivery in scenario.deliveries:
        if delivery.node not in nodes:
            nodes.append(delivery.node)

    missing = [node for node in nodes if node not in graph]
    if missing:
        raise ValueError(
            f"{len(missing)} scenario node(s) are not in the graph: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    return nodes


def _matrix_from_paths(
    graph: nx.Graph, nodes: list[Hashable], table: dict, length_attr: str
) -> tuple[np.ndarray, np.ndarray, list[tuple[Hashable, Hashable]]]:
    """Dense the all-pairs result into time and distance arrays.

    Distance is measured *along the fastest path*, not by a second search
    minimising length. Those are different routes in general, and the one a
    vehicle actually drives is the fastest — so that is the one whose metres it
    covers and whose fuel it burns.
    """
    size = len(nodes)
    matrix = np.full((size, size), np.inf, dtype=float)
    distances = np.full((size, size), np.inf, dtype=float)
    unreachable: list[tuple[Hashable, Hashable]] = []

    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            path, cost = table[(source, target)]
            matrix[i, j] = cost
            if np.isinf(cost):
                if i != j:
                    unreachable.append((source, target))
                continue
            distances[i, j] = _path_length(graph, path, length_attr)

    return matrix, distances, unreachable


def _path_length(graph: nx.Graph, path: list[Hashable], length_attr: str) -> float:
    """Total length in metres of the roads a path runs along.

    A self-pair has a one-node path and therefore no edges, so it is zero.
    """
    total = 0.0
    for u, v in zip(path, path[1:]):
        data = graph.adj[u][v]
        if graph.is_multigraph():
            total += min(_edge_length(attrs, length_attr) for attrs in data.values())
        else:
            total += _edge_length(data, length_attr)
    return total


def _edge_length(attributes: dict, length_attr: str) -> float:
    """One edge's length, or raise naming the missing attribute.

    Deliberately not defaulted to zero. A missing length would silently price a
    road at no distance and no fuel, which is a wrong answer that looks like a
    cheap route — the same reasoning that makes an unreachable pair raise above
    rather than become an ``inf`` nobody notices.
    """
    try:
        return float(attributes[length_attr])
    except KeyError:
        raise ValueError(
            f"edge is missing the {length_attr!r} attribute, so its distance and "
            f"fuel cannot be priced; available attributes: "
            f"{sorted(attributes)}"
        ) from None


def _edges_along(table: dict) -> frozenset[tuple[Hashable, Hashable]]:
    """Every road edge traversed by any of the table's paths.

    An unreachable pair contributes nothing, since its path is empty.
    """
    edges: set[tuple[Hashable, Hashable]] = set()
    for path, _cost in table.values():
        for u, v in zip(path, path[1:]):
            edges.add((u, v))
    return frozenset(edges)
