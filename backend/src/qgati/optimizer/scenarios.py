"""Scenario generation for VRP instances.

The one rule that matters here: **every node a scenario uses comes from the
largest strongly-connected subgraph of the road graph, never the raw node list.**

A real OSM extract contains one-way stubs that can be entered but not left — in
the current Delhi extract, 37 of 2033 nodes. A scenario that placed a delivery
on one of those would have no feasible tour, which is a *modelling* error that
would surface as a mysterious optimizer failure. Rather than document that as a
caveat, :func:`build_random_scenario` restricts the pool itself, so a caller
cannot pass the full node list and get an infeasible instance.
"""

from __future__ import annotations

import math
import random
from typing import Hashable

import networkx as nx

from qgati.graph.graph_builder import largest_strongly_connected_subgraph
from qgati.optimizer.models import Delivery, Depot, Scenario, Vehicle

__all__ = ["build_random_scenario", "servable_nodes"]


def servable_nodes(graph: nx.Graph) -> list[Hashable]:
    """Nodes a VRP scenario may legally use: the largest mutually-reachable set.

    Every node returned can reach every other, so any permutation of them forms
    a feasible tour. Sorted for reproducibility across runs.
    """
    component = largest_strongly_connected_subgraph(graph)
    nodes = list(component.nodes)
    try:
        return sorted(nodes)
    except TypeError:  # heterogeneous node id types
        return sorted(nodes, key=repr)


def _coordinates(graph: nx.Graph, node: Hashable) -> tuple[float, float]:
    """Lat/lon for a node, tolerating both OSMnx and synthetic graph layouts."""
    data = graph.nodes[node]
    if "y" in data and "x" in data:  # OSMnx
        return float(data["y"]), float(data["x"])
    if "pos" in data:  # synthetic test graph stores (x, y)
        x, y = data["pos"]
        return float(y), float(x)
    return 0.0, 0.0


def build_random_scenario(
    graph: nx.Graph,
    n_deliveries: int,
    n_vehicles: int,
    seed: int = 0,
    demand_range: tuple[int, int] = (1, 9),
    capacity_slack: float = 1.6,
) -> Scenario:
    """Generate a random, guaranteed-feasible VRP instance over ``graph``.

    Parameters
    ----------
    graph
        Road graph. Nodes are drawn from its largest strongly-connected
        subgraph, so the result is always servable.
    n_deliveries, n_vehicles
        Instance size.
    seed
        Fixes the instance; the same seed always yields the same scenario.
    demand_range
        Inclusive integer range for each delivery's demand.
    capacity_slack
        Fleet capacity is ``ceil(total_demand * slack / n_vehicles)`` per
        vehicle. Values above 1.0 leave room to consolidate, which is what makes
        the instance interesting rather than trivially one-route-per-stop.

    Raises
    ------
    ValueError
        If the graph has too few mutually-reachable nodes, or the arguments are
        nonsensical.
    """
    if n_deliveries < 1:
        raise ValueError("n_deliveries must be at least 1")
    if n_vehicles < 1:
        raise ValueError("n_vehicles must be at least 1")
    if capacity_slack < 1.0:
        raise ValueError(
            "capacity_slack must be >= 1.0, or the fleet cannot carry the demand"
        )
    low, high = demand_range
    if low < 0 or high < low:
        raise ValueError(f"invalid demand_range {demand_range}")

    pool = servable_nodes(graph)
    needed = n_deliveries + 1  # deliveries plus the depot
    if len(pool) < needed:
        raise ValueError(
            f"graph offers only {len(pool)} mutually-reachable nodes but the "
            f"scenario needs {needed}"
        )

    rng = random.Random(seed)
    chosen = rng.sample(pool, needed)
    depot_node, delivery_nodes = chosen[0], chosen[1:]

    demands = [rng.randint(low, high) for _ in range(n_deliveries)]
    total_demand = sum(demands)

    # Per-vehicle capacity: the fleet-wide fair share, inflated by the slack, but
    # never below the largest single demand (or one delivery could not be served
    # by any vehicle, making the instance infeasible regardless of slack).
    per_vehicle = math.ceil(total_demand * capacity_slack / n_vehicles)
    capacity = max(per_vehicle, max(demands))

    lat, lon = _coordinates(graph, depot_node)
    return Scenario(
        depot=Depot(node=depot_node, lat=lat, lon=lon),
        deliveries=tuple(
            Delivery(id=f"D{index}", node=node, demand=demand)
            for index, (node, demand) in enumerate(
                zip(delivery_nodes, demands, strict=True)
            )
        ),
        vehicles=tuple(
            Vehicle(id=f"V{index}", capacity=capacity)
            for index in range(n_vehicles)
        ),
    )
