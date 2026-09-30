"""The weighted objective: what a second, a metre and a litre are worth.

``DESIGN_DECISIONS.md`` commits to "one rupee-equivalent cost composed of time
cost, distance cost, fuel cost, penalties". These tests pin that the weights act
as *prices* — the property that makes them arguable — and that the fuel model
behaves the way the design claims, in particular that it responds to congestion.
That last one is the whole reason fuel is a separate term rather than a constant
multiplied into distance, so it is tested directly rather than inferred from a
scenario result.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from qgati.graph import build_cost_matrix
from qgati.optimizer import (
    DEFAULT_WEIGHTS,
    CostWeights,
    Delivery,
    Depot,
    FuelModel,
    Scenario,
    Solution,
    Vehicle,
    evaluate,
)
from qgati.optimizer.objective import price_legs

# --------------------------------------------------------------------------- #
# Weights are prices
# --------------------------------------------------------------------------- #
def test_default_weights_are_prices_in_rupees() -> None:
    """An hour, a kilometre and a litre each cost exactly what they are quoted at."""
    assert DEFAULT_WEIGHTS.time_per_second == pytest.approx(150.0 / 3600.0)
    assert DEFAULT_WEIGHTS.distance_per_metre == pytest.approx(5.0 / 1000.0)
    assert DEFAULT_WEIGHTS.fuel_per_litre == pytest.approx(90.0)

    assert DEFAULT_WEIGHTS.cost(3600.0, 1000.0, 1.0) == pytest.approx(
        150.0 + 5.0 + 90.0
    )


def test_weights_reject_negative_prices() -> None:
    """A negative price would pay a solver to drive further; refuse it outright."""
    with pytest.raises(ValueError, match="time_per_hour"):
        CostWeights(time_per_hour=-1.0)
    with pytest.raises(ValueError, match="fuel_per_litre"):
        CostWeights(fuel_per_litre=-1.0)


def test_zero_prices_isolate_a_single_goal() -> None:
    """Zeroing two prices leaves the third measurable on its own.

    Not a curiosity: it is what lets the objective change be evaluated honestly.
    The time-only objective is reachable from this same code path, so a
    difference between two runs is attributable to the weights and nothing else.
    """
    time = np.array([[0.0, 10.0], [10.0, 0.0]])
    distance = np.array([[0.0, 100.0], [100.0, 0.0]])

    # One rupee per second, so the objective reads directly in seconds.
    time_only = CostWeights(
        time_per_hour=3600.0, distance_per_km=0.0, fuel_per_litre=0.0
    )
    _, objective = price_legs(time, distance, time_only)
    assert objective[0, 1] == pytest.approx(10.0)

    # And one rupee per metre, so it reads in metres.
    distance_only = CostWeights(
        time_per_hour=0.0, distance_per_km=1000.0, fuel_per_litre=0.0
    )
    _, objective = price_legs(time, distance, distance_only)
    assert objective[0, 1] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# The fuel curve
# --------------------------------------------------------------------------- #
def test_fuel_curve_has_a_single_minimum() -> None:
    """The curve must fall then rise; a constant L/km could not price congestion."""
    model = FuelModel()
    assert model.most_efficient_kph == pytest.approx(65.0)
    assert model.litres_per_100km(65.0) == pytest.approx(7.5)

    speeds = np.array([10.0, 20.0, 40.0, 65.0, 80.0, 100.0, 120.0])
    values = model.litres_per_100km(speeds)
    assert int(values.argmin()) == 3  # 65 km/h
    assert values[0] > values[3] > 0.0
    assert values[-1] > values[3]


def test_fuel_model_rejects_a_constant() -> None:
    """Without speed dependence it is a constant L/km wearing a curve's name."""
    with pytest.raises(ValueError, match="constant L/km"):
        FuelModel(idle_coefficient=0.0, speed_coefficient=0.0)


def test_congestion_raises_fuel_per_kilometre() -> None:
    """The property the fuel term exists for.

    Identical road, identical distance, a third of the speed. Distance alone is
    blind to this, and the time term prices the delay in driver-hours but not in
    diesel. This is the case the brief's "function of distance and average
    speed" is there to catch.
    """
    distance = np.array([[0.0, 1000.0], [1000.0, 0.0]])
    fast = np.array([[0.0, 60.0], [60.0, 0.0]])     # 1 km in 60 s  = 60 km/h
    slow = np.array([[0.0, 180.0], [180.0, 0.0]])   # 1 km in 180 s = 20 km/h

    fast_fuel, fast_cost = price_legs(fast, distance)
    slow_fuel, slow_cost = price_legs(slow, distance)

    assert fast_fuel[0, 1] == pytest.approx(0.0752, abs=1e-3)
    assert slow_fuel[0, 1] == pytest.approx(0.1256, abs=1e-3)
    assert slow_fuel[0, 1] > fast_fuel[0, 1]
    assert slow_cost[0, 1] > fast_cost[0, 1]


def test_zero_distance_leg_burns_no_fuel() -> None:
    """Two deliveries at one road node cover no ground, so they cost no diesel."""
    time = np.array([[0.0, 0.0], [0.0, 0.0]])
    distance = np.zeros((2, 2))
    fuel, objective = price_legs(time, distance)
    assert fuel[0, 1] == 0.0
    assert objective[0, 1] == 0.0


def test_unreachable_legs_stay_infinite_even_at_zero_price() -> None:
    """``inf * 0`` is ``nan``, and a nan objective is not a rejected leg.

    An unreachable pair priced by naive arithmetic could turn into a free leg the
    moment a caller zeroes a weight — which is exactly what the time-only
    comparison above does.
    """
    time = np.array([[0.0, np.inf], [1.0, 0.0]])
    distance = np.array([[0.0, np.inf], [100.0, 0.0]])

    fuel, objective = price_legs(time, distance)
    assert np.isinf(fuel[0, 1]) and np.isinf(objective[0, 1])

    free = CostWeights(time_per_hour=0.0, distance_per_km=0.0, fuel_per_litre=0.0)
    _, objective_at_zero = price_legs(time, distance, free)
    assert np.isinf(objective_at_zero[0, 1])


# --------------------------------------------------------------------------- #
# What the matrix builder measures
# --------------------------------------------------------------------------- #
def _scenario(depot_node: object, delivery_node: object) -> Scenario:
    return Scenario(
        depot=Depot(node=depot_node, lat=0.0, lon=0.0),
        deliveries=(Delivery(id="A", node=delivery_node, demand=1.0),),
        vehicles=(Vehicle(id="V0", capacity=10.0),),
    )


def test_distance_is_measured_along_the_fastest_path() -> None:
    """A leg's distance is the route actually driven, not the shortest one.

    Routing minimises time, so the vehicle takes the fast road — and the metres
    it covers are that road's. Pricing the 100 m crawl instead would report a
    distance (and a fuel burn) for a journey nobody makes.
    """
    graph = nx.DiGraph()
    graph.add_edge(0, 1, length=100.0, weight=600.0)   # direct, 100 m, crawling
    graph.add_edge(0, 2, length=200.0, weight=10.0)    # the long way round
    graph.add_edge(2, 1, length=200.0, weight=10.0)
    graph.add_edge(1, 0, length=100.0, weight=600.0)
    graph.add_edge(2, 0, length=200.0, weight=10.0)
    graph.add_edge(1, 2, length=200.0, weight=10.0)

    matrix = build_cost_matrix(graph, _scenario(0, 1))

    assert matrix.cost(0, 1) == pytest.approx(20.0)          # the fast road wins
    assert matrix.distance_matrix[0, 1] == pytest.approx(400.0)  # and is longer


def test_build_cost_matrix_refuses_edges_without_a_length() -> None:
    """A missing length would price a road as free — a wrong answer shaped like
    a cheap route, so it raises instead."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=10.0)
    graph.add_edge(1, 0, weight=10.0)

    with pytest.raises(ValueError, match="length"):
        build_cost_matrix(graph, _scenario(0, 1))


def test_weights_apply_without_touching_the_routing() -> None:
    """Changing the prices moves the objective, never the shortest path."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, length=1000.0, weight=10.0)
    graph.add_edge(1, 0, length=1000.0, weight=10.0)

    default = build_cost_matrix(graph, _scenario(0, 1))
    expensive_fuel = build_cost_matrix(
        graph,
        _scenario(0, 1),
        weights=CostWeights(fuel_per_litre=180.0),
    )

    assert expensive_fuel.matrix[0, 1] == pytest.approx(default.matrix[0, 1])
    assert expensive_fuel.distance_matrix[0, 1] == pytest.approx(
        default.distance_matrix[0, 1]
    )
    assert expensive_fuel.objective_matrix[0, 1] > default.objective_matrix[0, 1]


# --------------------------------------------------------------------------- #
# The breakdown the rest of the system reports
# --------------------------------------------------------------------------- #
def test_evaluation_breakdown_sums_to_the_objective() -> None:
    """``travel_cost`` is the parts, added up — the number a report can audit."""
    graph = nx.DiGraph()
    for u, v in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)):
        graph.add_edge(u, v, length=500.0, weight=60.0)

    scenario = Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node=1, demand=1.0),
            Delivery(id="B", node=2, demand=1.0),
        ),
        vehicles=(Vehicle(id="V0", capacity=10.0),),
    )
    matrix = build_cost_matrix(graph, scenario)
    evaluation = evaluate(Solution(routes=((0, 1),)), scenario, matrix)

    assert evaluation.travel_cost == pytest.approx(
        matrix.weights.cost(
            evaluation.travel_time, evaluation.distance, evaluation.fuel
        )
    )
    assert evaluation.travel_time == pytest.approx(sum(evaluation.route_times))
    assert evaluation.distance == pytest.approx(sum(evaluation.route_distances))
    assert evaluation.fuel == pytest.approx(sum(evaluation.route_fuels))
    assert evaluation.travel_cost == pytest.approx(sum(evaluation.route_costs))
    assert evaluation.travel_time > 0.0 and evaluation.distance > 0.0


def test_penalties_dominate_an_objective_quoted_in_rupees() -> None:
    """Penalties scale off the objective, so they stay decisive in any unit.

    Anchored to travel seconds while the objective is in rupees, a penalty would
    be about a hundredth of the saving it is meant to outweigh — and an
    infeasible solution could win.
    """
    graph = nx.DiGraph()
    for u, v in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)):
        graph.add_edge(u, v, length=1000.0, weight=120.0)

    scenario = Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=(
            Delivery(id="A", node=1, demand=1.0),
            Delivery(id="B", node=2, demand=1.0),
        ),
        vehicles=(Vehicle(id="V0", capacity=10.0),),
    )
    matrix = build_cost_matrix(graph, scenario)

    complete = evaluate(Solution(routes=((0, 1),)), scenario, matrix)
    missing_one = evaluate(Solution(routes=((0,),)), scenario, matrix)

    assert complete.feasible and not missing_one.feasible
    assert missing_one.travel_cost < complete.travel_cost  # dropping A did save
    assert missing_one.fitness > complete.fitness          # and still loses
