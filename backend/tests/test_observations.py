"""Tier 1: a measured travel time, and where it outranks everything else.

``DESIGN_DECISIONS.md`` ranks three answers to "what does this road cost?": a
fleet vehicle's **measurement**, an operator's **report** (the flat x2.9
placeholder), and the **model** from the clock. This file pins that ranking,
and pins the two places it is deliberately *not* a simple ordering — a closure
beats a measurement, and an incident's multiplier is not applied on top of one.

The graph is the two-corridor tour graph ``test_incidents.py`` and
``test_detection.py`` use, so all three files describe one network.

Most of this is the pure layer, called directly. The last section goes through
the API, because the point of the override is not that the function returns the
right number but that a measured road is priced and *logged* with it — and only
the app's own path shows that.
"""

from __future__ import annotations

import math
from datetime import datetime

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from qgati.analytics import RunLogStore
from qgati.api.main import create_app, get_graph, get_log_store, get_run_store, get_store
from qgati.api.store import ScenarioStore
from qgati.traffic import (
    ACCIDENT,
    CLOSURE,
    GPS,
    PEAK_FACTOR,
    ActiveConditions,
    Observation,
    TrafficLogRow,
    TrafficLogStore,
    TrafficState,
    observation_for,
    price_scenario,
    simulated_travel_time,
    traffic_weight_function,
    update_edge_from_observation,
)

#: The same fixtures as ``test_incidents.py``/``test_detection.py``.
ARTERIAL_S = 10.0
SIDE_STREET_S = 11.0
NORMAL_TIME = "2026-09-21T02:00:00+05:30"

ARTERIAL = (0, 1)
SIDE_STREET = (2, 1)

LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05
TOY_SPEED_KPH = 30.0
SPUR_S = 500.0

#: A measurement of the arterial that is neither its modelled 10.0s nor 2.9x it.
MEASURED_S = 12.5


def _length_of(travel_time: float) -> float:
    return travel_time * (TOY_SPEED_KPH / 3.6)


def build_tour_graph() -> nx.DiGraph:
    """Two disjoint corridors through three nodes, plus a dead-end spur."""
    graph = nx.DiGraph()
    positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (2.0, 2.0)}
    for node, (x, y) in positions.items():
        graph.add_node(
            node, x=LON_ORIGIN + x * DEGREE_SPAN, y=LAT_ORIGIN + y * DEGREE_SPAN
        )

    for u, v in ((0, 1), (1, 2), (2, 0)):
        graph.add_edge(
            u, v, highway="secondary", travel_time=ARTERIAL_S, weight=ARTERIAL_S,
            length=_length_of(ARTERIAL_S),
        )
    for u, v in ((0, 2), (2, 1), (1, 0)):
        graph.add_edge(
            u, v, highway="residential", travel_time=SIDE_STREET_S,
            weight=SIDE_STREET_S, length=_length_of(SIDE_STREET_S),
        )
    graph.add_edge(
        0, 3, highway="residential", travel_time=SPUR_S, weight=SPUR_S,
        length=_length_of(SPUR_S),
    )
    return graph


@pytest.fixture(scope="module")
def tour_graph() -> nx.DiGraph:
    return build_tour_graph()


@pytest.fixture
def log_store() -> TrafficLogStore:
    return TrafficLogStore(":memory:")


@pytest.fixture
def store() -> ScenarioStore:
    return ScenarioStore()


@pytest.fixture
def client(tour_graph, store, log_store):
    """An app over the tour graph.

    Only the four usual overrides: the fleet registry is not touched here, because
    nothing in this file starts a watcher. The run history is, because one test
    calls ``/optimize`` and that writes a row.
    """
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: tour_graph
    application.dependency_overrides[get_store] = lambda: store
    application.dependency_overrides[get_log_store] = lambda: log_store
    application.dependency_overrides[get_run_store] = lambda: RunLogStore(":memory:")
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


def state_with(
    observations, *, conditions: ActiveConditions | None = None
) -> TrafficState:
    """A normal-band state carrying ``observations``."""
    return TrafficState(
        timestamp=datetime.fromisoformat(NORMAL_TIME),
        conditions=conditions or ActiveConditions(),
        observations=observations,
    )


def measure(edge, travel_time: float = MEASURED_S) -> dict:
    """A one-entry observation mapping for ``edge``."""
    return update_edge_from_observation(
        {},
        u=edge[0],
        v=edge[1],
        travel_time=travel_time,
        observed_at=NORMAL_TIME,
    )


# --------------------------------------------------------------------------- #
# The overlay itself
# --------------------------------------------------------------------------- #
def test_an_observation_replaces_the_modelled_cost(tour_graph) -> None:
    """The whole of Tier 1, in one assertion."""
    modelled = state_with({})
    measured = state_with(measure(ARTERIAL))

    assert simulated_travel_time(
        tour_graph, ARTERIAL, modelled
    ) == pytest.approx(ARTERIAL_S)
    assert simulated_travel_time(
        tour_graph, ARTERIAL, measured
    ) == pytest.approx(MEASURED_S)


def test_a_second_measurement_replaces_the_first(tour_graph) -> None:
    """One entry per road, newest wins — not a growing history.

    The live overlay decides what a route costs *now*, so it holds the latest
    reading and nothing else. The history is the log's job.
    """
    observations = measure(ARTERIAL, 12.5)
    observations = update_edge_from_observation(
        observations, u=0, v=1, travel_time=20.0, observed_at=NORMAL_TIME
    )

    assert len(observations) == 1
    assert observations[(0, 1)].travel_time == pytest.approx(20.0)


def test_the_update_does_not_mutate_what_it_was_given() -> None:
    """It returns a new mapping: the one handed in belongs to a live request."""
    original = measure(ARTERIAL, 12.5)
    update_edge_from_observation(
        original, u=1, v=2, travel_time=30.0, observed_at=NORMAL_TIME
    )

    assert set(original) == {(0, 1)}


def test_an_observation_is_not_a_multiplier(tour_graph) -> None:
    """``multiplier`` keeps reporting the model, because it is not a measurement.

    Returning the ratio of a measured time to the base would tie the measurement
    to the model it exists to correct: recompute the base and the measurement
    would silently move with it. So the two are reported through different calls,
    and this pins that they stay different.
    """
    measured = state_with(measure(ARTERIAL))

    assert measured.multiplier(ARTERIAL) == pytest.approx(1.0)
    assert simulated_travel_time(tour_graph, ARTERIAL, measured) == pytest.approx(
        MEASURED_S
    )


def test_an_observation_may_be_found_either_way() -> None:
    """A mapping is the fast path; a sequence is what a hand-built state holds."""
    observation = Observation(
        u=0, v=1, travel_time=12.5, observed_at=NORMAL_TIME
    )

    assert observation_for({(0, 1): observation}, 0, 1) is observation
    assert observation_for([observation], 0, 1) is observation
    assert observation_for(None, 0, 1) is None
    assert observation_for([observation], 1, 0) is None


def test_the_default_source_is_gps() -> None:
    observation = Observation(u=0, v=1, travel_time=12.5, observed_at=NORMAL_TIME)

    assert observation.source == GPS
    assert observation.edge == (0, 1)
    assert observation.to_dict()["source"] == "gps"


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_a_measurement_must_be_a_finite_positive_time(bad) -> None:
    """Refused at construction, not merely at the API boundary.

    The weight function is reachable from the watcher and from a caller building
    a state by hand, and a non-finite cost here would propagate into a cost matrix
    and then into a response body that cannot be rendered.
    """
    with pytest.raises(ValueError, match="finite, positive"):
        Observation(u=0, v=1, travel_time=bad, observed_at=NORMAL_TIME)


# --------------------------------------------------------------------------- #
# The ranking
# --------------------------------------------------------------------------- #
def test_a_measurement_outranks_an_incidents_placeholder(tour_graph) -> None:
    """Tier 2 is what a measurement exists to retire.

    A road reported slow is priced at a flat x2.9 — a conservative guess at a
    delay nobody has measured. Once a vehicle has driven it, the guess has been
    replaced, and the number charged is the one that was measured.
    """
    conditions = ActiveConditions(accident_edges=frozenset({ARTERIAL}))
    guess = state_with({}, conditions=conditions)
    measured = state_with(measure(ARTERIAL), conditions=conditions)

    assert simulated_travel_time(tour_graph, ARTERIAL, guess) == pytest.approx(
        PEAK_FACTOR * ARTERIAL_S
    )
    assert simulated_travel_time(tour_graph, ARTERIAL, measured) == pytest.approx(
        MEASURED_S
    )
    # Not the placeholder scaled, and not a fresh multiplier on top of it: the
    # measurement is a time, and a time is what comes back.
    assert simulated_travel_time(tour_graph, ARTERIAL, measured) != pytest.approx(
        PEAK_FACTOR * MEASURED_S
    )


def test_a_closure_outranks_a_measurement(tour_graph) -> None:
    """A closure is a statement about passability, not an estimate of speed.

    Tier 1 replaces the model's *guess at how long a road takes*. It does not
    reopen a road that has been reported shut — a measurement taken on one is
    contradictory rather than more authoritative, and the closed set is tested
    first for exactly that reason.
    """
    conditions = ActiveConditions(closed_edges=frozenset({ARTERIAL}))
    measured = state_with(measure(ARTERIAL), conditions=conditions)

    assert simulated_travel_time(tour_graph, ARTERIAL, measured) is None
    assert measured.multiplier(ARTERIAL) == math.inf


def test_a_measurement_replaces_congestion_not_compounds_it(tour_graph) -> None:
    """A measured road is not charged the clock's multiplier as well.

    ``MEASURED_S`` was taken at 02:00, but the same road under peak must still
    cost the measurement: applying x2.9 on top would charge the same delay twice
    and would resurrect the estimate the measurement replaced.
    """
    peak = TrafficState(
        timestamp=datetime.fromisoformat("2026-09-21T09:00:00+05:30"),
        observations=measure(ARTERIAL),
    )

    assert peak.congestion == "peak"
    assert simulated_travel_time(tour_graph, ARTERIAL, peak) == pytest.approx(
        MEASURED_S
    )


def test_an_unmeasured_road_is_priced_exactly_as_before(tour_graph) -> None:
    """The default changes nothing — an empty mapping is not a slower path."""
    timestamp = datetime.fromisoformat(NORMAL_TIME)
    without = TrafficState(timestamp=timestamp)
    explicitly_empty = TrafficState(timestamp=timestamp, observations={})

    for edge in ((0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)):
        assert simulated_travel_time(
            tour_graph, edge, without
        ) == simulated_travel_time(tour_graph, edge, explicitly_empty)


def test_the_weight_function_agrees_with_the_reported_cost(tour_graph) -> None:
    """A measured edge is charged the measurement by router *and* reporter.

    ``simulated_travel_time`` is documented as the same computation the weight
    function performs, which is what makes a logged row provably the number the
    optimizer was charged. A measurement has to flow through both, or the log
    would record the model's estimate of a road the fleet had already corrected.
    """
    measured = state_with(measure(ARTERIAL))
    weight = traffic_weight_function(tour_graph, measured)

    for u, v in tour_graph.edges:
        charged = weight(u, v, tour_graph.adj[u][v])
        reported = simulated_travel_time(tour_graph, (u, v), measured)
        assert charged == pytest.approx(reported)


# --------------------------------------------------------------------------- #
# TrafficState equality and hashing
# --------------------------------------------------------------------------- #
def test_two_states_with_different_measurements_are_not_equal() -> None:
    assert state_with(measure(ARTERIAL, 12.5)) != state_with(measure(ARTERIAL, 13.5))
    assert state_with(measure(ARTERIAL)) != state_with({})


def test_a_state_carrying_measurements_is_still_hashable() -> None:
    """``observations`` is a mapping, which is unhashable on its own.

    The field is excluded from the hash rather than the dataclass being made
    unhashable, so a measured state can still go in a set — which is what the
    per-condition keying and the scenario caches rely on.
    """
    measured = state_with(measure(ARTERIAL))

    assert len({measured, state_with(measure(ARTERIAL))}) == 1
    assert len({measured, state_with({})}) == 2


# --------------------------------------------------------------------------- #
# Reach: pricing and logging
# --------------------------------------------------------------------------- #
def test_a_measured_road_is_logged_with_its_measured_time(tour_graph) -> None:
    """The row carries the measurement, not the model's estimate of it.

    ``TrafficLogRow.from_edges`` reads through ``simulated_travel_time``, so a row
    logged against a state carrying the reading records the seconds a vehicle
    actually took. That is the whole reason the overlay lives on the state rather
    than being passed separately to the weight function.
    """
    measured = state_with(measure(ARTERIAL))

    [row] = TrafficLogRow.from_edges(tour_graph, [ARTERIAL], measured)

    assert row.travel_time == pytest.approx(MEASURED_S)
    assert row.incident_type is None


def test_an_incident_row_still_carries_its_effect_word(tour_graph) -> None:
    """A reading taken under a live incident is excluded from a later baseline.

    Which is the point: a baseline trained on anomalies cannot detect them, so a
    fleet ping from a road an operator has reported slow must record *why* it was
    slow, or the detector would learn that slow is normal.
    """
    conditions = ActiveConditions(accident_edges=frozenset({ARTERIAL}))
    measured = state_with(measure(ARTERIAL), conditions=conditions)

    [row] = TrafficLogRow.from_edges(tour_graph, [ARTERIAL], measured)

    assert row.travel_time == pytest.approx(MEASURED_S)
    assert row.incident_type == ACCIDENT
    assert measured.incident_type(ARTERIAL) != CLOSURE


def test_a_scenario_priced_from_measurements_carries_them_into_its_matrix(
    tour_graph,
) -> None:
    """A measurement reaches the cost matrix, which is what the solvers see.

    Priced through ``price_scenario`` — the same call the API makes — so the
    assertion is about the app's real path and not a hand-assembled one.
    """
    from qgati.optimizer.models import Delivery, Depot, Scenario, Vehicle

    scenario = Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=(Delivery(id="D0", node=1, demand=1),),
        vehicles=(Vehicle(id="V0", capacity=1),),
    )
    states = (state_with({}), state_with(measure(ARTERIAL)))

    without, with_measurement = (
        price_scenario(tour_graph, scenario, state, log_store=None).cost_matrix
        for state in states
    )

    assert without.matrix[0, 1] == pytest.approx(ARTERIAL_S)
    assert with_measurement.matrix[0, 1] == pytest.approx(MEASURED_S)


# --------------------------------------------------------------------------- #
# Through the API
# --------------------------------------------------------------------------- #
def create_scenario(client, *, timestamp: str = NORMAL_TIME) -> dict:
    payload = {
        "kind": "explicit",
        "depot": {"node": 0},
        "deliveries": [
            {"id": "D0", "node": 1, "demand": 1},
            {"id": "D1", "node": 2, "demand": 1},
        ],
        "vehicles": [{"id": "V0", "capacity": 2}],
        "conditions": {"timestamp": timestamp},
    }
    response = client.post("/scenarios", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_the_api_does_not_write_measurements_itself(client, store) -> None:
    """Nothing in this phase sets an observation from a request.

    A client cannot post a measurement directly — ``/detect`` judges one and
    discards it — so a freshly created scenario has none. This is the assertion
    that keeps a later phase from making that accidental: a scenario's costs move
    only when a fleet or an incident says so.
    """
    created = create_scenario(client)

    record = store.get(created["scenario_id"])
    assert record.observations == {}
    assert record.effective_traffic_state().observations == {}


def test_a_scenario_without_measurements_prices_through_the_api_as_before(
    client, store
) -> None:
    """The end-to-end version of the default-changes-nothing assertion."""
    created = create_scenario(client)
    response = client.post(f"/optimize/{created['scenario_id']}")

    assert response.status_code == 200, response.text
    body = response.json()
    # Depot -> D0 -> D1 -> depot over arterial roads: 10 + 10 + 10 seconds.
    assert body["travel_time"] == pytest.approx(3 * ARTERIAL_S)
    assert body["routes"][0]["travel_cost"] > 0.0
