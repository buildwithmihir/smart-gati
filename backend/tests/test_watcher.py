"""The simulated GPS fleet: what a tick measures, and what it does with it.

Two things make this testable, and both are design decisions rather than
test conveniences.

**A tick is a function.** ``ScenarioWatcher.tick`` is what the timer thread calls
and what ``POST /scenarios/{id}/watcher/tick`` calls, so nearly everything here
drives it directly through the API and asserts on the answer. Only one test
sleeps, and it is the one whose subject *is* the background thread.

**Time is a parameter.** ``time_scale`` says how many seconds of driving a tick
represents. The tour graph's roads take 10-11 seconds, so the default (one
second per tick) keeps a vehicle on the same road for several ticks — which is
what lets a test watch a road's cost move rather than watching vehicles teleport.

The graph is the same two-corridor tour graph the other traffic tests use.
"""

from __future__ import annotations

import time as wall_clock

import networkx as nx
import pytest
from fastapi.testclient import TestClient
from networkx.utils import graphs_equal

from qgati.analytics import RunLogStore
from qgati.api.main import (
    create_app,
    get_graph,
    get_log_store,
    get_run_store,
    get_store,
    get_watchers,
)
from qgati.api.store import ScenarioStore
from qgati.fleet import (
    DEFAULT_SEED,
    NOISE_SIGMA,
    WatcherRegistry,
    noisy_reading,
)
from qgati.traffic import (
    FALLBACK_FACTOR,
    PEAK_FACTOR,
    SLOW,
    TrafficLogStore,
    edge_of,
    simulated_travel_time,
)

ARTERIAL_S = 10.0
SIDE_STREET_S = 11.0

#: 02:00 prices as ``normal``, so every modelled road costs its own base time and
#: an assertion can name the number rather than derive it.
NORMAL_TIME = "2026-09-21T02:00:00+05:30"

LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05
TOY_SPEED_KPH = 30.0
SPUR_S = 500.0

#: One second of driving per tick, so a vehicle stays on a 10-11 s road for ten
#: ticks and a road's cost can be watched moving rather than assumed.
#:
#: The two numbers are what matter, not the interval on its own: a tick advances
#: the fleet by ``interval_seconds * time_scale``, so a 300 s interval paired with
#: a 1/300 scale drives the same second per tick while leaving the *background
#: thread* effectively asleep. That is deliberate. With ``interval_seconds=1`` the
#: timer would fire during a test and advance a vehicle between two explicit
#: ticks, which is exactly the movement ``test_repeat_ticks_do_not_replay_the_
#: same_reading`` asserts has not happened. Only the two tests whose subject is
#: the thread itself ask for a short interval.
IDLE_INTERVAL = 300.0
SCRIPTED_STEP_SECONDS = 1.0
ONE_SECOND = {
    "interval_seconds": IDLE_INTERVAL,
    "time_scale": SCRIPTED_STEP_SECONDS / IDLE_INTERVAL,
}

#: A deterministic solver. QPSO is the production default but its routes depend on
#: the seed, and these tests are about the fleet rather than about the search.
DETERMINISTIC = {"solver": "savings"}

#: Every field a tick's response may carry. Asserted as an exact set, because the
#: absence of a route, a cost or a solver from it is the structural evidence that
#: measuring a fleet does not re-optimize it.
TICK_FIELDS = frozenset(
    {
        "scenario_id",
        "tick",
        "at",
        "readings",
        "changed_legs",
        "rows_logged",
        "conditions_mutated",
    }
)


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
def watchers() -> WatcherRegistry:
    """A registry of its own, so one test's fleet cannot tick against another's."""
    return WatcherRegistry()


@pytest.fixture
def client(tour_graph, store, log_store, watchers):
    """An app over the tour graph, with a fleet registry that lives for the test.

    All the overrides are callables returning one instance: a dependency override
    is invoked per request, so passing a class would hand every request a fresh,
    empty one. The app's shutdown stops whatever this registry holds, which is why
    a test can start a watcher and walk away from it.

    The run history is in-memory for the same reason the traffic log is, and one
    more: starting a fleet *solves* the scenario, so without this override every
    test in this file would append a dispatch row to the developer's real
    ``backend/data/run_history.db``.
    """
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: tour_graph
    application.dependency_overrides[get_store] = lambda: store
    application.dependency_overrides[get_log_store] = lambda: log_store
    application.dependency_overrides[get_run_store] = lambda: RunLogStore(":memory:")
    application.dependency_overrides[get_watchers] = lambda: watchers
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def create_scenario(client, *, timestamp: str = NORMAL_TIME, vehicles=None) -> dict:
    """Create the two-stop tour scenario through the API.

    One vehicle of capacity two by default, which covers both stops in a single
    route. ``vehicles`` gives the multi-route case the fleet's spreading is
    about — two one-unit vehicles cannot share a route, so the solver must
    dispatch both.
    """
    payload = {
        "kind": "explicit",
        "depot": {"node": 0},
        "deliveries": [
            {"id": "D0", "node": 1, "demand": 1},
            {"id": "D1", "node": 2, "demand": 1},
        ],
        "vehicles": vehicles or [{"id": "V0", "capacity": 2}],
        "conditions": {"timestamp": timestamp},
    }
    response = client.post("/scenarios", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def start(client, scenario_id: str, **body) -> dict:
    """Start a watcher, or fail loudly with the API's own reason."""
    response = client.post(
        f"/scenarios/{scenario_id}/watcher", json={**DETERMINISTIC, **ONE_SECOND, **body}
    )
    assert response.status_code == 201, response.text
    return response.json()


def tick(client, scenario_id: str) -> dict:
    response = client.post(f"/scenarios/{scenario_id}/watcher/tick")
    assert response.status_code == 200, response.text
    return response.json()


def reading_for(body: dict, edge) -> dict:
    """The one reading in ``body`` that measured ``edge``."""
    found = [
        reading for reading in body["readings"] if reading["edge"] == edge_json(edge)
    ]
    assert len(found) == 1, f"expected exactly one reading on {edge}: {body['readings']}"
    return found[0]


def edge_json(edge) -> dict:
    return {"u": edge[0], "v": edge[1]}


def cost_of(store, graph, scenario_id: str, edge) -> float | None:
    """What the app currently charges for one *road*, under the scenario's state.

    Read through the scenario's effective state and ``simulated_travel_time`` —
    the same call routing makes — because a cost matrix is between a scenario's
    stops, not between road-graph nodes. There is no other honest way to ask what
    a road costs, and asking it this way is what makes the answer the number the
    optimizer is charged.
    """
    record = store.get(scenario_id)
    return simulated_travel_time(
        graph, edge_of(graph, edge[0], edge[1]), record.effective_traffic_state()
    )


# --------------------------------------------------------------------------- #
# The noise
# --------------------------------------------------------------------------- #
def test_the_noise_is_centred_on_the_model() -> None:
    """A lognormal multiplier with mean exactly 1.0, not approximately.

    If the fleet were biased the detector would slowly learn a wrong normal, so
    the mean is asserted over enough draws to distinguish 1.0 from 1.02.
    """
    import random

    rng = random.Random(11)
    draws = [noisy_reading(100.0, rng) / 100.0 for _ in range(20_000)]

    assert sum(draws) / len(draws) == pytest.approx(1.0, abs=0.005)


def test_the_noise_is_always_a_positive_time() -> None:
    """A travel time can be wrong, but it cannot be zero or negative."""
    import random

    rng = random.Random(3)

    assert all(noisy_reading(10.0, rng) > 0.0 for _ in range(5_000))


def test_the_noise_is_small_enough_not_to_trip_the_fallback() -> None:
    """The amplitude is derived from the detector's own margin, not chosen.

    An undisturbed road is measured against the detector's incident-free
    expectation, so a reading more than ``FALLBACK_FACTOR`` above it is flagged.
    That must be rare: a fleet that flags one ordinary drive in ten would make the
    detector useless. Asserted directly rather than left to a demo to notice.
    """
    import random

    rng = random.Random(5)
    draws = [noisy_reading(100.0, rng) for _ in range(20_000)]
    flagged = sum(1 for value in draws if value > FALLBACK_FACTOR * 100.0)

    assert flagged / len(draws) < 0.005, (
        f"{flagged} of {len(draws)} ordinary readings exceeded "
        f"{FALLBACK_FACTOR}x the expectation at sigma={NOISE_SIGMA}"
    )


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_starting_a_fleet_dispatches_routes_and_takes_a_first_tick(
    client, store, tour_graph
) -> None:
    """The response is self-sufficient: routes, vehicles, and one reading each.

    Starting a watcher is what solves the scenario, so the routes come back with
    it rather than needing a second ``/optimize`` call that would solve the same
    instance again.
    """
    created = create_scenario(client)

    body = start(client, created["scenario_id"])

    assert body["running"] is True
    assert body["ticks"] == 1
    assert body["solver"] == "savings"
    assert body["routes"], "a dispatched fleet has at least one route"
    assert body["vehicles"], "a dispatched fleet has at least one vehicle"
    assert body["first_tick"]["tick"] == 1
    assert len(body["first_tick"]["readings"]) == len(body["vehicles"])
    assert body["first_tick"]["rows_logged"] == len(
        [r for r in body["first_tick"]["readings"] if r["observed_travel_time"]]
    )


def test_a_vehicle_is_reported_on_a_road_that_exists(client, tour_graph) -> None:
    """A ping names a real road, and the modelled time is that road's own."""
    created = create_scenario(client)

    body = start(client, created["scenario_id"])

    for vehicle in body["vehicles"]:
        edge = (vehicle["edge"]["u"], vehicle["edge"]["v"])
        assert tour_graph.has_edge(*edge)
        assert 0.0 <= vehicle["progress"] <= 1.0


def test_a_second_fleet_on_one_scenario_is_a_conflict(client) -> None:
    """Two watchers would fight over the same compare-and-swap every tick."""
    created = create_scenario(client)
    start(client, created["scenario_id"])

    response = client.post(
        f"/scenarios/{created['scenario_id']}/watcher", json={**DETERMINISTIC, **ONE_SECOND}
    )

    assert response.status_code == 409
    assert "already has a fleet" in response.json()["detail"]


def test_status_reports_the_fleet_and_stopping_ends_it(client) -> None:
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    running = client.get(f"/scenarios/{scenario_id}/watcher")
    assert running.status_code == 200
    assert running.json()["running"] is True

    stopped = client.delete(f"/scenarios/{scenario_id}/watcher")
    assert stopped.status_code == 200
    assert stopped.json()["running"] is False
    # Stopping is not erasing: the vehicles are reported where they stood.
    assert stopped.json()["vehicles"]

    assert client.get(f"/scenarios/{scenario_id}/watcher").status_code == 404


def test_stopping_a_fleet_that_is_not_running_is_404(client) -> None:
    created = create_scenario(client)

    response = client.delete(f"/scenarios/{created['scenario_id']}/watcher")

    assert response.status_code == 404
    assert "no fleet running" in response.json()["detail"]


def test_a_fleet_on_an_unknown_scenario_is_404(client) -> None:
    assert client.get("/scenarios/nope/watcher").status_code == 404
    assert client.post("/scenarios/nope/watcher/tick").status_code == 404


@pytest.mark.parametrize("interval", [0, -1, 10_000])
def test_the_interval_is_bounded(client, interval) -> None:
    """A tick every 10,000 s is not a fleet, and one every 0 s is a spin loop."""
    created = create_scenario(client)

    response = client.post(
        f"/scenarios/{created['scenario_id']}/watcher",
        json={**DETERMINISTIC, "interval_seconds": interval},
    )

    assert response.status_code == 422


def test_an_unknown_solver_is_refused_before_anything_starts(client) -> None:
    """The same 422 ``/optimize`` gives, from the same helper."""
    created = create_scenario(client)

    response = client.post(
        f"/scenarios/{created['scenario_id']}/watcher", json={"solver": "nope"}
    )

    assert response.status_code == 422
    assert "nope" in response.json()["detail"]
    assert client.get(f"/scenarios/{created['scenario_id']}/watcher").status_code == 404


def test_the_background_thread_actually_ticks(client) -> None:
    """The one test that sleeps, because the thread is its subject.

    Everything else drives ``POST .../watcher/tick`` and asserts on a synchronous
    answer. This one asserts the thing no synchronous test can: that a started
    fleet advances on its own, on the interval it was given.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id, interval_seconds=1, time_scale=1.0)

    deadline = wall_clock.monotonic() + 10.0
    ticks = 1
    while wall_clock.monotonic() < deadline and ticks < 3:
        wall_clock.sleep(0.25)
        ticks = client.get(f"/scenarios/{scenario_id}/watcher").json()["ticks"]

    assert ticks >= 3, f"the fleet only reached tick {ticks} in ten seconds"

    client.delete(f"/scenarios/{scenario_id}/watcher")


# --------------------------------------------------------------------------- #
# What a tick measures
# --------------------------------------------------------------------------- #
def test_a_tick_reports_one_reading_per_vehicle(client, tour_graph) -> None:
    created = create_scenario(
        client,
        vehicles=[{"id": "V0", "capacity": 1}, {"id": "V1", "capacity": 1}],
    )
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    body = tick(client, scenario_id)

    reported = [r["vehicle_id"] for r in body["readings"] if r["observed_travel_time"]]
    assert sorted(reported) == sorted(
        vehicle["vehicle_id"]
        for vehicle in client.get(f"/scenarios/{scenario_id}/watcher").json()["vehicles"]
    )
    assert len(body["readings"]) == 2


def test_repeat_ticks_do_not_replay_the_same_reading(client) -> None:
    """The anti-script assertion: a measurement, not a recording.

    One second of driving per tick keeps a vehicle on its first road for ten
    ticks, so two consecutive ticks measure the *same* road — and must not report
    the same number twice. A log of identical rows is the state this whole phase
    exists to change.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    first = tick(client, scenario_id)
    second = tick(client, scenario_id)

    edge = first["readings"][0]["edge"]
    assert edge == second["readings"][0]["edge"], "the vehicle should not have moved"
    assert (
        first["readings"][0]["observed_travel_time"]
        != second["readings"][0]["observed_travel_time"]
    )


def test_a_seed_makes_the_fleet_reproducible(client) -> None:
    """Same seed, same readings; different seed, different ones.

    Asserted across two *separate* scenarios, so it is the fleet's randomness
    being pinned rather than a shared object being ticked twice. A deterministic
    solver is used so the routes cannot be the thing that differs.
    """
    first = create_scenario(client)["scenario_id"]
    second = create_scenario(client)["scenario_id"]
    third = create_scenario(client)["scenario_id"]

    same_a = start(client, first, seed=7)["first_tick"]["readings"]
    same_b = start(client, second, seed=7)["first_tick"]["readings"]
    other = start(client, third, seed=8)["first_tick"]["readings"]

    assert [r["observed_travel_time"] for r in same_a] == [
        r["observed_travel_time"] for r in same_b
    ]
    assert [r["observed_travel_time"] for r in same_a] != [
        r["observed_travel_time"] for r in other
    ]


def test_the_default_seed_is_stated_not_incidental() -> None:
    """``DEFAULT_SEED`` fixes the noise, so a demo prints the same table twice."""
    assert DEFAULT_SEED == 0


def test_an_ordinary_reading_is_not_flagged(client) -> None:
    """The fleet's own noise must not read as an anomaly.

    The road is measured against the detector's incident-free expectation, and
    the two differ only by the noise. See the amplitude test above for the
    arithmetic — that one is the statistical claim, over 20,000 draws; this is
    the same claim end to end, through the tick, on one seeded fleet.

    Seeded because the claim is about a *draw*: at ``sigma=0.06`` the fallback
    trips roughly one reading in a thousand, so an unseeded four-tick run would
    fail about once in 235 runs on noise alone. A fixed seed makes that either
    a steady pass or an immediate, reproducible failure, which is the honest
    way to assert a probabilistic property — and it is the same reason
    ``demo_watcher.py`` takes a seed.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id, seed=3)

    for _ in range(4):
        body = tick(client, scenario_id)
        # Deterministic half: a freshly created scenario's fleet has no history to
        # score against, so every reading is judged by the rule-of-thumb. That is
        # the claim the module docstring makes, and this pins it without depending
        # on a draw.
        assert all(r["rule"] == "insufficient_samples" for r in body["readings"])
        assert all(not r["flag"] for r in body["readings"]), body["readings"]
        assert body["changed_legs"] >= 0


def test_a_vehicle_leaves_the_road_when_it_reaches_the_end(client) -> None:
    """A finished route is a legitimate state, not an error to report loudly.

    Four hundred seconds per tick crosses the whole 30-second corridor in one go,
    so the next tick has nothing to measure and says so in ``note`` instead of
    inventing a reading.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id, time_scale=400.0)

    body = tick(client, scenario_id)

    assert all(r["observed_travel_time"] is None for r in body["readings"])
    assert all("not on a road" in r["note"] for r in body["readings"])
    assert body["changed_legs"] == 0
    assert body["rows_logged"] == 0


# --------------------------------------------------------------------------- #
# Tier 1, end to end
# --------------------------------------------------------------------------- #
def test_an_incident_does_not_displace_a_measurement(client, store, tour_graph) -> None:
    """The tier rule, end to end, and it is the surprising direction.

    ``x2.9`` is a placeholder for a delay nobody has measured. The fleet has
    measured this road — starting a watcher runs a tick inline, and that tick
    reads the road every vehicle is on — so reporting it slow does not reprice it
    at all. A measurement outranks a placeholder, which is what "Tier 1" means.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    edge = _first_edge(client, scenario_id)
    measured = cost_of(store, tour_graph, scenario_id, edge)
    assert measured is not None, "the first tick measured the road the vehicle is on"

    _report_slow(client, scenario_id, edge)

    assert cost_of(store, tour_graph, scenario_id, edge) == pytest.approx(measured), (
        "the incident's placeholder replaced a measurement it is ranked below"
    )


def test_a_tick_replaces_the_placeholder_with_the_measurement(
    client, store, tour_graph
) -> None:
    """The headline claim, and the reason the prompt asks for it specifically.

    The detector measures the reading against the incident's flat ``x2.9`` — the
    placeholder, exactly. What the road is charged afterwards is the fleet's own
    number instead: a time, not a factor, and therefore not ``x2.9`` of anything.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    edge = _first_edge(client, scenario_id)
    _report_slow(client, scenario_id, edge)

    body = tick(client, scenario_id)
    reading = reading_for(body, edge)
    after = cost_of(store, tour_graph, scenario_id, edge)

    assert reading["modelled_travel_time"] == pytest.approx(PEAK_FACTOR * ARTERIAL_S), (
        "the reading was drawn around the incident's flat placeholder"
    )
    assert after == pytest.approx(reading["observed_travel_time"])
    assert after != pytest.approx(PEAK_FACTOR * ARTERIAL_S)
    assert body["changed_legs"] > 0


def test_the_road_under_an_incident_is_flagged(client) -> None:
    """The detector fires on the road the fleet is reporting from.

    A 2.9x reading against the detector's incident-free expectation is far past
    the 1.2x margin, so the fallback flags it without needing any history at all —
    which is the state a freshly created scenario is always in.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    edge = _first_edge(client, scenario_id)
    _report_slow(client, scenario_id, edge)

    reading = reading_for(tick(client, scenario_id), edge)

    assert reading["flag"] is True
    assert reading["condition"] == "normal"
    assert reading["modelled_travel_time"] == pytest.approx(PEAK_FACTOR * ARTERIAL_S)
    assert reading["observed_travel_time"] > FALLBACK_FACTOR * ARTERIAL_S
    assert reading["reason"]


def test_the_detector_is_judging_against_an_unmeasured_road(client) -> None:
    """The measurement must not be scored against itself.

    The tick builds its own expectation from an observation-free state. If it
    reused the scenario's live state — which carries the fleet's own readings —
    every reading would match its expectation exactly, nothing would ever be
    flagged, and the fleet would drift upward by one random step per tick.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    edge = _first_edge(client, scenario_id)

    first = reading_for(tick(client, scenario_id), edge)
    second = reading_for(tick(client, scenario_id), edge)

    # Both readings are of the same road, whose modelled time has not moved even
    # though the road now carries a measurement from the tick before.
    assert first["modelled_travel_time"] == pytest.approx(second["modelled_travel_time"])
    assert first["modelled_travel_time"] == pytest.approx(ARTERIAL_S)


def test_a_closed_road_is_not_fabricated_a_reading(client, store, tour_graph) -> None:
    """The graph is only read, so a closure is reported, never measured away."""
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    edge = _first_edge(client, scenario_id)

    response = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": "closure", "edge": edge_json(edge)},
    )
    assert response.status_code == 201, response.text

    body = tick(client, scenario_id)
    reading = reading_for(body, edge)

    assert reading["observed_travel_time"] is None
    assert "impassable" in reading["note"]
    assert cost_of(store, tour_graph, scenario_id, edge) is None


# --------------------------------------------------------------------------- #
# Ingestion, and what the log learns
# --------------------------------------------------------------------------- #
def test_a_tick_writes_one_row_per_reading(client, log_store) -> None:
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    before = log_store.count()

    body = tick(client, scenario_id)

    assert body["rows_logged"] == len(
        [r for r in body["readings"] if r["observed_travel_time"]]
    )
    assert log_store.count() == before + body["rows_logged"]


def test_a_logged_row_carries_the_measured_seconds(client, log_store) -> None:
    """Not the model's estimate of them — the whole point of Tier 1.

    A row recording the estimate would teach a later baseline the number the
    measurement just replaced, and the fleet would have contributed nothing the
    model did not already know. Rows come back newest first, so the tick's own
    row is the one at the top for that road.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    edge = _first_edge(client, scenario_id)

    body = tick(client, scenario_id)
    observed = reading_for(body, edge)["observed_travel_time"]

    rows, _ = log_store.read(limit=20, road_u=edge[0], road_v=edge[1])

    # ``abs`` rather than pytest's default relative tolerance: the column stores
    # milliseconds (``TrafficLogRow.from_edges`` rounds to 3 dp), so a reading of
    # ~10 s lands up to 5e-4 away from the float the response reported. The
    # default 1e-6 relative tolerance is 1e-5 here, fifty times tighter than the
    # precision the column actually keeps.
    assert rows[0].travel_time == pytest.approx(observed, abs=1e-3)
    assert rows[0].incident_type is None
    # The scenario's creation row for the same road is still there underneath,
    # carrying the model's own number — which is what the measurement displaced
    # rather than what was deleted.
    assert ARTERIAL_S in [row.travel_time for row in rows]


def test_a_reading_under_an_incident_carries_the_effect_word(client, log_store) -> None:
    """So a later baseline skips it: a baseline trained on anomalies cannot detect them."""
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    edge = _first_edge(client, scenario_id)
    _report_slow(client, scenario_id, edge)

    tick(client, scenario_id)

    rows, _ = log_store.read(limit=20, road_u=edge[0], road_v=edge[1])
    assert any(row.incident_type == "accident" for row in rows)


def test_an_ordinary_reading_is_plain_history(client, log_store) -> None:
    """And so counts towards the baseline the next scenario is judged against.

    This is the ingestion the detection phase deferred, and it is what finally
    gives a road a history with a spread in it.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    edge = _first_edge(client, scenario_id)

    tick(client, scenario_id)

    rows, _ = log_store.read(limit=20, road_u=edge[0], road_v=edge[1])
    fleet_rows = [row for row in rows if row.travel_time is not None]

    assert fleet_rows
    assert all(row.incident_type is None for row in fleet_rows)


# --------------------------------------------------------------------------- #
# The boundaries
# --------------------------------------------------------------------------- #
def test_the_graph_is_never_mutated(client, tour_graph) -> None:
    """Weighting is a function, never a write.

    The Delhi graph is a cached, shared, read-only object, so "the edge's weight
    was updated" can only ever mean *the cost the app charges for that road*. A
    closure expressed by deleting an edge would leak into every later request.
    """
    before = tour_graph.copy()
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    _report_slow(client, scenario_id, _first_edge(client, scenario_id))
    tick(client, scenario_id)

    assert graphs_equal(before, tour_graph)


def test_a_tick_returns_no_solution(client) -> None:
    """A reading is a signal, not an action — the same boundary ``/detect`` draws.

    Re-optimizing on what the fleet found is ``reopt``'s job, in a later phase,
    and the structural evidence is that this response has nowhere to put a route.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    body = tick(client, scenario_id)

    assert set(body) == TICK_FIELDS


def test_a_scenario_is_still_optimizable_while_a_fleet_runs(client) -> None:
    """The fleet changes costs; it does not take the scenario away."""
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)
    tick(client, scenario_id)

    response = client.post(f"/optimize/{scenario_id}")

    assert response.status_code == 200, response.text
    assert response.json()["routes"]


def test_reading_the_fleet_does_not_move_it(client) -> None:
    """``GET`` does not tick, and neither does an optimize."""
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    client.get(f"/scenarios/{scenario_id}/watcher")
    client.get(f"/scenarios/{scenario_id}/watcher")
    client.post(f"/optimize/{scenario_id}")

    assert client.get(f"/scenarios/{scenario_id}/watcher").json()["ticks"] == 1


def test_a_fleet_does_not_block_a_request(client) -> None:
    """The whole reason it is a thread rather than a request."""
    created = create_scenario(client)
    start(client, created["scenario_id"], interval_seconds=1)

    assert client.get("/health").json() == {"status": "ok"}

    client.delete(f"/scenarios/{created['scenario_id']}/watcher")


def test_a_watcher_stops_when_its_scenario_disappears(client, store) -> None:
    """The tick loop's exit condition, driven directly.

    Left running it would log a failure every interval forever against a scenario
    that no longer exists.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]
    start(client, scenario_id)

    store.clear()

    response = client.post(f"/scenarios/{scenario_id}/watcher/tick")

    assert response.status_code == 404
    assert client.get(f"/scenarios/{scenario_id}/watcher").status_code == 404


def test_several_fleets_run_independently(client, tour_graph) -> None:
    """One registry, one entry per scenario — and no shared randomness.

    Each watcher owns its ``random.Random``, so a fleet's readings depend on its
    own seed and not on how many other fleets happen to be running.
    """
    first = create_scenario(client)["scenario_id"]
    second = create_scenario(client)["scenario_id"]

    start(client, first, seed=1)
    start(client, second, seed=1)

    left = tick(client, first)["readings"][0]["observed_travel_time"]
    right = tick(client, second)["readings"][0]["observed_travel_time"]

    # Both are tick 3 of their own fleet with the same seed and the same roads,
    # so they must agree — which they would not if one RNG were shared.
    assert left == right


# --------------------------------------------------------------------------- #
# Small helpers that read the app's own answers
# --------------------------------------------------------------------------- #
def _first_edge(client, scenario_id: str) -> tuple[int, int]:
    """The road the first vehicle is currently on, as the API reports it."""
    vehicles = client.get(f"/scenarios/{scenario_id}/watcher").json()["vehicles"]
    edge = vehicles[0]["edge"]
    assert edge is not None, "a dispatched vehicle starts on a road"
    return edge["u"], edge["v"]


def _report_slow(client, scenario_id: str, edge) -> dict:
    """Report one road as slow, the way an operator would."""
    response = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": SLOW, "edge": edge_json(edge)},
    )
    assert response.status_code == 201, response.text
    return response.json()
