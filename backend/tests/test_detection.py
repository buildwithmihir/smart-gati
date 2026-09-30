"""Statistical anomaly detection: z-scores, and the fallback when there is too little history.

The rules are from ``DESIGN_DECISIONS.md`` § Detection and have not been softened
here: mean and standard deviation per **(road, condition)**, ``z > 2`` flags, and
below ten samples the z-score is skipped entirely in favour of the simulator's own
expectation with a 1.2x margin.

Two things about this fixture shape matter to what can be asserted on it.

**History is seeded, not simulated.** The prompt's own scenario is "feed synthetic
historical samples for one edge", and it has to be synthetic: an ordinary logged
``travel_time`` is ``base_travel_time x congestion_multiplier``, both deterministic,
so a road's real history at one condition band is the *same number* every time and
its standard deviation is exactly zero. That case is real and is tested here — but
it cannot show a z-score above 1 working, because no z-score above 1 exists in it.
Seeding values with spread is what exercises the arithmetic, and the zero-spread
case is then pinned separately rather than left to chance.

**Rows are written straight to the log store**, not through a scenario. That is
the only way to give a road a history: pricing a scenario logs one row per road,
which is one sample, which is always the fallback. ``seed_history`` is that seam.

The graph is the same two-corridor tour graph ``test_incidents.py`` uses — every
leg has an alternative, one constant speed throughout, and a dead-end spur so an
edge no route uses is available.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from qgati.api.main import create_app, get_graph, get_log_store, get_store
from qgati.api.store import ScenarioStore
from qgati.traffic import (
    CLEARED,
    CLOSURE,
    FALLBACK_FACTOR,
    INSUFFICIENT_SAMPLES,
    MIN_SAMPLES,
    ROAD_CLOSURE,
    SLOW,
    Z_SCORE,
    Z_THRESHOLD,
    RoadStats,
    TrafficLogRow,
    TrafficLogStore,
    build_baseline,
    detect,
    z_score,
)

#: The arterial hop and the side-street one, in seconds — the same fixture as
#: ``test_incidents.py``, so the two files describe one graph.
ARTERIAL_S = 10.0
SIDE_STREET_S = 11.0

#: 02:00 prices as ``normal`` (outside the 06:00-22:00 daytime band) and 09:00 as
#: ``peak``. Two bands, so per-condition keying is testable.
NORMAL_TIME = "2026-09-21T02:00:00+05:30"
PEAK_TIME = "2026-09-21T09:00:00+05:30"
NORMAL = "normal"
PEAK = "peak"

#: The arterial road the seeded history is attached to, and the side street that
#: carries the fallback's test of *whose* expectation is used.
ARTERIAL = (0, 1)
SIDE_STREET = (2, 1)

LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05
TOY_SPEED_KPH = 30.0
SPUR_S = 500.0

#: A history with spread. Mean 40.0s, sample standard deviation ``sqrt(2.5)``.
SPREAD_VALUES = [40.0, 42.0, 38.0, 41.0, 39.0] * 3


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
    """An empty in-memory log, so a test never touches collected data."""
    return TrafficLogStore(":memory:")


@pytest.fixture
def store() -> ScenarioStore:
    """One scenario store per test, handed to the app and to the test alike."""
    return ScenarioStore()


@pytest.fixture
def client(tour_graph, store, log_store):
    """An app over the tour graph, with stores that live for the test.

    All three overrides must be callables returning one instance: a dependency
    override is invoked per request, so passing the class would hand every request
    a brand-new empty store.
    """
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: tour_graph
    application.dependency_overrides[get_store] = lambda: store
    application.dependency_overrides[get_log_store] = lambda: log_store
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def seed_history(
    log_store: TrafficLogStore,
    edge: tuple[int, int],
    values,
    *,
    condition: str = NORMAL,
    incident_type: str | None = None,
    timestamp: str = NORMAL_TIME,
) -> int:
    """Write one log row per value for ``edge``, as that road's past.

    Straight to the store rather than through a scenario: pricing a scenario logs
    one row per road, and one row is one sample, which is always the fallback.
    This is how a road is given a history worth computing a standard deviation
    from.
    """
    u, v = edge
    rows = [
        TrafficLogRow(
            road_u=u,
            road_v=v,
            timestamp=timestamp,
            day_of_week="Monday",
            time_of_day=timestamp[11:16],
            traffic_condition=condition,
            incident_type=incident_type,
            travel_time=value,
        )
        for value in values
    ]
    return log_store.write(rows)


def create_scenario(client, *, timestamp: str = NORMAL_TIME) -> dict:
    """Create the two-stop tour scenario through the API.

    Creation is where the baseline is taken, so anything seeded after this is
    invisible to the scenario — which is the once-per-scenario behaviour, asserted
    directly below.
    """
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


def observe(client, scenario_id: str, value: float, edge=ARTERIAL, **extra) -> dict:
    """POST one observation and return the verdict body."""
    response = client.post(
        f"/scenarios/{scenario_id}/detect",
        json={"edge": {"u": edge[0], "v": edge[1]}, "travel_time": value, **extra},
    )
    assert response.status_code == 200, response.text
    return response.json()


def stats_for(u: int, v: int, condition: str, values) -> RoadStats:
    """A :class:`RoadStats` over ``values``, computed the way the module does.

    Recomputed from the same numbers the fixture seeded, so a test asserting
    ``mean + 3 sigma`` is asserting against the module's own arithmetic rather
    than against a second one written into the test.
    """
    rows = [
        TrafficLogRow(
            road_u=u,
            road_v=v,
            timestamp=NORMAL_TIME,
            day_of_week="Monday",
            time_of_day="02:00",
            traffic_condition=condition,
            travel_time=value,
        )
        for value in values
    ]
    found = build_baseline(rows).for_edge(u, v, condition)
    assert found is not None
    return found


def flat(count: int, mean: float, std_dev: float = 0.0) -> RoadStats:
    """A hand-made :class:`RoadStats`, for pinning a rule at its boundary.

    Handed to :func:`detect` directly, where a test needs the observation to sit
    exactly on a threshold — which a value computed from stored floats does not
    reliably do.
    """
    return RoadStats(
        road_u=0, road_v=1, condition=NORMAL, count=count, mean=mean, std_dev=std_dev
    )


# --------------------------------------------------------------------------- #
# Building the baseline
# --------------------------------------------------------------------------- #
def test_history_is_keyed_by_road_and_condition() -> None:
    """The same road's peak and normal histories are separate samples."""
    rows = [
        TrafficLogRow(
            road_u=0, road_v=1, timestamp=NORMAL_TIME, day_of_week="Monday",
            time_of_day="02:00", traffic_condition=NORMAL, travel_time=10.0,
        ),
        TrafficLogRow(
            road_u=0, road_v=1, timestamp=PEAK_TIME, day_of_week="Monday",
            time_of_day="09:00", traffic_condition=PEAK, travel_time=29.0,
        ),
    ]
    baseline = build_baseline(rows)

    assert baseline.for_edge(0, 1, NORMAL).mean == 10.0
    assert baseline.for_edge(0, 1, PEAK).mean == 29.0
    assert baseline.for_edge(0, 1, NORMAL).count == 1


def test_a_road_with_no_history_has_no_entry() -> None:
    """Absent, not zero — there is a difference between none and nothing."""
    assert build_baseline([]).for_edge(0, 1, NORMAL) is None


def test_mean_and_standard_deviation_are_the_sample_estimator() -> None:
    """Checked against values computed by hand, not against the implementation.

    ``[10, 12, 14, 16, 18]``: mean 14, and the *sample* standard deviation
    (dividing by n-1 = 4) is ``sqrt(40/4) = sqrt(10)``. The population figure
    would be ``sqrt(40/5) = sqrt(8)``, so this pins which estimator is in use.
    """
    stats = stats_for(0, 1, NORMAL, [10.0, 12.0, 14.0, 16.0, 18.0])

    assert stats.count == 5
    assert stats.mean == pytest.approx(14.0)
    assert stats.std_dev == pytest.approx(math.sqrt(10.0))
    assert stats.std_dev != pytest.approx(math.sqrt(8.0))


def test_a_single_sample_has_no_spread() -> None:
    """One point is perfectly repeatable as far as anything can tell."""
    stats = stats_for(0, 1, NORMAL, [10.0])

    assert stats.count == 1
    assert stats.std_dev == 0.0


def test_incident_rows_are_not_history(log_store) -> None:
    """A baseline trained on anomalies cannot detect them.

    The slow rows here are the *only* variance in the data. If they counted, the
    mean would be pulled up and the spread invented — and a road routinely shut at
    18:00 would report a shut road at 18:00 as perfectly ordinary.
    """
    seed_history(log_store, ARTERIAL, [10.0] * 12, condition=NORMAL)
    seed_history(log_store, ARTERIAL, [29.0] * 6, condition=NORMAL, incident_type=SLOW)
    seed_history(
        log_store, ARTERIAL, [None], condition=NORMAL, incident_type=ROAD_CLOSURE
    )

    rows, _ = log_store.read(limit=100)
    stats = build_baseline(rows).for_edge(*ARTERIAL, NORMAL)

    assert stats.count == 12
    assert stats.mean == pytest.approx(10.0)
    assert stats.std_dev == 0.0


def test_a_closed_road_is_not_an_observation_of_a_travel_time() -> None:
    """``travel_time IS NULL`` is how an impassable road is logged.

    There is nothing to average, so the row carries no sample. Both filters
    exclude it independently here, which is why it is asserted on its own.
    """
    rows = [
        TrafficLogRow(
            road_u=0, road_v=1, timestamp=NORMAL_TIME, day_of_week="Monday",
            time_of_day="02:00", traffic_condition=NORMAL, travel_time=None,
        ),
    ]

    assert build_baseline(rows).for_edge(0, 1, NORMAL) is None


def test_a_road_cleared_is_not_history_either(log_store) -> None:
    """``road_clear`` is an operator action, not evidence of ordinary running."""
    seed_history(log_store, ARTERIAL, [10.0], condition=NORMAL, incident_type=CLEARED)

    rows, _ = log_store.read(limit=10)
    assert build_baseline(rows).for_edge(*ARTERIAL, NORMAL) is None


# --------------------------------------------------------------------------- #
# The z-score
# --------------------------------------------------------------------------- #
def test_z_score_is_the_deviation_over_the_spread() -> None:
    stats = flat(12, mean=40.0, std_dev=4.0)

    assert z_score(50.0, stats) == pytest.approx(2.5)
    assert z_score(40.0, stats) == pytest.approx(0.0)
    assert z_score(34.0, stats) == pytest.approx(-1.5)


def test_a_history_with_no_spread_is_infinite_not_undefined() -> None:
    """The case the simulated log actually produces, given its own arithmetic.

    A road that has only ever taken 10.0s and reports 29.0s is not a finite number
    of standard deviations out: there is no standard deviation for it to be out
    by. Zero over zero is answered, not guarded against.
    """
    stats = flat(12, mean=10.0)

    assert z_score(10.0, stats) == 0.0
    assert z_score(29.0, stats) == math.inf
    assert z_score(4.0, stats) == -math.inf


def test_an_observation_three_sigma_out_is_flagged(client, log_store) -> None:
    seed_history(log_store, ARTERIAL, SPREAD_VALUES)
    created = create_scenario(client)
    stats = stats_for(*ARTERIAL, NORMAL, SPREAD_VALUES)

    body = observe(client, created["scenario_id"], stats.mean + 3 * stats.std_dev)

    assert body["flagged"] is True
    assert body["rule"] == Z_SCORE
    assert body["z_score"] == pytest.approx(3.0)
    assert body["sample_count"] == 15
    assert body["threshold"] == Z_THRESHOLD


def test_an_observation_inside_the_threshold_is_not_flagged(client, log_store) -> None:
    seed_history(log_store, ARTERIAL, SPREAD_VALUES)
    created = create_scenario(client)
    stats = stats_for(*ARTERIAL, NORMAL, SPREAD_VALUES)

    body = observe(client, created["scenario_id"], stats.mean + 1.5 * stats.std_dev)

    assert body["flagged"] is False
    assert body["z_score"] == pytest.approx(1.5)
    assert body["rule"] == Z_SCORE


def test_the_threshold_is_strict() -> None:
    """Exactly two standard deviations out is *not* over it.

    Asserted on a history handed to :func:`detect` directly, because an
    observation of ``mean + 2 sigma`` computed from a stored float lands a hair
    off the exact value and could not decide whether the rule or the rounding was
    being tested.
    """
    stats = flat(12, mean=10.0, std_dev=2.0)

    assert detect(14.0, stats=stats, expected=10.0, condition=NORMAL).flagged is False
    assert detect(14.2, stats=stats, expected=10.0, condition=NORMAL).flagged is True


def test_a_road_reading_faster_than_usual_is_not_flagged(client, log_store) -> None:
    """One-sided on purpose: being early is not a reason to re-route."""
    seed_history(log_store, ARTERIAL, SPREAD_VALUES)
    created = create_scenario(client)
    stats = stats_for(*ARTERIAL, NORMAL, SPREAD_VALUES)

    body = observe(client, created["scenario_id"], stats.mean - 3 * stats.std_dev)

    assert body["flagged"] is False
    assert body["z_score"] == pytest.approx(-3.0)


def test_the_zero_spread_case_through_the_api(client, log_store) -> None:
    """The real-log case end to end: matching is fine, differing is anomalous.

    The differing verdict reports ``z_score`` of ``None`` rather than the
    infinity the arithmetic produces. JSON cannot carry one — Starlette renders
    responses with ``allow_nan=False`` and would raise — so the number is left to
    ``reason``, and ``std_dev`` of 0 is what says why it is not there.
    """
    seed_history(log_store, ARTERIAL, [10.0] * 12)
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    matching = observe(client, scenario_id, 10.0)
    assert matching["flagged"] is False
    assert matching["z_score"] == 0.0
    assert matching["std_dev"] == 0.0
    assert "no spread" in matching["reason"]

    differing = observe(client, scenario_id, 13.0)
    assert differing["flagged"] is True
    assert differing["rule"] == Z_SCORE
    assert differing["z_score"] is None
    assert differing["std_dev"] == 0.0
    assert "any difference is anomalous" in differing["reason"]


def test_a_reading_is_judged_in_its_own_condition_band(client, log_store) -> None:
    """Peak history does not inform a normal-band reading.

    The whole point of keying by condition: a pooled figure would score an
    ordinary 09:00 reading as a wild outlier against a mostly-02:00 history.
    """
    seed_history(log_store, ARTERIAL, [29.0] * 12, condition=PEAK, timestamp=PEAK_TIME)
    created = create_scenario(client, timestamp=NORMAL_TIME)

    body = observe(client, created["scenario_id"], 29.0)

    assert body["condition"] == NORMAL
    # No normal-band history at all, so the fallback ran — not a z-score against
    # the peak band's twelve samples.
    assert body["rule"] == INSUFFICIENT_SAMPLES
    assert body["sample_count"] == 0


def test_an_observation_may_name_its_own_band(client, log_store) -> None:
    seed_history(log_store, ARTERIAL, [29.0] * 12, condition=PEAK, timestamp=PEAK_TIME)
    created = create_scenario(client, timestamp=NORMAL_TIME)

    body = observe(client, created["scenario_id"], 29.0, timestamp=PEAK_TIME)

    assert body["condition"] == PEAK
    assert body["rule"] == Z_SCORE
    assert body["sample_count"] == 12


# --------------------------------------------------------------------------- #
# The fallback
# --------------------------------------------------------------------------- #
def test_below_the_minimum_the_score_is_not_taken_at_all() -> None:
    """Skipped, not computed-and-ignored: a zero would claim a measurement."""
    verdict = detect(
        40.0,
        stats=flat(MIN_SAMPLES - 1, mean=10.0, std_dev=1.0),
        expected=10.0,
        condition=NORMAL,
    )

    assert verdict.rule == INSUFFICIENT_SAMPLES
    assert verdict.z_score is None
    assert verdict.mean is None
    assert verdict.std_dev is None


def test_the_sample_boundary_is_exact() -> None:
    """Nine samples falls back, ten computes a score — asserted from both sides."""
    nine = flat(MIN_SAMPLES - 1, mean=10.0, std_dev=1.0)
    ten = flat(MIN_SAMPLES, mean=10.0, std_dev=1.0)

    assert detect(20.0, stats=nine, expected=10.0, condition=NORMAL).rule == (
        INSUFFICIENT_SAMPLES
    )
    assert detect(20.0, stats=ten, expected=10.0, condition=NORMAL).rule == Z_SCORE


def test_no_history_at_all_falls_back() -> None:
    verdict = detect(20.0, stats=None, expected=10.0, condition=NORMAL)

    assert verdict.rule == INSUFFICIENT_SAMPLES
    assert verdict.sample_count == 0
    assert verdict.z_score is None


def test_the_fallback_margin_is_strict() -> None:
    """``observed > 1.2 x expected``: exactly 1.2x is not over it."""
    stats = flat(0, mean=0.0)

    on_it = detect(12.0, stats=stats, expected=10.0, condition=NORMAL)
    over_it = detect(12.1, stats=stats, expected=10.0, condition=NORMAL)

    assert on_it.flagged is False
    assert on_it.expected == pytest.approx(10.0)
    assert over_it.flagged is True
    assert FALLBACK_FACTOR == 1.2


def test_the_fallback_flags_through_the_api(client, log_store) -> None:
    """A road with four samples: too few for a score, so the margin decides."""
    seed_history(log_store, ARTERIAL, [10.0] * 4)
    created = create_scenario(client)

    body = observe(client, created["scenario_id"], 20.0)

    assert body["rule"] == INSUFFICIENT_SAMPLES
    assert body["sample_count"] == 4
    assert body["flagged"] is True
    assert body["expected_travel_time"] == pytest.approx(ARTERIAL_S)
    assert body["z_score"] is None
    assert body["mean"] is None
    assert body["threshold"] == FALLBACK_FACTOR


def test_the_fallback_uses_the_roads_own_modelled_time(client) -> None:
    """Not the scenario's, and not a constant: the side street's own 11s.

    The margin is 1.2x, so the side street's ceiling is 13.2s where the arterial's
    would be 12.0s. A 13.0s reading therefore sits under one and over the other,
    which is what makes this test about *whose* expectation was used rather than
    about the arithmetic.
    """
    created = create_scenario(client)

    under = observe(client, created["scenario_id"], 13.0, edge=SIDE_STREET)
    over = observe(client, created["scenario_id"], 14.0, edge=SIDE_STREET)

    assert under["expected_travel_time"] == pytest.approx(SIDE_STREET_S)
    assert under["flagged"] is False
    assert over["expected_travel_time"] == pytest.approx(SIDE_STREET_S)
    assert over["flagged"] is True


def test_an_impassable_road_has_nothing_to_compare_against() -> None:
    """A closed road is a legitimate state, not an error to raise.

    Reachable only through :func:`detect`; the test below it shows why the API
    cannot produce this.
    """
    verdict = detect(99.0, stats=None, expected=None, condition=NORMAL)

    assert verdict.flagged is False
    assert verdict.expected is None
    assert "impassable" in verdict.reason


# --------------------------------------------------------------------------- #
# Expected is incident-free
# --------------------------------------------------------------------------- #
def test_a_live_incident_does_not_raise_the_expectation(client) -> None:
    """Otherwise the fallback would be blind to the change it exists to catch.

    A slow report makes the road *actually* 29s. If the expectation followed it,
    a 29s reading would look perfectly ordinary and nothing would ever be flagged
    on a road an operator had already touched.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    before = observe(client, scenario_id, 20.0)
    response = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": SLOW, "edge": {"u": 0, "v": 1}},
    )
    assert response.status_code == 201, response.text
    after = observe(client, scenario_id, 20.0)

    assert before["expected_travel_time"] == pytest.approx(ARTERIAL_S)
    assert after["expected_travel_time"] == pytest.approx(ARTERIAL_S)
    assert after["flagged"] is True


def test_a_live_closure_leaves_the_expectation_passable(client) -> None:
    """The same rule from the other side: a shut road still has an ordinary time.

    The expectation is computed with manual conditions cleared, so a closure
    reported against this road cannot make it *impassable* for the purposes of
    judging a reading. It is shut on the scenario's costs and still has a normal
    to be measured against.
    """
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    response = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": CLOSURE, "edge": {"u": 0, "v": 1}},
    )
    assert response.status_code == 201, response.text

    body = observe(client, scenario_id, 20.0)

    assert body["expected_travel_time"] == pytest.approx(ARTERIAL_S)


# --------------------------------------------------------------------------- #
# When the baseline is taken
# --------------------------------------------------------------------------- #
def test_the_baseline_is_taken_once_at_scenario_start(client, log_store) -> None:
    """History added after creation is invisible — the documented simplification.

    A production system would refresh this on a daily batch. This one takes one
    reading per scenario, so two observations on one scenario are judged against
    one history rather than against whatever the log happened to hold that second.
    """
    seed_history(log_store, ARTERIAL, [10.0] * 12)
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    assert observe(client, scenario_id, 10.0)["sample_count"] == 12

    seed_history(log_store, ARTERIAL, [11.0] * 5)

    assert observe(client, scenario_id, 10.0)["sample_count"] == 12


def test_a_scenario_does_not_contribute_to_its_own_baseline(
    client, store, log_store
) -> None:
    """Its own priced rows are written after the snapshot, so they are not history.

    Read the other way round, a scenario would be judged against a history
    containing the very reading it is supposed to be an exception to.
    """
    created = create_scenario(client)

    record = store.get(created["scenario_id"])
    assert record.baseline is not None
    # Pricing wrote rows for this scenario, and none of them reached its baseline.
    assert log_store.count() > 0
    assert record.baseline.stats == {}


def test_the_same_observation_gets_the_same_verdict(client, log_store) -> None:
    """Determinism, which is what a snapshot baseline buys."""
    seed_history(log_store, ARTERIAL, SPREAD_VALUES)
    created = create_scenario(client)
    scenario_id = created["scenario_id"]

    assert observe(client, scenario_id, 52.0) == observe(client, scenario_id, 52.0)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def test_an_unknown_scenario_is_404(client) -> None:
    response = client.post(
        "/scenarios/nope/detect",
        json={"edge": {"u": 0, "v": 1}, "travel_time": 10.0},
    )

    assert response.status_code == 404


def test_an_observation_edge_must_exist_in_the_road_graph(client) -> None:
    created = create_scenario(client)

    response = client.post(
        f"/scenarios/{created['scenario_id']}/detect",
        json={"edge": {"u": 0, "v": 99}, "travel_time": 10.0},
    )

    assert response.status_code == 422
    assert "no road from 0 to 99" in response.json()["detail"]


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_an_observation_must_be_a_positive_time(client, value) -> None:
    """A zero or negative travel time is not an observation of anything.

    The echoed input is asserted as well: the app rewrites every validation
    error on its way out, so a finite value has to survive that pass unchanged.
    """
    created = create_scenario(client)

    response = client.post(
        f"/scenarios/{created['scenario_id']}/detect",
        json={"edge": {"u": 0, "v": 1}, "travel_time": value},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["input"] == value


def test_an_infinite_observation_is_rejected(client) -> None:
    """``gt=0`` does not exclude an infinity, so the schema has to.

    Sent as a raw body rather than through ``json=``, because httpx refuses to
    *encode* an infinity — it calls ``json.dumps`` with ``allow_nan=False`` and
    raises in the client, so the request would never reach the API and the test
    would prove nothing about it.

    ``1e400`` is the case that matters: a **valid** JSON number literal that
    overflows to infinity when parsed, which any conforming client can send. The
    bare ``Infinity`` token is not legal JSON at all, but Python's parser accepts
    it, so it is refused for the same reason.

    Refusing it turned out to be only half the job. Pydantic echoes the offending
    value back in its error, and Starlette writes every response with
    ``allow_nan=False`` — so the 422 raised *because* the value was infinite
    could not itself be serialised, and the client got a 500 from the serializer
    rather than the 422 the validation produced. ``_json_safe`` nulls the echoed
    input; the assertions on the body are what hold that in place.
    """
    created = create_scenario(client)
    probe = '{"edge": {"u": 0, "v": 1}, "travel_time": %s}'

    for travel_time in ("1e400", "Infinity"):
        response = client.post(
            f"/scenarios/{created['scenario_id']}/detect",
            content=probe % travel_time,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422, travel_time

        # A rejection that renders at all is the point of the paragraph above.
        [error] = response.json()["detail"]
        assert error["type"] == "finite_number", travel_time
        assert error["input"] is None, travel_time


# --------------------------------------------------------------------------- #
# The boundary with re-optimization
# --------------------------------------------------------------------------- #
#: Every field the verdict may carry. Asserted as an exact set, because the
#: absence of a route, a cost or a solver from it is the structural evidence that
#: this route does not re-optimize.
VERDICT_FIELDS = frozenset(
    {
        "scenario_id",
        "edge",
        "observed_travel_time",
        "condition",
        "rule",
        "flagged",
        "expected_travel_time",
        "sample_count",
        "mean",
        "std_dev",
        "z_score",
        "threshold",
        "reason",
    }
)


def test_detection_does_not_re_optimize(client, log_store) -> None:
    """A flag is a signal, not an action.

    Re-optimizing on an anomaly is the ``reopt`` module's job, in a later phase.
    """
    seed_history(log_store, ARTERIAL, [10.0] * 12)
    created = create_scenario(client)

    body = observe(client, created["scenario_id"], 50.0)

    assert body["flagged"] is True
    assert set(body) == VERDICT_FIELDS


def test_detection_writes_nothing_to_the_log(client, log_store) -> None:
    """Persisting the reading would feed the anomaly back into its own baseline."""
    seed_history(log_store, ARTERIAL, [10.0] * 12)
    created = create_scenario(client)
    before = log_store.count()

    observe(client, created["scenario_id"], 50.0)
    observe(client, created["scenario_id"], 50.0)

    assert log_store.count() == before
