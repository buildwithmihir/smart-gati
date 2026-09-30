"""The run history: what gets written, and what deliberately does not.

Two kinds of test, split the way the rest of the suite splits them.

The **store** tests build rows by hand and check the table against itself —
round-trip, paging, filters, and the aggregate arithmetic. Hand-built rows are
the only way to be exact about averages: a test that drove four solves and then
averaged their runtimes would be asserting the solver's timing rather than the
store's summing.

The **API** tests drive the real endpoints over the same tour graph the other
fleet suites use, and assert the two things that matter and hold wherever the
fleet happens to be: that each solve path writes exactly one row of the right
kind, and that the row's numbers are the ones the response reported. The second
is the receipt check this table exists to make possible — a history row that
could disagree with what the caller was told would be worse than no row.

The negative case is here too, and it is not an afterthought: a **refused**
re-optimization must write nothing, because that is the difference between a
history of runs and a log of requests.
"""

from __future__ import annotations

from typing import get_args

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from qgati.analytics import KINDS, RunLogRow, RunLogStore
from qgati.api.main import (
    create_app,
    get_graph,
    get_log_store,
    get_run_store,
    get_store,
    get_watchers,
)
from qgati.api.schemas import RunEntry
from qgati.api.store import ScenarioStore
from qgati.fleet import WatcherRegistry
from qgati.traffic import TrafficLogStore

LON_ORIGIN, LAT_ORIGIN, DEGREE_SPAN = 77.20, 28.60, 0.05
TOY_SPEED_KPH = 30.0

#: 02:00 prices as ``normal``, so every modelled road costs its own base time.
NORMAL_TIME = "2026-09-21T02:00:00+05:30"

#: A long interval with a tiny scale: the background thread stays effectively
#: asleep while one explicit tick still moves the fleet. Same device as the other
#: fleet suites, and for the same reason.
IDLE_INTERVAL = 300.0
ONE_SECOND = {"interval_seconds": IDLE_INTERVAL, "time_scale": 1.0 / IDLE_INTERVAL}

#: The seed ``test_reopt`` pins as the one whose ordinary readings the detector
#: does not flag. Starting the fleet on it is what makes "nothing warrants a
#: re-plan" a reliable 409 rather than a coin flip on the noise draw.
QUIET_SEED = 3

DETERMINISTIC = {"solver": "savings"}


def _length_of(travel_time: float) -> float:
    return travel_time * (TOY_SPEED_KPH / 3.6)


# --------------------------------------------------------------------------- #
# Store: the table, on its own
# --------------------------------------------------------------------------- #
def row(**overrides) -> RunLogRow:
    """A plausible ``optimize`` row, with anything a test cares about overridden."""
    fields = {
        "timestamp": "2026-09-30T10:00:00+00:00",
        "kind": "optimize",
        "scenario_id": "s1",
        "n_deliveries": 12,
        "n_vehicles": 3,
        "solver": "qpso",
        "solver_name": "QPSO",
        "cost": 1000.0,
        "travel_cost": 900.0,
        "travel_time": 600.0,
        "distance_m": 5000.0,
        "fuel_litres": 1.2,
        "feasible": True,
        "runtime_ms": 250.0,
    }
    fields.update(overrides)
    return RunLogRow(**fields)


@pytest.fixture
def store() -> RunLogStore:
    return RunLogStore(":memory:")


def test_a_row_comes_back_the_way_it_went_in(store: RunLogStore) -> None:
    """Round-trip, including the two columns that are not strings or floats.

    ``feasible`` and ``id`` are the ones at risk: SQLite has no boolean type, so
    ``feasible`` leaves as a Python bool, is stored as 0 or 1, and has to be a bool
    again on the way out, and ``id`` is assigned by the database rather than by the
    caller.
    """
    written = store.write(
        row(feasible=False, moved=3, trigger="incident", old_eta_seconds=700.0)
    )
    assert written > 0

    rows, total = store.read()
    assert total == 1
    assert len(rows) == 1

    got = rows[0]
    assert got.id == written
    assert got.feasible is False
    assert got.moved == 3
    assert got.trigger == "incident"
    assert got.old_eta_seconds == 700.0
    assert got.scenario_id == "s1"
    assert got.solver_name == "QPSO"


def test_the_eta_pair_is_derived_and_absent_when_there_was_nothing_to_beat(
    store: RunLogStore,
) -> None:
    """``new_eta_seconds`` is ``travel_time``, and the pair is null without a baseline.

    This is the whole reason ``new_eta_seconds`` is not a column: for a re-plan it
    *is* the row's own travel time, and storing the same number twice is an
    invitation for the two to drift. An ``optimize`` has no earlier plan, so both
    derived fields are ``None`` — not ``0.0``, which would say the previous plan
    was instantaneous.
    """
    store.write(row(kind="optimize", travel_time=600.0))
    store.write(row(kind="reoptimize", travel_time=450.0, old_eta_seconds=700.0))

    newest, _ = store.read(limit=1)
    assert newest[0].kind == "reoptimize"
    assert newest[0].new_eta_seconds == 450.0
    assert newest[0].eta_saved_seconds == 250.0

    oldest, _ = store.read(limit=1, offset=1)
    assert oldest[0].kind == "optimize"
    assert oldest[0].old_eta_seconds is None
    assert oldest[0].new_eta_seconds is None
    assert oldest[0].eta_saved_seconds is None


def test_a_replan_that_came_out_worse_reports_a_negative_saving(
    store: RunLogStore,
) -> None:
    """The saving is signed, and a loss is not clamped to zero.

    A re-solve over the remaining stops can land on a worse arrangement than the
    one the fleet is already driving — it is a heuristic on a small instance, not a
    guarantee. Reporting that as "no saving" would hide the one case an operator
    most needs to see.
    """
    store.write(row(kind="reoptimize", travel_time=800.0, old_eta_seconds=700.0))
    assert store.read()[0][0].eta_saved_seconds == pytest.approx(-100.0)


def test_rows_come_back_newest_first_and_page_without_losing_the_total(
    store: RunLogStore,
) -> None:
    """Paging is by insertion, and ``total`` describes the filter rather than the page.

    Ordering by ``timestamp`` rather than by ``id`` would put these in the wrong
    order on purpose: the newest *insert* carries the oldest *stamp*, which is what
    a table written across a clock adjustment looks like.
    """
    store.write(row(timestamp="2026-09-30T09:00:00+00:00", scenario_id="old"))
    store.write(row(timestamp="2026-09-30T08:00:00+00:00", scenario_id="newest"))

    first, total = store.read(limit=1)
    assert total == 2, "the total counts the filter, not the page"
    assert len(first) == 1
    assert first[0].scenario_id == "newest", "ordered by insert, not by timestamp"

    second, _ = store.read(limit=1, offset=1)
    assert second[0].scenario_id == "old"
    assert store.read(limit=1, offset=2)[0] == [], "past the end is empty, not an error"


def test_filters_narrow_the_rows_and_the_total_together(store: RunLogStore) -> None:
    """Every filter is optional, and an applied one narrows the count with the page."""
    store.write(row(kind="optimize", solver="qpso", scenario_id="a"))
    store.write(row(kind="reoptimize", solver="savings", scenario_id="a"))
    store.write(row(kind="dispatch", solver="qpso", scenario_id="b"))

    assert store.read(kind="optimize")[1] == 1
    assert store.read(solver="qpso")[1] == 2
    assert store.read(scenario_id="a")[1] == 2
    assert store.read(scenario_id="a", solver="savings")[1] == 1
    assert store.read(kind="nothing-writes-this")[1] == 0


def test_an_empty_history_has_no_averages_rather_than_zero_ones(
    store: RunLogStore,
) -> None:
    """The distinction the API is built around: no data is not a measurement of zero.

    Every average is ``None`` and every count is ``0``. Returning ``0.0`` for the
    average runtime would be a claim about performance that nothing supports, which
    is the same reason the dashboard reports ``feasible`` as null for a plan it
    cannot score.
    """
    summary = store.summary()

    assert summary.total_runs == 0
    assert summary.first_run_at is None
    assert summary.last_run_at is None
    assert summary.runs_by_kind == {}
    assert summary.runs_by_solver == {}
    assert summary.avg_runtime_ms is None
    assert summary.min_runtime_ms is None
    assert summary.max_runtime_ms is None
    assert summary.avg_cost is None
    assert summary.avg_travel_cost is None
    assert summary.avg_travel_time_seconds is None
    assert summary.eta_runs == 0
    assert summary.avg_saved_seconds is None
    assert summary.total_saved_seconds is None
    # Counts, unlike averages, are genuinely zero.
    assert summary.feasible_runs == 0
    assert summary.infeasible_runs == 0
    assert summary.incident_triggered_runs == 0
    assert summary.improved_runs == 0


def test_the_summary_adds_up_over_a_known_set(store: RunLogStore) -> None:
    """Averages, extremes and the ETA split, against numbers computed by hand.

    Three rows: two re-plans that had a baseline — one ahead, one behind — and one
    dispatch that did not. The dispatch must stay out of the ETA figures entirely,
    or its absent baseline would be averaged in as a zero saving and understate
    both real ones.
    """
    store.write(
        row(
            kind="dispatch",
            solver="qpso",
            runtime_ms=100.0,
            cost=100.0,
            travel_cost=90.0,
            travel_time=60.0,
            feasible=True,
        )
    )
    store.write(
        row(
            kind="reoptimize",
            solver="qpso",
            runtime_ms=300.0,
            cost=200.0,
            travel_cost=180.0,
            travel_time=400.0,
            old_eta_seconds=500.0,
            trigger="incident",
            feasible=True,
        )
    )
    store.write(
        row(
            kind="avoid_road",
            solver="savings",
            runtime_ms=200.0,
            cost=300.0,
            travel_cost=270.0,
            travel_time=700.0,
            old_eta_seconds=600.0,
            trigger="override",
            feasible=False,
        )
    )

    summary = store.summary()

    assert summary.total_runs == 3
    assert summary.runs_by_kind == {"dispatch": 1, "reoptimize": 1, "avoid_road": 1}
    assert summary.runs_by_solver == {"qpso": 2, "savings": 1}
    assert summary.feasible_runs == 2
    assert summary.infeasible_runs == 1
    # One, not two. The avoid-road row carries a trigger too — "override" — and it
    # is deliberately *not* counted here: a driver reporting a road is not the
    # system reacting to an incident or to a reading the detector flagged, and
    # counting it would overstate how much of the history was the world pushing
    # back rather than a person asking.
    assert summary.incident_triggered_runs == 1

    assert summary.avg_runtime_ms == pytest.approx(200.0)
    assert summary.min_runtime_ms == pytest.approx(100.0)
    assert summary.max_runtime_ms == pytest.approx(300.0)
    assert summary.avg_cost == pytest.approx(200.0)
    assert summary.avg_travel_cost == pytest.approx(180.0)
    assert summary.avg_travel_time_seconds == pytest.approx(1160.0 / 3.0)

    assert summary.first_run_at == "2026-09-30T10:00:00+00:00"
    assert summary.last_run_at == "2026-09-30T10:00:00+00:00"

    # Only the two baselined rows: +100 saved and -100 lost.
    assert summary.eta_runs == 2
    assert summary.avg_saved_seconds == pytest.approx(0.0)
    assert summary.total_saved_seconds == pytest.approx(0.0)
    assert summary.improved_runs == 1
    assert summary.worsened_runs == 1
    assert summary.unchanged_runs == 0


# --------------------------------------------------------------------------- #
# API: the endpoints write, and the readers read
# --------------------------------------------------------------------------- #
def build_tour_graph() -> nx.DiGraph:
    """A corridor depot -> 1 -> 2 -> depot, one edge per leg, ten seconds each.

    The fleet graph the other suites use. A vehicle dispatched onto it starts on a
    real road, which is what ``avoid-road`` and an incident both need.
    """
    graph = nx.DiGraph()
    for node in (0, 1, 2):
        graph.add_node(node, x=LON_ORIGIN + node * DEGREE_SPAN, y=LAT_ORIGIN)
    for u, v in ((0, 1), (1, 2), (2, 0)):
        graph.add_edge(
            u, v, highway="secondary", travel_time=10.0, weight=10.0,
            length=_length_of(10.0),
        )
    return graph


@pytest.fixture
def run_store() -> RunLogStore:
    """The store the app writes to, kept in the test so it can be read directly."""
    return RunLogStore(":memory:")


@pytest.fixture
def client(run_store: RunLogStore) -> TestClient:
    """An app over the tour graph, with a fleet registry and a history of its own.

    The run store override is not decoration. Without it these endpoints reach the
    process-wide store and write into the developer's real ``run_history.db`` — the
    exact thing ``SMART_GATI_RUN_DB``'s comment says the suite is arranged to avoid.
    """
    application = create_app()
    graph = build_tour_graph()
    application.dependency_overrides[get_graph] = lambda: graph
    application.dependency_overrides[get_store] = lambda: ScenarioStore()
    application.dependency_overrides[get_log_store] = lambda: TrafficLogStore(":memory:")
    application.dependency_overrides[get_run_store] = lambda: run_store
    application.dependency_overrides[get_watchers] = lambda: WatcherRegistry()
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


def create_scenario(client: TestClient) -> str:
    """Two stops on the corridor, one vehicle that can carry both."""
    response = client.post(
        "/scenarios",
        json={
            "kind": "explicit",
            "depot": {"node": 0},
            "deliveries": [
                {"id": "D0", "node": 1, "demand": 1},
                {"id": "D1", "node": 2, "demand": 1},
            ],
            "vehicles": [{"id": "V0", "capacity": 2}],
            "conditions": {"timestamp": NORMAL_TIME},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["scenario_id"]


def start_fleet(client: TestClient, scenario_id: str, **body) -> dict:
    response = client.post(
        f"/scenarios/{scenario_id}/watcher",
        json={**DETERMINISTIC, "seed": QUIET_SEED, **ONE_SECOND, **body},
    )
    assert response.status_code == 201, response.text
    return response.json()


def first_edge(client: TestClient, scenario_id: str) -> dict:
    vehicles = client.get(f"/scenarios/{scenario_id}/watcher").json()["vehicles"]
    edge = vehicles[0]["edge"]
    assert edge is not None, "a dispatched vehicle starts on a road"
    return {"u": edge["u"], "v": edge["v"]}


def runs(client: TestClient, **params) -> dict:
    response = client.get("/analytics/runs", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_an_optimize_writes_one_row_carrying_the_numbers_it_reported(
    client: TestClient, run_store: RunLogStore
) -> None:
    """The receipt check: every figure on the row is one the response reported.

    Nothing here is compared against a hand-computed value, because a test that
    recomputed the cost would be asserting its own arithmetic. What is asserted is
    that the row and the response agree — which is the property that makes the
    history worth reading.
    """
    scenario_id = create_scenario(client)

    solved = client.post(f"/optimize/{scenario_id}", json={"seed": 0, "iterations": 5})
    assert solved.status_code == 200, solved.text
    body = solved.json()

    rows, total = run_store.read()
    assert total == 1, "one solve, one row"
    entry = rows[0]

    assert entry.kind == "optimize"
    assert entry.scenario_id == scenario_id
    assert entry.solver == body["solver"]
    assert entry.solver_name == body["solver_name"]
    assert entry.n_deliveries == 2
    assert entry.n_vehicles == 1
    assert entry.cost == pytest.approx(body["cost"])
    assert entry.travel_cost == pytest.approx(body["travel_cost"])
    assert entry.travel_time == pytest.approx(body["travel_time"])
    assert entry.distance_m == pytest.approx(body["distance_m"])
    assert entry.fuel_litres == pytest.approx(body["fuel_litres"])
    assert entry.feasible == body["feasible"]
    assert entry.runtime_ms == pytest.approx(body["runtime_ms"])
    # No earlier plan, so no baseline — and a dispatch/optimize is never forced to
    # invent one.
    assert entry.old_eta_seconds is None
    assert entry.trigger is None


def test_dispatching_a_fleet_is_a_run_and_is_recorded_as_one(
    client: TestClient, run_store: RunLogStore
) -> None:
    """``POST /watcher`` solves, so it writes a row — with a runtime that is not zero.

    This is the row the History tab opens onto after a sample load, and it is the
    reason the watcher handler had to start timing its solve: an untimed dispatch
    would put a zero in the milliseconds column for the run a user is most likely
    to be looking at.
    """
    scenario_id = create_scenario(client)
    started = start_fleet(client, scenario_id, include_geometry=False)

    rows, total = run_store.read()
    assert total == 1
    entry = rows[0]

    assert entry.kind == "dispatch"
    assert entry.solver == started["solver"]
    assert entry.travel_time == pytest.approx(
        sum(route["travel_time"] for route in started["routes"])
    )
    assert entry.runtime_ms > 0.0, "the dispatch's solve was actually timed"
    assert entry.old_eta_seconds is None


def test_a_refused_replan_writes_nothing(
    client: TestClient, run_store: RunLogStore
) -> None:
    """A 409 is not a run. The table records solves, not attempts.

    Nothing has happened to this scenario — no incident, no flagged reading — so the
    endpoint refuses before it reaches a solver, and there is nothing to record. The
    assertion is on the *whole* store rather than on a filtered read, because the
    claim is that no row of any kind appeared.
    """
    scenario_id = create_scenario(client)
    start_fleet(client, scenario_id, include_geometry=False)
    assert run_store.count() == 1, "the dispatch is the only row so far"

    refused = client.post(f"/scenarios/{scenario_id}/reoptimize", json={})
    assert refused.status_code == 409, refused.text

    assert run_store.count() == 1, "the refusal added nothing"


def test_an_incident_triggered_replan_records_the_eta_pair_it_computed(
    client: TestClient, run_store: RunLogStore
) -> None:
    """The row's old/new ETA is the before/after the response reports, and it is real.

    ``old_eta_seconds`` is checked against the sum of the ``before`` routes'
    travel times — the number a reader can see on the panel — rather than against a
    recomputation. That is what makes it a receipt: the same figure, reached from
    the response rather than from the store's own side of the wall.
    """
    scenario_id = create_scenario(client)
    start_fleet(client, scenario_id, include_geometry=False)

    reported = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": "slow", "edge": first_edge(client, scenario_id)},
    )
    assert reported.status_code == 201, reported.text

    replanned = client.post(f"/scenarios/{scenario_id}/reoptimize", json={})
    assert replanned.status_code == 200, replanned.text
    body = replanned.json()

    rows, total = run_store.read()
    assert total == 2, "the dispatch and the re-plan"
    entry = rows[0]

    assert entry.kind == "reoptimize"
    assert entry.trigger == "incident"
    assert entry.trigger_detail == body["trigger"]["detail"]
    assert entry.old_eta_seconds == pytest.approx(
        sum(route["travel_time"] for route in body["before"])
    )
    assert entry.new_eta_seconds == pytest.approx(body["travel_time"])
    assert entry.eta_saved_seconds == pytest.approx(
        entry.old_eta_seconds - entry.new_eta_seconds
    )
    assert entry.moved == len(body["moved"])


def test_an_avoid_road_run_is_recorded_against_the_vehicle_that_reported(
    client: TestClient, run_store: RunLogStore
) -> None:
    """The driver override is a re-plan, so it is a run — with its reporter named.

    Its ETA pair has to come from the **fleet-sized** evaluation on both sides. The
    one-vehicle instance the endpoint actually solves would compare a fleet's
    remaining travel time against a single vehicle's, which is a saving made of
    scope rather than of routing.
    """
    scenario_id = create_scenario(client)
    start_fleet(client, scenario_id, include_geometry=False)

    response = client.post(
        f"/scenarios/{scenario_id}/vehicles/V0/avoid-road",
        json={"edge": first_edge(client, scenario_id), "treatment": "slow"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    rows, total = run_store.read()
    assert total == 2
    entry = rows[0]

    assert entry.kind == "avoid_road"
    assert entry.affected_vehicle == "V0"
    assert entry.trigger == "override"
    assert entry.old_eta_seconds == pytest.approx(
        sum(route["travel_time"] for route in body["before"])
    )
    assert entry.new_eta_seconds == pytest.approx(
        sum(route["travel_time"] for route in body["after"])
    )


def test_the_two_readers_agree_with_what_was_written(
    client: TestClient, run_store: RunLogStore
) -> None:
    """End to end: the endpoints page the rows and total the same set.

    The summary is taken over exactly the rows the list returns, so a count that
    disagreed between them would be one of the two lying. ``has_more`` is the
    response's own field rather than arithmetic here, which is the point — it is
    what a client pages on.
    """
    scenario_id = create_scenario(client)
    client.post(f"/optimize/{scenario_id}", json={"seed": 0, "iterations": 5})
    start_fleet(client, scenario_id, include_geometry=False)
    assert run_store.count() == 2

    page = runs(client, limit=1)
    assert page["total"] == 2
    assert page["has_more"] is True
    assert len(page["items"]) == 1
    assert page["items"][0]["kind"] == "dispatch", "newest first"

    tail = runs(client, limit=1, offset=1)
    assert tail["total"] == 2
    assert tail["has_more"] is False
    assert tail["items"][0]["kind"] == "optimize"

    filtered = runs(client, kind="optimize")
    assert filtered["total"] == 1
    assert filtered["items"][0]["kind"] == "optimize"

    by_scenario = runs(client, scenario_id=scenario_id)
    assert by_scenario["total"] == 2
    assert runs(client, scenario_id="nobody")["total"] == 0

    summary = client.get("/analytics/summary")
    assert summary.status_code == 200, summary.text
    stats = summary.json()

    assert stats["total_runs"] == 2
    assert stats["runs_by_kind"] == {"optimize": 1, "dispatch": 1}
    # Two keys, not one: the dispatch pinned `savings` but the optimize sent no
    # solver at all, so it ran the production default. Asserting a single key here
    # would be asserting that `/optimize`'s default is `savings`, which it is not.
    assert stats["runs_by_solver"] == {"qpso": 1, "savings": 1}
    assert stats["first_run_at"] is not None
    assert stats["last_run_at"] is not None
    # Neither run was a re-plan, so there is nothing to average and the API says so
    # rather than reporting a saving of zero.
    assert stats["incident_triggered_runs"] == 0
    assert stats["eta"]["runs"] == 0
    assert stats["eta"]["avg_saved_seconds"] is None
    assert stats["eta"]["total_saved_seconds"] is None
    assert stats["avg_runtime_ms"] is not None


def test_the_store_and_the_schema_agree_on_what_a_kind_may_be() -> None:
    """The four kinds are spelled out twice, and only this connects them.

    ``KINDS`` is what the call sites write; ``RunEntry.kind`` is a ``Literal`` that
    response validation enforces. A fifth solve path added to one and not the other
    would be accepted by the store and then rejected on the way out — as a 500, on
    the one request that exercised it. Cheap to check, and silently expensive not to.
    """
    assert set(KINDS) == set(get_args(RunEntry.model_fields["kind"].annotation))
