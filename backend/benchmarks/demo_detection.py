#!/usr/bin/env python
"""Flag an anomalous travel time from a road's own logged history, over real HTTP.

This is the anomaly-detection demonstration, and it shows the three things that
were asked for, against the real application on the real Delhi graph:

1. **Mean and standard deviation per road**, computed from logged rows.
2. **A z-score on a new observation**, flagging at ``z > 2`` and not below it.
3. **The fallback when there is too little history** — under ten samples the
   z-score is skipped entirely and the simulator's expectation with a 1.2x margin
   decides instead.

It closes on the case that actually dominates a real log. An ordinary
``travel_time`` is ``base_travel_time x congestion_multiplier``, both
deterministic, so every sample for one ``(road, condition)`` key is the *same
number*, its standard deviation is exactly zero, and the z-score is a division by
zero that is not an error. Seeding a flat history reproduces that, and the run
prints what the detector does with it.

**This demo seeds synthetic history, so it runs against a throwaway log.** The
real ``backend/data/traffic_log.db`` is an accumulating dataset — a road's rows
there are its actual past, and writing invented samples into them would corrupt
that road's baseline for every later run. A temp file is created for the run and
removed with it, and the path is printed so there is no doubt which log was read.

The roads it uses are not invented either: they are taken from the cheapest paths
the solved scenario itself runs along, so each is provably a road this instance
was priced over.

Requests go through ``TestClient``, which drives the real FastAPI application
in-process — so there is no server to start, while routing, validation,
serialisation and the history lookup are all exercised exactly as a client would.

Usage
-----
    uv run python benchmarks/demo_detection.py
    uv run python benchmarks/demo_detection.py --deliveries 12 --seed 7
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import networkx as nx
from fastapi.testclient import TestClient

from qgati.api.main import create_app, get_graph, get_log_store, get_store
from qgati.traffic import (
    FALLBACK_FACTOR,
    MIN_SAMPLES,
    Z_THRESHOLD,
    ActiveConditions,
    TrafficLogRow,
    TrafficLogStore,
    TrafficState,
    build_baseline,
    congestion_state,
    edge_of,
    simulated_travel_time,
    traffic_weight_function,
)

#: Delhi's fixed offset. 02:00 is outside the 06:00-22:00 daytime band, so every
#: road prices at x1.0 and the numbers below are the graph's own base times —
#: which makes the modelled expectation and the seeded mean directly comparable.
WHEN = datetime(2026, 9, 21, 2, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

#: The two things this demo needs beside the log its own run creates: how many
#: samples to give the road that exercises the z-score, and how many to give the
#: one that exercises the fallback.
SPREAD_COUNT = 15

#: Fractions of the road's own modelled time to seed a spread from. They average
#: exactly 1.0, so the history's mean lands on the model's expectation and the
#: z-score below is measuring a departure from the road's own normal rather than
#: from a number this demo chose.
SPREAD_FRACTIONS = (0.90, 0.95, 1.00, 1.05, 1.10)

#: Well under :data:`MIN_SAMPLES`, so the second road takes the fallback.
SCARCE_COUNT = 4

#: Readings sent to the z-score road, as multiples of the standard deviation.
#: 1.5 is inside :data:`Z_THRESHOLD` and 3.0 is outside it.
INSIDE_SIGMAS = 1.5
OUTSIDE_SIGMAS = 3.0

#: How much slower the reading on the flat-history road is. Anything but an exact
#: match is anomalous when there is no spread, so the size is only there to make
#: the point concrete.
FLAT_DELTA = 0.30

RULE = "-" * 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deliveries", type=int, default=8)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=21, help="fixes the instance")
    return parser


# --------------------------------------------------------------------------- #
# The roads, taken from the solution the scenario was priced over
# --------------------------------------------------------------------------- #
def roads_for_demo(
    graph: nx.Graph, solution: dict, needed: int
) -> list[tuple]:
    """Distinct directed roads along the cheapest paths the solution uses.

    Walks the solved routes in order and collects the hops of each depot-to-stop
    cheapest path, under the scenario's *current* conditions — the same weight
    function the cost matrix was built with. Every road returned is therefore one
    this instance actually priced over, rather than a road picked at random and
    attached to a scenario it has nothing to do with.
    """
    record = get_store().get(solution["scenario_id"])
    weight = traffic_weight_function(graph, record.effective_traffic_state())
    depot = record.scenario.depot.node

    roads: list[tuple] = []
    seen: set[tuple] = set()
    for route in solution["routes"]:
        # Each route starts at the depot, so the chain restarts per vehicle.
        origin = depot
        for stop in route["stops"]:
            try:
                path = nx.shortest_path(graph, origin, stop["node"], weight=weight)
            except nx.NetworkXNoPath:
                continue
            origin = stop["node"]
            for hop in zip(path, path[1:]):
                if hop not in seen:
                    seen.add(hop)
                    roads.append(hop)
            if len(roads) >= needed:
                return roads[:needed]
    return roads


def modelled_seconds(graph: nx.Graph, u, v) -> float | None:
    """The road's cost under the clock alone — the detector's own expectation.

    Deliberately the same call the ``/detect`` route makes, with manual
    conditions cleared, so the numbers printed here are the numbers the verdict
    was decided against.
    """
    return simulated_travel_time(
        graph,
        edge_of(graph, u, v),
        TrafficState(timestamp=WHEN, conditions=ActiveConditions()),
    )


# --------------------------------------------------------------------------- #
# Seeding a road's past
# --------------------------------------------------------------------------- #
def seed(log_store: TrafficLogStore, edge: tuple, values, condition: str) -> int:
    """Append one row per value, as ``edge``'s history in that condition band."""
    u, v = edge
    return log_store.write(
        TrafficLogRow(
            road_u=u,
            road_v=v,
            timestamp=WHEN.isoformat(),
            day_of_week=WHEN.strftime("%A"),
            time_of_day=WHEN.strftime("%H:%M"),
            traffic_condition=condition,
            travel_time=value,
        )
        for value in values
    )


def read_back(log_store: TrafficLogStore, edge: tuple, condition: str):
    """The road's history as the detector would summarise it, from the log itself."""
    rows, total = log_store.read(limit=1000)
    collected = list(rows)
    while len(collected) < total:
        page, _ = log_store.read(limit=1000, offset=len(collected))
        if not page:
            break
        collected.extend(page)
    return build_baseline(collected).for_edge(*edge, condition)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def create_scenario(client: TestClient, args) -> dict:
    response = client.post(
        "/scenarios",
        json={
            "kind": "generate",
            "n_deliveries": args.deliveries,
            "n_vehicles": args.vehicles,
            "seed": args.seed,
            "conditions": {"timestamp": WHEN.isoformat()},
        },
    )
    response.raise_for_status()
    return response.json()


def solve(client: TestClient, scenario_id: str) -> dict:
    response = client.post(
        f"/optimize/{scenario_id}", json={"seed": 0, "include_geometry": False}
    )
    response.raise_for_status()
    return response.json()


def observe(client: TestClient, scenario_id: str, edge: tuple, seconds: float) -> dict:
    """POST one observed travel time and return the verdict."""
    response = client.post(
        f"/scenarios/{scenario_id}/detect",
        json={"edge": {"u": edge[0], "v": edge[1]}, "travel_time": seconds},
    )
    response.raise_for_status()
    return response.json()


def road(edge: tuple) -> str:
    return f"{edge[0]} -> {edge[1]}"


def print_verdict(label: str, seconds: float, body: dict) -> None:
    """One observation, and what the detector made of it.

    Two decimals, not one: the readings below sit a stated number of standard
    deviations from the mean and land exactly on the 1.2x margin, so rounding the
    column would print a number that no longer satisfies the arithmetic the label
    claims — ``6.66`` as ``6.7`` is not 1.5 standard deviations above ``6.0``.
    """
    state = "FLAGGED" if body["flagged"] else "not flagged"
    score = "—" if body["z_score"] is None else f"{body['z_score']:.2f}"
    print(f"  {label:<24} {seconds:>9.2f} s   {state:<12} z={score}")
    print(f"    rule {body['rule']}, threshold {body['threshold']}, "
          f"{body['sample_count']} sample(s)")
    print(f"    {body['reason']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="qgati-demo-detection-") as tmp:
        log_path = Path(tmp) / "traffic_log.db"
        log_store = TrafficLogStore(log_path)
        try:
            return run(args, log_store, log_path)
        finally:
            # Windows will not unlink an open SQLite file, and the temp directory
            # is about to be removed.
            log_store.close()


def run(args, log_store: TrafficLogStore, log_path: Path) -> int:
    application = create_app()
    application.dependency_overrides[get_log_store] = lambda: log_store
    condition = congestion_state(WHEN)

    print("Q-Gati — anomaly detection demo")
    print("=" * 78)
    print(f"log        {log_path}")
    print("           a throwaway seeded for this run; the collected log at")
    print("           backend/data/traffic_log.db is never written to or read")
    print(f"instance   {args.deliveries} deliveries, {args.vehicles} vehicles, "
          f"seed {args.seed}")
    print(f"priced at  {WHEN.isoformat()}  ({condition} band, x1.0 everywhere)")

    with TestClient(application) as client:
        # A scenario has to exist before anything can be solved, but its baseline
        # is snapshotted the moment it is created — so a scenario created *after*
        # the history is seeded is the only one that can be judged against it.
        # This first one is a probe: it is created, solved, and never observed.
        # Its only job is to name the roads this instance actually runs on.
        probe = create_scenario(client, args)
        solution = solve(client, probe["scenario_id"])

        found = roads_for_demo(get_graph(), solution, 3)
        if len(found) < 3:
            raise SystemExit(
                f"the solution's cheapest paths yielded only {len(found)} road(s); "
                "this demo needs three. Raise --deliveries."
            )
        scored, scarce, flat_road = found

        # The probe's pricing rows are scratch: they record the roads it touched,
        # which is not history any of the verdicts below should be measured
        # against. Dropped while the log is still this run's own temp file, so the
        # counts printed in step 1 are exactly the history that was seeded.
        log_store.clear()

        base = modelled_seconds(get_graph(), *scored)
        seed(
            log_store,
            scored,
            [round(base * fraction, 3) for fraction in SPREAD_FRACTIONS]
            * (SPREAD_COUNT // len(SPREAD_FRACTIONS)),
            condition,
        )
        scarce_base = modelled_seconds(get_graph(), *scarce)
        seed(log_store, scarce, [round(scarce_base, 3)] * SCARCE_COUNT, condition)
        flat_value = round(modelled_seconds(get_graph(), *flat_road), 3)
        seed(log_store, flat_road, [flat_value] * MIN_SAMPLES, condition)

        # Read back through the log, so what is printed is what was stored rather
        # than what was intended — and before the scenario below is created, so it
        # is reading exactly the rows that scenario's baseline will hold.
        scored_stats = read_back(log_store, scored, condition)
        scarce_stats = read_back(log_store, scarce, condition)
        flat_stats = read_back(log_store, flat_road, condition)

        # Created last, on the same seed as the probe, so it is the same instance
        # — this time with the history already in the log when the baseline is
        # taken. This is the scenario every verdict below is made against.
        created = create_scenario(client, args)
        scenario_id = created["scenario_id"]

        print(f"scenario   {scenario_id}")
        print()
        print("1. The baseline, snapshotted when the scenario was created")
        print(RULE)
        print(f"  {'road':<28} {'band':<8} {'samples':>7} {'mean':>9} {'std dev':>9}")
        for stats in (scored_stats, scarce_stats, flat_stats):
            print(f"  {road((stats.road_u, stats.road_v)):<28} {stats.condition:<8} "
                  f"{stats.count:>7} {stats.mean:>8.1f}s {stats.std_dev:>8.2f}s")
        print()
        print("  Every key is (road, condition), so a 09:00 reading is judged")
        print("  against that road's own 09:00 history. The cost of that is fewer")
        print("  samples per key, which is why a verdict reports its sample count.")
        print()
        print(f"  {road(scored)} has a spread because this demo seeded one. The")
        print(f"  {road(flat_road)} history is flat, which is what the real log")
        print("  produces — see step 4.")

        # -- the z-score ---------------------------------------------------- #
        print()
        print(f"2. POST /scenarios/{{id}}/detect   on {road(scored)}   "
              f"z > {Z_THRESHOLD} flags")
        print(RULE)
        for label, seconds in (
            ("at the mean", scored_stats.mean),
            (f"mean + {INSIDE_SIGMAS} sd", scored_stats.mean + INSIDE_SIGMAS * scored_stats.std_dev),
            (f"mean + {OUTSIDE_SIGMAS} sd", scored_stats.mean + OUTSIDE_SIGMAS * scored_stats.std_dev),
        ):
            print_verdict(label, seconds, observe(client, scenario_id, scored, seconds))
        print()
        print(f"  mean {scored_stats.mean:.1f}s, std dev "
              f"{scored_stats.std_dev:.2f}s — and the road's modelled "
              f"{condition} time")
        print(f"  is {base:.1f}s, so the history's mean sits on the model's own")
        print("  expectation and the score measures a departure from the road's")
        print("  normal rather than from a number this demo picked.")

        # -- the fallback --------------------------------------------------- #
        print()
        print(f"3. POST /scenarios/{{id}}/detect   on {road(scarce)}   "
              f"{scarce_stats.count} samples, under the {MIN_SAMPLES} a z-score needs")
        print(RULE)
        expectation = modelled_seconds(get_graph(), *scarce)
        print_verdict(
            f"exactly {FALLBACK_FACTOR}x the model",
            FALLBACK_FACTOR * expectation,
            observe(client, scenario_id, scarce, FALLBACK_FACTOR * expectation),
        )
        print_verdict(
            f"{1.5}x the model",
            1.5 * expectation,
            observe(client, scenario_id, scarce, 1.5 * expectation),
        )
        print()
        print(f"  the z-score is skipped entirely here, not computed and ignored:")
        print(f"  z_score comes back null, because reporting a number would claim a")
        print(f"  measurement that was never taken. The margin is strict, so an")
        print(f"  observation sitting exactly on {FALLBACK_FACTOR}x is not flagged.")

        # -- the case the real log produces --------------------------------- #
        print()
        print(f"4. POST /scenarios/{{id}}/detect   on {road(flat_road)}   "
              f"a history with no spread")
        print(RULE)
        print_verdict(
            "matching the history",
            flat_value,
            observe(client, scenario_id, flat_road, flat_value),
        )
        slower = round(flat_value * (1 + FLAT_DELTA), 3)
        print_verdict(
            f"{FLAT_DELTA:.0%} slower",
            slower,
            observe(client, scenario_id, flat_road, slower),
        )
        print()
        print("  This is the ordinary state of the collected log, not a corner")
        print("  case. A logged travel_time is base_travel_time x multiplier, both")
        print("  deterministic, so every sample for one (road, condition) key is the")
        print("  same number and the standard deviation is exactly zero. The")
        print("  z-score is then a division by zero that is not an error: equal to")
        print("  the history is 0 standard deviations out, and differing from it is")
        print("  infinitely many. An infinity cannot be written to JSON, so the")
        print("  verdict reports z_score null and std_dev 0 says why.")

        # -- the boundary --------------------------------------------------- #
        before = log_store.count()
        verdict = observe(client, scenario_id, scored, scored_stats.mean)
        print()
        print("5. A flag is a signal, not an action")
        print(RULE)
        print(f"  the verdict carries exactly: {', '.join(sorted(verdict))}")
        print("  no route, cost or solver field — re-optimizing on an anomaly is")
        print("  the reopt module's job, in a later prompt.")
        print(f"  rows in the log before that call: {before}; after: "
              f"{log_store.count()}")
        print("  nothing is written. Persisting an observation would feed the")
        print("  anomaly back into the baseline meant to catch it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
