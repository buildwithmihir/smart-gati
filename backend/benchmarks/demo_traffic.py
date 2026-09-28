#!/usr/bin/env python
"""Step 5 demo: peak hour changes the route, and the log collects the roads.

Two things this shows, both from real requests against the real application on
the real Delhi graph — no fixtures, no synthetic graph:

1. **One delivery round, priced twice.** The identical instance (same seed, so
   the same stops and the same fleet) is created once for 09:00 and once for
   14:00. Peak hour weights congestion by road class, so the arterial-heavy tour
   that wins off-peak loses under peak and the optimizer takes a different route.
2. **The traffic log fills up on its own.** Every scenario creation appends a row
   per affected road as a side effect. ``GET /traffic/log`` is then read back and
   printed, so the rows shown are real response bodies from the running app.

Why road class matters, in one line: a multiplier applied uniformly to every road
scales every tour equally and therefore cannot change which tour wins. The
peak factor is weighted by road class precisely so that it can.

Requests go through ``TestClient``, which drives the real FastAPI application
in-process. That keeps the demo to one command with no server to start, while
still exercising routing, validation, serialisation and the log write path
exactly as a client would.

Usage
-----
    uv run python benchmarks/demo_traffic.py
    uv run python benchmarks/demo_traffic.py --deliveries 15 --vehicles 4 --seed 7
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from fastapi.testclient import TestClient

from qgati.api.main import app
from qgati.graph import load_delhi_graph
from qgati.traffic import DEFAULT_LOG_DB_PATH, TrafficLogStore

#: Delhi's fixed offset, so the two runs land on a known side of the peak window.
IST = timezone(timedelta(hours=5, minutes=30))

#: 09:00 is inside the 08:00-10:00 morning peak; 14:00 is not.
PEAK_TIME = datetime(2026, 9, 21, 9, 0, tzinfo=IST)
OFF_PEAK_TIME = datetime(2026, 9, 21, 14, 0, tzinfo=IST)

RULE = "-" * 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deliveries", type=int, default=12)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=21, help="fixes the instance")
    parser.add_argument(
        "--solver", default="aco", help="production default unless overridden"
    )
    parser.add_argument("--log-rows", type=int, default=8, help="rows to print")
    return parser


def create_scenario(client: TestClient, args, when: datetime) -> dict:
    """Create the instance priced at ``when``, and return the response body."""
    response = client.post(
        "/scenarios",
        json={
            "kind": "generate",
            "n_deliveries": args.deliveries,
            "n_vehicles": args.vehicles,
            "seed": args.seed,
            "conditions": {"timestamp": when.isoformat()},
        },
    )
    response.raise_for_status()
    return response.json()


def optimize(client: TestClient, scenario_id: str, solver: str) -> dict:
    response = client.post(
        f"/optimize/{scenario_id}",
        json={"solver": solver, "seed": 0, "include_geometry": False},
    )
    response.raise_for_status()
    return response.json()


def describe_routes(body: dict) -> list[str]:
    """One readable line per vehicle: its stops, in visit order."""
    lines = []
    for route in body["routes"]:
        if not route["stops"]:
            continue
        stops = " -> ".join(stop["delivery_id"] for stop in route["stops"])
        lines.append(f"    {route['vehicle_id']}: {stops}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("Q-Gati — traffic simulation demo")
    print("=" * 78)
    print(f"graph      real Delhi extract ({load_delhi_graph().number_of_edges()} edges)")
    print(f"instance   {args.deliveries} deliveries, {args.vehicles} vehicles, "
          f"seed {args.seed}  (identical for both runs)")
    print(f"solver     {args.solver}")
    print(f"log db     {DEFAULT_LOG_DB_PATH}")

    before = TrafficLogStore().count() if Path(DEFAULT_LOG_DB_PATH).exists() else 0

    with TestClient(app) as client:
        peak = create_scenario(client, args, PEAK_TIME)
        off_peak = create_scenario(client, args, OFF_PEAK_TIME)

        peak_solution = optimize(client, peak["scenario_id"], args.solver)
        off_peak_solution = optimize(client, off_peak["scenario_id"], args.solver)

        log_page = client.get("/traffic/log", params={"limit": args.log_rows}).json()
        total_rows = client.get("/traffic/log", params={"limit": 1}).json()["total"]

    print()
    print("1. The same delivery round, priced at two times of day")
    print(RULE)
    for label, created, solved in (
        ("off-peak 14:00", off_peak, off_peak_solution),
        ("peak     09:00", peak, peak_solution),
    ):
        conditions = created["conditions"]
        print(
            f"  {label}   {conditions['traffic_condition']:<9} "
            f"travel cost {solved['travel_cost']:>9.1f} s"
        )
        for line in describe_routes(solved):
            print(line)

    same = [
        [stop["delivery_id"] for stop in route["stops"]]
        for route in peak_solution["routes"]
    ] == [
        [stop["delivery_id"] for stop in route["stops"]]
        for route in off_peak_solution["routes"]
    ]
    verdict = "IDENTICAL — the peak factor would be decorative" if same else "DIFFERENT"
    print()
    print(f"  routes: {verdict}")

    print()
    print("2. The traffic log, written as a side effect of the two creations above")
    print(RULE)
    print(f"  rows logged by these two scenarios: "
          f"{peak['traffic_rows_logged']} + {off_peak['traffic_rows_logged']}")
    print(f"  total rows in the table now:       {total_rows} "
          f"(was {before} before this run)")
    print()
    print(f"  GET /traffic/log?limit={args.log_rows}  ->  {len(log_page['items'])} of "
          f"{log_page['total']} matching rows")
    print()

    header = (
        f"  {'road_id':<24} {'time':<6} {'day':<4} {'weather':<8} "
        f"{'traffic':<9} {'incident':<13} {'travel_time':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for entry in log_page["items"]:
        travel_time = entry["travel_time"]
        rendered = "impassable" if travel_time is None else f"{travel_time:>11.3f}"
        print(
            f"  {entry['road_id']:<24} {entry['time_of_day']:<6} "
            f"{entry['day_of_week'][:3]:<4} {entry['weather_condition']:<8} "
            f"{entry['traffic_condition']:<9} {str(entry['incident_type'] or '-'):<13} "
            f"{rendered}"
        )

    print()
    print("  travel_time is seconds under these conditions; NULL means the road is")
    print("  impassable, with incident_type carrying the reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
