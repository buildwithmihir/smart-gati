#!/usr/bin/env python
"""Inject a live incident and watch the cost matrix move, over real HTTP.

This is the Prompt 5 demonstration, and it shows the three things that prompt
asked for, in order, against the real application on the real Delhi graph — no
fixtures, no synthetic graph:

1. **A road closes or slows *after* the scenario exists.** The scenario is priced
   once at creation and keeps those costs. The incident is the operator override
   layered on top, and it applies to that one scenario.
2. **The edge weight genuinely changed in the cost matrix.** The leg is read back
   out of the stored matrix on the next read, before and after, and printed. The
   solve that follows is *evidence that the costs moved*, not the point of the
   call: the incident endpoint itself never runs a solver.
3. **A ``traffic_log`` row was written.**  ``GET /traffic/log`` is read back
   afterwards and the audit row printed — edge, timestamp, type, day, time.

The road it reports on is not invented. It is the first edge of the cheapest path
from the depot to the first stop *under the scenario's own conditions*, so it is
provably a road the scenario's cost matrix was built from.

Two honest caveats, both handled by the run:

- ``changed_legs`` counts objective-matrix entries that moved and **can be zero**.
  A road no cheapest path uses changes nothing about this instance, and the demo
  prints the number rather than picking a road that flatters it.
- A ``slow`` report at **peak** changes nothing on a through road, because the
  incident factor *replaces* that road's congestion multiplier rather than
  compounding with it. This demo runs at 02:00, outside the daytime band, which
  is where a report is actually visible.

Requests go through ``TestClient``, which drives the real FastAPI application
in-process — so there is no server to start, while routing, validation,
serialisation and the log write path are all exercised exactly as a client would.

Usage
-----
    uv run python benchmarks/demo_incident.py
    uv run python benchmarks/demo_incident.py --slow
    uv run python benchmarks/demo_incident.py --deliveries 12 --seed 7
    uv run python benchmarks/demo_incident.py --u 123 --v 456    # a named road
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Sequence

import networkx as nx
from fastapi.testclient import TestClient

from qgati.api.main import app, get_graph, get_store
from qgati.traffic import (
    CLEARED,
    CLOSURE,
    SLOW,
    Edge,
    simulated_travel_time,
    traffic_weight_function,
)

#: Delhi's fixed offset. 02:00 is outside the 06:00-22:00 daytime band, so every
#: road prices at x1.0 and a report on one is not swallowed by congestion that
#: was already there.
NORMAL_TIME = datetime(2026, 9, 21, 2, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

RULE = "-" * 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deliveries", type=int, default=8)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=21, help="fixes the instance")
    parser.add_argument(
        "--solver", default=None, help="defaults to the registry's production solver"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="report the road slow-but-passable instead of closing it",
    )
    parser.add_argument("--u", default=None, help="incident edge tail (graph node id)")
    parser.add_argument("--v", default=None, help="incident edge head (graph node id)")
    return parser


def create_scenario(client: TestClient, args) -> dict:
    response = client.post(
        "/scenarios",
        json={
            "kind": "generate",
            "n_deliveries": args.deliveries,
            "n_vehicles": args.vehicles,
            "seed": args.seed,
            "conditions": {"timestamp": NORMAL_TIME.isoformat()},
        },
    )
    response.raise_for_status()
    return response.json()


def solve(client: TestClient, scenario_id: str, solver: str | None) -> dict:
    """Run the solver, with geometry off so the route is readable."""
    body: dict = {"seed": 0, "include_geometry": False}
    if solver is not None:
        body["solver"] = solver
    response = client.post(f"/optimize/{scenario_id}", json=body)
    response.raise_for_status()
    return response.json()


def first_stop(solution: dict):
    """The first node the first non-empty route visits, or ``None``."""
    for route in solution["routes"]:
        if route["stops"]:
            return route["stops"][0]["node"]
    return None


def pick_road(client: TestClient, solution: dict, args) -> tuple:
    """A directed road on the cheapest path from the depot to the first stop.

    Returns ``(road_u, road_v, leg_from, leg_to)``, and the two pairs are not
    interchangeable. The first is a real **road edge** — what the report names and
    what the audit row is written for. The second is a pair of **scenario nodes**,
    the depot and the stop it is heading for, because the cost matrix is indexed
    by scenario node and a road's endpoints are usually junctions that appear in
    no delivery list.

    Choosing the road this way rather than inventing one means it is provably a
    road the scenario's own matrix was built from. Override with ``--u``/``--v``
    to name one explicitly.
    """
    record = get_store().get(solution["scenario_id"])
    depot = record.scenario.depot.node
    stop = first_stop(solution)
    if stop is None:
        raise SystemExit("the solved instance has no route to report on")

    if args.u is not None and args.v is not None:
        return int(args.u), int(args.v), depot, stop

    graph: nx.Graph = get_graph()
    weight = traffic_weight_function(graph, record.effective_traffic_state())
    path = nx.shortest_path(graph, depot, stop, weight=weight)
    if len(path) < 2:
        raise SystemExit("the depot and the first stop are the same node")
    return path[0], path[1], depot, stop


def matrix_leg(scenario_id: str, u, v) -> float:
    """One entry of the scenario's stored cost matrix, in seconds.

    Read straight off the record the API is serving from, which is what makes
    this the answer to "did the edge weight actually change on the next read?".

    Both nodes must be *scenario* nodes — the depot or a delivery. A junction
    in the middle of a road has no row or column of its own.
    """
    matrix = get_store().get(scenario_id).cost_matrix
    i, j = matrix.nodes.index(u), matrix.nodes.index(v)
    return float(matrix.matrix[i, j])


def edge_seconds(scenario_id: str, u, v) -> float | None:
    """The road's own cost under the scenario's current conditions, in seconds.

    Deliberately *not* the same number as the matrix leg. Routing prices the
    cheapest **path**, so a slowed road with a cheaper way round it pushes the leg
    onto the detour while the road itself is as slow as the report says. Printing
    both is what keeps the demo from implying the matrix entry is the road's cost.
    ``None`` when the road is impassable.
    """
    record = get_store().get(scenario_id)
    graph: nx.Graph = get_graph()
    return simulated_travel_time(graph, Edge(u, v), record.effective_traffic_state())


def seconds(value: float | None) -> str:
    return "impassable" if value is None else f"{value:.1f} s"


def incident(client: TestClient, scenario_id: str, payload: dict) -> dict:
    response = client.post(f"/scenarios/{scenario_id}/incident", json=payload)
    response.raise_for_status()
    return response.json()


def revert(client: TestClient, scenario_id: str, incident_id: str) -> dict:
    response = client.delete(f"/scenarios/{scenario_id}/incident/{incident_id}")
    response.raise_for_status()
    return response.json()


def log_rows(client: TestClient, **filters) -> list:
    response = client.get("/traffic/log", params={"limit": 20, **filters})
    response.raise_for_status()
    return response.json()["items"]


def describe_routes(body: dict) -> list:
    lines = []
    for route in body["routes"]:
        if not route["stops"]:
            continue
        stops = " -> ".join(str(stop["node"]) for stop in route["stops"])
        lines.append(f"    {route['vehicle_id']}: {stops}")
    return lines


def print_log_row(entry: dict) -> None:
    print(
        f"    {entry['road_id']:<18} {entry['time_of_day']:<6} "
        f"{entry['day_of_week'][:3]:<4} {entry['traffic_condition']:<8} "
        f"{str(entry['incident_type']):<11} {seconds(entry['travel_time'])}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kind = SLOW if args.slow else CLOSURE

    print("Q-Gati — live incident demo")
    print("=" * 78)
    print(f"instance   {args.deliveries} deliveries, {args.vehicles} vehicles, "
          f"seed {args.seed}")
    print(f"priced at  {NORMAL_TIME.isoformat()}  (normal band, x1.0 everywhere)")
    print(f"report     {kind}")

    with TestClient(app) as client:
        created = create_scenario(client, args)
        scenario_id = created["scenario_id"]
        before_solution = solve(client, scenario_id, args.solver)
        road_u, road_v, leg_from, leg_to = pick_road(client, before_solution, args)

        print(f"scenario   {scenario_id}")
        print(f"stop order {leg_from} (depot) -> {leg_to} (first delivery)")
        print(f"road       {road_u} -> {road_v}   "
              f"(first hop of that leg's cheapest path)")

        # -- before --------------------------------------------------------- #
        before_leg = matrix_leg(scenario_id, leg_from, leg_to)
        before_edge = edge_seconds(scenario_id, road_u, road_v)

        print()
        print("1. Before the report")
        print(RULE)
        print(f"  cost matrix [{leg_from}][{leg_to}]   {before_leg:.1f} s   "
              f"conditions.mutated = {created['conditions']['mutated']}")
        print(f"  that road on its own      {seconds(before_edge)}")
        print(f"  {before_solution['solver_name']}: "
              f"time {before_solution['travel_time']:.1f} s  "
              f"cost Rs{before_solution['travel_cost']:.2f}")
        for line in describe_routes(before_solution):
            print(line)

        # -- inject --------------------------------------------------------- #
        applied = incident(
            client,
            scenario_id,
            {"incident_type": kind, "edge": {"u": road_u, "v": road_v}},
        )
        after_leg = matrix_leg(scenario_id, leg_from, leg_to)
        after_edge = edge_seconds(scenario_id, road_u, road_v)
        after_solution = solve(client, scenario_id, args.solver)

        print()
        print("2. POST /scenarios/{id}/incident   ->  201")
        print(RULE)
        print(f"  applied          {applied['applied']}")
        print(f"  changed_legs     {applied['changed_legs']}")
        print(f"  rows logged      {applied['traffic_rows_logged']}")
        print()
        print(f"  that road on its own      {seconds(before_edge)}  ->  "
              f"{seconds(after_edge)}")
        print(f"  cost matrix [{leg_from}][{leg_to}]   {before_leg:.1f} s  ->  "
              f"{after_leg:.1f} s")

        if after_leg == before_leg:
            print()
            print("  the leg did not move. The road did — the two numbers above are")
            print("  different on purpose: routing prices the cheapest *path*, so a")
            print("  slower road with an equally good way round it changes nothing")
            print("  about this instance. changed_legs counts what moved.")
        elif after_leg != after_edge:
            print()
            print("  the leg moved to a value that is not the road's own cost, because")
            print("  the fastest path now goes round it. Both numbers are correct:")
            print("  the road is as slow as reported, and the leg is the cheapest way")
            print("  between those two stops. The audit row below records the road.")

        conditions = applied["conditions"]
        print()
        print(f"  conditions.mutated   {conditions['mutated']}")
        for item in conditions["incidents"]:
            print(
                f"  incident             {item['incident_id'][:8]}  "
                f"{item['incident_type']}  {item['edge']['u']} -> {item['edge']['v']}"
                f"   reported {item['created_at']}"
            )

        # -- the audit trail ------------------------------------------------ #
        rows = log_rows(client, incident_type=kind)
        print()
        print(f"3. GET /traffic/log?incident_type={kind}   ->  {len(rows)} row(s)")
        print(RULE)
        for entry in rows:
            print_log_row(entry)
        if rows:
            row = rows[0]
            print()
            print(f"  the row carries the *scenario's* timestamp ({row['timestamp']}),")
            print("  not the wall clock of the report — so it is a valid observation")
            print("  of the conditions this scenario is priced under, and pairs with")
            print("  the rows already collected for it.")
            print()
            print("  travel_time is the road's own cost under the report; the leg")
            print("  above is the cheapest *path*, which may be a way round it.")

        # -- the boundary --------------------------------------------------- #
        print()
        print("4. The incident endpoint never ran a solver")
        print(RULE)
        print(f"  its response carries exactly: {', '.join(sorted(applied))}")
        print("  no route, cost or solver field — re-optimizing on an incident is")
        print("  the reopt module's job, in a later prompt.")
        print()
        print(f"  the solve above was a separate POST /optimize, and it changed "
              f"because")
        print(f"  the costs moved: {before_solution['travel_time']:.1f} s -> "
              f"{after_solution['travel_time']:.1f} s")

        # -- revert --------------------------------------------------------- #
        reverted = revert(client, scenario_id, applied["incident"]["incident_id"])
        restored_leg = matrix_leg(scenario_id, leg_from, leg_to)
        restored_edge = edge_seconds(scenario_id, road_u, road_v)

        print()
        print("5. DELETE /scenarios/{id}/incident/{incident_id}   ->  200")
        print(RULE)
        print(f"  changed_legs     {reverted['changed_legs']}")
        print(f"  that road on its own      {seconds(after_edge)}  ->  "
              f"{seconds(restored_edge)}   (was {seconds(before_edge)} before)")
        print(f"  cost matrix [{leg_from}][{leg_to}]   {after_leg:.1f} s  ->  "
              f"{restored_leg:.1f} s   (was {before_leg:.1f} s before)")
        print(f"  conditions.mutated   {reverted['conditions']['mutated']}"
              f"   incidents={len(reverted['conditions']['incidents'])}")
        print()
        for entry in log_rows(client, incident_type=CLEARED):
            print_log_row(entry)
        print()
        print("  a revert is exact, not subtractive: the conditions the scenario was")
        print("  created with are kept underneath and re-folded, so anything it was")
        print("  born with survives the round trip.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
