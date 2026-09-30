#!/usr/bin/env python
"""TEMPORARY — prints the actual rows the run-history table holds. Not to be committed.

The request this answers is "show me actual rows after running a few
optimizations and one incident trigger during testing", so that is exactly what
it does, against the **real** store — `backend/data/run_history.db`, the one
`GET /analytics/runs` reads and the History tab renders. The test suite
deliberately overrides this dependency to an in-memory store; here the override
is the thing being demonstrated, so there is none.

The sequence, in the order the dashboard would produce it:

    POST /scenarios                                 create the sample instance
    POST /optimize/{id}          x2                 "a few optimizations"
    POST /scenarios/{id}/watcher                    dispatch — solves, writes row
    POST .../vehicles/{v}/avoid-road                the driver's own report
    DELETE /scenarios/{id}/incident/{incident_id}   withdraw it, so the next
                                                    re-plan has one trigger only
    POST /scenarios/{id}/incident                   close a road a route runs along
    POST /scenarios/{id}/reoptimize                 re-plan it, with an ETA pair
    POST /optimize/{id}                             one more, after the incident

then every row, a filtered page, a paged read, and the aggregate summary.

The road is chosen the way a *click* chooses one — a segment of a route's own
drawn polyline, resolved through the graph's node and line features — so the
incident lands on a road the plan actually uses. That resolution is
`incident_flow_smoke.py`'s, imported rather than copied: two scripts that
disagree about what a click resolves to would make the incident land on a
different road in each, and this one would then be reporting on a network the
smoke test had not checked.

**Running it twice writes the rows twice.** That is the point of it.

Usage
-----
    uv run python run_history_demo.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

from fastapi.testclient import TestClient

from incident_flow_smoke import both_directions, click_somewhere_on, line_index, node_index
from qgati.analytics import DEFAULT_RUN_DB_PATH
from qgati.api.main import app

# Mirrors SAMPLE_SCENARIO_PAYLOAD / SAMPLE_SOLVER_SEED in frontend/lib/api.ts.
SAMPLE_PAYLOAD = {"kind": "generate", "n_deliveries": 12, "n_vehicles": 3, "seed": 21}
SAMPLE_SEED = 0
SAMPLE_INTERVAL_SECONDS = 10
SAMPLE_TIME_SCALE = 1.0

#: How many of the newest rows to dump as raw JSON, verbatim.
RAW_ROWS = 3


def banner(text: str) -> None:
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def eta_of(row: dict) -> str:
    """The row's ETA pair, or an em dash when the row had no baseline."""
    if row["old_eta_seconds"] is None or row["new_eta_seconds"] is None:
        return "—"
    return (
        f"{row['old_eta_seconds']:8.1f} → {row['new_eta_seconds']:8.1f} "
        f"({row['eta_saved_seconds']:+8.1f} s)"
    )


def print_rows(rows: list[dict]) -> None:
    """The table as the History tab lays it out, minus the formatting."""
    header = (
        f"{'id':>4}  {'when (local)':<22}  {'kind':<11}  {'scenario':<9}  "
        f"{'solver':<5}  {'cost (₹)':>11}  {'time':>9}  {'runtime':>10}  "
        f"{'ETA before → after':<40}  moved"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        # Stored UTC, shown local — the same conversion `formatRunTimestamp` does.
        when = datetime.fromisoformat(row["timestamp"]).astimezone()
        print(
            f"{row['id']:>4}  {when.strftime('%d %b, %H:%M:%S'):<22}  {row['kind']:<11}  "
            f"{row['scenario_id'][:8]:<9}  {row['solver_name']:<5}  "
            f"{row['travel_cost']:>11,.2f}  {row['travel_time']:>7.1f}s  "
            f"{row['runtime_ms']:>7.0f} ms  {eta_of(row):<40}  {row['moved']}"
        )


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    banner("0 — THE STORE THIS WILL WRITE TO")
    print(f"run history file   {DEFAULT_RUN_DB_PATH}")
    print(f"exists             {DEFAULT_RUN_DB_PATH.exists()}")

    with TestClient(app) as client:
        before = client.get("/analytics/summary").json()
        print(f"runs already there {before['total_runs']}")
        if before["total_runs"]:
            print(
                f"  by kind          {before['runs_by_kind']}  "
                f"(this run will add to these, not replace them)"
            )

        banner("1 — A FEW OPTIMIZATIONS: POST /optimize")
        created = client.post("/scenarios", json=SAMPLE_PAYLOAD)
        created.raise_for_status()
        sid = created.json()["scenario_id"]
        print(f"scenario_id        {sid}")

        optimizes = []
        for seed in (SAMPLE_SEED, 7):
            response = client.post(f"/optimize/{sid}", json={"seed": seed})
            response.raise_for_status()
            body = response.json()
            optimizes.append(body)
            print(
                f"  seed {seed}  {body['solver_name']:<5}  cost ₹{body['travel_cost']:,.2f}  "
                f"{body['travel_time']:.1f} s  {body['runtime_ms']:.0f} ms  "
                f"feasible={body['feasible']}"
            )

        banner("2 — DISPATCH: POST /watcher, which is also a solve")
        started = client.post(
            f"/scenarios/{sid}/watcher",
            json={
                "seed": SAMPLE_SEED,
                "interval_seconds": SAMPLE_INTERVAL_SECONDS,
                "time_scale": SAMPLE_TIME_SCALE,
                "include_geometry": True,
            },
        )
        started.raise_for_status()
        dispatch = started.json()
        routes = dispatch["routes"]
        print(f"solver             {dispatch['solver']}")
        print(f"routes             {len(routes)}")
        print(f"dispatch time      {sum(r['travel_time'] for r in routes):.1f} s")

        graph = client.get(f"/graph/delhi?scenario_id={sid}")
        graph.raise_for_status()
        geojson = graph.json()
        nodes = node_index(geojson)
        lines = line_index(geojson)

        # A vehicle with work to do, which is what both re-planning routes need.
        working = next((route for route in routes if route["stops"]), None)
        if working is None:
            print("  the dispatch produced no route with a stop on it")
            return 1
        vehicle_id = working["vehicle_id"]

        # Two different roads on the drawn routes: one for the driver's own
        # report, one for the operator's incident.
        first = click_somewhere_on([working], nodes, lines)
        if first is None:
            print("  no road on the dispatched routes resolved to a graph edge")
            return 1
        rest = click_somewhere_on(
            [route for route in routes if route is not working] + [working], nodes, lines
        )
        second = rest if rest != first else None

        banner(f"3 — THE DRIVER'S OWN REPORT: avoid-road on {first}")
        avoided = client.post(
            f"/scenarios/{sid}/vehicles/{vehicle_id}/avoid-road",
            json={"edge": {"u": first[0], "v": first[1]}, "treatment": "slow"},
        )
        print(f"status             {avoided.status_code}")
        if avoided.status_code == 200:
            body = avoided.json()
            filed = body["incident"]
            print(f"trigger            {body['trigger']['primary']} — {body['trigger']['detail']}")
            print(f"before / after     {sum(r['travel_time'] for r in body['before']):.1f} s "
                  f"/ {sum(r['travel_time'] for r in body['after']):.1f} s")
            print(f"moved              {len(body['moved'])}")
            # Withdrawn again so the re-plan below has exactly one trigger and the
            # demo's ETA pair is attributable to the incident and not to this too.
            if filed is not None:
                reverted = client.delete(f"/scenarios/{sid}/incident/{filed['incident_id']}")
                print(f"withdrawn          {reverted.status_code}  (so the next re-plan has one trigger)")
        else:
            print(f"detail             {avoided.json().get('detail')}")

        banner("4 — THE INCIDENT: close a road a drawn route runs along, then re-plan")
        if second is None:
            print("  only one road resolved; re-using it for the incident")
            second = first
        u, v = second
        street = both_directions(lines, u, v)
        print(f"clicked            {u} -> {v}   carriageways {street}")

        incidents: list[dict] = []
        for edge_u, edge_v in street:
            reported = client.post(
                f"/scenarios/{sid}/incident",
                json={"incident_type": "closure", "edge": {"u": edge_u, "v": edge_v}},
            )
            reported.raise_for_status()
            incidents.append(reported.json()["incident"])
            print(
                f"  ({edge_u}, {edge_v})  changed_legs={reported.json()['changed_legs']}  "
                f"rows_logged={reported.json()['traffic_rows_logged']}"
            )
        check("the incident was filed", len(incidents) > 0)

        replanned = client.post(f"/scenarios/{sid}/reoptimize", json={"seed": SAMPLE_SEED})
        print(f"reoptimize status  {replanned.status_code}")
        if replanned.status_code != 200:
            print(f"detail             {replanned.json().get('detail')}")
            return 1
        report = replanned.json()
        before_s = sum(route["travel_time"] for route in report["before"])
        after_s = sum(route["travel_time"] for route in report["after"])
        print(f"trigger            {report['trigger']['primary']} — {report['trigger']['detail']}")
        print(f"before             {before_s:.1f} s   (what the fleet was already driving)")
        print(f"after              {after_s:.1f} s   (what the re-plan takes)")
        print(f"difference         {after_s - before_s:+.1f} s")
        print(f"moved              {len(report['moved'])}")

        banner("5 — ONE MORE OPTIMIZATION, now that the road is shut")
        final = client.post(f"/optimize/{sid}", json={"seed": SAMPLE_SEED})
        final.raise_for_status()
        print(
            f"  {final.json()['solver_name']}  cost ₹{final.json()['travel_cost']:,.2f}  "
            f"{final.json()['travel_time']:.1f} s"
        )

        banner("6 — GET /analytics/runs — every row this run wrote")
        listed = client.get("/analytics/runs?limit=50")
        listed.raise_for_status()
        page = listed.json()
        print(f"total rows in the table   {page['total']}")
        print(f"rows on this page         {len(page['items'])}")
        print(f"has_more                  {page['has_more']}")
        print()
        print_rows(page["items"])

        written = [row for row in page["items"] if row["scenario_id"] == sid]
        print()
        print(f"rows this script wrote    {len(written)}  (they are at the head of the table)")

        kinds = [row["kind"] for row in reversed(written)]
        print(f"kinds, oldest first       {kinds}")

        banner(f"7 — THE NEWEST {RAW_ROWS} ROWS, as JSON — the exact objects the API returns")
        for row in page["items"][:RAW_ROWS]:
            print(json.dumps(row, indent=2))

        banner("8 — THE READERS: a filtered page, a paged read, and the summary")
        filtered = client.get("/analytics/runs?kind=reoptimize")
        filtered.raise_for_status()
        print(
            f"?kind=reoptimize           {len(filtered.json()['items'])} rows on the page, "
            f"total {filtered.json()['total']}"
        )
        by_scenario = client.get(f"/analytics/runs?scenario_id={sid}")
        by_scenario.raise_for_status()
        print(
            f"?scenario_id={sid[:8]}     total {by_scenario.json()['total']}"
        )
        one = client.get("/analytics/runs?limit=1&offset=1")
        one.raise_for_status()
        print(
            f"?limit=1&offset=1          {len(one.json()['items'])} row, "
            f"total {one.json()['total']}, offset {one.json()['offset']}, "
            f"has_more {one.json()['has_more']}"
        )

        summary = client.get("/analytics/summary")
        summary.raise_for_status()
        summary_body = summary.json()
        print()
        print(json.dumps(summary_body, indent=2))

        # -- the receipts --------------------------------------------------- #
        banner("9 — DO THE ROWS SAY WHAT THE RESPONSES SAID?")
        # Newest first, so the *first* row seen for a kind is that kind's newest —
        # which is why this is a `setdefault` loop and not `{row["kind"]: row ...}`.
        # A comprehension would overwrite as it went and hand back the oldest
        # optimize of the three this script writes, while the variable still said
        # "latest".
        latest: dict[str, dict] = {}
        for row in page["items"]:
            if row["scenario_id"] == sid:
                latest.setdefault(row["kind"], row)
        check(
            "all four kinds were recorded",
            set(kinds) == {"optimize", "dispatch", "avoid_road", "reoptimize"},
            f"saw {sorted(set(kinds))}",
        )
        check(
            "the dispatch row's travel time matches the watcher's own routes",
            abs(
                latest["dispatch"]["travel_time"]
                - sum(route["travel_time"] for route in routes)
            )
            < 1e-6,
            f"row {latest['dispatch']['travel_time']:.3f} s "
            f"vs routes {sum(r['travel_time'] for r in routes):.3f} s",
        )
        check(
            "the re-plan row's old ETA is the response's own `before` travel time",
            abs(latest["reoptimize"]["old_eta_seconds"] - before_s) < 1e-6,
            f"row {latest['reoptimize']['old_eta_seconds']:.3f} s vs before {before_s:.3f} s",
        )
        check(
            "the re-plan row's new ETA is the response's own `after` travel time",
            abs(latest["reoptimize"]["new_eta_seconds"] - after_s) < 1e-6,
            f"row {latest['reoptimize']['new_eta_seconds']:.3f} s vs after {after_s:.3f} s",
        )
        check(
            "the re-plan row names the incident as its trigger",
            latest["reoptimize"]["trigger"] == "incident",
            f"trigger={latest['reoptimize']['trigger']!r}",
        )
        check(
            "the avoid-road row names the override and the vehicle",
            latest["avoid_road"]["trigger"] == "override"
            and latest["avoid_road"]["affected_vehicle"] == vehicle_id,
            f"trigger={latest['avoid_road']['trigger']!r} "
            f"vehicle={latest['avoid_road']['affected_vehicle']!r}",
        )
        check(
            "every row carries a runtime, and none of them is a placeholder zero",
            all(row["runtime_ms"] > 0 for row in written),
            f"min {min(row['runtime_ms'] for row in written):.1f} ms",
        )
        check(
            "the summary's total grew by exactly the rows this script wrote",
            summary_body["total_runs"] == before["total_runs"] + len(written),
            f"{before['total_runs']} + {len(written)} = {summary_body['total_runs']}",
        )
        check(
            "the summary counts the incident-triggered re-plan",
            summary_body["incident_triggered_runs"] >= 1,
            f"incident_triggered_runs={summary_body['incident_triggered_runs']}",
        )
        check(
            "the ETA block is computed over the re-plans only, and says how many",
            summary_body["eta"]["runs"] >= 2,
            f"eta.runs={summary_body['eta']['runs']} "
            f"(a dispatch and an optimize are never averaged into it)",
        )

    banner("RESULT")
    failed = [name for name, ok, _ in checks if not ok]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    print()
    print(f"{len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
