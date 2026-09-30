#!/usr/bin/env python
"""TEMPORARY — proves the dashboard's incident flow end to end over real HTTP.

Not to be committed. This exists to check the exact request sequence the new
frontend code performs, because the frontend itself cannot be compiled or run
from here:

    load sample  ->  POST /scenarios
                     POST /scenarios/{id}/watcher      (solves + dispatches)
                     GET  /graph/delhi?scenario_id=…
    click a road ->  POST /scenarios/{id}/incident     x1 per directed edge
                     POST /scenarios/{id}/reoptimize
    clear        ->  DELETE /scenarios/{id}/incident/{incident_id}

The road is chosen the way a *click* chooses one, not the way an operator would
if they had the node ids: a segment of a route's own drawn polyline is taken,
its endpoints are read back as nodes, and `bothDirectionsOf`'s twin lookup is
reproduced from the GeoJSON. That is what the map does — see
`frontend/lib/road-picking.ts` — so this exercises the same resolution the UI
depends on, including the part where a street is two directed edges.

Pass criteria printed at the end; the script exits non-zero if any fail.

Usage
-----
    uv run python incident_flow_smoke.py
    uv run python incident_flow_smoke.py --kind slow
"""

from __future__ import annotations

import argparse
import sys

from fastapi.testclient import TestClient

from qgati.api.main import app

# Mirrors SAMPLE_SCENARIO_PAYLOAD / SAMPLE_SOLVER_SEED in frontend/lib/api.ts.
SAMPLE_PAYLOAD = {"kind": "generate", "n_deliveries": 12, "n_vehicles": 3, "seed": 21}
SAMPLE_SEED = 0
SAMPLE_INTERVAL_SECONDS = 10
SAMPLE_TIME_SCALE = 1.0


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def plan_seconds(routes: list[dict]) -> float:
    """The dashboard's `planSeconds` — summed per-route travel time."""
    return sum(route["travel_time"] for route in routes)


def route_shape(routes: list[dict]) -> str:
    """The dashboard's `routesChanged` — ordered stop sequences per vehicle."""
    return "|".join(
        f"{route['vehicle_id']}:"
        + ",".join(stop["delivery_id"] for stop in route["stops"])
        for route in routes
        if route["stops"]
    )


def node_index(geojson: dict) -> dict[tuple[float, float], int]:
    """(lon, lat) -> node id, from the graph's Point features."""
    lookup: dict[tuple[float, float], int] = {}
    for feature in geojson["features"]:
        if feature["geometry"]["type"] != "Point":
            continue
        lon, lat = feature["geometry"]["coordinates"]
        lookup[(round(lon, 6), round(lat, 6))] = feature["properties"]["id"]
    return lookup


def line_index(geojson: dict) -> dict[tuple[int, int], list[list[float]]]:
    """(u, v) -> polyline, over every directed edge the graph emits."""
    lookup: dict[tuple[int, int], list[list[float]]] = {}
    for feature in geojson["features"]:
        if feature["geometry"]["type"] != "LineString":
            continue
        u = feature["properties"]["u"]
        v = feature["properties"]["v"]
        lookup[(u, v)] = feature["geometry"]["coordinates"]
    return lookup


def click_somewhere_on(routes: list[dict], nodes: dict, lines: dict) -> tuple[int, int] | None:
    """The road a click lands on, resolved the way `pickRoad` resolves it.

    Walks the first route's drawn polyline and returns the first segment whose
    endpoints resolve to nodes and whose edge — or whose reverse twin — is in
    the graph. The click point itself is not modelled: any point on the segment
    resolves to the same edge, which is the property being relied on.
    """
    for route in routes:
        geometry = route.get("geometry") or []
        for start, end in zip(geometry, geometry[1:]):
            u = nodes.get((round(start[0], 6), round(start[1], 6)))
            v = nodes.get((round(end[0], 6), round(end[1], 6)))
            if u is None or v is None or u == v:
                continue
            if (u, v) in lines or (v, u) in lines:
                return (u, v)
    return None


def both_directions(lines: dict, u: int, v: int) -> list[tuple[int, int]]:
    """`bothDirectionsOf` — the clicked edge first, then its reverse twin."""
    edges = [(u, v)] if (u, v) in lines else []
    if (v, u) in lines and u != v:
        edges.append((v, u))
    return edges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="closure", choices=["closure", "slow"])
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    with TestClient(app) as client:
        banner("1 — LOAD SAMPLE: create, then dispatch a fleet")
        created = client.post("/scenarios", json=SAMPLE_PAYLOAD)
        created.raise_for_status()
        sid = created.json()["scenario_id"]
        print(f"scenario_id        {sid}")

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
        dispatched_routes = dispatch["routes"]
        print(f"solver             {dispatch['solver']}")
        print(f"ticks              {dispatch['ticks']}")
        print(f"routes             {len(dispatched_routes)}")
        print(f"dispatch time      {plan_seconds(dispatched_routes):.1f} s")
        print(f"first tick legs    {dispatch['first_tick']['changed_legs']} changed")

        check("the watcher dispatched at least one route", len(dispatched_routes) > 0)
        check(
            "the routes carry geometry, so the map can draw them",
            all(route["geometry"] for route in dispatched_routes),
        )
        check("the watcher is running", dispatch["running"] is True)

        graph = client.get(f"/graph/delhi?scenario_id={sid}")
        graph.raise_for_status()
        geojson = graph.json()
        nodes = node_index(geojson)
        lines = line_index(geojson)
        print(f"graph features     {len(geojson['features'])}")
        check("the scoped graph carries both node and road features", len(nodes) > 0 and len(lines) > 0)

        banner("2 — CLICK A ROAD: resolve the street the way the map does")
        clicked = click_somewhere_on(dispatched_routes, nodes, lines)
        if clicked is None:
            print("  no road on the dispatched routes resolved to a graph edge")
            return 1
        u, v = clicked
        street = both_directions(lines, u, v)
        print(f"clicked            {u} -> {v}")
        print(f"carriageways       {len(street)}  {street}")
        check("the clicked road resolved to at least one directed edge", len(street) > 0)
        check(
            "both directions were found, so a two-way street is closed both ways",
            len(street) == 2,
            "one-way street — one carriageway is correct here",
        )

        banner(f"3 — REPORT IT: POST /incident once per carriageway ({args.kind})")
        incidents: list[dict] = []
        changed_legs = 0
        for edge_u, edge_v in street:
            reported = client.post(
                f"/scenarios/{sid}/incident",
                json={"incident_type": args.kind, "edge": {"u": edge_u, "v": edge_v}},
            )
            reported.raise_for_status()
            body = reported.json()
            incidents.append(body["incident"])
            changed_legs += body["changed_legs"]
            print(
                f"  ({edge_u}, {edge_v})  applied={body['applied']}  "
                f"changed_legs={body['changed_legs']}  rows_logged={body['traffic_rows_logged']}"
            )

        print(f"total changed_legs  {changed_legs}")
        print(f"live incidents      {len(incidents)}")
        check("every carriageway report was accepted", len(incidents) == len(street))
        check(
            "the incident re-priced at least one leg",
            changed_legs > 0,
            "zero is legitimate but means this road is on no cheapest path",
        )

        banner("4 — RE-PLAN: POST /reoptimize, exactly as the panel does")
        replanned = client.post(f"/scenarios/{sid}/reoptimize", json={"seed": SAMPLE_SEED})
        print(f"status             {replanned.status_code}")
        if replanned.status_code != 200:
            print(f"detail             {replanned.json().get('detail')}")
            return 1
        report = replanned.json()

        before_s = plan_seconds(report["before"])
        after_s = plan_seconds(report["after"])
        changed = route_shape(report["before"]) != route_shape(report["after"])

        print(f"trigger kinds      {report['trigger']['kinds']}")
        print(f"trigger detail     {report['trigger']['detail']}")
        print(f"before             {before_s:.1f} s over {len(report['before'])} vehicles")
        print(f"after              {after_s:.1f} s over {len(report['after'])} vehicles")
        print(f"difference         {after_s - before_s:+.1f} s")
        print(f"route changed      {'yes' if changed else 'no'}")
        print(f"stops re-assigned  {len(report['moved'])} of {len(report['replanned'])}")
        print(f"unserved stops     {len(report['replanned'])}")
        print(f"replanned vehicles {report['replanned_vehicles']}")
        check("the re-plan was triggered by the incident", "incident" in report["trigger"]["kinds"])
        check("the report carries both before and after routes", bool(report["before"]) and bool(report["after"]))
        check(
            "the after routes answer for one vehicle per vehicle that has work left",
            len(report["after"]) <= len(dispatched_routes),
            "after covers only the vehicles with stops remaining",
        )

        banner("5 — CLEAR: DELETE each incident, then confirm the world is back")
        for incident in incidents:
            reverted = client.delete(f"/scenarios/{sid}/incident/{incident['incident_id']}")
            reverted.raise_for_status()
            body = reverted.json()
            print(
                f"  {incident['incident_id'][:8]}  applied={body['applied']}  "
                f"live incidents={len(body['conditions']['incidents'])}"
            )

        # Read back through `GET /scenarios/{id}`, which echoes the stored
        # scenario's conditions — the same list the reports were filed into.
        stored = client.get(f"/scenarios/{sid}")
        stored.raise_for_status()
        live = stored.json()["conditions"]["incidents"]
        print(f"incidents remaining {len(live)}")
        check("no incident is left on the scenario", len(live) == 0)

        # The dashboard does *not* re-plan on clear; it puts the dispatched plan
        # back, because `/reoptimize` writes nothing and so the fleet is still
        # driving exactly those routes. This is the check that the claim is true
        # rather than merely convenient: with the incident gone there is no
        # trigger left, and the endpoint says so by refusing.
        after_clear = client.post(f"/scenarios/{sid}/reoptimize", json={"seed": SAMPLE_SEED})
        print(f"reoptimize now     {after_clear.status_code}")
        if after_clear.status_code != 200:
            print(f"detail             {after_clear.json().get('detail')}")
        check(
            "clearing the incident removes the trigger, so the dispatched plan stands",
            after_clear.status_code == 409,
            "409 = no trigger, which is what makes reverting to the dispatch truthful",
        )

        fleet_after = client.get(f"/scenarios/{sid}/watcher")
        fleet_after.raise_for_status()
        check(
            "the fleet was never disturbed by any of this",
            fleet_after.json()["running"] is True,
            f"ticks={fleet_after.json()['ticks']}",
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
