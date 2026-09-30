#!/usr/bin/env python
"""TEMPORARY — prints one real decision trace, end to end, over real HTTP.

Not to be committed. The "why this route" layer makes a claim that is easy to
state and hard to check — *every sentence is built from a number the re-plan
actually computed* — and the only way to check it is to look at what a real
re-plan against the real Delhi graph actually emits. This script is that look.

It performs the sequence the dashboard performs and nothing else:

    load sample  ->  POST /scenarios                    (the 12/3/21 sample)
                     POST /scenarios/{id}/watcher       (QPSO solves, fleet drives)
    click a road ->  POST /scenarios/{id}/incident      (as `slow`, then as `closure`)
                     POST /scenarios/{id}/reoptimize    (the trace is in the response)

and prints, for each incident: the trigger's own sentence, the road report, every
vehicle's statements with their figures, and the convergence series the chart
plots. The road is resolved the way a *click* resolves it — a segment of a
route's drawn polyline, read back as nodes, with the reverse carriageway looked
up from the GeoJSON — so this exercises `frontend/lib/road-picking.ts`, not a
convenient stand-in.

Both incident kinds are shown on purpose. A **slow** has a before-time and an
after-time and can honestly say "a delay of Z minutes". A **closure** has no
after-time at all, so the same sentence would have to invent one; what it says
instead is the thing this layer exists to get right.

Usage
-----
    uv run python explain_demo.py
"""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from qgati.api.main import app

# Mirrors SAMPLE_SCENARIO_PAYLOAD / SAMPLE_SOLVER_SEED in frontend/lib/api.ts.
SAMPLE_PAYLOAD = {"kind": "generate", "n_deliveries": 12, "n_vehicles": 3, "seed": 21}
SAMPLE_SEED = 0

# One simulated second per tick, and the next tick an hour of wall clock away, so
# the fleet is genuinely running and genuinely still where it started. The demo
# would otherwise report a different trace depending on how fast the machine is.
INTERVAL_SECONDS = 300.0
TIME_SCALE = 1.0 / 300.0


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def heading(text: str) -> None:
    print()
    print(f"--- {text} " + "-" * max(0, 72 - len(text)))


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
        lookup[(feature["properties"]["u"], feature["properties"]["v"])] = feature[
            "geometry"
        ]["coordinates"]
    return lookup


def click_somewhere_on(routes: list[dict], nodes: dict, lines: dict) -> tuple[int, int] | None:
    """The road a click lands on, resolved the way `pickRoad` resolves it."""
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


def describe_figures(statement: dict) -> str:
    """The receipt line: the figures the sentence was built from, in order."""
    return "  ·  ".join(
        f"{figure['label']}={figure['value']:.4g} {figure['unit']}"
        for figure in statement["figures"]
    )


def print_explanation(explanation: dict | None) -> None:
    if explanation is None:
        print("  (no explanation on this response)")
        return

    heading("headline — the trigger's own sentence, passed through")
    print(f"  {explanation['headline']}")

    heading(f"roads — {len(explanation['roads'])} reported")
    for road in explanation["roads"]:
        print(f"  road {road['u']!r} -> {road['v']!r}")
        for statement in road["statements"]:
            print(f"    “{statement['text']}”")
            print(f"      {describe_figures(statement)}")

    heading(f"routes — {len(explanation['routes'])} vehicle(s)")
    for route in explanation["routes"]:
        print(f"  {route['vehicle_id']}  [{route['headline']}]")
        for index, statement in enumerate(route["statements"], start=1):
            print(f"    {index}. “{statement['text']}”")
            print(f"       {describe_figures(statement)}")
        print()


def print_convergence(convergence: list[float], solver: str) -> None:
    heading(f"convergence — {solver}, {len(convergence)} iteration(s)")
    if not convergence:
        print("  (this solver reported no history)")
        return
    ahead = ", ".join(f"{value:.2f}" for value in convergence[:4])
    tail = ", ".join(f"{value:.2f}" for value in convergence[-4:])
    print(f"  first 4   {ahead}")
    print(f"  last 4    {tail}")
    print(f"  best      ₹{convergence[-1]:.2f}")
    dropped = convergence[0] - min(convergence)
    print(f"  fell      ₹{dropped:.2f} from ₹{convergence[0]:.2f} at iteration 1")
    monotone = all(
        later <= earlier + 1e-9 for earlier, later in zip(convergence, convergence[1:])
    )
    print(f"  monotone  {monotone}  (best-so-far: the chart's y-series)")


def report(client: TestClient, sid: str, kind: str, street: list[tuple[int, int]]) -> None:
    """File one incident on every carriageway, re-plan, and print the trace."""
    banner(f"RE-PLAN after a {kind.upper()} on {street}")

    incidents = []
    for edge_u, edge_v in street:
        body = client.post(
            f"/scenarios/{sid}/incident",
            json={"incident_type": kind, "edge": {"u": edge_u, "v": edge_v}},
        ).json()
        incidents.append(body["incident"])
        print(
            f"  reported ({edge_u}, {edge_v})  applied={body['applied']}  "
            f"changed_legs={body['changed_legs']}"
        )

    response = client.post(f"/scenarios/{sid}/reoptimize", json={"seed": SAMPLE_SEED})
    print(f"  reoptimize -> {response.status_code}")
    if response.status_code != 200:
        print(f"  detail: {response.json().get('detail')}")
        return
    body = response.json()

    print(
        f"  solver={body['solver']}  moved={len(body['moved'])}/"
        f"{len(body['replanned'])}  vehicles={body['replanned_vehicles']}"
    )
    # The trace, as the API ships it. This is the payload the panel renders, and
    # the frontend composes no prose of its own from it.
    print()
    print(json.dumps(body["explanation"], indent=2, ensure_ascii=False))

    print()
    print_explanation(body["explanation"])
    print_convergence(body["convergence"], body["solver_name"])

    for incident in incidents:
        client.delete(f"/scenarios/{sid}/incident/{incident['incident_id']}")
    print(f"\n  reverted {len(incidents)} incident(s)")


def main() -> int:
    with TestClient(app) as client:
        banner("LOAD THE SAMPLE SCENARIO, AND DISPATCH THE FLEET")
        created = client.post("/scenarios", json=SAMPLE_PAYLOAD)
        created.raise_for_status()
        sid = created.json()["scenario_id"]

        started = client.post(
            f"/scenarios/{sid}/watcher",
            json={
                # No solver named: the production default, which is what the
                # dashboard gets and the only one whose convergence is a search.
                "seed": SAMPLE_SEED,
                "interval_seconds": INTERVAL_SECONDS,
                "time_scale": TIME_SCALE,
                "include_geometry": True,
            },
        )
        started.raise_for_status()
        dispatch = started.json()
        routes = dispatch["routes"]
        print(f"  scenario_id  {sid}")
        print(f"  solver       {dispatch['solver']} ({dispatch['solver_name']})")
        print(f"  routes       {len(routes)}")
        for route in routes:
            stops = " -> ".join(stop["delivery_id"] for stop in route["stops"])
            print(f"    {route['vehicle_id']}  {stops}")

        print()
        print_convergence(dispatch["convergence"], dispatch["solver_name"])

        graph = client.get(f"/graph/delhi?scenario_id={sid}").json()
        nodes, lines = node_index(graph), line_index(graph)
        clicked = click_somewhere_on(routes, nodes, lines)
        if clicked is None:
            print("no road on the dispatched routes resolved to a graph edge")
            return 1
        u, v = clicked
        street = both_directions(lines, u, v)
        print(f"\n  clicked road {u} -> {v}; carriageways {street}")

        # A slow first: it is the case the request's own example phrasing
        # describes, and the case where a before-time and an after-time both
        # exist to be quoted.
        report(client, sid, "slow", street)
        # Then a closure of the same street, which is the case that has no
        # after-time. Same code path, different truth, and the wording has to
        # differ with it or one of the two is a lie.
        report(client, sid, "closure", street)

    return 0


if __name__ == "__main__":
    sys.exit(main())
