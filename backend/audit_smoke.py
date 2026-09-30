#!/usr/bin/env python
"""TEMPORARY audit smoke test — not to be committed.

Walks the requested end-to-end sequence:
  create scenario -> optimize with QPSO -> block an edge a route uses
  -> reopt -> confirm the route changed and avoids that edge -> cost delta
  -> clear it.

It drives the real FastAPI app in-process over the real Delhi graph.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from qgati.api.main import app
from qgati.graph import load_delhi_graph


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def edges_of_route(route: dict, coord_to_node: dict) -> list[tuple]:
    """(u, v) road edges the drawn polyline walks along."""
    geometry = [tuple(point) for point in route["geometry"]]
    edges = []
    for start, end in zip(geometry, geometry[1:]):
        u = coord_to_node.get((round(start[0], 6), round(start[1], 6)))
        v = coord_to_node.get((round(end[0], 6), round(end[1], 6)))
        if u is not None and v is not None and u != v:
            edges.append((u, v))
    return edges


def main() -> int:
    graph = load_delhi_graph()
    print(f"graph: {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges")

    with TestClient(app) as client:
        banner("STEP 1 — create a scenario (moderate, generated)")
        created = client.post(
            "/scenarios",
            json={
                "kind": "generate",
                "n_deliveries": 12,
                "n_vehicles": 3,
                "seed": 21,
                "conditions": {"timestamp": "2026-09-21T14:00:00+05:30"},
            },
        )
        created.raise_for_status()
        scenario = created.json()
        sid = scenario["scenario_id"]
        print(f"scenario_id      {sid}")
        print(f"condition        {scenario['conditions']['traffic_condition']}")
        print(f"rows logged      {scenario['traffic_rows_logged']}")

        banner("STEP 2 — optimize with the production default (no solver named)")
        solved = client.post(f"/optimize/{sid}", json={"seed": 0})
        solved.raise_for_status()
        before = solved.json()
        print(f"solver           {before['solver']} ({before['solver_name']})")
        print(f"travel_cost      {before['travel_cost']:.1f} s")
        print(f"feasible         {before['feasible']}")
        for route in before["routes"]:
            stops = " -> ".join(stop["delivery_id"] for stop in route["stops"])
            print(f"  {route['vehicle_id']}: {stops or '(unused)'}"
                  f"   {route['travel_cost']:.1f} s")

        # Coordinate -> node lookup, the same join the frontend does.
        geojson = client.get("/graph/delhi",
                             params={"scenario_id": sid}).json()
        coord_to_node = {
            (round(f["geometry"]["coordinates"][0], 6),
             round(f["geometry"]["coordinates"][1], 6)): f["properties"]["id"]
            for f in geojson["features"]
            if f["geometry"]["type"] == "Point"
        }

        used: list[tuple] = []
        for route in before["routes"]:
            for edge in edges_of_route(route, coord_to_node):
                if edge not in used:
                    used.append(edge)
        print(f"\nedges on the solved routes: {len(used)}")

        banner("STEP 3 — block an edge that a route uses, then reopt")
        print("NOTE: probing for an incident endpoint. There is none — see below.")

        # The only way to apply a closure is to create a NEW scenario. Try
        # candidate edges until one is accepted and actually moves the route.
        chosen = None
        for candidate in used:
            payload = {
                "kind": "generate",
                "n_deliveries": 12,
                "n_vehicles": 3,
                "seed": 21,
                "conditions": {
                    "timestamp": "2026-09-21T14:00:00+05:30",
                    "closed_edges": [{"u": candidate[0], "v": candidate[1]}],
                },
            }
            attempt = client.post("/scenarios", json=payload)
            if attempt.status_code != 201:
                print(f"  edge {candidate} -> {attempt.status_code} "
                      f"(closure severs the instance); trying next")
                continue
            trial_sid = attempt.json()["scenario_id"]
            trial = client.post(f"/optimize/{trial_sid}",
                                json={"seed": 0, "include_geometry": False})
            trial.raise_for_status()
            body = trial.json()

            after_stops = [[s["delivery_id"] for s in r["stops"]]
                           for r in body["routes"]]
            before_stops = [[s["delivery_id"] for s in r["stops"]]
                            for r in before["routes"]]
            if after_stops != before_stops:
                chosen = (candidate, trial_sid, body, after_stops)
                break

        if chosen is None:
            print("  no candidate edge changed the route")
            return 1

        (edge, closed_sid, after, after_stops) = chosen
        print(f"\nblocked edge     {edge[0]} -> {edge[1]}")
        print(f"new scenario_id  {closed_sid}  (a NEW id — the original was not "
              f"mutated)")
        print(f"travel_cost      {after['travel_cost']:.1f} s")
        print(f"delta            {after['travel_cost'] - before['travel_cost']:+.1f} s "
              f"({100.0 * (after['travel_cost'] - before['travel_cost']) / before['travel_cost']:+.2f}%)")
        for route in after["routes"]:
            stops = " -> ".join(stop["delivery_id"] for stop in route["stops"])
            print(f"  {route['vehicle_id']}: {stops or '(unused)'}"
                  f"   {route['travel_cost']:.1f} s")

        print()
        print("route changed?   "
              + ("YES" if after_stops != before_stops else "NO"))

        banner("STEP 4 — does the new route avoid the blocked edge?")
        # Re-solve the closed scenario WITH geometry so the drawn line can be
        # walked back into edges.
        traced = client.post(f"/optimize/{closed_sid}",
                             json={"seed": 0, "include_geometry": True}).json()
        uses_blocked = False
        for route in traced["routes"]:
            if edge in edges_of_route(route, coord_to_node):
                uses_blocked = True
        print(f"blocked edge appears in the new geometry: {uses_blocked} "
              f"(expected False)")
        print(f"graph still has the edge (never deleted): {graph.has_edge(*edge)}")

        banner("STEP 5 — clear the incident")
        probe = client.post(f"/scenarios/{closed_sid}/incidents",
                            json={"clear": True})
        print(f"POST /scenarios/{{id}}/incidents -> {probe.status_code}")
        print(f"response: {probe.text[:200]}")
        print("\nThe original scenario's costs are immutable by design; "
              "'clearing' means creating yet another scenario.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
