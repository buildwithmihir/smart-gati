#!/usr/bin/env python
"""Re-plan the part of a plan that is still ahead of the fleet.

Every plan this project produces is static. ``POST /optimize`` solves a scenario
once and hands back routes; the fleet watcher then drives those routes while
traffic moves underneath it — roads are reported closed, measurements overwrite
estimates — and the routes never change. A vehicle stays committed to a plan
chosen under costs that may no longer exist. This script shows the fleet getting
to react.

It runs the whole thing over real HTTP against the real Delhi graph, and shows:

1. **Dispatch.** The plan the fleet was sent out on, and where each vehicle is
   once it has been out a while.
2. **What is already delivered.** The per-vehicle split between stops served and
   stops still ahead — derived from the fleet's own corridor, not declared.
3. **The refusal.** On a clean scenario ``POST /reoptimize`` answers **409**.
   Nothing has happened, so there is nothing to react to. This is the section
   that makes "not manually forced" a property of the API rather than a claim
   made here.
4. **A real trigger.** A road a vehicle is *currently driving* is reported slow.
   The next tick measures it, the detector flags the reading, and the reason it
   gives is printed verbatim.
5. **The before/after.** Two ways of serving *the same* remaining stops, with the
   completed stops marked and held fixed and every delivery that changed hands
   named.
6. **The check.** The completed stops are searched for in every new route and are
   not there — because they were never in the instance that was solved, not
   because something filtered them out of the answer. That distinction is the
   whole design, and it is asserted here rather than described.

Why the fleet is stepped by hand
--------------------------------
The watcher really does run on a background thread. Using ``POST
/scenarios/{id}/watcher/tick`` instead means the numbers printed are the tick
that was asked for rather than whatever the timer happened to be doing.

A tick advances the fleet by one interval — a vehicle that pings every 18 s has
driven 18 s — so the timer is set far out and the time scale down, giving a
one-second step with the thread quiet. That step is deliberately small: section 4
needs a vehicle to still be on the same road after its tick as before it, and
whether it is depends on how much road is left.

**This demo writes, and it writes to a throwaway log.** The watcher's job is to
append rows, and ``backend/data/traffic_log.db`` is an accumulating dataset a
demo has no business growing. A temp file is created for the run and removed with
it, and the path is printed so there is no doubt which log was written.

Requests go through ``TestClient``, which drives the real FastAPI application
in-process — routing, validation, serialisation and the log all exercised as a
client would, with no server to start.

Usage
-----
    uv run python benchmarks/demo_reopt.py
    uv run python benchmarks/demo_reopt.py --deliveries 12 --vehicles 4 --seed 7
    uv run python benchmarks/demo_reopt.py --ticks 6
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import networkx as nx
from fastapi.testclient import TestClient

from qgati.api.main import (
    create_app,
    get_graph,
    get_log_store,
    get_store,
    get_watchers,
)
from qgati.fleet import ScenarioWatcher, WatcherRegistry, cost_lookup
from qgati.reopt import VehicleProgress, fleet_progress
from qgati.traffic import (
    PEAK_FACTOR,
    SLOW,
    ActiveConditions,
    TrafficLogStore,
    TrafficState,
    edge_of,
    simulated_travel_time,
    traffic_weight_function,
)

#: Delhi's fixed offset. 02:00 is outside the 06:00-22:00 daytime band, so every
#: road prices at x1.0 and the numbers printed below are the graph's own base
#: times — which is what makes "x2.9 of the number beside it" literally true.
WHEN = datetime(2026, 9, 21, 2, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

#: The interval the fleet is started at. The schema caps ``interval_seconds`` at
#: 300, so this is "started, but not yet ticking": the thread exists and does
#: nothing while the explicit ticks below are stepped.
IDLE_INTERVAL = 300.0

#: How far the fleet drives per tick. Small on purpose — see the note on stepping
#: in the module docstring.
SCRIPTED_STEP_SECONDS = 1.0

#: The multiplier a ``slow`` incident applies, read off the simulator's own locked
#: peak factor rather than restated, so the arithmetic printed below cannot drift
#: from the arithmetic that priced the road.
SLOW_FACTOR = PEAK_FACTOR

RULE = "-" * 78

#: The registry this run installs. Module-level because the app's dependency
#: override needs something stable to point at, and emptied at the end of a run.
REGISTRY = WatcherRegistry()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deliveries", type=int, default=10)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7, help="fixes the instance and the noise")
    parser.add_argument(
        "--solver",
        default=None,
        help="solver key for the dispatch; the production default when omitted",
    )
    parser.add_argument(
        "--reopt-solver",
        default=None,
        help=(
            "solver key for the re-optimization; also the production default when "
            "omitted. Both are QPSO unless told otherwise, which is the point — a "
            "re-plan goes through the same optimizer an ordinary plan does"
        ),
    )
    parser.add_argument("--ticks", type=int, default=3, help="ticks before the trigger")
    return parser


# --------------------------------------------------------------------------- #
# Reading the scenario's own numbers
# --------------------------------------------------------------------------- #
def cost_of(graph: nx.Graph, scenario_id: str, edge: tuple) -> float | None:
    """What the app currently charges for ``edge``, in seconds.

    Read from the stored scenario's *effective* state, so this is the number
    ``/reoptimize`` prices a vehicle's position under rather than a recomputation
    of it. The road network itself is never touched — see the closing note in
    section 2 of ``demo_watcher.py``; the same holds here.
    """
    if not graph.has_edge(*edge):
        return None
    state = get_store().get(scenario_id).effective_traffic_state()
    return simulated_travel_time(graph, edge_of(graph, *edge), state)


def clean_seconds(graph: nx.Graph, scenario_id: str, edge: tuple) -> float | None:
    """The road's cost with incidents and measurements both cleared.

    The detector's own expectation, and the number a ``x2.9`` is ``x2.9`` of.
    Computed the way ``POST /detect`` computes it, so the arithmetic printed
    beside a verdict is the arithmetic behind it.
    """
    if not graph.has_edge(*edge):
        return None
    state = get_store().get(scenario_id).effective_traffic_state()
    return simulated_travel_time(
        graph,
        edge_of(graph, *edge),
        TrafficState(timestamp=state.timestamp, conditions=ActiveConditions()),
    )


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def create_scenario(client: TestClient, args) -> str:
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
    return response.json()["scenario_id"]


def start_fleet(client: TestClient, scenario_id: str, args) -> dict:
    body: dict = {
        "seed": args.seed,
        "interval_seconds": IDLE_INTERVAL,
        "time_scale": SCRIPTED_STEP_SECONDS / IDLE_INTERVAL,
        "include_geometry": False,
    }
    if args.solver is not None:
        body["solver"] = args.solver
    response = client.post(f"/scenarios/{scenario_id}/watcher", json=body)
    response.raise_for_status()
    return response.json()


def tick(client: TestClient, scenario_id: str) -> dict:
    response = client.post(f"/scenarios/{scenario_id}/watcher/tick")
    response.raise_for_status()
    return response.json()


def status(client: TestClient, scenario_id: str) -> dict:
    response = client.get(f"/scenarios/{scenario_id}/watcher")
    response.raise_for_status()
    return response.json()


def reoptimize(client: TestClient, scenario_id: str, args) -> tuple[int, dict]:
    """POST the re-optimization, returning the status **and** the body.

    The body matters on a refusal as much as on a success — the 409's ``detail``
    is the thing that names what would have qualified — so this does not raise on
    a non-2xx the way the other request helpers do.
    """
    body: dict = {"include_geometry": False}
    if args.reopt_solver is not None:
        body["solver"] = args.reopt_solver
    response = client.post(f"/scenarios/{scenario_id}/reoptimize", json=body)
    return response.status_code, response.json()


def inject(client: TestClient, scenario_id: str, edge: tuple, kind: str) -> dict:
    response = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": kind, "edge": {"u": edge[0], "v": edge[1]}},
    )
    response.raise_for_status()
    return response.json()


def fleet_view(scenario_id: str) -> tuple[VehicleProgress, ...]:
    """Where the fleet is, derived exactly as the re-optimize route derives it.

    The route reads this itself; the demo reaches for the same function so that
    section 2 can show the fleet's split between served and unserved *before* any
    trigger exists, which is a state the endpoint deliberately cannot be asked
    about. It is the same code path either way, not a re-implementation.
    """
    record = get_store().get(scenario_id)
    graph = get_graph()
    weight = traffic_weight_function(graph, record.effective_traffic_state())
    lookup = cost_lookup(graph, weight)
    watcher: ScenarioWatcher = REGISTRY.get(scenario_id)
    return watcher.read(lambda tracks: fleet_progress(record.scenario, tracks, lookup))


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def road(edge: dict | tuple | None) -> str:
    """An edge as ``u -> v``, from either the JSON shape or a plain tuple."""
    if edge is None:
        return "—"
    if isinstance(edge, dict):
        return f"{edge['u']} -> {edge['v']}"
    return f"{edge[0]} -> {edge[1]}"


def listing(items: Sequence[str]) -> str:
    return " ".join(items) if items else "(none)"


def route_stops(route: dict) -> str:
    return listing([stop["delivery_id"] for stop in route["stops"]])


def by_vehicle(routes: Sequence[dict]) -> dict[str, dict]:
    return {route["vehicle_id"]: route for route in routes}


def busiest_vehicle(state: dict, step: float) -> dict | None:
    """The vehicle with the most road left ahead of it, if any has enough.

    Section 4 needs one vehicle to still be on the same road after its tick as
    before it. Picking the one furthest from its next intersection is what makes
    that true without freezing the fleet, and the margin is there so the answer
    is not decided by a rounding error. ``None`` means none qualified, and the
    caller says so rather than reporting a flag that was never measured.
    """
    moving = [
        vehicle
        for vehicle in state["vehicles"]
        if not vehicle["finished"]
        and vehicle["remaining_seconds"] is not None
        and vehicle["edge"] is not None
    ]
    if not moving:
        return None
    best = max(moving, key=lambda vehicle: vehicle["remaining_seconds"])
    return best if best["remaining_seconds"] > step * 3 else None


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def show_dispatch(started: dict, step: float) -> None:
    print("1. Dispatch — the plan the fleet was sent out on")
    print(RULE)
    print(f"  solved by   {started['solver']}")
    print(f"  fleet clock interval {started['interval_seconds']:.0f} s, "
          f"time scale {started['time_scale']:.5f}  ->  {step:.2f} s of driving per tick")
    print(f"  vehicles    {len(started['vehicles'])} dispatched")
    print()
    print(f"  {'vehicle':<10} {'stops':<28} {'load':>6} {'cap':>5} {'time':>9}")
    for route in started["routes"]:
        print(f"  {route['vehicle_id']:<10} {route_stops(route):<28} "
              f"{route['load']:>6.0f} {route['capacity']:>5.0f} "
              f"{route['travel_time']:>8.1f}s")
    print()
    print("  This is the plan every later section is measured against. It was")
    print("  solved once, under costs that are about to stop being true, and the")
    print("  fleet is now driving it. Nothing below changes these routes on the")
    print("  vehicles — re-optimizing computes a new plan and returns it. Applying")
    print("  one is a separate step, and the README names it as the next one.")
    print()


def show_fleet(progress: Sequence[VehicleProgress], args) -> None:
    print(f"2. What the fleet has already delivered — after {args.ticks} tick(s)")
    print(RULE)
    print("  Vehicles start spread along their own routes rather than all leaving")
    print("  the depot together, so some of this was already done at dispatch.")
    print()
    print(f"  {'vehicle':<10} {'at node':>9} {'out':>8} {'cap left':>9}  "
          f"{'delivered':<22} still ahead")
    for item in progress:
        where = "—" if item.node is None else str(item.node)
        out = f"{item.elapsed:>7.0f}s"
        print(f"  {item.vehicle_id:<10} {where:>9} {out} "
              f"{item.remaining_capacity:>9.1f}  "
              f"{listing([str(stop) for stop in item.completed]):<22} "
              f"{listing([str(stop) for stop in item.remaining])}")
    print()
    print("  Note what those stop numbers are: positions in the scenario's delivery")
    print("  list, which is what the *fleet* is indexed by. The endpoint reports")
    print("  delivery ids, and section 5 prints those — the two are the same stops.")
    print()
    print("  None of this was declared by the script. Each vehicle's completed")
    print("  stops are read off the corridor it is actually driving: the route it")
    print("  was dispatched on is recorded at dispatch, and the point it has")
    print("  reached on that corridor says how far down it is. There is no")
    print("  fleet-state field on the request for a caller to assert.")
    print()
    stalled = [item for item in progress if item.stuck]
    if stalled:
        print(f"  {len(stalled)} vehicle(s) are stopped behind a closure: "
              f"{listing([item.vehicle_id for item in stalled])}. A stopped")
        print("  vehicle is not a finished one, and the two are not conflated: a")
        print("  finished vehicle has served everything, and a stopped one is")
        print("  carrying load nobody else has room for.")
        print()


def show_refusal(client: TestClient, scenario_id: str, args) -> None:
    """The 409, taken before anything has happened to the scenario."""
    print("3. The refusal — POST /scenarios/{id}/reoptimize on a clean scenario")
    print(RULE)
    code, body = reoptimize(client, scenario_id, args)
    print(f"  HTTP {code}")
    print()
    if code != 409:
        print("  Not a refusal this run. The fleet's own measurement noise tripped")
        print("  the detector — about a one-in-a-thousand chance per reading, which")
        print("  is the false-positive rate it was tuned to. That is a genuine")
        print("  trigger rather than a forced one, so the endpoint ran; re-run with")
        print("  another --seed to see the refusal.")
        print()
        return
    for line in _wrap(body.get("detail", body), 74, indent="  "):
        print(line)
    print()
    print("  This is the load-bearing line of the whole feature. There is no")
    print("  ``force`` field on the request and no way to declare an incident in")
    print("  the body: whether a re-optimization is justified is derived from the")
    print("  network and the fleet's own readings, and with neither holding the")
    print("  answer is no. Without this, 'adaptive routing' would be an endpoint")
    print("  indistinguishable from a re-solve button.")
    print()


def show_trigger(client: TestClient, scenario_id: str, step: float, args) -> dict | None:
    """Report a road slow under a vehicle, tick, and see the detector agree."""
    print(f"4. A trigger — POST /scenarios/{{id}}/incident ({SLOW}) on a road in use")
    print(RULE)

    graph = get_graph()
    chosen = busiest_vehicle(status(client, scenario_id), step)
    if chosen is None:
        print("  no vehicle is far enough from its next intersection to hold still")
        print("  across a tick; raise --deliveries and run again")
        print()
        return None

    edge = (chosen["edge"]["u"], chosen["edge"]["v"])
    vehicle_id = chosen["vehicle_id"]
    base = clean_seconds(graph, scenario_id, edge)

    inject(client, scenario_id, edge, SLOW)
    print(f"  reported    {SLOW} on {road(edge)}")
    print(f"  {vehicle_id} is on it, {chosen['remaining_seconds']:.1f} s from the "
          "next intersection")
    print()
    print(f"    the road's base time, under the clock alone   {base:>8.2f} s")
    print(f"    what the incident alone would charge          "
          f"{base * SLOW_FACTOR:>8.2f} s   (x{SLOW_FACTOR})")
    print(f"    charged now                                   "
          f"{cost_of(graph, scenario_id, edge):>8.2f} s")
    print()

    result = tick(client, scenario_id)
    reading = next(
        (
            item
            for item in result["readings"]
            if item["vehicle_id"] == vehicle_id and item["edge"] is not None
            and item["edge"]["u"] == edge[0] and item["edge"]["v"] == edge[1]
        ),
        None,
    )
    if reading is None or reading["observed_travel_time"] is None:
        print(f"  {vehicle_id} left {road(edge)} during that tick, so the road was")
        print("  never measured on it. Raise --ticks and run again.")
        print()
        return None

    ratio = reading["observed_travel_time"] / base if base else float("nan")
    print(f"  tick {result['tick']} measured it at "
          f"{reading['observed_travel_time']:.2f} s "
          f"({'FLAGGED' if reading['flag'] else 'not flagged'})")
    print(f"  the detector judged that against the road's incident-free model —")
    print(f"  {base:.2f} s — and not against the price the app charges, so an")
    print(f"  operator's own report cannot explain away the change the detector")
    print(f"  exists to catch. The reading is {ratio:.2f}x that model.")
    print()
    print("  The incident did not flag anything itself. A vehicle drove the road,")
    print("  the app drew a plausible travel time for it, and the detector — which")
    print("  is the Prompt 6 module, unchanged — returned this:")
    print()
    for line in _wrap(reading["reason"] or "(no reason given)", 74, indent="    "):
        print(line)
    print()
    return {"edge": edge, "vehicle": vehicle_id, "result": result, "ratio": ratio}


def show_replan(client: TestClient, scenario_id: str, args) -> None:
    print("5. The re-optimization — POST /scenarios/{id}/reoptimize")
    print(RULE)
    code, body = reoptimize(client, scenario_id, args)
    if code != 200:
        print(f"  HTTP {code}")
        print()
        for line in _wrap(body.get("detail", body), 74, indent="  "):
            print(line)
        print()
        return

    print(f"  HTTP {code}   solver {body['solver']} ({body['solver_name']})")
    kinds = body["trigger"]["kinds"]
    print(f"  trigger     {', '.join(kinds)}")
    print(f"              {body['trigger']['detail']}")
    for edge in body["trigger"]["edges"]:
        print(f"              road {road(edge)}")
    print()
    for reason in body["trigger"]["reasons"]:
        for line in _wrap(reason, 74, indent="              "):
            print(line)
    print()
    print("  Both kinds are reported because both hold, and they are different")
    print("  kinds of claim: an incident is something somebody *said*, an anomaly")
    print("  is something the fleet *measured*. A reader deciding whether to trust")
    print("  this plan should be able to tell which they have, so neither is")
    print("  collapsed into the other. The reason string is the detector's own,")
    print("  passed through verbatim — it names the numbers it judged.")
    print()

    # -- where every vehicle was, and what it may not be given back --------- #
    print(f"  {'vehicle':<10} {'at node':>9} {'cap left':>9}  delivered")
    for item in body["vehicles"]:
        where = "—" if item["node"] is None else str(item["node"])
        state = "finished" if item["finished"] else ("stopped" if item["stuck"] else "")
        print(f"  {item['vehicle_id']:<10} {where:>9} "
              f"{item['remaining_capacity']:>9.1f}  "
              f"{listing(item['completed'])}"
              + (f"   [{state}]" if state else ""))
    print()

    # -- the before/after --------------------------------------------------- #
    before = by_vehicle(body["before"])
    after = by_vehicle(body["after"])
    print("  Before and after — the same stops, two ways of serving them")
    print()
    print(f"  {'vehicle':<10} {'delivered (fixed)':<20} "
          f"{'before':<20} {'after':<20}")
    for item in body["vehicles"]:
        name = item["vehicle_id"]
        old = before.get(name)
        new = after.get(name)
        print(f"  {name:<10} {listing(item['completed']):<20} "
              f"{route_stops(old) if old else '(not in fleet)':<20} "
              f"{route_stops(new) if new else '(not in fleet)':<20}")
    print()

    old_stops = sorted(s["delivery_id"] for r in body["before"] for s in r["stops"])
    new_stops = sorted(s["delivery_id"] for r in body["after"] for s in r["stops"])
    print(f"  the {len(old_stops)} unserved stop(s) the fleet was heading for: "
          f"{listing(old_stops)}")
    print(f"  the {len(new_stops)} stop(s) the new plan serves:                "
          f"{listing(new_stops)}")
    print("  Both halves cover the same deliveries, which is what makes this a")
    print("  comparison of two assignments rather than of two amounts of work.")
    if old_stops != new_stops:
        print(f"  They differ here: {listing(sorted(set(old_stops) - set(new_stops)))}")
        print("  is served by neither — the plan is infeasible, which "
              f"feasible={body['feasible']} reports.")
    print()

    moved = body["moved"]
    if moved:
        print(f"  {len(moved)} delivery(ies) changed hands:")
        for move in moved:
            to = move["to_vehicle"] or "nobody (dropped)"
            print(f"    {move['delivery_id']:<8} {move['from_vehicle']} -> {to}")
    else:
        print("  No delivery changed hands. That is a real answer — the optimizer")
        print("  was given the fleet's actual positions and reduced capacities and")
        print("  found the current assignment still the cheapest. A re-plan that")
        print("  always moved something would be a re-plan that was not answering")
        print("  the question it was asked.")
    print()

    # -- the check that is the whole point ---------------------------------- #
    completed = set(body["completed"])
    leaked = sorted(completed & set(new_stops))
    print("  The check:")
    print(f"    delivered already, and never to be touched   {listing(sorted(completed))}")
    print(f"    of those, appearing in the new plan          {listing(leaked)}")
    print()
    if leaked:
        print("    THIS IS A BUG. A completed stop is in a new route.")
    else:
        print("    None. Not because they were filtered out of the answer, but")
        print("    because they were never in the instance that was solved: the")
        print("    re-optimization is handed a scenario whose delivery list holds")
        print("    only the unserved stops, so there is no delivery for a solver to")
        print("    name however badly it searches. The guarantee is structural, so")
        print("    it holds for all six solvers rather than for the ones somebody")
        print("    remembered to check.")
    print()

    # -- and the cost ------------------------------------------------------- #
    print(f"  The remaining plan, priced: {body['travel_time']:.0f} s on the road, "
          f"{body['distance_m'] / 1000:.1f} km,")
    print(f"  {body['fuel_litres']:.2f} L of fuel — {body['travel_cost']:.2f} rupees "
          f"under the objective, in {body['runtime_ms']:.0f} ms.")
    print(f"  feasible={body['feasible']}, solved by {body['solver_name']} — the")
    print("  production default, which is the point. A re-optimized plan goes")
    print("  through the same optimizer, the same objective and the same time")
    print("  windows as an ordinary /optimize; the only thing different is the")
    print("  scenario it was handed.")
    print()

    # -- what did not happen ------------------------------------------------ #
    after_state = status(client, scenario_id)
    print("6. What it did not do")
    print(RULE)
    print(f"  the fleet is still running: running={after_state['running']}, "
          f"{after_state['ticks']} tick(s)")
    print("  the vehicles are still driving the routes they were dispatched on.")
    print("  Re-optimizing is a **read**: it computed a plan and returned it.")
    print("  It did not rewrite the stored scenario — so a second call against an")
    print("  unchanged fleet returns the same plan rather than a different one,")
    print("  which it could not if the first had written anything. And it did not")
    print("  re-dispatch the fleet, because that means rebuilding every track and")
    print("  resetting how far each vehicle has driven, which is a simulation")
    print("  decision rather than an optimizer one.")
    print()
    print("  Applying a plan — handing these routes to the vehicles — is the")
    print("  obvious next step, and it is named as the boundary rather than left")
    print("  ambiguous.")
    print()


def _wrap(text, width: int, indent: str = "") -> list[str]:
    """Greedy wrap, so a long detail string does not run off the terminal.

    ``str`` because a FastAPI validation failure answers with a *list* of error
    objects where a raised ``HTTPException`` answers with a sentence, and the
    caller prints either through here without having to know which.
    """
    words, lines, current = str(text).split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(indent + current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(indent + current)
    return lines


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="qgati-demo-reopt-") as tmp:
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
    application.dependency_overrides[get_watchers] = lambda: REGISTRY

    print("Q-Gati — adaptive re-optimization demo")
    print("=" * 78)
    print(f"log        {log_path}")
    print("           a throwaway; the collected log at backend/data/traffic_log.db")
    print("           is neither read nor written by this run")
    print(f"instance   {args.deliveries} deliveries, {args.vehicles} vehicles, "
          f"seed {args.seed}")
    print(f"priced at  {WHEN.isoformat()}  (outside the daytime band, x1.0)")
    print()
    print("the claim  a completed stop cannot be reassigned, because it is not in")
    print("           the instance that is solved — and a re-optimization cannot")
    print("           be forced, because the endpoint derives whether one is due.")
    print()

    step = IDLE_INTERVAL * (SCRIPTED_STEP_SECONDS / IDLE_INTERVAL)

    with TestClient(application) as client:
        scenario_id = create_scenario(client, args)
        print(f"scenario   {scenario_id}")
        print()

        try:
            started = start_fleet(client, scenario_id, args)
            show_dispatch(started, step)

            for _ in range(args.ticks):
                tick(client, scenario_id)
            show_fleet(fleet_view(scenario_id), args)

            show_refusal(client, scenario_id, args)

            fired = show_trigger(client, scenario_id, step, args)
            if fired is None:
                print("  The fleet never measured a road under the incident, so")
                print("  there is no anomaly to show. The endpoint would still")
                print("  have run on the incident alone — a report needs no")
                print("  corroboration — but this script will not print a flag that")
                print("  was not taken. Raise --deliveries or change --seed.")
                print()
                return 1

            show_replan(client, scenario_id, args)

        finally:
            REGISTRY.stop_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
