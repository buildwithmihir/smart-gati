#!/usr/bin/env python
"""Watch a simulated GPS fleet measure the roads its vehicles are driving.

This is the fleet-telemetry demonstration. There is no real fleet in this
project — no GPS device, no telemetry protocol, no vehicle — so
:mod:`qgati.fleet` stands in for one, and this script drives it over real HTTP
against the real Delhi graph and shows the things that were asked for:

1. **Dispatch.** ``POST /scenarios/{id}/watcher`` solves the scenario and puts
   the fleet on the road, spread along its own routes rather than all leaving
   the depot together.
2. **Ticking.** Each tick places every vehicle on the road it is currently on,
   draws a *plausible* travel time for it — the modelled time plus noise, not a
   replay of it — and reports what it read.
3. **The cost actually moving.** The charge the application makes for a road is
   replaced by the fleet's measurement of it. The road network object itself is
   never touched; it is the *price* that moves.
4. **Anomaly detection firing, and the two tiers.** With an incident injected on
   a road a vehicle is driving, the next tick flags it. The same section shows
   why the incident's flat ``x2.9`` does not price that road — the fleet had
   already measured it, and a measurement outranks a placeholder — and shows the
   placeholder in force on a road nothing has measured.

The last section reads the traffic log back and shows what the fleet wrote: one
row per measured road, carrying the *measured* seconds rather than the estimate
they replaced, and carrying the incident word when one was in force — which is
what keeps an incident-time reading out of a later baseline.

The scenario is created at 02:00, outside the 06:00-22:00 daytime band the
simulator applies congestion in. Every road therefore prices at its own base
time, which is what makes the arithmetic printed below exact rather than
approximately right: a ``x2.9`` placeholder is ``x2.9`` of the number beside it.

Why the ticks are stepped explicitly
------------------------------------
The watcher really does run on a background thread, and the last section starts
it at the interval you ask for and lets it tick on its own clock. Every section
before that uses ``POST /scenarios/{id}/watcher/tick`` instead, so what is
printed is exactly the tick that was asked for rather than whatever the timer
happened to be doing.

A tick also advances the fleet by one interval — a vehicle that pings every 18 s
has driven 18 s — so the scripted sections set the timer far out and the time
scale down, to get a one-second step while the thread stays quiet. Section 5
restarts the fleet at real time.

**This demo writes, and it writes to a throwaway log.** The watcher's whole job
is to append rows, and the real ``backend/data/traffic_log.db`` is an
accumulating dataset that a demo has no business growing. A temp file is created
for the run and removed with it, and the path is printed so there is no doubt
which log was written.

Requests go through ``TestClient``, which drives the real FastAPI application
in-process — so there is no server to start, while routing, validation,
serialisation and the log are all exercised exactly as a client would.

Usage
-----
    uv run python benchmarks/demo_watcher.py
    uv run python benchmarks/demo_watcher.py --deliveries 12 --vehicles 4 --seed 7
    uv run python benchmarks/demo_watcher.py --interval 1 --live-seconds 8
"""

from __future__ import annotations

import argparse
import tempfile
import time
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
from qgati.fleet import WatcherRegistry
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
#: road prices at x1.0 and the numbers printed here are the graph's own base
#: times — which is what makes "x2.9 of the number beside it" literally true.
WHEN = datetime(2026, 9, 21, 2, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

#: The interval the scripted sections start the fleet at. The schema caps
#: ``interval_seconds`` at 300, so this is "started, but not yet ticking" — the
#: thread exists and does nothing while the explicit ticks below are stepped.
#: Without it a short ``--interval`` would have the timer firing between the
#: sections and the counts printed here would depend on how fast the machine is.
IDLE_INTERVAL = 300.0

#: How far the scripted fleet drives per tick. Small on purpose: the before/after
#: comparison below needs a vehicle to still be on the same road after a tick as
#: before it, and that is decided by whether the road has more than this much
#: left to run.
SCRIPTED_STEP_SECONDS = 1.0

#: The multiplier a ``slow`` incident applies. Read off the simulator's own
#: locked peak factor rather than restated, so the arithmetic printed below
#: cannot drift from the arithmetic that priced the road.
SLOW_FACTOR = PEAK_FACTOR

RULE = "-" * 78

#: The registry this run installs. A module-level object because the app's
#: dependency override needs something stable to point at, and emptied at the
#: end of every run.
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
    parser.add_argument("--ticks", type=int, default=3, help="scripted ticks in section 2")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between ticks for the live section",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="seconds of driving per tick for the live section",
    )
    parser.add_argument(
        "--live-seconds",
        type=float,
        default=6.0,
        help="how long to let the background thread run; 0 skips the live section",
    )
    return parser


# --------------------------------------------------------------------------- #
# Reading the scenario's own numbers
# --------------------------------------------------------------------------- #
def _state(scenario_id: str) -> TrafficState:
    """The scenario's effective state — incidents and measurements folded in."""
    return get_store().get(scenario_id).effective_traffic_state()


def cost_of(graph: nx.Graph, scenario_id: str, edge: tuple) -> float | None:
    """What the app currently charges for ``edge``, in seconds.

    Read from the stored scenario's *effective* state, so this is the number
    ``/optimize`` would search under rather than a recomputation of it.

    The road network is not touched. Everything the fleet does to the app's
    prices goes through this state, which is why the graph object can be read
    alongside and found unchanged.
    """
    if not graph.has_edge(*edge):
        return None
    return simulated_travel_time(graph, edge_of(graph, *edge), _state(scenario_id))


def measured_before(scenario_id: str, edge: tuple) -> float | None:
    """The fleet's last measurement of ``edge``, if it has made one."""
    observation = _state(scenario_id).observation(*edge)
    return None if observation is None else observation.travel_time


def clean_seconds(graph: nx.Graph, scenario_id: str, edge: tuple) -> float | None:
    """The road's cost with incidents and measurements both cleared.

    The detector's own expectation, and the number a ``x2.9`` is ``x2.9`` of.
    Computed the same way ``POST /scenarios/{id}/detect`` computes it, so the
    arithmetic printed beside a verdict is the arithmetic behind it.
    """
    if not graph.has_edge(*edge):
        return None
    state = _state(scenario_id)
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


def start_fleet(
    client: TestClient, scenario_id: str, args, *, interval: float, time_scale: float
) -> dict:
    body: dict = {
        "seed": args.seed,
        "interval_seconds": interval,
        "time_scale": time_scale,
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


def stop_fleet(client: TestClient, scenario_id: str) -> dict:
    response = client.delete(f"/scenarios/{scenario_id}/watcher")
    response.raise_for_status()
    return response.json()


def inject(client: TestClient, scenario_id: str, edge: tuple, kind: str) -> dict:
    response = client.post(
        f"/scenarios/{scenario_id}/incident",
        json={"incident_type": kind, "edge": {"u": edge[0], "v": edge[1]}},
    )
    response.raise_for_status()
    return response.json()


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


def as_tuple(edge: dict | None) -> tuple | None:
    return None if edge is None else (edge["u"], edge["v"])


def same_edge(edge: dict | None, other: tuple) -> bool:
    return edge is not None and edge["u"] == other[0] and edge["v"] == other[1]


def reading_for(result: dict, vehicle_id: str, edge: tuple) -> dict | None:
    """The reading this tick took of ``edge`` by ``vehicle_id``, if it took one."""
    return next(
        (
            item
            for item in result["readings"]
            if item["vehicle_id"] == vehicle_id and same_edge(item["edge"], edge)
        ),
        None,
    )


def print_readings(result: dict) -> None:
    """One tick's table: where each vehicle was, and what it read there."""
    details = "   [incident live]" if result["conditions_mutated"] else ""
    print(f"  tick {result['tick']}  at {result['at']}   "
          f"{result['changed_legs']} leg(s) repriced, "
          f"{result['rows_logged']} row(s) logged{details}")
    print(f"    {'vehicle':<10} {'road':<20} {'modelled':>9} {'observed':>9}   verdict")
    for reading in result["readings"]:
        observed = reading["observed_travel_time"]
        if observed is None:
            print(f"    {reading['vehicle_id']:<10} {road(reading['edge']):<20} "
                  f"{'—':>9} {'—':>9}   {reading['note']}")
            continue
        verdict = "FLAGGED" if reading["flag"] else "not flagged"
        print(f"    {reading['vehicle_id']:<10} {road(reading['edge']):<20} "
              f"{reading['modelled_travel_time']:>9.2f} {observed:>9.2f}   {verdict}")
    print()


def busiest_vehicle(state: dict, step: float) -> dict | None:
    """The vehicle with the most road left ahead of it, if any has enough.

    The before/after comparison needs one vehicle to still be on the same road
    after a tick as before it. Picking the one furthest from its next
    intersection is what makes that true without freezing the fleet, and the
    margin is there so the answer is not decided by a rounding error. ``None``
    means no vehicle qualified, and the caller says so rather than printing a
    comparison that quietly compared two different roads.
    """
    moving = [
        vehicle
        for vehicle in state["vehicles"]
        if not vehicle["finished"] and vehicle["remaining_seconds"] is not None
    ]
    if not moving:
        return None
    best = max(moving, key=lambda vehicle: vehicle["remaining_seconds"])
    return best if best["remaining_seconds"] > step * 3 else None


def ahead_of(graph: nx.Graph, scenario_id: str, start: tuple, target) -> tuple | None:
    """The first hop of the cheapest path from ``start`` to ``target``.

    Used to name a road the fleet will drive but has not reached — one no
    vehicle has reported on yet, which is the only state in which an incident's
    ``x2.9`` placeholder still prices a road. Traced under the scenario's own
    live weight function, so it is a road this instance would really take
    rather than one picked at random and attached to it.
    """
    weight = traffic_weight_function(graph, _state(scenario_id))
    try:
        path = nx.shortest_path(graph, start, target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return None if len(path) < 2 else (path[0], path[1])


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def show_dispatch(started: dict, step: float) -> None:
    print("1. Dispatch — POST /scenarios/{id}/watcher")
    print(RULE)
    print(f"  solved by   {started['solver']}")
    print(f"  fleet clock interval {started['interval_seconds']:.0f} s, "
          f"time scale {started['time_scale']:.5f}  ->  {step:.2f} s of driving per tick")
    print(f"  vehicles    {len(started['vehicles'])} dispatched")
    print()
    print(f"  {'vehicle':<10} {'start road':<20} {'remaining':>10} {'along':>7}")
    for vehicle in started["vehicles"]:
        print(f"  {vehicle['vehicle_id']:<10} {road(vehicle['edge']):<20} "
              f"{vehicle['remaining_seconds']:>9.1f}s {vehicle['progress']:>6.0%}")
    print()
    print("  The routes themselves, as solved:")
    for route in started["routes"]:
        # Node ids are integers on the Delhi graph, so they are cast to str on the
        # way into the join — the same thing demo_incident.py's route printer does.
        stops = " ".join(str(stop["node"]) for stop in route["stops"]) or "(none)"
        print(f"    {route['vehicle_id']:<10} {route['travel_time']:>8.1f}s  "
              f"{route['distance_m']:>8.0f} m  {stops}")
    print()
    print("  Vehicles start spread along their own routes — vehicle i of n at")
    print("  (i+1)/(n+1) of the way — because a fleet that all left the depot at")
    print("  once would report on the same first road three times.")
    print()
    first = started["first_tick"]
    print(f"  The first tick ran inline, so this response already carries it: "
          f"{len(first['readings'])} reading(s),")
    print(f"  {first['changed_legs']} leg(s) repriced, {first['rows_logged']} row(s) "
          "logged. Every road those vehicles were on has")
    print("  just been measured for the first time, and their prices are now the")
    print("  measurements rather than the model's estimates of them.")
    print()


def show_ticks(
    client: TestClient, scenario_id: str, args, step: float
) -> tuple[list[dict], tuple | None, str | None]:
    """Step the fleet, showing on the first step what happened to a road's price."""
    print(f"2. {args.ticks} ticks — POST /scenarios/{{id}}/watcher/tick")
    print(RULE)
    print()

    graph = get_graph()
    results: list[dict] = []
    compared: tuple | None = None
    compared_vehicle: str | None = None

    for number in range(args.ticks):
        snapshot = None
        prior = None
        if number == 0:
            state = status(client, scenario_id)
            chosen = busiest_vehicle(state, step)
            if chosen is not None:
                candidate = as_tuple(chosen["edge"])
                # Both read before the tick, so "before" is the state the tick
                # started from rather than the one it produced.
                snapshot = cost_of(graph, scenario_id, candidate)
                prior = measured_before(scenario_id, candidate)
                compared = candidate
                compared_vehicle = chosen["vehicle_id"]

        result = tick(client, scenario_id)
        results.append(result)
        print_readings(result)

        if snapshot is None or compared is None or compared_vehicle is None:
            continue
        reading = reading_for(result, compared_vehicle, compared)
        if reading is None:
            # The vehicle crossed onto another road during the tick. Said out
            # loud rather than compared anyway: a before/after across two
            # different roads would look like a price change and not be one.
            print(f"  {compared_vehicle} left {road(compared)} during that tick, so")
            print("  there is no before/after to show for it; the comparison needs a")
            print("  vehicle with more road left ahead of it. Raise --deliveries.")
            print()
            continue

        print(f"  What one of those readings did to a price — "
              f"{compared_vehicle} on {road(compared)}:")
        print(f"    the model's own estimate, no measurement    "
              f"{reading['modelled_travel_time']:>8.2f} s")
        print(f"    charged before this tick                    {snapshot:>8.2f} s"
              + ("   (already a measurement, from the first tick)"
                 if prior is not None else "   (the model's estimate)"))
        print(f"    this tick's measurement                     "
              f"{reading['observed_travel_time']:>8.2f} s")
        print(f"    charged after this tick                     "
              f"{cost_of(graph, scenario_id, compared):>8.2f} s")
        print()
        print("    The charge followed the measurement, not the model. Note the first")
        print("    two lines: the road's estimated time and the price it is charged")
        print("    differ, because a vehicle drove it and reported what it actually")
        print("    took. That difference is the entire content of Tier 1.")
        print()

    print("  The road did not move; the price did. The Delhi graph is a cached,")
    print("  shared, read-only object, and writing a price into it would leak the")
    print("  change into every later request. Every price in this project is a")
    print("  function of a TrafficState, so a measurement is applied by putting it")
    print("  on the state and repricing — which is also why the graph compares")
    print("  byte-identical before and after a tick.")
    print()
    print("  Each reading is drawn around the road's modelled time with noise, from")
    print("  a lognormal whose mean multiplier is exactly 1.0 — so the fleet is")
    print("  unbiased about a road while every individual reading differs from the")
    print("  last. That spread is the entire reason a standard deviation can be")
    print("  computed from this table at all: a rule-based row is")
    print("  base_travel_time x multiplier, both deterministic, so every sample of")
    print("  one road in one band is the same number and its deviation is zero.")
    print()
    return results, compared, compared_vehicle


def show_anomaly(
    client: TestClient,
    scenario_id: str,
    step: float,
    compared: tuple | None,
    compared_vehicle: str | None,
) -> tuple[dict, tuple | None]:
    """Inject a slow incident on a road a vehicle is driving, then tick."""
    print(f"3. An incident on a road the fleet is driving — "
          f"POST /scenarios/{{id}}/incident ({SLOW})")
    print(RULE)

    graph = get_graph()
    state = status(client, scenario_id)
    chosen = busiest_vehicle(state, step)
    if chosen is None:
        print("  no vehicle is far enough from its next intersection to hold still")
        print("  across a tick; raise --deliveries and run again")
        print()
        return {}, None

    edge = as_tuple(chosen["edge"])
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
    charged = cost_of(graph, scenario_id, edge)
    print(f"    charged now                                   {charged:>8.2f} s")
    print()
    if charged is not None and base is not None and charged < base * SLOW_FACTOR * 0.95:
        print("    The placeholder is not pricing this road, and that is the design")
        print("    rather than an oversight. x2.9 is a stand-in for a delay nobody")
        print("    has measured; this road has been measured, and the two tiers are")
        print("    ranked rather than blended — a measurement outranks a placeholder,")
        print("    which is what 'Tier 1' means. An incident's job on a road the")
        print("    fleet already covers is to say where to look, not what to charge.")
    else:
        print("    The placeholder is pricing this road, because nothing has measured")
        print("    it yet: it stands until real data arrives.")
    print()

    result = tick(client, scenario_id)
    print_readings(result)

    reading = reading_for(result, vehicle_id, edge)
    if reading is None:
        print(f"  {vehicle_id} left {road(edge)} during that tick, so nothing was")
        print("  measured on its road; raise --ticks and run again.")
        print()
        return result, edge

    after = cost_of(graph, scenario_id, edge)
    ratio = reading["observed_travel_time"] / base if base else float("nan")
    print(f"  The detector fired. It judged the reading against the road's")
    print(f"  incident-free model — {base:.2f} s — and not against the price the app")
    print(f"  charges, so an operator's own report cannot explain away the change the")
    print(f"  detector exists to catch. The reading is {ratio:.2f}x that model, far")
    print("  past the 1.2x margin the fallback rule allows.")
    print()
    print(f"    {reading['reason']}")
    print()
    print(f"  The price moved too, this time: {charged:.2f} -> {after:.2f} s. Not")
    print("  because of the incident, but because a vehicle drove the road while the")
    print("  incident was on and measured it. A road ends up priced at what the")
    print("  fleet took to drive it; the incident only says where to look.")
    print()

    # -- the other half of the tier rule ----------------------------------- #
    ahead = ahead_of(
        graph, scenario_id, edge[1], get_store().get(scenario_id).scenario.depot.node
    )
    if ahead is not None and measured_before(scenario_id, ahead) is None:
        inject(client, scenario_id, ahead, SLOW)
        ahead_base = clean_seconds(graph, scenario_id, ahead)
        ahead_cost = cost_of(graph, scenario_id, ahead)
        print(f"  For contrast, a road nothing has measured — {road(ahead)}, the next")
        print("  one along from where that vehicle is:")
        print()
        print(f"    base time                {ahead_base:>8.2f} s")
        print(f"    charged under the incident {ahead_cost:>8.2f} s   "
              f"= x{ahead_cost / ahead_base:.2f} its base time")
        print()
        print("  The placeholder is in force there, which is what it is for: no")
        print("  vehicle has reported on that road, so there is nothing better to")
        print("  price it with. The first fleet vehicle to drive it retires the")
        print("  placeholder — which is exactly what happened to every road in")
        print("  section 2, in the tick the fleet measured it.")
        print()
        return result, edge

    print("  (No unmeasured road was reachable ahead of the fleet to show the")
    print("  placeholder in force; every road it runs on has now been measured.)")
    print()
    return result, edge


def show_log(log_store: TrafficLogStore, edges: set[tuple]) -> None:
    """What the fleet actually wrote, and which of it a baseline will keep."""
    print("4. What landed in the log")
    print(RULE)
    rows, _ = log_store.read(limit=1000)
    written = [row for row in rows if (row.road_u, row.road_v) in edges]
    print(f"  {len(written)} row(s) for the {len(edges)} road(s) the fleet reported "
          "on, newest first:")
    print()
    print(f"  {'road':<20} {'condition':<10} {'incident':<10} {'travel_time':>11}")
    for row in written[:12]:
        seconds = "—" if row.travel_time is None else f"{row.travel_time:.2f} s"
        print(f"  {f'{row.road_u} -> {row.road_v}':<20} {row.traffic_condition:<10} "
              f"{(row.incident_type or '—'):<10} {seconds:>11}")
    if len(written) > 12:
        print(f"  … {len(written) - 12} more")
    print()
    print("  The travel_time column holds the *measured* seconds, not the model's")
    print("  estimate they replaced. The rows are written against the repriced")
    print("  state — the one the readings have already been folded into — because")
    print("  the model's own number is exactly what the measurement superseded, and")
    print("  recording it would lose the only new information the fleet produced.")
    print()

    flagged = [row for row in written if row.incident_type is not None]
    print(f"  {len(written) - len(flagged)} row(s) carry no incident word and "
          f"{len(flagged)} carry one.")
    print("  That column decides whether a row can ever be evidence. build_baseline")
    print("  skips any row with an incident_type, because a baseline trained on")
    print("  anomalies cannot detect them: a road's normal would come to include the")
    print("  day it was closed. So an ordinary ping enters that road's history and an")
    print("  incident-time ping does not — and the detector can still flag the")
    print("  incident-time reading, because it judges it against a history that")
    print("  excludes it.")
    print()
    print("  The rows outlive the scenario they were taken under. Costs are per")
    print("  scenario — the same scenario id always prices the same way, which is")
    print("  what makes a solver comparison meaningful — but this log is the road")
    print("  network's history, and every later scenario's baseline is built from it.")
    print()


def show_live(client: TestClient, scenario_id: str, args) -> None:
    """Let the background thread tick on its own clock."""
    print("5. The background thread")
    print(RULE)
    # Stopped first, and it has to be: a second fleet on one scenario is a 409,
    # because two watchers would fight over the same compare-and-swap every tick.
    scripted = stop_fleet(client, scenario_id)
    print(f"  the scripted fleet ran {scripted['ticks']} tick(s); DELETE stopped it")
    print(f"  restarting it at --interval {args.interval:g} s, time scale "
          f"{args.time_scale:g}")
    print()

    started = start_fleet(
        client, scenario_id, args, interval=args.interval, time_scale=args.time_scale
    )
    print(f"  restarted   {started['solver']}, {len(started['vehicles'])} vehicle(s), "
          f"first tick inline at {started['first_tick']['tick']}")
    print(f"  waiting     {args.live_seconds:g} s of wall clock…")
    time.sleep(args.live_seconds)

    state = status(client, scenario_id)
    print()
    print(f"  ticks so far   {state['ticks']}  "
          f"(1 inline, the rest from the timer at {state['interval_seconds']:g} s apart)")
    print(f"  last tick at   {state['last_tick_at']}")
    print(f"  running        {state['running']}")
    print()
    print(f"  {'vehicle':<10} {'road':<20} {'remaining':>10} {'along':>7}")
    for vehicle in state["vehicles"]:
        remaining = (
            "—" if vehicle["remaining_seconds"] is None
            else f"{vehicle['remaining_seconds']:.1f}s"
        )
        print(f"  {vehicle['vehicle_id']:<10} {road(vehicle['edge']):<20} "
              f"{remaining:>10} {vehicle['progress']:>6.0%}")
    print()
    print("  None of that was requested. The fleet ticks on its own daemon thread,")
    print("  and every tick measures roads, judges the readings and reprices the")
    print("  scenario — which is why the positions have moved on their own, and why")
    print("  the numbers printed in section 2 would not come out the same way twice")
    print("  while this is running.")
    print()
    print("  Ticks here are ones the scripted sections did not take, at the real")
    print("  interval rather than the one-second step they were stepped at.")
    print("  Restarting did not reset the scenario — the measurements the first")
    print("  fleet took are still in its cost matrix, and its log rows are permanent.")
    print()

    final = stop_fleet(client, scenario_id)
    print(f"  DELETE stopped it: running={final['running']}, ticks={final['ticks']}.")
    print("  The readings survive the watcher — only the ticking stops.")
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="qgati-demo-watcher-") as tmp:
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
    graph = get_graph()

    print("Q-Gati — simulated GPS fleet demo")
    print("=" * 78)
    print(f"log        {log_path}")
    print("           a throwaway; the collected log at backend/data/traffic_log.db")
    print("           is neither read nor written by this run")
    print(f"instance   {args.deliveries} deliveries, {args.vehicles} vehicles, "
          f"seed {args.seed}")
    print(f"priced at  {WHEN.isoformat()}  (outside the daytime band, x1.0)")
    print("fleet      a simulation, not fleet integration: there is no GPS device,")
    print("           no vehicle and no telemetry protocol. The pings are generated")
    print("           from the same traffic model the rest of the app prices with.")
    print()

    scripted_scale = SCRIPTED_STEP_SECONDS / IDLE_INTERVAL
    step = IDLE_INTERVAL * scripted_scale

    edges: set[tuple] = set()
    with TestClient(application) as client:
        rows_before = log_store.count()
        scenario_id = create_scenario(client, args)
        print(f"scenario   {scenario_id}")
        print()

        try:
            started = start_fleet(
                client,
                scenario_id,
                args,
                interval=IDLE_INTERVAL,
                time_scale=scripted_scale,
            )
            show_dispatch(started, step)
            for reading in started["first_tick"]["readings"]:
                if reading["edge"] is not None:
                    edges.add(as_tuple(reading["edge"]))

            results, compared, compared_vehicle = show_ticks(
                client, scenario_id, args, step
            )
            for result in results:
                for reading in result["readings"]:
                    if reading["edge"] is not None:
                        edges.add(as_tuple(reading["edge"]))

            anomaly_tick, incident_edge = show_anomaly(
                client, scenario_id, step, compared, compared_vehicle
            )
            for reading in anomaly_tick.get("readings", []):
                if reading["edge"] is not None:
                    edges.add(as_tuple(reading["edge"]))
            if incident_edge is not None:
                edges.add(incident_edge)

            show_log(log_store, edges)
            print(f"  rows in the log before the fleet started: {rows_before}")
            print(f"  rows in the log now:                      {log_store.count()}")
            print()

            if args.live_seconds > 0:
                show_live(client, scenario_id, args)

        finally:
            # A daemon thread writing to a log store that is about to be closed
            # is the one thing here that could outlive the run.
            REGISTRY.stop_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
