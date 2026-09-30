#!/usr/bin/env python
"""A driver says the road is worse than the model thinks, and gets a new route.

Two of the ways a fleet earns a re-plan are *derived*: an incident somebody
reported, or a reading the detector flagged on the fleet's last tick. That rule is
what stops ``POST /reoptimize`` from being a re-solve button, and it is enforced
rather than asserted — with neither holding, the route answers 409.

It also has a blind spot, and ``DESIGN_DECISIONS.md`` names it: an incident's
``x2.9`` is a **flat placeholder**. If the real delay is five times the modelled
time, the statistical route to noticing it runs through a fleet that has to drive
the road, be measured, and have the detector — at ``NOISE_SIGMA = 0.06`` — agree
the trip was unusual. The one party who already knows is sitting in the vehicle.

This script shows the place they can say so, over real HTTP against the real
Delhi graph:

1. **Dispatch.** The plan the fleet went out on, where each vehicle is now, and
   the road under each of them.
2. **The report.** A driver names their own road as ``slow``. Their remaining
   stops are re-planned from where they are; nobody else is told anything.
3. **The escalation.** The same driver reports it ``closed``, and the new route
   begins at the **near** end of that road rather than the far one — a vehicle
   cannot be planned through a road it cannot drive.
4. **The before/after, for the whole fleet.** Every other vehicle's route is
   searched for a change and there is none, asserted here rather than described.
5. **The revert.** The report is a real incident on the scenario, with an id, and
   ``DELETE`` takes it back off.

Why the fleet is stepped by hand
--------------------------------
The watcher really does run on a background thread. Using ``POST
/scenarios/{id}/watcher/tick`` instead means the numbers printed are the tick's
own, in the order they happened, rather than whatever the thread had reached by
the time the next request was handled.

Why this writes, when ``/reoptimize`` does not
----------------------------------------------
A derived trigger means nothing has changed and a plan is merely computed. A
driver's report means the network has changed, and the plan is the response to
it — so the road is re-priced, an audit row is written, and the report lands in
the scenario's own incident list, through the same code ``POST /incident`` uses.
That is also what keeps ``/reoptimize``'s rule intact rather than cutting a hole
in it: the driver changed the network, and the next fleet-wide call is justified
by a live incident like any other.
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
    CLOSURE,
    PEAK_FACTOR,
    SLOW,
    TrafficLogStore,
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
            "solver key for the driver's re-plan; also the production default when "
            "omitted. Both are QPSO unless told otherwise, which is the point — a "
            "driver-scoped re-plan goes through the same optimizer a fleet-wide "
            "one does, on a smaller instance"
        ),
    )
    parser.add_argument("--ticks", type=int, default=2, help="ticks before the report")
    return parser


# --------------------------------------------------------------------------- #
# Reading the scenario's own numbers
# --------------------------------------------------------------------------- #
def cost_of(graph: nx.Graph, scenario_id: str, edge: tuple) -> float | None:
    """What the app currently charges for ``edge``, in seconds.

    Read from the stored scenario's *effective* state, so this is the number the
    route prices a vehicle's position under rather than a recomputation of it.
    """
    if not graph.has_edge(*edge):
        return None
    state = get_store().get(scenario_id).effective_traffic_state()
    return simulated_travel_time(graph, edge_of(graph, *edge), state)


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


def avoid(
    client: TestClient, scenario_id: str, vehicle_id: str, edge: tuple,
    treatment: str, args,
) -> tuple[int, dict]:
    """POST the driver's report, returning the status **and** the body.

    The body matters on a refusal as much as on a success — the 409's ``detail``
    says which of the three reasons applied — so this does not raise on a non-2xx
    the way the other request helpers do.
    """
    body: dict = {
        "edge": {"u": edge[0], "v": edge[1]},
        "treatment": treatment,
        "include_geometry": False,
    }
    if args.reopt_solver is not None:
        body["solver"] = args.reopt_solver
    response = client.post(
        f"/scenarios/{scenario_id}/vehicles/{vehicle_id}/avoid-road", json=body
    )
    return response.status_code, response.json()


def revert(client: TestClient, scenario_id: str, incident_id: str) -> dict:
    response = client.delete(f"/scenarios/{scenario_id}/incident/{incident_id}")
    response.raise_for_status()
    return response.json()


def fleet_view(scenario_id: str) -> tuple[VehicleProgress, ...]:
    """Where the fleet is, derived exactly as the endpoint derives it.

    The route reads this itself, under the conditions as they stand *before* the
    report is applied. The demo reaches for the same function for the same
    reason section 1 prints it: where a vehicle is is a fact about the vehicle,
    not about the prices, and reading it after a closure would report the driver
    as stopped with no position at all — which is the case the endpoint exists
    to answer.
    """
    record = get_store().get(scenario_id)
    graph = get_graph()
    weight = traffic_weight_function(graph, record.effective_traffic_state())
    lookup = cost_lookup(graph, weight)
    watcher: ScenarioWatcher = REGISTRY.get(scenario_id)
    return watcher.read(lambda tracks: fleet_progress(record.scenario, tracks, lookup))


# --------------------------------------------------------------------------- #
# Picking the driver
# --------------------------------------------------------------------------- #
def busiest_driver(progress: Sequence[VehicleProgress]) -> VehicleProgress | None:
    """The vehicle with the most work left that is still on a road.

    The fleet spreads itself along its own corridors, so the last vehicle is
    often finished. A finished one has nothing to re-plan and a stopped one has
    no road to name, and neither is who this demo is about.
    """
    driving = [item for item in progress if item.remaining and item.edge is not None]
    if not driving:
        return None
    return max(driving, key=lambda item: len(item.remaining))


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


def seconds(value: float | None) -> str:
    """A cost in seconds, or a dash when the road is not currently priceable."""
    return "—" if value is None else f"{value:.2f} s"


def route_stops(route: dict) -> str:
    return listing([stop["delivery_id"] for stop in route["stops"]])


def by_vehicle(routes: Sequence[dict]) -> dict[str, dict]:
    return {route["vehicle_id"]: route for route in routes}


def _wrap(text, width: int, indent: str = "") -> list[str]:
    """Greedy wrap, so a long detail string does not run off the terminal."""
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
# Sections
# --------------------------------------------------------------------------- #
def show_dispatch(started: dict, progress: Sequence[VehicleProgress]) -> None:
    print("1. Dispatch — the plan the fleet went out on, and where it is now")
    print(RULE)
    print(f"  solved by   {started['solver']}")
    print(f"  vehicles    {len(started['routes'])} dispatched")
    print()
    print(f"  {'vehicle':<10} {'stops':<24} {'on the road':<18} "
          f"{'delivered':<16} still ahead")
    for route in started["routes"]:
        item = next(
            (entry for entry in progress if entry.vehicle_id == route["vehicle_id"]),
            None,
        )
        on = road(item.edge) if item is not None else "—"
        done = listing([str(stop) for stop in item.completed]) if item else "(none)"
        ahead = listing([str(stop) for stop in item.remaining]) if item else "(none)"
        print(f"  {route['vehicle_id']:<10} {route_stops(route):<24} {on:<18} "
              f"{done:<16} {ahead}")
    print()
    print("  The stop numbers are positions in the scenario's delivery list, which")
    print("  is what the *fleet* is indexed by; the endpoint reports delivery ids,")
    print("  and the sections below print those. The two are the same stops.")
    print()
    print("  None of the 'delivered' column was declared by this script. Each")
    print("  vehicle's progress is read off the corridor it is actually driving,")
    print("  and the point it has reached on that corridor says how far down it")
    print("  is. There is no fleet-state field on any request for a caller to")
    print("  assert — which is also why the driver below cannot claim to have")
    print("  delivered anything, only to have seen something.")
    print()


def show_report(
    client: TestClient, scenario_id: str, treatment: str, args
) -> tuple[dict, tuple, str] | None:
    """One driver's report, one re-plan, and the road it was about."""
    driver = busiest_driver(fleet_view(scenario_id))
    if driver is None:
        print("  no vehicle still has work ahead of it and a road under it; raise")
        print("  --deliveries and run again")
        print()
        return None

    # Read here rather than reused between sections: a report changes the prices,
    # which changes which road a vehicle counts as being on, and a driver reports
    # the road they are on *now*.
    edge = driver.edge
    assert edge is not None  # busiest_driver only returns vehicles on a road

    print(f"  driver      {driver.vehicle_id}, "
          f"{len(driver.remaining)} stop(s) still ahead")
    print(f"  reports     {road(edge)} as {treatment}")
    print(f"  the road    currently charged "
          f"{seconds(cost_of(get_graph(), scenario_id, edge))}")
    if treatment == SLOW:
        base = cost_of(get_graph(), scenario_id, edge)
        if base is not None:
            print(f"              a '{SLOW}' report re-prices it to x{SLOW_FACTOR} "
                  f"— {base * SLOW_FACTOR:.2f} s — which is the flat")
            print("              placeholder a driver correcting it upward is "
                  "correcting")
    print()

    code, body = avoid(client, scenario_id, driver.vehicle_id, edge, treatment, args)
    print(f"  POST /scenarios/{{id}}/vehicles/{driver.vehicle_id}/avoid-road -> "
          f"HTTP {code}")
    if code != 200:
        for line in _wrap(body.get("detail", body), 74, indent="  "):
            print(line)
        print()
        return None
    print()

    incident = body["incident"]
    print(f"  trigger     {', '.join(body['trigger']['kinds'])}  "
          f"(primary: {body['trigger']['primary']})")
    print(f"              {body['trigger']['detail']}")
    print(f"  incident    {incident['incident_id']}  "
          f"({incident['incident_type']} on {road(incident['edge'])})")
    print(f"  re-planned  {listing(body['replanned_vehicles'])}  "
          f"— and only these were in the instance that was solved")
    print()

    where = next(
        item for item in body["vehicles"] if item["vehicle_id"] == driver.vehicle_id
    )
    print(f"  {driver.vehicle_id} begins its new route at node {where['node']} "
          f"({road(where['edge'])})")
    print()

    before = by_vehicle(body["before"])[driver.vehicle_id]
    after = by_vehicle(body["after"])[driver.vehicle_id]
    print(f"  before      {route_stops(before):<34} "
          f"{before['travel_time']:>7.0f} s  {before['travel_cost']:>8.2f} rupees")
    print(f"  after       {route_stops(after):<34} "
          f"{after['travel_time']:>7.0f} s  {after['travel_cost']:>8.2f} rupees")
    print()
    same_work = sorted(route_stops(before).split()) == sorted(route_stops(after).split())
    print(f"  the same stops on both sides: {same_work}. A driver-scoped re-plan")
    print("  cannot hand work to anybody — there is nobody else in the instance —")
    print("  so all it can change is the order and the road it starts on.")
    print()

    return body, edge, driver.vehicle_id


def show_untouched(body: dict, driver_id: str) -> None:
    """Every other vehicle's route, compared across the call."""
    print("4. The before/after, for the whole fleet — and what did not move")
    print(RULE)
    before = by_vehicle(body["before"])
    after = by_vehicle(body["after"])
    print(f"  {'vehicle':<10} {'before':<26} {'after':<26} {'':<8}")
    for name, old in before.items():
        new = after[name]
        old_stops = [stop["delivery_id"] for stop in old["stops"]]
        new_stops = [stop["delivery_id"] for stop in new["stops"]]
        if name == driver_id:
            mark = "re-planned"
        elif old_stops == new_stops and old["travel_cost"] == new["travel_cost"]:
            mark = "untouched"
        else:
            mark = "CHANGED"
        print(f"  {name:<10} {listing(old_stops) or '(nothing left)':<26} "
              f"{listing(new_stops) or '(nothing left)':<26} {mark:<8}")
    print()

    # The claim, checked rather than asserted in prose. `before` and `after` are
    # priced by one evaluate over one scenario, so an untouched route is identical
    # on both sides by construction — and this is where that construction is
    # verified against the response rather than trusted.
    violations = []
    for name, old in before.items():
        if name == driver_id:
            continue
        new = after[name]
        if [stop["delivery_id"] for stop in old["stops"]] != [
            stop["delivery_id"] for stop in new["stops"]
        ] or old["travel_cost"] != new["travel_cost"]:
            violations.append(name)
    print(f"  vehicles other than {driver_id} whose route changed: "
          f"{listing(violations) if violations else 'none'}")
    if violations:
        raise AssertionError(
            f"a driver-scoped re-plan moved {violations}, which is not in the "
            "instance that was solved"
        )
    print()
    print("  Every other vehicle is absent from the instance the optimizer was")
    print("  handed, so no solver could have moved it however badly it searched —")
    print("  the guarantee is structural, like the completed-stop one. A vehicle")
    print("  with nothing left is '(nothing left)' on both sides, which is a true")
    print("  if easy instance of the same thing.")
    print()
    print(f"  solved by   {body['solver']} ({body['solver_name']}), "
          f"scoped to {len(body['replanned'])} stop(s), {body['runtime_ms']:.0f} ms")
    print(f"  priced at   {body['travel_time']:.0f} s, "
          f"{body['distance_m'] / 1000:.1f} km, {body['fuel_litres']:.2f} L "
          f"— {body['travel_cost']:.2f} rupees")
    print(f"  feasible={body['feasible']}   moved={len(body['moved'])} "
          "(empty by construction: there is nobody to hand work to)")
    print()


def show_reverts(client: TestClient, scenario_id: str, incident_ids: Sequence[str]) -> None:
    """Put the network back, one report at a time."""
    print("5. The revert — the reports are real incidents, and they come back off")
    print(RULE)
    before = get_store().get(scenario_id)
    print(f"  before any DELETE   mutated={before.conditions_mutated}  "
          f"{len(before.incidents)} live incident(s)")
    print()

    for incident_id in reversed(list(incident_ids)):
        body = revert(client, scenario_id, incident_id)
        print(f"  DELETE /scenarios/{{id}}/incident/{incident_id}")
        print(f"    -> HTTP 200, applied={body['applied']}, "
              f"{body['changed_legs']} leg(s) re-priced back")
    print()

    after = get_store().get(scenario_id)
    print(f"  after all DELETEs   mutated={after.conditions_mutated}  "
          f"{len(after.incidents)} live incident(s)")
    print()
    print("  What a revert takes back off is exactly what the report added: an")
    print("  incident is a fold over the scenario's own stored state, so removing")
    print("  it restores the conditions it replaced rather than leaving a drift")
    print("  behind. The fleet's own measurements are a separate layer and are")
    print("  untouched by this — they were never the report's to undo.")
    print()
    print("  This is why the report goes through ``POST /incident``'s machinery")
    print("  instead of being a flag on the re-plan request. The driver changed")
    print("  the network, so the network shows it, an audit row records it, and")
    print("  an operator can take it back off. And it is what keeps")
    print("  ``POST /reoptimize``'s rule intact: the next fleet-wide call is")
    print("  justified by a live incident like any other, rather than by a hole")
    print("  cut in the rule for drivers.")
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="qgati-demo-avoid-") as tmp:
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

    print("Q-Gati — manual override demo (a driver avoids their own road)")
    print("=" * 78)
    print(f"log        {log_path}")
    print("           a throwaway; the collected log at backend/data/traffic_log.db")
    print("           is neither read nor written by this run")
    print(f"instance   {args.deliveries} deliveries, {args.vehicles} vehicles, "
          f"seed {args.seed}")
    print(f"priced at  {WHEN.isoformat()}  (outside the daytime band, x1.0)")
    print()
    print("the claim  a driver can name their own road and be re-planned around")
    print("           it, while every other vehicle keeps exactly the route it was")
    print("           already driving — and the report is a real incident on the")
    print("           scenario, not a way around the trigger rule.")
    print()

    with TestClient(application) as client:
        scenario_id = create_scenario(client, args)
        print(f"scenario   {scenario_id}")
        print()

        incidents: list[str] = []
        try:
            started = start_fleet(client, scenario_id, args)
            for _ in range(args.ticks):
                tick(client, scenario_id)
            show_dispatch(started, fleet_view(scenario_id))

            print(f"2. The report — POST /scenarios/{{id}}/vehicles/{{id}}/avoid-road "
                  f"({SLOW})")
            print(RULE)
            slow = show_report(client, scenario_id, SLOW, args)
            if slow is None:
                return 1
            incidents.append(slow[0]["incident"]["incident_id"])

            print("3. The escalation — the driver, and the road now shut under them")
            print(RULE)
            print("  A driver who said 'slow' and then watches the road close has")
            print("  said something the model cannot express: the road is not")
            print("  drivable at all. This is the case a re-plan has to answer")
            print("  differently, because a route cannot begin on the far side of a")
            print("  road the vehicle cannot drive.")
            print()
            print("  The road is read fresh from the fleet rather than carried over")
            print("  from section 2, and that is not a convenience: re-pricing the")
            print("  road changed how long it takes, which changed where along its")
            print("  own corridor the vehicle now counts as being. A driver reports")
            print("  the road they are on *now*.")
            print()
            shut = show_report(client, scenario_id, CLOSURE, args)
            if shut is None:
                return 1
            incidents.append(shut[0]["incident"]["incident_id"])

            body, edge, driver_id = shut
            where = next(
                item for item in body["vehicles"] if item["vehicle_id"] == driver_id
            )
            near, far = edge[0], edge[1]
            print(f"  {driver_id} was driving {road(edge)}. Under a {SLOW} report it")
            print(f"  would begin its new route at the far end, node {far}. Under a")
            print(f"  {CLOSURE} it begins at node {where['node']} — the near end, "
                  f"node {near}.")
            if where["node"] != near:
                raise AssertionError(
                    f"a closure on the driver's own road should restart them at "
                    f"node {near}, but the response says {where['node']}"
                )
            print("  It turned around. The same rule covers a vehicle stopped")
            print("  behind somebody else's closure, which is the other thing this")
            print("  endpoint can do and `POST /reoptimize` cannot: a fleet-wide")
            print("  re-plan has nowhere to put a stopped vehicle's load, so it")
            print("  refuses, and here the stopped vehicle is the only one that")
            print("  matters.")
            print()
            print("  Note that the report was applied *after* the vehicle's")
            print("  position was read. Where a vehicle is is a fact about the")
            print("  vehicle, not about the prices — under the closure it just")
            print("  filed, its own road is unpriceable, and reading it afterwards")
            print("  would report the driver as stopped with no position at all.")
            print()

            show_untouched(body, driver_id)
            show_reverts(client, scenario_id, incidents)

        finally:
            REGISTRY.stop_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
