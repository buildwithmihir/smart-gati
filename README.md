# Smart-Gati

Intelligent vehicle-routing and fleet-optimization system for Delhi.

Smart-Gati loads the real Delhi road network and solves the Vehicle Routing Problem —
which vehicle serves which delivery, in what order — then keeps adjusting as
traffic changes.

The pipeline:

1. **Road graph** — fetch and cache the real Delhi network via OSMnx/NetworkX
2. **Routing** — Dijkstra / A* shortest paths over that graph
3. **Optimization** — QPSO (Quantum Particle Swarm Optimization) solves the VRP,
   benchmarked against GA, classical PSO, Clarke-Wright Savings, and brute force.
   ACO was benchmarked against QPSO too and then removed from the project; the
   finding that it reached a lower mean cost is recorded in
   [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md#removed-aco)
4. **Traffic** — rule-based traffic simulator (no ML, no external API), live
   incidents, anomaly detection, and a simulated GPS fleet that measures the roads
   it drives
5. **Re-optimization** — partial, per vehicle: re-plan only the unserved
   deliveries, from wherever each vehicle currently is — triggered by an incident,
   an anomaly, or a driver reporting their own road
6. **Explainability** — why a re-plan looks the way it does, as sentences built
   from that re-plan's own figures: what moved, what it cost, and what the closure
   cost in seconds
7. **Run history** — every solve recorded to SQLite with its solver, cost, runtime
   and (for a re-plan) the remaining travel time before and after, then aggregated
   into summary statistics

## Structure

```
smart-gati/
├── backend/                  # Python service and optimization engine (uv-managed)
│   └── src/qgati/
│       ├── graph/            # road graph construction and caching (OSMnx/NetworkX)
│       ├── routing/          # shortest paths — Dijkstra, A*
│       ├── optimizer/        # VRP solvers — QPSO, GA, classical PSO, Savings, brute force
│       ├── traffic/          # rule-based traffic simulator (no ML)
│       ├── fleet/            # simulated GPS fleet — positions, pings, the watcher
│       ├── reopt/            # adaptive re-optimization, and the explanation of it
│       ├── analytics/        # the run history — SQLite log and its aggregates
│       ├── explain/          # unused placeholder (see Status)
│       └── api/              # FastAPI application
├── frontend/                 # Next.js app — Dashboard, History, Compare
└── README.md
```

## Running the backend

See [`backend/README.md`](backend/README.md). In short:

```bash
cd backend
uv sync
uv run uvicorn qgati.api.main:app --reload
```

Health check at http://127.0.0.1:8000/health — API docs at `/docs`.

## Status

Phase 1 (road graph), Phase 2 (routing), Phase 3 (VRP solvers including QPSO),
Phase 4 (benchmark runner), Phase 5 (FastAPI backend), Phase 6 (Next.js frontend),
and the traffic, fleet, re-optimization and run-history layers are complete:
`qgati.traffic` simulates and measures conditions, `qgati.fleet` drives a simulated
GPS fleet over them, `qgati.reopt` re-plans the deliveries still outstanding —
never the ones already made — and explains the re-plan it produced, and
`qgati.analytics` records every solve to SQLite and aggregates them.

The frontend has three views: the **Dashboard** (map, plan, incident panel, and the
"why this route" explanation), **History** (past runs and their summary), and
**Compare** (all five solvers run against the loaded scenario).

`explain/` is the one remaining empty package, and it is a **placeholder that
nothing imports**: explainability was built in `reopt/explain.py`, because a
decision trace belongs with the re-plan that produced the numbers it quotes. The
directory is left in place rather than deleted — removing it is a rename, not a
feature, and it is not worth a commit of its own.
