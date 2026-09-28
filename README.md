# Q-Gati

Intelligent vehicle-routing and fleet-optimization system for Delhi.

Q-Gati loads the real Delhi road network and solves the Vehicle Routing Problem —
which vehicle serves which delivery, in what order — then keeps adjusting as
traffic changes.

The pipeline:

1. **Road graph** — fetch and cache the real Delhi network via OSMnx/NetworkX
2. **Routing** — Dijkstra / A* shortest paths over that graph
3. **Optimization** — QPSO (Quantum Particle Swarm Optimization) solves the VRP,
   benchmarked against GA, classical PSO, ACO, Clarke-Wright Savings, and brute force
4. **Traffic** — rule-based traffic simulator (no ML, no external API)
5. **Re-optimization** — stub; planned for a later phase
6. **Explainability** — stub; planned for a later phase

## Structure

```
q-gati/
├── backend/                  # Python service and optimization engine (uv-managed)
│   └── src/qgati/
│       ├── graph/            # road graph construction and caching (OSMnx/NetworkX)
│       ├── routing/          # shortest paths — Dijkstra, A*
│       ├── optimizer/        # VRP solvers — QPSO, GA, PSO, ACO, Savings, brute force
│       ├── traffic/          # rule-based traffic simulator (no ML)
│       ├── reopt/            # stub — warehouse for partial re-optimization (later phase)
│       ├── explain/          # stub — warehouse for decision traces (later phase)
│       └── api/              # FastAPI application
├── frontend/                 # Next.js app — Phase 6
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
Phase 4 (benchmark runner), Phase 5 (FastAPI backend), and Phase 6 (Next.js frontend)
are complete. Traffic simulation works (`qgati.traffic`). The `reopt/` and `explain/`
packages are empty stubs that will be filled in later phases.
