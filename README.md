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
4. **Traffic** — rule-based simulator plus an ML travel-time predictor
5. **Re-optimization** — adaptive partial re-solve when conditions shift
6. **Explainability** — decision traces showing *why* a route was chosen

## Structure

```
q-gati/
├── backend/                  # Python service and optimization engine (uv-managed)
│   └── src/qgati/
│       ├── graph/            # road graph construction and caching (OSMnx/NetworkX)
│       ├── routing/          # shortest paths — Dijkstra, A*
│       ├── optimizer/        # VRP solvers — QPSO, GA, PSO, ACO, Savings, brute force
│       ├── traffic/          # rule-based simulator + ML travel-time predictor
│       ├── reopt/            # adaptive partial re-optimization
│       ├── explain/          # decision traces / explainability
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

Phases 0–1: project skeleton in place, API boots with a health endpoint.
Routing, optimization, and the frontend are not implemented yet.
