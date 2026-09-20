# Q-Gati Backend

FastAPI service and optimization engine. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
cd backend
uv sync
```

## Run the API

```bash
cd backend
uv run uvicorn qgati.api.main:app --reload
```

Then check it is alive:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

Interactive docs are at http://127.0.0.1:8000/docs.

## Tests

```bash
cd backend
uv run pytest
```

## Layout

| Package             | Responsibility                                              |
| ------------------- | ----------------------------------------------------------- |
| `qgati.graph`       | Road graph construction and caching (OSMnx/NetworkX)         |
| `qgati.routing`     | Shortest paths — Dijkstra, A*                                |
| `qgati.optimizer`   | VRP solvers — QPSO, GA, classical PSO, ACO, Savings, brute force |
| `qgati.traffic`     | Rule-based traffic simulator + ML travel-time predictor      |
| `qgati.reopt`       | Adaptive partial re-optimization                             |
| `qgati.explain`     | Decision traces / explainability                             |
| `qgati.api`         | FastAPI application                                          |

`data/` holds cached graphs and scenario configs. `data/cache/` is gitignored.
