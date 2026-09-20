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

## Road graph

The Delhi drive network is fetched once from OpenStreetMap and cached, so only
the very first run touches the network:

```python
from qgati.graph import load_delhi_graph

graph = load_delhi_graph()   # ~20 s first time, ~0.2 s from cache thereafter
```

The default extract is a 4 km × 4 km box around Connaught Place: **2033 nodes /
4778 edges**, of which 1996 nodes form one strongly-connected core. Widen it with
`load_delhi_graph(dist=3000, force_refresh=True)`.

Edge attributes: `length` (metres), `speed_kph`, `travel_time` (seconds), and
`weight` — which mirrors `travel_time`, so routing reasons about minutes on the
road rather than metres. Use `largest_strongly_connected_subgraph(graph)` before
solving a VRP; OSM extracts contain one-way stubs that can be entered but not
left, and a tour needs every stop reachable from every other.

## Routing

`dijkstra` and `astar` are original implementations, deliberately independent of
networkx's so the test suite can check one against the other.

```python
from qgati.routing import dijkstra, astar, make_haversine_heuristic, dijkstra_all_pairs

path, seconds = dijkstra(graph, source, target)
path, seconds = astar(graph, source, target, heuristic=make_haversine_heuristic(graph))

# Cost matrix for the VRP layer: every ordered pair, one search per source.
table = dijkstra_all_pairs(graph, stops)      # {(src, dst): (path, cost)}
```

A `21 × 21` cost matrix over real Delhi nodes takes ~160 ms, against ~810 ms for
the naive one-search-per-pair equivalent.

## Tests

```bash
cd backend
uv run pytest                  # 26 fast tests, no network
```

The suite runs against a synthetic random graph, so it is fast and offline. One
test exercises the real Delhi graph and is skipped by default; opt in with:

```bash
QGATI_RUN_SLOW=1 uv run pytest         # bash
$env:QGATI_RUN_SLOW = "1"; uv run pytest   # PowerShell
```

It needs `backend/data/cache/delhi_drive_2000m.graphml` to exist, which one call
to `load_delhi_graph()` creates.

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

`data/` holds cached graphs and scenario configs. `data/cache/` is gitignored: it
holds the fetched `.graphml`, plus `data/cache/osm_http/` for OSMnx's raw
Overpass responses.
