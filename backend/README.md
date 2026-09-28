# Q-Gati Backend

A **multi-algorithm VRP framework** — six solvers behind one contract, benchmarked
against each other on identical cost matrices — plus the FastAPI service around
it. **QPSO is the production default.** The problem statement (SIH PS 26137) names
quantum-inspired search as the focus of this work — benchmarked against
conventional metaheuristics and exact methods — so QPSO is the solver the API
serves, and the other five exist to measure it against: four conventional
metaheuristics (Savings, GA, classical PSO, ACO) and brute force as exact ground
truth.

The benchmark is reported in full, including where QPSO loses: ACO reaches lower
raw cost at n=15 and n=25. What the results do support is the comparison QPSO was
chosen for — at equal budget it beats classical PSO, which shares its
representation and differs *only* in the update rule, so the gap is attributable
to the quantum sampling itself. See
[What the benchmark actually shows](#what-the-benchmark-actually-shows).

Managed with [uv](https://docs.astral.sh/uv/).

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

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/solvers` | Every registered solver, and which is the production default |
| `GET` | `/graph/servable` | How many mutually-reachable nodes scenarios draw from |
| `POST` | `/scenarios` | Create a scenario — generated at random, or described explicitly, optionally under simulated traffic conditions |
| `GET` | `/scenarios` | List stored scenarios |
| `GET` | `/scenarios/{scenario_id}` | Fetch one |
| `GET` | `/graph/delhi` | The road network as GeoJSON, optionally scoped to a scenario or a bbox |
| `POST` | `/optimize/{scenario_id}` | Run one solver on a stored scenario |
| `GET` | `/optimize/{scenario_id}/compare` | Run every solver on the same cost matrix |
| `GET` | `/traffic/log` | Inspect the collected road-condition log, paginated |

**`POST /optimize/{scenario_id}` defaults to QPSO**, the production solver. Note
that the scenario is named by the **URL**, never the body: the body carries only
*how* to solve, so the two kinds of input cannot drift into one payload. `GET
/optimize/{id}/compare` runs all six — that is where QPSO's headline result sits
next to the conventional solvers it is benchmarked against, on identical costs.

### A worked example

```bash
# Create an instance on the real Delhi graph and keep its id.
SID=$(curl -s -X POST localhost:8000/scenarios -H 'Content-Type: application/json' \
  -d '{"kind":"generate","n_deliveries":12,"n_vehicles":3,"seed":21}' \
  | python -c "import json,sys; print(json.load(sys.stdin)['scenario_id'])")

# Optimize it with the production default; no solver name needed.
# The scenario id goes in the URL, not the body.
curl -s -X POST "localhost:8000/optimize/$SID" -H 'Content-Type: application/json' \
  -d '{"seed":0}'
```

The response resolves the solver's internal delivery *indices* back into ids and
road-graph nodes, so it is directly renderable:

```json
{
  "solver": "aco", "cost": 1768.7, "travel_cost": 1768.7, "feasible": true,
  "runtime_ms": 456, "iterations": 100, "population": 30,
  "routes": [
    {"vehicle_id": "V0", "load": 29, "capacity": 29, "travel_cost": 988.8,
     "stops": [{"delivery_id": "D1", "node": 928490302}, ...]}
  ]
}
```

Compare, on a scenario small enough for an exact answer — an `n=10, k=3`
instance created with `{"kind":"generate","n_deliveries":10,"n_vehicles":3,
"seed":21}` and solved at solver `seed=0`:

```
GET /optimize/{scenario_id}/compare?seed=0

exact optimum: 1492.7
solver               travel_cost     ms gap_opt%
Brute Force               1492.7     25     0.00
Savings                   1622.3      2     8.68
ACO                       1502.3    309     0.65
Genetic Algorithm         1492.7    215     0.00
Classical PSO             1622.3    144     8.69
QPSO                      1502.3    149     0.65
```

Read that as one instance, not a ranking: at `n=10` three of the six solvers land
exactly on the optimum, so the instance barely separates them. The ordering that
matters comes from the multi-instance benchmark below, not from a single row of
this table.

### API design notes

**The road graph is a dependency, not a global.** `get_graph` is a FastAPI
dependency, so tests substitute a synthetic graph through
`app.dependency_overrides` and exercise the whole surface with no cached Delhi
extract and no network. It is loaded lazily, because a cold `load_delhi_graph()`
is ~20 s.

**The cost matrix is cached with the scenario.** Building it runs an all-pairs
Dijkstra — ~160 ms for a 21-stop instance — and every optimizer call needs it.
Caching it also guarantees that two calls against one `scenario_id` are
optimizing *identical* costs, which is what makes `/compare` meaningful.

**Scenarios are in memory.** Restarting the process drops them. Persistence is
out of scope for this phase; the store's interface is small enough that swapping
in a database would not touch the routes.

**Coordinates are snapped by a linear scan.** `nearest_node` is ~2000 distance
comparisons on the Delhi graph — microseconds — and works on any graph carrying
coordinates, including synthetic test graphs, which declare no CRS for a spatial
index to interpret.

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

## VRP layer

`qgati.optimizer` solves the Vehicle Routing Problem on top of the cost matrix —
it never touches the road graph.

```python
from qgati.graph import build_cost_matrix
from qgati.optimizer import build_random_scenario, solve_brute_force, clarke_wright_savings, evaluate

scenario = build_random_scenario(graph, n_deliveries=6, n_vehicles=2, seed=1)
costs    = build_cost_matrix(graph, scenario)     # dense (n+1)^2 travel-time array

optimal  = solve_brute_force(scenario, costs)     # exact, <= 10 deliveries
heuristic = clarke_wright_savings(scenario, costs)

print(evaluate(optimal, scenario, costs).summary())
```

| Piece                  | Role                                                        |
| ---------------------- | ----------------------------------------------------------- |
| `models`               | `Depot`, `Delivery`, `Vehicle`, `Scenario`, `Solution` — the contract every solver consumes |
| `scenarios`            | `build_random_scenario` — draws only from the largest strongly-connected subgraph |
| `fitness`              | `evaluate` — shared cost + penalty function all solvers are scored by |
| `decoding`             | permutation/random-keys → routes, via Prins' optimal capacity split — shared by every metaheuristic |
| `brute_force`          | Exact (Held-Karp + partition search), ground truth for small instances |
| `savings`              | Clarke-Wright Savings — the constructive baseline           |
| `qpso`                 | Quantum-behaved PSO — the research contribution              |
| `classical_pso`        | Velocity-driven PSO — the control that isolates the quantum update |
| `genetic_algorithm`    | GA — order crossover + swap mutation                        |
| `aco`                  | Ant Colony Optimization — pheromone construction           |

`Solution.routes` always holds **exactly one entry per vehicle** (unused vehicles
empty), so "more vehicles than the fleet has" is unrepresentable rather than
something each solver must remember to check. Routes list *delivery indices*,
not node ids — which keeps two deliveries at one address distinguishable.

### Two invariants worth knowing

**Scenarios are always servable.** `build_random_scenario` samples only from
`largest_strongly_connected_subgraph`. A real extract contains one-way stubs you
can enter but not leave (37 of 2033 nodes in the current Delhi extract); a
delivery placed on one would have no feasible tour, which is a modelling error
that would surface as a mysterious optimizer failure. `build_cost_matrix` raises
on an unreachable pair rather than substituting `inf`, for the same reason.

**Infeasible can never outscore feasible.** `evaluate` returns travel cost plus
capacity, coverage and shape penalties, with weights auto-scaled to the matrix
magnitude — so no constant tuning is needed when moving between toy costs and
real Delhi travel times.

### Baseline quality

Against the exact solver on 40 sampled Delhi instances, Clarke-Wright lands a
median ~7-9% above optimal (0% on a symmetric euclidean equivalent). The gap
comes from one-way streets: Delhi scenario matrices average ~12% relative
asymmetry, and the classical savings formula is a symmetric one.

## QPSO

`qgati.optimizer.qpso` is the project's research contribution: the genuine quantum-behaved
particle swarm (Sun et al., 2004), not classical PSO renamed. Particles are
sampled from a quantum potential well rather than driven by velocity.

```python
from qgati.optimizer import run_qpso

solution, cost, history = run_qpso(costs, scenario, seed=42)
# history[i] = best-so-far cost after generation i — feed it to the explain layer
```

`num_particles` (30), `num_iterations` (100) and the contraction-expansion
schedule (`beta_start` 1.0 → `beta_end` 0.5) are all configurable. A given seed
reproduces a run exactly.

### Encoding — the decision that matters most

QPSO is a **continuous** optimizer; the VRP is combinatorial. Two choices bridge
that gap:

**Positions are random keys.** A particle is `n` reals; sorting them gives a
delivery permutation. This keeps positions continuous so the quantum update
applies unchanged, while every real vector decodes to a valid permutation — and
it is order-preserving, so nearby positions mean similar routes. That locality is
what QPSO's gradient-free guidance depends on; a direct integer encoding would
destroy it.

**Route boundaries come from an optimal split, not encoded split points.** Fixed
split points would leave capacity feasibility to the search, making QPSO repair
overloaded vehicles instead of shortening routes. Instead a short DP (Prins'
split) cuts the permutation into the cheapest at-most-*k* capacity-feasible
routes. Capacity holds by construction, and because the split is solved *exactly*
rather than heuristically, it dominates any fixed split points the swarm could
have encoded.

### Results

Against the exact optimum on the real Delhi graph (30 particles, 100 iterations,
5 seeds):

| Instance | Brute force | Savings | QPSO best |
| --- | --- | --- | --- |
| n=6, k=2 | 1123.6 s | 1303.7 s (+16.0%) | **1123.6 s (+0.0%)** |
| n=8, k=3 | 1412.3 s | 1603.1 s (+13.5%) | **1412.3 s (+0.0%)** |
| n=10, k=3 | 1318.5 s | 1428.1 s (+8.3%) | **1318.5 s (+0.0%)** |

Beyond exact reach, QPSO beats Savings by ~10% (n=15: 1838.4 s vs 2054.5 s;
n=20: 2604.7 s vs 2876.6 s).

At the 100-iteration default, individual runs land within ~2% of optimal; hitting
the optimum in *most* runs needs ~300 iterations (measured on n=8: 5/10 exact at
100, 6/10 at 200, 9/10 at 300).

## Baseline solvers

Four metaheuristics now sit behind one contract, so they can be compared on
identical instances — see [Benchmarks](#benchmarks).

| Solver | Operators / update | Role |
| --- | --- | --- |
| `run_qpso` | quantum well sampling, `beta` 1.0 → 0.5 | the research contribution |
| `run_classical_pso` | `v = w·v + c₁r₁(pbest−x) + c₂r₂(gbest−x)`, `w` 0.9 → 0.4 | **the control** — isolates what the quantum update adds |
| `run_genetic_algorithm` | k-way tournament, order crossover (OX1), swap mutation, 1 elite | evolutionary baseline |
| `run_aco` | `tau^α · (1/cost)^β` roulette construction, evaporation 0.1 + reinforcement | pheromone baseline |

All four take `(cost_matrix, scenario, ...)`, return
`(solution, cost, convergence_history)`, and decode through `qgati.optimizer.decoding`.

```python
from qgati.optimizer import run_qpso, run_classical_pso, run_genetic_algorithm, run_aco

for run in (run_qpso, run_classical_pso, run_genetic_algorithm, run_aco):
    solution, cost, history = run(costs, scenario, seed=42)
    print(run.__name__, cost)
```

### Why classical PSO is here

`classical_pso` exists only to make the QPSO claim falsifiable. It shares QPSO's
representation, decoder, cost function, initialisation and contraction schedule
(`w` 0.9 → 0.4 against QPSO's `beta` 1.0 → 0.5), and even **imports** QPSO's
`num_particles` / `num_iterations` constants rather than restating them, so
tuning one solver cannot silently stop the pair from being comparable. The only
difference is the update rule: classical PSO moves particles through an
accumulated velocity, QPSO samples the next position from a probability
distribution and can therefore tunnel out of a local optimum in one step.

### Why every metaheuristic shares one decoder

Capacity behaviour lives in `decoding`, not in the solvers. Every candidate is
made feasible by construction — the permutation is cut by an exact split, and
ACO's ants filter their candidate set to what fits the current vehicle. The
result is that a difference between two rows of the benchmark table is a
difference in *search quality*, not in how well a solver happened to handle
capacity. It also means all four return feasible solutions on every instance
measured so far, with no repair step anywhere.

## Benchmarks

`benchmarks/run_comparison.py` runs every solver on the same cost matrices and
prints a comparison table plus an iteration sweep.

```bash
cd backend
uv run python benchmarks/run_comparison.py            # synthetic graph, n=8/15/25
uv run python benchmarks/run_comparison.py --real     # real Delhi graph
uv run python benchmarks/run_comparison.py --quick     # smoke test
```

Results land in `benchmarks/results/` as a timestamped set: a `.json` with the
complete record (every per-seed cost and runtime) plus `_comparison.csv` and
`_sweep.csv` for pasting into a report.

Two methodological notes, because they affect how the numbers should be read:

**Timings use min-of-seeds as well as the mean.** A laptop throttles under
sustained load, and a first version of this runner showed ACO at 511 ms in one
table and 109 ms in another for the identical work — an artifact of it always
being measured last. The runner now interleaves solvers per seed so they all
sample the same thermal conditions, and reports `ms min` alongside `ms mean`.
Compare solvers on `ms min`; the gap between them is how much the machine was
throttling.

**The sweep is the evidence for the default budget.** 30 particles × 100
iterations is a choice, not a law, and the sweep is what defends it — see the
iteration-sweep table in the results.

### Results — 5 seeds, equal iteration budget

Synthetic graph (45 nodes, `edge_prob` 0.22, seed 11), population 30 for all four
solvers. `best` / `mean` are travel cost; lower is better.

**n=8 k=3 — exact optimum 8025.4**

| Solver | best | mean | exact hits | ms min |
| --- | --- | --- | --- | --- |
| brute force | 8025.4 | 8025.4 | 1/1 | 3 |
| savings | 10000.1 | 10000.1 | 0/1 | 0 |
| QPSO | 8025.4 | 8078.0 | 3/5 | 145 |
| Classical PSO | 8025.4 | 8205.0 | 1/5 | 141 |
| Genetic Algorithm | 8025.4 | **8025.4** | **5/5** | 199 |
| ACO | 8025.4 | **8025.4** | **5/5** | 275 |

**n=15 k=3 — no exact answer available**

| Solver | best | mean | gap vs best | ms min |
| --- | --- | --- | --- | --- |
| savings | 12056.0 | 12056.0 | +23.9% | 0 |
| QPSO | **9728.4** | 10365.9 | 0.0% | 231 |
| Classical PSO | 10500.6 | 11348.8 | +7.9% | 227 |
| Genetic Algorithm | 10240.3 | 10794.3 | +5.3% | 293 |
| ACO | 9768.7 | **9906.9** | +0.4% | 542 |

**n=25 k=5 — no exact answer available**

| Solver | best | mean | gap vs best | ms min |
| --- | --- | --- | --- | --- |
| savings | 23118.1 | 23118.1 | +42.8% | 0 |
| QPSO | 20485.4 | 21386.1 | +26.5% | 505 |
| Classical PSO | 18226.5 | 19161.9 | +12.6% | 500 |
| Genetic Algorithm | 17945.5 | 18725.2 | +10.8% | 580 |
| ACO | **16190.4** | **16343.2** | 0.0% | 1037 |

### What the benchmark actually shows

QPSO is the production default because the problem statement names it as this
project's focus — quantum-inspired search, benchmarked against conventional
metaheuristics and exact methods. That is a requirement, and it is why the default
is QPSO rather than whichever solver happened to score lowest. The benchmark's job
is to report honestly how the required algorithm performs, and that includes where
it does not win.

**On raw cost, ACO is the strongest solver here, and by a wide margin at n=25.**
Its mean of 16,343.2 against QPSO's 21,386.1 is a 23.6% gap, and it holds the best
mean at every size (tied with GA at n=8), despite costing about twice the wall time
per iteration. GA is second at n=25. QPSO is second at n=15 and last of the four
metaheuristics at n=25 under the 100-iteration budget. Stated plainly rather than
buried: on final cost alone, the required algorithm is not the best one in the
table.

Two measured results qualify that, and neither is an argument from intent.

**QPSO beats classical PSO — the controlled comparison.** The two share one
representation and one decoder and differ *only* in the update rule, quantum
sampling against the classical velocity update, so a gap between them is
attributable to the quantum sampling and not to encoding or budget. At equal budget
QPSO wins: at n=8 it reaches the exact optimum in 3/5 runs against classical PSO's
1/5, and at n=25 with 3,000 iterations it is 5.9% better (17,359.8 against
18,446.3). This is the problem statement's own claim about quantum-inspired
sampling, and it is the comparison that actually tests it.

**QPSO is a long-budget technique, so the 100-iteration tables understate it.** The
same sweep shows QPSO improving 19% from 100 to 3,000 iterations — the most
budget-sensitive solver in the set — where ACO moves 0.4% over the same 30×. At the
100-iteration default there is not enough budget for the quantum update to pay off,
and QPSO sits *behind* classical PSO (21,386 against 19,162); by 3,000 iterations it
is clearly ahead. The mechanism fits: broad quantum sampling explores widely,
converging more slowly early and better asymptotically. The same pattern shows in
miniature at n=15, where QPSO produced the best single run of any solver (9,728.4
against ACO's 9,768.7) even though ACO's mean there is better (9,906.9 against
10,365.9).

**ACO is nearly budget-insensitive**, which is the more interesting result of the
two: 16,343 → 16,283 mean over 100 → 3,000 iterations, a 0.4% gain for 30× the
compute. Its strength is the greedy `1/cost^2` construction, not the pheromone
learning, and it is effectively a very strong constructive heuristic wearing an
ACO's clothes.

### The equal-budget sweep, n=25 k=5 (mean cost)

| iterations | QPSO | Classical PSO | Genetic Algorithm | ACO |
| --- | --- | --- | --- | --- |
| 100 | 21386.1 | 19161.9 | 18725.2 | **16343.2** |
| 300 | 19496.8 | 19135.5 | 17984.6 | **16343.2** |
| 1000 | 17920.6 | 18305.7 | 17278.0 | **16322.8** |
| 3000 | 17359.8 | 18446.3 | 16797.3 | **16282.6** |

QPSO is the most budget-sensitive solver (a 19% improvement from 100 → 3,000
iterations) and still finishes third. One caveat on the timing column of this
sweep: the run that produced it straddled a laptop suspend, so the `ms mean` of
one cell (`ACO @ 300`) is inflated by ~6.7 hours of wall clock. Quality figures
are unaffected — they are deterministic per seed and reproduced bit-identically
across two independent runs — and `ms min` stays valid, which is why it exists.

## Traffic simulation

`qgati.traffic` prices the road network under simulated conditions *before* the
optimizer runs, so a route computed at 9am is not the route computed at 3pm.

**This is a simulation for demonstrating dynamic routing, not real-world traffic
data.** The factors below are plausible round numbers, not calibrated against
measured Delhi traffic.

| Condition | Factor |
| --- | --- |
| Peak hour, 08:00–10:00 and 17:00–20:00 | ×1.70 on through roads, ×1.21 on side streets |
| Rain | ×1.40, every road |
| Accident | ×3.00, on the named roads only |
| Road closure | impassable |
| Combinations | multiply — peak rain on an accident road is 7.14× |

Conditions are fixed when a scenario is created (`POST /scenarios` takes an
optional `conditions` block) and recorded with it. The same `scenario_id` always
optimizes the same costs, which is what keeps a solver comparison on it
meaningful; create two scenarios from one seed with different timestamps to
compare peak against off-peak.

### A flat multiplier cannot reroute anything

This is the finding that shaped the model. A tour is a sum of legs. Scale every
edge by 1.7 and every tour scales by 1.7, so **the cheapest tour is still the
cheapest tour** — a uniform peak factor returns identical routes at a higher
price. A flat ×1.7 would have made the dynamic routing this layer exists to
demonstrate impossible to show.

Peak congestion is therefore weighted by road class, using the OSM `highway` tag
already on every edge:

```
peak_multiplier(edge) = 1 + (1.7 - 1) * sensitivity(edge)

  motorway / trunk / primary / secondary / tertiary (+ _link)   sensitivity 1.00 -> x1.70
  residential / living_street / service / unclassified, unknown  sensitivity 0.30 -> x1.21
```

The split is load-bearing on this extract rather than cosmetic. Through roads run
at 48.6 kph against 35.0 kph on residential streets, so off-peak an arterial is
**1.39× faster**; under peak their effective speeds converge to 28.6 against
28.9 kph, and the arterial's advantage disappears. Routing moves onto side
streets. `PEAK_FACTOR` is the through-road figure, so the headline ×1.7 is
exactly what an arterial takes.

**Rain is applied flat at ×1.4, and by the same argument it reroutes nothing on
its own** — it raises every route by exactly 40%. That is a property of the
model, not a defect: rain changes the *price* of a route, and only bites on
*routing* in combination with something road-dependent, like peak hour or an
accident. `test_rain_alone_scales_every_tour_equally` pins it.

Road closures are never implemented by deleting edges. The Delhi graph is a
cached, shared, read-only object, so a deletion would leak into every later
request; instead a closed edge is given infinite weight, which Dijkstra discards
for the same reason it discards any unaffordable edge. The graph is only read.

### Traffic log

Every scenario creation writes a row per affected road to
`data/traffic_log.db` (SQLite), as a side effect of ordinary use — there is no
separate step to run. `GET /traffic/log` reads it back, paginated.

| Column | |
| --- | --- |
| `road_id`, `road_u`, `road_v` | the directed edge, as `"u->v"` and as its endpoints |
| `timestamp`, `day_of_week`, `time_of_day` | when the condition was applied |
| `weather_condition`, `traffic_condition` | `clear`/`rain`, `peak`/`off_peak` |
| `incident_type` | `accident`, `road_closure`, or `NULL` |
| `travel_time` | seconds under these conditions; `NULL` for a closed road |

Two kinds of road are logged, and both are needed. **Traversed roads** are every
edge the scenario's cheapest paths run along, so each row is a road a real route
weighed rather than an arbitrary slice of the graph; pricing the same instance
off-peak and at peak logs the same roads under different conditions, which is
what makes the table *paired*. **Incident roads** are logged whether or not any
route used them — a closed edge has infinite weight, so no cheapest path can ever
include it, and without this half the `road_closure` value would never once
appear.

A closed road logs `travel_time` as `NULL`: it is impassable, so there is no
travel time to record, and `incident_type` carries the reason.

**No prediction is built on this data.** There is no model, no training pipeline
and no "predicted travel time" field — the log exists so that a later phase has a
`(road, time, weather, incident) -> travel_time` history to learn from.

## Tests

```bash
cd backend
uv run pytest                  # 175 fast tests, no network
```

The suite runs against a synthetic random graph, so it is fast and offline.
Three tests exercise the real Delhi graph and are skipped by default; opt in
with:

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
| `qgati.traffic`     | Rule-based traffic simulator, and the log it collects         |
| `qgati.reopt`       | Adaptive partial re-optimization                             |
| `qgati.explain`     | Decision traces / explainability                             |
| `qgati.api`         | FastAPI application — schemas, in-memory store, routes       |

`data/` holds cached graphs and scenario configs. `data/cache/` is gitignored: it
holds the fetched `.graphml`, plus `data/cache/osm_http/` for OSMnx's raw
Overpass responses. `data/traffic_log.db` is gitignored too — it is collected at
runtime and grows with use.
