# Smart-Gati Backend

A **multi-algorithm VRP framework** — five solvers behind one contract, benchmarked
against each other on identical cost matrices — plus the FastAPI service around
it. **QPSO is the production default.** The problem statement (SIH PS 26137) names
quantum-inspired search as the focus of this work — benchmarked against
conventional metaheuristics and exact methods — so QPSO is the solver the API
serves, and the other four exist to measure it against: three conventional
metaheuristics (Savings, GA, classical PSO) and brute force as exact ground
truth.

The benchmark is reported in full, including where QPSO loses. **ACO was
benchmarked here too and reached a lower mean raw cost than QPSO at n=15 and
n=25; it has since been removed from the project.** Its figures are kept in the
tables below as measurements, and the finding is recorded in
[`DESIGN_DECISIONS.md`](../DESIGN_DECISIONS.md#removed-aco) — with ACO gone,
"QPSO is the best of these" is a claim about a smaller field. What the remaining
results do support is the comparison QPSO was chosen for — at equal budget it
beats classical PSO, which shares its representation and differs *only* in the
update rule, so the gap is attributable to the quantum sampling itself. See
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
| `POST` | `/scenarios/{scenario_id}/incident` | Inject a live incident — a closed or slow road — into a stored scenario. Re-prices it; does not re-optimize |
| `DELETE` | `/scenarios/{scenario_id}/incident/{incident_id}` | Revert one incident, restoring the conditions it replaced |
| `POST` | `/scenarios/{scenario_id}/detect` | Judge one observed travel time against that road's logged history. Flags an anomaly; does not re-optimize |
| `POST` | `/scenarios/{scenario_id}/watcher` | Dispatch a simulated GPS fleet: solve the scenario, put its vehicles on the road, and start measuring every `interval_seconds` |
| `GET` | `/scenarios/{scenario_id}/watcher` | What that fleet is doing, and where each vehicle currently is. Does not tick |
| `POST` | `/scenarios/{scenario_id}/watcher/tick` | Advance the fleet one interval, synchronously — the same call the timer thread makes |
| `DELETE` | `/scenarios/{scenario_id}/watcher` | Stop the fleet. The readings it took stay in the scenario and in the log |
| `POST` | `/scenarios/{scenario_id}/reoptimize` | Re-plan **only the unserved deliveries** of a scenario whose fleet is out, from each vehicle's current position. `409` unless an incident or a flagged reading justifies it. Writes nothing |
| `POST` | `/scenarios/{scenario_id}/vehicles/{vehicle_id}/avoid-road` | A driver names a road and a treatment and is re-planned around it, **alone** — every other vehicle keeps its route. Files the report as a real, revertable incident, then re-plans only that vehicle's remaining stops |
| `GET` | `/graph/delhi` | The road network as GeoJSON, optionally scoped to a scenario or a bbox |
| `POST` | `/optimize/{scenario_id}` | Run one solver on a stored scenario |
| `GET` | `/optimize/{scenario_id}/compare` | Run every solver on the same cost matrix |
| `GET` | `/traffic/log` | Inspect the collected road-condition log, paginated |
| `GET` | `/analytics/runs` | The run history — every recorded solve, newest first. Paginated, and filterable by kind, solver or scenario |
| `GET` | `/analytics/summary` | Aggregate statistics over the whole run history — counts by kind and solver, runtime, and what re-planning has saved |

**`POST /optimize/{scenario_id}` defaults to QPSO**, the production solver. Note
that the scenario is named by the **URL**, never the body: the body carries only
*how* to solve, so the two kinds of input cannot drift into one payload. `GET
/optimize/{id}/compare` runs all five — that is where QPSO's headline result sits
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
road-graph nodes, so it is directly renderable. `travel_cost` is the combined
objective in **rupees**, and the three raw components it was built from are
reported next to it so the total can be checked rather than taken on trust:

```json
{
  "solver": "qpso", "cost": <rupees>, "travel_cost": <rupees>, "feasible": true,
  "travel_time": <seconds>, "distance_m": <metres>, "fuel_litres": <litres>,
  "runtime_ms": 456, "iterations": 100, "population": 30,
  "routes": [
    {"vehicle_id": "V0", "load": 29, "capacity": 29,
     "travel_cost": <rupees>, "travel_time": <seconds>,
     "distance_m": <metres>, "fuel_litres": <litres>,
     "stops": [{"delivery_id": "D1", "node": 928490302}, ...]}
  ]
}
```

> The numeric fields are left blank deliberately: the captured response this
> block used to show was produced under the time-only objective, so every cost in
> it is stale *and* in the wrong unit. Re-running the curl above regenerates it.

`cost` is `travel_cost` plus penalties, so the two are equal on a feasible
answer. `travel_cost` is **not** seconds — it was, before the objective combined
all three goals, and any caller reading it as a duration is now wrong by a factor
of about 100.

Compare, on a scenario small enough for an exact answer — an `n=10, k=3`
instance created with `{"kind":"generate","n_deliveries":10,"n_vehicles":3,
"seed":21}` and solved at solver `seed=0`:

> The table below is from the pre-change run: the objective was travel time in
> seconds, so these are seconds, not rupees, and the gap column is a gap on time
> alone. Re-run `/compare` on the current build to get the combined-objective
> figures. It also predates the removal of ACO, so it has one row fewer than it
> did.

```
GET /optimize/{scenario_id}/compare?seed=0

exact optimum: 1492.7
solver               travel_cost     ms gap_opt%
Brute Force               1492.7     25     0.00
Savings                   1622.3      2     8.68
Genetic Algorithm         1492.7    215     0.00
Classical PSO             1622.3    144     8.69
QPSO                      1502.3    149     0.65
```

Read that as one instance, not a ranking: at `n=10` three of the five solvers land
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
costs    = build_cost_matrix(graph, scenario)     # dense (n+1)^2 rupee objective

optimal  = solve_brute_force(scenario, costs)     # exact, <= 10 deliveries
heuristic = clarke_wright_savings(scenario, costs)

print(evaluate(optimal, scenario, costs).summary())
# cost=Rs… [time=…s distance=…m fuel=…L] feasible
```

`costs.objective_matrix` — not `costs.matrix` — is what every solver minimises.
`costs.matrix` keeps its original meaning (travel **seconds**), because the
traffic layer logs travel times and would misreport if that changed underneath
it. `costs.distance_matrix` and `costs.fuel_matrix` carry the other two
components, and `costs.weights` records the prices that combined them.

| Piece                  | Role                                                        |
| ---------------------- | ----------------------------------------------------------- |
| `models`               | `Depot`, `Delivery`, `Vehicle`, `Scenario`, `Solution` — the contract every solver consumes |
| `scenarios`            | `build_random_scenario` — draws only from the largest strongly-connected subgraph |
| `fitness`              | `evaluate` — shared objective + penalty function all solvers are scored by |
| `objective`            | the prices turning time, distance and fuel into one rupee objective; the fuel curve |
| `decoding`             | permutation/random-keys → routes, via Prins' optimal capacity split — shared by every metaheuristic |
| `brute_force`          | Exact (Held-Karp + partition search), ground truth for small instances |
| `savings`              | Clarke-Wright Savings — the constructive baseline           |
| `qpso`                 | Quantum-behaved PSO — the research contribution              |
| `classical_pso`        | Velocity-driven PSO — the control that isolates the quantum update |
| `genetic_algorithm`    | GA — order crossover + swap mutation                        |

ACO was a sixth solver here and has been **removed from the project** — the module
is a tombstone and nothing imports it. Its benchmark figures survive in the tables
below; see [`DESIGN_DECISIONS.md`](../DESIGN_DECISIONS.md#removed-aco).

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

**Infeasible can never outscore feasible.** `evaluate` returns the weighted travel
cost plus capacity, coverage and shape penalties, with weights auto-scaled to the
matrix magnitude — so no constant tuning is needed when moving between toy costs
and real Delhi costs. The penalties scale off the *objective*, not off travel
seconds: anchored to seconds against a rupee objective they would be about a
hundredth of the saving they exist to outweigh.

**One objective, formed once.** Time, distance and fuel are priced into a single
rupee figure when the cost matrix is built (`optimizer/objective.py`), and every
solver minimises that same array. None of them gets a chance to weight the three
goals differently, which is what keeps the benchmark below a comparison of search
algorithms rather than of accounting. See `DESIGN_DECISIONS.md` → "Objective
function" for the prices, their basis, and the fuel model's assumptions.

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

Three metaheuristics now sit behind one contract, so they can be compared on
identical instances — see [Benchmarks](#benchmarks).

| Solver | Operators / update | Role |
| --- | --- | --- |
| `run_qpso` | quantum well sampling, `beta` 1.0 → 0.5 | the research contribution |
| `run_classical_pso` | `v = w·v + c₁r₁(pbest−x) + c₂r₂(gbest−x)`, `w` 0.9 → 0.4 | **the control** — isolates what the quantum update adds |
| `run_genetic_algorithm` | k-way tournament, order crossover (OX1), swap mutation, 1 elite | evolutionary baseline |

All three take `(cost_matrix, scenario, ...)`, return
`(solution, cost, convergence_history)`, and decode through `qgati.optimizer.decoding`.

```python
from qgati.optimizer import run_qpso, run_classical_pso, run_genetic_algorithm

for run in (run_qpso, run_classical_pso, run_genetic_algorithm):
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
made feasible by construction — the permutation is cut by an exact split. The
result is that a difference between two rows of the benchmark table is a
difference in *search quality*, not in how well a solver happened to handle
capacity. It also means all three return feasible solutions on every instance
measured so far, with no repair step anywhere.

## Benchmarks

`benchmarks/run_comparison.py` runs every solver on the same cost matrices and
prints a comparison table plus an iteration sweep. It reports the combined
objective, so the numbers it prints are rupees.

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
sustained load, and a first version of this runner showed one solver at 511 ms in
one table and 109 ms in another for the identical work — an artifact of it always
being measured last. The runner now interleaves solvers per seed so they all
sample the same thermal conditions, and reports `ms min` alongside `ms mean`.
Compare solvers on `ms min`; the gap between them is how much the machine was
throttling.

**The sweep is the evidence for the default budget.** 30 particles × 100
iterations is a choice, not a law, and the sweep is what defends it — see the
iteration-sweep table in the results.

### Results — 5 seeds, equal iteration budget

> **These tables predate the combined objective and have not been re-measured.**
> Every figure below was produced when the objective was travel time alone, so the
> absolute values are in seconds and are not comparable to the rupee costs the
> code now reports. More importantly the *ranking* is not guaranteed to survive:
> the solvers are now minimising a different function, and a method that was
> strongest on time need not be strongest on time + distance + fuel. The
> qualitative findings in the next section should be treated as unverified until
> `uv run python benchmarks/run_comparison.py` is re-run, which writes fresh
> `.json`/`.csv` results with the new objective.

> **ACO rows below are measurements of a solver that has since been removed.** They
> are kept rather than struck out because they are the only record of the strongest
> counter-example to QPSO in this repository — see
> [`DESIGN_DECISIONS.md`](../DESIGN_DECISIONS.md#removed-aco). The command above no
> longer reproduces them.

Synthetic graph (45 nodes, `edge_prob` 0.22, seed 11), population 30 for every
solver. `best` / `mean` are travel cost — travel **seconds** under this run's
objective; lower is better.

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

> Carried over from the time-only run described above; see the banner there. The
> argument below is about *which comparison is controlled*, and that part does not
> depend on the objective — but the numbers do.

QPSO is the production default because the problem statement names it as this
project's focus — quantum-inspired search, benchmarked against conventional
metaheuristics and exact methods. That is a requirement, and it is why the default
is QPSO rather than whichever solver happened to score lowest. The benchmark's job
is to report honestly how the required algorithm performs, and that includes where
it does not win.

**On raw cost, ACO was the strongest solver here, and by a wide margin at n=25.**
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
budget-sensitive solver in the set — where ACO moved 0.4% over the same 30×. At the
100-iteration default there is not enough budget for the quantum update to pay off,
and QPSO sits *behind* classical PSO (21,386 against 19,162); by 3,000 iterations it
is clearly ahead. The mechanism fits: broad quantum sampling explores widely,
converging more slowly early and better asymptotically. The same pattern shows in
miniature at n=15, where QPSO produced the best single run of any solver (9,728.4
against ACO's 9,768.7) even though ACO's mean there is better (9,906.9 against
10,365.9).

**ACO was nearly budget-insensitive**, which is the more interesting result of the
two: 16,343 → 16,283 mean over 100 → 3,000 iterations, a 0.4% gain for 30× the
compute. Its strength was the greedy `1/cost^2` construction, not the pheromone
learning, and it was effectively a very strong constructive heuristic wearing
ACO's clothes. It is worth knowing that the removed solver was the *robust* one,
not the fragile one.

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
optimizer runs, so a route computed at 9am is not the route computed at 2pm.

**This is a simulation for demonstrating dynamic routing, not real-world traffic
data.** The factors are anchored to the TomTom Traffic Index 2025 figure for New
Delhi, but the road-class split below is a modelling choice, not a measurement.

Time of day alone picks the state — there is no operator toggle, so a demo left
running unattended always prices under something sensible:

| State | Window | Factor |
| --- | --- | --- |
| Normal | outside the daytime band | ×1.0, every road |
| Moderate | 06:00–22:00, outside the peak windows | ×1.6 on through roads, ×1.18 on side streets |
| Peak | 08:00–10:00 and 17:00–20:00 | ×2.9 on through roads, ×1.57 on side streets |
| Accident | the named roads only | ×2.9, flat |
| Road closure | the named roads only | impassable |

The congestion figures come from the TomTom Traffic Index 2025 for New Delhi:
average congestion 60.2%, which is a travel time of 1.602× free-flow and rounds
to 1.6, and 192% at the 6pm peak, which is 2.92× and rounds to 2.9. The peak
windows are the demo's "morning and evening rush". The 06:00–22:00 daytime
boundary is a modelling choice and **not** a TomTom figure — the index publishes
congestion by hour, not a definition of daytime.

An accident **replaces** that edge's congestion multiplier with a flat ×2.9
rather than compounding with it. ×2.9 is already the conservative placeholder
for a delay nobody has measured, so stacking peak hour on top would charge the
same congestion twice. It is applied flat across road classes, unlike congestion
itself, which is scaled — erring high on an unmeasured delay is the safe
direction.

Conditions are fixed when a scenario is created (`POST /scenarios` takes an
optional `conditions` block) and recorded with it. The same `scenario_id` always
optimizes the same costs, which is what keeps a solver comparison on it
meaningful; create two scenarios from one seed with different timestamps to
compare two states.

### A flat multiplier cannot reroute anything

This is the finding that shaped the model. A tour is a sum of legs. Scale every
edge by 2.9 and every tour scales by 2.9, so **the cheapest tour is still the
cheapest tour** — a uniform congestion factor returns identical routes at a
higher price. A flat factor would have made the dynamic routing this layer
exists to demonstrate impossible to show.

Congestion is therefore weighted by road class, using the OSM `highway` tag
already on every edge:

```
congestion_multiplier(edge) = 1 + (FACTOR - 1) * sensitivity(edge)

  motorway / trunk / primary / secondary / tertiary (+ _link)   sensitivity 1.00 -> x2.90 peak, x1.60 moderate
  residential / living_street / service / unclassified, unknown  sensitivity 0.30 -> x1.57 peak, x1.18 moderate
```

The split is load-bearing on this extract rather than cosmetic. Through roads run
at 48.6 kph against 35.0 kph on residential streets, so in the normal band an
arterial is **1.39× faster**. Under moderate the two nearly converge — 30.4
against 29.7 kph — and at peak the side street is plainly the quicker road, 22.3
against 16.8 kph. Routing moves onto side streets. `PEAK_FACTOR` is the
through-road figure, so the headline ×2.9 is exactly what an arterial takes.

Rain used to take a flat ×1.4 here. It has no factor any more: by the invariance
above, a flat factor reroutes nothing, so it was a price rise wearing a
condition's clothes.

Road closures are never implemented by deleting edges. The Delhi graph is a
cached, shared, read-only object, so a deletion would leak into every later
request; instead a closed edge is given infinite weight, which Dijkstra discards
for the same reason it discards any unaffordable edge. The graph is only read.

### Traffic log

Every scenario creation writes a row per affected road to
`data/traffic_log.db` (SQLite), as a side effect of ordinary use — there is no
separate step to run. Applying or reverting an incident writes one more, for the
incident edge alone. `GET /traffic/log` reads it back, paginated.

| Column | |
| --- | --- |
| `road_id`, `road_u`, `road_v` | the directed edge, as `"u->v"` and as its endpoints |
| `timestamp`, `day_of_week`, `time_of_day` | when the condition was applied |
| `traffic_condition` | `normal` / `moderate` / `peak` |
| `incident_type` | `accident`, `road_closure`, `closure`, `slow`, `road_clear`, or `NULL` |
| `travel_time` | seconds under these conditions; `NULL` for a closed road |

Two kinds of road are logged, and both are needed. **Traversed roads** are every
edge the scenario's cheapest paths run along, so each row is a road a real route
weighed rather than an arbitrary slice of the graph; pricing the same instance
at moderate and at peak logs the same roads under different conditions, which is
what makes the table *paired*. **Incident roads** are logged whether or not any
route used them — a closed edge has infinite weight, so no cheapest path can ever
include it, and without this half the `road_closure` value would never once
appear.

A closed road logs `travel_time` as `NULL`: it is impassable, so there is no
travel time to record, and `incident_type` carries the reason.

`incident_type` records **what was reported**, which is why it holds five values
in two generations. `accident` and `road_closure` come from the `conditions`
block a scenario was created with, where the caller names the *cause*.
`closure`, `slow` and `road_clear` come from the live-incident routes below,
where an operator names what they saw — including `road_clear` when they lift an
override. `accident` and `slow` are the same effect (a flat ×2.9) under two
names, and `road_closure` and `closure` likewise. Renaming the older pair to
match would have split an accumulating dataset across two spellings of the same
state, so the older rows were left alone and the report's own word is used
alongside them.

### Live incidents

`POST /scenarios/{id}/incident` takes an `incident_type` of `"closure"` or
`"slow"` and a directed `edge`, and applies the corresponding multiplier to that
road **for that scenario only**:

| Report | Effect |
| --- | --- |
| `closure` | edge weight ∞ — impassable |
| `slow` | edge base time ×2.9, flat |

The scenario's cost matrix is rebuilt and one audit row is written for the
incident edge, stamped with the scenario's own timestamp so it pairs with the
rows already collected for it. `DELETE .../incident/{incident_id}` reverts it,
restoring the conditions the scenario was created with — a revert is exact, not
subtractive, so any accident or closure the scenario was born with survives.

**Neither route runs a solver.** An incident moves the *costs*; routes a client
already holds stay where they are until it asks for a new solution with
`POST /optimize/{id}`. Re-optimizing in response to an incident is a separate
concern with its own module.

Two consequences worth knowing:

- The response's `changed_legs` counts objective-matrix entries that moved, and
  it can legitimately be **zero** — an incident on a road no cheapest path uses
  changes nothing about that instance.
- A `slow` report on a through road **at peak changes nothing at all**, because
  the incident factor *replaces* the edge's congestion multiplier rather than
  compounding with it, and both are ×2.9. Inject at normal or moderate to see a
  difference.

Re-pricing is not re-collection: the scenario's roads were logged when it was
priced, so a mutation writes the one row it is actually about rather than a
hundred near-duplicates.

### Anomaly detection

`POST /scenarios/{id}/detect` takes an observed `travel_time` for a directed road
and answers one question: **is this road slower than it normally is?** Plain
statistics over the rows already in `traffic_log` — `statistics.fmean` and
`statistics.stdev` — with no model and no training.

```json
{"edge": {"u": 0, "v": 1}, "travel_time": 12.0, "timestamp": null}
```

```json
{
  "scenario_id": "…", "edge": {"u": 0, "v": 1},
  "observed_travel_time": 12.0, "condition": "normal",
  "rule": "z_score", "flagged": true,
  "expected_travel_time": 10.0,
  "sample_count": 15, "mean": 10.4, "std_dev": 0.5, "z_score": 3.2,
  "threshold": 2.0, "reason": "12.0s is 3.2 standard deviations above …"
}
```

Two rules decide, and the verdict says which one ran:

| Rule | When | Flags when |
| --- | --- | --- |
| `z_score` | ≥ 10 samples for this `(road, condition)` key | `z > 2` |
| `insufficient_samples` | fewer than 10 — **the z-score is skipped entirely** | `observed > 1.2 × expected` |

History is keyed per **`(road, condition)`**, so a 09:00 reading is compared with
that road's own 09:00-band history rather than with a pooled figure in which peak
*is* the anomaly. The price of that is fewer samples per key, which is why every
verdict returns `sample_count`.

Four things about this route are deliberate:

- **It never runs a solver, and writes nothing.** A flag is a signal for the
  re-optimization phase to act on, not an action; the response model has no field
  a route, a cost or a solver could be returned in. Recording the observation
  would feed the anomaly into the baseline meant to catch it, so ingestion is
  `POST /scenarios/{id}/watcher`'s job — see **Simulated GPS watcher** below.
- **`expected` excludes live incidents.** Both rules compare against the road's
  incident-free modelled cost under the clock alone, so a reported incident cannot
  explain a slow reading away — otherwise the fallback would be blind to exactly
  the change it exists to catch.
- **The baseline is a snapshot, taken once when the scenario is created.** A
  production system would refresh it on a daily batch; this one takes one reading
  per scenario, which makes two observations against one scenario comparable. It
  is taken *before* the scenario is priced, so a scenario's own rows never enter
  the history it is judged against.
- **An infinite observation is refused, and the refusal is deliverable.** `gt=0`
  alone would admit an infinity, so `travel_time` also carries
  `allow_inf_nan=False` and answers 422. Pydantic echoes the offending value back
  in that error, though, and Starlette renders every response with
  `allow_nan=False` — so the 422 raised *because* the value was infinite could not
  itself be written, and `1e400` (a valid JSON number literal that overflows to
  infinity when parsed) got a 500 from the serialiser instead. The app installs
  its own `RequestValidationError` handler that nulls the echoed input and
  otherwise returns FastAPI's body unchanged.

**The standard deviation is 0 on real logged data**, and that is the case worth
knowing about. An ordinary `travel_time` is `base_travel_time ×
congestion_multiplier`, both deterministic, so every sample for one key is the
same number. The z-score is then `0/0`, which is answered rather than guarded:
equal to the history is `z = 0` and is not flagged, differing from it is infinite
and is. An infinity cannot go on the wire — Starlette renders with
`allow_nan=False` and `JSON.parse` rejects `Infinity` — so the verdict reports
`z_score: null` with `std_dev: 0.0` saying why, and the flag is decided on the
real value regardless.

`demo_detection.py` shows all of it: the seeded baseline, a reading at the mean
and at `mean + 1.5σ` (not flagged) against one at `mean + 3σ` (flagged), a road
with four samples taking the fallback at its strict 1.2× margin, and the
zero-spread case above. It **seeds synthetic history, so it runs against a
throwaway log** — the collected `data/traffic_log.db` is never written to or read,
because inventing samples for a real road would corrupt that road's baseline
permanently. The path it used is printed.

**No prediction is built on this data.** Detection is arithmetic over rows we
already have: no model, no training pipeline, no "predicted travel time" field.
The log exists so that a later phase has a `(road, time, condition, incident) ->
travel_time` history to learn from, and the detector reads the same rows a model
eventually would.

### Simulated GPS watcher

Everything above consumes fleet telemetry and nothing produced any, which is why
every `(road, condition)` key in the log has a standard deviation of exactly
zero. `POST /scenarios/{id}/watcher` is the stand-in for the fleet that would fix
that.

**It is a simulation, not fleet integration.** There is no GPS device, no
telemetry protocol and no vehicle — the same caveat the traffic simulator
carries. What there is: a background daemon thread per watched scenario that,
every `interval_seconds`, places each of that scenario's vehicles on the road it
is currently driving, draws a plausible travel time for it, and does the two
things a real ping would do.

| Route | Does |
| --- | --- |
| `POST /scenarios/{id}/watcher` | Solves the scenario with the production default (or a named solver), puts its vehicles on the roads their routes run along, runs the first tick inline, and returns `201` with the routes and that first tick. A second fleet on one scenario is `409` |
| `GET /scenarios/{id}/watcher` | `running`, `ticks`, `interval_seconds`, and each vehicle's current road, seconds remaining and progress. **Does not tick** — reading the fleet does not move it |
| `POST /scenarios/{id}/watcher/tick` | One tick, synchronously: readings, verdicts, `changed_legs`, `rows_logged`. The same function the timer calls, which is why no test sleeps |
| `DELETE /scenarios/{id}/watcher` | Stops it and returns the final state. The readings it took stay in the scenario and in the log |

The vehicles start **spread along their own routes** — vehicle `i` of `n` at
`(i+1)/(n+1)` of its corridor — rather than all leaving the depot together,
which would have three vehicles reporting on the same first road. A vehicle
advances by *seconds of driving*, not by edges, so a road that slows down keeps
it there longer.

```json
POST /scenarios/{id}/watcher  {"seed": 7, "interval_seconds": 18}
{
  "scenario_id": "…", "solver": "qpso", "running": true,
  "interval_seconds": 18.0, "time_scale": 1.0, "ticks": 1,
  "vehicles": [{"vehicle_id": "veh-0", "edge": {"u": 0, "v": 1},
                "remaining_seconds": 12.4, "progress": 0.25, "finished": false}],
  "routes": [ … ],
  "first_tick": {
    "tick": 1, "at": "…",
    "readings": [{"vehicle_id": "veh-0", "edge": {"u": 0, "v": 1},
                  "modelled_travel_time": 41.2, "observed_travel_time": 42.0,
                  "flag": false, "rule": "insufficient_samples",
                  "condition": "normal", "sample_count": 0, …}],
    "changed_legs": 3, "rows_logged": 3, "conditions_mutated": false
  }
}
```

A tick does six things: read the scenario's effective state; advance every
vehicle and ask where it is under the *current* costs; read the road's modelled
time and draw a noisy measurement from it; judge the measurement and fold it in;
re-price the scenario and write one log row per reading; write the record back
with a compare-and-swap.

**The noise is the point, not decoration.** A rule-based row is
`base_travel_time × multiplier`, both deterministic, so without this the log
stays a table of constants and no standard deviation can be computed from it.
Readings are drawn from a lognormal whose mean multiplier is exactly 1.0 —
strictly positive, right-skewed, and unbiased about the road while every
individual reading differs from the last. Its width is **derived, not chosen**:
the detector's fallback flags anything above `1.2 ×` the modelled time, and for a
lognormal multiplier `1.2×` sits `(ln 1.2 + σ²/2)/σ` standard deviations above
the mean, so σ decides the false-positive rate on its own — `0.06` puts it at ~3σ
(about one reading in 1000) where `0.15` would flag one drive in ten. An
incident's flat `×2.9` sits at ~18σ either way, so a wider spread would not make
true positives more visible, only bury them in false ones.

**Two tiers, ranked rather than blended.** Applying a reading is *Tier 1*: a
measured time **overwrites** the rule-based estimate for that road, which is what
retires the flat `×2.9` an incident applies. An incident's number is a
placeholder for a delay nobody has measured, so it prices a road only until
something does. A closure still outranks both, because passability is not an
estimate of speed.

Each tick re-prices, so a later `POST /optimize` searches under what the fleet
actually saw. The rows are written against the **repriced** state, so
`travel_time` holds the *measured* seconds rather than the model's estimate they
replaced, and carry the incident word when one was in force — which is what keeps
an incident-time reading out of a later baseline. This is the ingestion half that
`POST /detect` deliberately does not do; `/detect` still writes nothing.

Two consequences worth knowing:

- **A watcher left running makes a scenario's costs non-repeatable**, by design.
  The store's "same id, same costs" invariant holds only while nothing is
  measuring the roads. Costs are still per scenario, though, while the log rows
  are global: the fleet's history teaches every *later* scenario's baseline even
  though a scenario's own matrix does not change from someone else's readings.
- **Nothing is written to the graph.** The Delhi extract is a cached, shared,
  read-only object, so a measurement is applied by putting it on the
  `TrafficState` and repricing — a test asserts every edge attribute is
  identical across a tick. "The edge weight was updated" means the *charge*
  moved, and only that.

Five runnable demos, all driving the real app in-process over `TestClient` on the
real Delhi graph, so there is no server to start:

```bash
uv run python benchmarks/demo_traffic.py      # same round at 09:00 vs 14:00
uv run python benchmarks/demo_incident.py     # inject an incident, read the log
uv run python benchmarks/demo_detection.py    # flag a slow road from its own history
uv run python benchmarks/demo_watcher.py      # a simulated fleet measures the roads
uv run python benchmarks/demo_reopt.py        # re-plan what is left, hold what is done
uv run python benchmarks/demo_avoid_road.py   # a driver avoids their own road
```

`demo_incident.py` is the Prompt 5 demonstration: it picks a road off the
cheapest path the scenario's own matrix was built from, reports it, prints the
leg before and after, prints the audit row `GET /traffic/log` returns, and
reverts. `--slow` reports it slow-but-passable rather than closing it.
`demo_detection.py` is the Prompt 6 one, described under **Anomaly detection**
above. `demo_watcher.py` is the Prompt 7 one: it dispatches a fleet, steps three
ticks, shows a road's price being replaced by the fleet's measurement of it,
injects an incident on a road a vehicle is driving to show the detector fire and
the tier rule at work, reads back the rows the fleet wrote, and finally restarts
the fleet at a real interval so the background thread can be watched ticking.
`--interval 1 --live-seconds 8` makes that last part quick; the scripted sections
step the fleet explicitly so their output is exact. `demo_reopt.py` is the Prompt
8 one, described under **Adaptive re-optimization** below, and
`demo_avoid_road.py` is the manual-override one, described under **The manual
override** below that.

### Adaptive re-optimization

Everything above leaves the plan alone. `POST /optimize` solves a scenario once
and hands back routes; the watcher then drives those routes while traffic moves
underneath it, and the routes never change. A vehicle stays committed to a plan
chosen under costs that may no longer exist. `POST /scenarios/{id}/reoptimize` is
where the fleet gets to react.

| Condition | Response |
| --- | --- |
| unknown scenario | `404` |
| no fleet running | `404`, naming `POST /scenarios/{id}/watcher` |
| no incident and no flagged reading | **`409`** |
| unknown solver, or an exact solver over what remains | `422` |
| a live closure severs the remaining stops from a vehicle | `409`, naming the vehicles' positions |

```json
POST /scenarios/{id}/reoptimize  {"include_geometry": false}
{
  "solver": "qpso", "solver_name": "QPSO", "seed": 7,
  "trigger": {"kinds": ["incident", "anomaly"], "primary": "incident",
              "detail": "1 live incident(s) on the network (slow); the fleet's most
                         recent tick flagged 1 reading(s) as anomalous",
              "edges": [{"u": 43, "v": 91}], "reasons": ["observed 118.4 s against a
                         model of 41.2 s — 2.87x, past the 1.20x margin"]},
  "vehicles": [{"vehicle_id": "veh-0", "node": 91, "completed": ["D0", "D4"],
                "remaining": ["D7"], "elapsed_seconds": 812.4,
                "remaining_capacity": 4.0, "stuck": false, "finished": false,
                "available": true}],
  "before": [ … ], "after": [ … ],
  "completed": ["D0", "D4"], "replanned": ["D7", "D9"],
  "moved": [{"delivery_id": "D9", "from_vehicle": "veh-1", "to_vehicle": "veh-2"}],
  "cost": 214.6, "travel_time": 611.0, "distance_m": 5402.1,
  "fuel_litres": 0.43, "feasible": true, "runtime_ms": 96.2
}
```

**A completed stop cannot be reassigned, because it is not in the problem.**
The re-optimization builds a *new* scenario whose delivery list holds only the
unserved stops, each vehicle's capacity is what it has left to give, and each
vehicle's start is where it currently is. A completed delivery therefore has no
entry in `Scenario.deliveries`, so there is no index a solver could return and no
route that could contain it. The guarantee is structural — it holds for all five
solvers at any budget — rather than a filter applied to an answer. The demo
asserts it rather than describing it.

**Both halves of the answer cover the same stops.** `before` is the rest of the
plan the fleet was already driving, in its original order and priced from each
vehicle's *real* position; `after` is the new assignment of exactly those stops.
Comparing them is therefore two ways of serving one set of work, not two
different amounts of work. Completed deliveries appear in neither — they are
reported once on `vehicles` — and there is deliberately no field in
`ReoptimizeResponse` that could carry one in an `after` route.

**The trigger is derived and enforced.** There is no `force` field and no
`trigger` field on the request, and no way to declare fleet state either:
positions, completed stops and remaining capacity are read from the running
watcher. A request arriving when nothing has happened is a `409`. Without that,
"adaptive routing" would be an endpoint indistinguishable from a re-solve button.
An **incident** is a report and needs no corroboration; an **anomaly** is a
measurement the detector flagged on the fleet's most recent tick, reported with
the detector's own reason string verbatim. Both commonly hold at once, and both
are reported rather than collapsed, because "we were told" and "we measured" are
different strengths of evidence.

**Making "start from where it is" work meant changing the depot model.** The
depot was hard-coded at matrix index 0 in `CostMatrix.DEPOT_INDEX`,
`route_metrics`' leg construction and `optimal_split`'s rank-1 decomposition, so
a per-vehicle start was unrepresentable rather than merely unimplemented.
`Scenario.starts` and `CostMatrix.vehicle_start_index` extend that, with the
depot still at index 0 and every route still returning to it — only the outbound
leg moves. `vehicle_start_index` is **empty** rather than a tuple of zeros for
the ordinary case, and two decoder branches are gated on `has_custom_starts`
precisely so that a scenario without starts decodes bit-for-bit as it did before.
`benchmarks/run_comparison.py` must still produce its old numbers; that is the
check that the extension is additive, and it is re-run rather than argued.

**Re-optimizing is a read.** It computes a plan and returns it. It does not
rewrite the stored scenario — so a second call against an unchanged fleet returns
the same plan — and it does not re-dispatch the fleet, because that means
rebuilding every track and resetting how far each vehicle has travelled. That is
a simulation decision, not an optimizer one. **Applying a plan is the obvious
next step, and it is the boundary this feature stops at.**

Full reasoning, including the time-window re-basing approximation and why the
shift is by the smallest elapsed time, is in the **Re-optimization** section of
`DESIGN_DECISIONS.md`.

`demo_reopt.py` is the Prompt 8 demonstration. It dispatches a fleet, steps it,
shows each vehicle's split between delivered and still-ahead, gets a **409** on a
clean scenario first, then reports a road a vehicle is driving as slow, ticks,
prints the detector's own reason for flagging the reading, and prints the
before/after table with the completed stops marked and held fixed and every
delivery that changed hands named. It closes by searching every new route for a
completed stop and asserting none is there.

### The manual override

`POST /scenarios/{id}/vehicles/{vehicle_id}/avoid-road` is the third way a
re-plan is earned, and the only one a person asks for by name. A driver names a
road and a treatment — `slow`, or `closure` — and gets a new route for **their own
remaining stops**, while every other vehicle keeps exactly the route it was on.

| Condition | Response |
| --- | --- |
| unknown scenario, or no fleet running | `404` |
| unknown `vehicle_id` | `404`, naming the fleet's own vehicle ids |
| the road is not in the graph | `422` — the same check and message `POST /incident` gives |
| the vehicle has no unserved stops | `409` |
| the vehicle is on no road at all | `409` |
| the closure severs the instance | `422`, from the incident machinery |
| unknown solver | `422`, and nothing is written — the solver is resolved before the report is filed |
| the new plan cannot be built under it | `409` |

```json
POST /scenarios/{id}/vehicles/veh-1/avoid-road
  {"edge": {"u": 43, "v": 91}, "treatment": "closure", "include_geometry": false}
{
  "trigger": {"kinds": ["override"], "primary": "override",
              "detail": "vehicle veh-1 reported the road 43 -> 91 as closure, and
                         asked to be routed around it",
              "edges": [{"u": 43, "v": 91}], "reasons": []},
  "vehicles": [{"vehicle_id": "veh-1", "node": 43, "edge": {"u": 43, "v": 91}, …}],
  "replanned_vehicles": ["veh-1"],
  "incident": {"incident_id": "9f2c…", "incident_type": "closure",
               "edge": {"u": 43, "v": 91}, "created_at": "…"},
  "before": [ …every vehicle… ], "after": [ …every vehicle… ],
  "completed": ["D0", "D4"], "replanned": ["D7"], "moved": [],
  "cost": 88.3, "feasible": true, "runtime_ms": 41.7
}
```

**Why it exists.** An incident's effect is a *flat placeholder* — `slow` prices a
road at `PEAK_FACTOR` (×2.9) and that is all it can say. A driver who is looking
at five times the modelled delay has no way to say so, and the statistical route
to noticing runs through a fleet that has to drive the road, be measured, and have
the detector agree the trip was unusual.

**The report is a real incident, not a way around the trigger rule.** It goes
through `POST /incident`'s own code: the road is re-priced for the whole scenario,
an audit row is written, it lands in the scenario's incident list, it is
revertable with `DELETE /scenarios/{id}/incident/{incident_id}`, and a later
fleet-wide `POST /reoptimize` sees it. That is what keeps the `409` rule intact
rather than cutting a hole in it for the most convenient caller — the driver
*changed the network* rather than asserting a justification, so the next
fleet-wide call is justified by a live incident like any other. `detect_trigger`
was not modified and knows nothing about overrides.

**What the override adds is scope.** One vehicle is in the instance that is
solved, so no solver can move another one however badly it searches. `before` and
`after` nonetheless cover the whole fleet — that is what makes "everybody else is
untouched" visible rather than asserted — `replanned_vehicles` names the one
vehicle that was solved for, and `moved` is empty by construction, because there
is nobody to hand work to.

**Where the new route begins follows reachability.** Normally the far end of the
road the vehicle is on, as ever. But a plan cannot begin on the far side of a road
the vehicle cannot drive: a `closure` on the driver's own road moves the start to
the **near** end — the driver turns around — while a `slow` report leaves the road
drivable and the ordinary rule stands. The same rule covers a vehicle already
stopped behind somebody else's closure, which is the case a fleet-wide
`/reoptimize` has nothing to offer: it has nowhere to put a stopped vehicle's
load, refuses with `409`, and leaves the load on board a vehicle going nowhere.
`VehicleProgressOut.edge` is what makes that possible — the road a stopped vehicle
is stopped behind is the only record of where it is.

**This route writes, and `/reoptimize` does not.** The write is the *report*, not
the re-plan: `qgati.reopt` is still read-only and gained no I/O dependency.

`demo_avoid_road.py` is the manual-override demonstration. It reads a driver's own
road out of the fleet, reports it `slow` and shows the re-plan, then reports it
`closed` and shows the new route starting from the *near* intersection instead.
It prints the before/after table for the whole fleet with the untouched vehicles
marked and **asserts** that no other route changed, then reverts both reports one
at a time and shows the scenario back to `mutated=False` with no live incidents —
noting as it does that the fleet's own measurements are a separate layer, and were
never the report's to undo.

## Tests

```bash
cd backend
uv run pytest                  # the fast suite: synthetic graphs, no network
```

The suite runs against a synthetic random graph, so it is fast and offline.
Three tests exercise the real Delhi graph and are skipped by default; opt in
with:

```bash
SMART_GATI_RUN_SLOW=1 uv run pytest         # bash
$env:SMART_GATI_RUN_SLOW = "1"; uv run pytest   # PowerShell
```

It needs `backend/data/cache/delhi_drive_2000m.graphml` to exist, which one call
to `load_delhi_graph()` creates.

## Layout

| Package             | Responsibility                                              |
| ------------------- | ----------------------------------------------------------- |
| `qgati.graph`       | Road graph construction and caching (OSMnx/NetworkX)         |
| `qgati.routing`     | Shortest paths — Dijkstra, A*                                |
| `qgati.optimizer`   | VRP solvers — QPSO, GA, classical PSO, Savings, brute force  |
| `qgati.traffic`     | Rule-based traffic simulator, live incidents, the log they collect, and anomaly detection over it |
| `qgati.fleet`       | The simulated GPS fleet — where each vehicle is, how long its road took, and the background watcher that ticks |
| `qgati.reopt`       | Adaptive re-optimization — where each vehicle is, what has already been delivered, re-planning only what is left, and where a re-planned route may begin. Also the **explanation** of a re-plan (`explain.py`), in sentences built from that re-plan's own figures |
| `qgati.analytics`   | The run history — a SQLite log of every solve (solver, cost, runtime, and the before/after ETA of a re-plan), and the aggregates read over it |
| `qgati.explain`     | **Empty placeholder that nothing imports.** Explainability was built in `qgati.reopt.explain` instead, beside the re-plan whose numbers it quotes |
| `qgati.api`         | FastAPI application — schemas, in-memory store, routes       |

`data/` holds cached graphs and scenario configs. `data/cache/` is gitignored: it
holds the fetched `.graphml`, plus `data/cache/osm_http/` for OSMnx's raw
Overpass responses. `data/traffic_log.db` and `data/run_history.db` are gitignored
too — both are written at runtime and grow with use. The run history is a record
with no reader but `/analytics/*`; the traffic log is not, since anomaly detection
reads each road's logged history as its baseline.
