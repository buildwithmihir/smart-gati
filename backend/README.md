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
| `qpso`                 | Quantum-behaved PSO — the headline solver                   |
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

`qgati.optimizer.qpso` is the headline solver: the genuine quantum-behaved
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
| `run_qpso` | quantum well sampling, `beta` 1.0 → 0.5 | the headline solver |
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

**QPSO is not the strongest solver here.** ACO wins at every size, GA is second
at n=8 and n=25, and QPSO is third — last but one — despite ACO costing about
twice the wall time per iteration. This is worth stating plainly rather than
burying: the project's headline solver does not currently justify its place on
quality alone, and the equal-budget n=25 sweep below shows that giving QPSO a 30×
larger budget does not change the ordering.

**What *is* supported is the narrower claim.** At equal budget QPSO beats
classical PSO — by 5.9% at n=25 with 3,000 iterations, and on n=8 it hits the
exact optimum in 3/5 runs against classical PSO's 1/5. Since the two differ
*only* in the update rule, that difference is attributable to the quantum
sampling. The pattern behind it is consistent: quantum sampling explores more
broadly, so it converges slowly early and better asymptotically. At n=25 with
only 100 iterations there is not enough budget to exploit that, and QPSO is
*behind* classical PSO (21,386 against 19,162) — the quantum update is a
long-budget technique.

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
