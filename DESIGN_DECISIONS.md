# Q-Gati Design Decisions

This file records the project's architectural and algorithmic decisions.
All later work must follow what is recorded here.

## Solver and benchmark

- **Production default: QPSO** (as required by SIH PS 26137).
- Benchmarked against: Brute Force, Savings (Clarke-Wright), GA, Classical PSO.
  ACO was also benchmarked and then removed from the project — see
  [Removed: ACO](#removed-aco).
- Benchmark methodology: identical cost matrices, interleaved per-seed execution,
  iteration sweep to defend budget choice.

## Removed: ACO

**ACO (ant colony optimization) was removed from the project on request.** The
code is gone from `qgati.optimizer`, the registry, the API, the frontend's
Compare tab and the tests. `backend/src/qgati/optimizer/aco.py` is a tombstone
docstring rather than an implementation, and `SOLVERS` holds five solvers.

**This deleted a result, not a redundancy, and the finding is kept here so it is
not quietly lost.** On the Phase 4 benchmark, ACO reached a **lower mean raw cost
than QPSO at n=15 (9,906.9 against 10,365.9) and by a wide margin at n=25
(16,343.2 against 21,386.1)** — and it held the best mean at every instance size,
tied with the GA at n=8. ACO was the strongest solver in the set on raw cost. The
reason QPSO ships anyway is unchanged and was never a benchmark claim: PS 26137
names quantum-inspired search, so the default is a requirement of the problem
statement rather than an outcome of the comparison. What the removal changes is the
*field*: with ACO gone, "QPSO is the best of these" is a statement about a smaller
set of solvers, not a new result. The two figures above are from the time-only
objective and carry the same staleness banner as the tables they come from.

Two things were deliberately **not** done:

- **The benchmark result files under `backend/benchmarks/results/` are kept as
  raw measurements.** They are a record of what was run, not a claim that ACO is
  still available, and re-running them without ACO would destroy the only
  evidence for the paragraph above.
- **The two benchmark tables in `backend/README.md` that carry ACO's figures are
  kept, with their existing staleness banners.** They are measurements from a
  benchmark that ran; deleting the row would make the surviving numbers
  unreadable as a comparison.

The cost of the removal is recorded plainly: **there is no longer a solver in
this project that beats QPSO on raw cost**, and that is a fact about the solver
set rather than a result. Any "QPSO performs best" statement in this repository
should be read against this section.

## Traffic states

Traffic states are multipliers applied per edge via the road-class sensitivity
mechanism (through roads ×1.00 sensitivity, side streets ×0.30).

| State       | Multiplier | Notes                                       |
|-------------|-----------|---------------------------------------------|
| Normal      | ×1.0      | Baseline (no congestion)                     |
| Moderate    | ×1.6      | Default daytime band                         |
| Peak        | ×2.9      | Heavy congestion                             |
| Closure     | ∞         | Edge impassable                              |

**Source:** TomTom Traffic Index 2025, New Delhi — average congestion 60.2%
(corresponding to ×1.6), 6 pm peak 192% (×2.9).

No rain factor, no accident factor. The existing road-class scaling is retained.

The state is selected by the clock alone. There is no manual toggle, so a demo
left running unattended always prices under a real band:

| Band               | Window                   |
|--------------------|--------------------------|
| Peak               | 08:00-10:00, 17:00-20:00 |
| Daytime (moderate) | 06:00-22:00              |
| Everything else    | normal                   |

A peak window outranks the daytime band it sits inside. The 06:00-22:00 daytime
boundary is a modelling choice, **not** a TomTom figure — the index publishes
congestion by hour, not a definition of daytime.

An accident is x2.9 **flat**, and replaces that edge's congestion multiplier
rather than compounding with it: x2.9 is already the conservative worst case, so
multiplying peak hour on top would charge the same congestion twice.

## Incidents

Incidents are operator-reported:

| Report     | Effect                                      |
|------------|---------------------------------------------|
| "blocked"  | Edge weight set to ∞ (impassable)           |
| "slow"     | Edge base time ×2.9, mode "estimated"       |
| GPS observation | A real observed edge time from any fleet vehicle **overwrites** the estimate — see *Fleet telemetry* |
| "road clear" | Removes any operator-placed override       |

That GPS row is the only one of the four that is a measurement rather than a
reported factor, and it is ranked above the other two overrides. The tiering is
set out in full under *Fleet telemetry* below.

Both multipliers already exist as the two condition buckets in the simulator
(`closed_edges` → ∞, `accident_edges` → ×2.9). An incident **replaces** that
road's congestion multiplier rather than compounding with it, the same rule as
above — which is why a "slow" report on a through road at peak changes nothing:
the road is already at ×2.9.

### Live and mutable, per scenario

Conditions start frozen — a scenario is priced once, and the same `scenario_id`
answering with the same costs is what makes one solver comparison meaningful.
The operator override is layered on top rather than replacing that:

- `POST /scenarios/{id}/incident` reports one road as `closure` or `slow` and
  re-prices that scenario.
- `DELETE /scenarios/{id}/incident/{incident_id}` reverts it.
- The creation-time conditions stay stored underneath, so a revert is exact
  rather than subtractive: accidents and closures the scenario was born with
  survive a inject/revert cycle untouched.
- **Neither route runs a solver.** They rebuild the cost matrix and stop, so an
  operator's report cannot silently move vehicles that are already on the road.
  Re-optimizing in response to an incident is the re-optimization section below,
  and is a separate module.

### Vocabulary

The `traffic_log.incident_type` column records **what was reported**, so it
holds five values in two generations: `accident` / `road_closure` from a
scenario's creation-time `conditions` block, and `closure` / `slow` /
`road_clear` from the live-incident routes. `accident` and `slow` are the same
effect under two names, as are `road_closure` and `closure`. Renaming the older
pair to match would split an accumulating dataset across two spellings of the
same state, so the older rows were left alone and the report's own word is
written alongside them.

`road_clear` is what a revert writes — the observation that the road is back to
its normal cost, not merely the absence of a row.

### The audit row

A mutation writes exactly one row, for the incident edge. The scenario's full
road set was already logged when it was priced, so re-logging it per incident
would add hundreds of near-duplicate rows.

The row carries the **scenario's own pricing timestamp**, not the wall-clock
moment of the change, so it is a valid observation of the conditions that
scenario is priced under and pairs with the rows already collected for it. The
wall-clock moment is returned instead as `incident.created_at`, which keeps the
"when was this reported?" answer without a log row claiming a congestion band the
scenario was never priced under.

## Detection and re-optimization trigger

Per-road, per-traffic-condition statistics maintained:

- Mean and standard deviation of past travel times for each (road, condition) pair.
- Z-score = (actual − mean) / std
- Z > 2 flags the road as anomalous.
- Fewer than 10 samples → fallback: actual > 1.2 × expected flags it.

Implemented in `backend/src/qgati/traffic/detection.py` and reached over HTTP at
`POST /scenarios/{id}/detect`. Plain statistics — `statistics.fmean` and
`statistics.stdev` over rows already in `traffic_log` — with no model and no
training. The thresholds are constants in that module: `MIN_SAMPLES = 10`,
`Z_THRESHOLD = 2.0`, `FALLBACK_FACTOR = 1.2`.

### Detecting is not re-optimizing

A verdict **flags**; it does not act. The route runs no solver, and its response
model has no field a route, a cost or a solver could be returned in — the same
structural boundary the incident routes draw. Acting on a flag is the
re-optimization module's job, below.

**Nothing is written.** The observation is judged and discarded. Recording it
would feed the anomaly into the baseline that is supposed to catch it, so
ingestion is not this route's job: it is the fleet watcher's, in *Fleet
telemetry* below.

### Two rules, and what "normal" means to each

Both rules answer the same question — *is this road slower than it normally is?* —
against the same notion of normal: the road's incident-free modelled cost under
the clock alone. `expected` is computed with manual conditions **cleared**, so a
live incident cannot explain a slow reading away. Were it otherwise the fallback
would be blind to exactly the change it exists to catch, and the two rules would
disagree about what "normal" meant.

Both comparisons are strict (`>`), so an observation sitting exactly on a
threshold is not flagged.

### The case that dominates real data: std dev ≡ 0

An ordinary logged `travel_time` is `base_travel_time(attributes) ×
congestion_multiplier(road_class, condition)` — both deterministic, so every
sample for one `(road, condition)` key is the **same number**. The standard
deviation is therefore exactly zero and the z-score is `(x − μ) / 0`. This is the
ordinary state of this log, not a corner case.

It is answered rather than guarded against. Equal to the history is `z = 0.0` and
is not flagged — an observation that is exactly what the road always does is not
anomalous. Differing from it is `z = ±inf` and is flagged: a road that has only
ever taken 10 s and now reports 29 s is not a finite number of standard deviations
out, because there is no standard deviation for it to be out by.

An infinity cannot go on the wire — Starlette renders responses with
`allow_nan=False`, and `JSON.parse` rejects `Infinity` — so the verdict reports
`z_score: null` with `std_dev: 0.0` saying why and the reason sentence carrying
the reading in words. The flag itself is decided on the real value, so the verdict
is unaffected.

An infinite *observation* is refused at the boundary: `gt=0` alone would admit
one, so `ObservationRequest.travel_time` carries `allow_inf_nan=False` and answers
422. That refusal needs a handler of its own, because Pydantic echoes the offending
value back inside its error and the same `allow_nan=False` then applies to the
*error* body — so `1e400`, a valid JSON literal that overflows to infinity when
parsed, produced a 500 from the serialiser rather than the 422 the validation had
just decided on. The app installs its own `RequestValidationError` handler, which
nulls the echoed input and otherwise returns FastAPI's body unchanged.

### Baselines exclude incidents, and are snapshotted once

Two kinds of row are skipped when building a baseline, for the same reason —
neither is an observation of a road being itself:

- **Any row with an `incident_type`.** A baseline trained on anomalies cannot
  detect them: a road routinely shut at 18:00 would report a shut road at 18:00 as
  perfectly ordinary.
- **Any row with a NULL `travel_time`.** That is how a closed road is logged — it
  is impassable, so there is no travel time to record.

**Simplification versus production.** The baseline is computed **once, when the
scenario is created**, and reused for that scenario's lifetime. A production
system would refresh it on a daily batch, or over a rolling window, so that
history stays current; this one takes a single reading per scenario, which is
enough for a demonstration and makes two observations against one scenario
provably comparable. It is taken *before* the scenario is priced, so a scenario's
own rows never enter the history it is then judged against.

Keys are per `(road, condition)`, as above, so a 09:00 reading is compared with
that road's own 09:00-band history rather than against a pooled figure in which
peak *is* the anomaly. The accepted cost is a third as many samples per key, so
the fallback fires more often — which the verdict reports rather than hides, by
always returning `sample_count`.

## Fleet telemetry

There is no real fleet in this project, and a demonstration of dynamic routing
needs one: without vehicles reporting on roads, the traffic log only ever holds
what the model already knew — `base_travel_time × multiplier`, both
deterministic, and therefore a table with no spread for a detector to detect
from. `backend/src/qgati/fleet/` is the stand-in, and it is **a simulation, not
fleet integration**: no GPS device, no telemetry protocol, no vehicle. The pings
are generated from the same traffic model the rest of the app prices with, plus
noise. Same caveat as *Data sources*.

A **watcher** is one background daemon thread per scenario. Every
`interval_seconds` — 18 s by default, the middle of the 15-20 s a real fleet
would ping at — it places each of that scenario's vehicles on the road it is
currently driving, draws a plausible travel time for it, judges it against that
road's history, and applies it. Reached over HTTP at
`POST/GET/DELETE /scenarios/{id}/watcher` and
`POST /scenarios/{id}/watcher/tick`.

### Two tiers, ranked rather than blended

The incident table above lists a GPS observation beside "blocked" and "slow" as
though the three were peers. They are not — they are ranked, and only the first
is a measurement:

| Rank | Source | What it says | Wins because |
|------|--------|--------------|--------------|
| 1 | Closure | the road cannot be driven | passability is not an estimate of speed |
| 2 | **GPS observation** | this road took 34.6 s | a measurement beats a model |
| 3 | "slow" / accident | the road takes ×2.9 | a placeholder for a delay nobody has measured |
| 4 | Congestion band | the road takes ×1.6 / ×2.9 | the rule-based default |

**Tier 2 retires Tier 3, and that is the whole point of it.** ×2.9 is a stand-in
for a delay nobody has measured, so it prices a road only until somebody does:
the first fleet vehicle to drive a slow road replaces the flat placeholder with
what it actually took. Measured, that number lands near ×2.9 rather than exactly
on it — which is the difference between a rule and an observation, and the reason
the placeholder is described as lasting only until real data arrives.

**Tier 1 outranks Tier 2 because passability is not a speed.** A closure is not
an estimate that could be refined by a better one; it is a statement that the
road is shut. So `get_traffic_multiplier` tests `closed_edges` first and a closed
road stays infinite however many vehicles report on it — which in practice means
a measurement can never resurrect a road an operator has shut.

### A measured time is not a multiplier

The congestion and incident layers produce a *factor*; an observation is a
*duration in seconds*. They are not interchangeable and the code does not pretend
they are. `get_traffic_multiplier` still answers with a factor and knows nothing
about observations; the override happens one level down, in `_simulated_cost`,
which returns the measured seconds directly rather than dividing them by a base
time to invent a multiplier the road may not have.

The mechanical consequence: an observation is only meaningful for a road it was
taken on. A ×2.9 could be applied anywhere; 34.6 s could not.

### Why the noise, and how wide

An ordinary logged `travel_time` is deterministic, so every sample of one
`(road, condition)` key is the same number and its standard deviation is exactly
zero — the case *Detection* above documents at length. A fleet changes that: two
drivers on the same road at the same hour do not take the same time, and the
spread between them is the signal a detector needs. The noise is therefore not
decoration, and it is the reason this section exists at all.

Readings are drawn from a **lognormal** multiplier with mean exactly 1.0:
strictly positive (a normal multiplier would eventually produce a zero or
negative time), right-skewed (a road can be much slower than usual far more
easily than much faster), and unbiased — the fleet is right about the road on
average and only ever disagrees about the individual trip.

Its width is **derived from the detector's own locked margin, not chosen**.
`FALLBACK_FACTOR = 1.2` means a reading above 1.2× the modelled time is flagged,
and for a lognormal multiplier 1.2× sits `(ln 1.2 + σ²/2)/σ` standard deviations
above the mean:

| σ | where 1.2× sits | false-positive rate |
|---|---|---|
| 0.06 | ~3.0 σ | ~1 reading in 1000 |
| 0.15 | ~1.3 σ | ~1 reading in 10 |

`NOISE_SIGMA = 0.06`, for the first line. A detector that flags one drive in ten
is a detector nobody reads. An incident's flat ×2.9 sits at ~18 σ either way, so
a wider spread would not make true positives any more visible — it would only
bury them.

### What a tick measures against

The measurement and the expectation must both come from **observation-free**
states, or the fleet would score every reading against its own last one and drift
upward by a random step each tick. Within that there are two model numbers, and
they answer different questions:

- **`live`** — the scenario's own conditions, incidents *included* and
  measurements excluded. This is what the vehicle is actually driving through, so
  it is what the measurement is drawn around. Measured from the incident-free
  model instead, a road reported slow would never read as slow.
- **`clean`** — incidents cleared as well. This is what the detector compares
  against, the same choice `POST /scenarios/{id}/detect` makes, for the reason
  recorded there.

Collapsing the two into one would break one end or the other: with only `clean`
the fleet could never report the incident it is driving through, and with only
`live` an incident would explain itself away.

### Per-scenario costs, global history

Observations live on the scenario, beside its incidents, so a scenario's cost
matrix is still a function of that scenario alone. The store's invariant — the
same `scenario_id` answers with the same costs, which is what makes a solver
comparison meaningful — is preserved.

**But the rows a watcher writes are global.** They go to the same
`traffic_log` every scenario's baseline is built from, so the fleet teaches every
*later* scenario even though one scenario's readings never move another's
matrix. Costs are per scenario; history is not.

One consequence to state plainly: **a watcher left running makes a scenario's
costs non-repeatable, by design.** The invariant above holds only while nothing
is measuring the roads.

### Nothing is written to the graph

A tick changes what the app *charges* for a road, never the road. The Delhi
extract is a cached, shared, read-only object, so a price written into it would
leak into every later request; every price in this project is a function of a
`TrafficState`, so a measurement is applied by putting it on the state and
re-pricing. A test asserts the graph's edge attributes are identical across a
tick. "The edge weight was updated" means the charge moved, and only that.

### It is the ingestion phase, and it is the first concurrent code here

Recording an observation is what `POST /scenarios/{id}/detect` deliberately does
not do. A watcher's tick does it: one `traffic_log` row per measured road, stamped
with the *measured* seconds — not the estimate they replaced — and with the
incident word when one was in force, so an incident-time reading stays out of a
later baseline.

That makes the watcher the first thing in the project that mutates a scenario
without a request asking it to. The store's compare-and-swap was built for a
mutation that spends time re-pricing before it writes, and this is what uses it
unprompted: losing the race is not an error, because the readings were still
taken and their rows still written, and the next tick re-reads and tries again.
One lock per watcher is held across a whole tick, so `stop()` cannot return while
a tick is still writing.

## Re-optimization

- **Scope:** partial, per vehicle, remaining stops only. Completed stops and
  current vehicle position are fixed.
- **Solver:** QPSO (the production default).
- **Cost matrix** rebuilt after any edge-weight change.

### A completed stop is not filtered out — it is not in the problem

The invariant the whole feature rests on is that **a completed stop can never be
reassigned**. It would be easy to implement that as a check: solve everything,
then reject any answer that moves a delivery already made. That implementation is
wrong in a way that would not show up until it mattered, because it makes the
guarantee a property of the checking code — and the guarantee then holds only for
the solvers and code paths somebody remembered to check.

Instead, re-optimization builds a *new scenario* whose delivery list holds only
the unserved stops. A completed delivery has no entry in
`Scenario.deliveries`, so there is no index a solver could return, no route that
could contain it, and no assignment to reject. The guarantee is structural: it
holds for all five solvers, at any search budget, because the object they are
searching over does not contain the answer they would have to give to break it.

The same instinct as `Scenario.__post_init__` refusing an infeasible instance
rather than letting a solver return a bad one. Making a wrong answer
unrepresentable beats detecting it.

Two substitutions go with it. Each vehicle's **capacity** is what it has left to
give — dispatched capacity less the demand it has already dropped, which is what
actually frees up as a vehicle unloads — and each vehicle's **start** is where it
currently is. That is the whole of "modified starting conditions".

### The depot was hard-coded, and that was the real work

"Each vehicle starts somewhere different" turned out not to be unimplemented but
**unrepresentable**. Three places assumed a route both begins and ends at matrix
index 0:

- `CostMatrix.DEPOT_INDEX = 0`, and `route_metrics` building its legs as
  `[(depot, first), ..., (last, depot)]`;
- `optimal_split`'s rank-1 decomposition, which splits a route's cost into
  `matrix[depot, perm[i]] + ... + matrix[perm[j], depot]`;
- `_scenario_nodes`, which never emitted any node but the depot and the
  deliveries.

So `Scenario` gained `starts` — one node per vehicle, empty meaning "every
vehicle leaves the depot" — and `CostMatrix` gained `vehicle_start_index` with a
`start_index(vehicle)` accessor. The **depot stays at index 0** and a route still
*returns* to it; only the outbound leg moves. That asymmetry is forced: one index
cannot describe a different start per vehicle, so the starts are a separate
mapping rather than a relocated depot. `vehicle_start_index` is **empty** rather
than a tuple of zeros for the ordinary case, so a matrix built the usual way is
identical to one built before the field existed.

The decoder changed least of all, because `optimal_split`'s outer loop already
counted routes: route `r` is vehicle `r-1`, so per-vehicle starts and per-vehicle
capacities slot into the existing loop with no change to the DP's `O(k·n²)` shape.

**Two branches exist purely to preserve today's numbers, and both are gated on
`has_custom_starts`.** Without starts, every route is still bounded by the fleet's
largest capacity rather than its own vehicle's, and `assemble_routes` still sorts
routes by load onto vehicles. A per-vehicle capacity is arguably the more correct
rule in general — but it is not the rule the Phase 4 benchmark was run under, and
a benchmark that moves because a constraint got tighter is comparing two
different problems. `benchmarks/run_comparison.py` must reproduce its old numbers
exactly; that is the check that the extension is genuinely additive, and it is
re-run rather than argued.

### Re-optimizing cannot be forced

`POST /scenarios/{id}/reoptimize` refuses with **409** unless something has
actually happened: a live incident, or a reading the detector flagged on the
fleet's most recent tick. Neither → no re-plan.

This is not politeness about wasted CPU. Without it, "adaptive routing" is an
endpoint indistinguishable from a re-solve button, and the demo proving it reacts
to incidents would be proving only that the script called it after injecting one.
So the rule is derived and enforced rather than described: there is no `force`
field, no `trigger` field, and no way to declare a fleet state in the body either
— positions, completed stops and remaining capacity are read from the running
watcher, which 404s if no fleet is running.

The two kinds are reported separately because they are different strengths of
evidence. An **incident** is a report — somebody said a road is slow, and it needs
no corroboration. An **anomaly** is a measurement — the fleet drove a road and the
detector found the trip unlike that road's history. They commonly both hold: an
incident is injected, a vehicle drives it, and the detector flags the trip
independently. Reporting only the incident would throw away the measurement that
confirms it.

An anomaly is read from the **most recent tick only**. A flag from twenty ticks
ago describes a moment nobody is re-planning for, and a trigger that never expired
would make every later request look justified.

There is a third kind — `override`, a driver reporting their own road — and it is
not a relaxation of this rule. It is filed as a real incident *before* anything is
re-planned, so what justifies the re-plan is still a live incident on the network;
see *The manual override* below for why that shape was chosen over a `force`
field.

### The one approximation: time windows are re-based

Windows are quoted in seconds **from depot departure**, and a vehicle that has
been out for 900 seconds needs them shifted. The shift cannot be exact: a window
belongs to the delivery, but elapsed time belongs to whichever vehicle serves it —
and after re-optimization that may be a different vehicle than the one that was
900 seconds in.

The shift used is the **smallest elapsed time among participating vehicles**. That
is the optimistic choice, and it is chosen deliberately: shifting by the smallest
elapsed can only ever make a window *earlier*, so the shift alone never turns a
reachable window into an unreachable one. Lateness the other vehicles incur is
found by the optimizer and priced by the objective at `late_per_second`, which is
where it belongs — a real cost, reported, rather than a modelling artefact.
Windows are clamped at zero rather than allowed to go negative: a window that
closed while the vehicle was driving is one the re-plan is already late for, and
the objective should price that rather than the model refusing to represent it.

Worth knowing when reading this: the API cannot currently create a scenario with
windows at all (`DeliveryOut` carries only id, node and demand), so this is core
correctness with unit tests rather than something the demo exercises.

### What re-optimizing does not do

It is a **read**. It computes a plan and returns it; it does not rewrite the
stored scenario and it does not re-dispatch the fleet.

Not rewriting is what makes a second call against an unchanged fleet return the
same plan rather than a different one — which it could not if the first had
written anything. Not re-dispatching is because applying a plan means rebuilding
every vehicle's track and resetting how far it has travelled, which is a
simulation decision rather than an optimizer one. **Applying a plan is the
obvious next step and is named as the boundary rather than left ambiguous.**

`POST /scenarios/{id}/vehicles/{vehicle_id}/avoid-road` is the one exception, and
it is worth being exact about where it lives. That route *does* write — but what
it writes is the **driver's report**, filed through the incident routes' own code,
and not the re-plan. `qgati.reopt` is still read-only and gained no I/O dependency
to make it work; the write belongs to the route, next to the incident routes that
own the same machinery.

### The manual override: a driver says so

`POST /scenarios/{id}/vehicles/{vehicle_id}/avoid-road` is the third way a
re-plan is earned, and the only one a person asks for by name.

It exists because of a specific placeholder. An incident's effect on a road is a
**flat multiplier** — `slow` prices the road at `PEAK_FACTOR` (×2.9) and that is
it. If the real delay is five times the modelled time, the statistical route to
noticing runs through a fleet that has to drive the road, be measured, and have
the detector agree the trip was unusual — and at `NOISE_SIGMA = 0.06` a genuinely
much-worse road does trip it, but only after somebody has driven it. The party
who already knows is sitting in the vehicle, and this is the place they can say
so.

**The report is a real incident, not a bypass.** It goes through `POST
/incident`'s machinery unchanged: the road is re-priced for the whole scenario,
an audit row is written, it lands in the scenario's own incident list, it is
revertable with `DELETE /scenarios/{id}/incident/{incident_id}`, and a later
fleet-wide `POST /reoptimize` sees it.

That is the whole reason it is shaped this way. The alternative — a `force` flag,
or a trigger the re-plan request declares about itself — is exactly the hole the
rule above exists to prevent, and it would be a hole cut for the most convenient
caller. Filing the report instead means the driver has *changed the network*
rather than *asserted a justification*, and `detect_trigger`, which knows nothing
about overrides and was not modified for them, reaches the same verdict on the
next call through the incident branch it already had.

What the override adds on top of all that is **scope**: it re-plans one vehicle
rather than the fleet, because one vehicle is what the driver asked about. The
scoping is not a filter over the answer — the other vehicles are *absent from the
instance that was solved*, so no solver can move one however badly it searches.
The response's `before` and `after` nonetheless cover the whole fleet, so
"everybody else is untouched" is visible in the diff rather than asserted in
prose; `replanned_vehicles` names the one vehicle that was actually solved for,
and `moved` is empty by construction because there is nobody to hand work to.

**Where the new route begins is the one part that is not about cost.** A
re-planned vehicle normally starts from the far end of the road it is on — the
next intersection it reaches — which is what keeps a new route from opening with
a leg already driven. But a route is a sequence of roads, and a plan cannot begin
on the far side of one the vehicle cannot drive. So:

- A **`slow`** report leaves the road drivable, the vehicle still reaches the far
  end, and the ordinary rule stands.
- A **`closure`** on the driver's own road makes the far end unreachable, so the
  start moves to the **near** end: the driver turns around.
- The same exception covers a vehicle **already stopped behind somebody else's
  closure**, which is a state the fleet reports as `stuck` with no position at
  all — and which is why the fleet's progress carries the *road* a stopped
  vehicle is stopped behind, not just the fact that it is stopped.

That last case is the one `POST /reoptimize` cannot answer and this can. A
fleet-wide re-plan has nowhere to put a stopped vehicle's load, so it refuses
with 409 and leaves the load on board a vehicle that is going nowhere. Here the
stopped vehicle is the one that reported, and it is the only one that matters.

## Objective function

One rupee-equivalent cost composed of:

- Time cost
- Distance cost
- Fuel cost
- Penalties (capacity, coverage, time windows)

The first three are **priced in rupees** and summed; penalties are added on top:

    cost = w1·time + w2·distance + w3·fuel + penalties

Implemented in `backend/src/qgati/optimizer/objective.py`. The weighted sum is
formed once per scenario, when the cost matrix is built, so all five solvers
minimise the identical array and a benchmark between them stays apples-to-apples.

Penalties are capacity, coverage and fleet shape — flat amounts, scaled off the
objective's own magnitude — plus **lateness**, which is priced per second rather
than flat. Time windows are implemented; see *Time windows* below.

### Time windows

A delivery may carry an optional **service window**: `earliest_arrival` and
`latest_arrival`, both nullable. A scenario where neither is set anywhere behaves
exactly as it did before windows existed, and takes the older, cheaper code paths.

**Measured in seconds from depot departure, not as clock times.** The optimizer
layer is deliberately free of wall-clock time, exactly as the cost matrix is.
Absolute time belongs to the traffic layer, which prices the road network under a
timestamp and then hands the optimizer an array of seconds; putting a clock in
the optimizer would mean the router and the optimizer disagreed about what
"09:00" means the moment congestion was involved. An API accepting ISO times is a
conversion at the boundary, not a change of unit in the core.

**Arriving early means waiting, and waiting is charged.** Waiting is what a
driver actually does — the goods cannot be handed over before the window opens —
so it is not a violation, but the vehicle and its driver are committed for that
hour exactly as much as for an hour of driving, so it is priced at w1. The clock
moves forward to the window's opening, which is what lets a window bite with
nothing in the route being late: a stop reached too early delays every later
arrival, and that can be what makes the next one miss.

**Arriving late is priced, not forbidden.** Lateness is charged at
**₹600/hour** — four times the vehicle-time price — applied to every second a
stop is served after its latest arrival. Two rates equal would make a solver
indifferent between an hour driving and an hour late, and it would then break a
window whenever the detour cost more than the delay it avoided, which is the
opposite of what a window is for.

The multiple is a judgement, not a measurement: ₹600/hour stands in for a
re-delivery, a waiting customer, or a contractual penalty. Replace it with the
operator's own service-level figure via `CostWeights(late_per_hour=...)`.

Worth knowing when reading the numbers: lateness is priced four times the *time*
price, but time is only part of the travel price. At 30 km/h the full objective
works out near ₹0.155/s against ₹0.167/s for lateness, so a second spent late
costs roughly 1.08 seconds of driving — the window has to be tight enough that
avoiding the lateness is worth the detour. Measured, not assumed: it is why the
time-window tests use a window that admits exactly one change of plan.

**Why windows cannot be folded into the cost matrix, when time, distance and
fuel were.** The other three are sums over a route's legs, so they collapse into
an `(n, n)` array once and every solver reads them. Arrival time is not a
property of a leg: it depends on the entire prefix that preceded it, and the wait
at an early stop is a non-linear function of when the vehicle got there. So the
matrix stays window-free and the clock is walked per route instead — which is why
`route_metrics` is the only place in the optimizer that knows what time it is.

That difference propagates into the exact solver. Held-Karp's `(subset, last)`
state holds one number because with an additive cost, the only thing that matters
about a partial tour is what it has cost. With windows it must instead carry a
Pareto frontier of `(cost, ready time)` labels, since two ways of visiting the
same subset can differ in both and the cheaper one may be the one that misses the
next window. Frontiers are pruned on both axes, which is safe because time is
priced — a label's cost already contains the price of every second it has spent,
waiting included — so a label that is both cheaper and earlier can never lose.
Brute force remains exact; the instance-size limit is unchanged.

### Why prices, not weights

The three goals are measured in seconds, metres and litres. A weighted sum of
unnormalised quantities is decided by whichever unit happens to be largest —
metres here, by three orders of magnitude over litres — rather than by intent.
Normalising each term to `[0, 1]` avoids that but hides the trade-off instead of
stating it: there is no fact of the matter about whether distance deserves 0.3,
and the right answer would move with the instance's scale.

Pricing each term in rupees fixes the comparison once, at the point where the
judgement belongs, and makes every weight falsifiable — a reviewer can disagree
with ₹150/hour by stating their own figure and why.

### Defaults

| Term | Default | Basis |
|------|---------|-------|
| Time (w1) | ₹150 / hour | Driver ≈ ₹94/h (₹22,000 a month ÷ 26 days ÷ 9 h) plus ≈ ₹55/h of vehicle time cost — insurance, permits, road tax, calendar depreciation |
| Distance (w2) | ₹5 / km | Tyres, servicing, oil and the distance-based share of depreciation, for a light commercial vehicle |
| Fuel (w3) | ₹90 / litre | Delhi diesel retail |
| Lateness | ₹600 / hour | 4× the time price. A stand-in for a re-delivery, a waiting customer or a contractual penalty — a service-level figure, not a cost

**w2 excludes fuel.** Fuel is priced separately as w3; including it in both would
charge every kilometre twice.

**These are assumptions, not measurements.** They are stated so they can be
argued with and replaced; `CostWeights(...)` overrides any of them, and
`build_cost_matrix(..., weights=...)` applies the result to a scenario.

### The fuel model

Fuel is estimated as a function of distance and average speed, which is the
fallback the brief sanctions when no better model is available. A real model
needs per-vehicle engine data, and this project consumes only the operator's own
fleet observations (see *Data sources*), so none is available offline.

The curve is the conventional three-term average-speed form:

    litres per 100 km  =  idle/v  +  rolling  +  drag·v        (v in km/h)

| Term | Reads as |
|------|----------|
| `idle/v` | Per-unit-*time* overhead — idling, friction, low gears — expressed per unit distance. Dividing by speed is what makes crawling expensive. |
| `rolling` | Rolling resistance: the distance-only part. |
| `drag·v` | Aerodynamic drag, which is why the curve turns back up at speed. |

It has one minimum, which is the physically correct shape. The coefficients are
**pinned to two stated points rather than fitted**, there being no dataset to fit
to: the minimum sits at a 65 km/h cruise and returns 7.5 L/100 km there, giving
`idle = 211.25`, `rolling = 1.0`, `drag = 0.05`. The curve reads 12.6 L/100 km at
20 km/h and 7.9 at 90.

**Assumption:** fuel depends only on distance and average speed. Gradient,
payload, time spent stationary at stops, and vehicle age are not modelled.

**Why fuel is its own term rather than folded into distance.** Fuel is distance
multiplied by a factor moving between 0.075 and 0.126 L/km across the usable
speed range, so the two terms largely agree about which route is short. What fuel
adds is *speed sensitivity*: traffic scales edge times, which lowers each leg's
implied average speed, which raises fuel per kilometre. Distance alone cannot see
that, and it is the only part of the objective that prices congestion in diesel
rather than only in driver-hours.

**Honest caveat:** distance and fuel are strongly correlated and are not two
independent signals. A cost breakdown should be read with that in mind.

### Distance, and which route it is measured along

A leg's distance is the length of roads its **fastest path** runs along — not the
shortest path by distance, and not the straight line. Routing minimises time, so
the fastest path is the route the vehicle actually drives; pricing a different
one would report metres and fuel for a journey nobody makes. A consequence worth
knowing: on a network with a slow direct road and a fast detour, the chosen route
can be both quicker and longer, and the distance term will show the trade rather
than hide it.

## Run history

Every solve is recorded to SQLite in `data/run_history.db`, read back through
`GET /analytics/runs` and `GET /analytics/summary`.

### It is a record of runs, not of requests

Four solve paths write a row — `optimize`, `dispatch` (starting a watcher),
`reoptimize`, and `avoid_road` — and each writes exactly one. A request that is
**refused writes nothing**: a re-plan that returns `409` because no incident
justifies it, or a scenario that does not exist, returns before the write. So the
table answers "what has this system solved", which is the question an operator
actually has, rather than "what has been asked of it".

Logging is best-effort: the write is wrapped so that a failure to record can never
fail a solve that succeeded. A solve that happened and was not logged is a lost
row; a solve that worked and returned a 500 because the log was locked is a lost
demo.

### `new_eta_seconds` is derived, not independently computed

On a re-plan the row carries the remaining travel time before and after. The
"after" figure is the same `travel_time` the row already stores for that re-plan,
written to both fields from one evaluation — so the two cannot drift apart, and
`eta_saved_seconds` (`old − new`) is a difference between two numbers that were
produced together.

It is **not clamped at zero**: a re-solve on a small instance can land on a worse
arrangement than the one in hand, and reporting that as a zero saving would hide
the case most worth seeing.

### `incident_triggered_runs` counts `incident` and `anomaly`, not `override`

Three things can justify a re-plan (`qgati.reopt.triggers`): a filed road incident,
a flagged reading from anomaly detection, and a driver reporting their own road.
Only the first two count as "the system reacted". A driver's override is an
operator saying so, and folding it into a statistic about automatic response would
overstate how much the detection layer is doing.

### Known inconsistency: `avoid-road`'s row is fleet-scoped, its response is not

`POST /scenarios/{id}/vehicles/{v}/avoid-road` returns a plan whose top-level
totals come from re-solving **that one vehicle**, while its `after` array covers
the **whole fleet's** remaining work. The run-history row is written from the
fleet-sized evaluation, deliberately, so that a row always describes the whole
fleet's outstanding work and is comparable with the other three kinds.

The consequence, recorded rather than fixed: for that one endpoint
`sum(response["after"]) != response["travel_time"]`. It is a pre-existing
discrepancy in the response, not something the history introduced, and every
consumer of that response — the dashboard's panels and the explanation — reads the
arrays rather than the top-level totals.

## The comparison surface

`GET /optimize/{scenario_id}/compare` runs every registered solver over one
scenario's cost matrix and returns each one's cost, gaps, travel time, distance,
fuel and runtime. The frontend's **Compare** tab is the only UI over it.

### It writes no run-history rows

A five-solver sweep is a benchmark, not a solve somebody asked for. Recording it
would put rows into the history that no operator request corresponds to, and
would swamp a table whose value is that every row is a real decision. The offline
`sweep` runner (`benchmarks/run_comparison.py`) is excluded for the same reason.

### The row order is the solver registry's, and a skipped solver is absent

Rows come back in `SOLVERS` order (`qgati.optimizer.registry`), so the table is not
a ranking — and the frontend renders the response's order rather than sorting, so
a sixth solver appears without any frontend change.

Brute force cannot run past `MAX_EXACT_DELIVERIES`. When it cannot, **every one of
its figures is `null`** and `skipped` says why, so a solver that did not run is
unambiguously absent rather than a zero-cost answer. The UI renders that sentence
and no figures, and draws no bar for it — a zero-height bar is the same
misreading in a different medium.

### The optimum is usually unknown, and the UI says so

`optimal` is populated only from brute force's own result, so on any instance
larger than its exact limit it is `null`. The dashboard's sample scenario (12
deliveries) is such an instance, which makes "no proven optimum" the **ordinary**
case on the demo path rather than an edge case.

So the comparison leads with gap-against-best-found, which is always available,
and reports the gap against the optimum only when it exists — labelled "not known"
rather than left blank, because the two are different claims: "0.4% above the best
anyone found here" is not "0.4% above proven optimal".

## Data sources

- **No ML models.** No external traffic API.
- Real-time signals come only from the operator's own fleet GPS or driver reports.
- Road network from OpenStreetMap (via OSMnx), cached locally.
- Traffic statistics derived from observed fleet data only.