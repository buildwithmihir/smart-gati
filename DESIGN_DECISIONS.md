# Q-Gati Design Decisions

This file records the project's architectural and algorithmic decisions.
All later work must follow what is recorded here.

## Solver and benchmark

- **Production default: QPSO** (as required by SIH PS 26137).
- Benchmarked against: Brute Force, Savings (Clarke-Wright), GA, Classical PSO, ACO.
- Benchmark methodology: identical cost matrices, interleaved per-seed execution,
  iteration sweep to defend budget choice.

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

## Incidents

Incidents are operator-reported:

| Report     | Effect                                      |
|------------|---------------------------------------------|
| "blocked"  | Edge weight set to ∞ (impassable)           |
| "slow"     | Edge base time ×2.9, mode "estimated"       |
| GPS observation | A real observed edge time from any fleet vehicle overwrites the estimate |
| "road clear" | Removes any operator-placed override       |

## Detection and re-optimization trigger

Per-road, per-traffic-condition statistics maintained:

- Mean and standard deviation of past travel times for each (road, condition) pair.
- Z-score = (actual − mean) / std
- Z > 2 triggers re-optimization.
- Fewer than 10 samples → fallback: actual > 1.2 × expected triggers re-optimization.

## Re-optimization

- **Scope:** partial, per vehicle, remaining stops only. Completed stops and
  current vehicle position are fixed.
- **Solver:** QPSO (the production default).
- **Cost matrix** rebuilt after any edge-weight change.

## Objective function

One rupee-equivalent cost composed of:

- Time cost
- Distance cost
- Fuel cost
- Penalties (capacity, coverage, time windows)

## Data sources

- **No ML models.** No external traffic API.
- Real-time signals come only from the operator's own fleet GPS or driver reports.
- Road network from OpenStreetMap (via OSMnx), cached locally.
- Traffic statistics derived from observed fleet data only.