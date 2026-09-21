#!/usr/bin/env python
"""Phase 4 benchmark: every VRP solver, identical instances, one comparison table.

The question this answers is "why QPSO?" — which is only answerable by running
the alternatives on the *same* cost matrices and reporting the honest numbers.
So every solver here is handed the identical ``CostMatrix`` for an instance, is
scored through the same :func:`~qgati.optimizer.fitness.evaluate`, and appears in
one table with its best cost, its runtime, and its optimality gap.

What is compared
----------------
=====================  ==========  ==================================================
solver                 kind        role
=====================  ==========  ==================================================
``brute force``        exact       ground truth — only for n <= 10
``savings``            heuristic   constructive baseline (Clarke-Wright)
``genetic algorithm``  metaheuristic
``classical PSO``      metaheuristic  the control: isolates QPSO's quantum update
``ACO``                metaheuristic
``QPSO``               metaheuristic  the project's headline solver
=====================  ==========  ==================================================

Two tables come out, and the second matters as much as the first:

1. **The comparison** — per instance, per algorithm: best / mean / worst cost,
   mean runtime, and gap against the exact optimum.
2. **The iteration sweep** — QPSO's (and the other metaheuristics') quality as a
   function of generation budget at 100 / 200 / 300 / 1000 iterations, with the
   rate at which each exact optimum is hit. The default budget of 100 is a
   choice, and this is the evidence that defends it.

Reading the numbers
-------------------
Runtimes are wall-clock on one machine and are indicative, not a formal
complexity result. Mean cost is only meaningful *within* an instance — the three
instances are different problems at different scales, so the aggregate table
normalises to a percentage gap. Brute force and Savings are deterministic, so
they are run once per instance; the others are run over ``--seeds`` seeds and
summarised.

Usage
-----
    uv run python benchmarks/run_comparison.py                       # full run
    uv run python benchmarks/run_comparison.py --quick                # smoke test
    uv run python benchmarks/run_comparison.py --real                 # Delhi graph
    uv run python benchmarks/run_comparison.py --instances 8:3@303 25:5

Results are written to ``benchmarks/results/`` as a timestamped set: ``.json``
(the complete record) plus ``_comparison.csv`` and ``_sweep.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from qgati.graph import build_cost_matrix, build_synthetic_graph, load_delhi_graph
from qgati.optimizer import (
    DEFAULT_NUM_ITERATIONS as DEFAULT_BUDGET,
)
from qgati.optimizer import (
    MAX_EXACT_DELIVERIES,
    Scenario,
    Solution,
    build_random_scenario,
    clarke_wright_savings,
    evaluate,
    run_aco,
    run_classical_pso,
    run_genetic_algorithm,
    run_qpso,
    solve_brute_force,
)

#: A stochastic run this close to the optimum counts as having found it.
EXACT_TOLERANCE = 1e-6

#: Costs and gaps are quoted to this many decimals in the tables.
COST_DECIMALS = 1
GAP_DECIMALS = 2

#: An algorithm counts as having "won" an instance within this percentage.
GAP_TOLERANCE_PCT = 0.01

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: The default instances: small enough for an exact answer, then two beyond
#: exact reach. k follows the project's existing fixtures (~n/5, but k=3 on n=8
#: so the instance is genuinely constrained rather than trivially one-route).
DEFAULT_INSTANCES = ("8:3", "15:3", "25:5")


# --------------------------------------------------------------------------- #
# Solver registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Metaheuristic:
    """A stochastic solver, plus how to feed it a budget.

    The four solvers name their budget and population arguments differently on
    purpose — ``num_generations`` reads better in a GA than ``num_iterations`` —
    so the adapter carries the names rather than distorting the solvers' APIs.
    """

    name: str
    run: Callable[..., tuple[Solution, float, list[float]]]
    budget_kwarg: str
    population_kwarg: str

    def __call__(
        self, cost_matrix, scenario: Scenario, *, seed: int, population: int, iterations: int
    ) -> tuple[Solution, float, list[float]]:
        return self.run(
            cost_matrix,
            scenario,
            seed=seed,
            **{self.budget_kwarg: iterations, self.population_kwarg: population},
        )


METAHEURISTICS: tuple[Metaheuristic, ...] = (
    Metaheuristic("QPSO", run_qpso, "num_iterations", "num_particles"),
    Metaheuristic("Classical PSO", run_classical_pso, "num_iterations", "num_particles"),
    Metaheuristic(
        "Genetic Algorithm", run_genetic_algorithm, "num_generations", "population_size"
    ),
    Metaheuristic("ACO", run_aco, "num_iterations", "num_ants"),
)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class AlgorithmRun:
    """One algorithm's results on one instance, across every seed."""

    algorithm: str
    costs: list[float]
    runtimes: list[float]
    feasible: int
    optimal: float | None
    best_known: float

    @property
    def runs(self) -> int:
        return len(self.costs)

    @property
    def best(self) -> float:
        return min(self.costs)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.costs)

    @property
    def worst(self) -> float:
        return max(self.costs)

    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.costs) if self.runs > 1 else 0.0

    @property
    def mean_ms(self) -> float:
        return 1000.0 * statistics.fmean(self.runtimes)

    @property
    def min_ms(self) -> float:
        """Fastest observed run.

        The throttle-robust estimate: a laptop under sustained load slows down
        partway through a long benchmark, which inflates a mean but cannot make a
        run faster than the machine can actually go. Comparing solvers on their
        minima is therefore more trustworthy here than comparing means, and both
        are reported so the gap between them is visible rather than hidden.
        """
        return 1000.0 * min(self.runtimes)

    @property
    def exact_hits(self) -> int | None:
        """Runs that landed on the optimum, or ``None`` when it is unknown."""
        if self.optimal is None:
            return None
        return sum(1 for cost in self.costs if cost <= self.optimal + EXACT_TOLERANCE)

    @property
    def gap_vs_optimal(self) -> float | None:
        if self.optimal is None or self.optimal <= 0.0:
            return None
        return 100.0 * (self.best - self.optimal) / self.optimal

    @property
    def gap_vs_best_known(self) -> float | None:
        if self.best_known <= 0.0:
            return None
        return 100.0 * (self.best - self.best_known) / self.best_known

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "runs": self.runs,
            "best": self.best,
            "mean": self.mean,
            "worst": self.worst,
            "stdev": self.stdev,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "feasible_runs": self.feasible,
            "gap_vs_optimal_pct": self.gap_vs_optimal,
            "gap_vs_best_known_pct": self.gap_vs_best_known,
            "exact_hits": self.exact_hits,
            "costs": self.costs,
            "runtimes_ms": [1000.0 * r for r in self.runtimes],
        }


@dataclass
class InstanceResult:
    label: str
    n_deliveries: int
    n_vehicles: int
    scenario_seed: int
    optimal: float | None
    algorithms: list[AlgorithmRun] = field(default_factory=list)

    @property
    def best_known(self) -> float:
        return min(run.best for run in self.algorithms)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_deliveries": self.n_deliveries,
            "n_vehicles": self.n_vehicles,
            "scenario_seed": self.scenario_seed,
            "optimal": self.optimal,
            "best_known": self.best_known,
            "algorithms": [run.to_dict() for run in self.algorithms],
        }


@dataclass
class SweepPoint:
    """One algorithm at one iteration budget, averaged over the sweep seeds."""

    algorithm: str
    iterations: int
    mean_cost: float
    best_cost: float
    mean_ms: float
    min_ms: float
    exact_hits: int | None
    seeds: int

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "iterations": self.iterations,
            "mean_cost": self.mean_cost,
            "best_cost": self.best_cost,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "exact_hits": self.exact_hits,
            "seeds": self.seeds,
        }


# --------------------------------------------------------------------------- #
# Table rendering
# --------------------------------------------------------------------------- #
def format_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], right_from: int = 1
) -> str:
    """Render an aligned plain-text table. Columns before ``right_from`` are left-aligned."""
    cells = [[str(head) for head in headers]]
    cells += [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[column]) for row in cells) for column in range(len(headers))]

    lines: list[str] = []
    for index, row in enumerate(cells):
        rendered = [
            cell.ljust(widths[column]) if column < right_from else cell.rjust(widths[column])
            for column, cell in enumerate(row)
        ]
        lines.append("  ".join(rendered).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def _cost(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.{COST_DECIMALS}f}"


def _gap(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.{GAP_DECIMALS}f}"


def _hits(run: AlgorithmRun) -> str:
    hits = run.exact_hits
    return "n/a" if hits is None else f"{hits}/{run.runs}"


# --------------------------------------------------------------------------- #
# Progress logging
# --------------------------------------------------------------------------- #
class ProgressLog:
    """Append-only JSONL record of every unit of work, flushed as it completes.

    Two problems this exists to solve, both of which bit this benchmark once:

    * **A killed run must not take its results with it.** A 6-minute sweep that
      dies at minute five previously left nothing behind, because results were
      only written at the very end.
    * **A redirected run is otherwise invisible.** Python block-buffers stdout
      when it is not a terminal, so a healthy run in the background produces no
      output at all until it exits — which is indistinguishable from a hang.

    One line per event, ``fsync``-ed, so what is on disk is always the truth
    about how far the run got.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    def record(self, event: str, **fields: object) -> None:
        if self._handle is None:
            return
        payload = {
            "event": event,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields,
        }
        self._handle.write(json.dumps(payload, default=str) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _timed(call: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def run_instance(
    scenario: Scenario,
    cost_matrix,
    *,
    seeds: Sequence[int],
    population: int,
    iterations: int,
    solvers: Sequence[Metaheuristic] = METAHEURISTICS,
    label: str,
    scenario_seed: int,
    progress: ProgressLog | None = None,
) -> InstanceResult:
    """Run every solver on one instance's cost matrix."""
    log = progress or ProgressLog(None)
    # (algorithm, per-seed costs, per-seed runtimes, feasible count)
    entries: list[tuple[str, list[float], list[float], int]] = []

    # The exact optimum, where it is reachable. Everything else is measured
    # against it — and it is what makes "exact hit rate" meaningful in the sweep.
    optimal: float | None = None
    if scenario.n_deliveries <= MAX_EXACT_DELIVERIES:
        brute_solution, brute_seconds = _timed(
            lambda: solve_brute_force(scenario, cost_matrix)
        )
        brute_evaluation = evaluate(brute_solution, scenario, cost_matrix)
        optimal = float(brute_evaluation.travel_cost)
        entries.append(
            (
                "brute force",
                [optimal],
                [brute_seconds],
                1 if brute_evaluation.feasible else 0,
            )
        )
    log.record(
        "instance_start",
        instance=label,
        n_deliveries=scenario.n_deliveries,
        n_vehicles=scenario.n_vehicles,
        scenario_seed=scenario_seed,
        optimal=optimal,
    )

    # Deterministic baselines: one run each. Repeating them per seed would add
    # identical rows, so they carry a single runtime and a single cost.
    savings_solution, savings_seconds = _timed(
        lambda: clarke_wright_savings(scenario, cost_matrix)
    )
    savings_evaluation = evaluate(savings_solution, scenario, cost_matrix)
    entries.append(
        (
            "savings",
            [float(savings_evaluation.travel_cost)],
            [savings_seconds],
            1 if savings_evaluation.feasible else 0,
        )
    )

    # Interleaved by seed rather than run solver-by-solver. A laptop throttles
    # under sustained load, so a solver measured in one contiguous block gets
    # whatever thermal state that block landed in; spreading every solver across
    # the whole run samples them all under the same conditions.
    costs: dict[str, list[float]] = {solver.name: [] for solver in solvers}
    runtimes: dict[str, list[float]] = {solver.name: [] for solver in solvers}
    feasible: dict[str, int] = {solver.name: 0 for solver in solvers}

    for seed in seeds:
        for solver in solvers:
            (solution, _cost_value, _history), seconds = _timed(
                lambda solver=solver, seed=seed: solver(
                    cost_matrix,
                    scenario,
                    seed=seed,
                    population=population,
                    iterations=iterations,
                )
            )
            evaluation = evaluate(solution, scenario, cost_matrix)
            costs[solver.name].append(float(evaluation.travel_cost))
            runtimes[solver.name].append(seconds)
            feasible[solver.name] += 1 if evaluation.feasible else 0
            # Per seed, so a run killed mid-instance still yields the quality
            # data for every solver-seed pair that did finish.
            log.record(
                "solver_seed",
                instance=label,
                algorithm=solver.name,
                seed=seed,
                iterations=iterations,
                population=population,
                cost=float(evaluation.travel_cost),
                feasible=evaluation.feasible,
                seconds=seconds,
            )

    for solver in solvers:
        entries.append(
            (solver.name, costs[solver.name], runtimes[solver.name], feasible[solver.name])
        )

    # best_known needs every solver's result, so it is stamped on afterwards.
    best_known = min(min(costs) for _, costs, _, _ in entries)
    result = InstanceResult(
        label=label,
        n_deliveries=scenario.n_deliveries,
        n_vehicles=scenario.n_vehicles,
        scenario_seed=scenario_seed,
        optimal=optimal,
        algorithms=[
            AlgorithmRun(
                algorithm=name,
                costs=costs,
                runtimes=runtimes,
                feasible=feasible,
                optimal=optimal,
                best_known=best_known,
            )
            for name, costs, runtimes, feasible in entries
        ],
    )
    log.record(
        "instance_done",
        instance=label,
        best_known=best_known,
        results=[run.to_dict() for run in result.algorithms],
    )
    return result


# --------------------------------------------------------------------------- #
# The iteration sweep
# --------------------------------------------------------------------------- #
def run_sweep(
    scenario: Scenario,
    cost_matrix,
    *,
    optimal: float | None,
    budgets: Sequence[int],
    seeds: Sequence[int],
    population: int,
    solvers: Sequence[Metaheuristic] = METAHEURISTICS,
    label: str = "",
    progress: ProgressLog | None = None,
) -> list[SweepPoint]:
    """Quality against generation budget, at a fixed population size.

    Holding the population constant is the point: it separates "does this solver
    need more generations?" from "does it need more particles?", which are
    different diagnoses with different fixes.
    """
    log = progress or ProgressLog(None)
    points: list[SweepPoint] = []
    # Interleaved by seed, for the same thermal reason as run_instance.
    for budget in budgets:
        costs: dict[str, list[float]] = {solver.name: [] for solver in solvers}
        runtimes: dict[str, list[float]] = {solver.name: [] for solver in solvers}
        hits: dict[str, int] = {solver.name: 0 for solver in solvers}

        for seed in seeds:
            for solver in solvers:
                (solution, _cost_value, _history), seconds = _timed(
                    lambda solver=solver, seed=seed, budget=budget: solver(
                        cost_matrix,
                        scenario,
                        seed=seed,
                        population=population,
                        iterations=budget,
                    )
                )
                evaluation = evaluate(solution, scenario, cost_matrix)
                cost = float(evaluation.travel_cost)
                costs[solver.name].append(cost)
                runtimes[solver.name].append(seconds)
                if optimal is not None and cost <= optimal + EXACT_TOLERANCE:
                    hits[solver.name] += 1
                # Per seed as well as per point: a single corrupted timing cell
                # is otherwise invisible, because the aggregate hides which seed
                # produced it. (A suspended laptop inflates one seed's wall time
                # by hours while leaving every other cell valid.)
                log.record(
                    "sweep_seed",
                    instance=label,
                    algorithm=solver.name,
                    iterations=budget,
                    seed=seed,
                    cost=cost,
                    feasible=evaluation.feasible,
                    seconds=seconds,
                )

        for solver in solvers:
            point = SweepPoint(
                algorithm=solver.name,
                iterations=budget,
                mean_cost=statistics.fmean(costs[solver.name]),
                best_cost=min(costs[solver.name]),
                mean_ms=1000.0 * statistics.fmean(runtimes[solver.name]),
                min_ms=1000.0 * min(runtimes[solver.name]),
                exact_hits=hits[solver.name] if optimal is not None else None,
                seeds=len(seeds),
            )
            points.append(point)
            # One line per completed (algorithm, budget) cell, so an interrupted
            # sweep keeps every cell that finished.
            log.record("sweep_point", instance=label, **point.to_dict())
    return points


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def print_instance_table(result: InstanceResult, *, population: int, iterations: int) -> str:
    lines = [
        f"Instance {result.label}   "
        f"(scenario seed {result.scenario_seed}, "
        f"budget {iterations} iterations x {population} population)",
    ]
    if result.optimal is not None:
        lines.append(f"  exact optimum: {_cost(result.optimal)}  (gap measured against this)")
    else:
        lines.append(
            "  no exact optimum: n > "
            f"{MAX_EXACT_DELIVERIES} deliveries; gap measured against the best any solver found"
        )

    rows = [
        [
            run.algorithm,
            _cost(run.best),
            _cost(run.mean),
            _cost(run.worst),
            _cost(run.stdev),
            _hits(run),
            f"{run.mean_ms:.0f}",
            f"{run.min_ms:.0f}",
            _gap(run.gap_vs_optimal),
            _gap(run.gap_vs_best_known),
            f"{run.feasible}/{run.runs}",
        ]
        for run in result.algorithms
    ]
    lines.append(
        format_table(
            [
                "algorithm",
                "best",
                "mean",
                "worst",
                "sd",
                "exact",
                "ms mean",
                "ms min",
                "gap opt%",
                "gap best%",
                "feasible",
            ],
            rows,
        )
    )
    lines.append(
        "  'ms min' is the fastest observed run and is the more trustworthy timing on a"
    )
    lines.append(
        "  laptop that throttles; 'ms mean' is the average. Compare solvers on 'ms min'."
    )
    return "\n".join(lines)


def print_aggregate_table(results: Sequence[InstanceResult], seeds: Sequence[int]) -> str:
    """Mean runtime and mean normalised gap per algorithm, across instances.

    Absolute mean cost is deliberately not aggregated: the instances are
    different problems at different scales, so averaging 5,000-second and
    20,000-second instances would produce a number that means nothing. The gap
    columns are scale-free and therefore comparable.
    """
    names: list[str] = []
    for result in results:
        for run in result.algorithms:
            if run.algorithm not in names:
                names.append(run.algorithm)

    rows = []
    for name in names:
        runs = [
            run for result in results for run in result.algorithms if run.algorithm == name
        ]
        gaps_opt = [run.gap_vs_optimal for run in runs if run.gap_vs_optimal is not None]
        gaps_best = [run.gap_vs_best_known for run in runs if run.gap_vs_best_known is not None]
        wins = sum(
            1
            for run in runs
            if run.gap_vs_best_known is not None
            and run.gap_vs_best_known <= GAP_TOLERANCE_PCT
        )
        feasible = sum(run.feasible for run in runs)
        total_runs = sum(run.runs for run in runs)
        mean_gap_opt = statistics.fmean(gaps_opt) if gaps_opt else None
        mean_gap_best = statistics.fmean(gaps_best) if gaps_best else None
        rows.append(
            [
                name,
                f"{statistics.fmean(run.mean for run in runs):.{COST_DECIMALS}f}",
                _gap(mean_gap_opt),
                _gap(mean_gap_best),
                f"{statistics.fmean(run.mean_ms for run in runs):.0f}",
                f"{wins}/{len(runs)}",
                f"{feasible}/{total_runs}",
            ]
        )

    lines = [
        f"Aggregate over {len(results)} instance(s); stochastic solvers averaged over "
        f"{len(seeds)} seed(s) each",
        format_table(
            [
                "algorithm",
                "mean cost",
                "mean gap opt%",
                "mean gap best%",
                "mean ms",
                "instances won",
                "feasible",
            ],
            rows,
        ),
        "  'mean cost' averages each algorithm's per-instance mean; compare it within a row,",
        "  not across rows of differently-sized instances. 'instances won' counts instances",
        "  where the algorithm's best cost matched the best any solver found.",
    ]
    return "\n".join(lines)


def print_sweep_table(
    sweep: Sequence[tuple[str, list[SweepPoint]]], seeds: Sequence[int]
) -> str:
    lines = [
        f"Iteration sweep   (population held fixed; {len(seeds)} seed(s) per point)",
    ]
    for label, points in sweep:
        rows = [
            [
                point.algorithm,
                str(point.iterations),
                f"{point.mean_cost:.{COST_DECIMALS}f}",
                f"{point.best_cost:.{COST_DECIMALS}f}",
                "n/a" if point.exact_hits is None else f"{point.exact_hits}/{point.seeds}",
                f"{point.mean_ms:.0f}",
                f"{point.min_ms:.0f}",
            ]
            for point in points
        ]
        lines.append("")
        lines.append(f"  {label}")
        lines.append(
            format_table(
                [
                    "algorithm",
                    "iters",
                    "mean cost",
                    "best cost",
                    "exact hits",
                    "ms mean",
                    "ms min",
                ],
                rows,
            )
        )
    lines.append("")
    lines.append(
        "  'exact hits' is the number of seeds whose run matched the brute-force optimum;"
    )
    lines.append(
        "  it is n/a where the instance is too large to solve exactly."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_results(
    payload: dict,
    comparison_rows: Sequence[Sequence[object]],
    sweep_rows: Sequence[Sequence[object]],
    out_dir: Path,
    timestamp: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"comparison_{timestamp}"
    written: list[Path] = []

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    written.append(json_path)

    for suffix, header, rows in (
        (
            "_comparison",
            [
                "instance",
                "n_deliveries",
                "n_vehicles",
                "scenario_seed",
                "algorithm",
                "runs",
                "best",
                "mean",
                "worst",
                "stdev",
                "mean_ms",
                "min_ms",
                "exact_hits",
                "gap_vs_optimal_pct",
                "gap_vs_best_known_pct",
                "feasible_runs",
            ],
            comparison_rows,
        ),
        (
            "_sweep",
            ["instance", "algorithm", "iterations", "mean_cost", "best_cost", "mean_ms",
             "min_ms", "exact_hits", "seeds"],
            sweep_rows,
        ),
    ):
        path = out_dir / f"{stem}{suffix}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        written.append(path)

    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_instance(text: str, default_seed: int) -> tuple[int, int, int]:
    """Parse ``n:k`` or ``n:k@scenario_seed``."""
    spec, _, seed_text = text.partition("@")
    try:
        n_text, k_text = spec.split(":")
        n, k = int(n_text), int(k_text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"instance must look like n:k or n:k@seed, got {text!r}"
        ) from None
    if n < 1 or k < 1:
        raise argparse.ArgumentTypeError(f"instance {text!r} needs positive n and k")
    seed = int(seed_text) if seed_text else default_seed
    return n, k, seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare every VRP solver on identical cost matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=list(DEFAULT_INSTANCES),
        metavar="N:K[@SEED]",
        help="instances to compare, as n:k or n:k@scenario_seed",
    )
    parser.add_argument("--seeds", type=int, default=5, help="seeds per stochastic solver")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed")
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_BUDGET, help="generation budget for the comparison"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=30,
        help="particles / population / ants, held equal across metaheuristics",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=7,
        help="scenario seed for instances given without an explicit @seed",
    )
    parser.add_argument(
        "--sweep-iterations",
        nargs="+",
        type=int,
        default=[100, 200, 300, 1000],
        help="generation budgets for the iteration sweep",
    )
    parser.add_argument("--sweep-seeds", type=int, default=10, help="seeds per sweep point")
    parser.add_argument(
        "--sweep-instances",
        nargs="+",
        default=None,
        metavar="N:K[@SEED]",
        help="instances to sweep; default is those with an exact optimum",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=[solver.name for solver in METAHEURISTICS],
        choices=[solver.name for solver in METAHEURISTICS],
        help="which metaheuristics to run",
    )
    parser.add_argument(
        "--real", action="store_true", help="use the real Delhi graph instead of a synthetic one"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-save", action="store_true", help="print tables only")
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="JSONL progress file; defaults to <out-dir>/<timestamp>_progress.jsonl",
    )
    parser.add_argument(
        "--no-progress-log", action="store_true", help="disable incremental progress logging"
    )
    parser.add_argument(
        "--quick", action="store_true", help="tiny budgets and few seeds, for a smoke test"
    )
    return parser


def resolve_args(args: argparse.Namespace) -> None:
    """Apply --quick, which is a shortcut for a set of smaller numbers."""
    if args.quick:
        args.instances = ["8:3", "12:3"]
        args.seeds = 2
        args.iterations = 40
        args.sweep_iterations = [40, 80]
        args.sweep_seeds = 3
        args.sweep_instances = ["8:3"]
    if args.sweep_instances is None:
        exact = [
            text
            for text in args.instances
            if parse_instance(text, args.scenario_seed)[0] <= MAX_EXACT_DELIVERIES
        ]
        args.sweep_instances = exact or [args.instances[0]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_args(args)

    # Line-buffered, so a redirected run reports progress as it goes rather than
    # appearing to hang: Python block-buffers stdout when it is not a terminal.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - exotic stdout object
        pass

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.no_progress_log or args.no_save:
        progress_path = args.progress_log
    else:
        progress_path = args.progress_log or (
            args.out_dir / f"comparison_{timestamp}_progress.jsonl"
        )

    progress = ProgressLog(progress_path)
    if progress_path is not None:
        print(f"progress log : {progress_path}")

    try:
        return _run(args, timestamp, progress)
    except BaseException as exc:
        # Leave the failure on disk next to the results, so a run that dies
        # unattended says why instead of just stopping.
        progress.record(
            "error",
            type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        progress.close()


def _run(args: argparse.Namespace, timestamp: str, progress: ProgressLog) -> int:
    selected = [solver for solver in METAHEURISTICS if solver.name in args.solvers]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    sweep_seeds = list(range(args.seed_start, args.seed_start + args.sweep_seeds))

    print("=" * 100)
    print("Q-Gati Phase 4 — VRP solver comparison")
    print("=" * 100)

    if args.real:
        graph = load_delhi_graph()
        graph_label = "real Delhi graph (OSMnx, cached)"
    else:
        graph = build_synthetic_graph(n_nodes=45, edge_prob=0.22, seed=11)
        graph_label = "synthetic random graph (45 nodes, edge_prob 0.22, seed 11)"

    print(f"graph        : {graph_label}")
    print(f"instances    : {', '.join(args.instances)}")
    print(f"solvers      : {', '.join(solver.name for solver in selected)}")
    print(f"baselines    : brute force (n<={MAX_EXACT_DELIVERIES}), savings")
    print(f"budget       : {args.iterations} iterations x {args.population} population")
    print(f"seeds        : comparison {seeds}, sweep {sweep_seeds}")
    print(f"sweep budgets: {args.sweep_iterations}")
    print()

    progress.record(
        "run_start",
        timestamp=timestamp,
        graph=graph_label,
        instances=list(args.instances),
        solvers=[solver.name for solver in selected],
        seeds=seeds,
        iterations=args.iterations,
        population=args.population,
        scenario_seed_default=args.scenario_seed,
        sweep_instances=list(args.sweep_instances),
        sweep_iterations=list(args.sweep_iterations),
        sweep_seeds=sweep_seeds,
    )

    # --- the comparison ---------------------------------------------------- #
    instance_results: list[InstanceResult] = []
    comparison_rows: list[list[object]] = []

    print("-" * 100)
    print("COMPARISON")
    print("-" * 100)
    for text in args.instances:
        n_deliveries, n_vehicles, scenario_seed = parse_instance(text, args.scenario_seed)
        scenario = build_random_scenario(
            graph, n_deliveries, n_vehicles, seed=scenario_seed
        )
        cost_matrix = build_cost_matrix(graph, scenario)
        label = f"n={n_deliveries} k={n_vehicles}"

        result = run_instance(
            scenario,
            cost_matrix,
            seeds=seeds,
            population=args.population,
            iterations=args.iterations,
            solvers=selected,
            label=label,
            scenario_seed=scenario_seed,
            progress=progress,
        )
        instance_results.append(result)

        print()
        print(print_instance_table(
            result, population=args.population, iterations=args.iterations
        ))

        for run in result.algorithms:
            comparison_rows.append(
                [
                    label,
                    n_deliveries,
                    n_vehicles,
                    scenario_seed,
                    run.algorithm,
                    run.runs,
                    f"{run.best:.4f}",
                    f"{run.mean:.4f}",
                    f"{run.worst:.4f}",
                    f"{run.stdev:.4f}",
                    f"{run.mean_ms:.2f}",
                    f"{run.min_ms:.2f}",
                    "" if run.exact_hits is None else run.exact_hits,
                    "" if run.gap_vs_optimal is None else f"{run.gap_vs_optimal:.4f}",
                    "" if run.gap_vs_best_known is None else f"{run.gap_vs_best_known:.4f}",
                    run.feasible,
                ]
            )

    print()
    print("-" * 100)
    print("AGGREGATE")
    print("-" * 100)
    aggregate_text = print_aggregate_table(instance_results, seeds)
    print(aggregate_text)

    infeasible = [
        (result.label, run.algorithm, run.runs - run.feasible)
        for result in instance_results
        for run in result.algorithms
        if run.feasible < run.runs
    ]
    if infeasible:
        print()
        print("  WARNING — infeasible solutions returned:")
        for label, algorithm, count in infeasible:
            print(f"    {label}: {algorithm} returned {count} infeasible run(s)")

    # --- the iteration sweep ----------------------------------------------- #
    print()
    print("-" * 100)
    print("ITERATION SWEEP")
    print("-" * 100)

    sweep_results: list[tuple[str, list[SweepPoint]]] = []
    sweep_rows: list[list[object]] = []
    for text in args.sweep_instances:
        n_deliveries, n_vehicles, scenario_seed = parse_instance(text, args.scenario_seed)
        scenario = build_random_scenario(
            graph, n_deliveries, n_vehicles, seed=scenario_seed
        )
        cost_matrix = build_cost_matrix(graph, scenario)
        label = f"n={n_deliveries} k={n_vehicles}"

        optimal: float | None = None
        if n_deliveries <= MAX_EXACT_DELIVERIES:
            optimal = float(
                evaluate(solve_brute_force(scenario, cost_matrix), scenario, cost_matrix).travel_cost
            )

        points = run_sweep(
            scenario,
            cost_matrix,
            optimal=optimal,
            budgets=args.sweep_iterations,
            seeds=sweep_seeds,
            population=args.population,
            solvers=selected,
            label=label,
            progress=progress,
        )
        sweep_results.append((label, points))
        sweep_rows.extend(
            [label, point.algorithm, point.iterations, f"{point.mean_cost:.4f}",
             f"{point.best_cost:.4f}", f"{point.mean_ms:.2f}", f"{point.min_ms:.2f}",
             "" if point.exact_hits is None else point.exact_hits, point.seeds]
            for point in points
        )
        print()
        print(f"  {label}: sweep complete")

    print()
    print(print_sweep_table(sweep_results, sweep_seeds))

    # --- save -------------------------------------------------------------- #
    print()
    if args.no_save:
        print("--no-save given: nothing written.")
        return 0

    payload = {
        "run": {
            "timestamp": timestamp,
            "graph": graph_label,
            "graph_kind": "real_delhi" if args.real else "synthetic",
            "instances": args.instances,
            "solvers": [solver.name for solver in selected],
            "seeds": seeds,
            "iterations": args.iterations,
            "population": args.population,
            "scenario_seed_default": args.scenario_seed,
            "sweep_iterations": args.sweep_iterations,
            "sweep_seeds": sweep_seeds,
            "sweep_instances": args.sweep_instances,
            "python": sys.version.split()[0],
        },
        "instances": [result.to_dict() for result in instance_results],
        "sweep": [
            {"instance": label, "points": [point.to_dict() for point in points]}
            for label, points in sweep_results
        ],
    }
    written = save_results(
        payload, comparison_rows, sweep_rows, args.out_dir, timestamp
    )
    progress.record("saved", files=[str(path) for path in written])
    print("Saved:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
