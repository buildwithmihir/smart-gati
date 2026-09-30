"use client";

/**
 * The five solvers side by side, on one scenario.
 *
 * A semantic `<table>`, following `runs-table.tsx`'s idioms: a `COLUMNS` const so
 * the header is stated once, a shared `CELL` class, an `sr-only` caption, a left
 * rule on the first cell, and `lib/format` for every figure rather than a local
 * formatter. The row order is the **response's**, which is the solver registry's
 * order — nothing here sorts or enumerates the solvers, so a sixth solver
 * appears in this table without this file changing.
 *
 * ## Two ways this table could lie, and what it does instead
 *
 * **A skipped solver.** Brute force cannot run past its exact limit, and past
 * that limit the backend sets every one of its figures to `null` — deliberately,
 * so that a solver that did not run cannot be read as a zero-cost answer. So a
 * skipped row renders its own `skipped` sentence across the figure columns, and
 * no figures at all. Never `₹0.00`, and never a zero-height bar in the chart
 * beside it: "did not run" and "ran and scored nothing" are different facts.
 *
 * **The optimum is usually not known.** `optimal` comes only from brute force, so
 * on any instance larger than its limit — including the sample scenario — it is
 * `null`. The gap against it is therefore a column that is *empty by
 * construction* on the ordinary path, and it is placed second-to-last and
 * labelled rather than dropped, because on a small instance it is the single most
 * informative figure here and hiding it would make the table's shape depend on
 * the instance size. `gap_vs_best_pct` is the always-available comparison and
 * leads.
 */

import { Badge } from "@/components/ui/badge";
import {
  formatCount,
  formatDistance,
  formatFigure,
  formatLitres,
  formatRupees,
  formatRuntime,
  formatSeconds,
} from "@/lib/format";
import type { CompareResponse, SolverCatalog, SolverResult } from "@/lib/api";

const CELL = "px-3 py-2.5 align-top";

/** Every column, so the header is stated once rather than twice. */
const COLUMNS = [
  "Solver",
  "Cost",
  "Gap vs best",
  "Gap vs optimal",
  "Travel time",
  "Distance",
  "Fuel",
  "Runtime",
] as const;

/** An em dash, for a figure that does not exist rather than one that is zero. */
function Absent({ reason }: { reason?: string }) {
  return (
    <span className="text-muted-foreground" title={reason}>
      —
    </span>
  );
}

/**
 * One gap, as a signed percentage.
 *
 * `formatFigure(…, "percent")` does the rendering, so a gap here reads exactly
 * like a percentage anywhere else on the dashboard. The sign is added because
 * these are deviations from a reference: `0.0%` on its own leaves the reader to
 * work out which direction it went, and the whole column is about direction.
 *
 * A zero gap is named rather than numbered — "best" and "optimal" are stronger,
 * shorter statements than `+0.0%`, and the difference between the two words is
 * exactly the difference between the two columns.
 */
function Gap({
  value,
  zeroLabel,
  zeroTitle,
  absentReason,
}: {
  value: number | null;
  zeroLabel: string;
  zeroTitle: string;
  absentReason: string;
}) {
  if (value === null) return <Absent reason={absentReason} />;
  // Tolerant of float noise: two solvers that found the same cost can differ in
  // the last bits, and printing "+0.0%" for a tie would invent a gap.
  if (Math.abs(value) < 0.05) {
    return (
      <span className="text-xs font-medium text-ok" title={zeroTitle}>
        {zeroLabel}
      </span>
    );
  }
  return (
    <span className="text-xs tabular-nums text-warn" title="Above the reference cost">
      +{formatFigure(value, "percent")}
    </span>
  );
}

/** The solver's name, its registry key, and what the catalog says about it. */
function SolverCell({
  result,
  isDefault,
  isExact,
}: {
  result: SolverResult;
  isDefault: boolean;
  isExact: boolean;
}) {
  return (
    <div className="flex flex-col items-start gap-1">
      <span className="text-xs font-medium">{result.solver_name}</span>
      <div className="flex flex-wrap items-center gap-1">
        <span className="font-mono text-2xs text-muted-foreground">{result.solver}</span>
        {isDefault ? (
          // The production default, which is QPSO — and which is *not* the same
          // claim as "the best of these". QPSO is the default because the problem
          // statement names it, not because it won a benchmark. This badge says
          // which one ships, not which one won.
          <Badge variant="secondary" title="The solver this app runs by default">
            default
          </Badge>
        ) : null}
        {isExact ? (
          <Badge variant="outline" title="Exhaustive — its answer is the optimum, where it can run">
            exact
          </Badge>
        ) : null}
      </div>
    </div>
  );
}

export default function CompareTable({
  response,
  catalog,
}: {
  response: CompareResponse;
  /** Null when the catalog could not be read — the badges are the only loss. */
  catalog: SolverCatalog | null;
}) {
  const optimalUnknown =
    response.optimal === null
      ? "No exact solver ran on this instance, so the optimum is not known"
      : undefined;

  return (
    <div className="overflow-hidden rounded-xl bg-card ring-1 ring-foreground/10">
      {/* Scrolls sideways on a narrow window rather than wrapping the figures,
          because a wrapped cost column stops being a column. */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">
            Every solver&apos;s result on this scenario, in the solver registry&apos;s order. Cost is
            the weighted objective in rupees. A solver that did not run shows why instead of
            figures.
          </caption>
          <thead>
            <tr className="border-b border-border">
              {COLUMNS.map((column, index) => (
                <th
                  key={column}
                  scope="col"
                  // The runtime column's honest label: this is one timed run at one
                  // seed, not the benchmark's min-of-seeds figure. Saying so in the
                  // header is cheaper than a reader assuming the stronger claim.
                  title={
                    column === "Runtime"
                      ? "One timed run at the seed above — not a minimum over seeds"
                      : undefined
                  }
                  className={`text-2xs font-medium tracking-wide text-muted-foreground uppercase ${
                    index === 0
                      ? "border-l-2 border-l-border py-2 pr-3 pl-3 text-left"
                      : `${CELL} text-left`
                  }`}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {response.results.map((result) => {
              const spec = catalog?.solvers.find((solver) => solver.key === result.solver);
              const solverCell = (
                <td className="border-l-2 border-l-border px-3 py-2.5 align-top">
                  <SolverCell
                    result={result}
                    isDefault={catalog?.default === result.solver}
                    isExact={spec?.is_exact ?? false}
                  />
                </td>
              );

              // Absent, not zero: no figures, one sentence, and the reason comes
              // from the backend so this file never restates the limit.
              if (result.skipped !== null) {
                return (
                  <tr key={result.solver} className="border-b border-border/60 last:border-0">
                    {solverCell}
                    <td
                      colSpan={COLUMNS.length - 1}
                      className={`${CELL} text-xs text-muted-foreground`}
                    >
                      <span className="mr-1.5" aria-hidden>
                        —
                      </span>
                      Did not run: {result.skipped}.
                    </td>
                  </tr>
                );
              }

              return (
                <tr
                  key={result.solver}
                  className="border-b border-border/60 last:border-0 hover:bg-muted/40"
                >
                  {solverCell}

                  <td className={CELL}>
                    <div className="flex flex-col gap-0.5">
                      <span className="text-xs tabular-nums">
                        {result.travel_cost === null ? <Absent /> : formatRupees(result.travel_cost)}
                      </span>
                      {/* The fitness actually minimised — the objective *plus*
                          constraint penalties. Only shown when it differs from
                          the objective, since on a feasible solve the two are the
                          same number and a second identical figure is noise. */}
                      {result.cost !== null && result.cost !== result.travel_cost ? (
                        <span
                          className="text-2xs whitespace-nowrap text-muted-foreground tabular-nums"
                          title="Fitness: the objective plus constraint penalties"
                        >
                          fitness {formatRupees(result.cost)}
                        </span>
                      ) : result.feasible === false ? (
                        <span className="text-2xs font-medium text-warn">infeasible</span>
                      ) : null}
                    </div>
                  </td>

                  <td className={CELL}>
                    <Gap
                      value={result.gap_vs_best_pct}
                      zeroLabel="best"
                      zeroTitle="The lowest cost any solver in this comparison reached"
                      absentReason="No solver reported a cost to compare against"
                    />
                  </td>

                  <td className={CELL}>
                    <Gap
                      value={result.gap_vs_optimal_pct}
                      zeroLabel="optimal"
                      zeroTitle="Matched the exhaustive solver's cost"
                      absentReason={optimalUnknown ?? "Not comparable"}
                    />
                  </td>

                  <td className={`${CELL} text-xs tabular-nums`}>
                    {result.travel_time === null ? <Absent /> : formatSeconds(result.travel_time)}
                  </td>

                  <td className={`${CELL} text-xs tabular-nums`}>
                    {result.distance_m === null ? (
                      <Absent />
                    ) : (
                      formatDistance(result.distance_m)
                    )}
                  </td>

                  <td className={`${CELL} text-xs tabular-nums`}>
                    {result.fuel_litres === null ? <Absent /> : formatLitres(result.fuel_litres)}
                  </td>

                  <td className={`${CELL} text-xs tabular-nums`}>
                    {result.runtime_ms === null ? <Absent /> : formatRuntime(result.runtime_ms)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="border-t border-border px-3 py-2 text-2xs text-muted-foreground">
        {formatCount(response.results.length)} solvers registered, listed in the registry&apos;s order
        rather than by rank. <span className="font-medium">Gap vs best</span> is measured against the
        lowest cost any solver here reached.{" "}
        <span className="font-medium">Gap vs optimal</span> is measured against the exhaustive
        solver&apos;s answer, so it is empty whenever that solver could not run — which is every
        instance larger than its exact limit.
      </p>
    </div>
  );
}
