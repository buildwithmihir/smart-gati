"use client";

/**
 * How the solver got to its answer: best-so-far cost against iteration.
 *
 * This plots the **best-so-far** series, which is what the solvers return and is
 * monotone non-increasing by construction — iteration *i* holds the best cost
 * seen in the first *i* iterations, so it can only fall or stay flat. That
 * matters for what the chart is allowed to say: a monotone staircase is a
 * truthful picture of a converging search, whereas the same numbers drawn as a
 * per-iteration sample would read as a noisy series that wanders upward, and no
 * QPSO run ever reports a best-so-far that gets worse.
 *
 * It is also why the line is drawn with `type="stepAfter"` rather than smoothed.
 * The value is constant between improvements and drops at an iteration, and a
 * curve through those points would draw progress that happened at no particular
 * iteration — an interpolation that is prettier and false.
 *
 * The last value and the iteration count are printed as text beside the chart.
 * A chart alone makes the reader hover to learn the number, and the number is
 * the point; the picture is what shows the *shape* of getting there.
 */

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatRupees } from "@/lib/format";

/** Height of the plot area, in pixels. Short — this sits inside a panel. */
const CHART_HEIGHT = 148;

type ConvergenceChartProps = {
  /** Best-so-far cost per iteration. Empty when the solver reported none. */
  history: number[];
  /** Named in the caption, so the chart says which solver converged. */
  solverName: string;
};

export default function ConvergenceChart({ history, solverName }: ConvergenceChartProps) {
  const data = useMemo(
    () => history.map((cost, index) => ({ iteration: index + 1, cost })),
    [history],
  );

  // Guarded rather than rendered empty: an axis with no line on it looks like a
  // chart that failed to load, when the truth is that this solver reported no
  // history at all.
  if (data.length === 0) {
    return (
      <p className="text-2xs text-muted-foreground">
        {solverName} reported no convergence history for this solve — the plan above is
        its answer, but the path it took to get there was not recorded.
      </p>
    );
  }

  const best = data[data.length - 1].cost;
  const first = data[0].cost;
  const improvement = first - best;

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="text-2xs text-muted-foreground">
          Best cost per iteration
        </span>
        <span className="text-xs font-medium tabular-nums">
          {formatRupees(best)}{" "}
          <span className="font-normal text-muted-foreground">
            · {data.length} {data.length === 1 ? "iteration" : "iterations"}
          </span>
        </span>
      </div>

      <div style={{ height: CHART_HEIGHT }} data-convergence-chart={data.length}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
            <CartesianGrid
              stroke="var(--color-border)"
              strokeDasharray="2 4"
              vertical={false}
            />
            <XAxis
              dataKey="iteration"
              tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
              tickLine={false}
              axisLine={false}
              minTickGap={24}
            />
            <YAxis
              // Left off deliberately. The y-range of a converging run is often a
              // few rupees on a base of thousands, so an axis from zero would
              // flatten the line into the baseline and hide the thing being
              // shown; an axis that does not start at zero invites misreading the
              // slope. The exact figures are in the caption and the tooltip, and
              // this is a shape rather than a measurement.
              hide
              domain={["dataMin", "dataMax"]}
            />
            <Tooltip
              cursor={{ stroke: "var(--color-border)" }}
              contentStyle={{
                background: "var(--color-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                fontSize: 11,
                padding: "4px 8px",
              }}
              labelFormatter={(iteration) => `Iteration ${iteration}`}
              formatter={(value) => [formatRupees(Number(value)), "best"]}
            />
            <Line
              type="stepAfter"
              dataKey="cost"
              stroke="var(--color-ink)"
              strokeWidth={1.75}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-1.5 text-2xs text-muted-foreground">
        {improvement > 0 ? (
          <>
            Fell {formatRupees(improvement)} from{" "}
            <span className="tabular-nums">{formatRupees(first)}</span> at the first
            iteration. The line is monotone because each point is the best cost seen so
            far, not that iteration&apos;s sample.
          </>
        ) : (
          <>
            Flat from the first iteration — the starting population was not improved
            on. That happens on a small instance where the initial assignment is
            already optimal.
          </>
        )}
      </p>
    </div>
  );
}
