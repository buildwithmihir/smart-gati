"use client";

/**
 * The five solvers' costs as bars.
 *
 * ## Why this axis starts at zero, when the convergence chart's does not
 *
 * `convergence-chart.tsx` hides its y-axis and spans `dataMin`→`dataMax`, and the
 * reasoning recorded there is right *for a line*: the line's height encodes a
 * position on a scale, so an axis that starts at the minimum removes dead space
 * without removing information.
 *
 * A bar encodes its value as **length**, and that inverts the argument. Truncating
 * a bar chart's baseline makes a bar twice as long for a 0.4% difference, which is
 * the single most common way a bar chart lies. Here the solvers are genuinely
 * close — that is the benchmark's actual finding — and a truncated axis would
 * dress a photo finish up as a rout. So the baseline is zero and every bar is
 * labelled with its own figure, which also means the axis does not have to be
 * read to get a number out of this.
 *
 * ## What is not drawn
 *
 * A solver that did not run has **no bar**. The backend nulls a skipped solver's
 * figures precisely so it cannot be read as a zero-cost answer, and a zero-height
 * bar is that same misreading in a different medium: it looks like a solver that
 * ran and scored nothing. The table beside this says which solvers those were and
 * why.
 */

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatRupees } from "@/lib/format";
import type { SolverResult } from "@/lib/api";

/** Height of the plot area, in pixels. Taller than the convergence line: this
 *  one has an axis, and labels above each bar need room to sit. */
const CHART_HEIGHT = 200;

/** Whole rupees on the axis — the paise are the table's job, not the axis's. */
function axisRupees(value: number): string {
  return `₹${Math.round(value).toLocaleString("en-IN")}`;
}

/** The label above each bar: the figure, at a size that fits five of them. */
function barLabel(value: unknown): string {
  return typeof value === "number" ? `₹${Math.round(value).toLocaleString("en-IN")}` : "";
}

export default function CompareChart({ results }: { results: SolverResult[] }) {
  const data = useMemo(
    () =>
      results
        // Absent, not zero — see the file docstring.
        .filter((result) => result.skipped === null && result.travel_cost !== null)
        .map((result) => ({
          key: result.solver,
          name: result.solver_name,
          cost: result.travel_cost as number,
          gap: result.gap_vs_best_pct,
        })),
    [results],
  );

  if (data.length === 0) {
    return (
      <p className="text-2xs text-muted-foreground">
        No solver produced a cost for this scenario, so there is nothing to draw. The table above
        says why each one was skipped.
      </p>
    );
  }

  // Ties are real here and both get the winning colour: two solvers that land on
  // the same cost are both the best found, and picking one by index would claim a
  // winner the numbers do not support.
  const best = Math.min(...data.map((row) => row.cost));
  const winners = data.filter((row) => row.cost === best).length;

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="text-2xs text-muted-foreground">Weighted objective, per solver</span>
        <span className="text-xs font-medium tabular-nums">
          {formatRupees(best)}{" "}
          <span className="font-normal text-muted-foreground">
            best of {data.length}
            {winners > 1 ? ` · ${winners}-way tie` : ""}
          </span>
        </span>
      </div>

      <div style={{ height: CHART_HEIGHT }} data-compare-chart={data.length}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 16, right: 4, bottom: 0, left: 4 }}>
            <CartesianGrid stroke="var(--color-border)" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--color-border)" }}
              interval={0}
            />
            <YAxis
              // Zero-based, unlike the convergence chart — the file docstring
              // carries the reasoning. Ticks are compact because the axis is a
              // scale here, not a set of figures.
              tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
              tickLine={false}
              axisLine={false}
              width={64}
              tickFormatter={axisRupees}
            />
            <Tooltip
              cursor={{ fill: "var(--color-muted-foreground)", fillOpacity: 0.08 }}
              contentStyle={{
                background: "var(--color-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                fontSize: 11,
                padding: "4px 8px",
              }}
              formatter={(value) => [formatRupees(Number(value)), "cost"]}
            />
            <Bar dataKey="cost" isAnimationActive={false} radius={[2, 2, 0, 0]}>
              {data.map((row) => (
                <Cell
                  key={row.key}
                  // The winner in the ok ink so the result is visible without
                  // reading the axis, the rest in the page's own ink. Both are
                  // existing tokens; nothing new is introduced for this chart.
                  fill={row.cost === best ? "var(--color-ok)" : "var(--color-ink)"}
                />
              ))}
              <LabelList
                dataKey="cost"
                position="top"
                formatter={barLabel}
                style={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-1.5 text-2xs text-muted-foreground">
        Bars start at zero, so equal heights mean equal costs. These solvers land close together on
        this instance — the difference is in the gap column of the table, not in the bar lengths.
        {winners > 1
          ? ` ${winners} solvers reached the same best cost.`
          : ""}
      </p>
    </div>
  );
}
