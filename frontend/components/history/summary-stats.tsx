"use client";

/**
 * The summary strip above the runs table.
 *
 * Four figures over the whole history, read from `GET /analytics/summary`, plus a
 * breakdown card. The one rule this file exists to keep is that **an unknown figure
 * renders as an em dash with its reason, never as a zero**. The backend returns
 * `null` for every average over an empty table — SQL's `AVG` over no rows is
 * `NULL` — precisely so that "no re-plan has ever run" cannot be displayed as
 * "re-planning has saved 0.0 s". Those are different statements and only one of
 * them is true.
 *
 * The ETA card is the one with a trap in it, and it is labelled rather than
 * smoothed over: its average is taken over the rows that had a baseline, which
 * is a subset of the runs above it. So it carries its own denominator, and the
 * improved/worsened split, which is also the only place a *negative* saving
 * becomes visible — a re-solve can land on a worse arrangement than the one in
 * hand, and that is worth being able to see.
 */

import { Card, CardContent } from "@/components/ui/card";
import {
  formatCount,
  formatRuntime,
  formatRunTimestamp,
  formatRupees,
  formatSeconds,
  formatSignedSeconds,
} from "@/lib/format";
import type { AnalyticsSummary } from "@/lib/api";

/**
 * How each kind reads. A kind the API grows later is not in this map and falls
 * back to its raw key in `Distribution` — legible, and visibly not-yet-labelled
 * rather than missing.
 */
const KIND_LABELS: Record<string, string> = {
  dispatch: "dispatch",
  optimize: "optimize",
  reoptimize: "re-plan",
  avoid_road: "avoid road",
};

/**
 * One figure, with a label, a sub-line and the reason when it is unknown.
 *
 * `value` is `null` for an unknown, and the caller passes the `reason` that
 * makes the dash readable — an em dash on its own is a hole, and "no re-plans
 * yet" is an answer.
 */
function Stat({
  label,
  value,
  detail,
  reason,
  tone = "default",
}: {
  label: string;
  value: string | null;
  detail: string | null;
  reason: string;
  tone?: "default" | "ok" | "warn";
}) {
  const valueTone =
    tone === "ok" ? "text-ok" : tone === "warn" ? "text-warn" : "text-foreground";

  return (
    <Card size="sm">
      <CardContent className="flex flex-col gap-1">
        <span className="text-2xs font-medium tracking-wide text-muted-foreground uppercase">
          {label}
        </span>
        {value === null ? (
          <>
            <span className="text-lg leading-none font-semibold text-muted-foreground" title={reason}>
              —
            </span>
            <span className="text-2xs text-muted-foreground">{reason}</span>
          </>
        ) : (
          <>
            <span className={`text-lg leading-none font-semibold tabular-nums ${valueTone}`}>
              {value}
            </span>
            {detail ? (
              <span className="text-2xs text-muted-foreground">{detail}</span>
            ) : (
              <span className="text-2xs text-muted-foreground">&nbsp;</span>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * A distribution as chips: "dispatch 1 · re-plan 1".
 *
 * Counts are `formatCount`ed rather than left bare, because every figure on this
 * page is a count of something different and a naked numeral beside a label is a
 * question rather than an answer.
 */
function Distribution({
  label,
  counts,
  labels = {},
  empty,
}: {
  label: string;
  counts: Record<string, number>;
  labels?: Record<string, string>;
  empty: string;
}) {
  const entries = Object.entries(counts);
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-2xs font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </span>
      {entries.length === 0 ? (
        <span className="text-xs text-muted-foreground">{empty}</span>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {entries.map(([key, count]) => (
            <span
              key={key}
              className="inline-flex items-center gap-1.5 rounded-md bg-muted px-2 py-1 text-xs"
            >
              <span className="text-muted-foreground">{labels[key] ?? key}</span>
              <span className="font-medium tabular-nums">{formatCount(count)}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/** The split of re-plans, as one sentence: "1 better · 0 worse · 0 unchanged". */
function etaSplit(summary: AnalyticsSummary): string | null {
  const { improved_runs: better, worsened_runs: worse, unchanged_runs: same } = summary.eta;
  if (summary.eta.runs === 0) return null;
  const parts = [`${formatCount(better)} better`];
  // Only mentioned once it has happened. "0 worse" is noise on a healthy
  // history; a non-zero worse is the single most interesting number here.
  if (worse > 0) parts.push(`${formatCount(worse)} worse`);
  if (same > 0) parts.push(`${formatCount(same)} unchanged`);
  return parts.join(" · ");
}

export default function SummaryStats({ summary }: { summary: AnalyticsSummary }) {
  const { eta } = summary;
  const span =
    summary.first_run_at && summary.last_run_at
      ? `${formatRunTimestamp(summary.first_run_at)} → ${formatRunTimestamp(summary.last_run_at)}`
      : null;
  const split = etaSplit(summary);

  return (
    <section className="space-y-3" aria-label="Run history summary">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Runs recorded"
          value={formatCount(summary.total_runs)}
          detail={span}
          reason="Nothing has been solved yet"
        />
        <Stat
          label="Average runtime"
          value={summary.avg_runtime_ms === null ? null : formatRuntime(summary.avg_runtime_ms)}
          detail={
            summary.min_runtime_ms === null || summary.max_runtime_ms === null
              ? null
              : `fastest ${formatRuntime(summary.min_runtime_ms)} · slowest ${formatRuntime(summary.max_runtime_ms)}`
          }
          reason="No runtime recorded yet"
        />
        <Stat
          label="Incident re-plans"
          value={formatCount(summary.incident_triggered_runs)}
          detail={
            summary.incident_triggered_runs === 0
              ? "none triggered by an incident or a flagged reading"
              : `of ${formatCount(summary.total_runs)} runs`
          }
          reason="Nothing reported against a road yet"
        />
        <Stat
          label="Average ETA saved"
          value={eta.avg_saved_seconds === null ? null : formatSignedSeconds(eta.avg_saved_seconds)}
          // The sign, and why it can be either. `eta_saved_seconds` is
          // old − new, so a positive average is the re-plan coming out ahead —
          // and it is deliberately not clamped, because a re-solve can be worse.
          detail={
            eta.avg_saved_seconds === null
              ? null
              : `over ${formatCount(eta.runs)} re-plan${eta.runs === 1 ? "" : "s"}${
                  eta.total_saved_seconds === null
                    ? ""
                    : ` · ${formatSignedSeconds(eta.total_saved_seconds)} total`
                }`
          }
          reason="Nothing has been re-planned yet"
          tone={
            eta.avg_saved_seconds === null
              ? "default"
              : eta.avg_saved_seconds > 0
                ? "ok"
                : eta.avg_saved_seconds < 0
                  ? "warn"
                  : "default"
          }
        />
      </div>

      <Card size="sm">
        <CardContent className="grid gap-4 sm:grid-cols-2">
          <Distribution
            label="By kind"
            counts={summary.runs_by_kind}
            labels={KIND_LABELS}
            empty="No runs recorded."
          />
          <Distribution
            label="By solver"
            counts={summary.runs_by_solver}
            empty="No runs recorded."
          />
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-border pt-3 text-2xs text-muted-foreground sm:col-span-2">
            <span>
              {formatCount(summary.feasible_runs)} feasible ·{" "}
              {formatCount(summary.infeasible_runs)} infeasible
            </span>
            {split ? <span>{split}</span> : null}
            {summary.avg_travel_cost === null ? null : (
              <span>average cost {formatRupees(summary.avg_travel_cost)}</span>
            )}
            {summary.avg_travel_time_seconds === null ? null : (
              <span>average travel time {formatSeconds(summary.avg_travel_time_seconds)}</span>
            )}
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
