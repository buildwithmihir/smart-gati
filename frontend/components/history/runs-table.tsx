"use client";

/**
 * The past runs, newest first.
 *
 * A semantic `<table>` with Tailwind token classes, and deliberately not a
 * `components/ui/table` primitive: there is no table in that folder, this is its
 * only caller, and adding a component library's worth of surface for one view is
 * more than the feature needs. The same call `why-this-route.tsx` made when it
 * built a disclosure out of `Button` rather than introduce an accordion.
 *
 * ## What the columns mean, and the one way they can mislead
 *
 * | When | Kind | Scenario | Solver | Cost | Runtime | ETA |
 *
 * `cost`, `travel_time` and the rest cover **different work from one kind to the
 * next**: for `dispatch` and `optimize` they are the whole scenario, for
 * `reoptimize` the unserved remainder, and for `avoid_road` the whole fleet's
 * remainder after the reporting vehicle was re-solved — the unserved stops, from
 * wherever each vehicle had got to, which is necessarily less. The Scenario column
 * therefore reports the *scenario's* size, not the run's coverage, and says so in
 * its own sub-line. Comparing a re-plan's cost against a dispatch's and reading the
 * difference as a saving is the mistake this table cannot prevent but can at least
 * not invite.
 *
 * The ETA column is the other half of that: it appears only on a row that had a
 * baseline, and on such a row the two numbers *are* comparable, because both are
 * evaluations of the unserved work under the same conditions.
 *
 * ## Why it also reads as a timeline
 *
 * A left rule down the first column and a monospace timestamp. The request asked
 * for "a table/timeline" and this is both at once: evenly-aligned columns to
 * compare rows, and a descending spine that shows the order they happened in.
 */

import { ChevronLeft, ChevronRight, History } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  formatCount,
  formatRupees,
  formatRunTimestamp,
  formatRuntime,
  formatSeconds,
  formatSignedSeconds,
} from "@/lib/format";
import type { RunEntry, RunPage, RunKind } from "@/lib/api";

/** One badge per kind, so a row's origin is readable without reading the row. */
function KindBadge({ kind }: { kind: RunKind }) {
  switch (kind) {
    case "dispatch":
      return <Badge>dispatch</Badge>;
    case "optimize":
      return <Badge variant="secondary">optimize</Badge>;
    case "reoptimize":
      return <Badge variant="outline">re-plan</Badge>;
    case "avoid_road":
      // The warn ink rather than the destructive one: this is an operator
      // overriding a road, which is the third state the palette already has, and
      // painting it red would say something went wrong when the operator asked
      // for exactly this.
      return (
        <Badge variant="outline" className="border-warn/40 bg-warn/5 text-warn">
          avoid road
        </Badge>
      );
  }
}

/** The sub-line under a kind: what caused the run, when something did. */
function triggerOf(run: RunEntry): string | null {
  const parts: string[] = [];
  if (run.trigger) parts.push(run.trigger);
  if (run.affected_vehicle) parts.push(run.affected_vehicle);
  return parts.length > 0 ? parts.join(" · ") : null;
}

/**
 * The ETA cell: the before/after pair, its change, and what moved.
 *
 * `eta_saved_seconds` is `old − new`, so the *change* shown here is its negation
 * — negative meaning the re-plan came out cheaper, which is the sign convention
 * the rest of the app uses for deltas. A `null` pair (every first answer) is an
 * em dash, because a dispatch has nothing it could have improved on.
 */
function EtaCell({ run }: { run: RunEntry }) {
  if (run.old_eta_seconds === null || run.new_eta_seconds === null) {
    return (
      <span className="text-muted-foreground" title="A first answer — nothing to compare against">
        —
      </span>
    );
  }

  const change = run.new_eta_seconds - run.old_eta_seconds;

  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs tabular-nums">
        {formatSeconds(run.old_eta_seconds)}
        <span className="px-1 text-muted-foreground" aria-hidden>
          →
        </span>
        <span className="sr-only">became</span>
        {formatSeconds(run.new_eta_seconds)}
      </span>
      <span
        className={`text-2xs font-medium tabular-nums ${
          change < 0 ? "text-ok" : change > 0 ? "text-warn" : "text-muted-foreground"
        }`}
        // The two magnitudes are above; this is the difference between them.
        title="Remaining travel time, before and after re-planning — both priced with the incident in place"
      >
        {formatSignedSeconds(change)}
        {run.moved > 0 ? (
          <span className="ml-1.5 font-normal text-muted-foreground">
            · {formatCount(run.moved)} moved
          </span>
        ) : null}
      </span>
    </div>
  );
}

/** Every column this table has, so the header is stated once rather than twice. */
const COLUMNS = ["When", "Kind", "Scenario", "Solver", "Cost", "Runtime", "ETA"] as const;

const CELL = "px-3 py-2.5 align-top";

export default function RunsTable({
  page,
  busy,
  onOffsetChange,
}: {
  page: RunPage;
  /** True while a page is in flight — the buttons disable rather than queue. */
  busy: boolean;
  onOffsetChange: (offset: number) => void;
}) {
  if (page.items.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border px-6 py-12 text-center">
        <span className="flex size-9 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <History className="size-4" aria-hidden />
        </span>
        <p className="text-sm font-medium">No runs recorded yet</p>
        <p className="max-w-sm text-xs text-muted-foreground">
          Load the sample scenario, file an incident and re-plan around it, or call{" "}
          <code className="font-mono">POST /optimize/&#123;id&#125;</code>. Every solve writes a row
          — a refused one does not.
        </p>
      </div>
    );
  }

  const first = page.offset + 1;
  const last = page.offset + page.items.length;
  const canGoBack = page.offset > 0;

  return (
    <div className="overflow-hidden rounded-xl bg-card ring-1 ring-foreground/10">
      {/* The table scrolls sideways on a narrow window rather than wrapping the
          figures, because a wrapped cost column stops being a column. */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">
            Past optimization runs, newest first. Cost and distance cover the whole scenario for a
            dispatch or an optimize, and only the unserved stops for a re-plan.
          </caption>
          <thead>
            <tr className="border-b border-border">
              {COLUMNS.map((column, index) => (
                <th
                  key={column}
                  scope="col"
                  className={`text-2xs font-medium tracking-wide text-muted-foreground uppercase ${
                    index === 0 ? "border-l-2 border-l-border py-2 pr-3 pl-3 text-left" : `${CELL} text-left`
                  }`}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page.items.map((run) => {
              const trigger = triggerOf(run);
              return (
                <tr
                  key={run.id ?? `${run.timestamp}-${run.kind}`}
                  className="border-b border-border/60 last:border-0 hover:bg-muted/40"
                >
                  <td className="border-l-2 border-l-border px-3 py-2.5 align-top">
                    {/* Monospace and unbroken: the one column whose width is
                        fixed by its content, which is what makes the rows read
                        as one descending spine rather than a ragged table. */}
                    <time
                      dateTime={run.timestamp}
                      className="font-mono text-xs whitespace-nowrap tabular-nums"
                    >
                      {formatRunTimestamp(run.timestamp)}
                    </time>
                  </td>

                  <td className={CELL}>
                    <div className="flex flex-col items-start gap-1">
                      <KindBadge kind={run.kind} />
                      {trigger ? (
                        <span className="text-2xs text-muted-foreground" title={run.trigger_detail ?? undefined}>
                          {trigger}
                        </span>
                      ) : null}
                    </div>
                  </td>

                  <td className={CELL}>
                    <div className="flex flex-col gap-0.5">
                      <span className="font-mono text-xs" title={run.scenario_id}>
                        {run.scenario_id.slice(0, 8)}
                      </span>
                      {/* The scenario's own size, not this run's coverage — see
                          the file docstring. Scenarios live in memory and do not
                          survive a restart, so the size is the only thing that
                          identifies the instance a row was solved against. */}
                      <span className="text-2xs whitespace-nowrap text-muted-foreground">
                        {formatCount(run.n_deliveries)} stops · {formatCount(run.n_vehicles)} vehicles
                      </span>
                    </div>
                  </td>

                  <td className={CELL}>
                    <div className="flex flex-col gap-0.5">
                      <span className="text-xs font-medium">{run.solver_name}</span>
                      <span className="font-mono text-2xs text-muted-foreground">
                        {run.seed === null ? "no seed" : `seed ${run.seed}`}
                      </span>
                    </div>
                  </td>

                  <td className={CELL}>
                    <div className="flex flex-col gap-0.5">
                      <span className="text-xs tabular-nums">{formatRupees(run.travel_cost)}</span>
                      <span className="text-2xs whitespace-nowrap text-muted-foreground tabular-nums">
                        {formatSeconds(run.travel_time)}
                        {run.feasible ? null : (
                          <span className="ml-1.5 font-medium text-warn">· infeasible</span>
                        )}
                      </span>
                    </div>
                  </td>

                  <td className={CELL}>
                    <div className="flex flex-col gap-0.5">
                      <span className="text-xs tabular-nums">{formatRuntime(run.runtime_ms)}</span>
                      {run.iterations === null ? null : (
                        <span className="text-2xs whitespace-nowrap text-muted-foreground tabular-nums">
                          {formatCount(run.iterations)} iter
                          {run.population === null ? "" : ` · pop ${formatCount(run.population)}`}
                        </span>
                      )}
                    </div>
                  </td>

                  <td className={CELL}>
                    <EtaCell run={run} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between gap-3 border-t border-border bg-muted/30 px-3 py-2">
        <span className="text-2xs text-muted-foreground tabular-nums">
          {formatCount(first)}–{formatCount(last)} of {formatCount(page.total)}
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            // `Next` is driven by the response's own `has_more`, not by
            // arithmetic on `total`: the server is the one that knows how many
            // rows matched, and a client that recomputed the boundary would
            // offer a page that does not exist the moment the two disagreed.
            disabled={busy || !canGoBack}
            onClick={() => onOffsetChange(Math.max(0, page.offset - page.limit))}
          >
            <ChevronLeft className="size-3.5" aria-hidden />
            Previous
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={busy || !page.has_more}
            onClick={() => onOffsetChange(page.offset + page.limit)}
          >
            Next
            <ChevronRight className="size-3.5" aria-hidden />
          </Button>
        </div>
      </div>
    </div>
  );
}
