"use client";

/**
 * The History tab: a summary strip over a page of past runs.
 *
 * ## What it fetches, and when
 *
 * Both endpoints on mount, and both again on Refresh. It is **mounted only while
 * the tab is up** — the app shell renders it conditionally rather than hiding it
 * — so every visit is a fresh read rather than a table left over from ten
 * minutes ago. That is the behaviour a history view wants, and it also sidesteps
 * the offset-vs-cursor problem: `GET /analytics/runs` pages by offset, so a solve
 * landing mid-read shifts every row down one and a second fetch of the same page
 * can repeat a row. Refetching on demand keeps that window to the gap between
 * opening the tab and clicking Refresh, rather than to the whole time the tab
 * was hidden.
 *
 * ## Why the two are fetched together and not with `Promise.all`
 *
 * They are independent, and a failure of one should not blank the other — the
 * summary is a handful of aggregates and the table is the evidence, and a reader
 * whose summary failed but whose rows arrived is better served by seeing the
 * rows. So each half has its own state, and each renders its own failure.
 *
 * ## Three states, all explicit
 *
 * Loading, empty and data are distinct on screen, and so is "the backend is not
 * running" — which for this tab is the likely failure, since it is the only view
 * that talks to the API without a scenario behind it.
 */

import { useCallback, useEffect, useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";

import RunsTable from "./runs-table";
import SummaryStats from "./summary-stats";
import ApiFailure from "@/components/ui/api-failure";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  fetchRuns,
  fetchRunSummary,
  type AnalyticsSummary,
  type RunPage,
} from "@/lib/api";

/**
 * Rows per page.
 *
 * The API's own maximum is far higher and its default is 100; 25 is chosen for
 * the screen, where a longer table is scrolled past rather than read. The value
 * is echoed back on every page response, so nothing here re-derives what the
 * server did.
 */
const PAGE_SIZE = 25;

function messageOf(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * What this tab specifically needs from the backend.
 *
 * `ApiFailure` renders the shared failure and adds this only when the API was
 * unreachable, which for this tab is the likely cause: it is the one view that
 * reads the API without a scenario behind it.
 */
const UNREACHABLE_HINT =
  "The run history lives in SQLite on the backend, so this tab needs it running.";

/** Placeholders shaped like the summary cards they become. */
function SummarySkeleton() {
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {[0, 1, 2, 3].map((index) => (
        <div key={index} className="space-y-2 rounded-xl bg-card p-3 ring-1 ring-foreground/10">
          <Skeleton className="h-2.5 w-20 rounded-sm" />
          <Skeleton className="h-4 w-16 rounded-sm" />
          <Skeleton className="h-2.5 w-24 rounded-sm" />
        </div>
      ))}
    </div>
  );
}

/** Placeholders shaped like the table's rows, so the page resolves in place. */
function TableSkeleton() {
  return (
    <div className="space-y-2 rounded-xl bg-card p-3 ring-1 ring-foreground/10">
      {[0, 1, 2, 3, 4].map((index) => (
        <div key={index} className="flex items-center gap-4">
          <Skeleton className="h-3.5 w-28 shrink-0 rounded-sm" />
          <Skeleton className="h-4 w-16 shrink-0 rounded-full" />
          <Skeleton className="h-3.5 w-20 shrink-0 rounded-sm" />
          <Skeleton className="h-3.5 w-14 shrink-0 rounded-sm" />
          <Skeleton className="h-3.5 flex-1 rounded-sm" />
        </div>
      ))}
    </div>
  );
}

export default function HistoryView() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const [page, setPage] = useState<RunPage | null>(null);
  const [runsError, setRunsError] = useState<string | null>(null);

  const [offset, setOffset] = useState(0);
  /** True while a page fetch is in flight, so the buttons disable rather than queue. */
  const [paging, setPaging] = useState(false);
  /** True only for the first load, which is the one that gets skeletons. */
  const [loading, setLoading] = useState(true);

  /**
   * Both reads, each landing in its own state.
   *
   * `silent` is for the paging and refresh paths: the rows already on screen are
   * still the truth while the next page is fetched, so replacing them with
   * skeletons would be a flicker that says the data was lost and found again.
   */
  const load = useCallback(async (nextOffset: number, silent: boolean) => {
    if (!silent) setLoading(true);
    setPaging(true);

    const [summaryResult, runsResult] = await Promise.allSettled([
      fetchRunSummary(),
      fetchRuns({ limit: PAGE_SIZE, offset: nextOffset }),
    ]);

    if (summaryResult.status === "fulfilled") {
      setSummary(summaryResult.value);
      setSummaryError(null);
    } else {
      // The summary keeps whatever it last had rather than being cleared. A
      // stale aggregate labelled as stale beats an empty box, and the error
      // below is what says it is stale.
      setSummaryError(messageOf(summaryResult.reason));
    }

    if (runsResult.status === "fulfilled") {
      setPage(runsResult.value);
      setRunsError(null);
    } else {
      setRunsError(messageOf(runsResult.reason));
    }

    setLoading(false);
    setPaging(false);
  }, []);

  // Mount only — the component is unmounted whenever the tab is left, so this
  // runs once per visit rather than once per page of results.
  useEffect(() => {
    void load(0, false);
  }, [load]);

  const changeOffset = useCallback(
    (next: number) => {
      setOffset(next);
      void load(next, true);
    },
    [load],
  );

  const refresh = useCallback(() => {
    // Reloads the page the reader is actually on, not page 0: a Refresh that
    // silently jumped back to the top would lose their place in the table.
    void load(offset, true);
  }, [load, offset]);

  // Only the first load is skeletoned: after that the previous results stay on
  // screen while the new ones arrive.
  const showSkeletons = loading && summary === null && page === null;

  const summaryFailed = summaryError !== null && summary === null;
  const runsFailed = runsError !== null && page === null;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-5 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="font-heading text-lg font-semibold tracking-tight">Run history</h1>
            <p className="mt-1 max-w-2xl text-xs text-muted-foreground">
              Every solve the backend has run, newest first — a dispatch, an{" "}
              <code className="font-mono">optimize</code>, a re-plan, or an{" "}
              <code className="font-mono">avoid-road</code>. Each row is the solve&apos;s own
              reported figures, and an incident-triggered re-plan also carries the remaining
              travel time before and after.
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={refresh} disabled={paging}>
            <RefreshCw className={paging ? "size-3.5 animate-spin" : "size-3.5"} aria-hidden />
            Refresh
          </Button>
        </div>

        {/* A summary that failed after a good load is a warning, not a teardown:
            the numbers below are real, they are just not current. */}
        {summaryError !== null && summary !== null ? (
          <p className="rounded-lg bg-warn/5 px-3 py-2 text-2xs text-warn">
            The summary could not be refreshed — the figures below are from the last successful
            read. {summaryError}
          </p>
        ) : null}

        {showSkeletons ? <SummarySkeleton /> : null}

        {summaryFailed ? (
          <ApiFailure
            title="Could not load the summary"
            message={summaryError}
            onRetry={refresh}
            hint={UNREACHABLE_HINT}
          />
        ) : null}

        {summary !== null ? <SummaryStats summary={summary} /> : null}

        {showSkeletons ? <TableSkeleton /> : null}

        {runsFailed ? (
          <ApiFailure
            title="Could not load the runs"
            message={runsError}
            onRetry={refresh}
            hint={UNREACHABLE_HINT}
          />
        ) : null}

        {page !== null ? (
          <RunsTable page={page} busy={paging} onOffsetChange={changeOffset} />
        ) : null}

        {loading && !showSkeletons ? (
          <p className="flex items-center gap-2 text-2xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" aria-hidden />
            Reading the run history…
          </p>
        ) : null}
      </div>
    </div>
  );
}
