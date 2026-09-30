"use client";

/**
 * The Compare tab: every solver's answer to the loaded scenario, side by side.
 *
 * ## Why this does not fetch on mount
 *
 * Every other view in the app loads what it shows. This one must not: a
 * comparison is **five solves**, brute force among them, and the dashboard's
 * sample scenario is already past the size where that is instant. Charging that
 * to anyone who clicks the tab — most of whom want to look at the table they
 * already ran, not run a new one — would make the tab feel broken on the way in.
 * So there is a **Run comparison** button and the tab opens idle.
 *
 * The cost of that choice is one more click, and it buys the thing the seed input
 * needs anyway: a moment where the parameters are visible and settable *before*
 * the work starts, rather than a result that arrived under settings the reader
 * never saw.
 *
 * ## Why the seed input exists at all
 *
 * Three of these five solvers are stochastic. Without a seed, two runs of the same
 * scenario differ for reasons that have nothing to do with solver quality, and a
 * reader comparing "QPSO 8 020" against "QPSO 8 240" would be comparing noise.
 * The default is the same seed the dashboard's own plan uses, so the comparison
 * and the plan on screen are answering the same question.
 *
 * ## Why the catalog is fetched once and separately
 *
 * `GET /solvers` is a static registry listing — it supplies the "default" and
 * "exact" badges and nothing else. The table is driven entirely by the comparison
 * response. So it is read once on mount and its failure is survivable: a
 * comparison without badges is a comparison, and blanking the table because a
 * decoration could not be fetched would be the wrong trade.
 */

import { useCallback, useEffect, useState } from "react";
import { BarChart3, Loader2 } from "lucide-react";

import CompareChart from "./compare-chart";
import CompareTable from "./compare-table";
import ApiFailure from "@/components/ui/api-failure";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  fetchCompare,
  fetchSolvers,
  SAMPLE_SOLVER_SEED,
  type CompareResponse,
  type SolverCatalog,
} from "@/lib/api";
import { formatRupees } from "@/lib/format";

function messageOf(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/** Placeholders shaped like the table to come, so the page resolves in place. */
function ResultsSkeleton() {
  return (
    <div className="space-y-4">
      <div className="space-y-2 rounded-xl bg-card p-4 ring-1 ring-foreground/10">
        <Skeleton className="h-3 w-40 rounded-sm" />
        <Skeleton className="h-48 w-full rounded-md" />
      </div>
      <div className="space-y-2 rounded-xl bg-card p-3 ring-1 ring-foreground/10">
        {[0, 1, 2, 3, 4, 5].map((index) => (
          <div key={index} className="flex items-center gap-4">
            <Skeleton className="h-3.5 w-24 shrink-0 rounded-sm" />
            <Skeleton className="h-3.5 w-20 shrink-0 rounded-sm" />
            <Skeleton className="h-3.5 w-16 shrink-0 rounded-sm" />
            <Skeleton className="h-3.5 flex-1 rounded-sm" />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function CompareView({ scenarioId }: { scenarioId: string | null }) {
  const [response, setResponse] = useState<CompareResponse | null>(null);
  const [catalog, setCatalog] = useState<SolverCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  /**
   * The seed, held as text.
   *
   * Text rather than a number so the field can be empty while it is being typed —
   * a number state would collapse `""` to `0` and turn a half-typed seed into a
   * valid one. It is parsed at the moment of the run, where an unparseable value
   * means "no seed" and is sent as such rather than as a zero.
   */
  const [seedText, setSeedText] = useState(String(SAMPLE_SOLVER_SEED));

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const [catalogResult] = await Promise.allSettled([fetchSolvers()]);
      if (cancelled) return;
      // A failed catalog leaves `catalog` null, which the table already handles —
      // so there is nothing to report here and no error state to set.
      if (catalogResult.status === "fulfilled") setCatalog(catalogResult.value);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const run = useCallback(async () => {
    if (scenarioId === null) return;
    // Bound to a local after the guard so the `await` below is unambiguously
    // working with a `string`. Narrowing a parameter does survive an await, but
    // this is one line and removes the question.
    const id = scenarioId;
    setRunning(true);
    try {
      const trimmed = seedText.trim();
      const parsed = Number(trimmed);
      const result = await fetchCompare(id, {
        // An empty or non-numeric field means "let the solvers choose", which is
        // a different request from seed 0 — so it is omitted, not defaulted.
        seed: trimmed !== "" && Number.isInteger(parsed) ? parsed : undefined,
      });
      setResponse(result);
      setError(null);
    } catch (cause) {
      // The previous result is kept rather than cleared: a comparison that
      // succeeded a moment ago is still a true comparison, and the banner below
      // says it is not current. Replacing it with an empty page would lose real
      // information to report a fetch failure.
      setError(messageOf(cause));
    } finally {
      setRunning(false);
    }
  }, [scenarioId, seedText]);

  // Nothing to compare against yet. This is the state the tab opens in whenever
  // the dashboard has not loaded a scenario, which is the ordinary first visit.
  if (scenarioId === null) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl space-y-5 p-5">
          <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border px-6 py-14 text-center">
            <span className="flex size-9 items-center justify-center rounded-full bg-muted text-muted-foreground">
              <BarChart3 className="size-4" aria-hidden />
            </span>
            <div>
              <p className="text-sm font-medium">No scenario to compare yet</p>
              <p className="mx-auto mt-1 max-w-md text-xs text-muted-foreground">
                This view runs all five solvers over the scenario the dashboard currently has
                loaded. Load the sample scenario on the Dashboard tab first, then come back.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const ran = response === null ? [] : response.results.filter((row) => row.skipped === null);
  const skippedCount = response === null ? 0 : response.results.length - ran.length;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-5 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="font-heading text-lg font-semibold tracking-tight">
              Compare algorithms
            </h1>
            <p className="mt-1 max-w-2xl text-xs text-muted-foreground">
              Every registered solver run against the loaded scenario, scored by the same fitness
              function over the same cost matrix — so a difference between two rows is a difference
              in search quality, not in accounting. Five solves, so this takes longer than a single
              plan.
            </p>
          </div>

          <div className="flex items-end gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-2xs font-medium tracking-wide text-muted-foreground uppercase">
                Seed
              </span>
              <input
                type="text"
                inputMode="numeric"
                value={seedText}
                onChange={(event) => setSeedText(event.target.value)}
                disabled={running}
                // The stochastic solvers use this; the deterministic ones ignore
                // it. Left empty, each solver picks its own — which is why the
                // field is not prefilled with a zero when cleared.
                title="Seeds the stochastic solvers. Blank lets each one choose."
                // The Button's own focus idiom, copied rather than invented: this
                // is the only bare input in the app, and a focus ring that reads
                // differently from every other control would look like a bug.
                className="h-8 w-24 rounded-md border border-border bg-card px-2 text-xs tabular-nums outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
              />
            </label>
            <Button size="sm" onClick={() => void run()} disabled={running}>
              {running ? (
                <Loader2 className="size-3.5 animate-spin" aria-hidden />
              ) : (
                <BarChart3 className="size-3.5" aria-hidden />
              )}
              {running ? "Running…" : response === null ? "Run comparison" : "Run again"}
            </Button>
          </div>
        </div>

        {/* A failure after a good run is a warning, not a teardown — the results
            below are real, they are just from the previous run. */}
        {error !== null && response !== null ? (
          <p className="rounded-lg bg-warn/5 px-3 py-2 text-2xs text-warn">
            This comparison could not be re-run — the results below are from the last successful
            run. {error}
          </p>
        ) : null}

        {running && response === null ? <ResultsSkeleton /> : null}

        {error !== null && response === null ? (
          <ApiFailure
            title="Could not run the comparison"
            message={error}
            onRetry={() => void run()}
            hint="The comparison is solved on the backend, so this tab needs it running."
          />
        ) : null}

        {response !== null ? (
          <>
            <Card size="sm">
              <CardContent className="flex flex-wrap items-baseline gap-x-5 gap-y-1.5 text-xs">
                <span>
                  <span className="text-muted-foreground">Best found </span>
                  <span className="font-medium tabular-nums">
                    {response.best_known === null ? "—" : formatRupees(response.best_known)}
                  </span>
                </span>
                <span>
                  <span className="text-muted-foreground">Exact optimum </span>
                  {response.optimal === null ? (
                    <span
                      className="text-muted-foreground"
                      title="Only the exhaustive solver can prove an optimum, and it did not run on an instance this size"
                    >
                      not known
                    </span>
                  ) : (
                    <span className="font-medium tabular-nums">{formatRupees(response.optimal)}</span>
                  )}
                </span>
                <span className="text-muted-foreground">
                  {ran.length} of {response.results.length} solvers ran
                  {skippedCount > 0 ? ` · ${skippedCount} skipped` : ""}
                </span>
              </CardContent>
            </Card>

            <Card size="sm">
              <CardContent>
                <CompareChart results={response.results} />
              </CardContent>
            </Card>

            <CompareTable response={response} catalog={catalog} />
          </>
        ) : null}

        {running && response !== null ? (
          <p className="flex items-center gap-2 text-2xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" aria-hidden />
            Running five solvers — the figures above are the previous run.
          </p>
        ) : null}
      </div>
    </div>
  );
}
