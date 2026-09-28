"use client";

/**
 * The dashboard shell: owns the scenario/solution state and lays out the four
 * regions — header, stop list, map, summary, stat bar.
 *
 * The pipeline is exactly the documented one: `POST /scenarios` to build an
 * instance, then `POST /optimize/{id}` to solve it and `GET /graph/delhi` for
 * the road geometry. The graph fetch is only possible once a scenario exists
 * (it is scoped by scenario id), so it runs in parallel with the solve rather
 * than after it.
 *
 * The map area is a stack of up to three surfaces, and the order matters:
 * the MapLibre canvas is always mounted at the bottom so it can spend the whole
 * solve initialising and pulling basemap tiles behind whatever is covering it;
 * the skeleton sits over that while the first solve runs; and the hero sits on
 * top of both only while there is no solution at all. Every handover between
 * them is a cross-fade, because the transitions here are what Phase 8's
 * incident simulator will be judged on.
 */

import dynamic from "next/dynamic";
import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Loader2, MapPin, Play } from "lucide-react";

import Header from "./header";
import HeroBackdrop from "./hero-backdrop";
import MapSkeleton from "./map-skeleton";
import RouteSummary from "./route-summary";
import StatBar from "./stat-bar";
import StopsPanel from "./stops-panel";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { REDRAW_MS } from "@/lib/motion";
import {
  SAMPLE_SCENARIO_PAYLOAD,
  createScenario,
  fetchGraph,
  optimizeScenario,
  type GraphGeoJSON,
  type OptimizeResponse,
  type ScenarioResponse,
} from "@/lib/api";

// MapLibre needs `window` and WebGL, so the map is client-only. Its chunk is
// small and the basemap is slow, so the placeholder here is only ever on screen
// for a frame — the map skeleton covers the wait that actually matters.
const RouteMap = dynamic(() => import("./route-map"), {
  ssr: false,
  loading: () => <MapSkeleton visible />,
});

export default function Dashboard() {
  const [scenario, setScenario] = useState<ScenarioResponse | null>(null);
  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [graph, setGraph] = useState<GraphGeoJSON | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSample = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const created = await createScenario(SAMPLE_SCENARIO_PAYLOAD);
      // The graph is scoped by scenario id, so it can only start once the
      // scenario exists — but it does not depend on the solve, so it overlaps it.
      const [solved, roadGraph] = await Promise.all([
        optimizeScenario(created.scenario_id),
        fetchGraph(created.scenario_id),
      ]);
      setScenario(created);
      setResult(solved);
      setGraph(roadGraph);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * The skeleton outlives the first solution by one fade.
   *
   * Unmounting it the moment `result` lands would cut from a full-screen
   * placeholder to a map whose routes have only just started fading up — a
   * blank frame in the middle, which is the exact flicker this pass exists to
   * remove. A timeout rather than `transitionend`, because under a
   * reduced-motion preference there is no transition to end and the event would
   * never arrive.
   */
  const [skeletonRetired, setSkeletonRetired] = useState(false);

  useEffect(() => {
    if (!result) {
      setSkeletonRetired(false);
      return;
    }
    const timer = window.setTimeout(() => setSkeletonRetired(true), REDRAW_MS + 80);
    return () => window.clearTimeout(timer);
  }, [result]);

  const routes = result?.routes ?? [];
  const activeRoutes = routes.filter((route) => route.stops.length > 0);
  const totalStops = activeRoutes.reduce((sum, route) => sum + route.stops.length, 0);

  // With no solution there is exactly one surface, and it shows either the
  // invitation or the failure — never both, and never a blank map.
  const showHero = !result;
  const showEmptyState = showHero && !loading && !error;
  const showErrorState = showHero && !loading && Boolean(error);
  const showMapSkeleton = loading && !result && !skeletonRetired;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas">
      <Header />

      <div className="flex min-h-0 flex-1">
        <StopsPanel
          scenario={scenario}
          routes={routes}
          loading={loading}
          onLoadSample={loadSample}
        />

        <main className="relative min-w-0 flex-1">
          <RouteMap graph={graph} routes={routes} depotNode={scenario?.depot.node ?? null} />

          {/* Mounted for one fade past the first solution; see `skeletonRetired`. */}
          {skeletonRetired ? null : (
            <MapSkeleton visible={showMapSkeleton} />
          )}

          {showHero ? (
            <HeroBackdrop visible={!loading}>
              {showErrorState ? (
                <Card className="max-w-md shadow-2xl [--card-spacing:--spacing(6)]">
                  <CardContent className="flex flex-col items-center gap-3 text-center">
                    <span className="flex size-10 items-center justify-center rounded-full bg-danger/10 text-danger">
                      <AlertTriangle className="size-5" aria-hidden />
                    </span>
                    <div>
                      <p className="text-base font-semibold">Could not load the scenario</p>
                      <p className="mt-1 text-xs break-words text-muted-foreground">{error}</p>
                      <p className="mt-2 text-2xs text-muted-foreground">
                        Check the backend is running on{" "}
                        <code className="font-mono">
                          {process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}
                        </code>
                        .
                      </p>
                    </div>
                    <Button onClick={loadSample} size="lg">
                      <Play className="size-4" aria-hidden />
                      Try again
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                <Card className="max-w-md shadow-2xl [--card-spacing:--spacing(6)]">
                  <CardContent className="flex flex-col items-center gap-3 text-center">
                    <span className="flex size-10 items-center justify-center rounded-full bg-ink text-white">
                      <MapPin className="size-5" aria-hidden />
                    </span>
                    <div>
                      <p className="text-base font-semibold">No scenario loaded</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Load a sample scenario to solve a 12-delivery instance on the real Delhi
                        road network and see the routes drawn here.
                      </p>
                    </div>
                    <Button onClick={loadSample} size="lg">
                      Load Sample Scenario
                    </Button>
                  </CardContent>
                </Card>
              )}
            </HeroBackdrop>
          ) : null}

          {loading ? (
            <div className="qgati-fade-in pointer-events-none absolute top-3 left-1/2 z-40 -translate-x-1/2">
              <div className="flex items-center gap-2 rounded-full bg-ink px-3 py-1.5 text-xs text-white shadow-lg">
                <Loader2 className="size-3.5 animate-spin" aria-hidden />
                Running ACO on the Delhi graph…
              </div>
            </div>
          ) : null}
        </main>

        <aside className="w-panel shrink-0 space-y-4 overflow-y-auto border-l border-border bg-canvas p-4">
          <RouteSummary result={result} loading={loading} />
        </aside>
      </div>

      <StatBar
        stops={result ? totalStops : null}
        vehiclesUsed={result ? activeRoutes.length : null}
        vehiclesTotal={result ? routes.length : null}
        travelTime={result ? result.travel_cost : null}
        solverName={result ? result.solver_name : null}
        loading={loading}
      />
    </div>
  );
}
