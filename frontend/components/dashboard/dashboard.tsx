"use client";

/**
 * The dashboard shell: owns the scenario, the plan, the fleet and the incident,
 * and lays out the four regions — stop list, map, summary, stat bar.
 *
 * The header and the viewport frame live in `app-shell.tsx`, which is what
 * decides whether this is on screen at all — see `active` below.
 *
 * The pipeline is exactly the documented one, with one substitution that is
 * worth knowing before reading any of it: **a scenario is loaded by dispatching
 * a fleet, not by solving it.** `POST /optimize/{id}` would give a plan, but
 * `POST /reoptimize` — the whole point of this page — answers 404 without a
 * running watcher, because it derives where every vehicle is and what it has
 * left by reading one. So the load does `POST /scenarios` → `POST /watcher`,
 * and the watcher's response *is* the dispatched plan. One solve rather than
 * two, and the routes drawn are exactly the routes the vehicles are driving.
 *
 * ## What the incident flow is, in order
 *
 * 1. A kind is armed, and `picking` turns the map's cursor into a crosshair.
 * 2. A click arrives as a `PickedRoad` — the street, resolved to the one or two
 *    *directed* edges that make it up. See `lib/road-picking`.
 * 3. The highlight goes up immediately, with a fresh pulse token. It is the
 *    acknowledgement that the click landed, and it has to precede the answer
 *    rather than accompany it.
 * 4. One `POST /incident` per directed edge, then one `POST /reoptimize`.
 * 5. On success the re-planned routes become the plan and the dispatched ones
 *    stay on the map faintly underneath. On 409/422 the incident stands and the
 *    plan does not move — which is the truth, because nothing re-planned it.
 *
 * `POST /reoptimize` writes nothing, so "Clear Incident" does not need to ask
 * the backend for a plan: it reverts the incidents and puts the dispatched plan
 * back, which is provably the plan the fleet is still driving.
 *
 * The map area is a stack of up to three surfaces, and the order matters:
 * the MapLibre canvas is always mounted at the bottom so it can spend the whole
 * load initialising and pulling basemap tiles behind whatever is covering it;
 * the skeleton sits over that while the first solve runs; and the hero sits on
 * top of both only while there is no plan at all.
 */

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Loader2, MapPin, Play } from "lucide-react";

import HeroBackdrop from "./hero-backdrop";
import IncidentPanel, { type IncidentState, type PanelNotice } from "./incident-panel";
import MapSkeleton from "./map-skeleton";
import RouteSummary from "./route-summary";
import StatBar from "./stat-bar";
import StopsPanel from "./stops-panel";
import type { IncidentHighlight } from "./route-map";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { REDRAW_MS } from "@/lib/motion";
import { type PickedRoad } from "@/lib/road-picking";
import {
  ApiError,
  FLEET_POLL_MS,
  SAMPLE_FLEET_INTERVAL_SECONDS,
  SAMPLE_FLEET_TIME_SCALE,
  SAMPLE_SCENARIO_PAYLOAD,
  SAMPLE_SOLVER_SEED,
  clearIncident,
  createScenario,
  fetchGraph,
  fetchSolvers,
  injectIncident,
  planFromReopt,
  planFromWatcherStart,
  reoptimizeScenario,
  startWatcher,
  stopWatcher,
  watcherStatus,
  type DashboardPlan,
  type GraphGeoJSON,
  type IncidentKind,
  type IncidentOut,
  type ReoptimizeResponse,
  type ScenarioResponse,
  type WatcherStatus,
} from "@/lib/api";

// MapLibre needs `window` and WebGL, so the map is client-only. Its chunk is
// small and the basemap is slow, so the placeholder here is only ever on screen
// for a frame — the map skeleton covers the wait that actually matters.
const RouteMap = dynamic(() => import("./route-map"), {
  ssr: false,
  loading: () => <MapSkeleton visible />,
});

/**
 * The incident currently on the map, as this component needs it.
 *
 * Broader than the panel's view of the same thing, and that is why it lives
 * here: the map needs the street's *geometry* to draw the highlight, and the
 * panel needs the *reports* to show what they did. Both come out of the one
 * click.
 *
 * `incidents` is empty between the click and the first response, and that state
 * is visible rather than hidden — the panel says the road is being reported.
 * Anything else would mean showing a street as closed before the backend had
 * agreed to close it.
 */
type LiveIncident = IncidentState & {
  /** The street as clicked, in both directions, with its polyline. */
  road: PickedRoad;
  /** Bumped on every pick. The map pulses on a *change* in this. */
  pulse: number;
};

/** A failure, as a sentence. `ApiError.detail` is already written for a human. */
function messageOf(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * A message for the panel, framed by what kind of answer it was.
 *
 * **409 is not a failure.** `POST /reoptimize` answers it when it looked at the
 * network, the incident and the fleet and found nothing warranting a re-plan —
 * the endpoint working correctly and saying so. Painting that red would tell an
 * operator something broke when the system had just made a judgement, so it is
 * carried through as the other tone and rendered as a note.
 */
function noticeOf(cause: unknown): PanelNotice {
  return {
    detail: messageOf(cause),
    tone: cause instanceof ApiError && cause.status === 409 ? "refusal" : "failure",
  };
}

/**
 * Whether this view is the one on screen.
 *
 * The app shell keeps the dashboard *mounted* while History is up — unmounting
 * would discard the loaded scenario, the plan and the filed incident, and there
 * is no rehydrate path, so coming back would land on the empty hero while the
 * fleet it dispatched kept ticking. The cost of that choice is paid by MapLibre,
 * which sizes its canvas from the container and listens for *window* resize: a
 * canvas inside `display:none` measures 0×0 and stays there. `active` is how the
 * map finds out it has been given a size again — see `route-map.tsx`.
 */
type DashboardProps = {
  active?: boolean;
  /**
   * Reports which scenario is loaded, so the Compare tab can solve the same
   * instance the dashboard is showing.
   *
   * Called **from `loadSample`, not from an effect**. An effect watching
   * `scenarioId` would call the shell's `setState` on every render that touched
   * its dependencies, and since the value it sets is what a parent re-renders
   * on, that is a loop. It is an event — "a scenario finished loading" — and an
   * event handler is what it belongs in.
   */
  onScenarioChange?: (scenarioId: string | null) => void;
};

export default function Dashboard({ active = true, onScenarioChange }: DashboardProps) {
  const [scenario, setScenario] = useState<ScenarioResponse | null>(null);
  /** The plan the fleet was dispatched on. Never changes after a load. */
  const [dispatch, setDispatch] = useState<DashboardPlan | null>(null);
  /** The plan on screen: the dispatched one, or the re-planned one over it. */
  const [plan, setPlan] = useState<DashboardPlan | null>(null);
  const [graph, setGraph] = useState<GraphGeoJSON | null>(null);
  const [fleet, setFleet] = useState<WatcherStatus | null>(null);
  const [loading, setLoading] = useState(false);
  /** True while a report, a re-plan or a revert is in flight. */
  const [busy, setBusy] = useState(false);
  /** A failure that stopped a scenario from loading at all — shown in the hero. */
  const [error, setError] = useState<string | null>(null);
  /** What the backend last said about the incident flow — shown in the panel. */
  const [notice, setNotice] = useState<PanelNotice | null>(null);
  const [arming, setArming] = useState<IncidentKind | null>(null);
  const [incident, setIncident] = useState<LiveIncident | null>(null);
  const [report, setReport] = useState<ReoptimizeResponse | null>(null);

  /** The scenario whose fleet is out, so a reload can stand it down. */
  const liveFleetRef = useRef<string | null>(null);

  const scenarioId = scenario?.scenario_id ?? null;

  const loadSample = useCallback(async () => {
    setLoading(true);
    setError(null);
    setNotice(null);
    setArming(null);
    setIncident(null);
    setReport(null);
    setFleet(null);
    setDispatch(null);
    setPlan(null);

    // A fleet left running against the previous scenario would keep ticking,
    // keep logging rows and keep holding a solved instance that nothing on
    // screen refers to any more. Best effort: it is a simulator holding no
    // state this UI needs, so failing to stand it down is not a reason to
    // refuse to load.
    const previous = liveFleetRef.current;
    liveFleetRef.current = null;
    if (previous) {
      stopWatcher(previous).catch(() => {});
    }

    try {
      const created = await createScenario(SAMPLE_SCENARIO_PAYLOAD);

      // The graph is scoped by scenario id, so it can only start once the
      // scenario exists — but it depends on nothing else here, so it overlaps
      // both the solve and the solver catalogue.
      const [started, roadGraph, catalog] = await Promise.all([
        startWatcher(created.scenario_id, {
          seed: SAMPLE_SOLVER_SEED,
          intervalSeconds: SAMPLE_FLEET_INTERVAL_SECONDS,
          timeScale: SAMPLE_FLEET_TIME_SCALE,
        }),
        fetchGraph(created.scenario_id),
        // Only used for the solver's display name and the effort it ran at, so
        // a failure here degrades the label to the solver key upper-cased
        // rather than taking the scenario down with it.
        fetchSolvers().catch(() => null),
      ]);

      const dispatched = planFromWatcherStart(started, catalog);
      liveFleetRef.current = created.scenario_id;

      setScenario(created);
      setGraph(roadGraph);
      setDispatch(dispatched);
      setPlan(dispatched);
      setFleet(started);
      // After the state above, so a listener that reads this as "the dashboard
      // now has scenario X" is not told so a render before it does.
      onScenarioChange?.(created.scenario_id);
    } catch (cause) {
      liveFleetRef.current = null;
      setError(messageOf(cause));
      // The previous scenario is gone — a fleet was stood down and the state
      // above was cleared — so the shell must not keep offering a comparison
      // against an instance this view can no longer show.
      onScenarioChange?.(null);
    } finally {
      setLoading(false);
    }
  }, [onScenarioChange]);

  /**
   * The skeleton outlives the first solution by one fade.
   *
   * Unmounting it the moment the plan lands would cut from a full-screen
   * placeholder to a map whose routes have only just started fading up — a
   * blank frame in the middle, which is the exact flicker this pass exists to
   * remove. A timeout rather than `transitionend`, because under a
   * reduced-motion preference there is no transition to end and the event would
   * never arrive.
   */
  const [skeletonRetired, setSkeletonRetired] = useState(false);

  useEffect(() => {
    if (!plan) {
      setSkeletonRetired(false);
      return;
    }
    const timer = window.setTimeout(() => setSkeletonRetired(true), REDRAW_MS + 80);
    return () => window.clearTimeout(timer);
  }, [plan]);

  // -- the fleet's position ------------------------------------------------ //
  const watcherRunning = fleet?.running ?? false;

  useEffect(() => {
    if (!scenarioId || !watcherRunning) return;

    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const status = await watcherStatus(scenarioId);
        if (!cancelled) setFleet(status);
      } catch {
        // A 404 means the fleet is gone — stopped out from under this page, or
        // the backend restarted. Either way there is nothing left to poll, and
        // ticking a failing request every five seconds forever is worse than
        // the pill quietly going out.
        if (!cancelled) setFleet(null);
      }
    }, FLEET_POLL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [scenarioId, watcherRunning]);

  // Only the vehicles array is read here, and only by the map — memoised so an
  // unrelated re-render does not look like the fleet moved.
  const fleetVehicles = useMemo(() => fleet?.vehicles ?? [], [fleet]);

  // -- the incident flow --------------------------------------------------- //
  const arm = useCallback((kind: IncidentKind) => {
    setNotice(null);
    setArming(kind);
  }, []);

  const handlePickRoad = useCallback(
    async (road: PickedRoad) => {
      if (!scenarioId || !arming) return;

      const kind = arming;
      // Disarmed on the click rather than on the response: a second click while
      // the first is still being priced would file a second incident, and the
      // panel has no way to draw two.
      setArming(null);
      setBusy(true);
      setNotice(null);
      setReport(null);

      // The pulse token only ever has to be *different* from the last one, so
      // the wall clock is enough and no counter has to be threaded through.
      const pulse = Date.now();

      // Up before the first request goes out. The pulse is the acknowledgement
      // that the click landed, and an acknowledgement that arrives with the
      // answer is not one — so the highlight covers the whole round trip and
      // the plan arrives underneath it.
      setIncident({ kind, road, u: road.u, v: road.v, incidents: [], changedLegs: 0, pulse });

      const appliedIncidents: IncidentOut[] = [];
      let changedLegs = 0;
      try {
        // One call per directed edge. The graph emits each carriageway of a
        // two-way street separately — see `bothDirectionsOf` — so closing only
        // the one that was clicked would leave the street open the other way.
        for (const edge of road.edges) {
          const response = await injectIncident(scenarioId, kind, { u: edge.u, v: edge.v });
          appliedIncidents.push(response.incident);
          // Each response is a diff against the matrix as it stood when that
          // call was made, so these accumulate rather than overlap. A single
          // leg both reports re-price would be counted in both; the two
          // carriageways are different directed edges, so that is a corner
          // rather than the rule.
          changedLegs += response.changed_legs;
        }
      } catch (cause) {
        // The report was refused, so there is no incident to show — and a
        // highlight left on a road that is not closed would be the one thing on
        // screen that is untrue.
        //
        // The reports are per directed edge and only the failing one is rolled
        // back by the backend, so anything already accepted has to be withdrawn
        // here. Skipping this would leave a carriageway closed in the stored
        // cost matrix with nothing on screen saying so: every later solve on
        // this scenario would quietly route around a road this page had
        // forgotten about. Best effort, because a failed *un*-report leaves the
        // same state we are trying to avoid and there is nothing further to try.
        for (const applied of appliedIncidents) {
          try {
            await clearIncident(scenarioId, applied.incident_id);
          } catch {
            // Rolled back as far as it can be; the notice below is what the
            // operator reads either way.
          }
        }
        setIncident(null);
        setNotice(noticeOf(cause));
        setBusy(false);
        return;
      }

      setIncident((current) =>
        current ? { ...current, incidents: appliedIncidents, changedLegs } : current,
      );

      try {
        const replanned = await reoptimizeScenario(scenarioId);
        setReport(replanned);
        setPlan(planFromReopt(replanned));
      } catch (cause) {
        // A refusal is an answer. 409 says nothing has happened that warrants a
        // re-plan; 422 says the remaining work cannot be re-planned under these
        // conditions. Either way the incident stands and the plan does not
        // move — which is exactly what happened, and is why the map is left
        // alone here.
        setNotice(noticeOf(cause));
      } finally {
        setBusy(false);
      }
    },
    [arming, scenarioId],
  );

  const cancelArm = useCallback(() => setArming(null), []);

  const revert = useCallback(async () => {
    if (!scenarioId || !incident) return;
    setBusy(true);
    setNotice(null);

    // Every report is attempted even after one fails: a revert that gives up
    // halfway leaves a road closed in one direction and open in the other,
    // which is the state least likely to be noticed and most likely to confuse
    // whoever looks at the map next.
    const remaining: IncidentOut[] = [];
    let failure: unknown = null;
    for (const applied of incident.incidents) {
      try {
        await clearIncident(scenarioId, applied.incident_id);
      } catch (cause) {
        if (failure === null) failure = cause;
        remaining.push(applied);
      }
    }

    if (remaining.length > 0) {
      // Something is still reported, so the re-planned routes are still the
      // truthful plan and the highlight is still a live road.
      setIncident((current) => (current ? { ...current, incidents: remaining } : current));
      setNotice(noticeOf(failure));
      setBusy(false);
      return;
    }

    // `POST /reoptimize` never dispatched anything, so the plan the fleet is on
    // is still the one it was sent out with. Putting it back is a redraw, not a
    // request.
    setIncident(null);
    setReport(null);
    setPlan(dispatch);
    setBusy(false);
  }, [incident, dispatch, scenarioId]);

  // -- what the map is handed ---------------------------------------------- //
  //
  // Memoised on the fields rather than on `incident`, and that is load-bearing:
  // the incident object is replaced when the reports come back, and a new
  // highlight object would re-run the map's pulse effect — setting the road
  // flashing again every time a report landed. These three values do not change
  // across that patch, so the identity does not either.
  const highlightRoad = incident?.road ?? null;
  const highlightKind = incident?.kind ?? null;
  const highlightPulse = incident?.pulse ?? 0;

  const highlight = useMemo<IncidentHighlight | null>(
    () =>
      highlightRoad && highlightKind
        ? { kind: highlightKind, edges: highlightRoad.edges, pulse: highlightPulse }
        : null,
    [highlightRoad, highlightKind, highlightPulse],
  );

  // The dispatched plan is only drawn faintly while a *different* plan is on
  // screen. Before a re-plan lands — and after a refused one — the two are the
  // same routes and a ghost would be a second copy of them.
  const ghostRoutes = useMemo(
    () => (report && dispatch ? dispatch.routes : []),
    [report, dispatch],
  );

  const panelIncident = useMemo<IncidentState | null>(
    () =>
      incident
        ? {
            kind: incident.kind,
            u: incident.u,
            v: incident.v,
            incidents: incident.incidents,
            changedLegs: incident.changedLegs,
          }
        : null,
    [incident],
  );

  const routes = plan?.routes ?? [];
  const activeRoutes = routes.filter((route) => route.stops.length > 0);
  const totalStops = activeRoutes.reduce((sum, route) => sum + route.stops.length, 0);

  // The fleet's size, taken from the dispatch rather than from the plan on
  // screen. A re-optimization answers for the vehicles that had work left, so
  // counting its routes would shrink the denominator partway through the demo
  // and turn "one vehicle has finished" into "there were only ever two".
  const fleetSize = dispatch?.routes.length ?? routes.length;

  // With no plan there is exactly one surface, and it shows either the
  // invitation or the failure — never both, and never a blank map.
  const showHero = !plan;
  const showEmptyState = showHero && !loading && !error;
  const showErrorState = showHero && !loading && Boolean(error);
  const showMapSkeleton = loading && !plan && !skeletonRetired;

  return (
    <>
      <div className="flex min-h-0 flex-1">
        <StopsPanel
          scenario={scenario}
          routes={routes}
          loading={loading}
          onLoadSample={loadSample}
          explanation={plan?.explanation ?? null}
          convergence={plan?.convergence ?? []}
          solverName={plan?.solver_name ?? ""}
        />

        <main className="relative min-w-0 flex-1">
          <RouteMap
            graph={graph}
            routes={routes}
            ghostRoutes={ghostRoutes}
            depotNode={scenario?.depot.node ?? null}
            fleet={fleetVehicles}
            highlight={highlight}
            picking={arming !== null}
            onPickRoad={handlePickRoad}
            active={active}
          />

          {/* Mounted for one fade past the first solution; see `skeletonRetired`. */}
          {skeletonRetired ? null : <MapSkeleton visible={showMapSkeleton} />}

          {/* The watcher, at its quietest. It is the only thing on this page
              that changes without being asked to, so it gets a mark; it is also
              the least important, so it gets the smallest one there is and
              nothing else — no controls attach to it. */}
          {fleet?.running ? (
            <div
              className="qgati-fade-in pointer-events-none absolute top-3 right-3 z-30 flex items-center gap-2 rounded-full bg-ink/85 px-2.5 py-1 text-2xs text-white shadow"
              title={`Simulated fleet running · ${fleet.vehicles.length} vehicles · one tick every ${fleet.interval_seconds}s`}
            >
              <span className="qgati-live-dot size-1.5 shrink-0 rounded-full bg-white" aria-hidden />
              <span className="font-medium tracking-wide uppercase">live</span>
              <span className="text-white/60 tabular-nums">tick {fleet.ticks}</span>
            </div>
          ) : null}

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
                        . A scenario needs both its solve and its fleet — if the fleet could not
                        be dispatched, this message is why.
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
                        road network and put a simulated fleet on it. Then close a road and watch
                        the fleet be re-planned around it.
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
                Solving and dispatching a fleet…
              </div>
            </div>
          ) : null}
        </main>

        <aside className="w-panel shrink-0 space-y-4 overflow-y-auto border-l border-border bg-canvas p-4">
          <IncidentPanel
            arming={arming}
            incident={panelIncident}
            report={report}
            busy={busy}
            notice={notice}
            onArm={arm}
            onCancelArm={cancelArm}
            onClear={revert}
          />
          <RouteSummary
            result={plan}
            loading={loading}
            fleetSize={dispatch?.routes.length}
            caption={
              report
                ? "Re-planned — the unserved stops only, so this covers less than the dispatched round did."
                : undefined
            }
          />
        </aside>
      </div>

      {/* The stat bar reads figures only this component has — the plan's totals,
          the fleet's size, whether a re-plan is on screen — so it stays here
          rather than moving up to the shell. The shell wraps the pair in a flex
          column and the row above takes the slack. */}
      <StatBar
        stops={plan ? totalStops : null}
        vehiclesUsed={plan ? activeRoutes.length : null}
        vehiclesTotal={plan ? fleetSize : null}
        travelTime={plan ? plan.travel_time : null}
        cost={plan ? plan.travel_cost : null}
        solverName={plan ? plan.solver_name : null}
        note={report ? "Re-planned · remaining stops" : null}
        loading={loading}
      />
    </>
  );
}
