"use client";

/**
 * The app frame: the header, and which view is inside it.
 *
 * Two views, one of which must not be destroyed when the other is shown.
 *
 * ## Why the dashboard is hidden rather than unmounted
 *
 * `Dashboard` owns everything the demo has accumulated — the loaded scenario,
 * the dispatched plan, the filed incident, and the `liveFleetRef` it would need
 * to stand the fleet down. It has no props to rehydrate from and no route to
 * come back through, so unmounting it and returning would land on the empty hero
 * while the fleet it dispatched kept ticking against a scenario nothing on
 * screen refers to any more. So History is rendered *beside* a dashboard that
 * stays mounted, and switching is a `hidden` class.
 *
 * That choice has one consequence, and it is MapLibre's: a canvas inside
 * `display:none` measures 0×0, and MapLibre sizes itself once and then listens
 * for *window* resize, which never fires here. `active` is the prop that tells
 * the map to re-measure when it is shown again — see `route-map.tsx`.
 *
 * ## Why not `<Activity>`
 *
 * React's [`<Activity>`](https://react.dev/reference/react/Activity) is the
 * platform's own answer to this — `preserving-ui-state.md` in the bundled Next
 * docs names tabs as its use case, and it hides with `display: none` exactly as
 * this does. It is not used here for one reason, and it is worth writing down
 * rather than rediscovering: **Activity runs effect cleanup functions while
 * hidden, as on unmount.** `route-map.tsx`'s map lifecycle effect cleans up by
 * calling `map.remove()`, so every trip to History would destroy the MapLibre
 * instance and every trip back would rebuild it — re-requesting basemap tiles
 * and re-framing the camera on a map whose whole job was to stay put. A plain
 * `hidden` class keeps the map alive, which is the point of keeping the
 * dashboard alive.
 *
 * ## Why History and Compare are conditional instead of hidden too
 *
 * Both are plain reads with no state worth keeping: no fleet to keep ticking, no
 * incident to hold open. So they mount when you look at them and refetch, which
 * makes every visit a fresh read — the thing you actually want from a history
 * view, and the only safe thing for a comparison — rather than a stale table left
 * over from ten minutes ago. Hiding them would buy nothing and cost that
 * freshness.
 *
 * ## Why the shell holds the scenario id
 *
 * `Compare` is the one view that needs something the dashboard owns, and the
 * dashboard cannot be asked for it — it keeps its scenario in local state behind
 * no props. Rather than lift the whole scenario up, or have the compare view
 * re-derive one (which would mean a second scenario, solved separately, and a
 * comparison of a different instance from the one on screen), the dashboard
 * reports its scenario id outward when it loads one. That is the only fact the
 * shell needs, and it is one the dashboard already had.
 */

import { useState } from "react";

import CompareView from "@/components/compare/compare-view";
import Dashboard from "@/components/dashboard/dashboard";
import Header, { type View } from "@/components/dashboard/header";
import HistoryView from "@/components/history/history-view";

export default function AppShell() {
  const [view, setView] = useState<View>("dashboard");
  /**
   * The scenario the dashboard currently has loaded, or `null` before it loads
   * one. Set by the dashboard as a side effect of loading, never as a source of
   * truth: `Dashboard` is still the owner, and this is a copy it pushes out.
   */
  const [scenarioId, setScenarioId] = useState<string | null>(null);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas">
      <Header view={view} onNavigate={setView} />

      {/* `hidden` rather than a conditional, and `flex flex-col` so the dashboard
          can return its row and its stat bar as two children of one column. The
          dashboard fills what the header leaves. */}
      <div
        className={view === "dashboard" ? "flex min-h-0 flex-1 flex-col" : "hidden"}
      >
        <Dashboard active={view === "dashboard"} onScenarioChange={setScenarioId} />
      </div>

      {view === "history" ? <HistoryView /> : null}

      {/* Keyed by the scenario so that loading a different one resets this view
          rather than leaving the previous instance's results on screen under a
          heading that now describes a different problem. React's own answer to
          "reset state when a prop changes" is a key, which is cheaper and less
          error-prone than an effect that clears four pieces of state in order. */}
      {view === "compare" ? <CompareView key={scenarioId ?? "none"} scenarioId={scenarioId} /> : null}
    </div>
  );
}
