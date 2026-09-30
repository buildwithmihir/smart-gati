"use client";

/**
 * Left panel: the stop list, grouped by vehicle.
 *
 * Grouping is what makes the per-vehicle colours legible — a flat list would
 * show twelve coloured badges with nothing to decode them against. The number
 * on each badge is the stop's position within *its own vehicle's* route, which
 * is the same number drawn on the map marker, so a stop can be found in either
 * place from the other.
 *
 * The panel sits on the canvas tone rather than on white, so the stop cards
 * read as cards. On a white panel they were separated from it by their ring
 * alone, which is the weakest signal in the system and the first thing to
 * disappear on a projector.
 */

import { Loader2, PackageOpen, Play } from "lucide-react";

import { PlanReasoning, RouteReasoning } from "@/components/dashboard/why-this-route";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import type { Explanation, Route, ScenarioResponse } from "@/lib/api";
import { colorForVehicle, tintForVehicle } from "@/lib/vehicle-colors";

/** How many placeholders to draw while the first solve runs. */
const SKELETON_STOPS = 6;

type StopsPanelProps = {
  scenario: ScenarioResponse | null;
  routes: Route[];
  loading: boolean;
  onLoadSample: () => void;
  /**
   * The decision trace for the plan on screen, or `null` for a first dispatch.
   *
   * Optional so the panel still renders standalone; a caller that has no trace
   * to give gets the same "nothing to compare against yet" wording a dispatch
   * gets, rather than a panel that looks broken.
   */
  explanation?: Explanation | null;
  /** Best-so-far cost per iteration for the solve behind these routes. */
  convergence?: number[];
  /** Named on the convergence chart. */
  solverName?: string;
};

/**
 * A placeholder shaped like a stop card — badge, title, subtitle, demand chip —
 * rather than a uniform grey bar.
 *
 * The anatomy is the point: a skeleton that matches the card it becomes means
 * the list resolves in place instead of reflowing when the data lands. The
 * stagger is what keeps six of them from pulsing in lockstep, which reads as a
 * wave across the list rather than a page-wide blink.
 */
function StopCardSkeleton({ index }: { index: number }) {
  return (
    <div
      className="flex animate-pulse items-center gap-3 rounded-xl bg-card px-3 py-3 ring-1 ring-foreground/10"
      style={{ animationDelay: `${index * 90}ms` }}
      aria-hidden
    >
      <Skeleton className="size-7 shrink-0 rounded-full" />
      <div className="min-w-0 flex-1 space-y-1.5">
        <Skeleton className="h-3.5 w-24 rounded-sm" />
        <Skeleton className="h-2.5 w-16 rounded-sm" />
      </div>
      <Skeleton className="h-5 w-9 shrink-0 rounded-full" />
    </div>
  );
}

export default function StopsPanel({
  scenario,
  routes,
  loading,
  onLoadSample,
  explanation = null,
  convergence = [],
  solverName = "",
}: StopsPanelProps) {
  const activeRoutes = routes.filter((route) => route.stops.length > 0);

  const demandById = new Map(
    (scenario?.deliveries ?? []).map((delivery) => [delivery.id, delivery.demand]),
  );

  // Indexed by vehicle so a section can find its own trace without scanning the
  // list per vehicle, and so a vehicle the re-plan did not touch is simply
  // absent — which `RouteReasoning` renders as "nothing to compare against",
  // the truthful answer, rather than as the nearest other vehicle's reasoning.
  const reasonsByVehicle = new Map(
    (explanation?.routes ?? []).map((route) => [route.vehicle_id, route]),
  );

  return (
    <aside className="flex w-panel shrink-0 flex-col border-r border-border bg-canvas">
      <div className="border-b border-border p-4">
        <Button onClick={onLoadSample} disabled={loading} className="w-full" size="lg">
          {loading ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <Play className="size-4" aria-hidden />
          )}
          {loading ? "Optimizing…" : "Load Sample Scenario"}
        </Button>
        <p className="mt-2 text-xs text-muted-foreground">
          {scenario
            ? `Scenario ${scenario.scenario_id.slice(0, 8)} · ${scenario.n_deliveries} deliveries · ${scenario.n_vehicles} vehicles`
            : "Generates a 12-delivery, 3-vehicle instance on the real Delhi graph."}
        </p>
      </div>

      {/* Vehicle legend — the key that decodes every colour in the UI. */}
      {activeRoutes.length > 0 ? (
        <div className="flex flex-wrap gap-x-4 gap-y-2 border-b border-border px-4 py-3">
          {activeRoutes.map((route) => (
            <div key={route.vehicle_id} className="flex items-center gap-2">
              <span
                className="size-2.5 rounded-full"
                style={{ backgroundColor: colorForVehicle(route.vehicle_id) }}
                aria-hidden
              />
              <span className="text-xs font-medium">{route.vehicle_id}</span>
              <span className="text-xs text-muted-foreground tabular-nums">
                {route.stops.length}
              </span>
            </div>
          ))}
        </div>
      ) : null}

      <ScrollArea className="min-h-0 flex-1">
        <div className="p-4">
          {loading && activeRoutes.length === 0 ? (
            <div className="space-y-2">
              {Array.from({ length: SKELETON_STOPS }).map((_, index) => (
                <StopCardSkeleton key={index} index={index} />
              ))}
            </div>
          ) : activeRoutes.length === 0 ? (
            /* Deliberately not a second call to action. The hero behind the map
               already carries the invitation and the button, and the panel's own
               header carries it a third time — so this says what the panel will
               hold rather than repeating what to press. */
            <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border px-4 py-10 text-center">
              <PackageOpen className="size-7 text-muted-foreground" aria-hidden />
              <p className="text-sm font-medium">No stops yet</p>
              <p className="text-xs text-muted-foreground">
                Solved stops appear here, grouped by vehicle.
              </p>
            </div>
          ) : (
            <div className="space-y-5">
              {/* Plan-level first: what happened to the network is the context
                  every vehicle's sentences below are read against. */}
              <PlanReasoning
                explanation={explanation}
                convergence={convergence}
                solverName={solverName}
              />

              {activeRoutes.map((route) => (
                <section key={route.vehicle_id}>
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className="size-2.5 rounded-full"
                        style={{ backgroundColor: colorForVehicle(route.vehicle_id) }}
                        aria-hidden
                      />
                      <span className="text-xs font-semibold tracking-wide uppercase">
                        {route.vehicle_id}
                      </span>
                    </div>
                    <span className="text-2xs text-muted-foreground tabular-nums">
                      {route.stops.length} stops · load {Math.round(route.load)}/
                      {Math.round(route.capacity)}
                    </span>
                  </div>

                  <div className="space-y-2">
                    {route.stops.map((stop, index) => (
                      <Card
                        key={`${route.vehicle_id}-${stop.delivery_id}`}
                        size="sm"
                        className="shadow-sm"
                      >
                        <CardContent className="flex items-center gap-3 px-3">
                          {/* The number badge: colour = vehicle, number = position in that route. */}
                          <span
                            className="flex size-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold text-white tabular-nums"
                            style={{ backgroundColor: colorForVehicle(route.vehicle_id) }}
                          >
                            {index + 1}
                          </span>
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-sm font-medium">{stop.delivery_id}</div>
                            <div className="truncate font-mono text-2xs text-muted-foreground">
                              node {stop.node}
                            </div>
                          </div>
                          {demandById.has(stop.delivery_id) ? (
                            <Badge
                              variant="outline"
                              style={{
                                backgroundColor: tintForVehicle(route.vehicle_id),
                                borderColor: colorForVehicle(route.vehicle_id),
                              }}
                              className="shrink-0 tabular-nums"
                            >
                              {demandById.get(stop.delivery_id)}u
                            </Badge>
                          ) : null}
                        </CardContent>
                      </Card>
                    ))}
                  </div>

                  <div className="mt-2">
                    <RouteReasoning
                      explanation={reasonsByVehicle.get(route.vehicle_id)}
                    />
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </aside>
  );
}
