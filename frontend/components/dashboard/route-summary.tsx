"use client";

/**
 * Right panel: the Route Summary card.
 *
 * Every figure is read from the optimize response — nothing here is hardcoded,
 * including the solver name, which comes from `solver_name` ("QPSO" while QPSO
 * remains the backend's production default).
 *
 * The cost figure is travel **time in seconds**, not distance: the graph's
 * `weight` mirrors `travel_time`. It is labelled as time rather than dressed up
 * as a distance, because calling seconds "km" would be a confident lie.
 */

import type { ReactNode } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { OptimizeResponse } from "@/lib/api";
import { colorForVehicle } from "@/lib/vehicle-colors";

type RouteSummaryProps = {
  result: OptimizeResponse | null;
  loading: boolean;
};

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm font-medium tabular-nums">{value}</span>
    </div>
  );
}

/**
 * A placeholder shaped like a `Row` — label left, figure right — so the card
 * resolves in place rather than reflowing. Staggered for the same reason the
 * stop list is: six bars pulsing in lockstep reads as a blink, a wave reads as
 * loading.
 */
function RowSkeleton({ index }: { index: number }) {
  return (
    <div
      className="flex animate-pulse items-center justify-between py-1.5"
      style={{ animationDelay: `${index * 90}ms` }}
      aria-hidden
    >
      <Skeleton className="h-3 w-24 rounded-sm" />
      <Skeleton className="h-3.5 w-12 rounded-sm" />
    </div>
  );
}

/** 1768.7s -> "29m 28.7s"; keeps the seconds legible at a glance. */
function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}m ${rest.toFixed(1)}s`;
}

export default function RouteSummary({ result, loading }: RouteSummaryProps) {
  const activeRoutes = (result?.routes ?? []).filter((route) => route.stops.length > 0);
  const totalStops = activeRoutes.reduce((sum, route) => sum + route.stops.length, 0);

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Route Summary</CardTitle>
      </CardHeader>
      <CardContent className="px-(--card-spacing)">
        {loading && !result ? (
          <div className="divide-y divide-border py-1">
            {Array.from({ length: 4 }).map((_, index) => (
              <RowSkeleton key={index} index={index} />
            ))}
          </div>
        ) : !result ? (
          <p className="py-2 text-xs text-muted-foreground">
            No solution yet — figures appear once a scenario is solved.
          </p>
        ) : (
          <>
            <div className="divide-y divide-border">
              <Row label="Total stops" value={totalStops} />
              <Row
                label="Vehicles used"
                value={`${activeRoutes.length} / ${result.routes.length}`}
              />
              <Row label="Total travel time" value={formatSeconds(result.travel_cost)} />
              <Row
                label="Solver"
                value={
                  <span className="rounded-md bg-ink px-2 py-0.5 text-xs font-semibold text-white">
                    {result.solver_name}
                  </span>
                }
              />
            </div>

            <div className="mt-3 border-t border-border pt-3">
              <div className="mb-2 text-2xs font-medium tracking-wide text-muted-foreground uppercase">
                Per vehicle
              </div>
              <div className="space-y-1.5">
                {activeRoutes.map((route) => (
                  <div key={route.vehicle_id} className="flex items-center gap-2">
                    <span
                      className="size-2.5 shrink-0 rounded-full"
                      style={{ backgroundColor: colorForVehicle(route.vehicle_id) }}
                      aria-hidden
                    />
                    <span className="w-8 text-xs font-medium">{route.vehicle_id}</span>
                    <span className="flex-1 text-xs text-muted-foreground tabular-nums">
                      {route.stops.length} stops
                    </span>
                    <span className="text-xs tabular-nums">
                      {formatSeconds(route.travel_cost)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-3 flex items-center justify-between border-t border-border pt-3 text-2xs text-muted-foreground">
              <span>
                Solved in <span className="tabular-nums">{Math.round(result.runtime_ms)} ms</span>
                {result.iterations ? (
                  <>
                    {" · "}
                    <span className="tabular-nums">
                      {result.iterations} iters × {result.population}
                    </span>
                  </>
                ) : null}
              </span>
              <span
                className={
                  result.feasible ? "font-medium text-ok" : "font-medium text-danger"
                }
              >
                {result.feasible ? "Feasible" : "Infeasible"}
              </span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
