"use client";

/**
 * Right panel: the Route Summary card.
 *
 * Every figure is read from the plan — nothing here is hardcoded, including the
 * solver name, which comes from `solver_name` ("QPSO" while QPSO remains the
 * backend's production default).
 *
 * The card shows the objective and its three parts separately, because a single
 * combined figure hides what the solver actually traded. "Cost" is the weighted
 * objective in rupees; time, distance and fuel are the raw quantities it was
 * priced from. Each is labelled with its own unit — a rupee figure shown as a
 * duration, or metres shown as rupees, would be a confident lie.
 *
 * It reads a `DashboardPlan` rather than the optimize response, because the
 * dashboard's plan can also come from dispatching a fleet or from a
 * re-optimization. Two fields are genuinely unknown for a dispatched plan —
 * `feasible` and `runtime_ms` — and the card shows them as unknown rather than
 * inferring them; see `DashboardPlan` in `lib/api`.
 */

import type { ReactNode } from "react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { DashboardPlan } from "@/lib/api";
import {
  formatDistance,
  formatLitres,
  formatRupees,
  formatSeconds,
} from "@/lib/format";
import { colorForVehicle } from "@/lib/vehicle-colors";

type RouteSummaryProps = {
  result: DashboardPlan | null;
  loading: boolean;
  /**
   * What the figures are a total *of*, shown under the title while it is not
   * the obvious answer.
   *
   * A re-planned plan covers only the stops the fleet has left, so its travel
   * time is legitimately smaller than the dispatched plan's was. Without a line
   * saying so, a number that drops the moment an incident is injected reads as
   * the trip having got shorter — which is the opposite of what happened.
   */
  caption?: string;
  /**
   * How many vehicles the scenario has, when that is not the same as how many
   * routes the plan on screen carries.
   *
   * A re-optimization answers only for the vehicles that had work left, so
   * without this the denominator would shrink partway through the demo and
   * "one vehicle has finished" would read as "there were only ever two".
   */
  fleetSize?: number;
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

export default function RouteSummary({ result, loading, caption, fleetSize }: RouteSummaryProps) {
  const activeRoutes = (result?.routes ?? []).filter((route) => route.stops.length > 0);
  const totalStops = activeRoutes.reduce((sum, route) => sum + route.stops.length, 0);

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Route Summary</CardTitle>
        {caption ? <CardDescription className="text-2xs">{caption}</CardDescription> : null}
      </CardHeader>
      <CardContent className="px-(--card-spacing)">
        {loading && !result ? (
          <div className="divide-y divide-border py-1">
            {Array.from({ length: 6 }).map((_, index) => (
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
                value={`${activeRoutes.length} / ${fleetSize ?? result.routes.length}`}
              />
              <Row label="Travel time" value={formatSeconds(result.travel_time)} />
              <Row label="Distance" value={formatDistance(result.distance_m)} />
              <Row label="Fuel" value={formatLitres(result.fuel_litres)} />
              <Row label="Cost" value={formatRupees(result.travel_cost)} />
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
                      {route.stops.length} stops · {formatDistance(route.distance_m)}
                    </span>
                    <span className="text-xs tabular-nums">
                      {formatSeconds(route.travel_time)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-3 flex items-center justify-between border-t border-border pt-3 text-2xs text-muted-foreground">
              <span>
                {result.runtime_ms !== null ? (
                  <>
                    Solved in{" "}
                    <span className="tabular-nums">{Math.round(result.runtime_ms)} ms</span>
                  </>
                ) : (
                  // A dispatched plan's solve is not timed anywhere that reaches
                  // the client, so this says where the plan came from instead of
                  // printing a duration it does not have.
                  "Dispatched with the fleet"
                )}
                {result.iterations !== null ? (
                  <>
                    {" · "}
                    <span className="tabular-nums">
                      {result.iterations} iters × {result.population}
                    </span>
                  </>
                ) : null}
              </span>
              {result.feasible === null ? (
                <span
                  className="text-muted-foreground"
                  title="The dispatch reports the routes it put on the road, not a feasibility check over them."
                >
                  —
                </span>
              ) : (
                <span
                  className={
                    result.feasible ? "font-medium text-ok" : "font-medium text-danger"
                  }
                >
                  {result.feasible ? "Feasible" : "Infeasible"}
                </span>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
