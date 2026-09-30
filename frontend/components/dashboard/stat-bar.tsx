"use client";

import type { ReactNode } from "react";
import { Route } from "lucide-react";

import { formatRupees, formatSeconds } from "@/lib/format";

/**
 * A single live figure. Units are spelled out rather than implied, because the
 * two figures that used to be one number are now different quantities: travel
 * time is seconds, and the cost beside it is the weighted objective in rupees.
 * Showing either without its unit would invite reading one as the other.
 */
function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-baseline gap-2 px-4">
      <span className="text-2xs tracking-wide text-muted-foreground uppercase">{label}</span>
      <span className="text-sm font-semibold tabular-nums">{value}</span>
    </div>
  );
}

export default function StatBar({
  stops,
  vehiclesUsed,
  vehiclesTotal,
  travelTime,
  cost,
  solverName,
  note,
  loading,
}: {
  stops: number | null;
  vehiclesUsed: number | null;
  vehiclesTotal: number | null;
  travelTime: number | null;
  cost: number | null;
  solverName: string | null;
  /**
   * Replaces the standing footer note while the figures on this bar are not the
   * whole job.
   *
   * The bar sits at the bottom of the screen and reads as a running total for
   * the scenario. When a re-plan is drawn it is a total for what is *left*, and
   * the travel time drops accordingly — so it has to say so, here, next to the
   * number rather than somewhere up the page.
   */
  note?: string | null;
  loading: boolean;
}) {
  const placeholder = <span className="text-muted-foreground">—</span>;

  return (
    <footer className="flex h-11 shrink-0 items-center border-t border-border bg-white">
      <Stat label="Stops" value={stops ?? placeholder} />
      <span className="h-4 w-px bg-border" aria-hidden />
      <Stat
        label="Vehicles"
        value={
          vehiclesUsed !== null && vehiclesTotal !== null
            ? `${vehiclesUsed} / ${vehiclesTotal}`
            : placeholder
        }
      />
      <span className="h-4 w-px bg-border" aria-hidden />
      <Stat label="Travel time" value={travelTime !== null ? formatSeconds(travelTime) : placeholder} />
      <span className="h-4 w-px bg-border" aria-hidden />
      <Stat label="Cost" value={cost !== null ? formatRupees(cost) : placeholder} />
      <span className="h-4 w-px bg-border" aria-hidden />
      <Stat
        label="Solver"
        value={
          solverName ? (
            <span className="rounded bg-ink px-1.5 py-0.5 text-xs font-semibold text-white">
              {solverName}
            </span>
          ) : (
            placeholder
          )
        }
      />

      <div className="ml-auto flex items-center gap-2 pr-4 text-2xs text-muted-foreground">
        <Route className="size-3.5" aria-hidden />
        <span>{loading ? "Solving…" : (note ?? "FastAPI backend · real Delhi road graph")}</span>
      </div>
    </footer>
  );
}
