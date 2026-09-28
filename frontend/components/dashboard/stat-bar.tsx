"use client";

import type { ReactNode } from "react";
import { Route } from "lucide-react";

/**
 * A single live figure. Unit is spelled out rather than implied — the cost
 * numbers are travel time in seconds, so "s" is not decoration.
 */
function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-baseline gap-2 px-4">
      <span className="text-2xs tracking-wide text-muted-foreground uppercase">{label}</span>
      <span className="text-sm font-semibold tabular-nums">{value}</span>
    </div>
  );
}

function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${(seconds - minutes * 60).toFixed(1)}s`;
}

export default function StatBar({
  stops,
  vehiclesUsed,
  vehiclesTotal,
  travelTime,
  solverName,
  loading,
}: {
  stops: number | null;
  vehiclesUsed: number | null;
  vehiclesTotal: number | null;
  travelTime: number | null;
  solverName: string | null;
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
        <span>{loading ? "Solving…" : "FastAPI backend · real Delhi road graph"}</span>
      </div>
    </footer>
  );
}
