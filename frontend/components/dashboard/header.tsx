"use client";

import { Route } from "lucide-react";

/**
 * The dark header. The nav links are deliberately inert: they mark where the
 * other views will live rather than pretending to route somewhere. They are
 * `aria-disabled` and carry a title explaining why, so nothing here looks
 * clickable and then does nothing.
 *
 * The two white opacities are step values on the ink, not free choices: /45 for
 * the strapline sat at 3.6:1, under the 4.5:1 bar for body-sized text, so it is
 * /60. The inactive links stay quieter than the active one at /55 — still
 * clearly out of play, but legible enough to be read rather than guessed at.
 */
const NAV_ITEMS = [
  { label: "Dashboard", active: true },
  { label: "Scenarios", active: false },
  { label: "Benchmarks", active: false },
] as const;

export default function Header() {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between bg-ink px-5">
      <div className="flex items-center gap-3">
        <span className="flex size-7 items-center justify-center rounded-md bg-white/10 text-white">
          <Route className="size-4" aria-hidden />
        </span>
        <span className="font-heading text-base font-semibold tracking-tight text-white">
          Q-Gati
        </span>
        <span className="hidden text-xs text-white/60 sm:inline">Vehicle routing for Delhi</span>
      </div>

      <nav className="flex items-center gap-1">
        {NAV_ITEMS.map((item) => (
          <span
            key={item.label}
            aria-disabled={!item.active}
            title={item.active ? undefined : "Available in a later phase"}
            className={
              item.active
                ? "rounded-md bg-white/10 px-3 py-1.5 text-sm font-medium text-white"
                : "cursor-not-allowed rounded-md px-3 py-1.5 text-sm text-white/55"
            }
          >
            {item.label}
          </span>
        ))}
      </nav>
    </header>
  );
}
