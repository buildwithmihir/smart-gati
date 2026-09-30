"use client";

import { Route } from "lucide-react";

/**
 * The dark header, and the app's only navigation.
 *
 * Three of the four items lead somewhere and one does not, and the difference is
 * carried by the *element* rather than by styling: a destination is a `<button>`
 * and a view that does not exist yet is a `<span aria-disabled>`. Making the
 * latter clickable-but-inert would be the worse lie — a button that does nothing
 * reads as broken, where a disabled span reads as not-yet.
 *
 * `aria-current="page"` marks the active view, so the highlight is announced
 * rather than being a colour a screen reader cannot see.
 *
 * The two white opacities are step values on the ink, not free choices: /45 for
 * the strapline sat at 3.6:1, under the 4.5:1 bar for body-sized text, so it is
 * /60. The inactive links stay quieter than the active one at /55 — still
 * clearly out of play, but legible enough to be read rather than guessed at.
 */

/**
 * The views this app has. `History` is the run history (`history-view.tsx`);
 * `Compare` is the five-solver comparison (`compare-view.tsx`).
 */
export type View = "dashboard" | "history" | "compare";

type HeaderProps = {
  view: View;
  onNavigate: (view: View) => void;
};

/**
 * The nav, in order, with the one item that is not written yet marked as such.
 *
 * `view` is `null` for that one, which is what the render branches on — there is
 * no separate "disabled" flag to keep in step with it.
 *
 * **There used to be a second disabled item here, "Benchmarks", and it is gone
 * rather than promoted.** The offline runner (`benchmarks/run_comparison.py`) is
 * a CLI script with no HTTP surface, so a nav item for it could only ever have
 * been a disabled span — and a permanently-dead tab sitting beside a working
 * "Compare" that answers the same question is worse than no tab at all. The
 * script is still there and still the way to sweep many instances; it just is not
 * a page in this app, because it is not one.
 */
const NAV_ITEMS: { label: string; view: View | null }[] = [
  { label: "Dashboard", view: "dashboard" },
  { label: "History", view: "history" },
  { label: "Compare", view: "compare" },
  { label: "Scenarios", view: null },
];

export default function Header({ view, onNavigate }: HeaderProps) {
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
        {NAV_ITEMS.map((item) => {
          if (item.view === null) {
            return (
              <span
                key={item.label}
                aria-disabled
                title="Available in a later phase"
                className="cursor-not-allowed rounded-md px-3 py-1.5 text-sm text-white/55"
              >
                {item.label}
              </span>
            );
          }

          const active = item.view === view;
          return (
            <button
              key={item.label}
              type="button"
              onClick={() => onNavigate(item.view as View)}
              aria-current={active ? "page" : undefined}
              className={
                active
                  ? "rounded-md bg-white/10 px-3 py-1.5 text-sm font-medium text-white"
                  : "rounded-md px-3 py-1.5 text-sm text-white/55 transition-colors hover:bg-white/5 hover:text-white/80"
              }
            >
              {item.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
