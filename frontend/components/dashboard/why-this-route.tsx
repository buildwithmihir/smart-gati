"use client";

/**
 * The "why this route" layer: the sentences behind a re-plan, and their receipts.
 *
 * Two disclosures, at the two scopes the data actually has:
 *
 * * `PlanReasoning` — what happened to the network. A road's delay is a fact
 *   about the road, identical for every vehicle, so it is stated once. The
 *   convergence chart belongs here for the same reason: one series per solve.
 * * `RouteReasoning` — one vehicle's part in it, which is the only place a
 *   sentence can be per-vehicle, because a cost delta is the only quantity here
 *   that is.
 *
 * Both render the backend's sentences **verbatim**. Nothing on this side
 * composes prose from numbers — a UI that wrote its own sentences would be a
 * second place for the reasoning to be wrong, and the one nobody tests. The
 * figures are drawn underneath each sentence instead, which is the other half of
 * the contract: the sentence arrives with the numbers it was built from, so a
 * reader can check it without trusting either side.
 *
 * Absent data is stated, never faked. A first dispatch has no trace because it
 * has nothing to be different from, and a vehicle the re-plan did not touch has
 * no statements; both say so in words rather than rendering an empty box.
 */

import { useState, type ReactNode } from "react";
import { ChevronRight, Info, Sigma } from "lucide-react";

import ConvergenceChart from "@/components/dashboard/convergence-chart";
import { Button } from "@/components/ui/button";
import type {
  Explanation,
  ExplanationStatement,
  RouteExplanation,
} from "@/lib/api";
import { formatFigure } from "@/lib/format";
import { colorForVehicle } from "@/lib/vehicle-colors";

/**
 * A disclosure, built from the pieces already in the design system.
 *
 * There is no collapsible primitive in `components/ui/`, and adding one for two
 * callers would be more surface than the feature needs. What matters is that the
 * trigger is a real `button` with `aria-expanded` and `aria-controls`, and that
 * the body is `hidden` rather than unmounted-and-styled-away, so the disclosure
 * is navigable and announced correctly.
 */
function Disclosure({
  id,
  label,
  hint,
  children,
  defaultOpen = false,
}: {
  id: string;
  label: string;
  hint?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = `${id}-body`;

  return (
    <div className="rounded-xl ring-1 ring-border">
      <Button
        variant="ghost"
        size="xs"
        className="h-auto w-full justify-start gap-1.5 px-2.5 py-2 text-2xs font-medium"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((current) => !current)}
      >
        <ChevronRight
          className={`size-3 shrink-0 transition-transform duration-150 ${
            open ? "rotate-90" : ""
          }`}
          aria-hidden
        />
        <span>{label}</span>
        {hint ? (
          <span className="ml-auto truncate font-normal text-muted-foreground">{hint}</span>
        ) : null}
      </Button>

      <div id={bodyId} hidden={!open} className="space-y-2.5 px-2.5 pt-0.5 pb-2.5">
        {children}
      </div>
    </div>
  );
}

/**
 * One sentence, with the numbers it was built from laid out beneath it.
 *
 * The figures are a list rather than a sentence-tail so that a reader comparing
 * two statements is comparing aligned columns, and so an unrecognised unit shows
 * up as an odd row rather than as odd prose.
 */
function StatementBlock({ statement }: { statement: ExplanationStatement }) {
  return (
    <div className="rounded-lg bg-muted/60 px-2.5 py-2">
      <p className="text-xs leading-relaxed">{statement.text}</p>

      {statement.figures.length > 0 ? (
        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 border-t border-border/60 pt-1.5">
          {statement.figures.map((figure) => (
            <span
              key={`${figure.label}-${figure.unit}`}
              className="text-2xs text-muted-foreground"
            >
              {figure.label}{" "}
              <span className="font-medium text-foreground tabular-nums">
                {formatFigure(figure.value, figure.unit)}
              </span>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/**
 * The plan-level trace: what happened, and how the solve got there.
 *
 * Open by default only once there is something in it — an incident and a
 * re-plan — because a collapsed disclosure on a freshly loaded scenario would
 * hide the only thing on this panel that is not a number.
 */
export function PlanReasoning({
  explanation,
  convergence,
  solverName,
}: {
  explanation: Explanation | null;
  convergence: number[];
  solverName: string;
}) {
  const roads = explanation?.roads ?? [];

  return (
    <Disclosure
      id="plan-reasoning"
      label="Why this route? · whole plan"
      hint={roads.length > 0 ? `${roads.length} reported road(s)` : undefined}
      defaultOpen={roads.length > 0}
    >
      {explanation ? (
        <p className="text-xs leading-relaxed text-muted-foreground">
          {explanation.headline}
        </p>
      ) : (
        <p className="flex items-start gap-1.5 text-2xs text-muted-foreground">
          <Info className="mt-px size-3 shrink-0" aria-hidden />
          <span>
            First dispatch. A decision trace is built from the difference between two
            plans, and this fleet has nothing to be different from yet — report an
            incident and the reasoning appears here.
          </span>
        </p>
      )}

      {roads.flatMap((road) =>
        road.statements.map((statement, index) => (
          <StatementBlock
            key={`${road.u}-${road.v}-${index}`}
            statement={statement}
          />
        )),
      )}

      <div className="border-t border-border pt-2.5">
        <ConvergenceChart history={convergence} solverName={solverName} />
      </div>
    </Disclosure>
  );
}

/**
 * One vehicle's trace, under its stop list.
 *
 * The vehicle's colour is carried on the heading so the panel matches the
 * stop badges above it and the route on the map — the same decoding the rest of
 * this panel relies on.
 */
export function RouteReasoning({
  explanation,
}: {
  explanation: RouteExplanation | undefined;
}) {
  if (!explanation) {
    return (
      <Disclosure id="route-reasoning-empty" label="Why this route?">
        <p className="text-2xs text-muted-foreground">
          Nothing to compare against yet. This vehicle&apos;s reasoning appears once a
          re-plan has changed something it was responsible for.
        </p>
      </Disclosure>
    );
  }

  return (
    <Disclosure
      id={`route-reasoning-${explanation.vehicle_id}`}
      label="Why this route?"
      hint={explanation.headline}
    >
      <div className="flex items-center gap-1.5">
        <span
          className="size-2 rounded-full shrink-0"
          style={{ backgroundColor: colorForVehicle(explanation.vehicle_id) }}
          aria-hidden
        />
        <span className="text-2xs font-medium">{explanation.vehicle_id}</span>
        <span className="ml-auto flex items-center gap-1 text-2xs text-muted-foreground">
          <Sigma className="size-2.5" aria-hidden />
          {explanation.statements.length} statement
          {explanation.statements.length === 1 ? "" : "s"}
        </span>
      </div>

      {explanation.statements.map((statement, index) => (
        <StatementBlock key={index} statement={statement} />
      ))}
    </Disclosure>
  );
}
