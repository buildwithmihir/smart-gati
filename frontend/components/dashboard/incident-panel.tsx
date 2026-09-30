"use client";

/**
 * Right panel, above the summary: the incident controls, and the re-plan that
 * answers them.
 *
 * This is the only place in the app where the user changes the world rather
 * than reading it, so most of what is here is about saying back what happened.
 * Two things in particular, both of which are easy to get wrong in a way that
 * makes the numbers lie:
 *
 * 1. **Both figures in the re-plan table are priced with the incident already
 *    in place, and both cover only the stops that are still unserved.** So the
 *    difference between them is what *re-planning* recovered — it is not what
 *    the incident cost, and it is not a change in the whole job. Read either
 *    way the number is wrong, and neither reading is unreasonable, so the card
 *    says which one it is instead of hoping.
 * 2. **A refusal is an answer.** `POST /reoptimize` answers 409 when nothing
 *    has happened that warrants one, and `changed_legs: 0` says the road that
 *    was clicked is on no route. Both are reported as findings rather than as
 *    failures, because both are things an operator wants to know.
 *
 * The panel does not own any of the state it draws — the dashboard does, since
 * it also has to drive the map. Everything here is props down, events up.
 */

import type { ReactNode } from "react";
import { Ban, Clock, Gauge, Info, Loader2, MapPin, TriangleAlert, Undo2, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  planSeconds,
  routesChanged,
  type IncidentKind,
  type IncidentOut,
  type ReoptimizeResponse,
} from "@/lib/api";
import { formatCount, formatSeconds, formatSignedSeconds } from "@/lib/format";
import { describeRoad } from "@/lib/road-picking";

/** How each kind reads on a button and in a sentence. */
const KIND_LABEL: Record<IncidentKind, string> = {
  closure: "Road Closure",
  slow: "Slow / Accident",
};

/**
 * The live report, as the panel draws it.
 *
 * `incidents` is a list because a street is two directed edges on this graph —
 * closing one carriageway of a two-way road would leave it open in the
 * direction nobody checked. See `bothDirectionsOf` for how a click becomes the
 * one or two of them.
 */
export type IncidentState = {
  kind: IncidentKind;
  /** The edge that was clicked; the street is named by it in both directions. */
  u: number;
  v: number;
  incidents: IncidentOut[];
  /** Objective-matrix entries the report moved. Zero is a legitimate answer. */
  changedLegs: number;
};

type IncidentPanelProps = {
  /** The kind being armed, or null when the map is not waiting for a click. */
  arming: IncidentKind | null;
  incident: IncidentState | null;
  /** The re-plan that followed the incident, if one was produced. */
  report: ReoptimizeResponse | null;
  /** True while an incident request or a re-plan is in flight. */
  busy: boolean;
  notice: PanelNotice | null;
  onArm: (kind: IncidentKind) => void;
  onCancelArm: () => void;
  onClear: () => void;
};

/**
 * Something the backend said back, and which of two very different things it
 * was.
 *
 * The distinction is not cosmetic. **409 from `/reoptimize` means nothing has
 * happened that warrants re-planning** — the system looked, found the plan
 * still correct, and declined. That is the endpoint working, and painting it
 * red would tell an operator something failed when nothing did. A 422, where
 * the change itself was refused and no plan exists, is a failure. Both carry
 * FastAPI's `detail`, which is already a sentence written for a person; only
 * the framing differs, so only the framing is carried here.
 */
export type PanelNotice = {
  detail: string;
  tone: "refusal" | "failure";
};

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm font-medium tabular-nums">{value}</span>
    </div>
  );
}

export default function IncidentPanel({
  arming,
  incident,
  report,
  busy,
  notice,
  onArm,
  onCancelArm,
  onClear,
}: IncidentPanelProps) {
  return (
    <>
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Incident Control</CardTitle>
          {incident ? (
            <CardAction>
              <Badge variant={incident.kind === "closure" ? "destructive" : "outline"}>
                {KIND_LABEL[incident.kind]}
              </Badge>
            </CardAction>
          ) : null}
        </CardHeader>

        <CardContent className="px-(--card-spacing)">
          {incident && incident.incidents.length === 0 ? (
            // The click has landed and the highlight is already on the map, but
            // the backend has not agreed to the report yet. Saying "0
            // carriageways, 0 legs" here would be reporting as a fact something
            // that is still a request.
            <p className="flex items-center gap-2 py-1 text-xs text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" aria-hidden />
              Reporting {describeRoad(incident.u, incident.v)}…
            </p>
          ) : incident ? (
            <>
              <div className="divide-y divide-border">
                <Row
                  label="Street"
                  value={
                    <span className="font-mono text-xs">
                      {describeRoad(incident.u, incident.v)}
                    </span>
                  }
                />
                <Row
                  label="Reported"
                  value={
                    incident.incidents.length === 1
                      ? "1 carriageway"
                      : `${formatCount(incident.incidents.length)} carriageways`
                  }
                />
                <Row label="Legs re-priced" value={formatCount(incident.changedLegs)} />
              </div>

              <p className="mt-2 text-2xs text-muted-foreground">
                {incident.incidents.length === 1
                  ? "Reported in this direction only — this road is one-way on the graph."
                  : "Reported in both directions: the graph carries each carriageway separately, and one of them alone would leave the street half open."}
              </p>

              {incident.changedLegs === 0 ? (
                <p className="mt-2 flex items-start gap-1.5 rounded-lg bg-muted px-2.5 py-2 text-2xs text-muted-foreground">
                  <TriangleAlert className="mt-px size-3 shrink-0" aria-hidden />
                  <span>
                    No route in the current solution uses this road, so nothing was re-priced
                    and nothing will move. Try a street one of the drawn routes runs along.
                  </span>
                </p>
              ) : null}

              <Button
                variant="outline"
                size="lg"
                className="mt-3 w-full"
                disabled={busy}
                onClick={onClear}
              >
                <Undo2 aria-hidden />
                Clear Incident
              </Button>
            </>
          ) : arming ? (
            <div className="flex items-center gap-2 rounded-lg border border-dashed border-ink/30 bg-ink/5 px-2.5 py-2">
              <MapPin className="size-3.5 shrink-0 text-ink" aria-hidden />
              <p className="flex-1 text-xs">
                Click a road on the map to report it as{" "}
                <span className="font-medium">{KIND_LABEL[arming].toLowerCase()}</span>.
              </p>
              <Button variant="ghost" size="xs" onClick={onCancelArm}>
                <X aria-hidden />
                Cancel
              </Button>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2">
                <Button variant="destructive" size="lg" disabled={busy} onClick={() => onArm("closure")}>
                  <Ban aria-hidden />
                  Road Closure
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  className="text-warn"
                  disabled={busy}
                  onClick={() => onArm("slow")}
                >
                  <Gauge aria-hidden />
                  Slow / Accident
                </Button>
              </div>
              <p className="mt-2 text-2xs text-muted-foreground">
                Report a road, then watch the fleet be re-planned around it. One incident at a
                time — clear it to report another.
              </p>
            </>
          )}
        </CardContent>
      </Card>

      {notice ? (
        notice.tone === "failure" ? (
          <div className="flex items-start gap-2 rounded-xl bg-danger/10 px-3 py-2.5 text-xs text-danger ring-1 ring-danger/20">
            <TriangleAlert className="mt-px size-3.5 shrink-0" aria-hidden />
            <span className="break-words">{notice.detail}</span>
          </div>
        ) : (
          <div className="flex items-start gap-2 rounded-xl bg-muted px-3 py-2.5 text-xs text-muted-foreground ring-1 ring-border">
            <Info className="mt-px size-3.5 shrink-0" aria-hidden />
            <span className="break-words">{notice.detail}</span>
          </div>
        )
      ) : null}

      {incident ? (
        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle>Re-plan</CardTitle>
            {report ? (
              <CardAction>
                <Badge variant="outline">{report.solver_name}</Badge>
              </CardAction>
            ) : null}
          </CardHeader>

          <CardContent className="px-(--card-spacing)">
            {busy && !report ? (
              <p className="flex items-center gap-2 py-1 text-xs text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin" aria-hidden />
                Re-planning the remaining stops…
              </p>
            ) : !report ? (
              <p className="py-1 text-xs text-muted-foreground">
                No re-plan was produced — see the message above.
              </p>
            ) : (
              <ReplanBody report={report} />
            )}
          </CardContent>
        </Card>
      ) : null}
    </>
  );
}

/**
 * The before/after, and the caveat that makes it readable.
 *
 * Split out only so the two branches above stay readable as a whole; it holds
 * no state of its own.
 */
function ReplanBody({ report }: { report: ReoptimizeResponse }) {
  const before = planSeconds(report.before);
  const after = planSeconds(report.after);
  const delta = after - before;
  const changed = routesChanged(report.before, report.after);

  return (
    <>
      <p className="mb-2 text-xs text-muted-foreground">{report.trigger.detail}</p>

      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        {report.trigger.kinds.map((kind) => (
          <Badge key={kind} variant="secondary">
            {kind}
          </Badge>
        ))}
      </div>

      <div className="divide-y divide-border">
        <Row
          label="Before"
          value={
            <span className="flex items-center gap-1.5">
              <Clock className="size-3 text-muted-foreground" aria-hidden />
              {formatSeconds(before)}
            </span>
          }
        />
        <Row
          label="After"
          value={
            <span className="flex items-center gap-1.5">
              <Clock className="size-3 text-muted-foreground" aria-hidden />
              {formatSeconds(after)}
            </span>
          }
        />
        <Row
          label="Difference"
          value={
            <span className={delta < 0 ? "text-ok" : delta > 0 ? "text-danger" : undefined}>
              {formatSignedSeconds(delta)}
            </span>
          }
        />
        <Row
          label="Route changed"
          value={
            changed ? (
              <span className="text-ok">yes</span>
            ) : (
              <span className="text-muted-foreground">no</span>
            )
          }
        />
        {changed ? (
          <Row
            label="Stops re-assigned"
            value={`${formatCount(report.moved.length)} of ${formatCount(report.replanned.length)}`}
          />
        ) : null}
        <Row label="Unserved stops" value={formatCount(report.replanned.length)} />
      </div>

      {!changed ? (
        <p className="mt-2 text-2xs text-muted-foreground">
          The re-plan was handed every vehicle's real position and remaining work, and kept the
          assignment it found. That is a result, not a failure: under these conditions the fleet
          is already on the cheapest routes left to it.
        </p>
      ) : null}

      <div className="mt-3 border-t border-border pt-3 text-2xs text-muted-foreground">
        <p>
          Both figures price the same {formatCount(report.replanned.length)} unserved stops with
          the incident already in place — so the difference is what re-planning recovered, not
          what the incident cost. The incident's delay is inside both numbers.
        </p>
        <p className="mt-1.5">
          Drawn, not dispatched: the fleet keeps driving the routes it was sent on.
        </p>
      </div>
    </>
  );
}
