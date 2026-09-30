/**
 * Number formatting for the dashboard's figures.
 *
 * These lived as private copies in `route-summary.tsx` and `stat-bar.tsx` — the
 * same four functions, written twice. The incident panel needs all of them a
 * third time, and a duration that renders as "29m 28.7s" in one card and
 * something else in another is the kind of drift nobody notices until a
 * projector does. So they live here and all three import them.
 *
 * The unit is always spelled out. Two of the figures this dashboard shows used
 * to be one number — `travel_cost` was seconds before distance and fuel joined
 * the objective, and is rupees now — so a bare numeral next to a label is a
 * question rather than an answer. Every formatter here attaches its unit.
 */

/** 1768.7 -> "29m 28.7s"; under a minute, "48.2 s". */
export function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${(seconds - minutes * 60).toFixed(1)}s`;
}

/**
 * A *change* in seconds, with its sign.
 *
 * Separate from `formatSeconds` because a delta has to keep a leading `+`: a
 * saving and a loss that both render as "12.0 s" are indistinguishable, and the
 * whole point of the before/after panel is which direction the number went. A
 * negative value carries its own minus, so only the positive branch adds one.
 */
export function formatSignedSeconds(seconds: number): string {
  const sign = seconds > 0 ? "+" : seconds < 0 ? "−" : "";
  return `${sign}${formatSeconds(Math.abs(seconds))}`;
}

/** The objective is a price, so it carries a currency mark and no unit suffix. */
export function formatRupees(rupees: number): string {
  return `₹${rupees.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

/** 812 -> "812 m"; past a kilometre, "4.31 km". */
export function formatDistance(metres: number): string {
  if (metres < 1000) return `${metres.toFixed(0)} m`;
  return `${(metres / 1000).toFixed(2)} km`;
}

export function formatLitres(litres: number): string {
  return `${litres.toFixed(2)} L`;
}

/** A 1-based position, for "stop 3 of 7". */
export function formatCount(value: number): string {
  return value.toLocaleString("en-IN");
}

/** Month names, so the output does not depend on the runtime's locale data. */
const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
] as const;

/**
 * A run's recorded timestamp, as a local date and clock time.
 *
 * The history stores ISO-8601 **UTC** and the reader is looking at their own
 * clock, so the conversion is the point rather than an accident — a run at
 * 06:23 UTC is 11:53 in Delhi, and showing the stored string would be off by
 * five and a half hours in the one place a reader is most likely to check it
 * against their watch.
 *
 * Built from the `Date` getters rather than `toLocaleString`, so the output does
 * not move with the runtime's locale data. An unparseable value comes back
 * verbatim: a bad row should look wrong, not look like today.
 */
export function formatRunTimestamp(iso: string): string {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return iso;
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${at.getDate()} ${MONTHS[at.getMonth()]}, ${pad(at.getHours())}:${pad(at.getMinutes())}:${pad(at.getSeconds())}`;
}

/**
 * A runtime in milliseconds, at a scale a human reads.
 *
 * Sub-second solves are the common case here and milliseconds are the honest
 * unit for them, but a solver given a large budget can run for minutes, and
 * "184320 ms" is a number a reader has to do work on. So the unit switches where
 * it stops being useful and says which one it used.
 */
export function formatRuntime(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds.toFixed(0)} ms`;
  return formatSeconds(milliseconds / 1000);
}

/**
 * One figure from a decision trace, rendered from its `unit` word.
 *
 * The backend sends explanations' numbers as a value plus a bare unit word
 * (`"rupees"`, `"seconds"`, `"percent"`, …) rather than pre-formatted, so this
 * is the single place that decides how each reads. That matters more than it
 * looks: a statement's figure is the *receipt* for a sentence, and a receipt
 * that renders differently from the same quantity elsewhere on the dashboard
 * would make the sentence unverifiable by eye — which is the one thing the
 * pairing exists to allow.
 *
 * An unrecognised unit falls back to the bare number **without inventing a
 * unit**. If the backend ever grows a seventh kind, this shows the value and
 * nothing else, which reads as incomplete rather than as a confident claim in
 * the wrong units.
 */
export function formatFigure(value: number, unit: string): string {
  switch (unit) {
    case "rupees":
      return formatRupees(value);
    case "seconds":
      // Unsigned, unlike the before/after panel's deltas. Most second-figures in
      // a trace are not changes: "before" and "late by" are both plain
      // magnitudes, and `formatSignedSeconds` would stamp a `+` on each of them
      // — a receipt claiming an increase for a number the sentence introduced
      // with no direction at all. The sign still survives where it is real,
      // because `toFixed` keeps a negative one, and the labels ("delay",
      // "late by") say what each figure is either way.
      return formatSeconds(value);
    case "metres":
      return formatDistance(value);
    case "litres":
      return formatLitres(value);
    case "percent":
      return `${value.toFixed(1)}%`;
    case "multiple":
      return `${value.toFixed(2)}×`;
    case "stops":
      return `${formatCount(value)} ${value === 1 ? "stop" : "stops"}`;
    case "count":
      return formatCount(value);
    default:
      return value.toFixed(2);
  }
}
