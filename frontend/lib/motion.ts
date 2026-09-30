/**
 * How long a redraw takes to read as a dissolve rather than a swap.
 *
 * These live in their own module because two places have to agree on them and
 * neither owns the other: the map schedules its layer and marker fades from
 * here, and the dashboard holds the loading skeleton on screen for one fade
 * past the first solution so the placeholder does not cut away before the
 * routes have faded up underneath it.
 *
 * `REDRAW_MS` is also published to CSS as `--qgati-redraw-ms` on the map
 * container, which is what keeps the stylesheet's marker fades in step with the
 * layer fades that JavaScript drives. Change it here and all three move
 * together.
 */

/**
 * The one duration. Tuned by ear: lengthen it and the dissolve turns syrupy,
 * shorten it and the snap comes back.
 */
export const REDRAW_MS = 420;

/** Gap between consecutive markers dealing in. */
export const MARKER_STAGGER_MS = 22;

/**
 * Ceiling on the stagger, in markers. Past a dozen the cumulative delay starts
 * to read as the map struggling rather than the map arriving.
 */
export const MARKER_STAGGER_CAP = 12;

/**
 * How long an injected incident's road pulses before settling.
 *
 * The pulse is the acknowledgement that the click landed — it starts on the
 * click, while the incident is being priced and the re-optimization is still
 * running, so it is answering "yes, that road" rather than "here is the answer".
 * That is why it is short and why it does not wait for the response: an
 * acknowledgement that arrives with the result is not an acknowledgement.
 *
 * It ends at a steady highlight rather than at nothing, because the incident
 * stays live until it is cleared and a road that is closed should keep saying
 * so. Deliberately under `REDRAW_MS` × 3: the pulse has to be over before the
 * routes finish dissolving, or the eye is pulled to the road instead of to the
 * change.
 */
export const PULSE_MS = 1200;

/** Pulses within `PULSE_MS`. Three reads as a signal; two reads as a flicker. */
export const PULSE_COUNT = 3;
