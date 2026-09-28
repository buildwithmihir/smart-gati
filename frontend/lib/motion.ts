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
