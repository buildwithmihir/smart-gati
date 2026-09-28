/**
 * Vehicle colour assignment.
 *
 * These are not hand-picked hues. They are slots 1-8 of the reference
 * categorical palette, assigned in fixed order, and the ordering is the
 * colourblind-safety mechanism rather than decoration — it was validated with
 * the palette validator against the actual basemap surface (`#fafaf8`, CARTO
 * Positron's background), not judged by eye.
 *
 * Result for the three vehicles this sample scenario uses, all-pairs:
 *
 *   lightness band    PASS   all 3 inside L 0.43-0.77
 *   chroma floor      PASS   all 3 >= 0.1
 *   CVD separation    PASS   worst pair aqua<->orange dE 9.2 (deutan)
 *   normal-vision     PASS   worst pair aqua<->blue   dE 24.0
 *   contrast          WARN   aqua at 2.69:1 — relief required
 *
 * That contrast warning is why every stop also carries a visible number and
 * label in text ink: identity is never encoded by colour alone. Keeping the
 * numbers is not decoration either — it is the accessibility mitigation the
 * warning obligates.
 *
 * Past three vehicles, all-pairs separation cannot be met by any ordering (red
 * and orange sit at dE 7.1, below the hard floor, and no reordering fixes a
 * pairlist that ignores order). The numbers on the markers are what carry
 * identity from there, and nothing here generates a new hue: a 9th vehicle
 * folds into a single documented "other" grey rather than cycling the palette,
 * because a cycled hue silently reintroduces a collision.
 */

/** Fixed categorical order. Never reordered, never cycled, never generated. */
export const VEHICLE_PALETTE = [
  "#2a78d6", // 1 blue
  "#eb6834", // 2 orange
  "#1baf7a", // 3 aqua
  "#eda100", // 4 yellow
  "#e87ba4", // 5 magenta
  "#008300", // 6 green
  "#4a3aa7", // 7 violet
  "#e34948", // 8 red
] as const;

/** The documented fold-to-"other" tone for a 9th and beyond. */
export const OTHER_VEHICLE_COLOR = "#6b7280";

/**
 * Stable colour for a vehicle.
 *
 * Keyed on the vehicle's own identity (the number in `V0`, `V1`, ...) rather
 * than its position in the current response, so a vehicle keeps its colour
 * across the whole UI — map line, marker, badge, legend and summary — and would
 * keep it even if the solver returned its routes in a different order. Colour
 * follows the entity, not its rank.
 */
export function colorForVehicle(vehicleId: string): string {
  const match = /(\d+)/.exec(vehicleId);
  if (!match) return OTHER_VEHICLE_COLOR;
  const index = Number(match[1]);
  return index < VEHICLE_PALETTE.length ? VEHICLE_PALETTE[index] : OTHER_VEHICLE_COLOR;
}

/** A vehicle's colour at reduced alpha, for badge backgrounds and washes. */
export function tintForVehicle(vehicleId: string, alpha = 0.12): string {
  const hex = colorForVehicle(vehicleId).replace("#", "");
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
