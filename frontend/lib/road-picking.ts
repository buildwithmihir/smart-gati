/**
 * Turning a click on the map into the roads an incident can name.
 *
 * A click lands on a *street*, but `POST /incident` names a **directed** edge
 * `(u, v)`. The gap between those two is this module, and it is worth being
 * explicit about because the graph does not close it for you.
 *
 * `graph_to_geojson` emits **every edge including the reverse twin** —
 * `backend/src/qgati/graph/geometry.py:233` — and says why: Delhi's network is
 * asymmetric, so the two directions of a street are genuinely different roads
 * with different travel times, and collapsing them would misrepresent the thing
 * this project routes on. That is right for routing and awkward for a map: the
 * two twins are drawn on top of each other, a click can only hit whichever one
 * the renderer puts on top, and closing that one leaves the other carriageway
 * open.
 *
 * So a click is read as *the street*, and `bothDirectionsOf` expands it back into
 * the one or two directed edges that make it up. For a closure that is what an
 * operator means by "this road is shut"; for a slow report it is what a street
 * jammed in both directions looks like. Nothing here decides *which* of those is
 * being reported — it only resolves the geometry, and the caller picks the
 * treatment.
 */

import { isLineFeature, type GraphGeoJSON } from "@/lib/api";

/** One directed road, with the polyline to draw for it. */
export type DirectedRoad = {
  u: number;
  v: number;
  /** GeoJSON `[lon, lat]` pairs, as the graph emitted them. */
  coordinates: [number, number][];
};

/** A street the user clicked: the node pair, and every directed edge of it. */
export type PickedRoad = {
  /** The endpoints of the edge that was actually hit, as clicked. */
  u: number;
  v: number;
  /** The clicked edge first, then its reverse twin when the street has one. */
  edges: DirectedRoad[];
};

/** Every directed edge of a graph, keyed by `"u->v"`, for twin lookups. */
function lineIndex(graph: GraphGeoJSON): Map<string, DirectedRoad> {
  const roads = new Map<string, DirectedRoad>();
  for (const feature of graph.features) {
    if (!isLineFeature(feature)) continue;
    const { u, v } = feature.properties;
    roads.set(`${u}->${v}`, { u, v, coordinates: feature.geometry.coordinates });
  }
  return roads;
}

/**
 * The street through `u -> v`, as one or two directed edges.
 *
 * The clicked edge comes first so a caller that only wants one — a report about
 * the carriageway the operator actually pointed at — can take `edges[0]` and be
 * right. When the road is one-way there is no twin and the result has a single
 * entry, which is not a special case for the caller to handle: the same loop
 * that files both accidents files the one.
 */
export function bothDirectionsOf(graph: GraphGeoJSON, u: number, v: number): DirectedRoad[] {
  const roads = lineIndex(graph);
  const clicked = roads.get(`${u}->${v}`);
  const edges: DirectedRoad[] = clicked ? [clicked] : [];
  const twin = roads.get(`${v}->${u}`);
  // `u === v` would make the twin the clicked edge itself; a self-loop is not a
  // road anybody drives and must not be filed twice.
  if (twin && u !== v) edges.push(twin);
  return edges;
}

/**
 * A street named the way an operator reads one: `"249782331 -> 249782340"`.
 *
 * Node ids rather than a street name, because the graph is not guaranteed to
 * carry one — and it is the id that a backend error message will quote back if
 * the incident is refused, so showing the same thing is what lets the two be
 * matched up.
 */
export function describeRoad(u: number, v: number): string {
  return `${u} → ${v}`;
}
