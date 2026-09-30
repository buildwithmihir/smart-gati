"use client";

/**
 * The MapLibre canvas: base roads, one coloured line per vehicle, numbered stops
 * — and, since Phase 8, the incident surface: a clickable road network, a
 * pulsing highlight on a reported road, the dispatched plan left faintly under a
 * re-planned one, and the live fleet as small dots.
 *
 * MapLibre touches `window` and WebGL, so this module is loaded through
 * `next/dynamic` with `ssr: false` — it must never be evaluated on the server.
 *
 * Note the import style: maplibre-gl v6 ships **no default export**, only named
 * ones, so `import maplibregl from "maplibre-gl"` fails to typecheck.
 *
 * Two join problems are solved here, both of which come from what the API
 * actually returns:
 *
 * 1. **Stops have no coordinates.** `Route.stops[]` carries `{delivery_id,
 *    node}` — a node id, not a lat/lon. The positions come from the `Point`
 *    features of the scoped `/graph/delhi` response, which is fetched for
 *    exactly this reason as well as for the road layer.
 * 2. **Unused vehicles come back as empty routes.** A vehicle with no work has
 *    `stops: []` and `geometry: []`, so it is filtered out here rather than
 *    drawn as a zero-length line or given a marker it has not earned.
 *
 * ## Redraws cross-fade
 *
 * A re-solve produces a different set of routes, and MapLibre cannot tween
 * GeoJSON geometry — `setData` with new coordinates snaps. So a redraw is not
 * an edit, it is a handover: the outgoing generation is faded to nothing and
 * removed on a timer, while the incoming one is added at zero opacity and
 * faded up. The two overlap for the length of the fade, which is what turns a
 * jarring swap into a dissolve.
 *
 * That handover is the reason this file is more careful than it looks. The
 * fade-out is asynchronous, so a second redraw can arrive while the first is
 * still in flight; `retiringRef` is what keeps generations from stacking up,
 * and it is flushed — not just cancelled — when a new redraw starts, because
 * layers left behind are layers drawn forever.
 *
 * ## Layering is decided by insertion anchor, not by order of arrival
 *
 * Four kinds of layer stack here, bottom to top: roads, ghosts (the dispatched
 * plan), live routes, the incident highlight. Getting that order by adding them
 * in the right sequence would only work on the first pass — a later pass adds
 * layers *on top*, so after one re-plan the routes would bury the closure
 * highlight and the whole point of the panel would be invisible. So the two
 * insertions that can happen at any time carry an explicit `beforeId`:
 * everything the routes and ghosts add goes *below* the incident glow when it
 * exists. Order is then a property of the code rather than of the history.
 */

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  GeoJSONSource,
  LngLatBounds,
  MapLibreMap,
  Marker,
  setWorkerUrl,
  type ErrorEvent as MapLibreErrorEvent,
  type MapMouseEvent,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import {
  isLineFeature,
  isPointFeature,
  type GraphGeoJSON,
  type IncidentKind,
  type Route,
  type VehicleTrack,
} from "@/lib/api";
import {
  MARKER_STAGGER_CAP,
  MARKER_STAGGER_MS,
  PULSE_COUNT,
  PULSE_MS,
  REDRAW_MS,
} from "@/lib/motion";
import { bothDirectionsOf, type DirectedRoad, type PickedRoad } from "@/lib/road-picking";
import { colorForVehicle } from "@/lib/vehicle-colors";

/**
 * Point MapLibre at a worker the server will actually serve.
 *
 * MapLibre parses every source — basemap tiles and our GeoJSON alike — inside a
 * web worker, and derives the worker's URL from `import.meta.url` plus a
 * sibling filename. Under Turbopack that resolves to a path that is not an
 * emitted asset, so the request comes back as the HTML shell, the module worker
 * is rejected on its MIME type, and the worker never starts. The failure is
 * quiet and looks like a styling bug: the style's background layer paints, the
 * DOM markers appear, and every vector source silently stays empty — no roads,
 * no routes, and none of the basemap's own tiles.
 *
 * So the two worker files are served as static assets from `public/maplibre/`,
 * and this is what points at them. It is a side effect at module scope on
 * purpose: it has to be in place before the first map is constructed.
 */
setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");

/** CARTO Positron — free and key-less, which is why it is the default here. */
const STYLE_URL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

/** Roughly Connaught Place, the centre of the cached extract. */
const DELHI_CENTER: [number, number] = [77.209, 28.6139];

const ROADS_SOURCE = "qgati-roads";

/**
 * Padding between the solution's bounding box and the edge of the viewport.
 *
 * Wide enough to clear the chips that float over the canvas — the solve chip
 * top-centre, the live pill top-right, the tile-error notice bottom-left — so a
 * first solve does not frame the routes underneath them.
 */
const FRAME_PADDING = 72;

/** The highlight's source, and the core layer drawn from it. The glow is `${}-glow`. */
const INCIDENT_SOURCE = "qgati-incident";

/**
 * How far from the cursor a click still counts as hitting a road, in pixels.
 *
 * The road line is 1.2 px wide, so a pixel-exact hit test would demand
 * precision nobody has with a mouse and nobody has at all with a trackpad. Ten
 * pixels is about a fingertip at typical zoom, and still well under the
 * spacing of the parallel carriageways in this extract, so it widens the
 * target without merging two streets into one.
 */
const PICK_RADIUS_PX = 10;

/** The dispatched plan, left under the live one. Faint enough to read as context. */
const GHOST_OPACITY = 0.28;

/**
 * Where the incident highlight rests between pulses and after them.
 *
 * The core is nearly solid because it is a small line and has to survive being
 * drawn over a route; the glow is a wide, blurred halo and only has to be
 * present, so it sits low. The pulse's peaks are separate constants below
 * because the two have very different headroom — the core has almost none, the
 * glow has most of the range.
 */
const INCIDENT_CORE_OPACITY = 0.9;
const INCIDENT_GLOW_OPACITY = 0.2;
const INCIDENT_CORE_PEAK = 1;
const INCIDENT_GLOW_PEAK = 0.7;

/**
 * Clamp an opacity into MapLibre's valid range.
 *
 * MapLibre rejects a paint value outside `[0, 1]`, and it does so by firing a
 * style-validation error — which arrives on the same `error` event a basemap
 * outage does, and which the handler below used to report as "tiles
 * unavailable". One out-of-range opacity therefore produced a console error *per
 * animation frame* plus a notice blaming the network.
 *
 * The pulse is the only opacity in this file that is *computed* rather than
 * written as a literal, and its arithmetic looks bounded: the core runs
 * `0.9 + 0.1 · lift` with `lift ∈ [0, 1]`, so it should top out at exactly 1.
 * Floating point does not honour that reasoning — `0.9 + 0.1` has no obligation
 * to be exactly `1` — and a value above 1 was observed reaching the style. The
 * fix is a clamp rather than a rearranged formula, because no rearrangement makes
 * a derived float provably in range.
 */
function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/**
 * Read a design token out of CSS.
 *
 * MapLibre paint values are JavaScript while the palette lives in
 * `globals.css`, and the two have to agree. The way they stop drifting is that
 * only one of them exists: the token is read at runtime, and the literal beside
 * each name is a fallback matching the stylesheet, so a failed read degrades to
 * the previous hardcoded colour rather than to nothing at all.
 *
 * Lazily, on first use, rather than at module scope — in development the
 * stylesheet is injected by script, and may not be in the document yet when
 * this module evaluates.
 */
let cachedTokens: { road: string; basemap: string; danger: string; warn: string } | null = null;

function paletteTokens() {
  if (cachedTokens) return cachedTokens;
  const read = (name: string, fallback: string) => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  };
  cachedTokens = {
    road: read("--color-road", "#b5b3ac"),
    basemap: read("--color-basemap", "#fafaf8"),
    danger: read("--color-danger", "#d03b3b"),
    warn: read("--color-warn", "#b45309"),
  };
  return cachedTokens;
}

/** A redraw generation that is on its way out, and the timer that finishes it. */
type Retiring = {
  routeIds: string[];
  markers: Marker[];
  timer: number;
};

/** Removes a generation's layers and sources, children before parents. */
function discardLayers(map: MapLibreMap, routeIds: string[]) {
  for (const id of routeIds) {
    if (map.getLayer(`${id}-casing`)) map.removeLayer(`${id}-casing`);
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }
}

/** True when the current viewport already contains all of `bounds`. */
function viewCovers(map: MapLibreMap, bounds: LngLatBounds): boolean {
  const view = map.getBounds();
  return view.contains(bounds.getSouthWest()) && view.contains(bounds.getNorthEast());
}

/** Squared distance from a point to the segment `a`→`b`, all in screen pixels. */
function squaredDistanceToSegment(
  point: [number, number],
  a: [number, number],
  b: [number, number],
): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return (point[0] - a[0]) ** 2 + (point[1] - a[1]) ** 2;
  // Where the point projects onto the line, clamped to the segment.
  const t = Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / lengthSq));
  const cx = a[0] + t * dx;
  const cy = a[1] + t * dy;
  return (point[0] - cx) ** 2 + (point[1] - cy) ** 2;
}

/**
 * Distance from the cursor to a drawn road, in pixels.
 *
 * Against the *segments*, not the vertices. A road in this extract can be a
 * polyline of a dozen vertices or a straight two-point block, and on a long
 * block the nearest vertex can be hundreds of pixels away while the cursor sits
 * squarely on the line — comparing vertices would make long straight roads
 * unpickable exactly where they are easiest to point at.
 */
function distanceToPolyline(
  point: [number, number],
  coordinates: [number, number][],
): number {
  if (coordinates.length === 0) return Infinity;
  if (coordinates.length === 1) {
    return Math.hypot(point[0] - coordinates[0][0], point[1] - coordinates[0][1]);
  }
  let best = Infinity;
  for (let index = 1; index < coordinates.length; index += 1) {
    const distance = squaredDistanceToSegment(point, coordinates[index - 1], coordinates[index]);
    if (distance < best) best = distance;
  }
  return Math.sqrt(best);
}

/**
 * The street under a click, resolved to the directed edge(s) an incident names.
 *
 * Two steps, and the second is the one that is easy to skip. `queryRenderedFeatures`
 * is asked over a box rather than a point (see `PICK_RADIUS_PX`), which means it
 * can return several roads; each is scored by its true pixel distance to the
 * cursor and the closest wins. Long roads are scored over their whole polyline
 * and projected into screen space first, so zoom level does not change which
 * road is considered nearest.
 *
 * The result is then widened from one directed edge to the street it belongs to
 * by `bothDirectionsOf` — the graph emits both carriageways as separate edges;
 * see that module for why, and for why a click is read as the street.
 */
function pickRoad(
  map: MapLibreMap,
  point: { x: number; y: number },
  graph: GraphGeoJSON,
): PickedRoad | null {
  // `queryRenderedFeatures` with a non-existent layer returns nothing and warns,
  // so the guard is only about not filling the console on a click that arrives
  // before the roads have been added.
  if (!map.getLayer(ROADS_SOURCE)) return null;

  // The box is typed rather than inferred: the query takes a `PointLike` pair
  // and an inferred `number[][]` is not one, because a `PointLike` is exactly
  // two numbers and not an array of any length.
  const box: [[number, number], [number, number]] = [
    [point.x - PICK_RADIUS_PX, point.y - PICK_RADIUS_PX],
    [point.x + PICK_RADIUS_PX, point.y + PICK_RADIUS_PX],
  ];

  const hits = map.queryRenderedFeatures(box, { layers: [ROADS_SOURCE] });

  const cursor: [number, number] = [point.x, point.y];
  let best: { u: number; v: number; distance: number } | null = null;

  for (const hit of hits) {
    const { u, v } = hit.properties;
    if (typeof u !== "number" || typeof v !== "number") continue;
    const geometry = hit.geometry;
    if (geometry.type !== "LineString") continue;

    const screen = geometry.coordinates.map(([lon, lat]) => {
      const corner: [number, number] = [lon, lat];
      const projected = map.project(corner);
      return [projected.x, projected.y] as [number, number];
    });

    const distance = distanceToPolyline(cursor, screen);
    if (best === null || distance < best.distance) best = { u, v, distance };
  }

  if (!best) return null;

  const edges = bothDirectionsOf(graph, best.u, best.v);
  if (edges.length === 0) return null;
  return { u: best.u, v: best.v, edges };
}

/**
 * Linear interpolation along a polyline, by cumulative planar length.
 *
 * Length in degrees rather than metres — the two differ by a factor of about
 * `cos(latitude)`, which is constant across a viewport, so it cancels out of a
 * *fraction* along the line. Converting properly would buy nothing here.
 */
function pointAlong(
  coordinates: [number, number][],
  fraction: number,
): [number, number] | null {
  if (coordinates.length === 0) return null;
  if (coordinates.length === 1) return coordinates[0];

  const legs: number[] = [];
  let total = 0;
  for (let index = 1; index < coordinates.length; index += 1) {
    const leg = Math.hypot(
      coordinates[index][0] - coordinates[index - 1][0],
      coordinates[index][1] - coordinates[index - 1][1],
    );
    legs.push(leg);
    total += leg;
  }
  if (total <= 0) return coordinates[0];

  let target = Math.min(Math.max(fraction, 0), 1) * total;
  for (let index = 0; index < legs.length; index += 1) {
    if (target > legs[index]) {
      target -= legs[index];
      continue;
    }
    const ratio = legs[index] === 0 ? 0 : target / legs[index];
    const [lonA, latA] = coordinates[index];
    const [lonB, latB] = coordinates[index + 1];
    return [lonA + (lonB - lonA) * ratio, latA + (latB - latA) * ratio];
  }
  return coordinates[coordinates.length - 1];
}

/** One directed road's polyline and its cost, indexed for fleet placement. */
type EdgeGeometry = { coordinates: [number, number][]; travel_time: number };

/** Where a vehicle is *on its edge*, from what the watcher reports. */
function fleetPosition(
  track: VehicleTrack,
  nodeCoords: Map<number, [number, number]>,
  edgeGeometry: Map<string, EdgeGeometry>,
): [number, number] | null {
  if (!track.edge || track.finished) return null;

  const edge = edgeGeometry.get(`${track.edge.u}->${track.edge.v}`);
  const from = nodeCoords.get(track.edge.u);
  const to = nodeCoords.get(track.edge.v);

  const coordinates: [number, number][] | null =
    edge?.coordinates ??
    (from && to ? [from, to] : from ? [from] : null);
  if (!coordinates || coordinates.length < 2) return null;

  // The watcher reports the road a vehicle is on and the seconds left before
  // the next intersection — never a point. Those two *are* enough for a
  // fraction, because the road's own cost is in the graph: driving `cost -
  // remaining` of a road that costs `cost` is that far along it. Both numbers
  // are seconds under the *current* costs while the graph carries the base
  // ones, so a road under an incident is slightly off; nothing else is.
  //
  // Where that fails — a finished vehicle, or an edge missing from this scoped
  // graph — the midpoint is used, which claims nothing beyond what is certain:
  // the vehicle is on this block, somewhere between its ends.
  const fraction =
    track.remaining_seconds !== null && edge && edge.travel_time > 0
      ? 1 - track.remaining_seconds / edge.travel_time
      : 0.5;

  return pointAlong(coordinates, fraction);
}

/** The road an incident was reported on, plus a token that re-arms its pulse. */
export type IncidentHighlight = {
  kind: IncidentKind;
  /** Both carriageways of the street, in the order they were reported. */
  edges: DirectedRoad[];
  /**
   * Bumped for every new report. The pulse runs on a *change* in this number
   * rather than on every render, so a re-render — a poll landing, a panel
   * opening — does not set the road flashing again.
   */
  pulse: number;
};

type RouteMapProps = {
  graph: GraphGeoJSON | null;
  routes: Route[];
  depotNode: number | null;
  /**
   * The plan the fleet was dispatched on. Drawn faintly under `routes` so a
   * shorter re-planned route reads as *remaining* work rather than as the whole
   * job having shrunk. Empty when the live plan is the dispatched one.
   */
  ghostRoutes?: Route[];
  /** Live vehicle positions from the watcher. */
  fleet?: VehicleTrack[];
  highlight?: IncidentHighlight | null;
  /** True while the map is waiting for a road to be clicked. */
  picking?: boolean;
  onPickRoad?: (road: PickedRoad) => void;
  /**
   * Whether this map is on screen, when its container may have been hidden.
   *
   * The app shell hides the dashboard with `display:none` rather than unmounting
   * it, so the loaded scenario, the plan and the incident survive a trip to the
   * History tab. MapLibre sizes its canvas from the container's box and listens
   * for *window* resize — not for the container appearing — so a canvas hidden
   * at 0×0 comes back at 0×0 and stays there even once the container has a size
   * again. `resize()` is the only way to tell it to re-measure.
   *
   * Defaults to `true`: a caller that never hides the map is already correct and
   * should not have to say so.
   */
  active?: boolean;
};

export default function RouteMap({
  graph,
  routes,
  depotNode,
  ghostRoutes = [],
  fleet = [],
  highlight = null,
  picking = false,
  onPickRoad,
  active = true,
}: RouteMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markersRef = useRef<Marker[]>([]);
  const routeIdsRef = useRef<string[]>([]);
  const ghostIdsRef = useRef<string[]>([]);
  const fleetMarkersRef = useRef<Map<string, Marker>>(new Map());
  const retiringRef = useRef<Retiring | null>(null);
  /** Whether the camera has been aimed at a solution at least once. */
  const framedRef = useRef(false);
  /** Bumped on every redraw; see where the route ids are built. */
  const generationRef = useRef(0);
  const [ready, setReady] = useState(false);
  const [tileError, setTileError] = useState<string | null>(null);

  // The click handler is registered once, against the map, and lives for the
  // life of the map — so it cannot close over the props it needs to read. These
  // refs are how it sees the current ones without being torn down and re-added
  // on every render, which would be a listener leak waiting to happen.
  const pickingRef = useRef(picking);
  const onPickRoadRef = useRef(onPickRoad);
  const graphRef = useRef(graph);

  useEffect(() => {
    pickingRef.current = picking;
    onPickRoadRef.current = onPickRoad;
    graphRef.current = graph;
  });

  /**
   * Re-measure when this map comes back from `display:none`.
   *
   * The effect runs after the DOM is committed, so the container already has its
   * real size by the time `resize()` reads it — which is why this is an effect
   * and not a layout measurement. Guarded on `mapRef.current` rather than on
   * `ready`, because the map may still be initialising: `resize()` on a live
   * instance that has not finished loading is harmless, and skipping it would
   * leave the first switch back to Dashboard blank until something else resized
   * the window.
   */
  useEffect(() => {
    if (active) mapRef.current?.resize();
  }, [active]);

  // Only vehicles that actually received work are drawn.
  const activeRoutes = useMemo(() => routes.filter((route) => route.stops.length > 0), [routes]);

  const activeGhosts = useMemo(
    () => ghostRoutes.filter((route) => route.stops.length > 0 && route.geometry.length > 0),
    [ghostRoutes],
  );

  /** node id -> [lon, lat], from the graph's Point features. */
  const nodeCoords = useMemo(() => {
    const lookup = new Map<number, [number, number]>();
    if (!graph) return lookup;
    for (const feature of graph.features) {
      if (isPointFeature(feature)) lookup.set(feature.properties.id, feature.geometry.coordinates);
    }
    return lookup;
  }, [graph]);

  /** `"u->v"` -> the road's polyline and base cost, for placing the fleet. */
  const edgeGeometry = useMemo(() => {
    const lookup = new Map<string, EdgeGeometry>();
    if (!graph) return lookup;
    for (const feature of graph.features) {
      if (!isLineFeature(feature)) continue;
      const { u, v, travel_time } = feature.properties;
      lookup.set(`${u}->${v}`, { coordinates: feature.geometry.coordinates, travel_time });
    }
    return lookup;
  }, [graph]);

  /** LineStrings only — the Point features are lookup data, not geometry to draw. */
  const roads = useMemo(() => {
    if (!graph) return null;
    return {
      type: "FeatureCollection" as const,
      features: graph.features.filter(isLineFeature),
    };
  }, [graph]);

  // -- map lifecycle ------------------------------------------------------ //
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new MapLibreMap({
      container: containerRef.current,
      style: STYLE_URL,
      center: DELHI_CENTER,
      zoom: 12,
      attributionControl: { compact: true },
    });
    mapRef.current = map;

    // Readiness is gated on **`style.load`**, not on `load`.
    //
    // `load` fires only once the basemap's own tiles have finished, which makes
    // it a dependency on a third party: on a slow link, behind a proxy, or if
    // CARTO is having a bad day, `load` never fires and every route stays
    // undrawn — even though the routes and roads are GeoJSON we already hold
    // and do not need a single basemap tile to render. `style.load` is the
    // event that actually matters here, because a parsed style is the only
    // precondition for `addSource`/`addLayer`.
    //
    // `load` is still listened for as a belt-and-braces fallback, and the
    // synchronous `isStyleLoaded()` check covers a style that resolved from
    // cache before this effect ran.
    const markReady = () => setReady(true);
    map.on("style.load", markReady);
    map.on("load", markReady);
    if (map.isStyleLoaded()) markReady();

    map.on("error", (event: MapLibreErrorEvent) => {
      // **Only a tile failure is a basemap problem.** MapLibre routes every style
      // and configuration error through this same event — a paint value out of
      // range, a layer that was never added, a malformed filter — and reporting
      // those as "basemap tiles unavailable" tells the reader to ignore a bug
      // that is ours, and to look at the network instead of the code. A tile
      // error is the one kind that identifies itself: the source fires it with
      // `{ tile }` attached (`SourceCache`, on a fetch failure), and nothing else
      // does. So that is the test.
      const { tile } = event as MapLibreErrorEvent & { tile?: unknown };
      if (!tile) return;
      setTileError(event.error?.message ?? "unknown tile error");
    });

    // One click handler for the whole picking mode. It is deliberately not
    // added and removed with the mode: a listener attached on the transition
    // can miss the click that arrives in the same frame, and the flag it reads
    // is already the cheapest possible guard.
    map.on("click", (event: MapMouseEvent) => {
      if (!pickingRef.current) return;
      const current = graphRef.current;
      const report = onPickRoadRef.current;
      if (!current || !report) return;
      const road = pickRoad(map, event.point, current);
      if (road) report(road);
    });

    return () => {
      // A pending fade-out outlives the map it was scheduled against, so its
      // timer has to die with the map — otherwise it fires against a removed
      // instance and throws.
      const retiring = retiringRef.current;
      if (retiring) {
        window.clearTimeout(retiring.timer);
        retiringRef.current = null;
      }
      markersRef.current.forEach((marker) => marker.remove());
      markersRef.current = [];
      fleetMarkersRef.current.forEach((marker) => marker.remove());
      fleetMarkersRef.current = new Map();
      routeIdsRef.current = [];
      ghostIdsRef.current = [];
      framedRef.current = false;
      map.remove();
      mapRef.current = null;
      setReady(false);
    };
  }, []);

  // The crosshair is the entire affordance for picking on the canvas itself;
  // the strip in the incident panel says what it is for. Set on the canvas
  // rather than on the container because MapLibre writes `cursor` on the canvas
  // during drags and would otherwise overwrite it.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    map.getCanvas().style.cursor = picking ? "crosshair" : "";
  }, [picking, ready]);

  // -- data lifecycle ----------------------------------------------------- //
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;

    const tokens = paletteTokens();

    // Base roads are added once and then only re-pointed at new data.
    if (roads) {
      const existing = map.getSource(ROADS_SOURCE) as GeoJSONSource | undefined;
      if (existing) {
        existing.setData(roads);
      } else {
        map.addSource(ROADS_SOURCE, { type: "geojson", data: roads });
        map.addLayer({
          id: ROADS_SOURCE,
          type: "line",
          source: ROADS_SOURCE,
          paint: {
            "line-color": tokens.road,
            "line-width": 1.2,
            "line-opacity": 0.75,
          },
        });
      }
    }

    // Finish any generation still fading from an earlier redraw, now rather
    // than on its timer. Two dissolves at once is mud; a stale generation that
    // outlives its fade is a layer that never leaves.
    const retiring = retiringRef.current;
    if (retiring) {
      window.clearTimeout(retiring.timer);
      discardLayers(map, retiring.routeIds);
      retiring.markers.forEach((marker) => marker.remove());
      retiringRef.current = null;
    }

    // Hand the outgoing generation over to the fade, and take the incoming one
    // into the empty slots it leaves behind.
    const outgoingRouteIds = routeIdsRef.current;
    const outgoingMarkers = markersRef.current;

    for (const id of outgoingRouteIds) {
      if (map.getLayer(`${id}-casing`)) {
        map.setPaintProperty(`${id}-casing`, "line-opacity", 0);
      }
      if (map.getLayer(id)) {
        map.setPaintProperty(id, "line-opacity", 0);
      }
    }
    for (const marker of outgoingMarkers) {
      marker.getElement().setAttribute("data-retiring", "true");
    }

    routeIdsRef.current = [];
    markersRef.current = [];

    if (outgoingRouteIds.length > 0 || outgoingMarkers.length > 0) {
      const pending: Retiring = {
        routeIds: outgoingRouteIds,
        markers: outgoingMarkers,
        timer: 0,
      };
      pending.timer = window.setTimeout(() => {
        discardLayers(map, pending.routeIds);
        pending.markers.forEach((marker) => marker.remove());
        if (retiringRef.current === pending) retiringRef.current = null;
      }, REDRAW_MS);
      retiringRef.current = pending;
    }

    // Shared by both layers of a route: the fade-up, and the fade they will
    // need on the way out. MapLibre reads a property's transition from the
    // layer's own paint spec, so it has to be declared here and not at the
    // `setPaintProperty` call, which takes no transition argument.
    const fade = { duration: REDRAW_MS, delay: 0 };

    // Where new layers go so they do not bury the incident highlight. MapLibre
    // appends, so a redraw that happens while a road is reported would draw
    // straight over the mark saying which road — the one thing on screen the
    // re-plan is about. Passing the glow as `beforeId` puts everything below it.
    // `undefined` is correct and means "on top": with no incident there is
    // nothing to stay beneath.
    const incidentAnchor = map.getLayer(`${INCIDENT_SOURCE}-glow`)
      ? `${INCIDENT_SOURCE}-glow`
      : undefined;

    // Layer and source ids are scoped to this generation, and they have to be.
    // A stable per-vehicle id is what the previous version used, and it is
    // wrong the moment a cross-fade exists: the whole point is that two
    // generations are alive at once, so the incoming `addSource("route-V0")`
    // collides with the outgoing one and throws "already exists" — on the
    // second solve, and every solve after it.
    const generation = (generationRef.current += 1);

    let markerIndex = 0;

    for (const route of activeRoutes) {
      if (route.geometry.length === 0) continue;

      const id = `route-${generation}-${route.vehicle_id}`;
      const color = colorForVehicle(route.vehicle_id);

      map.addSource(id, {
        type: "geojson",
        data: {
          type: "Feature",
          properties: {},
          geometry: { type: "LineString", coordinates: route.geometry },
        },
      });

      // Surface casing first, then the coloured line over it. Both are added
      // invisible and then faded up, which is what makes the arrival a
      // dissolve rather than a pop.
      map.addLayer(
        {
          id: `${id}-casing`,
          type: "line",
          source: id,
          layout: { "line-cap": "round", "line-join": "round" },
          paint: {
            "line-color": tokens.basemap,
            "line-width": 7,
            "line-opacity": 0,
            "line-opacity-transition": fade,
          },
        },
        incidentAnchor,
      );
      map.addLayer(
        {
          id,
          type: "line",
          source: id,
          layout: { "line-cap": "round", "line-join": "round" },
          paint: {
            "line-color": color,
            "line-width": 4,
            "line-opacity": 0,
            "line-opacity-transition": fade,
          },
        },
        // The same anchor again, which lands this *above* the casing just added
        // and still below the highlight — inserting before a layer puts you
        // immediately beneath it, so both go under the same mark in order.
        incidentAnchor,
      );
      routeIdsRef.current.push(id);

      map.setPaintProperty(`${id}-casing`, "line-opacity", 0.9);
      map.setPaintProperty(id, "line-opacity", 1);

      // Numbered stop markers, numbered by position within *this* vehicle's
      // route — the same number the left panel shows on the stop's badge.
      route.stops.forEach((stop, index) => {
        const coords = nodeCoords.get(stop.node);
        if (!coords) return;

        // The wrapper is what MapLibre positions and fades; the dot inside is
        // what scales in. See the marker rules in `globals.css` for why they
        // cannot be the same element.
        const element = document.createElement("div");
        element.className = "qgati-marker";
        element.style.setProperty(
          "--marker-delay",
          `${Math.min(markerIndex, MARKER_STAGGER_CAP) * MARKER_STAGGER_MS}ms`,
        );
        markerIndex += 1;

        const dot = document.createElement("div");
        dot.className = "qgati-marker-dot";
        dot.style.setProperty("--marker-color", color);
        dot.textContent = String(index + 1);
        element.appendChild(dot);

        element.title = `${stop.delivery_id} — stop ${index + 1} of ${route.vehicle_id}`;

        markersRef.current.push(
          new Marker({ element, anchor: "center" }).setLngLat(coords).addTo(map),
        );
      });
    }

    // The depot is not a numbered stop, so it gets its own quieter marker.
    if (depotNode !== null) {
      const coords = nodeCoords.get(depotNode);
      if (coords) {
        const element = document.createElement("div");
        element.className = "qgati-depot-marker";
        element.title = "Depot";
        const dot = document.createElement("div");
        dot.className = "qgati-depot-dot";
        element.appendChild(dot);

        markersRef.current.push(
          new Marker({ element, anchor: "center" }).setLngLat(coords).addTo(map),
        );
      }
    }

    // Ghosts: the plan the fleet was dispatched on, under the live one.
    //
    // Their ids are stable and carry no generation, which is the opposite of
    // the choice made for the routes above and is right for the same reason
    // inverted: ghosts never cross-fade, so there is never a second generation
    // of them to collide with, and a stable id is what lets them be re-pointed
    // with `setData` instead of rebuilt.
    //
    // They are inserted *beneath* the routes just added rather than on top of
    // them, which is the whole point — an underlay drawn above what it
    // underlays is just a second route.
    const ghostAnchor = routeIdsRef.current[0];
    const ghostBefore = ghostAnchor ? `${ghostAnchor}-casing` : incidentAnchor;
    const wanted = new Set<string>();

    for (const route of activeGhosts) {
      const id = `ghost-${route.vehicle_id}`;
      wanted.add(id);

      const data = {
        type: "Feature" as const,
        properties: {},
        geometry: { type: "LineString" as const, coordinates: route.geometry },
      };

      const existing = map.getSource(id) as GeoJSONSource | undefined;
      if (existing) {
        existing.setData(data);
        continue;
      }

      map.addSource(id, { type: "geojson", data });
      map.addLayer(
        {
          id,
          type: "line",
          source: id,
          layout: { "line-cap": "round", "line-join": "round" },
          paint: {
            "line-color": colorForVehicle(route.vehicle_id),
            "line-width": 3,
            "line-opacity": GHOST_OPACITY,
          },
        },
        ghostBefore,
      );
    }

    for (const id of ghostIdsRef.current) {
      if (wanted.has(id)) continue;
      if (map.getLayer(id)) map.removeLayer(id);
      if (map.getSource(id)) map.removeSource(id);
    }
    ghostIdsRef.current = [...wanted];

    // Frame the solution. Bounds come from the route geometry, which already
    // starts and ends at the depot, so the depot is always included.
    if (activeRoutes.length > 0) {
      const bounds = new LngLatBounds();
      let extended = false;
      for (const route of activeRoutes) {
        for (const coord of route.geometry) {
          bounds.extend(coord);
          extended = true;
        }
      }

      if (extended) {
        if (!framedRef.current) {
          map.fitBounds(bounds, { padding: FRAME_PADDING, duration: 700, maxZoom: 15 });
          framedRef.current = true;
        } else if (!viewCovers(map, bounds)) {
          // A later solve only earns a camera move if it actually lands outside
          // the view. Re-framing on every redraw would yank the map out from
          // under someone who has just zoomed in to look at a street — which is
          // precisely what the incident controls invite them to do.
          map.fitBounds(bounds, { padding: FRAME_PADDING, duration: 500, maxZoom: 15 });
        }
      }
    }
  }, [ready, roads, activeRoutes, activeGhosts, nodeCoords, depotNode]);

  // -- the incident highlight --------------------------------------------- //
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;

    const tokens = paletteTokens();
    const color = highlight?.kind === "slow" ? tokens.warn : tokens.danger;

    const data = {
      type: "FeatureCollection" as const,
      features: (highlight?.edges ?? []).map((edge) => ({
        type: "Feature" as const,
        properties: {},
        geometry: { type: "LineString" as const, coordinates: edge.coordinates },
      })),
    };

    const glowId = `${INCIDENT_SOURCE}-glow`;
    const source = map.getSource(INCIDENT_SOURCE) as GeoJSONSource | undefined;

    if (source) {
      source.setData(data);
      map.setPaintProperty(glowId, "line-color", color);
      map.setPaintProperty(INCIDENT_SOURCE, "line-color", color);
      // Restored to the top on every change, not just at creation. The two
      // mechanisms cover opposite cases: the `beforeId` in the routes effect
      // keeps a redraw from drawing over a mark that is already there, and this
      // puts the mark back on top of routes that were drawn while it was not.
      // Both are needed, because a clear-and-report cycle can produce either
      // order.
      map.moveLayer(glowId);
      map.moveLayer(INCIDENT_SOURCE);
    } else {
      map.addSource(INCIDENT_SOURCE, { type: "geojson", data });
      // Glow under core. The glow is blurred as well as wide, because a hard
      // 14 px line over a pale basemap reads as a second road rather than as
      // attention being drawn to one.
      map.addLayer({
        id: glowId,
        type: "line",
        source: INCIDENT_SOURCE,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": color,
          "line-width": 14,
          "line-blur": 4,
          "line-opacity": 0,
        },
      });
      map.addLayer({
        id: INCIDENT_SOURCE,
        type: "line",
        source: INCIDENT_SOURCE,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": color,
          "line-width": 3.5,
          "line-opacity": 0,
        },
      });
    }

    // The single write path for every incident opacity, so the clamp below cannot
    // be forgotten at one of the four call sites.
    const paint = (glow: number, core: number) => {
      map.setPaintProperty(glowId, "line-opacity", clamp01(glow));
      map.setPaintProperty(INCIDENT_SOURCE, "line-opacity", clamp01(core));
    };

    if (highlight === null || highlight.edges.length === 0) {
      paint(0, 0);
      return;
    }

    // Nothing is *communicated* by the pulse — the road is marked either way —
    // so under a reduced-motion preference it is skipped and the highlight
    // simply appears at its resting weight.
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      paint(INCIDENT_GLOW_OPACITY, INCIDENT_CORE_OPACITY);
      return;
    }

    // The pulse starts on the click, before the incident has been priced and
    // while the re-optimization is still running, which is what makes it an
    // acknowledgement rather than a result. `lift` is that signal: it starts at
    // 1 — so the first frame is already at full — oscillates `PULSE_COUNT`
    // times, and decays to 0, leaving the highlight exactly at its resting
    // weight with no discontinuity at the handover.
    //
    // Paint property writes go straight to the style, so this is one style
    // invalidation per frame for the length of the pulse.
    const start = performance.now();
    let frame = 0;

    const step = (now: number) => {
      const elapsed = now - start;
      if (elapsed >= PULSE_MS) {
        paint(INCIDENT_GLOW_OPACITY, INCIDENT_CORE_OPACITY);
        return;
      }
      const wave = (1 + Math.cos((2 * Math.PI * PULSE_COUNT * elapsed) / PULSE_MS)) / 2;
      const lift = wave * (1 - elapsed / PULSE_MS);
      paint(
        INCIDENT_GLOW_OPACITY + (INCIDENT_GLOW_PEAK - INCIDENT_GLOW_OPACITY) * lift,
        INCIDENT_CORE_OPACITY + (INCIDENT_CORE_PEAK - INCIDENT_CORE_OPACITY) * lift,
      );
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);

    return () => cancelAnimationFrame(frame);
  }, [ready, highlight]);

  // -- the live fleet ------------------------------------------------------ //
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;

    const pool = fleetMarkersRef.current;
    const placed = new Map<string, [number, number]>();

    for (const track of fleet) {
      const position = fleetPosition(track, nodeCoords, edgeGeometry);
      if (position) placed.set(track.vehicle_id, position);
    }

    // Moved, not rebuilt. A marker re-created on every poll would restart its
    // entrance fade each time and the whole fleet would blink in unison every
    // five seconds; `setLngLat` slides the existing one instead.
    for (const [vehicleId, marker] of pool) {
      const position = placed.get(vehicleId);
      if (!position) {
        marker.remove();
        pool.delete(vehicleId);
        continue;
      }
      marker.setLngLat(position);
    }

    for (const [vehicleId, position] of placed) {
      if (pool.has(vehicleId)) continue;

      // One element, not the wrapper-and-dot pair the stop markers use. That
      // pair exists so the entrance can scale without fighting MapLibre for
      // `transform`; a fleet dot fades in instead of scaling, so it needs the
      // split for nothing. See `.qgati-fleet-dot` in `globals.css`.
      const element = document.createElement("div");
      element.className = "qgati-fleet-dot";
      element.style.setProperty("--fleet-color", colorForVehicle(vehicleId));
      element.title = `${vehicleId} — last reported position`;

      pool.set(vehicleId, new Marker({ element, anchor: "center" }).setLngLat(position).addTo(map));
    }
  }, [ready, fleet, nodeCoords, edgeGeometry]);

  return (
    // The `data-*` attributes are test seams: they let an end-to-end check
    // assert on what the map actually received without reaching into WebGL.
    // `--qgati-redraw-ms` is what keeps the CSS transitions in step with the
    // layer fades, which are driven from JavaScript.
    <div
      className="relative h-full w-full"
      style={{ "--qgati-redraw-ms": `${REDRAW_MS}ms` } as CSSProperties}
      data-map-ready={String(ready)}
      data-active-routes={activeRoutes.length}
      data-node-coords={nodeCoords.size}
      data-ghost-routes={activeGhosts.length}
      data-picking={String(picking)}
    >
      <div ref={containerRef} className="h-full w-full" />
      {tileError ? (
        <div className="pointer-events-none absolute bottom-3 left-3 z-10 max-w-sm rounded-lg bg-ink/90 px-3 py-2 text-xs text-white">
          Basemap tiles unavailable — road network and routes are still drawn.
          <span className="mt-0.5 block text-2xs text-white/60">{tileError}</span>
        </div>
      ) : null}
    </div>
  );
}
