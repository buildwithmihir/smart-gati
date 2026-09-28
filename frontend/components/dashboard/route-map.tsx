"use client";

/**
 * The MapLibre canvas: base roads, one coloured line per vehicle, numbered stops.
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
 * The capability is built now, ahead of the incident simulator that will drive
 * it hardest: there, re-solving is interactive, and an update that flickers
 * reads as a bug rather than as an answer.
 */

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  GeoJSONSource,
  LngLatBounds,
  MapLibreMap,
  Marker,
  setWorkerUrl,
  type ErrorEvent as MapLibreErrorEvent,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import { isLineFeature, isPointFeature, type GraphGeoJSON, type Route } from "@/lib/api";
import { MARKER_STAGGER_CAP, MARKER_STAGGER_MS, REDRAW_MS } from "@/lib/motion";
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

/** Padding between the solution's bounding box and the edge of the viewport. */
const FRAME_PADDING = 72;

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
let cachedTokens: { road: string; basemap: string } | null = null;

function paletteTokens() {
  if (cachedTokens) return cachedTokens;
  const read = (name: string, fallback: string) => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  };
  cachedTokens = {
    road: read("--color-road", "#b5b3ac"),
    basemap: read("--color-basemap", "#fafaf8"),
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

type RouteMapProps = {
  graph: GraphGeoJSON | null;
  routes: Route[];
  depotNode: number | null;
};

export default function RouteMap({ graph, routes, depotNode }: RouteMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markersRef = useRef<Marker[]>([]);
  const routeIdsRef = useRef<string[]>([]);
  const retiringRef = useRef<Retiring | null>(null);
  /** Whether the camera has been aimed at a solution at least once. */
  const framedRef = useRef(false);
  /** Bumped on every redraw; see where the route ids are built. */
  const generationRef = useRef(0);
  const [ready, setReady] = useState(false);
  const [tileError, setTileError] = useState<string | null>(null);

  // Only vehicles that actually received work are drawn.
  const activeRoutes = useMemo(() => routes.filter((route) => route.stops.length > 0), [routes]);

  /** node id -> [lon, lat], from the graph's Point features. */
  const nodeCoords = useMemo(() => {
    const lookup = new Map<number, [number, number]>();
    if (!graph) return lookup;
    for (const feature of graph.features) {
      if (isPointFeature(feature)) lookup.set(feature.properties.id, feature.geometry.coordinates);
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
      // Basemap tiles come from a third party; if that fails the roads and
      // routes still draw, so this reports the degradation rather than
      // pretending the map is intact.
      setTileError(event.error?.message ?? "unknown tile error");
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
      routeIdsRef.current = [];
      framedRef.current = false;
      map.remove();
      mapRef.current = null;
      setReady(false);
    };
  }, []);

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
      map.addLayer({
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
      });
      map.addLayer({
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
      });
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
          // precisely what the incident simulator will invite them to do.
          map.fitBounds(bounds, { padding: FRAME_PADDING, duration: 500, maxZoom: 15 });
        }
      }
    }
  }, [ready, roads, activeRoutes, nodeCoords, depotNode]);

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
