/**
 * Typed client for the Q-Gati FastAPI backend.
 *
 * Every type below was captured from the running service rather than inferred
 * from the docs — the shapes mirror `backend/src/qgati/api/schemas.py` as
 * actually serialised. Two details worth knowing, both verified against a live
 * response:
 *
 * - `OptimizeResponse.routes` always has one entry per vehicle, and an *unused*
 *   vehicle comes back with `stops: []` and `geometry: []`. So a route's
 *   emptiness — not its presence — is what marks a vehicle as unused.
 * - `Route.geometry` is a GeoJSON coordinate array in **lon, lat** order, as is
 *   everything else from the graph endpoints. MapLibre also expects lon, lat,
 *   so these pass through unchanged; do not "fix" the order.
 *
 * Costs are travel **time in seconds**, not distance: the graph's `weight`
 * mirrors `travel_time` (see `backend/README.md`).
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Depot = {
  node: number;
  lat: number;
  lon: number;
};

export type ScenarioDelivery = {
  id: string;
  node: number;
  demand: number;
};

export type ScenarioVehicle = {
  id: string;
  capacity: number;
};

export type ScenarioResponse = {
  scenario_id: string;
  depot: Depot;
  deliveries: ScenarioDelivery[];
  vehicles: ScenarioVehicle[];
  total_demand: number;
  total_capacity: number;
  n_deliveries: number;
  n_vehicles: number;
  exactly_solvable: boolean;
};

/** A single stop on a solved route. Carries a node id, not coordinates. */
export type RouteStop = {
  delivery_id: string;
  node: number;
};

export type Route = {
  vehicle_id: string;
  stops: RouteStop[];
  load: number;
  capacity: number;
  travel_cost: number;
  /** GeoJSON `[lon, lat]` pairs, depot first and last. Empty for unused vehicles. */
  geometry: [number, number][];
};

export type OptimizeResponse = {
  scenario_id: string;
  solver: string;
  /** Human-readable solver name, e.g. "ACO" — this is what the UI displays. */
  solver_name: string;
  seed: number | null;
  iterations: number | null;
  population: number | null;
  /** Fitness actually minimised: travel time plus any constraint penalties. */
  cost: number;
  travel_cost: number;
  feasible: boolean;
  runtime_ms: number;
  routes: Route[];
  /** Best-so-far cost per generation; empty for deterministic solvers. */
  convergence: number[];
};

export type GraphLineFeature = {
  type: "Feature";
  properties: { u: number; v: number; travel_time: number };
  geometry: { type: "LineString"; coordinates: [number, number][] };
};

export type GraphPointFeature = {
  type: "Feature";
  properties: { id: number };
  geometry: { type: "Point"; coordinates: [number, number] };
};

export type GraphFeature = GraphLineFeature | GraphPointFeature;

export type GraphGeoJSON = {
  type: "FeatureCollection";
  features: GraphFeature[];
};

/**
 * Type guards for the two feature kinds.
 *
 * TypeScript will not narrow this union from `feature.geometry.type` alone,
 * because the discriminant sits a level deeper than it follows, so the checks
 * are written out as predicates. They also read better than the inline
 * comparison at each call site, since this union is discriminated on a nested
 * field and every access needs the narrowing.
 */
export function isLineFeature(feature: GraphFeature): feature is GraphLineFeature {
  return feature.geometry.type === "LineString";
}

export function isPointFeature(feature: GraphFeature): feature is GraphPointFeature {
  return feature.geometry.type === "Point";
}

/**
 * The hardcoded sample the "Load Sample Scenario" button posts.
 *
 * `seed` is fixed so the same scenario comes back every time — a demo that
 * reshuffles its stops on every click is hard to reason about.
 */
export const SAMPLE_SCENARIO_PAYLOAD = {
  kind: "generate",
  n_deliveries: 12,
  n_vehicles: 3,
  seed: 21,
} as const;

/** Solver seed for `/optimize`, also fixed so results are reproducible. */
export const SAMPLE_SOLVER_SEED = 0;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!response.ok) {
    // FastAPI puts a human-readable reason in `detail`; surface it rather than
    // a bare status code, since most failures here are actionable (e.g. a
    // scenario whose stops are mutually unreachable).
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // Non-JSON error body — keep the status line.
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export function createScenario(payload: Record<string, unknown>): Promise<ScenarioResponse> {
  return request<ScenarioResponse>("/scenarios", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Solve a stored scenario. No solver name is sent, so the backend applies its
 * production default — QPSO — which is what the UI then reports back.
 */
export function optimizeScenario(scenarioId: string): Promise<OptimizeResponse> {
  return request<OptimizeResponse>(`/optimize/${scenarioId}`, {
    method: "POST",
    body: JSON.stringify({ seed: SAMPLE_SOLVER_SEED }),
  });
}

/**
 * The road network as GeoJSON, scoped to the scenario's bounding box.
 *
 * Scoping matters: unscoped this is the whole ~1 MB Delhi extract. Scoped it is
 * still sizeable, so the response is used for two things at once — the `Point`
 * features become the node→coordinate lookup that lets stops be placed on the
 * map (the optimize response identifies stops by node id only), and the
 * `LineString` features are drawn as the base road layer.
 */
export function fetchGraph(scenarioId: string): Promise<GraphGeoJSON> {
  return request<GraphGeoJSON>(`/graph/delhi?scenario_id=${encodeURIComponent(scenarioId)}`);
}
