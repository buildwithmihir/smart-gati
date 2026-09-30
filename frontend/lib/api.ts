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
 * Costs are the backend's weighted objective, in **rupees** — time, distance and
 * fuel combined under the prices in `qgati.optimizer.objective`. That is not the
 * same thing as `travel_time`, which is seconds and comes back alongside it. The
 * two were the same number before distance and fuel joined the objective, so any
 * figure previously read from `travel_cost` as a duration is now a price.
 *
 * The second half of the file is the incident and re-optimization surface: a
 * simulated fleet that ticks, reports against a road, and re-plans what is left.
 * Two things about it are easy to get wrong and are worth stating here.
 *
 * - **`before` and `after` are the *unserved* work, not the whole round.** A
 *   re-optimization solves what the fleet has left, so its routes start where
 *   each vehicle actually is rather than at the depot. Comparing them is
 *   comparing two ways to serve one set of stops — which is what makes the
 *   comparison mean anything — but they are shorter than the dispatched routes
 *   the map drew a moment ago, and that is not a bug.
 * - **`POST /reoptimize` writes nothing.** It is a proposal: the network is
 *   priced as it stands and a plan is computed, but the fleet keeps driving the
 *   routes it was dispatched on. `POST .../avoid-road` and `POST .../incident`
 *   are the writes, and both are revertible.
 *
 * A third follows from the first two and is the one the dashboard is built
 * around: **`POST /reoptimize` needs a running fleet.** It derives where every
 * vehicle is, what it has already delivered and how much room it has left by
 * reading the watcher, so with no fleet there is nothing to read and the route
 * answers 404. That is why the dashboard dispatches a fleet as part of loading a
 * scenario rather than as an optional extra — and why the plan it draws comes
 * from that dispatch rather than from a separate `POST /optimize`.
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
  /** The traffic state the scenario's costs are built under, incidents included. */
  conditions: ConditionsOut;
  /** Rows written to the traffic log when the scenario was priced. */
  traffic_rows_logged: number;
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
  /** Weighted objective for this route, in rupees. */
  travel_cost: number;
  /** Seconds on the road, depot to depot. */
  travel_time: number;
  /** Metres driven. */
  distance_m: number;
  /** Litres burned, estimated from distance and average speed. */
  fuel_litres: number;
  /** GeoJSON `[lon, lat]` pairs, depot first and last. Empty for unused vehicles. */
  geometry: [number, number][];
};

export type OptimizeResponse = {
  scenario_id: string;
  solver: string;
  /** Human-readable solver name, e.g. "QPSO" — this is what the UI displays. */
  solver_name: string;
  seed: number | null;
  iterations: number | null;
  population: number | null;
  /** Fitness actually minimised: the weighted cost plus any constraint penalties. */
  cost: number;
  /** The weighted objective in rupees. Not seconds — see `travel_time`. */
  travel_cost: number;
  /** Total seconds on the road across all vehicles. */
  travel_time: number;
  /** Total metres driven. */
  distance_m: number;
  /** Total litres burned. */
  fuel_litres: number;
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

/** Solver seed for the dispatch and every re-optimization, fixed for the same reason. */
export const SAMPLE_SOLVER_SEED = 0;

/**
 * How long the simulated fleet waits between ticks, in wall-clock seconds.
 *
 * Well under the backend's 18 s default, because this fleet exists to be
 * *watched*: at 18 s a demo sits looking at a stopped map. Not so low that the
 * fleet races — see `SAMPLE_FLEET_TIME_SCALE`, which is what the ticking rate
 * actually buys.
 */
export const SAMPLE_FLEET_INTERVAL_SECONDS = 10;

/**
 * How much driving each tick represents, in seconds of travel per tick.
 *
 * Held at real time. A tick covers `interval × time_scale` = 10 s of driving, and
 * the sample scenario's routes run 10-20 minutes, so the fleet stays mid-route
 * for something like a hundred ticks. That matters more than it looks: a
 * re-optimization with nothing left to serve answers **409**, and a demo that
 * reaches that state because the fleet drove off the end of its routes has
 * failed for a reason nobody watching would guess. Raising this trades that
 * headroom for speed.
 */
export const SAMPLE_FLEET_TIME_SCALE = 1;

/** How often the dashboard re-reads the fleet's position while it is running. */
export const FLEET_POLL_MS = 5000;

/**
 * A failed request, with the status code kept.
 *
 * `detail` is FastAPI's own sentence and is what gets displayed — most failures
 * here are actionable (a scenario whose stops are mutually unreachable, a
 * closure that severed the instance) and the backend already writes them for a
 * human to read.
 *
 * `status` is kept because the incident flow has to *branch* on it rather than
 * merely report it. A re-optimization answers **409** when nothing has happened
 * that warrants one — an answer, not a failure — while **422** means the change
 * itself was refused. Those want different words on screen, and a status buried
 * in a message string cannot be told apart from a status that happens to appear
 * in one.
 *
 * Extends `Error`, so every existing `catch (cause) { cause instanceof Error }`
 * still works.
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // Non-JSON error body — keep the status line.
    }
    throw new ApiError(response.status, detail);
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
 * Solve a stored scenario.
 *
 * **Not used by the dashboard, and deliberately.** The dashboard gets its routes
 * from `startWatcher` instead, which solves the scenario with this same
 * production default on its way to putting a fleet on the road — one solve
 * rather than two, so the routes drawn are exactly the routes the vehicles are
 * driving rather than a second answer to the same question.
 *
 * Kept because it is part of the API's surface and the only way to ask for a
 * plan *without* also dispatching a fleet.
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
 *
 * The `LineString` features are also what makes a road **clickable**: each one
 * carries its endpoints as `u`/`v`, so a click that lands on a road can be
 * turned into the directed edge an incident names.
 */
export function fetchGraph(scenarioId: string): Promise<GraphGeoJSON> {
  return request<GraphGeoJSON>(`/graph/delhi?scenario_id=${encodeURIComponent(scenarioId)}`);
}

// --------------------------------------------------------------------------- //
// Incidents, the simulated fleet, and re-optimization
// --------------------------------------------------------------------------- //

/** One directed road, named by its endpoints. Direction matters throughout. */
export type Edge = {
  u: number;
  v: number;
};

/** The two words `POST /incident` accepts. */
export type IncidentKind = "closure" | "slow";

export type IncidentOut = {
  incident_id: string;
  incident_type: IncidentKind;
  edge: Edge;
  created_at: string;
};

export type ConditionsOut = {
  timestamp: string;
  peak_hour: boolean;
  traffic_condition: string;
  accident_edges: Edge[];
  closed_edges: Edge[];
  mutated: boolean;
  incidents: IncidentOut[];
};

/**
 * The outcome of applying or reverting one incident.
 *
 * Deliberately not a solution — an incident moves the *costs* and stops. Nothing
 * a client already holds is moved by it.
 *
 * `changed_legs` is the evidence it landed, and **zero is a legitimate answer**:
 * an incident on a road no cheapest path uses changes nothing about this
 * instance. The demo's panel reports that rather than hiding it, because
 * "the road you clicked is not on any route" is the single most useful thing to
 * say to somebody who has just clicked a road and seen nothing happen.
 */
export type IncidentResponse = {
  scenario_id: string;
  incident: IncidentOut;
  applied: boolean;
  conditions: ConditionsOut;
  changed_legs: number;
  traffic_rows_logged: number;
};

/** Why a re-optimization was allowed to run, and on what evidence. */
export type Trigger = {
  kinds: ("incident" | "anomaly" | "override")[];
  primary: "incident" | "anomaly" | "override";
  detail: string;
  edges: Edge[];
  /** The detector's own explanations, verbatim — they name the numbers judged. */
  reasons: string[];
};

export type Move = {
  delivery_id: string;
  from_vehicle: string;
  to_vehicle: string | null;
};

export type VehicleProgress = {
  vehicle_id: string;
  node: number | null;
  edge: Edge | null;
  completed: string[];
  remaining: string[];
  elapsed_seconds: number;
  remaining_capacity: number;
  stuck: boolean;
  finished: boolean;
  available: boolean;
};

/**
 * A re-optimized plan, and the before/after that shows what changed.
 *
 * **`before` and `after` cover the unserved stops only**, and they are two ways
 * of serving *the same set of them* — which is what makes the comparison mean
 * something. They start where each vehicle actually is, not at the depot, so
 * they are shorter than the dispatched routes the map drew a moment ago. That is
 * not a bug and the UI says so.
 *
 * **Both halves are priced with the incident in place.** So the difference
 * between them is what *re-planning* recovered, not what the incident cost —
 * the incident's delay is already inside both numbers. Reading the delta as the
 * incident's cost is the one way this panel can mislead, so it is labelled.
 */
export type ReoptimizeResponse = {
  scenario_id: string;
  solver: string;
  solver_name: string;
  seed: number | null;
  iterations: number | null;
  population: number | null;
  trigger: Trigger;
  vehicles: VehicleProgress[];
  replanned_vehicles: string[];
  incident: IncidentOut | null;
  before: Route[];
  after: Route[];
  completed: string[];
  replanned: string[];
  moved: Move[];
  cost: number;
  travel_cost: number;
  travel_time: number;
  distance_m: number;
  fuel_litres: number;
  feasible: boolean;
  runtime_ms: number;
  convergence: number[];
  explanation: Explanation | null;
};

/**
 * One number a reasoning statement was built from.
 *
 * `unit` is a bare word (`"rupees"`, `"seconds"`, `"stops"`, `"percent"`,
 * `"multiple"`, `"count"`) rather than a formatted string, so this client does
 * the formatting — the same split the rest of this file uses for figures.
 *
 * Prefixed rather than named `Figure` for the same reason as
 * `ExplanationStatement`: these are only meaningful inside an explanation, and a
 * bare name here would read as a general-purpose type.
 */
export type ExplanationFigure = {
  label: string;
  value: number;
  unit: string;
};

/**
 * One sentence of a decision trace, and the numbers it is made of.
 *
 * `figures` is the receipt for `text`: every number quoted in the sentence
 * appears here with its label and unit, so the panel can render the arithmetic
 * underneath the prose and a reader can check it. Nothing in `text` is
 * generated on this side — the sentence arrives whole.
 *
 * Named `ExplanationStatement` and not `Statement` because the DOM already has a
 * global `Statement`, and a shadowed global in a file that also touches the DOM
 * is a trap for whoever edits it next.
 */
export type ExplanationStatement = {
  text: string;
  figures: ExplanationFigure[];
};

/** Why one vehicle's route looks the way it does after a re-plan. */
export type RouteExplanation = {
  vehicle_id: string;
  headline: string;
  statements: ExplanationStatement[];
};

/** What one reported road costs now, against what it cost before the report. */
export type RoadReport = {
  u: string | number;
  v: string | number;
  statements: ExplanationStatement[];
};

/**
 * The whole decision trace for a re-optimization.
 *
 * `roads` is plan-level and `routes` is per-vehicle. That split is not cosmetic:
 * a road's delay is a fact about the road, identical for every vehicle, so
 * putting it inside each vehicle's panel would print the same sentence three
 * times and imply it had been computed three times.
 */
export type Explanation = {
  headline: string;
  roads: RoadReport[];
  routes: RouteExplanation[];
};

export type VehicleTrack = {
  vehicle_id: string;
  edge: Edge | null;
  remaining_seconds: number | null;
  progress: number;
  finished: boolean;
};

export type WatcherStatus = {
  scenario_id: string;
  solver: string;
  running: boolean;
  interval_seconds: number;
  time_scale: number;
  seed: number | null;
  ticks: number;
  started_at: string;
  last_tick_at: string | null;
  vehicles: VehicleTrack[];
};

export type WatcherReading = {
  vehicle_id: string;
  edge: Edge | null;
  modelled_travel_time: number | null;
  observed_travel_time: number | null;
  flag: boolean;
  rule: "z_score" | "insufficient_samples" | null;
  condition: string | null;
  sample_count: number;
  z_score: number | null;
  reason: string | null;
  note: string | null;
};

export type WatcherTick = {
  scenario_id: string;
  tick: number;
  at: string;
  readings: WatcherReading[];
  changed_legs: number;
  rows_logged: number;
  conditions_mutated: boolean;
};

/**
 * The fleet that was just started: its routes, and its first tick.
 *
 * Carries `routes` because **starting a watcher is what solves the scenario**,
 * so this response is the dashboard's source of the dispatched plan.
 */
export type WatcherStartResponse = WatcherStatus & {
  routes: Route[];
  first_tick: WatcherTick;
  /**
   * Best-so-far cost per iteration for the solve that produced `routes`.
   *
   * Present because starting a watcher *is* how the dashboard gets its first
   * plan — the routes above are a solve's output, and how that solve converged
   * is the same kind of fact. Defaults to `[]` for a solver that reports no
   * history, which is why every consumer treats it as possibly empty.
   */
  convergence: number[];
};

export type SolverCatalog = {
  default: string;
  default_iterations: number;
  default_population: number;
  solvers: { key: string; name: string; is_stochastic: boolean; is_exact: boolean }[];
};

/**
 * A plan as the dashboard displays it.
 *
 * `OptimizeResponse` minus what a watcher-dispatched plan cannot supply.
 * `feasible` is the one that matters: the backend decides it in `evaluate`
 * against capacity, coverage, fleet shape **and time windows**, and the
 * watcher's start response reports routes rather than a solution summary — so a
 * client re-deriving it would be guessing at the window rule and could disagree
 * with the server. It is `null` there and the UI shows an em dash rather than a
 * claim it cannot support. `runtime_ms` is unknown for the same reason: the
 * dispatch's solve is not timed anywhere the response reaches.
 *
 * `OptimizeResponse.cost` — the fitness actually minimised, which is the
 * weighted objective *plus constraint penalties* — is deliberately not here.
 * Nothing on the dashboard shows it, and for a dispatched plan it cannot be
 * recovered from the routes at all: summing them gives the weighted objective
 * and silently drops the penalties, which is a number that looks like the
 * fitness and is not. `travel_cost` is the figure the UI actually labels "Cost",
 * and it is summed honestly.
 */
export type DashboardPlan = {
  scenario_id: string;
  solver: string;
  solver_name: string;
  seed: number | null;
  iterations: number | null;
  population: number | null;
  travel_cost: number;
  travel_time: number;
  distance_m: number;
  fuel_litres: number;
  feasible: boolean | null;
  runtime_ms: number | null;
  routes: Route[];
  convergence: number[];
  /**
   * Why the plan looks the way it does — absent for a first dispatch.
   *
   * Only a re-plan can explain itself: a statement is built from the difference
   * between two plans, and the plan a fleet is first sent out on has nothing to
   * be different from. A dispatched plan therefore carries no explanation and
   * the panel says so, rather than showing an empty box.
   */
  explanation: Explanation | null;
};

/** Totals over a set of routes — the same sums `evaluate` reports. */
function totals(routes: Route[]) {
  return routes.reduce(
    (sum, route) => ({
      travel_cost: sum.travel_cost + route.travel_cost,
      travel_time: sum.travel_time + route.travel_time,
      distance_m: sum.distance_m + route.distance_m,
      fuel_litres: sum.fuel_litres + route.fuel_litres,
    }),
    { travel_cost: 0, travel_time: 0, distance_m: 0, fuel_litres: 0 },
  );
}

/**
 * The dispatched plan, from the watcher's start response.
 *
 * Every figure is a sum over the routes, which is what the backend's
 * `Evaluation` holds for the same plan — no number here is invented. The two
 * that cannot be summed are `feasible` and `runtime_ms`, and both are reported
 * as unknown rather than guessed; see `DashboardPlan`.
 */
export function planFromWatcherStart(
  started: WatcherStartResponse,
  catalog: SolverCatalog | null,
): DashboardPlan {
  const spec = catalog?.solvers.find((solver) => solver.key === started.solver);
  return {
    scenario_id: started.scenario_id,
    solver: started.solver,
    // The registry's own display name when the catalogue resolved, and the key
    // upper-cased when it did not — which is right for the production default
    // ("qpso" -> "QPSO") and merely legible for the rest.
    solver_name: spec?.name ?? started.solver.toUpperCase(),
    seed: started.seed,
    // Not sent on the start request, so the backend applied its own defaults.
    // The catalogue is where those are published, so the footer can show the
    // effort the dispatch actually used rather than leaving it blank.
    iterations: spec?.is_stochastic ? (catalog?.default_iterations ?? null) : null,
    population: spec?.is_stochastic ? (catalog?.default_population ?? null) : null,
    ...totals(started.routes),
    feasible: null,
    runtime_ms: null,
    routes: started.routes,
    convergence: started.convergence ?? [],
    // A dispatch has nothing to compare against, so it has no trace to give.
    explanation: null,
  };
}

/**
 * The re-planned plan, from the re-optimization's response.
 *
 * `after`, not `before` — the routes either side of the change are the report's
 * own business and are shown in its panel, not drawn on the map as the current
 * plan twice.
 */
export function planFromReopt(report: ReoptimizeResponse): DashboardPlan {
  return {
    scenario_id: report.scenario_id,
    solver: report.solver,
    solver_name: report.solver_name,
    seed: report.seed,
    iterations: report.iterations,
    population: report.population,
    travel_cost: report.travel_cost,
    travel_time: report.travel_time,
    distance_m: report.distance_m,
    fuel_litres: report.fuel_litres,
    feasible: report.feasible,
    runtime_ms: report.runtime_ms,
    routes: report.after,
    convergence: report.convergence,
    explanation: report.explanation ?? null,
  };
}

/** Every stop's total travel time across a set of routes — an ETA for the plan. */
export function planSeconds(routes: Route[]): number {
  return routes.reduce((sum, route) => sum + route.travel_time, 0);
}

/**
 * Whether a re-plan moved anything.
 *
 * Compares each vehicle's *ordered* stop sequence, which catches both things a
 * re-optimization can do: re-order one vehicle's route, and hand a stop from one
 * vehicle to another. Comparing only the set of stops would miss the first, and
 * the first is the common case for a driver-scoped re-plan.
 *
 * Stops are identified by `delivery_id` rather than by position, so the two
 * sides are comparable even though they are indexed differently.
 */
export function routesChanged(before: Route[], after: Route[]): boolean {
  const shape = (routes: Route[]) =>
    routes
      .filter((route) => route.stops.length > 0)
      .map((route) => `${route.vehicle_id}:${route.stops.map((s) => s.delivery_id).join(",")}`)
      .join("|");
  return shape(before) !== shape(after);
}

/**
 * Report one directed road as closed or slow.
 *
 * One call per **directed** edge. A two-way street is two edges on this graph
 * (Delhi's network is asymmetric, so the twins are not redundant), so closing a
 * street both ways is two of these — see `bothDirectionsOf` in `road-picking`.
 */
export function injectIncident(
  scenarioId: string,
  incidentType: IncidentKind,
  edge: Edge,
): Promise<IncidentResponse> {
  return request<IncidentResponse>(
    `/scenarios/${encodeURIComponent(scenarioId)}/incident`,
    {
      method: "POST",
      body: JSON.stringify({ incident_type: incidentType, edge }),
    },
  );
}

/** Revert one incident, restoring the conditions it replaced. */
export function clearIncident(
  scenarioId: string,
  incidentId: string,
): Promise<IncidentResponse> {
  return request<IncidentResponse>(
    `/scenarios/${encodeURIComponent(scenarioId)}/incident/${encodeURIComponent(incidentId)}`,
    { method: "DELETE" },
  );
}

/**
 * Dispatch a simulated fleet against a stored scenario.
 *
 * **This solves the scenario** — a fleet needs routes to drive and a stored
 * scenario has none until something solves it — and runs the first tick inline,
 * so the response already carries readings. It is also the only thing that makes
 * `POST /reoptimize` callable: that route reads the fleet's positions and
 * remaining stops, and without a fleet there is nothing for it to read.
 */
export function startWatcher(
  scenarioId: string,
  options: {
    seed?: number;
    intervalSeconds?: number;
    timeScale?: number;
    includeGeometry?: boolean;
  } = {},
): Promise<WatcherStartResponse> {
  return request<WatcherStartResponse>(
    `/scenarios/${encodeURIComponent(scenarioId)}/watcher`,
    {
      method: "POST",
      body: JSON.stringify({
        seed: options.seed ?? SAMPLE_SOLVER_SEED,
        interval_seconds: options.intervalSeconds ?? SAMPLE_FLEET_INTERVAL_SECONDS,
        time_scale: options.timeScale ?? SAMPLE_FLEET_TIME_SCALE,
        include_geometry: options.includeGeometry ?? true,
      }),
    },
  );
}

/** What the fleet is doing, and where each of its vehicles is. */
export function watcherStatus(scenarioId: string): Promise<WatcherStatus> {
  return request<WatcherStatus>(`/scenarios/${encodeURIComponent(scenarioId)}/watcher`);
}

/** Stop the fleet. The readings it took stay — they are already in the costs. */
export function stopWatcher(scenarioId: string): Promise<WatcherStatus> {
  return request<WatcherStatus>(`/scenarios/${encodeURIComponent(scenarioId)}/watcher`, {
    method: "DELETE",
  });
}

/**
 * Re-plan the unserved deliveries of a scenario whose fleet is out.
 *
 * **This is a read.** It computes a plan and returns it; the stored scenario is
 * untouched and the fleet keeps driving the routes it was already on. The
 * routes it returns are a *proposal*, which is what the map draws them as.
 *
 * Refusals are answers, not failures, and the caller is expected to read them:
 * **409** when nothing has happened that warrants re-planning (no live incident
 * and no flagged reading), **404** when no fleet is running, and **422** when
 * the remaining deliveries cannot be re-planned at all under current conditions.
 */
export function reoptimizeScenario(
  scenarioId: string,
): Promise<ReoptimizeResponse> {
  return request<ReoptimizeResponse>(
    `/scenarios/${encodeURIComponent(scenarioId)}/reoptimize`,
    { method: "POST", body: JSON.stringify({ seed: SAMPLE_SOLVER_SEED }) },
  );
}

/**
 * Every registered solver, and which one is the production default.
 *
 * Fetched for its key → display-name map, so the dispatched plan can be labelled
 * "QPSO" rather than "qpso" without the client hardcoding a registry it does not
 * own.
 */
export function fetchSolvers(): Promise<SolverCatalog> {
  return request<SolverCatalog>("/solvers");
}

// --------------------------------------------------------------------------- //
// Run history
// --------------------------------------------------------------------------- //

/**
 * Which solve a row records.
 *
 * `dispatch` is `POST /scenarios/{id}/watcher` — the solve a fleet is put on the
 * road with, and therefore the first row any demo produces. There is no row for
 * a refused solve: 409/422 return before the write, so this is a history of
 * *runs*, not of requests.
 */
export type RunKind = "dispatch" | "optimize" | "reoptimize" | "avoid_road";

/**
 * One recorded solve, as the history table stores it.
 *
 * Every figure is one the solve produced, captured where it was computed rather
 * than recomputed — `runtime_ms` is the same measured span, and the totals come
 * from the same evaluation. So a row is a receipt, and the only arithmetic this
 * client does is the ETA delta, which the backend also derives and sends.
 *
 * **`cost`, `travel_cost`, `travel_time`, `distance_m` and `fuel_litres` cover
 * different work from one `kind` to the next**, and comparing them across kinds
 * is the one way this table can mislead. For `optimize` and `dispatch` they are
 * the whole scenario; for `reoptimize` they are the unserved remainder; for
 * `avoid_road` they are the whole fleet's remainder after the reporting vehicle
 * was re-solved — deliberately not the figure at the top of that response, which
 * covers one vehicle. `n_deliveries` is always the *scenario's* size, so it
 * identifies the instance rather than describing the run's coverage.
 */
export type RunEntry = {
  id: number | null;
  /** ISO-8601 UTC, seconds precision. */
  timestamp: string;
  kind: RunKind;
  scenario_id: string;
  n_deliveries: number;
  n_vehicles: number;
  solver: string;
  solver_name: string;
  seed: number | null;
  iterations: number | null;
  population: number | null;
  /** Why a re-plan was allowed: `"incident"`, `"anomaly"` or `"override"`. */
  trigger: string | null;
  trigger_detail: string | null;
  /** The vehicle that reported, for an `avoid_road` run. */
  affected_vehicle: string | null;
  /**
   * The fitness actually minimised, constraint penalties included.
   *
   * `cost`, `travel_cost`, `travel_time`, `distance_m` and `fuel_litres` cover
   * **different work from one `kind` to the next**: the whole scenario for
   * `optimize`/`dispatch`, the unserved remainder for `reoptimize`, and the
   * whole fleet's remainder after the reporting vehicle was re-solved for
   * `avoid_road`. Reading down the Cost column across kinds compares unlike
   * things; `n_deliveries` is the scenario's size either way, so it says which
   * instance the run was against, never how much of it the run covered.
   */
  cost: number;
  /** The weighted objective in rupees — not seconds. */
  travel_cost: number;
  /** Total seconds on the road. Also the *new* ETA on a re-plan. */
  travel_time: number;
  distance_m: number;
  fuel_litres: number;
  feasible: boolean;
  runtime_ms: number;
  /** The baseline, on a re-plan only — `null` on every first answer. */
  old_eta_seconds: number | null;
  new_eta_seconds: number | null;
  /**
   * `old − new`, positive when the re-plan came out ahead.
   *
   * **Negative is a real answer** and is not clamped: a re-solve on a small
   * instance can land on a worse arrangement than the one in hand, and showing
   * that as a zero would hide the case an operator most wants to see.
   */
  eta_saved_seconds: number | null;
  /** Deliveries that changed hands in this re-plan. */
  moved: number;
};

/** One page of the run history, newest first. */
export type RunPage = {
  items: RunEntry[];
  /** Rows matching the filters across the whole table, not just this page. */
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

/** What re-planning has saved, over the re-plans that had a baseline. */
export type EtaSummary = {
  /** How many re-plans the averages below were computed over. */
  runs: number;
  avg_saved_seconds: number | null;
  total_saved_seconds: number | null;
  improved_runs: number;
  worsened_runs: number;
  unchanged_runs: number;
};

/**
 * Aggregate statistics over the whole run history.
 *
 * The averages are `null` and not `0.0` when the table is empty — SQL's `AVG`
 * over no rows is `NULL` and the API passes that through — so they must be
 * rendered as unknown rather than as a zero.
 */
export type AnalyticsSummary = {
  total_runs: number;
  first_run_at: string | null;
  last_run_at: string | null;
  runs_by_kind: Record<string, number>;
  runs_by_solver: Record<string, number>;
  feasible_runs: number;
  infeasible_runs: number;
  /** Runs a live incident or a flagged reading caused, of any kind. */
  incident_triggered_runs: number;
  avg_runtime_ms: number | null;
  min_runtime_ms: number | null;
  max_runtime_ms: number | null;
  avg_cost: number | null;
  avg_travel_cost: number | null;
  avg_travel_time_seconds: number | null;
  eta: EtaSummary;
};

/**
 * A page of past solves, newest first.
 *
 * `offset` rather than a cursor because the table only grows at the head: a new
 * run shifts every row's position by one, so a page held open across a solve can
 * repeat a row. That is acceptable here and is why the History tab refreshes on
 * demand rather than polling — a history that reorders itself under the reader
 * is worse than one that is a few seconds stale.
 */
export function fetchRuns(
  options: { limit?: number; offset?: number; kind?: RunKind; solver?: string } = {},
): Promise<RunPage> {
  const params = new URLSearchParams();
  if (options.limit !== undefined) params.set("limit", String(options.limit));
  if (options.offset !== undefined) params.set("offset", String(options.offset));
  if (options.kind) params.set("kind", options.kind);
  if (options.solver) params.set("solver", options.solver);
  const query = params.toString();
  return request<RunPage>(`/analytics/runs${query ? `?${query}` : ""}`);
}

/** Aggregate statistics over every recorded run — no parameters to give. */
export function fetchRunSummary(): Promise<AnalyticsSummary> {
  return request<AnalyticsSummary>("/analytics/summary");
}

// --------------------------------------------------------------------------- //
// The five-solver comparison
// --------------------------------------------------------------------------- //

/**
 * One solver's outcome in a comparison.
 *
 * **Every figure is `null` when `skipped` is set**, and that is the whole point of
 * the shape: a solver that did not run is *absent* from the comparison, not a
 * zero-cost infeasible answer. So a skipped row must render its `skipped`
 * sentence and dashes, never `₹0.00` — the backend nulls these fields precisely so
 * that mistake cannot be made by accident.
 *
 * `travel_cost` is the weighted objective in rupees; `travel_time`, `distance_m`
 * and `fuel_litres` are the raw quantities behind it. All four are carried
 * because a solver can buy seconds with extra kilometres, and only showing the
 * score would hide that it did.
 */
export type SolverResult = {
  /** The registry key — `"qpso"`, `"brute_force"`, … */
  solver: string;
  solver_name: string;
  /** The fitness actually minimised, constraint penalties included. */
  cost: number | null;
  /** The weighted objective in rupees — not seconds. */
  travel_cost: number | null;
  travel_time: number | null;
  distance_m: number | null;
  fuel_litres: number | null;
  feasible: boolean | null;
  runtime_ms: number | null;
  /** Percentage above the exact optimum; `null` when no exact solver ran. */
  gap_vs_optimal_pct: number | null;
  /** Percentage above the best cost any solver in this comparison found. */
  gap_vs_best_pct: number | null;
  /** Why this solver did not run, if it did not — e.g. brute force past its limit. */
  skipped: string | null;
};

/**
 * Every solver's result on one scenario.
 *
 * `optimal` is the *exact* optimum and is therefore only known when brute force
 * ran — which needs an instance of `MAX_EXACT_DELIVERIES` or fewer. The sample
 * scenario is larger than that, so on the ordinary demo path `optimal` is `null`
 * and `best_known` is the only reference available. The two are reported
 * separately rather than collapsed, because "0.4% above the best anyone found"
 * and "0.4% above proven optimal" are very different claims.
 */
export type CompareResponse = {
  scenario_id: string;
  /** The proven optimum, or `null` when no exact solver was able to run. */
  optimal: number | null;
  /** The best cost any solver in this comparison reached. */
  best_known: number | null;
  /** One row per registered solver, in the registry's own order. */
  results: SolverResult[];
};

/**
 * Run all five solvers over a stored scenario and compare them.
 *
 * Not the dashboard's `optimizeScenario`: that asks one solver for an answer,
 * where this asks every solver for one and returns the comparison. Slow by
 * construction — five solves, brute force among them — which is why nothing calls
 * it on mount.
 *
 * `seed` is passed through to the stochastic solvers so a comparison is
 * reproducible; leaving it out lets each solver pick its own, which makes two
 * runs of the same scenario differ for reasons that are not solver quality.
 * `iterations` and `population` fall to the backend's defaults when omitted, and
 * are sent as-is rather than defaulted here so the two cannot drift.
 */
export function fetchCompare(
  scenarioId: string,
  options: { seed?: number; iterations?: number; population?: number } = {},
): Promise<CompareResponse> {
  const params = new URLSearchParams();
  if (options.seed !== undefined) params.set("seed", String(options.seed));
  if (options.iterations !== undefined) params.set("iterations", String(options.iterations));
  if (options.population !== undefined) params.set("population", String(options.population));
  const query = params.toString();
  return request<CompareResponse>(
    `/optimize/${encodeURIComponent(scenarioId)}/compare${query ? `?${query}` : ""}`,
  );
}

