"""Traffic modelling: a rule-based simulator, and a log of what it applied.

Six pieces, deliberately separate:

:mod:`~qgati.traffic.simulator`
    The rules. Time of day and reported incidents turn into an edge-weight
    callable that any router or the cost-matrix builder can use. Pure functions,
    no I/O, no database.
:mod:`~qgati.traffic.incidents`
    Reports that arrive *after* a scenario exists — a road shut or crawling —
    and how they fold onto the conditions it was created with. Pure, no I/O.
:mod:`~qgati.traffic.observations`
    Measurements, which outrank every report and every rule: a fleet vehicle's
    own time for a road it has driven. Tier 1 of the ranking the simulator
    applies. Pure, no I/O.
:mod:`~qgati.traffic.log_store`
    Where observations are kept — a SQLite table of road conditions.
:mod:`~qgati.traffic.detection`
    What reads that table back: per-road mean and standard deviation, and a
    z-score on each newly observed travel time. Plain statistics, no model.
:mod:`~qgati.traffic.recorder`
    The wiring: price a scenario under a set of conditions, then record the
    roads that touched.

**There is no machine learning here, on purpose.** The log exists so that a
future phase has a dataset to train on; this phase only produces the data and
judges new readings against simple statistics. The simulator's factors are
anchored to the TomTom Traffic Index 2025 figure for New Delhi, with the
road-class split a modelling choice on top — *simulation for demonstrating
dynamic routing, not real-world traffic data.*

    >>> from qgati.traffic import TrafficState, price_scenario
    >>> state = TrafficState.now()                  # system clock, no incidents
    >>> priced = price_scenario(graph, scenario, state, log_store)
    >>> priced.cost_matrix.matrix[0, 1]             # depot -> first stop, seconds
    >>> priced.cost_matrix.objective_matrix[0, 1]   # the same leg, in rupees
"""

from qgati.traffic.detection import (
    FALLBACK_FACTOR,
    INSUFFICIENT_SAMPLES,
    MIN_SAMPLES,
    Z_SCORE,
    Z_THRESHOLD,
    Baseline,
    Detection,
    RoadStats,
    build_baseline,
    detect,
    z_score,
)
from qgati.traffic.incidents import (
    CLEARED,
    CLOSURE,
    INCIDENT_MULTIPLIERS,
    INCIDENT_TYPES,
    SLOW,
    Incident,
    conditions_for,
)
from qgati.traffic.log_store import (
    DEFAULT_LOG_DB_PATH,
    TrafficLogRow,
    TrafficLogStore,
    road_id_of,
)
from qgati.traffic.observations import (
    GPS,
    Observation,
    observation_for,
    update_edge_from_observation,
)
from qgati.traffic.recorder import TrafficPricing, price_scenario
from qgati.traffic.simulator import (
    ACCIDENT,
    CONGESTION_FACTORS,
    MODERATE,
    MODERATE_FACTOR,
    MODERATE_WINDOWS,
    NORMAL,
    NORMAL_FACTOR,
    PEAK,
    PEAK_FACTOR,
    PEAK_WINDOWS,
    ROAD_CLOSURE,
    THROUGH_ROAD_CLASSES,
    ActiveConditions,
    Edge,
    TrafficState,
    congestion_multiplier,
    congestion_state,
    edge_of,
    get_traffic_multiplier,
    is_peak_hour,
    road_class_of,
    sensitivity_of,
    simulated_travel_time,
    traffic_weight_function,
)

__all__ = [
    "ACCIDENT",
    "ActiveConditions",
    "Baseline",
    "CLEARED",
    "CLOSURE",
    "CONGESTION_FACTORS",
    "DEFAULT_LOG_DB_PATH",
    "Detection",
    "Edge",
    "FALLBACK_FACTOR",
    "GPS",
    "INCIDENT_MULTIPLIERS",
    "INCIDENT_TYPES",
    "INSUFFICIENT_SAMPLES",
    "Incident",
    "MIN_SAMPLES",
    "MODERATE",
    "MODERATE_FACTOR",
    "MODERATE_WINDOWS",
    "NORMAL",
    "NORMAL_FACTOR",
    "Observation",
    "PEAK",
    "PEAK_FACTOR",
    "PEAK_WINDOWS",
    "ROAD_CLOSURE",
    "RoadStats",
    "SLOW",
    "THROUGH_ROAD_CLASSES",
    "TrafficLogRow",
    "TrafficLogStore",
    "TrafficPricing",
    "TrafficState",
    "Z_SCORE",
    "Z_THRESHOLD",
    "build_baseline",
    "conditions_for",
    "congestion_multiplier",
    "congestion_state",
    "detect",
    "edge_of",
    "get_traffic_multiplier",
    "is_peak_hour",
    "observation_for",
    "price_scenario",
    "road_class_of",
    "road_id_of",
    "sensitivity_of",
    "simulated_travel_time",
    "traffic_weight_function",
    "update_edge_from_observation",
    "z_score",
]
