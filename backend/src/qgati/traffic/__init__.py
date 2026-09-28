"""Traffic modelling: a rule-based simulator, and a log of what it applied.

Three pieces, deliberately separate:

:mod:`~qgati.traffic.simulator`
    The rules. Peak hour, rain, accidents and closures turn into an edge-weight
    callable that any router or the cost-matrix builder can use. Pure functions,
    no I/O, no database.
:mod:`~qgati.traffic.log_store`
    Where observations are kept — a SQLite table of road conditions.
:mod:`~qgati.traffic.recorder`
    The wiring: price a scenario under a set of conditions, then record the
    roads that touched.

**There is no machine learning here, on purpose.** The log exists so that a
future phase has a dataset to train on; this phase only produces the data. The
simulator's factors are plausible round numbers chosen to make dynamic routing
demonstrable — *simulation for demonstrating dynamic routing, not real-world
traffic data.*

    >>> from qgati.traffic import TrafficState, price_scenario
    >>> state = TrafficState.now()                  # system clock, clear, no incidents
    >>> priced = price_scenario(graph, scenario, state, log_store)
    >>> priced.cost_matrix.matrix[0, 1]             # depot -> first stop, priced
"""

from qgati.traffic.log_store import (
    DEFAULT_LOG_DB_PATH,
    TrafficLogRow,
    TrafficLogStore,
    road_id_of,
)
from qgati.traffic.recorder import TrafficPricing, price_scenario
from qgati.traffic.simulator import (
    ACCIDENT,
    ACCIDENT_FACTOR,
    PEAK_FACTOR,
    PEAK_WINDOWS,
    RAIN_FACTOR,
    ROAD_CLOSURE,
    THROUGH_ROAD_CLASSES,
    ActiveConditions,
    Edge,
    TrafficState,
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
    "ACCIDENT_FACTOR",
    "DEFAULT_LOG_DB_PATH",
    "PEAK_FACTOR",
    "PEAK_WINDOWS",
    "RAIN_FACTOR",
    "ROAD_CLOSURE",
    "THROUGH_ROAD_CLASSES",
    "ActiveConditions",
    "Edge",
    "TrafficLogRow",
    "TrafficLogStore",
    "TrafficPricing",
    "TrafficState",
    "edge_of",
    "get_traffic_multiplier",
    "is_peak_hour",
    "price_scenario",
    "road_class_of",
    "road_id_of",
    "sensitivity_of",
    "simulated_travel_time",
    "traffic_weight_function",
]
