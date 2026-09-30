"""A simulated GPS fleet: the thing that produces fleet data in this project.

Everything in :mod:`qgati.traffic` consumes fleet telemetry — the log records
it, the detector judges it, the simulator lets a measured time displace its own
estimate. None of it produces any, because there is no fleet. This package is the
stand-in.

Two modules, split the way :mod:`qgati.traffic` splits:

:mod:`~qgati.fleet.pings`
    The rules of the simulation, and nothing else. Where a vehicle is, how long
    its road took this time, how a fleet is spread along its routes. Pure
    functions of their arguments: no threads, no clock, no database, and every
    question about the fleet's behaviour answerable without starting anything.
:mod:`~qgati.fleet.watcher`
    The wiring — a background thread per watched scenario, its lifecycle, and
    the one function a tick is.

**This is a simulation, not fleet integration.** There is no GPS device, no
telemetry protocol and no vehicle; the pings are generated from the same traffic
model the rest of the app prices with, plus noise. The noise is the part that
matters and it is not decoration: without it every reading of one road in one
condition band is the same number, which is the state the log is already in and
the reason its standard deviation is zero. *Simulation for demonstrating
dynamic routing, not real fleet telemetry* — the same caveat the traffic
simulator carries.

    >>> from qgati.fleet import ScenarioWatcher, initial_tracks
    >>> tracks = initial_tracks(scenario, solution, graph, weight)
    >>> watcher = ScenarioWatcher(scenario_id=..., store=..., graph=...,
    ...                           log_store=..., tracks=tracks, solver="qpso")
    >>> watcher.tick().readings          # one measurement per vehicle
"""

from qgati.fleet.pings import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_SEED,
    DEFAULT_TIME_SCALE,
    NOISE_SIGMA,
    CostOf,
    TrackPosition,
    VehicleTrack,
    blocked_edge,
    completed_stops,
    corridor,
    corridor_legs,
    cost_lookup,
    initial_tracks,
    noisy_reading,
    position,
    route_nodes,
)
from qgati.fleet.watcher import (
    Reading,
    ScenarioWatcher,
    TickResult,
    VehicleState,
    WatcherExists,
    WatcherRegistry,
    WatcherState,
)

__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "DEFAULT_SEED",
    "DEFAULT_TIME_SCALE",
    "NOISE_SIGMA",
    "CostOf",
    "Reading",
    "ScenarioWatcher",
    "TickResult",
    "TrackPosition",
    "VehicleState",
    "VehicleTrack",
    "WatcherExists",
    "WatcherRegistry",
    "WatcherState",
    "blocked_edge",
    "completed_stops",
    "corridor",
    "corridor_legs",
    "cost_lookup",
    "initial_tracks",
    "noisy_reading",
    "position",
    "route_nodes",
]
