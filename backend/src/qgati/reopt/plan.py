"""Build the re-optimization instance, and solve it with the same machinery.

The whole idea is in one sentence: **a completed stop cannot be reassigned,
because it is not in the instance to be solved.** A remaining-stops-only
:class:`~qgati.optimizer.models.Scenario` is built, each vehicle's reduced
capacity is carried onto its derived
:class:`~qgati.optimizer.models.Vehicle`, and each vehicle's current position
becomes its :attr:`~qgati.optimizer.models.Scenario.starts` entry. QPSO then runs
against it exactly as it runs against any other scenario — it is handed a scenario
and a cost matrix and knows nothing about any of this.

That is why completed stops are safe. Not because something filters them out of
an answer, but because there is no delivery in the instance whose id they carry,
so a solver cannot name one however badly it searches. The same trick as
:meth:`~qgati.optimizer.models.Scenario.__post_init__` refusing an infeasible
instance rather than letting a solver look bad: the guarantee is structural, so
it holds for every solver rather than for the ones somebody remembered to check.

What is deliberately *not* reused
---------------------------------
Nothing. The objective, the time windows, the capacity penalties, the decoder and
the split all come through
:func:`~qgati.optimizer.fitness.evaluate` and
:func:`~qgati.optimizer.registry.SolverSpec.__call__` unchanged. A re-optimized
plan and a plan solved from scratch are the same kind of object, produced the same
way, and the only difference between them is the scenario they were given.

The one approximation, stated plainly
-------------------------------------
A delivery's service window is quoted in seconds **from depot departure**, and a
vehicle that has been out for 900 seconds needs it shifted. The shift cannot be
exact: a window belongs to the delivery, but elapsed time belongs to whichever
vehicle serves it, and after re-optimization that may be a different vehicle than
the one that was 900 seconds in. This module shifts by the **smallest** elapsed
time among participating vehicles — the optimistic choice, which guarantees the
shift alone never turns a reachable window into an unreachable one. Any lateness
the other vehicles incur is found by the optimizer and priced by the objective,
which is where it belongs. It is a real approximation and it is stated here rather
than buried; note that this API cannot currently create a scenario with windows at
all, so it is core correctness rather than something a caller can observe yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Sequence

import networkx as nx

from qgati.graph.cost_matrix import CostMatrix, build_cost_matrix
from qgati.optimizer.fitness import Evaluation, evaluate
from qgati.optimizer.models import (
    CAPACITY_EPSILON,
    Delivery,
    Scenario,
    Solution,
    Vehicle,
)
from qgati.optimizer.registry import (
    DEFAULT_ITERATIONS,
    DEFAULT_POPULATION,
    DEFAULT_SOLVER_KEY,
    get_solver,
)
from qgati.reopt.state import VehicleProgress

__all__ = [
    "Move",
    "NothingToReplan",
    "ReoptPlan",
    "ReoptRefused",
    "SolverTooSmall",
    "before_view",
    "reoptimize",
]


class ReoptRefused(RuntimeError):
    """Base for a re-optimization that could not run, with a caller-facing reason.

    Distinct from an unexpected failure: every subclass here describes something
    the caller can fix or should know about, and the route maps each onto a status
    code rather than letting it surface as a 500.
    """


class NothingToReplan(ReoptRefused):
    """There is no remaining work, or no vehicle able to take it on."""


class SolverTooSmall(ReoptRefused):
    """An exact solver was named for a re-optimization larger than it can close."""


@dataclass(frozen=True, slots=True)
class Move:
    """One delivery that changed hands, or was dropped."""

    #: Position in the **original** scenario's delivery list.
    delivery: int
    from_vehicle: str
    #: ``None`` when the new plan does not serve it at all, which is a real
    #: answer — an infeasible plan — and one ``Evaluation.feasible`` reports.
    to_vehicle: str | None


@dataclass(frozen=True, slots=True)
class ReoptPlan:
    """A re-optimization: the instance it solved, and what came out.

    Carries the derived instance rather than only its solution, because a caller
    reporting the result needs both halves — ``solution``'s routes hold positions
    in ``instance.deliveries``, which mean nothing without the instance they index.
    """

    instance: Scenario
    matrix: CostMatrix
    solution: Solution
    evaluation: Evaluation
    #: Every vehicle, in scenario order, exactly as it was read.
    progress: tuple[VehicleProgress, ...]
    #: The subset of those that can still work, which is the derived fleet.
    active: tuple[VehicleProgress, ...]
    #: Original delivery indices that were in play, in derived-index order — so
    #: ``pool[k]`` is the original delivery that ``instance.deliveries[k]`` came
    #: from.
    pool: tuple[int, ...]
    moved: tuple[Move, ...]
    solver: str
    convergence: list[float]
    runtime_ms: float


def reoptimize(
    *,
    scenario: Scenario,
    progress: Sequence[VehicleProgress],
    graph: nx.Graph,
    weight: str | object = "weight",
    solver_key: str | None = None,
    seed: int | None = None,
    population: int = DEFAULT_POPULATION,
    iterations: int = DEFAULT_ITERATIONS,
) -> ReoptPlan:
    """Re-plan the unserved deliveries of ``scenario`` around ``progress``.

    Raises
    ------
    NothingToReplan
        If nothing is left to serve, or nothing can serve it.
    SolverTooSmall
        If an exact solver was named for a bigger instance than it can close.
    ValueError
        Propagated from :func:`~qgati.graph.cost_matrix.build_cost_matrix` when a
        vehicle's position cannot reach a remaining stop — a closure severing the
        network, which the route reports rather than serving as a 500.
    """
    pool = _pool(progress)
    if not pool:
        raise NothingToReplan(
            "every delivery has been served; there is nothing left to re-plan"
        )

    active = tuple(item for item in progress if item.available)
    if not active:
        raise NothingToReplan(
            f"{len(pool)} delivery(ies) remain but no vehicle can take them: "
            "every one is either finished or stopped behind a closure"
        )

    # Checked explicitly rather than left to `Scenario.__post_init__`, which would
    # catch it with a message about total demand and leave the caller to work out
    # why a scenario that was feasible a moment ago is not. The arithmetic is the
    # reason: the pool is every vehicle's undelivered load, but the capacity is
    # only the *available* vehicles' — a vehicle stopped behind a closure carries
    # load that no one else has room for, and that is a real answer, not a bug.
    capacity = sum(item.remaining_capacity for item in active)
    demand = sum(scenario.deliveries[stop].demand for stop in pool)
    if demand > capacity + CAPACITY_EPSILON:
        raise NothingToReplan(
            f"the {len(pool)} unserved delivery(ies) total {demand:g} of demand, but "
            f"the {len(active)} vehicle(s) that can still work have {capacity:g} of "
            "capacity left between them: a vehicle stopped behind a closure is "
            "carrying load no one else has room for"
        )

    instance = _derive(scenario, progress, pool, active)

    spec = get_solver(solver_key or DEFAULT_SOLVER_KEY)
    if spec.is_exact and instance.n_deliveries > (spec.exact_limit or 0):
        raise SolverTooSmall(
            f"{spec.name} is limited to {spec.exact_limit} deliveries; "
            f"{instance.n_deliveries} remain unserved. Use a heuristic."
        )

    # The same weight the scenario is currently priced under — the incident and
    # the measurements that triggered this are in it, so the new plan is solved
    # against the conditions it is reacting to rather than the ones that produced
    # the plan it is replacing.
    matrix = build_cost_matrix(graph, instance, weight=weight)

    start = time.perf_counter()
    solution, _cost, convergence = spec(
        matrix,
        instance,
        seed=seed,
        population=population,
        iterations=iterations,
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0

    return ReoptPlan(
        instance=instance,
        matrix=matrix,
        solution=solution,
        evaluation=evaluate(solution, instance, matrix),
        progress=tuple(progress),
        active=active,
        pool=pool,
        moved=_moves(progress, pool, instance, solution),
        solver=spec.key,
        convergence=convergence,
        runtime_ms=runtime_ms,
    )


def _pool(progress: Sequence[VehicleProgress]) -> tuple[int, ...]:
    """Every still-unserved delivery, in scenario then route order.

    Deterministic on purpose: two calls against an unchanged fleet build the same
    instance with its deliveries in the same positions, so the same seed gives
    the same plan and a diff between two runs means something.
    """
    return tuple(
        stop for item in progress for stop in item.remaining
    )


def _derive(
    scenario: Scenario,
    progress: Sequence[VehicleProgress],
    pool: tuple[int, ...],
    active: tuple[VehicleProgress, ...],
) -> Scenario:
    """The remaining-stops-only instance, with reduced capacities and real starts.

    Three substitutions, and each one is the entire content of the brief's
    "modified starting conditions":

    * the deliveries are the unserved ones, re-indexed, their original ids kept so
      a result can be mapped back;
    * the fleet is the vehicles that can still work, each capped at what it has
      left to give rather than what it set out with;
    * each of those vehicles begins where it currently is.
    """
    # One shift for every delivery, because a window belongs to the delivery and
    # not to a vehicle — see the module docstring for why this is the smallest
    # elapsed and why it is an approximation.
    shift = min(item.elapsed for item in active)
    deliveries = tuple(_rebased(scenario.deliveries[stop], shift) for stop in pool)
    vehicles = tuple(
        Vehicle(
            id=item.vehicle_id,
            # A vehicle whose remaining stops all weigh nothing has nothing left
            # to give, and `Vehicle` refuses a non-positive capacity outright.
            # The epsilon keeps it representable without inventing room it does
            # not have: the capacity check carries the same slack, so only
            # zero-demand work can use it.
            capacity=max(item.remaining_capacity, CAPACITY_EPSILON),
        )
        for item in active
    )
    return Scenario(
        depot=scenario.depot,
        deliveries=deliveries,
        vehicles=vehicles,
        starts=tuple(item.node for item in active),
    )


def before_view(
    scenario: Scenario,
    progress: Sequence[VehicleProgress],
    graph: nx.Graph,
    weight: str | object = "weight",
) -> tuple[Scenario, CostMatrix]:
    """The original scenario, re-priced with each vehicle starting where it is.

    The *before* half of a re-optimization is the rest of the plan the fleet is
    already driving, and it has to be measured on the same footing as the new
    plan or the comparison is between two pricings rather than two plans. Pricing
    the old routes from the depot would credit them with a departure none of these
    vehicles is anywhere near, so the same treatment is applied to both halves:
    the scenario gains the fleet's current positions as per-vehicle starts.

    Returned as ``(scenario, matrix)`` because a caller resolving route indices
    back into ids needs the scenario the routes index, and the matrix is the one
    they were priced on. Neither is stored anywhere.

    A vehicle with no known position — one stopped behind a closure — is quoted
    against the depot, because where it is is precisely the thing that is unknown.
    That is the one case this is approximate, it can only arise for a vehicle
    whose undelivered work is in the pool being reassigned anyway, and a finished
    vehicle is unaffected since it has no remaining route to price.
    """
    starts = tuple(
        scenario.depot.node if item.node is None else item.node for item in progress
    )
    priced = replace(scenario, starts=starts)
    return priced, build_cost_matrix(graph, priced, weight=weight)


def _rebased(delivery: Delivery, shift: float) -> Delivery:
    """One delivery's window moved to the vehicle's clock, or left absent.

    Clamped at zero rather than allowed to go negative: a window that closed
    while the vehicle was driving is one the re-plan is already late for, and the
    objective should price that lateness rather than the model refusing to
    represent it. Order is preserved by the subtraction, so a shift can turn a
    window into an immediate one but never into an empty one.
    """
    if not delivery.has_window:
        return delivery

    def moved(value: float | None) -> float | None:
        return None if value is None else max(value - shift, 0.0)

    return replace(
        delivery,
        earliest_arrival=moved(delivery.earliest_arrival),
        latest_arrival=moved(delivery.latest_arrival),
    )


def _moves(
    progress: Sequence[VehicleProgress],
    pool: tuple[int, ...],
    instance: Scenario,
    solution: Solution,
) -> tuple[Move, ...]:
    """Which deliveries changed vehicle, comparing the old plan with the new.

    Read off the two assignments rather than inferred from the cost, because this
    is the claim the feature makes — "assignments can change, and completed stops
    never do" — and a claim like that should be computed from the assignments
    themselves.
    """
    before = {stop: item.vehicle_id for item in progress for stop in item.remaining}
    after: dict[int, str] = {}
    for position, route in enumerate(solution.routes):
        if position >= instance.n_vehicles:  # pragma: no cover - Solution pads to the fleet
            continue
        served_by = instance.vehicles[position].id
        for stop in route:
            after[pool[stop]] = served_by

    return tuple(
        Move(delivery=stop, from_vehicle=before[stop], to_vehicle=after.get(stop))
        for stop in pool
        if after.get(stop) != before[stop]
    )
