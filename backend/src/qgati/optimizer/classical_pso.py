"""Classical (velocity-driven) Particle Swarm Optimization for the VRP.

This solver exists for one reason: to **isolate what QPSO's quantum behaviour
adds**. It is deliberately the same algorithm in every respect but the update
rule, so that a difference between the two rows of the benchmark table can only
be the quantum sampling term.

Held identical to :mod:`qgati.optimizer.qpso`
--------------------------------------------
* **Representation** — positions are random keys in ``[0, 1]``, decoded through
  the same :func:`~qgati.optimizer.decoding.decode_position`.
* **Cost function** — the shared :func:`~qgati.optimizer.fitness.evaluate`.
* **Budget** — ``num_particles`` and ``num_iterations`` default to QPSO's
  constants, *imported* rather than redefined. Restating 30 and 100 here would
  work right up until someone tuned one of them, at which point the comparison
  would silently stop being fair. Importing makes that drift impossible.
* **Initialisation** — the same ``rng.random((particles, n))`` draw.
* **Contraction schedule** — QPSO decays the quantum well width ``beta`` from
  1.0 to 0.5; this decays inertia ``w`` from 0.9 to 0.4. Both occupy the same
  slot in the update (how much of the previous state survives), so each is the
  natural counterpart of the other.

The update rule, and why it is the only difference
--------------------------------------------------
Classical PSO moves particles through a velocity that accumulates momentum::

    v = w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)
    x = x + v

QPSO instead samples the new position from a probability distribution centred on
an attractor, so a particle can *appear* far from where it was without travelling
there. That is the substantive difference: classical PSO can only reach a region
its accumulated velocity carries it to, whereas QPSO can tunnel out of a local
optimum in one step. On a rugged landscape such as the VRP, that is a real
distinction — and one this project should be able to *measure* rather than
assert, which is exactly what having both implementations buys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qgati.optimizer.decoding import decode_position
from qgati.optimizer.fitness import evaluate
from qgati.optimizer.models import Scenario, Solution
from qgati.optimizer.qpso import DEFAULT_NUM_ITERATIONS, DEFAULT_NUM_PARTICLES

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = [
    "DEFAULT_COGNITIVE",
    "DEFAULT_INERTIA_END",
    "DEFAULT_INERTIA_START",
    "DEFAULT_MAX_VELOCITY",
    "DEFAULT_SOCIAL",
    "run_classical_pso",
]

#: Inertia schedule, the counterpart of QPSO's ``beta`` 1.0 -> 0.5.
DEFAULT_INERTIA_START = 0.9
DEFAULT_INERTIA_END = 0.4

#: Cognitive (pull toward the particle's own best) and social (pull toward the
#: swarm's best) acceleration coefficients. The classical 2.0 / 2.0 pair.
DEFAULT_COGNITIVE = 2.0
DEFAULT_SOCIAL = 2.0

#: Positions live in [0, 1], so a velocity of 1.0 already spans the whole domain
#: in a single step. Clamping there stops the velocity accumulating without
#: bound, which is what makes an unclamped classical PSO diverge.
DEFAULT_MAX_VELOCITY = 1.0


def run_classical_pso(
    cost_matrix: CostMatrix,
    scenario: Scenario,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    seed: int | None = None,
    inertia_start: float = DEFAULT_INERTIA_START,
    inertia_end: float = DEFAULT_INERTIA_END,
    cognitive: float = DEFAULT_COGNITIVE,
    social: float = DEFAULT_SOCIAL,
    max_velocity: float = DEFAULT_MAX_VELOCITY,
) -> tuple[Solution, float, list[float]]:
    """Optimize a VRP instance with classical Particle Swarm Optimization.

    Parameters
    ----------
    cost_matrix, scenario
        A scenario and its precomputed costs, in the same order every other
        metaheuristic here takes them (see :func:`~qgati.optimizer.qpso.run_qpso`).
    num_particles, num_iterations
        Swarm size and generations. Default to QPSO's values by import.
    seed
        Fixes the run. The same seed reproduces the same result exactly.
    inertia_start, inertia_end
        Linear inertia decay, the counterpart of QPSO's contraction schedule.
    cognitive, social
        ``c1`` and ``c2`` — the pbest and gbest attraction weights.
    max_velocity
        Velocity clamp, in position units.

    Returns
    -------
    (best_solution, best_cost, convergence_history)
        Same shape as :func:`~qgati.optimizer.qpso.run_qpso`, so the benchmark
        runner can dispatch every metaheuristic uniformly.
        ``convergence_history[i]`` is the best-so-far cost after generation
        ``i``, is non-increasing, and has exactly ``num_iterations`` entries.
    """
    if num_particles < 1:
        raise ValueError("num_particles must be at least 1")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")
    if max_velocity <= 0:
        raise ValueError("max_velocity must be positive")

    n = scenario.n_deliveries
    rng = np.random.default_rng(seed)

    positions = rng.random((num_particles, n))
    velocities = np.zeros((num_particles, n))
    personal_best = positions.copy()
    personal_best_cost = np.full(num_particles, np.inf)

    def cost_of(position: np.ndarray) -> float:
        return evaluate(
            decode_position(position, scenario, cost_matrix), scenario, cost_matrix
        ).fitness

    global_best = positions[0].copy()
    global_best_cost = np.inf
    for i in range(num_particles):
        candidate = cost_of(positions[i])
        personal_best_cost[i] = candidate
        if candidate < global_best_cost:
            global_best_cost = candidate
            global_best = positions[i].copy()

    convergence: list[float] = [global_best_cost]

    for iteration in range(1, num_iterations):
        inertia = inertia_start + (inertia_end - inertia_start) * (
            iteration / (num_iterations - 1)
        )

        # Two independent draws per (particle, dimension), sampled as whole
        # arrays so the random stream stays reproducible for a given seed.
        r1 = rng.random((num_particles, n))
        r2 = rng.random((num_particles, n))

        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best - positions)
            + social * r2 * (global_best - positions)
        )
        np.clip(velocities, -max_velocity, max_velocity, out=velocities)
        positions = positions + velocities
        np.clip(positions, 0.0, 1.0, out=positions)

        for i in range(num_particles):
            candidate = cost_of(positions[i])
            if candidate < personal_best_cost[i]:
                personal_best_cost[i] = candidate
                personal_best[i] = positions[i].copy()
                if candidate < global_best_cost:
                    global_best_cost = candidate
                    global_best = positions[i].copy()

        convergence.append(global_best_cost)

    # Re-derive the returned solution from the winning position so the solution
    # and the reported cost can never disagree.
    best_solution = decode_position(global_best, scenario, cost_matrix)
    return best_solution, evaluate(best_solution, scenario, cost_matrix).fitness, convergence
