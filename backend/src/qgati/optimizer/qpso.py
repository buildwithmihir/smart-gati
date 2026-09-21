"""Quantum-behaved Particle Swarm Optimization for the VRP.

This is the project's headline solver. It is the genuine quantum-inspired
algorithm (Sun et al., 2004) — particles are attracted to a quantum potential
well and their positions sampled from a probability distribution — not classical
velocity-driven PSO wearing a different name. The classical variant lives beside
it in :mod:`qgati.optimizer.classical_pso` precisely so the two can be compared.

Representation
--------------
The encoding this solver searches over — random keys decoded through an optimal
capacity split — is documented in :mod:`qgati.optimizer.decoding`, together with
why it is chosen and why every metaheuristic here shares it. The short version:
positions stay continuous so the quantum update applies unchanged, and capacity
holds by construction so the swarm optimises pure travel cost.

Reference
---------
Sun, Feng, Xu (2004), *Particle swarm optimization with particles having quantum
behavior*, IEEE CEC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qgati.optimizer.decoding import decode_position, optimal_split
from qgati.optimizer.fitness import evaluate
from qgati.optimizer.models import Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = [
    "DEFAULT_BETA_END",
    "DEFAULT_BETA_START",
    "DEFAULT_NUM_ITERATIONS",
    "DEFAULT_NUM_PARTICLES",
    "decode_position",
    "optimal_split",
    "run_qpso",
]

DEFAULT_NUM_PARTICLES = 30
DEFAULT_NUM_ITERATIONS = 100

#: Contraction-expansion coefficient schedule. Standard practice is to decay
#: beta linearly from 1.0 to 0.5: large early displacement to explore, small
#: late displacement to converge. Beta controls the width of the quantum
#: potential well, and therefore how far a particle may tunnel from its
#: attractor.
DEFAULT_BETA_START = 1.0
DEFAULT_BETA_END = 0.5

#: Floor for the uniform sample feeding ``ln(1/u)``. u must stay strictly
#: positive or the logarithm diverges.
_U_FLOOR = 1e-12


def run_qpso(
    cost_matrix: CostMatrix,
    scenario: Scenario,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    seed: int | None = None,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
) -> tuple[Solution, float, list[float]]:
    """Optimize a VRP instance with Quantum-behaved Particle Swarm Optimization.

    The update, per particle per dimension, is the quantum one. With ``mbest``
    the mean of all personal bests and ``p`` an attractor drawn between the
    particle's own best and the swarm's best::

        p = phi * pbest + (1 - phi) * gbest          phi ~ U(0,1)
        x = p ± beta * |mbest - x| * ln(1/u)         u   ~ U(0,1)

    The sign is chosen at random and ``ln(1/u)`` is the quantum sampling term
    that lets a particle appear anywhere in the well rather than travelling
    there — the behaviour that distinguishes QPSO from classical PSO. ``beta``
    is the contraction-expansion coefficient, decayed linearly from
    ``beta_start`` to ``beta_end`` over the run.

    Parameters
    ----------
    cost_matrix, scenario
        A scenario and its precomputed costs. Note this argument order is the
        reverse of :func:`~qgati.optimizer.brute_force.solve_brute_force` and
        :func:`~qgati.optimizer.savings.clarke_wright_savings`, which take
        ``(scenario, cost_matrix)``. All four metaheuristics take
        ``(cost_matrix, scenario)``, so a benchmark can dispatch them uniformly.
    num_particles, num_iterations
        Swarm size and generations.
    seed
        Fixes the run. The same seed reproduces the same result exactly.
    beta_start, beta_end
        Contraction-expansion schedule.

    Returns
    -------
    (best_solution, best_cost, convergence_history)
        ``best_cost`` is the fitness actually minimised — travel cost plus any
        constraint penalties — and equals the travel cost whenever the solution
        is feasible, which the optimal-split decoder makes the normal case.
        ``convergence_history[i]`` is the best-so-far cost after generation
        ``i``; it is non-increasing and has exactly ``num_iterations`` entries.
    """
    if num_particles < 1:
        raise ValueError("num_particles must be at least 1")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    n = scenario.n_deliveries
    rng = np.random.default_rng(seed)

    # Random keys, uniform in [0, 1). Clipped back into that range after every
    # move: only the ordering of the keys carries meaning, so letting them drift
    # unboundedly would add nothing but numerical risk.
    positions = rng.random((num_particles, n))
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
        beta = beta_start + (beta_end - beta_start) * (iteration / (num_iterations - 1))

        # Mean of all personal bests — the centre of the quantum potential well
        # the whole swarm is attracted to.
        mean_best = personal_best.mean(axis=0)

        # Vectorised across the entire swarm at once. The three independent
        # draws per (particle, dimension) are sampled as whole arrays so the
        # random stream stays reproducible for a given seed.
        phi = rng.random((num_particles, n))
        u = 1.0 - rng.random((num_particles, n))  # (0, 1], so ln(1/u) >= 0
        positive = rng.random((num_particles, n)) < 0.5

        attractor = phi * personal_best + (1.0 - phi) * global_best
        displacement = (
            beta * np.abs(mean_best - positions) * np.log(1.0 / np.maximum(u, _U_FLOOR))
        )
        positions = np.where(positive, attractor + displacement, attractor - displacement)
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
    best_cost = evaluate(best_solution, scenario, cost_matrix).fitness
    return best_solution, best_cost, convergence
