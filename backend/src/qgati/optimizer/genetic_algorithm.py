"""Genetic Algorithm for the capacitated VRP.

The classical evolutionary baseline: a population of candidate routes evolved by
selection, crossover and mutation. Chromosomes are *permutations* of the
delivery indices and are decoded through the same optimal capacity split every
other solver here uses (:mod:`qgati.optimizer.decoding`), so capacity holds by
construction and the GA optimises pure travel cost like the swarms do.

Why permutations rather than QPSO's random keys — and what it costs
------------------------------------------------------------------
The brief asked for the same representation as QPSO "where practical". Random
keys are the wrong encoding for a GA: the crossover operator that makes a GA work
(order crossover) is defined on permutations, and applying it to real-valued keys
would splice two *orderings* together in a way that has no meaning. So the
chromosome is the permutation itself.

The comparison stays fair because the permutation is exactly what QPSO's random
keys *decode to*: both search the same space of orderings, and both are scored
after the same optimal split. Only the search strategy differs. What the GA gains
in encoding naturalness it loses in locality — a random-key position moves
smoothly, whereas OX1 can relocate half a tour at once. That trade is the
interesting part of the benchmark, not a flaw in it.

Operators
---------
* **Selection** — k-way tournament. Chosen over fitness-proportionate
  roulette because costs here are large and tightly clustered (a 5% spread on a
  10^4-second instance), which makes roulette nearly uniform and destroys
  selection pressure.
* **Crossover** — order crossover (OX1): copy a contiguous segment from one
  parent, fill the remaining positions with the other parent's genes in order.
  This is the standard operator for permutation problems because it preserves
  relative order while never producing an invalid chromosome.
* **Mutation** — swap two positions. Applied per-gene at rate ``1/n`` by
  default, i.e. about one swap per child regardless of instance size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from qgati.optimizer.decoding import decode_permutation
from qgati.optimizer.fitness import evaluate
from qgati.optimizer.models import Scenario, Solution

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = [
    "DEFAULT_CROSSOVER_RATE",
    "DEFAULT_ELITES",
    "DEFAULT_NUM_GENERATIONS",
    "DEFAULT_POPULATION_SIZE",
    "DEFAULT_TOURNAMENT_SIZE",
    "order_crossover",
    "run_genetic_algorithm",
    "swap_mutation",
]

#: Matches QPSO's 30 particles / 100 iterations. The benchmark's claim is that
#: the four metaheuristics are compared at equal budget, so these move together.
DEFAULT_POPULATION_SIZE = 30
DEFAULT_NUM_GENERATIONS = 100

DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_CROSSOVER_RATE = 0.9

#: One elite survives each generation untouched. Without it the best solution
#: found can be lost to a bad draw of children, which makes the convergence
#: history non-monotonic and the benchmark noisier than the search deserves.
DEFAULT_ELITES = 1


def order_crossover(
    parent_a: Sequence[int], parent_b: Sequence[int], rng: np.random.Generator
) -> list[int]:
    """OX1: splice a segment of ``parent_a`` into ``parent_b``'s ordering.

    A contiguous segment of ``parent_a`` is copied verbatim, then the remaining
    positions — walking forward from the segment's end and wrapping — are filled
    with the genes of ``parent_b`` in the order they appear there, skipping any
    already inherited. The result is always a valid permutation.
    """
    size = len(parent_a)
    if size < 2:  # pragma: no cover - Scenario forbids fewer than 1 delivery
        return list(parent_a)

    left, right = sorted(int(bound) for bound in rng.choice(size, size=2, replace=False))
    child: list[int] = [-1] * size
    child[left : right + 1] = parent_a[left : right + 1]

    inherited = set(parent_a[left : right + 1])
    donor = [gene for gene in parent_b if gene not in inherited]
    for offset, gene in enumerate(donor):
        child[(right + 1 + offset) % size] = gene
    return child


def swap_mutation(
    chromosome: list[int], rng: np.random.Generator, rate: float
) -> None:
    """Swap each gene with a random other position, independently, at ``rate``.

    Mutates ``chromosome`` in place. Expected swaps per call is ``n · rate``, so
    the ``1/n`` default yields roughly one swap per child at any instance size.
    """
    size = len(chromosome)
    if size < 2:  # pragma: no cover - Scenario forbids fewer than 1 delivery
        return
    for index in range(size):
        if rng.random() >= rate:
            continue
        other = int(rng.integers(0, size - 1))
        if other >= index:
            other += 1  # skip over `index`, so a gene never swaps with itself
        chromosome[index], chromosome[other] = chromosome[other], chromosome[index]


def _tournament(
    population: list[list[int]], costs: list[float], size: int, rng: np.random.Generator
) -> list[int]:
    """Return a copy of the fittest chromosome among ``size`` drawn at random."""
    contenders = rng.choice(len(population), size=size, replace=False)
    winner = int(contenders[0])
    for contender in contenders[1:]:
        candidate = int(contender)
        if costs[candidate] < costs[winner]:
            winner = candidate
    return list(population[winner])


def run_genetic_algorithm(
    cost_matrix: CostMatrix,
    scenario: Scenario,
    population_size: int = DEFAULT_POPULATION_SIZE,
    num_generations: int = DEFAULT_NUM_GENERATIONS,
    seed: int | None = None,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float | None = None,
    elites: int = DEFAULT_ELITES,
) -> tuple[Solution, float, list[float]]:
    """Optimize a VRP instance with a Genetic Algorithm.

    Parameters
    ----------
    cost_matrix, scenario
        A scenario and its precomputed costs, in the same order every other
        metaheuristic here takes them (see :func:`~qgati.optimizer.qpso.run_qpso`).
    population_size, num_generations
        Population and generation count.
    seed
        Fixes the run. The same seed reproduces the same result exactly.
    tournament_size
        Contenders per selection. Larger means stronger selection pressure.
    crossover_rate
        Probability a child is bred rather than cloned from one parent.
    mutation_rate
        Per-gene swap probability. ``None`` means ``1 / n_deliveries``.
    elites
        Fittest chromosomes carried over untouched each generation.

    Returns
    -------
    (best_solution, best_cost, convergence_history)
        Same shape as :func:`~qgati.optimizer.qpso.run_qpso`, so the benchmark
        runner can dispatch every metaheuristic uniformly.
        ``convergence_history[i]`` is the best-so-far cost after generation
        ``i``, is non-increasing, and has exactly ``num_generations`` entries.
    """
    if population_size < 1:
        raise ValueError("population_size must be at least 1")
    if num_generations < 1:
        raise ValueError("num_generations must be at least 1")
    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1")
    if tournament_size > population_size:
        # Otherwise numpy raises an opaque "cannot take a larger sample than
        # population" from inside the selection loop.
        raise ValueError(
            f"tournament_size ({tournament_size}) cannot exceed "
            f"population_size ({population_size})"
        )
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1")
    if not 0 <= elites < population_size:
        raise ValueError(
            f"elites must be fewer than population_size, got elites={elites} "
            f"and population_size={population_size}"
        )

    n = scenario.n_deliveries
    if mutation_rate is None:
        mutation_rate = 1.0 / n
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)

    def cost_of(chromosome: Sequence[int]) -> float:
        return evaluate(
            decode_permutation(chromosome, scenario, cost_matrix), scenario, cost_matrix
        ).fitness

    # Random permutations. Every solver here starts from an uninformed
    # population — seeding the GA with, say, the Savings solution would hand it a
    # head start the swarms do not get and quietly rig the comparison.
    population = [list(rng.permutation(n)) for _ in range(population_size)]
    costs = [cost_of(chromosome) for chromosome in population]

    best_index = int(np.argmin(costs))
    best_chromosome = list(population[best_index])
    best_cost = costs[best_index]
    convergence: list[float] = [best_cost]

    for _ in range(1, num_generations):
        order = sorted(range(population_size), key=lambda i: costs[i])
        offspring = [list(population[i]) for i in order[:elites]]

        while len(offspring) < population_size:
            parent_a = _tournament(population, costs, tournament_size, rng)
            if rng.random() < crossover_rate:
                parent_b = _tournament(population, costs, tournament_size, rng)
                child = order_crossover(parent_a, parent_b, rng)
            else:
                child = parent_a
            swap_mutation(child, rng, mutation_rate)
            offspring.append(child)

        population = offspring
        costs = [cost_of(chromosome) for chromosome in population]

        generation_best = int(np.argmin(costs))
        if costs[generation_best] < best_cost:
            best_cost = costs[generation_best]
            best_chromosome = list(population[generation_best])
        convergence.append(best_cost)

    # Re-derive the returned solution from the winning chromosome so the solution
    # and the reported cost can never disagree.
    best_solution = decode_permutation(best_chromosome, scenario, cost_matrix)
    return best_solution, evaluate(best_solution, scenario, cost_matrix).fitness, convergence
