"""The solver registry — one list of what this project can solve with.

The Phase 4 benchmark, the Phase 5 API, and anything later all read the solvers
from here. Without it, "which algorithms exist" would be answered separately in
each place and the answers would drift: the benchmark would gain a solver the API
never exposes, or one solver's budget argument would be renamed in one place and
not the other. This is the same argument that put every metaheuristic behind one
decoder in :mod:`qgati.optimizer.decoding`, applied to the solver list itself.

Production default
------------------
:data:`DEFAULT_SOLVER_KEY` is **ACO**, not QPSO. Phase 4's equal-budget benchmark
measured ACO dominant at every tested scale — best cost and best mean at n=15 and
n=25, exact optimum 5/5 at n=8, and nearly budget-insensitive (0.4% between 100
and 3,000 iterations). Selection followed the measurements rather than the
project's original premise. QPSO stays first-class for comparison and
explainability, which is where its research value lives: it beats classical PSO
at equal budget, and those two differ only in the update rule.

Uniform interface
-----------------
Every solver is callable as::

    solver(cost_matrix, scenario, seed=0, population=30, iterations=100)
        -> (Solution, cost, convergence_history)

The four metaheuristics name their arguments differently on purpose —
``num_generations`` reads better in a GA than ``num_iterations`` — so the spec
carries the argument names rather than forcing the solvers to rename them. The
deterministic solvers (brute force, Savings) accept and ignore the three
parameters, which is what lets a caller loop over every solver without
special-casing any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from qgati.optimizer.aco import run_aco
from qgati.optimizer.brute_force import MAX_EXACT_DELIVERIES, solve_brute_force
from qgati.optimizer.classical_pso import run_classical_pso
from qgati.optimizer.fitness import evaluate
from qgati.optimizer.genetic_algorithm import run_genetic_algorithm
from qgati.optimizer.models import Scenario, Solution
from qgati.optimizer.qpso import run_qpso
from qgati.optimizer.savings import clarke_wright_savings

if TYPE_CHECKING:  # avoid an optimizer <-> graph import cycle at runtime
    from qgati.graph.cost_matrix import CostMatrix

__all__ = [
    "DEFAULT_POPULATION",
    "DEFAULT_SOLVER_KEY",
    "DEFAULT_ITERATIONS",
    "SOLVERS",
    "SolverSpec",
    "default_solver",
    "get_solver",
    "solver_keys",
]

#: Default search effort, shared by every stochastic solver so a request that
#: names no parameters gets a like-for-like answer whichever solver it picks.
DEFAULT_POPULATION = 30
DEFAULT_ITERATIONS = 100


@dataclass(frozen=True)
class SolverSpec:
    """How to call one solver, and what it costs to call it."""

    key: str
    """Stable machine name — used in URLs and request payloads. Never renamed."""

    name: str
    """Display name, for tables and reports."""

    run: Callable[..., tuple[Solution, float, list[float]]]
    """The solver itself, under its own native signature."""

    budget_kwarg: str | None
    """Name of the generation-count argument, or ``None`` if it takes none."""

    population_kwarg: str | None
    """Name of the swarm/population-size argument, or ``None``."""

    is_stochastic: bool
    """True when the result depends on ``seed``. Deterministic solvers are run
    once and their runtime is reported as measured once."""

    is_exact: bool = False
    """True only for brute force: returns a provably optimal answer."""

    exact_limit: int | None = None
    """Largest instance this solver can handle exactly, if bounded."""

    def __call__(
        self,
        cost_matrix: CostMatrix,
        scenario: Scenario,
        *,
        seed: int | None = None,
        population: int = DEFAULT_POPULATION,
        iterations: int = DEFAULT_ITERATIONS,
    ) -> tuple[Solution, float, list[float]]:
        """Run the solver on a uniform signature.

        Returns ``(solution, cost, convergence_history)``. The deterministic
        solvers have no history and return an empty one, so callers can treat
        every solver identically.
        """
        if not self.is_stochastic:
            return self.run(cost_matrix, scenario, seed=None)

        parameters: dict[str, int] = {}
        if self.budget_kwarg is not None:
            parameters[self.budget_kwarg] = iterations
        if self.population_kwarg is not None:
            parameters[self.population_kwarg] = population
        return self.run(cost_matrix, scenario, seed=seed, **parameters)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "is_stochastic": self.is_stochastic,
            "is_exact": self.is_exact,
            "exact_limit": self.exact_limit,
        }


# --------------------------------------------------------------------------- #
# Adapters for the two solvers that predate the uniform interface
# --------------------------------------------------------------------------- #
def _run_brute_force(
    cost_matrix: CostMatrix, scenario: Scenario, **_ignored: object
) -> tuple[Solution, float, list[float]]:
    solution = solve_brute_force(scenario, cost_matrix)
    return solution, evaluate(solution, scenario, cost_matrix).fitness, []


def _run_savings(
    cost_matrix: CostMatrix, scenario: Scenario, **_ignored: object
) -> tuple[Solution, float, list[float]]:
    solution = clarke_wright_savings(scenario, cost_matrix)
    return solution, evaluate(solution, scenario, cost_matrix).fitness, []


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
#: Every solver, in the order they should be presented: the exact answer first,
#: then the constructive baseline, then the metaheuristics.
SOLVERS: tuple[SolverSpec, ...] = (
    SolverSpec(
        key="brute_force",
        name="Brute Force",
        run=_run_brute_force,
        budget_kwarg=None,
        population_kwarg=None,
        is_stochastic=False,
        is_exact=True,
        exact_limit=MAX_EXACT_DELIVERIES,
    ),
    SolverSpec(
        key="savings",
        name="Savings",
        run=_run_savings,
        budget_kwarg=None,
        population_kwarg=None,
        is_stochastic=False,
    ),
    SolverSpec(
        key="aco",
        name="ACO",
        run=run_aco,
        budget_kwarg="num_iterations",
        population_kwarg="num_ants",
        is_stochastic=True,
    ),
    SolverSpec(
        key="genetic_algorithm",
        name="Genetic Algorithm",
        run=run_genetic_algorithm,
        budget_kwarg="num_generations",
        population_kwarg="population_size",
        is_stochastic=True,
    ),
    SolverSpec(
        key="classical_pso",
        name="Classical PSO",
        run=run_classical_pso,
        budget_kwarg="num_iterations",
        population_kwarg="num_particles",
        is_stochastic=True,
    ),
    SolverSpec(
        key="qpso",
        name="QPSO",
        run=run_qpso,
        budget_kwarg="num_iterations",
        population_kwarg="num_particles",
        is_stochastic=True,
    ),
)

_BY_KEY: dict[str, SolverSpec] = {spec.key: spec for spec in SOLVERS}

#: The production default. See the module docstring for why this is ACO.
DEFAULT_SOLVER_KEY = "aco"


def solver_keys() -> tuple[str, ...]:
    """Every registered solver key, in presentation order."""
    return tuple(spec.key for spec in SOLVERS)


def get_solver(key: str) -> SolverSpec:
    """Look a solver up by key.

    Raises
    ------
    KeyError
        If no solver has that key. The message lists the valid keys, because the
        usual cause is a typo in a request payload.
    """
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown solver {key!r}; available: {', '.join(solver_keys())}"
        ) from None


def default_solver() -> SolverSpec:
    """The production default solver."""
    return get_solver(DEFAULT_SOLVER_KEY)
