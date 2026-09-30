"""The solver registry — one list of what this project can solve with.

The Phase 4 benchmark, the Phase 5 API, and anything later all read the solvers
from here. Without it, "which algorithms exist" would be answered separately in
each place and the answers would drift: the benchmark would gain a solver the API
never exposes, or one solver's budget argument would be renamed in one place and
not the other. This is the same argument that put every metaheuristic behind one
decoder in :mod:`qgati.optimizer.decoding`, applied to the solver list itself.

Production default
------------------
:data:`DEFAULT_SOLVER_KEY` is **QPSO**. The SIH problem statement (PS 26137) names
quantum-inspired search as the focus of this work — quantum particle swarm
optimization, benchmarked against conventional metaheuristics and exact methods —
so it is the solver the API serves by default, and the other four here exist to
benchmark it: three conventional metaheuristics (Savings, GA, classical PSO) and
one exact ground truth (brute force).

What the benchmark supports is the comparison QPSO was chosen for. At equal budget
QPSO beats classical PSO — 5.9% better at n=25 with 3,000 iterations, and 3/5 runs
hitting the exact optimum at n=8 against classical PSO's 1/5. Those two share one
representation and differ *only* in the update rule, so the gap is attributable to
the quantum sampling itself rather than to encoding or budget. That is the closest
controlled comparison available here, and it is the problem statement's own claim
about quantum-inspired sampling, measured. See the backend README, "What the
benchmark actually shows", for the numbers behind every sentence above.

A finding this registry no longer records, deliberately
-------------------------------------------------------
ACO was **removed from the project**, and it is worth being explicit that this
deleted a result rather than a solver that was merely redundant: when it was
benchmarked, ACO reached a *lower mean raw cost than QPSO* at n=15 and n=25, and
held the best mean at every instance size. The four
solvers remaining here are not a full account of what QPSO was measured against,
and the two README tables that still hold ACO's figures are kept as measurements
rather than re-run. See "Removed: ACO" in ``DESIGN_DECISIONS.md``. Any claim of
the form "QPSO is the best of these" is now true of a smaller field.

Uniform interface
-----------------
Every solver is callable as::

    solver(cost_matrix, scenario, seed=0, population=30, iterations=100)
        -> (Solution, cost, convergence_history)

The three metaheuristics name their arguments differently on purpose —
``num_generations`` reads better in a GA than ``num_iterations`` — so the spec
carries the argument names rather than forcing the solvers to rename them. The
deterministic solvers (brute force, Savings) accept and ignore the three
parameters, which is what lets a caller loop over every solver without
special-casing any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

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

#: The production default. See the module docstring for why this is QPSO.
DEFAULT_SOLVER_KEY = "qpso"


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
