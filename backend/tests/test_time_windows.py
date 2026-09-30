"""Service windows: arrival tracking, and a penalty that changes what solvers do.

The question worth asking about a constraint like this is not "is the penalty
computed" — that is easy and nearly worthless — but "does the optimizer route
differently because of it". A penalty that is calculated and then ignored by
every solver looks exactly like a working feature from the outside, so the
central tests below build an instance whose shortest tour breaks a window and
require the solvers to abandon it.

Two things about the fixture are deliberate and worth stating up front.

**It is directed.** A symmetric matrix would have made these tests vacuous, and
quietly so. On a symmetric instance a tour and its reverse cost exactly the same,
so a window that the tour misses on its last stop is met for free by running the
same loop backwards — the window would change the direction of travel and nothing
else. Real Delhi matrices are asymmetric (one-way streets; see the backend README),
so a directed fixture is also the more faithful one.

**Distance is derived from time at a fixed implied speed**, so the rupee objective
is exactly proportional to travel seconds. That is not a physically consistent
road network and is not claimed to be: it is a timing table, shaped so that every
comparison below can be checked by hand. The consequence is that a second of
driving always costs ₹0.1549, whatever the window does — which is what makes the
narrow margin in :data:`TIGHT_C` legible rather than a matter of trust.

Windows are in **seconds from depot departure**; see
:class:`~qgati.optimizer.models.Delivery` for why.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qgati.graph.cost_matrix import CostMatrix
from qgati.optimizer import (
    SOLVERS,
    CostWeights,
    Delivery,
    Depot,
    Scenario,
    Solution,
    Vehicle,
    evaluate,
    solve_brute_force,
)
from qgati.optimizer import brute_force as bf
from qgati.optimizer.fitness import route_metrics, route_total_cost

# --------------------------------------------------------------------------- #
# The hand-built instance every routing test below shares
# --------------------------------------------------------------------------- #
#: Depot at 0, deliveries A, B, C and D at nodes 1, 2, 3 and 4.
#:
#: Travel seconds, **per direction** — a row is the leg *out of* that stop:
#:
#:              to:  depot     A     B     C     D
#:      from depot     0    10    20    12    24
#:      A             10     0    10    30    28
#:      B             20    10     0    26    10
#:      C             12    18    26     0    34
#:      D             24    28    10    12     0
#:
#: Asymmetric in exactly two places, and both matter: ``C -> D`` is slow (34 s)
#: where ``D -> C`` is quick (12 s), and ``A -> C`` is slow (30 s) where
#: ``C -> A`` is quick (18 s). Everything else is symmetric.
#:
#: The shape that falls out:
#:
#: * The cheapest tour is ``depot -> A -> B -> D -> C -> depot`` at **54 s**. It
#:   serves C last, 42 s in.
#: * Running that loop backwards costs **76 s**, not 54, because of ``C -> D`` —
#:   so the usual trick of reversing a tour to reach its last stop first does not
#:   work here.
#: * The only way to reach C early is to go straight to it: C is 12 s from the
#:   depot by road and at least 36 s from it via any other stop.
NODES = (0, 1, 2, 3, 4)
DELIVERY_NODES = {"A": 1, "B": 2, "C": 3, "D": 4}

_TIME_TABLE = (
    (0.0, 10.0, 20.0, 12.0, 24.0),  # depot
    (10.0, 0.0, 10.0, 30.0, 28.0),  # A
    (20.0, 10.0, 0.0, 26.0, 10.0),  # B
    (12.0, 18.0, 26.0, 0.0, 34.0),  # C
    (24.0, 28.0, 10.0, 12.0, 0.0),  # D
)

#: The window on C: it must be served within 12 s of leaving the depot, which is
#: exactly the time the direct drive takes. So the window does not merely prefer
#: C first — it admits nothing else, and the solver's only remaining freedom is
#: how to order the other three after it.
TIGHT_C = 12.0

#: The optimal tour with no windows — 54 s of driving, C served at 42 s.
UNCONSTRAINED = (0, 1, 3, 2)

#: The optimal tour once C's window binds — 74 s of driving, C served at 12 s.
#: A different order of the same four stops; the window costs 20 s of detour.
WINDOWED = (2, 0, 1, 3)


def _scenario(
    *,
    latest: dict[str, float] | None = None,
    earliest: dict[str, float] | None = None,
    deliveries: tuple[str, ...] = ("A", "B", "C", "D"),
    vehicles: int = 2,
    node_of: dict[str, int] | None = None,
) -> Scenario:
    """The instance above, with whatever windows are asked for.

    Two vehicles by default so Clarke-Wright is never forced to merge past the
    fleet size: that fallback overrides the heuristic's own judgement, and a
    solver whose answer is decided by a fallback demonstrates nothing about how
    it handles a window.
    """
    latest = latest or {}
    earliest = earliest or {}
    node_of = node_of or DELIVERY_NODES
    return Scenario(
        depot=Depot(node=0, lat=0.0, lon=0.0),
        deliveries=tuple(
            Delivery(
                id=name,
                node=node_of[name],
                demand=1.0,
                earliest_arrival=earliest.get(name),
                latest_arrival=latest.get(name),
            )
            for name in deliveries
        ),
        vehicles=tuple(
            Vehicle(id=f"V{index}", capacity=10.0) for index in range(vehicles)
        ),
    )


def _tight() -> Scenario:
    """The instance with C's window: what the solvers must route around."""
    return _scenario(latest={"C": TIGHT_C})


def _matrix_for(rows: tuple[tuple[float, ...], ...], nodes) -> CostMatrix:
    """A directed timing table, priced with distance at a fixed 30 km/h."""
    times = np.array(rows, dtype=float)
    return CostMatrix(
        matrix=times,
        nodes=nodes,
        delivery_node_index=tuple(nodes[1:]),
        distance_matrix=times * (30.0 / 3.6),
    )


def _windows_matrix() -> CostMatrix:
    return _matrix_for(_TIME_TABLE, NODES)


def _shape(solution: Solution) -> tuple[tuple[int, ...], ...]:
    """The routes actually driven: direction preserved, empty routes dropped.

    Direction is deliberately **not** canonicalised away. On a directed matrix a
    reversal is a different route with a different cost and different arrival
    times, and it is precisely the kind of change a window is supposed to cause —
    normalising it out would hide the thing under test.
    """
    return tuple(sorted(route for route in solution.routes if route))


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def test_a_delivery_without_windows_behaves_exactly_as_before() -> None:
    delivery = Delivery(id="A", node=1, demand=2.0)
    assert delivery.earliest_arrival is None
    assert delivery.latest_arrival is None
    assert not delivery.has_window


def test_an_empty_window_is_rejected() -> None:
    """An earliest after its latest cannot be served, so it is refused outright."""
    with pytest.raises(ValueError, match="empty window"):
        Delivery(id="A", node=1, demand=1.0, earliest_arrival=500.0, latest_arrival=100.0)


def test_a_negative_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        Delivery(id="A", node=1, demand=1.0, earliest_arrival=-1.0)


def test_scenario_reports_whether_any_window_is_set() -> None:
    assert not _scenario().has_time_windows
    assert _tight().has_time_windows


def test_windows_are_reported_in_delivery_order() -> None:
    scenario = _scenario(latest={"C": TIGHT_C}, earliest={"A": 5.0})
    assert scenario.windows == ((5.0, None), (None, None), (None, TIGHT_C), (None, None))


# --------------------------------------------------------------------------- #
# Arrival tracking
# --------------------------------------------------------------------------- #
def test_arrival_times_accumulate_along_the_route() -> None:
    """A -> B -> C -> D: each arrival is the running sum of the legs."""
    metrics = route_metrics((0, 1, 2, 3), _scenario(), _windows_matrix())

    # depot->A 10, A->B 10, B->C 26, C->D 34, D->depot 24
    assert metrics.arrival_times == pytest.approx((10.0, 20.0, 46.0, 80.0))
    assert metrics.driving_time == pytest.approx(104.0)
    assert metrics.waiting == 0.0
    assert metrics.lateness == 0.0
    assert metrics.time == pytest.approx(104.0)


def test_arriving_early_waits_rather_than_counting_as_a_violation() -> None:
    """Being early is not a fault — the vehicle waits, and the wait is charged."""
    metrics = route_metrics(
        (0, 1, 2, 3), _scenario(earliest={"B": 50.0}), _windows_matrix()
    )

    # A is reached at 10 s; B would be reached at 20 s but opens at 50 s.
    assert metrics.waiting == pytest.approx(30.0)
    assert metrics.arrival_times[1] == pytest.approx(50.0)
    assert metrics.lateness == 0.0
    assert metrics.window_penalty == 0.0
    # The wait is not extra driving, but it is extra elapsed time at the time
    # price, and it pushes every later arrival back by the same 30 s.
    assert metrics.driving_time == pytest.approx(104.0)
    assert metrics.time == pytest.approx(134.0)
    assert metrics.arrival_times[2] == pytest.approx(76.0)


def test_lateness_is_penalised_in_proportion_to_how_late() -> None:
    matrix = _windows_matrix()
    metrics = route_metrics((0, 1, 2, 3), _scenario(latest={"D": 70.0}), matrix)

    # D is reached at 80 s against a latest of 70 s: 10 s late.
    assert metrics.lateness == pytest.approx(10.0)
    assert metrics.window_penalty == pytest.approx(10.0 * matrix.weights.late_per_second)
    assert metrics.total_cost == pytest.approx(metrics.cost + metrics.window_penalty)


def test_the_penalty_scales_with_the_price() -> None:
    """Twice the price of lateness, twice the penalty — a price, not a flag."""
    scenario = _scenario(latest={"D": 70.0})
    cheap = _windows_matrix()
    dear = CostMatrix(
        matrix=cheap.matrix,
        nodes=cheap.nodes,
        delivery_node_index=cheap.delivery_node_index,
        distance_matrix=cheap.distance_matrix,
        weights=CostWeights(late_per_hour=2.0 * cheap.weights.late_per_hour),
    )

    assert route_metrics((0, 1, 2, 3), scenario, dear).window_penalty == pytest.approx(
        2.0 * route_metrics((0, 1, 2, 3), scenario, cheap).window_penalty
    )


def test_waiting_carries_into_later_arrivals() -> None:
    """A window can fail a route with nothing in it being late when reached.

    B opens at 50 s, so the vehicle waits 30 s for it; that delay alone is what
    makes D miss a window it would otherwise have met with 5 s to spare.
    """
    matrix = _windows_matrix()
    without_wait = route_metrics((0, 1, 2, 3), _scenario(latest={"D": 85.0}), matrix)
    with_wait = route_metrics(
        (0, 1, 2, 3), _scenario(earliest={"B": 50.0}, latest={"D": 85.0}), matrix
    )

    assert without_wait.lateness == 0.0
    assert with_wait.lateness == pytest.approx(25.0)


def test_windows_are_ignored_when_none_are_set() -> None:
    """The no-window path is untouched, so existing instances are unaffected."""
    matrix = _windows_matrix()
    scenario = _scenario()
    plain = route_metrics((0, 1, 2, 3), scenario, matrix)

    assert plain.waiting == 0.0
    assert plain.lateness == 0.0
    assert plain.window_penalty == 0.0
    assert plain.time == pytest.approx(plain.driving_time)
    assert route_total_cost((0, 1, 2, 3), scenario, matrix) == pytest.approx(plain.cost)


# --------------------------------------------------------------------------- #
# The penalty reaches fitness
# --------------------------------------------------------------------------- #
def test_evaluate_folds_the_window_penalty_into_fitness_not_travel_cost() -> None:
    scenario = _scenario(latest={"D": 70.0})
    matrix = _windows_matrix()
    evaluation = evaluate(Solution(routes=((0, 1, 2, 3), ())), scenario, matrix)

    assert evaluation.window_penalty > 0.0
    assert evaluation.lateness == pytest.approx(10.0)
    assert not evaluation.feasible
    # Held out of travel_cost so "how much of this is lateness?" stays answerable.
    assert evaluation.travel_cost == pytest.approx(sum(evaluation.route_costs))
    assert evaluation.fitness == pytest.approx(
        evaluation.travel_cost
        + evaluation.window_penalty
        + evaluation.capacity_penalty
        + evaluation.coverage_penalty
        + evaluation.shape_penalty
    )


def test_a_punctual_solution_is_feasible_and_unpenalised() -> None:
    scenario = _scenario(latest={"D": 100.0})
    evaluation = evaluate(
        Solution(routes=((0, 1, 2, 3), ())), scenario, _windows_matrix()
    )

    assert evaluation.window_penalty == 0.0
    assert evaluation.feasible


# --------------------------------------------------------------------------- #
# The exact solver's two regimes agree when nothing binds
# --------------------------------------------------------------------------- #
def test_the_windowed_dp_reproduces_held_karp_when_windows_do_not_bind() -> None:
    """Every subset's tour must cost the same under both regimes.

    A window-aware Held-Karp that disagreed with the scalar one would still look
    correct on an instance whose optimal tour avoided the discrepancy, so this
    compares the whole cost table — all fifteen subsets — rather than one route.

    Generous windows are what put the windowed code path in play while leaving it
    nothing to trade off.
    """
    matrix = _windows_matrix()
    scenario = _scenario(
        earliest={name: 0.0 for name in DELIVERY_NODES},
        latest={name: 1e6 for name in DELIVERY_NODES},
    )
    n = scenario.n_deliveries
    local, local_time = bf._local_arrays(matrix, n)

    additive, _ = bf._additive_tours(n, local)
    windowed, _ = bf._windowed_tours(scenario, matrix, n, local, local_time)

    assert np.allclose(additive, windowed)


def test_the_windowed_dp_agrees_with_an_independent_enumeration() -> None:
    """The Pareto DP checked against pricing every tour by hand.

    The pruning is the part most likely to be quietly wrong, and the opening stop
    is the part most likely to be *forgotten*: it is the one arrival in a route
    with no predecessor to charge it against, so its wait and its lateness are
    dropped by any implementation that only handles windows on transitions. That
    bug is invisible on an instance whose first stop is served on time, which is
    why the scenarios below include one where the first stop cannot be.

    Enumerating permutations and pricing each with :func:`route_metrics` shares no
    code with the DP, so the two can only agree if both are right.
    """
    matrix = _windows_matrix()
    scenarios = (
        _tight(),
        _scenario(earliest={"A": 50.0}),  # the first stop must be waited for
        _scenario(latest={"D": 70.0}),
        _scenario(earliest={"B": 50.0}, latest={"D": 85.0}),
    )

    for scenario in scenarios:
        n = scenario.n_deliveries
        local, local_time = bf._local_arrays(matrix, n)
        windowed, _ = bf._windowed_tours(scenario, matrix, n, local, local_time)

        for mask in range(1, 1 << n):
            members = [d for d in range(n) if (mask >> d) & 1]
            best = min(
                route_total_cost(tour, scenario, matrix)
                for tour in itertools.permutations(members)
            )
            assert windowed[mask] == pytest.approx(best), (
                f"subset {members} on {scenario.delivery_ids()}: the DP says "
                f"{windowed[mask]}, enumerating the tours says {best}"
            )


def test_the_two_regimes_disagree_once_a_window_binds() -> None:
    """...and the equivalence above is not vacuous.

    Same comparison on the tight instance, where the cheapest tour of the full
    subset must now be a different one — otherwise the equivalence test would
    pass just as happily on a windowed DP that had become a no-op.
    """
    matrix = _windows_matrix()
    scenario = _tight()
    n = scenario.n_deliveries
    local, local_time = bf._local_arrays(matrix, n)

    additive, _ = bf._additive_tours(n, local)
    windowed, _ = bf._windowed_tours(scenario, matrix, n, local, local_time)

    assert not np.allclose(additive, windowed)
    # A window can only ever make a tour dearer, never cheaper.
    finite = np.isfinite(additive)
    assert np.all(windowed[finite] >= additive[finite] - 1e-9)


# --------------------------------------------------------------------------- #
# The one that matters: does the penalty change what solvers do?
# --------------------------------------------------------------------------- #
def test_the_unconstrained_optimum_is_what_we_think_it_is() -> None:
    """Pin the premise, so the tests below cannot drift from their instance."""
    matrix = _windows_matrix()

    assert _shape(solve_brute_force(_scenario(), matrix)) == (UNCONSTRAINED,)
    assert route_metrics(UNCONSTRAINED, _scenario(), matrix).time == pytest.approx(54.0)


def test_the_window_makes_the_shortest_tour_infeasible() -> None:
    """State the trap before testing that solvers escape it.

    And check the margin, because it is not large. With the default prices a
    second spent late costs about 1.08 times a second spent driving, so the
    window only wins if it is tight enough that avoiding the lateness is worth
    the detour it forces. Here it is: 30 s of lateness avoided buys 20 s of extra
    driving, and only just.
    """
    matrix = _windows_matrix()
    scenario = _tight()
    shortest = evaluate(Solution(routes=(UNCONSTRAINED, ())), scenario, matrix)
    punctual = evaluate(Solution(routes=(WINDOWED, ())), scenario, matrix)

    assert shortest.lateness == pytest.approx(30.0)  # C served at 42 s, window closes at 12
    assert not shortest.feasible

    assert punctual.lateness == 0.0
    assert punctual.travel_time == pytest.approx(74.0)
    # The punctual route drives further and still wins — which is the claim the
    # whole feature rests on. If it lost, the price of lateness would be too low
    # to ever change a decision.
    assert punctual.distance > shortest.distance
    assert punctual.fitness < shortest.fitness


def test_a_tight_window_changes_the_route_the_exact_solver_returns() -> None:
    """The headline: a window makes the *exact* solver abandon the shortest tour.

    Both answers here are unique, so this asserts the routes themselves rather
    than a property of them. Because the solver is exact, the windowed route is
    the provably best windowed answer and not a heuristic's guess at one.
    """
    matrix = _windows_matrix()

    assert _shape(solve_brute_force(_scenario(), matrix)) == (UNCONSTRAINED,)
    assert _shape(solve_brute_force(_tight(), matrix)) == (WINDOWED,)


def test_reversing_the_tour_is_not_a_way_round_the_window() -> None:
    """The reason the fixture is directed, asserted rather than assumed.

    If the reverse of the optimal tour were as cheap as the tour, a solver could
    satisfy C's window by driving the same loop backwards at no cost — and every
    test below would pass while proving nothing. It is not cheap: ``C -> D`` is
    34 s against ``D -> C``'s 12 s.
    """
    matrix = _windows_matrix()
    scenario = _tight()
    reversed_tour = tuple(reversed(UNCONSTRAINED))

    assert _shape(solve_brute_force(_scenario(), matrix)) == (UNCONSTRAINED,)
    # Reaching C first this way does satisfy the window...
    assert evaluate(Solution(routes=(reversed_tour, ())), scenario, matrix).lateness == 0.0
    # ...but it is dearer than the answer the solvers are required to find.
    assert route_total_cost(reversed_tour, scenario, matrix) > route_total_cost(
        WINDOWED, scenario, matrix
    )


#: Every solver that searches, and so every solver the penalty has to steer.
#: Clarke-Wright is the exception and is tested on its own below: it is
#: constructive rather than search-based, it cannot re-plan an answer it has
#: already built, and on this instance its greedy happens to serve C first
#: anyway — so "did the penalty change its search" is not a question it can
#: answer. What the penalty has to reach there is the merge rule, which the
#: dedicated test below exercises.
SEARCHING_SOLVERS = tuple(spec for spec in SOLVERS if spec.key != "savings")


@pytest.mark.parametrize("solver", SEARCHING_SOLVERS, ids=lambda spec: spec.name)
def test_every_searching_solver_reroutes_for_the_window(solver) -> None:
    """The penalty must change the route, not merely the score.

    Same instance, same seed, with and without the window. Three things are
    required of each solver, and the strict inequality is the one with teeth: a
    solver that computed the penalty but never consulted it would return the
    unconstrained answer, and would therefore score exactly what ignoring the
    window scores rather than beating it.

    Driving this off the registry rather than a hand-written list means a solver
    added later is covered without anyone remembering to add it here.
    """
    matrix = _windows_matrix()
    constrained = _tight()
    without = solver(matrix, _scenario(), seed=7)[0]
    with_window = solver(matrix, constrained, seed=7)[0]

    knowing = evaluate(with_window, constrained, matrix)
    ignoring = evaluate(without, constrained, matrix)

    assert knowing.lateness == 0.0, (
        f"{solver.name} returned a solution that misses C's window"
    )
    assert _shape(without) != _shape(with_window), (
        f"{solver.name} returned the same routes with and without the window"
    )
    assert knowing.fitness < ignoring.fitness, (
        f"{solver.name} did no better under the window than by ignoring it, so "
        f"the penalty is not reaching its search"
    )


def test_savings_serves_the_windowed_stop_in_time() -> None:
    """The baseline must not be made *worse* by the window.

    Weaker than the test above, and honestly so: Clarke-Wright builds a route once
    and cannot re-plan it, so the penalty reaches it only by refusing merges it
    would otherwise commit. Here its greedy already reaches C directly, so this
    pins that it stays punctual — the merge rule below is the other half of the
    claim, that the window does not stop it committing a merge worth committing.
    """
    matrix = _windows_matrix()
    savings = next(spec for spec in SOLVERS if spec.key == "savings")
    windowed = savings(matrix, _tight(), seed=7)[0]

    assert evaluate(windowed, _tight(), matrix).lateness == 0.0


# --------------------------------------------------------------------------- #
# Clarke-Wright, on an instance built to expose it
# --------------------------------------------------------------------------- #
#: Two stops, two vehicles, and a merge that is worth money but misses a window.
#:
#: ``X -> Y`` is a short hop, so joining the two deliveries saves 9 s of driving
#: and the savings rule accepts it — but X cannot be served until 100 s, so the
#: vehicle sits there and then reaches Y at 106 s against a 20 s deadline. Running
#: them separately keeps the wait inside X's own route and serves Y at 10 s.
_SAVINGS_TABLE = (
    (0.0, 5.0, 10.0),  # depot -> X 5, Y 10
    (5.0, 0.0, 6.0),   # X -> depot 5, Y 6
    (10.0, 6.0, 0.0),  # Y -> depot 10, X 6
)
_SAVINGS_NODES = (0, 1, 2)
_SAVINGS_NODE_OF = {"X": 1, "Y": 2}


def _savings_scenario(windowed: bool, vehicles: int = 2) -> Scenario:
    return _scenario(
        deliveries=("X", "Y"),
        earliest={"X": 100.0} if windowed else None,
        latest={"Y": 20.0} if windowed else None,
        vehicles=vehicles,
        node_of=_SAVINGS_NODE_OF,
    )


def _savings_matrix() -> CostMatrix:
    return _matrix_for(_SAVINGS_TABLE, _SAVINGS_NODES)


def test_savings_refuses_a_merge_that_would_break_a_window() -> None:
    """Clarke-Wright is the solver most able to ignore windows, so pin it down.

    Its merge loop compares leg savings, which know nothing about arrival times,
    and it commits a merge without ever re-checking the clock. The window reaches
    it only through the rule that a merged route must beat the two it replaces
    once lateness is priced — so this test asserts the rule's premise directly,
    then asserts the rule actually changes the answer.
    """
    matrix = _savings_matrix()
    savings = next(spec for spec in SOLVERS if spec.key == "savings")

    windowed = _savings_scenario(windowed=True)

    # The premise: merging is worth 9 s of driving, but costs far more in lateness.
    merged = route_total_cost((0, 1), windowed, matrix)
    separate = route_total_cost((0,), windowed, matrix) + route_total_cost(
        (1,), windowed, matrix
    )
    assert merged > separate, "the fixture no longer exercises the rule"

    # Which is exactly what decides the two answers.
    assert _shape(savings(matrix, _savings_scenario(windowed=False), seed=None)[0]) == (
        (0, 1),
    )
    assert _shape(savings(matrix, windowed, seed=None)[0]) == ((0,), (1,))


def test_savings_result_is_scored_with_the_window_like_everyone_else() -> None:
    """A missed window must show up in the baseline's score, not be lost.

    Savings cannot re-plan a route, so it may legitimately return a late one. The
    requirement is that this is *reported*: the benchmark compares it against the
    metaheuristics on the shared evaluator, so its lateness has to be priced like
    anyone's rather than quietly dropped.

    The instance is one where lateness is unavoidable — the only window here
    closes before any vehicle could reach it — so the assertion does not depend
    on which way round the greedy happens to build its route.
    """
    matrix = _savings_matrix()
    savings = next(spec for spec in SOLVERS if spec.key == "savings")
    unreachable = _scenario(
        deliveries=("X", "Y"),
        latest={"X": 1.0, "Y": 1.0},
        node_of=_SAVINGS_NODE_OF,
    )

    evaluation = evaluate(savings(matrix, unreachable, seed=None)[0], unreachable, matrix)

    assert evaluation.lateness > 0.0
    assert evaluation.window_penalty > 0.0
    assert not evaluation.feasible
    assert evaluation.fitness > evaluation.travel_cost


# --------------------------------------------------------------------------- #
# Degenerate and slack cases
# --------------------------------------------------------------------------- #
def test_an_unreachable_window_still_yields_a_rankable_answer() -> None:
    """An impossible window is a penalty, not an exception.

    Nothing can avoid being late here, so every route is infeasible. The point is
    that solvers still return their best attempt and the evaluator still ranks
    two of them against each other — a hard constraint would instead have made the
    instance unsolvable and told an operator nothing about how close they got.
    """
    matrix = _windows_matrix()
    impossible = _scenario(latest={name: 1.0 for name in DELIVERY_NODES})

    solution = solve_brute_force(impossible, matrix)
    evaluation = evaluate(solution, impossible, matrix)

    assert not evaluation.feasible
    assert evaluation.window_penalty > 0.0
    assert len(solution.routes) == impossible.n_vehicles


def test_windows_do_not_disturb_a_solution_that_already_meets_them() -> None:
    """Adding a window every tour already satisfies changes nothing.

    Guards against the feature quietly rerouting instances it need not touch: the
    windowed code path is a different algorithm and must agree with the old one
    wherever the windows are slack.
    """
    matrix = _windows_matrix()
    generous = _scenario(
        earliest={name: 0.0 for name in DELIVERY_NODES},
        latest={name: 1e6 for name in DELIVERY_NODES},
    )

    assert _shape(solve_brute_force(generous, matrix)) == _shape(
        solve_brute_force(_scenario(), matrix)
    )
