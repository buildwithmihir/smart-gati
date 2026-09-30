"""What a second, a metre and a litre are worth — the objective solvers minimise.

The problem statement's core optimization goals name three things to reduce:
travel **time**, **distance** and **fuel**. A search can only minimise one number,
so the three must be combined — and the only defensible way to add quantities
measured in seconds, metres and litres is to convert all three into one common
unit. This module converts them into **rupees**, because that is the unit a fleet
operator already reasons in, and it makes every weight a price that can be argued
with rather than a tuned constant that cannot.

The choice is not invented here. ``DESIGN_DECISIONS.md`` already records the
objective as "one rupee-equivalent cost composed of time cost, distance cost,
fuel cost, penalties"; this module implements that decision, and
:mod:`qgati.optimizer.fitness` adds the penalties on top.

Prices, not normalised weights
------------------------------
The obvious alternative is to normalise each term to ``[0, 1]`` and pick weights
summing to one. It is rejected because normalised weights *hide* the trade-off
instead of stating it. Nothing tells a reviewer whether 0.3 for distance is
right, the answer silently changes with the instance's scale, and there is no
fact of the matter to check it against. A price is falsifiable: someone who
disagrees with ₹150/hour can say what their number is and why.

It also removes a trap. Weighted sums of unnormalised quantities are dominated
by whichever term happens to be quoted in the largest unit — here metres, which
would drown out litres by three orders of magnitude regardless of the intended
importance. Pricing each term fixes the comparison at the point where the
weights are chosen, which is where that judgement belongs.

Replacing the defaults
----------------------
Every number below is an assumption, not a measurement, and the constructor
takes all of them. The defaults describe a light diesel delivery vehicle
operating in Delhi; an operator with a fuel card and a payroll ledger should
replace them, and the model is built so that doing so is a constructor argument
rather than an edit. They are documented in ``DESIGN_DECISIONS.md`` under
"Objective function".

The fuel model, and what it assumes
-----------------------------------
Fuel is estimated from distance and average speed, which is the fallback the
brief sanctions when no better model is available. There is no better model
available offline: a real one needs per-vehicle engine data, and this project is
explicit that it consumes only the operator's own fleet observations. What is
used instead is the standard three-term average-speed curve,

    litres per 100 km  =  idle/v  +  rolling  +  drag * v        (v in km/h)

whose terms are the conventional decomposition of road fuel use — see
:data:`DEFAULT_FUEL_MODEL`. It is calibrated to two stated points rather than
fitted to a dataset, and the resulting curve is a good deal better than the
constant L/km it replaces, because it is the only part of the objective that
responds to *congestion*: the traffic layer scales edge times, which lowers each
leg's implied average speed, which raises fuel per kilometre. A pure distance
term cannot see that.

The honest caveat is that fuel and distance are strongly correlated — fuel is
distance times a factor that moves between roughly 0.075 and 0.126 L/km over the
speed range above. So the two terms mostly agree about which route is short and
differ about which route is slow. That is worth knowing when reading a cost
breakdown: distance and fuel are not two independent signals, and the fuel term
earns its place through its speed sensitivity, not its distance sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CostWeights",
    "DEFAULT_FUEL_MODEL",
    "DEFAULT_LATE_PER_HOUR",
    "DEFAULT_WEIGHTS",
    "FuelModel",
    "price_legs",
]


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
#: Rupees per hour of vehicle time — the loaded cost of keeping one delivery
#: vehicle on the road for an hour, whether or not it covers ground.
#:
#: Two parts. A commercial driver in Delhi on about ₹22,000 a month, over 26
#: working days of roughly 9 hours, is about ₹94/h. The vehicle's own time-based
#: costs — insurance, permits, road tax, and the calendar share of depreciation —
#: add roughly ₹55/h at 9 hours a day. Rounded to ₹150.
#:
#: This is the term that makes a fast route worth taking even when it is longer.
DEFAULT_TIME_PER_HOUR = 150.0

#: Rupees per kilometre of running cost, **excluding fuel**, which is priced
#: separately below. Tyres, servicing, oil, and the distance-based share of
#: depreciation; for a light commercial vehicle in India this lands around
#: ₹4-6/km, and ₹5 is the middle of that.
#:
#: Excluding fuel is not tidiness — including it here and pricing it again below
#: would charge every kilometre twice.
DEFAULT_DISTANCE_PER_KM = 5.0

#: Rupees per litre of diesel at retail in Delhi, which has held near ₹90.
DEFAULT_FUEL_PER_LITRE = 90.0

#: Rupees per hour of *lateness*, applied to every second a stop is served after
#: its ``latest_arrival``.
#:
#: Set at four times the vehicle-time price on purpose. Two rates equal would
#: make a solver indifferent between an hour spent driving and an hour spent
#: late, and it would then break a window whenever the detour cost more than the
#: delay it avoided — which is the opposite of what a service window is for.
#: Pricing lateness higher is what makes punctuality worth trading route length
#: for, and it is the only reason the penalty changes routing rather than merely
#: being computed.
#:
#: The multiple is a judgement, not a measurement: a missed window means a
#: re-delivery, a waiting customer, or a contractual penalty, and ₹600/hour is a
#: deliberately round stand-in for that. Replace it with the operator's own
#: service-level figure; ``CostWeights(late_per_hour=...)`` takes it.
DEFAULT_LATE_PER_HOUR = 600.0


# --------------------------------------------------------------------------- #
# The fuel curve
# --------------------------------------------------------------------------- #
#: Bounds the average-speed curve is evaluated within.
#:
#: Speed is *derived*, not measured: it is a leg's distance divided by its
#: travel time. Two things can therefore push it somewhere the curve should not
#: be trusted. A leg with no distance and no time — a delivery sharing its road
#: node with another — has no speed at all. And congestion scales edge times, so
#: a peak-hour multiplier of 2.9 on a 30 km/h road implies 10 km/h, while a
#: closure or a pile-up can imply less. Clamping keeps the ``idle/v`` term from
#: diverging toward infinity at the bottom and keeps an implausible 200 km/h tag
#: in OSM data from under-pricing a leg at the top. Fuel for a zero-distance leg
#: is zero regardless, so the diagonal is unaffected.
_MIN_SPEED_KPH = 5.0
_MAX_SPEED_KPH = 130.0


@dataclass(frozen=True, slots=True)
class FuelModel:
    """Average-speed fuel consumption: litres per 100 km as ``a/v + b + c*v``.

    The shape is the conventional one and each term has a physical reading:

    * ``idle_coefficient / v`` is the per-unit-*time* overhead — engine idling,
      friction, low-gear running — expressed per unit distance. Dividing by
      speed is what makes crawling expensive: the same kilometre takes longer, so
      the fixed cost of turning the engine is spread over less ground.
    * ``rolling_coefficient`` is rolling resistance: the part that depends on
      distance alone.
    * ``speed_coefficient * v`` is aerodynamic drag, which is why the curve turns
      back up once the vehicle is moving quickly.

    One minimum, which is the physically correct shape and the reason a
    constant L/km figure cannot express what congestion does to fuel.
    """

    idle_coefficient: float = 211.25
    rolling_coefficient: float = 1.0
    speed_coefficient: float = 0.05

    def __post_init__(self) -> None:
        if self.idle_coefficient < 0.0 or self.rolling_coefficient < 0.0:
            raise ValueError("fuel coefficients must not be negative")
        if self.speed_coefficient < 0.0:
            raise ValueError("fuel coefficients must not be negative")
        if self.idle_coefficient == 0.0 and self.speed_coefficient == 0.0:
            raise ValueError(
                "a fuel model with no speed dependence is just a constant L/km; "
                "set rolling_coefficient instead"
            )

    @property
    def most_efficient_kph(self) -> float:
        """Speed at which consumption per distance is lowest — ``sqrt(a/c)``."""
        if self.speed_coefficient <= 0.0:  # pragma: no cover - guarded in __post_init__
            return float("inf")
        return float(np.sqrt(self.idle_coefficient / self.speed_coefficient))

    def litres_per_100km(self, speed_kph):
        """Consumption at ``speed_kph``, scalar or array, before clamping."""
        speed = np.asarray(speed_kph, dtype=float)
        return (
            self.idle_coefficient / speed
            + self.rolling_coefficient
            + self.speed_coefficient * speed
        )


#: The default curve, pinned to two stated points rather than fitted — there is
#: no consumption dataset in this project to fit to.
#:
#: The minimum is placed at 65 km/h, a normal open-road cruise, and the vehicle
#: is made to return 7.5 L/100 km there. ``sqrt(a/c) = 65`` and the value at the
#: minimum together fix the coefficients: a = 0.05 * 65^2 = 211.25, b = 7.5 -
#: 2 * 0.05 * 65 = 1.0, c = 0.05. The curve then reads 12.6 L/100 km at 20 km/h
#: and 7.9 at 90, which is the right order of magnitude and the right shape for a
#: light diesel van. Anyone with real figures should replace this.
DEFAULT_FUEL_MODEL = FuelModel()


# --------------------------------------------------------------------------- #
# The combined objective
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CostWeights:
    """Prices converting time, distance and fuel into one rupee cost.

    Immutable and hashable, so it can sit on a :class:`~qgati.graph.cost_matrix.CostMatrix`
    and be compared or cached along with it. Every field has a default, so
    ``CostWeights()`` is the documented policy and any single price can be
    overridden on its own.
    """

    time_per_hour: float = DEFAULT_TIME_PER_HOUR
    distance_per_km: float = DEFAULT_DISTANCE_PER_KM
    fuel_per_litre: float = DEFAULT_FUEL_PER_LITRE
    late_per_hour: float = DEFAULT_LATE_PER_HOUR
    fuel_model: FuelModel = DEFAULT_FUEL_MODEL

    def __post_init__(self) -> None:
        for name in (
            "time_per_hour",
            "distance_per_km",
            "fuel_per_litre",
            "late_per_hour",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative")

    @property
    def time_per_second(self) -> float:
        """``time_per_hour`` in the unit the cost matrix is quoted in."""
        return self.time_per_hour / 3600.0

    @property
    def late_per_second(self) -> float:
        """``late_per_hour`` in seconds, the unit lateness is measured in."""
        return self.late_per_hour / 3600.0

    @property
    def distance_per_metre(self) -> float:
        """``distance_per_km`` in the unit the distance matrix is quoted in."""
        return self.distance_per_km / 1000.0

    def cost(self, seconds: float, metres: float, litres: float) -> float:
        """The objective for one route or one leg, in rupees.

        The single definition of the weighted sum; everything else in the
        optimizer reaches it through here, which is what keeps solvers
        comparable rather than merely similar.

        Does **not** include the lateness penalty. That is not a property of a
        distance or a duration the way the other three are — it depends on the
        whole route's arrival times, not on any leg — so
        :mod:`qgati.optimizer.fitness` adds it on top rather than folding it in
        here, where it would have no meaning for a single leg.
        """
        return (
            self.time_per_second * seconds
            + self.distance_per_metre * metres
            + self.fuel_per_litre * litres
        )


#: The documented default policy. See the module docstring for why these are
#: prices and ``DESIGN_DECISIONS.md`` for the reasoning behind each number.
DEFAULT_WEIGHTS = CostWeights()


# --------------------------------------------------------------------------- #
# Vectorised pricing, used when the cost matrix is built
# --------------------------------------------------------------------------- #
def average_speed_kph(seconds, metres):
    """Leg speed implied by a travel time and a distance, clamped.

    A leg whose time is not finite has no speed; it is reported as the ceiling,
    which is arbitrary but harmless because such a leg's cost is infinite
    through the time term regardless. A zero-distance leg likewise reports the
    ceiling rather than dividing by zero.
    """
    seconds = np.asarray(seconds, dtype=float)
    metres = np.asarray(metres, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(
            (seconds > 0.0) & np.isfinite(seconds),
            (metres / 1000.0) / (seconds / 3600.0),
            np.inf,
        )
    return np.clip(speed, _MIN_SPEED_KPH, _MAX_SPEED_KPH)


def price_legs(
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    weights: CostWeights = DEFAULT_WEIGHTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Price every leg, returning ``(fuel_litres, objective_rupees)``.

    The one place the three goals become a single number for a whole matrix.
    Building it here rather than per solver is what makes the benchmark
    apples-to-apples: every optimizer is handed the same already-priced array
    and none of them gets a chance to weight the goals differently.

    An unreachable pair (infinite time) stays infinite in both outputs, so it can
    never be mistaken for a cheap leg. A zero-distance leg costs nothing in fuel.
    """
    time = np.asarray(time_matrix, dtype=float)
    distance = np.asarray(distance_matrix, dtype=float)
    if time.shape != distance.shape:
        raise ValueError(
            f"time and distance matrices disagree on shape: "
            f"{time.shape} vs {distance.shape}"
        )

    speed = average_speed_kph(time, distance)
    consumption = weights.fuel_model.litres_per_100km(speed)

    # inf * 0 is nan, which is what an unreachable leg would produce as soon as
    # any price is zero — and a zero price is a legitimate way to ask what the
    # other two goals alone would choose. errstate silences the warning; the
    # explicit where() below restores the infinities afterwards.
    with np.errstate(invalid="ignore"):
        fuel = np.where(distance > 0.0, distance / 1000.0 * consumption / 100.0, 0.0)
        fuel = np.where(np.isfinite(time), fuel, np.inf)
        objective = (
            time * weights.time_per_second
            + distance * weights.distance_per_metre
            + fuel * weights.fuel_per_litre
        )
        objective = np.where(np.isfinite(time), objective, np.inf)

    return fuel, objective
