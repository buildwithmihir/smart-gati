"""Anomaly detection: is this road behaving unlike itself?

The traffic log has been collecting ``(road, time, condition) -> travel_time``
rows since the logging phase and nothing has read them back. This module is what
reads them: it summarises a road's history, and judges each newly observed travel
time against it. Plain statistics — mean, standard deviation, a z-score — and
deliberately nothing more. There is no model here and no training, and the
thresholds are constants a reader can check by hand.

Two rules, and why there are two
--------------------------------
A z-score needs a standard deviation, and a standard deviation needs enough
samples to be worth estimating. Below :data:`MIN_SAMPLES` there is not enough, so
the test falls back to the simulator's own rule-based expectation with a flat
margin over it. The two rules answer the same question — *is this road slower than
it normally is?* — with different amounts of evidence, and both are given the same
notion of "normal" to compare against. See :func:`detect`.

Why "normal" excludes incidents
-------------------------------
The baseline is built from :func:`build_baseline`, which skips every row carrying
an ``incident_type``. A baseline that has been trained on anomalies cannot detect
them: a road that is routinely shut at 18:00 would report a shut road at 18:00 as
perfectly ordinary, and the closure would go unreported. Only rows that record a
road being *itself* — no incident, and a travel time at all — count as history.

The same reasoning decides the fallback's ``expected``. It is the road's modelled
cost under the clock alone, with manual conditions cleared, so both rules compare
against a normal the operator has not touched.

The case that dominates real data
---------------------------------
An ordinary logged ``travel_time`` is ``base_travel_time(attributes) *
congestion_multiplier(road_class, condition)`` — both deterministic. Every sample
for one ``(road, condition)`` key is therefore the *same number*, so the standard
deviation is exactly zero and the z-score is a division by zero that is not an
error. :func:`z_score` gives that case an explicit answer rather than a guard:
matching the history is 0 standard deviations out, and differing from it is
infinitely many.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

from qgati.traffic.log_store import TrafficLogRow

__all__ = [
    "FALLBACK_FACTOR",
    "INSUFFICIENT_SAMPLES",
    "MIN_SAMPLES",
    "Z_SCORE",
    "Z_THRESHOLD",
    "Baseline",
    "Detection",
    "RoadStats",
    "build_baseline",
    "detect",
    "z_score",
]

Node = Hashable

#: How many historical samples a ``(road, condition)`` key needs before its
#: standard deviation is trusted enough to divide by. Below this the rule-based
#: expectation decides instead. ``DESIGN_DECISIONS.md`` § Detection.
MIN_SAMPLES = 10

#: Standard deviations above the mean that count as anomalous.
#:
#: One-sided on purpose — a road running *faster* than usual is not a reason to
#: re-route anything, so a large negative z is left alone.
Z_THRESHOLD = 2.0

#: How far over the rule-based expectation an observation may sit while
#: :data:`MIN_SAMPLES` is unmet. The fallback has no spread to measure, so it uses
#: a blunt margin instead: 20% slower than the model expects.
FALLBACK_FACTOR = 1.2

#: Which rule produced a verdict. The values are the wire form.
Z_SCORE = "z_score"
INSUFFICIENT_SAMPLES = "insufficient_samples"


@dataclass(frozen=True, slots=True)
class RoadStats:
    """What a road's logged history says about it, for one condition band.

    ``count`` is the number of usable samples — incident-free rows that recorded
    a travel time — not the number of rows in the table for this road.

    ``std_dev`` is the **sample** standard deviation (``n - 1``), which is the
    estimator for a spread inferred from a sample rather than measured over a
    whole population. A single sample has no spread to estimate and reports
    ``0.0``; :func:`z_score` handles what that means.
    """

    road_u: Node
    road_v: Node
    condition: str
    count: int
    mean: float
    std_dev: float


@dataclass(frozen=True, slots=True)
class Baseline:
    """A frozen summary of the log, keyed by ``(road_u, road_v, condition)``.

    Keyed by condition as well as road, so a 09:00 observation is judged against
    that road's own 09:00-band history rather than against a pooled figure in
    which peak *is* the anomaly. The cost of that is a third as many samples per
    key, which is why the fallback exists and why :class:`Detection` reports the
    sample count it had.

    Frozen and handed around whole rather than queried live per request: it is a
    snapshot taken at a known moment, and a caller that could refresh it midway
    would be judging two observations against two different histories.
    """

    stats: Mapping[tuple[Node, Node, str], RoadStats]

    def for_edge(self, u: Node, v: Node, condition: str) -> RoadStats | None:
        """This road's history for one condition band, or ``None`` if it has none."""
        return self.stats.get((u, v, condition))


@dataclass(frozen=True, slots=True)
class Detection:
    """The verdict on one observed travel time.

    ``mean``, ``std_dev`` and ``z_score`` are ``None`` when the fallback decided,
    because no z-score was computed — reporting a zero would claim a measurement
    that was never taken. ``sample_count`` is reported either way, so a caller can
    always see how much evidence was behind the answer.

    ``z_score`` is also ``None`` when the score is not a finite number, which is
    the zero-spread case :func:`z_score` documents. JSON has no way to write an
    infinity — ``json.dumps`` refuses to with ``allow_nan=False``, and a browser's
    ``JSON.parse`` rejects it — so the verdict reports ``None`` and leaves the
    reading in ``reason``, where it is a sentence rather than a number. A reader
    can tell the two ``None``\\ s apart by ``std_dev``: zero there means there was
    no spread to divide by, and ``rule`` says which rule ran.
    """

    flagged: bool
    rule: str
    observed: float
    #: The road's modelled cost under the clock alone, incidents excluded.
    #: ``None`` when the road is impassable under those conditions.
    expected: float | None
    sample_count: int
    condition: str
    mean: float | None
    std_dev: float | None
    #: Standard deviations above the mean, or ``None`` when none was taken (the
    #: fallback ran) or none exists (a history with no spread). See the class
    #: docstring — an infinity would not survive the wire.
    z_score: float | None
    #: One sentence naming the number that decided it, for a human or a log line.
    reason: str


def build_baseline(rows: Iterable[TrafficLogRow]) -> Baseline:
    """Summarise logged rows into per-road, per-condition history.

    Two kinds of row are skipped, for the same reason: neither is an observation
    of a road being itself.

    - **Rows with an ``incident_type``.** A closure, an accident or an operator's
      report records a road *not* behaving normally, which is the thing being
      detected. Including them would teach the baseline that slow is normal.
    - **Rows with no ``travel_time``.** That is how a closed road is logged: it is
      impassable, so there is no travel time to record and ``incident_type``
      carries the reason. There is nothing to average.

    Every remaining row contributes exactly one sample, under the key its own
    ``traffic_condition`` names.
    """
    samples: dict[tuple[Node, Node, str], list[float]] = {}
    for row in rows:
        if row.incident_type is not None or row.travel_time is None:
            continue
        key = (row.road_u, row.road_v, row.traffic_condition)
        samples.setdefault(key, []).append(row.travel_time)

    return Baseline(
        stats={
            key: RoadStats(
                road_u=key[0],
                road_v=key[1],
                condition=key[2],
                count=len(values),
                mean=statistics.fmean(values),
                # stdev needs two points to have a spread at all, and raises on
                # one. A lone sample is perfectly repeatable as far as anything
                # can tell, which is what 0.0 says.
                std_dev=statistics.stdev(values) if len(values) >= 2 else 0.0,
            )
            for key, values in samples.items()
        }
    )


def z_score(observed: float, stats: RoadStats) -> float:
    """Standard deviations ``observed`` sits above this road's mean.

    Signed, so a road running faster than usual gives a negative z and is not
    flagged by a one-sided :data:`Z_THRESHOLD`.

    When the history has **no spread at all** — the ordinary case here, since the
    simulator is deterministic — the division has no answer, and the honest one is
    an infinity rather than a guard: a road that has only ever taken 10.0 s and
    reports 29.0 s is not a finite number of standard deviations out, because
    there is no standard deviation for it to be out by. Matching the history is
    ``0.0``: the observation is exactly as far out as the road always is, which is
    not at all.
    """
    if stats.std_dev > 0.0:
        return (observed - stats.mean) / stats.std_dev
    if observed == stats.mean:
        return 0.0
    return math.inf if observed > stats.mean else -math.inf


def detect(
    observed: float,
    *,
    stats: RoadStats | None,
    expected: float | None,
    condition: str,
) -> Detection:
    """Judge one observed travel time against a road's history.

    The single place the two rules meet, and the only thing a caller needs.

    With at least :data:`MIN_SAMPLES` samples the z-score decides: flagged when
    ``z > Z_THRESHOLD``. Below that — including when the road has no history at
    all — the z-score is skipped entirely and the rule-based expectation decides:
    flagged when ``observed > FALLBACK_FACTOR * expected``. Both comparisons are
    strict, so an observation sitting exactly on a threshold is not flagged.

    ``expected`` is ``None`` for a road that is impassable under these conditions.
    There is then nothing to be slower than, so nothing is flagged; a closed road
    is a legitimate state rather than an error, and the reason says which.

    The verdict's ``z_score`` is ``None`` rather than an infinity when the score
    is not finite, which :func:`z_score` explains is the ordinary case here. The
    flag is unaffected — the comparison is made against the real value — and the
    reason carries the number in words.
    """
    if stats is None or stats.count < MIN_SAMPLES:
        return _by_rule_of_thumb(observed, expected, condition, stats)

    z = z_score(observed, stats)
    flagged = z > Z_THRESHOLD
    return Detection(
        flagged=flagged,
        rule=Z_SCORE,
        observed=observed,
        expected=expected,
        sample_count=stats.count,
        condition=condition,
        mean=stats.mean,
        std_dev=stats.std_dev,
        z_score=z if math.isfinite(z) else None,
        reason=_describe_z(observed, stats, z, flagged),
    )


def _by_rule_of_thumb(
    observed: float, expected: float | None, condition: str, stats: RoadStats | None
) -> Detection:
    """The verdict while there are too few samples for a standard deviation."""
    count = 0 if stats is None else stats.count

    if expected is None:
        return Detection(
            flagged=False,
            rule=INSUFFICIENT_SAMPLES,
            observed=observed,
            expected=None,
            sample_count=count,
            condition=condition,
            mean=None,
            std_dev=None,
            z_score=None,
            reason=(
                f"the road is impassable under {condition} conditions, so there is "
                "no travel time to compare against"
            ),
        )

    ceiling = FALLBACK_FACTOR * expected
    flagged = observed > ceiling
    where = (
        f"above the modelled {condition} time of {expected:.1f}s"
        if flagged
        else f"within {FALLBACK_FACTOR}x the modelled {condition} time of "
        f"{expected:.1f}s"
    )
    return Detection(
        flagged=flagged,
        rule=INSUFFICIENT_SAMPLES,
        observed=observed,
        expected=expected,
        sample_count=count,
        condition=condition,
        mean=None,
        std_dev=None,
        z_score=None,
        reason=(
            f"{observed:.1f}s is {where}; with {count} sample(s) — fewer than the "
            f"{MIN_SAMPLES} a z-score needs — the {FALLBACK_FACTOR}x margin decides"
        ),
    )


def _describe_z(
    observed: float, stats: RoadStats, z: float, flagged: bool
) -> str:
    """The reason line for a z-score verdict, including the zero-spread case."""
    if stats.std_dev == 0.0:
        # Tested on the observation, not on `flagged`: a road running faster than
        # every sample on record also has a non-finite z, and calling that "an
        # exact match" would be the opposite of what happened.
        if observed == stats.mean:
            return (
                f"{observed:.1f}s matches this road's {stats.condition} history "
                f"exactly — {stats.count} samples with no spread at all"
            )
        if flagged:
            return (
                f"{observed:.1f}s differs from this road's {stats.condition} "
                f"history of {stats.mean:.1f}s, which has no spread across "
                f"{stats.count} samples; any difference is anomalous"
            )
        return (
            f"{observed:.1f}s is below this road's {stats.condition} history of "
            f"{stats.mean:.1f}s, which has no spread across {stats.count} "
            "samples; a road running fast is not flagged"
        )

    if flagged:
        return (
            f"{observed:.1f}s is {z:.1f} standard deviations above this road's "
            f"{stats.condition} mean of {stats.mean:.1f}s "
            f"(std dev {stats.std_dev:.2f}, threshold {Z_THRESHOLD})"
        )
    return (
        f"{observed:.1f}s is {z:.2f} standard deviations from this road's "
        f"{stats.condition} mean of {stats.mean:.1f}s, inside the "
        f"{Z_THRESHOLD} threshold"
    )
