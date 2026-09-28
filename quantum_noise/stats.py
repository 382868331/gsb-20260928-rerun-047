"""Measurement statistics with explicit low-confidence handling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ProportionEstimate:
    outcome: str
    count: int
    shots: int
    empirical: float
    exact: float
    standard_error: float
    confidence_95: tuple[float, float]

    @property
    def margin_95(self) -> float:
        return (self.confidence_95[1] - self.confidence_95[0]) / 2.0


@dataclass(frozen=True)
class DifferenceTest:
    outcome: str
    observed_difference: float
    standard_error: float
    confidence_95: tuple[float, float]
    significant: bool
    exact_difference: float
    has_exact_effect: bool
    minimum_expected_count: float
    enough_samples: bool

    @property
    def interpretation(self) -> str:
        if not self.enough_samples:
            return "inconclusive: expected counts are too low for a stable binomial test"
        if not self.has_exact_effect:
            return "no deterministic effect: ideal and noisy exact probabilities are equal"
        if not self.significant:
            return "no detected deterministic effect at 95% confidence"
        direction = "higher" if self.observed_difference > 0 else "lower"
        return f"detected {direction} noisy frequency at 95% confidence"


def proportions(
    counts: Mapping[str, int], exact_probabilities: Mapping[str, float]
) -> dict[str, ProportionEstimate]:
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("counts must contain at least one shot")
    estimates = {}
    for outcome, count in counts.items():
        empirical = count / shots
        exact = float(exact_probabilities.get(outcome, 0.0))
        standard_error = math.sqrt(max(0.0, empirical * (1.0 - empirical)) / shots)
        margin = 1.96 * standard_error
        estimates[outcome] = ProportionEstimate(
            outcome=outcome,
            count=count,
            shots=shots,
            empirical=empirical,
            exact=exact,
            standard_error=standard_error,
            confidence_95=(
                max(0.0, empirical - margin),
                min(1.0, empirical + margin),
            ),
        )
    return estimates


def compare_counts(
    ideal_counts: Mapping[str, int],
    noisy_counts: Mapping[str, int],
    ideal_probabilities: Mapping[str, float],
    noisy_probabilities: Mapping[str, float],
    outcomes: Sequence[str] | None = None,
    minimum_expected_count: float = 5.0,
) -> dict[str, DifferenceTest]:
    """Compare independent ideal and noisy samples using a two-proportion interval.

    Normal-approximation tests are deliberately suppressed when any exact
    expectation for an outcome is below ``minimum_expected_count``.  This keeps
    rare-event binomial fluctuations from being called a determined effect.
    """

    ideal_shots = sum(ideal_counts.values())
    noisy_shots = sum(noisy_counts.values())
    if ideal_shots <= 0 or noisy_shots <= 0:
        raise ValueError("both count tables must contain at least one shot")
    selected = list(outcomes) if outcomes is not None else sorted(set(ideal_counts) | set(noisy_counts))
    tests: dict[str, DifferenceTest] = {}
    for outcome in selected:
        ideal_p = ideal_counts.get(outcome, 0) / ideal_shots
        noisy_p = noisy_counts.get(outcome, 0) / noisy_shots
        ideal_exact = float(ideal_probabilities.get(outcome, 0.0))
        noisy_exact = float(noisy_probabilities.get(outcome, 0.0))
        smallest_expectation = min(
            ideal_exact * ideal_shots,
            (1.0 - ideal_exact) * ideal_shots,
            noisy_exact * noisy_shots,
            (1.0 - noisy_exact) * noisy_shots,
        )
        enough = smallest_expectation >= minimum_expected_count
        standard_error = math.sqrt(
            ideal_p * (1.0 - ideal_p) / ideal_shots
            + noisy_p * (1.0 - noisy_p) / noisy_shots
        )
        difference = noisy_p - ideal_p
        margin = 1.96 * standard_error
        interval = (difference - margin, difference + margin)
        exact_difference = noisy_exact - ideal_exact
        tests[outcome] = DifferenceTest(
            outcome=outcome,
            observed_difference=difference,
            standard_error=standard_error,
            confidence_95=interval,
            exact_difference=exact_difference,
            has_exact_effect=abs(exact_difference) > 1e-12,
            minimum_expected_count=smallest_expectation,
            enough_samples=enough,
            significant=enough and abs(exact_difference) > 1e-12
            and (interval[0] > 0.0 or interval[1] < 0.0),
        )
    return tests


def total_variation_distance(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    outcomes = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in outcomes)
