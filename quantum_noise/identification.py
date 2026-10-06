"""Identify common versus independent perturbations from joint count data.

A researcher supplies a small circuit (two or three qubits), joint count
data, and candidate event time windows.  The functions here scan the legal
probability range [0, 1] for each window and compare three model
classes -- fully common, fully independent, and mixed -- by exact
multinomial likelihood.  When the data cannot separate the classes, every
plausible explanation is kept together with the reason it survives.

Follow-up evidence is tagged by mode: counts gathered while continuing
along already observed conditions ("observed") and counts gathered
after resampling the environment ("resample") are never pooled, and
the original study keeps the conditions it was created with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .gates import Circuit
from .noise import JointEnvironmentEvent, NoiseModel, TimeSchedule
from .simulator import simulate

MODEL_CLASSES = ("common", "independent", "mixed")
FOLLOWUP_MODES = ("observed", "resample")


@dataclass(frozen=True)
class CandidateWindow:
    """A candidate perturbation time window with probabilities to be fitted."""

    name: str
    start: float
    end: float
    after_gate: int
    qubits: tuple[int, ...]
    channel: str = "bit_flip"

    def event(self, common_probability: float, independent_probability: float) -> JointEnvironmentEvent:
        return JointEnvironmentEvent(
            name=self.name,
            start=self.start,
            end=self.end,
            after_gate=self.after_gate,
            qubits=self.qubits,
            common_probability=common_probability,
            independent_probability=independent_probability,
            channel=self.channel,
        )


@dataclass(frozen=True)
class FittedModel:
    """One legal (window, class, probabilities) combination."""

    window: str
    model_class: str
    common_probability: float
    independent_probability: float
    log_likelihood: float
    distribution: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "window": self.window,
            "model_class": self.model_class,
            "common_probability": self.common_probability,
            "independent_probability": self.independent_probability,
            "log_likelihood": self.log_likelihood,
            "distribution": dict(self.distribution),
        }


@dataclass(frozen=True)
class OutcomeDeviation:
    outcome: str
    empirical: float
    predicted: float
    standard_error: float
    confidence_95: tuple[float, float]
    consistent: bool


@dataclass(frozen=True)
class PredictionTest:
    """Goodness-of-fit of fresh independent counts against a prediction."""

    shots: int
    enough_samples: bool
    consistent: bool
    deviations: tuple[OutcomeDeviation, ...]

    @property
    def interpretation(self) -> str:
        if not self.enough_samples:
            return "inconclusive: expected counts are too low for a stable test"
        if self.consistent:
            return "fresh samples are consistent with the predicted joint distribution"
        return "fresh samples reject the predicted joint distribution at 95% confidence"


@dataclass(frozen=True)
class IdentificationResult:
    """Outcome of scanning all candidate windows and model classes."""

    shots: int
    enough_samples: bool
    status: str
    reasons: tuple[str, ...]
    best_by_class: Mapping[str, FittedModel]
    aic_by_class: Mapping[str, float]
    plausible: tuple[FittedModel, ...]
    marginals_warning: Optional[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "shots": self.shots,
            "enough_samples": self.enough_samples,
            "status": self.status,
            "reasons": list(self.reasons),
            "best_by_class": {
                name: model.to_dict() for name, model in self.best_by_class.items()
            },
            "aic_by_class": dict(self.aic_by_class),
            "plausible": [model.to_dict() for model in self.plausible],
            "marginals_warning": self.marginals_warning,
        }


_CLASS_PARAMETERS = {"common": 1, "independent": 1, "mixed": 2}


def _validate_counts(circuit: Circuit, counts: Mapping[str, int]) -> int:
    if circuit.num_qubits not in (2, 3):
        raise ValueError("identification supports two- or three-qubit circuits")
    if not counts:
        raise ValueError("counts must contain at least one outcome")
    shots = 0
    for outcome, count in counts.items():
        if not outcome or any(bit not in "01" for bit in outcome):
            raise ValueError(f"outcome {outcome!r} is not a bit string")
        if len(outcome) != circuit.num_qubits:
            raise ValueError(
                f"outcome {outcome!r} does not match {circuit.num_qubits} qubits"
            )
        if int(count) < 0:
            raise ValueError("counts must be nonnegative")
        shots += int(count)
    if shots <= 0:
        raise ValueError("counts must contain at least one shot")
    return shots


def _distribution_for(
    circuit: Circuit,
    schedule: TimeSchedule,
    window: CandidateWindow,
    common_probability: float,
    independent_probability: float,
) -> dict[str, float]:
    model = NoiseModel(
        name=f"identify-{window.name}",
        rules=(),
        groups=(),
        environment_events=(window.event(common_probability, independent_probability),),
    )
    return simulate(circuit, model, schedule=schedule).probabilities


def _log_likelihood(counts: Mapping[str, int], distribution: Mapping[str, float]) -> float:
    total = 0.0
    for outcome, count in counts.items():
        if count == 0:
            continue
        probability = float(distribution.get(outcome, 0.0))
        if probability <= 0.0:
            return -math.inf
        total += count * math.log(probability)
    return total


def _grid(points: int) -> list[float]:
    if points < 2:
        raise ValueError("grid_points must be at least 2")
    return [index / (points - 1) for index in range(points)]


def _parameter_pairs(model_class: str, grid: Sequence[float]) -> list[tuple[float, float]]:
    if model_class == "common":
        return [(c, 0.0) for c in grid]
    if model_class == "independent":
        return [(0.0, q) for q in grid]
    return [(c, q) for c in grid for q in grid]


def _marginals(distribution: Mapping[str, float], num_qubits: int) -> tuple[float, ...]:
    values = []
    for qubit in range(num_qubits):
        position = num_qubits - 1 - qubit
        values.append(
            sum(
                probability
                for outcome, probability in distribution.items()
                if (int(outcome, 2) >> position) & 1
            )
        )
    return tuple(values)


def fit_event_models(
    circuit: Circuit,
    schedule: TimeSchedule,
    counts: Mapping[str, int],
    windows: Sequence[CandidateWindow],
    *,
    grid_points: int = 21,
    plausibility: float = 3.84,
    minimum_expected_count: float = 5.0,
) -> IdentificationResult:
    """Scan legal perturbation combinations and rank the three model classes.

    Every (window, class, probabilities) combination with probabilities in
    [0, 1] is scored by exact multinomial log-likelihood.  Because the
    mixed class nests the fully common and fully independent classes, model
    classes are compared by AIC (a class stays plausible while its AIC is
    within 2 of the best); individual combinations inside a plausible
    class are kept while their likelihood-ratio statistic stays below
    "plausibility".  When several classes or windows survive, or when the
    sample is too small, the result stays inconclusive and reports why.
    """

    shots = _validate_counts(circuit, counts)
    if not windows:
        raise ValueError("at least one candidate window is required")
    names = [window.name for window in windows]
    if len(set(names)) != len(names):
        raise ValueError("candidate windows must have unique names")

    grid = _grid(grid_points)
    fits: list[FittedModel] = []
    for window in windows:
        for model_class in MODEL_CLASSES:
            for common, independent in _parameter_pairs(model_class, grid):
                distribution = _distribution_for(circuit, schedule, window, common, independent)
                fits.append(
                    FittedModel(
                        window=window.name,
                        model_class=model_class,
                        common_probability=common,
                        independent_probability=independent,
                        log_likelihood=_log_likelihood(counts, distribution),
                        distribution=distribution,
                    )
                )

    best_by_class: dict[str, FittedModel] = {}
    for model_class in MODEL_CLASSES:
        best_by_class[model_class] = max(
            (fit for fit in fits if fit.model_class == model_class),
            key=lambda fit: fit.log_likelihood,
        )
    aic_by_class = {
        model_class: -2.0 * best_by_class[model_class].log_likelihood
        + 2.0 * _CLASS_PARAMETERS[model_class]
        for model_class in MODEL_CLASSES
    }
    best_aic = min(aic_by_class.values())
    plausible_classes = {
        model_class
        for model_class, aic in aic_by_class.items()
        if aic < best_aic + 2.0
    }
    best_overall = max(fit.log_likelihood for fit in fits)
    plausible = tuple(
        fit
        for fit in fits
        if fit.model_class in plausible_classes
        and 2.0 * (best_overall - fit.log_likelihood) <= plausibility
    )

    reasons: list[str] = []
    best_fit = max(fits, key=lambda fit: fit.log_likelihood)
    expected = [
        float(best_fit.distribution.get(outcome, 0.0)) * shots
        for outcome in counts
        if counts[outcome] > 0
    ]
    enough_samples = bool(expected) and min(expected) >= minimum_expected_count
    if not enough_samples:
        reasons.append(
            "sample too small: some expected counts are below "
            f"{minimum_expected_count:g}, so sampling noise can mimic a model difference"
        )

    plausible_windows = {fit.window for fit in plausible}
    if len(plausible_windows) > 1:
        reasons.append(
            "parameters are not identifiable: windows "
            + ", ".join(sorted(plausible_windows))
            + " explain the data equally well"
        )
    if len(plausible_classes) > 1:
        reasons.append(
            "model classes not separated: "
            + ", ".join(sorted(plausible_classes))
            + " all remain plausible (AIC within 2 of the best class)"
        )

    marginals_warning = _marginals_warning(
        best_by_class["common"], best_by_class["independent"], circuit.num_qubits, shots
    )
    if marginals_warning:
        reasons.append(marginals_warning)

    if not enough_samples or len(plausible_classes) > 1 or len(plausible_windows) > 1:
        status = "inconclusive"
    else:
        status = "identified"
        reasons.append(
            f"only {best_fit.model_class} models remain plausible for window "
            f"{best_fit.window!r}"
        )

    return IdentificationResult(
        shots=shots,
        enough_samples=enough_samples,
        status=status,
        reasons=tuple(reasons),
        best_by_class=best_by_class,
        aic_by_class=aic_by_class,
        plausible=plausible,
        marginals_warning=marginals_warning,
    )


def _marginals_warning(
    common_fit: FittedModel,
    independent_fit: FittedModel,
    num_qubits: int,
    shots: int,
) -> Optional[str]:
    common_marginals = _marginals(common_fit.distribution, num_qubits)
    independent_marginals = _marginals(independent_fit.distribution, num_qubits)
    marginal_gap = max(
        abs(a - b) for a, b in zip(common_marginals, independent_marginals)
    )
    outcomes = set(common_fit.distribution) | set(independent_fit.distribution)
    joint_gap = 0.5 * sum(
        abs(
            float(common_fit.distribution.get(outcome, 0.0))
            - float(independent_fit.distribution.get(outcome, 0.0))
        )
        for outcome in outcomes
    )
    sampling_slack = 1.96 * math.sqrt(0.25 / max(1, shots))
    if marginal_gap <= sampling_slack and joint_gap > sampling_slack:
        return (
            "equal per-qubit marginals do not imply model equivalence: the best "
            "common and independent fits share marginals within sampling error "
            f"(gap {marginal_gap:.4g}) but their joint distributions differ by "
            f"{joint_gap:.4g} in total variation"
        )
    return None


def predict_joint_distribution(
    model: FittedModel,
    windows: Sequence[CandidateWindow],
    circuit: Circuit,
    schedule: TimeSchedule,
) -> dict[str, float]:
    """Predict the joint distribution of a fitted candidate on a new circuit."""

    match = [window for window in windows if window.name == model.window]
    if not match:
        raise ValueError(f"unknown candidate window {model.window!r}")
    return _distribution_for(
        circuit,
        schedule,
        match[0],
        model.common_probability,
        model.independent_probability,
    )


def test_prediction(
    predicted: Mapping[str, float],
    counts: Mapping[str, int],
    *,
    minimum_expected_count: float = 5.0,
) -> PredictionTest:
    """Check fresh independent counts against a predicted joint distribution."""

    shots = sum(int(count) for count in counts.values())
    if shots <= 0:
        raise ValueError("counts must contain at least one shot")
    for outcome in counts:
        if outcome not in predicted:
            raise ValueError(f"outcome {outcome!r} is outside the predicted support")
    deviations = []
    consistent = True
    for outcome in sorted(set(predicted) | set(counts)):
        probability = float(predicted.get(outcome, 0.0))
        empirical = counts.get(outcome, 0) / shots
        standard_error = math.sqrt(max(0.0, probability * (1.0 - probability)) / shots)
        margin = 1.96 * standard_error
        interval = (probability - margin, probability + margin)
        ok = interval[0] <= empirical <= interval[1]
        consistent = consistent and ok
        deviations.append(
            OutcomeDeviation(
                outcome=outcome,
                empirical=empirical,
                predicted=probability,
                standard_error=standard_error,
                confidence_95=interval,
                consistent=ok,
            )
        )
    positive_expectations = [
        float(probability) * shots
        for probability in predicted.values()
        if float(probability) > 0.0
    ]
    enough = bool(positive_expectations) and min(
        positive_expectations
    ) >= minimum_expected_count
    return PredictionTest(
        shots=shots,
        enough_samples=enough,
        consistent=consistent,
        deviations=tuple(deviations),
    )


@dataclass(frozen=True)
class EvidenceRecord:
    """One batch of counts tagged with the way it was gathered."""

    mode: str
    counts: Mapping[str, int]
    shots: int
    note: str = ""


@dataclass
class PerturbationStudy:
    """A study bound to its original conditions plus separated follow-up evidence.

    The original circuit, schedule, windows, and counts are fixed at
    creation; follow-up counts must declare whether they continue along the
    already observed conditions ("observed") or resample the
    environment ("resample").  Pooling never mixes the two modes.
    """

    study_id: str
    circuit: Circuit
    schedule: TimeSchedule
    windows: tuple[CandidateWindow, ...]
    counts: Mapping[str, int]
    _evidence: list[EvidenceRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.study_id:
            raise ValueError("study requires an id")
        _validate_counts(self.circuit, self.counts)
        self.counts = {key: int(value) for key, value in self.counts.items()}
        self.windows = tuple(self.windows)

    def identify(self, **kwargs) -> IdentificationResult:
        """Fit the original counts; never touches follow-up evidence."""

        return fit_event_models(
            self.circuit, self.schedule, self.counts, self.windows, **kwargs
        )

    def add_followup(
        self,
        counts: Mapping[str, int],
        mode: str,
        *,
        note: str = "",
    ) -> EvidenceRecord:
        if mode not in FOLLOWUP_MODES:
            raise ValueError(
                "follow-up mode must be 'observed' or 'resample'; "
                "evidence from the two modes must stay separate"
            )
        _validate_counts(self.circuit, counts)
        record = EvidenceRecord(
            mode=mode,
            counts={key: int(value) for key, value in counts.items()},
            shots=sum(int(value) for value in counts.values()),
            note=note,
        )
        self._evidence.append(record)
        return record

    def evidence(self, mode: Optional[str] = None) -> tuple[EvidenceRecord, ...]:
        if mode is None:
            return tuple(self._evidence)
        if mode not in FOLLOWUP_MODES:
            raise ValueError(f"unknown evidence mode {mode!r}")
        return tuple(record for record in self._evidence if record.mode == mode)

    def pooled_counts(self, mode: str) -> dict[str, int]:
        """Pool counts of a single mode; mixing modes is refused."""

        if mode not in FOLLOWUP_MODES:
            raise ValueError(
                "pooling requires an explicit mode ('observed' or 'resample'); "
                "continuation and resampling evidence must not be mixed"
            )
        pooled: dict[str, int] = {}
        for record in self._evidence:
            if record.mode != mode:
                continue
            for outcome, count in record.counts.items():
                pooled[outcome] = pooled.get(outcome, 0) + count
        if not pooled:
            raise ValueError(f"no {mode!r} follow-up evidence recorded")
        return pooled
