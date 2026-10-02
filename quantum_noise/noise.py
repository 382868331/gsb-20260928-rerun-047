"""Explicitly parameterized, physically valid noise channels.

Besides independent channels, this module supports *time-correlated* noise:
a :class:`CorrelatedNoiseGroup` couples error events at several circuit
positions through a two-state Markov chain, so the disturbances seen by two
operations can be correlated.  Correlation strength c = 0 recovers
independent errors; c -> 1 makes errors at adjacent sites coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt, inf
from typing import Optional, Sequence

from .gates import I, X, Y, Z
from .linalg import Matrix

Probability = float


def _probability(value: float, name: str) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def pauli_channel(p_x: float = 0.0, p_y: float = 0.0, p_z: float = 0.0) -> list[Matrix]:
    """Return weighted Kraus operators sqrt(p) P, with identity filling trace."""
    probabilities = [float(p_x), float(p_y), float(p_z)]
    if any(value < 0 for value in probabilities) or sum(probabilities) > 1 + 1e-12:
        raise ValueError("Pauli probabilities must be nonnegative and sum to at most 1")
    p_i = max(0.0, 1.0 - sum(probabilities))
    weighted: list[Matrix] = []
    for probability, operator in ((p_i, I), (p_x, X), (p_y, Y), (p_z, Z)):
        if probability > 0:
            scale = sqrt(probability)
            weighted.append([[scale * value for value in row] for row in operator])
    return weighted


def bit_flip(probability: Probability) -> list[Matrix]:
    _probability(probability, "bit-flip probability")
    return pauli_channel(p_x=probability)


def phase_flip(probability: Probability) -> list[Matrix]:
    _probability(probability, "phase-flip probability")
    return pauli_channel(p_z=probability)


def depolarizing(probability: Probability) -> list[Matrix]:
    """Single-qubit depolarizing channel with equal X, Y and Z probability."""
    _probability(probability, "depolarizing probability")
    each = probability / 3.0
    return pauli_channel(p_x=each, p_y=each, p_z=each)


def amplitude_damping(gamma: Probability) -> list[Matrix]:
    _probability(gamma, "amplitude-damping gamma")
    survival = sqrt(max(0.0, 1.0 - gamma))
    return [
        [[1.0 + 0j, 0j], [0j, survival + 0j]],
        [[0j, sqrt(gamma) + 0j], [0j, 0j]],
    ]


@dataclass(frozen=True)
class NoiseRule:
    """A noise insertion rule.

    ``qubits`` identifies fixed target qubits.  ``after_gate`` is zero-based and
    ``None`` means after the complete ideal circuit.  Rules are applied in list
    order, so ordering is part of the experiment specification.
    """

    channel: str
    probability: float = 0.0
    qubits: tuple[int, ...] = (0,)
    after_gate: Optional[int] = None
    params: dict[str, float] = field(default_factory=dict)

    def kraus_operators(self) -> list[Matrix]:
        if self.channel == "bit_flip":
            return bit_flip(self.probability)
        if self.channel == "phase_flip":
            return phase_flip(self.probability)
        if self.channel == "depolarizing":
            return depolarizing(self.probability)
        if self.channel == "amplitude_damping":
            gamma = self.params.get("gamma", self.probability)
            return amplitude_damping(gamma)
        if self.channel == "pauli":
            return pauli_channel(
                self.params.get("p_x", 0.0),
                self.params.get("p_y", 0.0),
                self.params.get("p_z", 0.0),
            )
        raise ValueError(f"unknown noise channel: {self.channel!r}")


@dataclass(frozen=True)
class Perturbation:
    """A transient external disturbance active over ``[start, end)``.

    While active the disturbance replaces the environmental noise strength
    of its target channels with ``probability`` (and optionally scales the
    correlation time).  Events at exactly ``end`` see the restored baseline
    environment; the interval is half-open, so an instantaneous disturbance
    (``start == end``) never changes any event.  ``priority`` resolves
    overlapping disturbances: a higher-priority disturbance wins; equal
    priorities follow the later entry in the perturbation list.
    """

    start: float
    end: float
    probability: Optional[float] = None
    correlation_time_scale: float = 1.0
    name: str = "perturbation"
    targets: Optional[frozenset[str]] = None
    priority: int = 0

    def __post_init__(self) -> None:
        if float(self.start) < 0.0:
            raise ValueError("perturbation start must be nonnegative")
        if float(self.end) < float(self.start):
            raise ValueError("perturbation end must not precede its start")
        if self.probability is not None:
            _probability(float(self.probability), "perturbation probability")
        if float(self.correlation_time_scale) < 0.0:
            raise ValueError("correlation_time_scale must be nonnegative")

    @property
    def instantaneous(self) -> bool:
        return float(self.end) == float(self.start)

    def active_at(self, time: float) -> bool:
        return float(self.start) <= float(time) < float(self.end)

    def overlaps(self, other: "Perturbation") -> bool:
        if self.instantaneous or other.instantaneous:
            return False
        return float(self.start) < float(other.end) and float(other.start) < float(self.end)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": float(self.start),
            "end": float(self.end),
            "probability": None if self.probability is None else float(self.probability),
            "correlation_time_scale": float(self.correlation_time_scale),
            "targets": None if self.targets is None else sorted(self.targets),
            "priority": int(self.priority),
        }

    @staticmethod
    def from_dict(data: dict) -> "Perturbation":
        targets = data.get("targets")
        return Perturbation(
            start=float(data["start"]),
            end=float(data["end"]),
            probability=None if data.get("probability") is None else float(data["probability"]),
            correlation_time_scale=float(data.get("correlation_time_scale", 1.0)),
            name=data.get("name", "perturbation"),
            targets=None if targets is None else frozenset(targets),
            priority=int(data.get("priority", 0)),
        )


@dataclass(frozen=True)
class EnvironmentTimeline:
    """Resolved environmental state along physical time."""

    perturbations: tuple[Perturbation, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(item, Perturbation) for item in self.perturbations):
            raise ValueError("timeline entries must be Perturbation instances")

    def active(self, time: float) -> tuple[Perturbation, ...]:
        active = [item for item in self.perturbations if item.active_at(time)]
        active.sort(key=lambda item: (item.priority, self.perturbations.index(item)))
        return tuple(reversed(active))

    def effective_at(
        self, time: float, channel: str
    ) -> tuple[Optional[Perturbation], Optional[float]]:
        """Return the winning disturbance and effective probability at ``time``."""
        for perturbation in self.active(time):
            if perturbation.targets is None or channel in perturbation.targets:
                return perturbation, perturbation.probability
        return None, None

    def winner_at(self, time: float) -> Optional[Perturbation]:
        active = self.active(time)
        return active[0] if active else None

    def correlation_over(
        self,
        correlation: float,
        start_time: float,
        end_time: float,
        base_time: float,
        channel: str,
    ) -> float:
        """Correlation surviving a possibly piecewise-perturbed interval."""
        elapsed = end_time - start_time
        if elapsed < 0.0:
            raise ValueError("elapsed time must be nonnegative")
        if elapsed == 0.0 or base_time == inf:
            return correlation
        boundaries = {start_time, end_time}
        for perturbation in self.perturbations:
            if perturbation.targets is not None and channel not in perturbation.targets:
                continue
            if start_time < perturbation.start < end_time:
                boundaries.add(float(perturbation.start))
            if start_time < perturbation.end <= end_time:
                boundaries.add(float(perturbation.end))
        decay = 0.0
        ordered = sorted(boundaries)
        for left, right in zip(ordered, ordered[1:]):
            duration = right - left
            if duration <= 0.0:
                continue
            winner, _probability = self.effective_at((left + right) / 2.0, channel)
            scale = 1.0 if winner is None else winner.correlation_time_scale
            if scale == 0.0:
                return 0.0
            decay += duration / (base_time * scale)
        return correlation * exp(-decay)

    def change_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for index, perturbation in enumerate(self.perturbations):
            overlaps = [
                other.name
                for other_index, other in enumerate(self.perturbations)
                if other_index != index and perturbation.overlaps(other)
            ]
            for kind, time in (("start", perturbation.start), ("end", perturbation.end)):
                # End events sit on the half-open boundary, so active_at(end)
                # describes the restored environment.
                winning = self.winner_at(time)
                probability = winning.probability if winning is not None else None
                records.append(
                    {
                        "perturbation": perturbation.name,
                        "kind": kind,
                        "time": float(time),
                        "time_inclusive": kind == "start",
                        "probability": probability,
                        "correlation_time_scale": perturbation.correlation_time_scale,
                        "overlapping": overlaps,
                        "environment_owner": winning.name if winning is not None else None,
                        "restored": kind == "end",
                        "restored_environment": winning.name if winning is not None else "baseline",
                    }
                )
        records.sort(key=lambda record: (record["time"], 0 if record["kind"] == "start" else 1))
        return records

    def to_list(self) -> list[dict]:
        return [item.to_dict() for item in self.perturbations]

    @staticmethod
    def from_list(data: Optional[Sequence[dict]]) -> "EnvironmentTimeline":
        if not data:
            return EnvironmentTimeline()
        return EnvironmentTimeline(tuple(Perturbation.from_dict(item) for item in data))


def direct_perturbed_probabilities(
    base_probability: float,
    persistent_probability: float,
    perturbation: Perturbation,
    event_times: Sequence[float],
) -> list[float]:
    """Direct short-circuit reference for one channel at known event times."""
    results = []
    for time in event_times:
        if perturbation.active_at(time):
            results.append(persistent_probability)
        else:
            results.append(base_probability)
    return results


_CORRELATED_ERROR_OPERATORS = {
    "bit_flip": X,
    "phase_flip": Z,
}


@dataclass(frozen=True)
class TimeSchedule:
    """Physical timing of circuit operations and subsequent idle intervals.

    Both sequences are indexed by circuit operation.  A correlated event at
    ``after_gate=i`` happens after ``operation_durations[i]`` and
    ``idle_after[i]`` have elapsed, so the same circuit can be compared under
    different timings without changing its gate sequence.
    """

    operation_durations: tuple[float, ...] = ()
    idle_after: tuple[float, ...] = ()
    name: str = "schedule"

    def __post_init__(self) -> None:
        if not self.operation_durations:
            raise ValueError("time schedule must contain at least one operation")
        if any(float(value) < 0.0 for value in self.operation_durations):
            raise ValueError("operation durations must be nonnegative")
        if any(float(value) < 0.0 for value in self.idle_after):
            raise ValueError("idle durations must be nonnegative")

    @property
    def num_operations(self) -> int:
        return len(self.operation_durations)

    def normalized(self, num_operations: int) -> "TimeSchedule":
        if self.operation_durations and len(self.operation_durations) != num_operations:
            raise ValueError("schedule does not match circuit operation count")
        idle = list(self.idle_after)
        if len(idle) > num_operations:
            raise ValueError("schedule contains too many idle intervals")
        idle.extend([0.0] * (num_operations - len(idle)))
        if not self.operation_durations:
            durations = [1.0] * num_operations
        else:
            durations = [float(value) for value in self.operation_durations]
        return TimeSchedule(tuple(durations), tuple(idle), self.name)

    def event_time(self, after_gate: int) -> float:
        """Return the time at the end of operation ``after_gate`` and its idle gap."""
        if not 0 <= after_gate < self.num_operations:
            raise ValueError("operation index is outside schedule")
        return sum(self.operation_durations[: after_gate + 1]) + sum(
            self.idle_after[: after_gate + 1]
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "operation_durations": list(self.operation_durations),
            "idle_after": list(self.idle_after),
        }

    @staticmethod
    def from_dict(data: dict) -> "TimeSchedule":
        return TimeSchedule(
            operation_durations=tuple(float(v) for v in data.get("operation_durations", ())),
            idle_after=tuple(float(v) for v in data.get("idle_after", ())),
            name=data.get("name", "schedule"),
        )


@dataclass(frozen=True)
class CorrelatedNoiseGroup:
    """A time-correlated error process across several circuit positions.

    ``sites`` are ``(after_gate, qubit)`` pairs in temporal order.  Error
    occurrences follow a two-state Markov chain with marginal probability
    ``probability`` and adjacent-site Pearson correlation ``correlation``:

    * P(error at t+1 | error at t)    = p + c * (1 - p)
    * P(error at t+1 | no error at t) = p * (1 - c)

    Correlations between sites separated by k steps decay as c**k.  The joint
    trajectory distribution is enumerated exactly by the simulator; it is
    never replaced by independent per-site sampling.
    """

    channel: str
    sites: tuple[tuple[int, int], ...] = ()
    probability: float = 0.0
    correlation: float = 0.0
    name: str = "correlated"
    correlation_time: float = inf

    def __post_init__(self) -> None:
        if self.channel not in _CORRELATED_ERROR_OPERATORS:
            raise ValueError(
                "correlated groups support "
                f"{sorted(_CORRELATED_ERROR_OPERATORS)}, not {self.channel!r}"
            )
        _probability(self.probability, "correlated noise probability")
        if len(self.sites) < 2:
            raise ValueError("a correlated group needs at least two sites")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError("correlated group sites must be distinct")
        p = self.probability
        if p in (0.0, 1.0):
            lower = -1.0
        else:
            lower = max(-p / (1.0 - p), -(1.0 - p) / p)
        if not lower - 1e-12 <= self.correlation <= 1.0 + 1e-12:
            raise ValueError(
                f"correlation must be in [{lower:.6g}, 1] for p={p:g}"
            )
        if self.correlation_time <= 0.0:
            raise ValueError("correlation_time must be positive")

    def correlation_after(self, elapsed_time: Optional[float]) -> float:
        """Correlation across an elapsed physical time gap."""
        if elapsed_time is None or self.correlation_time == inf:
            return self.correlation
        if elapsed_time < 0.0:
            raise ValueError("elapsed time must be nonnegative")
        return self.correlation * exp(-float(elapsed_time) / self.correlation_time)

    @property
    def error_operator(self) -> Matrix:
        return _CORRELATED_ERROR_OPERATORS[self.channel]

    def trajectories(self) -> list[tuple[tuple[int, ...], float]]:
        """Enumerate (error pattern, joint probability) pairs exactly."""
        p = self.probability
        c = self.correlation
        stay_error = p + c * (1.0 - p)
        become_error = p * (1.0 - c)
        results: list[tuple[tuple[int, ...], float]] = [((), 1.0)]
        for site_index in range(len(self.sites)):
            next_results: list[tuple[tuple[int, ...], float]] = []
            for pattern, weight in results:
                if site_index == 0:
                    p_error = p
                else:
                    p_error = stay_error if pattern[-1] == 1 else become_error
                p_error = min(1.0, max(0.0, p_error))
                next_results.append((pattern + (1,), weight * p_error))
                next_results.append((pattern + (0,), weight * (1.0 - p_error)))
            results = next_results
        return [(pattern, weight) for pattern, weight in results if weight > 0.0]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "channel": self.channel,
            "sites": [list(site) for site in self.sites],
            "probability": self.probability,
            "correlation": self.correlation,
            "correlation_time": None if self.correlation_time == inf else self.correlation_time,
        }

    @staticmethod
    def from_dict(data: dict) -> "CorrelatedNoiseGroup":
        return CorrelatedNoiseGroup(
            channel=data["channel"],
            sites=tuple(tuple(site) for site in data["sites"]),
            probability=float(data["probability"]),
            correlation=float(data.get("correlation", 0.0)),
            name=data.get("name", "correlated"),
            correlation_time=float(data.get("correlation_time", inf))
            if data.get("correlation_time") is not None
            else inf,
        )


@dataclass(frozen=True)
class NoiseModel:
    name: str
    rules: Sequence[NoiseRule] = field(default_factory=tuple)
    groups: Sequence[CorrelatedNoiseGroup] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rules": [
                {
                    "channel": rule.channel,
                    "probability": rule.probability,
                    "qubits": list(rule.qubits),
                    "after_gate": rule.after_gate,
                    "params": dict(rule.params),
                }
                for rule in self.rules
            ],
            "groups": [group.to_dict() for group in self.groups],
        }

    @staticmethod
    def from_dict(data: dict) -> "NoiseModel":
        return NoiseModel(
            name=data["name"],
            rules=tuple(
                NoiseRule(
                    channel=item["channel"],
                    probability=float(item.get("probability", 0.0)),
                    qubits=tuple(item.get("qubits", (0,))),
                    after_gate=item.get("after_gate"),
                    params=dict(item.get("params", {})),
                )
                    for item in data.get("rules", [])
            ),
            groups=tuple(
                CorrelatedNoiseGroup.from_dict(item)
                for item in data.get("groups", [])
            ),
        )
