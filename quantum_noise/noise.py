"""Explicitly parameterized, physically valid noise channels.

Besides independent channels, this module supports *time-correlated* noise:
a :class:`CorrelatedNoiseGroup` couples error events at several circuit
positions through a two-state Markov chain, so the disturbances seen by two
operations can be correlated.  Correlation strength c = 0 recovers
independent errors; c -> 1 makes errors at adjacent sites coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
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


_CORRELATED_ERROR_OPERATORS = {
    "bit_flip": X,
    "phase_flip": Z,
}


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
        }

    @staticmethod
    def from_dict(data: dict) -> "CorrelatedNoiseGroup":
        return CorrelatedNoiseGroup(
            channel=data["channel"],
            sites=tuple(tuple(site) for site in data["sites"]),
            probability=float(data["probability"]),
            correlation=float(data.get("correlation", 0.0)),
            name=data.get("name", "correlated"),
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
