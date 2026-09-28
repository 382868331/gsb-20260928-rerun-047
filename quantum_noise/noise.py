"""Explicitly parameterized, physically valid noise channels."""

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


@dataclass(frozen=True)
class NoiseModel:
    name: str
    rules: Sequence[NoiseRule] = field(default_factory=tuple)

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
        )
