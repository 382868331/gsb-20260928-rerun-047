"""Time-correlated Pauli-noise trajectories.

Noise at two insertion times is a stationary two-state Markov chain per
qubit.  ``correlation`` is the lag correlation coefficient of the binary
error indicator: with probability :math:`p` of an error at every step and
correlation :math:`\\rho`, the joint distribution is

    P(e1=1, e2=1) = p^2 + rho*p*(1-p)
    P(e1=e2)    = 1 - 2*p*(1-p)*(1-rho)

so ``rho = 0`` reproduces independent draws and ``rho = 1`` makes both
perturbations identical.  Sampling the trajectory is driven by a noise-only
random stream, separate from the stream that draws measurement outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Optional, Sequence

from .gates import I, X, Y, Z
from .linalg import Matrix


def _probability(value: float, name: str) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


_PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


@dataclass(frozen=True)
class CorrelatedNoiseEvent:
    """A Pauli perturbation site inserted after gate ``after_gate``.

    Events sharing a ``trajectory`` key (by default the target qubit) form one
    correlated Markov chain in chronological (insertion) order.
    """

    channel: str  # "bit_flip" | "phase_flip" | "pauli_x" ...
    after_gate: int
    qubit: int
    probability: float = 0.0
    trajectory: Optional[str] = None
    correlation: float = 0.0
    params: dict[str, float] = field(default_factory=dict)

    def key(self) -> str:
        return self.trajectory if self.trajectory is not None else f"q{self.qubit}"

    def error_pauli(self) -> Matrix:
        table = {
            "bit_flip": X,
            "phase_flip": Z,
            "pauli_x": X,
            "pauli_y": Y,
            "pauli_z": Z,
        }
        if self.channel not in table:
            raise ValueError(f"correlated noise supports Pauli errors, got {self.channel!r}")
        return table[self.channel]

    def validate(self) -> None:
        _probability(self.probability, "error probability")
        p = self.probability
        rho = float(self.correlation)
        lower = -min(p, 1.0 - p) / max(p, 1.0 - p)
        if not lower - 1e-12 <= rho <= 1.0 + 1e-12:
            raise ValueError(
                f"correlation {rho:g} infeasible for p={p:g}; allowed range [{lower:g}, 1]"
            )


@dataclass(frozen=True)
class CorrelatedNoiseModel:
    name: str
    events: Sequence[CorrelatedNoiseEvent] = field(default_factory=tuple)

    def ordered_events(self) -> list[CorrelatedNoiseEvent]:
        return sorted(self.events, key=lambda event: event.after_gate)

    def validate(self, num_gates: int) -> None:
        for event in self.events:
            event.validate()
            if not 0 <= event.after_gate < num_gates:
                raise ValueError("correlated event points outside circuit operation indices")
            if event.qubit < 0:
                raise ValueError("event qubit index must be nonnegative")
        keys = {event.key() for event in self.events}
        for key in keys:
            chain = [event for event in self.ordered_events() if event.key() == key]
            qubits = {event.qubit for event in chain}
            if len(qubits) != 1:
                raise ValueError(f"trajectory {key!r} mixes qubits")
            probabilities = {round(event.probability, 12) for event in chain}
            correlations = {round(event.correlation, 12) for event in chain}
            if len(probabilities) != 1:
                raise ValueError(f"trajectory {key!r} needs a stationary probability")
            if len(correlations) != 1:
                raise ValueError(f"trajectory {key!r} needs one stationary correlation")

    def to_dict(self) -> dict:
        return {
            "kind": "correlated",
            "name": self.name,
            "events": [
                {
                    "channel": event.channel,
                    "after_gate": event.after_gate,
                    "qubit": event.qubit,
                    "probability": event.probability,
                    "trajectory": event.trajectory,
                    "correlation": event.correlation,
                    "params": dict(event.params),
                }
                for event in self.events
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "CorrelatedNoiseModel":
        return CorrelatedNoiseModel(
            name=data["name"],
            events=tuple(
                CorrelatedNoiseEvent(
                    channel=item["channel"],
                    after_gate=int(item["after_gate"]),
                    qubit=int(item["qubit"]),
                    probability=float(item.get("probability", 0.0)),
                    trajectory=item.get("trajectory"),
                    correlation=float(item.get("correlation", 0.0)),
                    params=dict(item.get("params", {})),
                )
                for item in data.get("events", [])
            ),
        )


def transition_probability(p: float, rho: float, previous_error: int) -> float:
    """P(e_next = 1 | e_previous) for the stationary binary Markov chain."""

    if previous_error:
        return p + (1.0 - p) * rho
    return p * (1.0 - rho)


def sample_trajectory(
    events: Sequence[CorrelatedNoiseEvent], rng: random.Random
) -> dict[int, int]:
    """Draw error indicators (0/1) for every event id along one correlated path.

    ``events`` must all share one trajectory key and be insertion ordered.
    The Markov transition is used, so repeated calls describe temporally
    correlated perturbations rather than independent resampling.
    """

    indicators: dict[int, int] = {}
    previous: Optional[int] = None
    for event in events:
        if previous is None:
            error = 1 if rng.random() < event.probability else 0
        else:
            rate = transition_probability(event.probability, event.correlation, previous)
            error = 1 if rng.random() < rate else 0
        indicators[id(event)] = error
        previous = error
    return indicators


def trajectory_paths(
    events: Sequence[CorrelatedNoiseEvent],
) -> list[tuple[tuple[int, ...], float]]:
    """Enumerate all binary error paths with exact Markov-chain probabilities."""

    paths: list[tuple[tuple[int, ...], float]] = [((), 1.0)]
    for step, event in enumerate(events):
        extended = []
        for values, weight in paths:
            if step == 0:
                rates = (1.0 - event.probability, event.probability)
            else:
                previous = values[-1]
                rate_one = transition_probability(event.probability, event.correlation, previous)
                rates = (1.0 - rate_one, rate_one)
            for error, conditional_probability in enumerate(rates):
                if conditional_probability > 0:
                    extended.append((values + (error,), weight * conditional_probability))
        paths = extended
    return paths
