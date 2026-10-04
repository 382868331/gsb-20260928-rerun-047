"""Durable environmental events shared by designated qubits.

An EnvironmentEvent is a finite-duration environmental process that
targets several qubits at once.  During the active window [start, end)
the event fires at every matching time-schedule boundary; each firing
uses one shared random realization:

* with probability common_probability a common flip is drawn and the same
  Pauli operator is applied to every target qubit;
* independently, each target qubit draws an independent flip with
  probability independent_probability.

The joint realization distribution is enumerated exactly (it is never
re-sampled per qubit), so fully common flips, fully independent flips and
mixtures remain distinguishable in the joint measurement distribution.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

from .gates import X, Z
from .noise import _probability


ENVIRONMENT_EVENT_OPERATORS = {
    "bit_flip": X,
    "phase_flip": Z,
}


class EnvironmentEvent:
    """A finite-duration common environmental event.

    While active over [start, end) the event fires at every schedule
    boundary inside the window.  One firing produces one shared flip
    (identical operator on every qubit in qubits) with probability
    common_probability and one independent flip per qubit with probability
    independent_probability; the two draws are independent.  start == end
    is legal and fires nowhere (zero-duration control).
    """

    def __init__(
        self,
        start: float,
        end: float,
        qubits: Sequence[int],
        common_probability: float = 0.0,
        independent_probability: float = 0.0,
        channel: str = "bit_flip",
        name: str = "environment-event",
        priority: int = 0,
    ) -> None:
        if float(start) < 0.0:
            raise ValueError("environment event start must be nonnegative")
        if float(end) < float(start):
            raise ValueError("environment event end must not precede its start")
        if channel not in ENVIRONMENT_EVENT_OPERATORS:
            raise ValueError(
                "environment events support "
                f"{sorted(ENVIRONMENT_EVENT_OPERATORS)}, not {channel!r}"
            )
        target_qubits = tuple(int(qubit) for qubit in qubits)
        if not target_qubits:
            raise ValueError("environment event must target at least one qubit")
        if len(set(target_qubits)) != len(target_qubits):
            raise ValueError("environment event qubits must be distinct")
        _probability(float(common_probability), "common_probability")
        _probability(float(independent_probability), "independent_probability")
        self.start = float(start)
        self.end = float(end)
        self.qubits = target_qubits
        self.common_probability = float(common_probability)
        self.independent_probability = float(independent_probability)
        self.channel = channel
        self.name = name
        self.priority = int(priority)

    @property
    def instantaneous(self) -> bool:
        return self.end == self.start

    def active_at(self, time: float) -> bool:
        return self.start <= float(time) < self.end

    @property
    def error_operator(self):
        return ENVIRONMENT_EVENT_OPERATORS[self.channel]

    def firing_indices(self, schedule, measurement_indices: Sequence[int] = ()) -> tuple[int, ...]:
        """Return non-measurement operation boundaries inside [start, end)."""
        measurements = set(measurement_indices)
        return tuple(
            index
            for index in range(schedule.num_operations)
            if index not in measurements and self.active_at(schedule.event_time(index))
        )

    def realizations(self) -> list[tuple[tuple[int, ...], float]]:
        """Exact joint (common bit, per-qubit independent bits) distribution.

        The common flip is drawn once; independent flips are drawn per
        qubit but enumerated jointly, so the shared component is never
        split into marginal resampling per qubit.
        """
        pc = self.common_probability
        pi = self.independent_probability
        results: list[tuple[tuple[int, ...], float]] = []
        for common_bit in (1, 0):
            common_weight = pc if common_bit else 1.0 - pc
            for independent_bits in product((1, 0), repeat=len(self.qubits)):
                weight = common_weight
                for bit in independent_bits:
                    weight *= pi if bit else 1.0 - pi
                results.append(((common_bit,) + tuple(independent_bits), weight))
        return [(bits, weight) for bits, weight in results if weight > 0.0]

    def realization_flips(self, bits: Sequence[int]) -> tuple[int, ...]:
        """Map a realization to the XOR flip bit of each target qubit."""
        common_bit = bits[0]
        return tuple(common_bit ^ independent for independent in bits[1:])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "qubits": list(self.qubits),
            "common_probability": self.common_probability,
            "independent_probability": self.independent_probability,
            "channel": self.channel,
            "priority": self.priority,
        }

    @staticmethod
    def from_dict(data: dict) -> "EnvironmentEvent":
        return EnvironmentEvent(
            start=float(data["start"]),
            end=float(data["end"]),
            qubits=tuple(data["qubits"]),
            common_probability=float(data.get("common_probability", 0.0)),
            independent_probability=float(data.get("independent_probability", 0.0)),
            channel=data.get("channel", "bit_flip"),
            name=data.get("name", "environment-event"),
            priority=int(data.get("priority", 0)),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvironmentEvent):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(
            (
                self.start,
                self.end,
                self.qubits,
                self.common_probability,
                self.independent_probability,
                self.channel,
                self.name,
                self.priority,
            )
        )

    def __repr__(self) -> str:
        return (
            f"EnvironmentEvent(name={self.name!r}, start={self.start:g}, "
            f"end={self.end:g}, qubits={self.qubits!r}, "
            f"common={self.common_probability:g}, "
            f"independent={self.independent_probability:g})"
        )


class EventCancellation:
    """A recorded decision to revoke an event at a definite physical time.

    A cancellation only prevents firings at boundaries t >= time; a
    duration event that already fired before the cancellation keeps its
    observed realization, so recorded studies are never rewritten.
    """

    def __init__(self, event_name: str, time: float) -> None:
        if float(time) < 0.0:
            raise ValueError("cancellation time must be nonnegative")
        self.event_name = event_name
        self.time = float(time)

    def cancels(self, event: EnvironmentEvent, firing_time: float) -> bool:
        return self.event_name == event.name and firing_time >= self.time

    def to_dict(self) -> dict:
        return {"event": self.event_name, "time": self.time}

    @staticmethod
    def from_dict(data: dict) -> "EventCancellation":
        return EventCancellation(data["event"], float(data["time"]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventCancellation):
            return NotImplemented
        return self.event_name == other.event_name and self.time == other.time

    def __hash__(self) -> int:
        return hash((self.event_name, self.time))


def firing_plan(
    events: Sequence[EnvironmentEvent],
    schedule,
    cancellations: Sequence[EventCancellation] = (),
    measurement_indices: Sequence[int] = (),
) -> dict[int, list[tuple[int, EnvironmentEvent]]]:
    """Map each operation boundary to (event index, event) firings.

    Firings at one boundary are ordered by priority (high first); ties
    follow event definition order.  Zero-duration events and firings
    revoked by a cancellation are omitted.
    """
    plan: dict[int, list[tuple[int, EnvironmentEvent]]] = {}
    for event_index, event in enumerate(events):
        for operation_index in event.firing_indices(schedule, measurement_indices):
            firing_time = schedule.event_time(operation_index)
            if any(
                cancellation.cancels(event, firing_time)
                for cancellation in cancellations
            ):
                continue
            plan.setdefault(operation_index, []).append((event_index, event))
    for entries in plan.values():
        entries.sort(key=lambda entry: (-entry[1].priority, entry[0]))
    return plan


def event_joint_probabilities(
    events: Sequence[EnvironmentEvent],
) -> list[tuple[list[tuple[int, ...]], float]]:
    """Exact product distribution over one shared realization per event.

    Each event contributes one shared draw common to all its firings;
    distinct events are mutually independent.  Returns (realization bits
    per event, joint probability) pairs.
    """
    per_event = [event.realizations() for event in events]
    combined: list[tuple[list[tuple[int, ...]], float]] = []
    for choices in product(*per_event):
        bits = [choice[0] for choice in choices]
        weight = 1.0
        for choice in choices:
            weight *= choice[1]
        if weight > 0.0:
            combined.append((bits, weight))
    return combined
