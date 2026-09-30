"""Explicit time schedules for experiments with finite-duration disturbances.

A :class:`Schedule` attaches a duration and an *idle wait* to every circuit
operation.  The schedule is part of the experiment specification: the same
operation sequence executed with different timings is a different experiment,
because the correlation of a time-correlated environment depends on the elapsed
time between noise sites (including idle intervals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class Schedule:
    """Timing of a fixed operation sequence.

    ``durations[i]`` is the duration of operation ``i`` and ``idle_after[i]``
    is the idle interval inserted after operation ``i``.  Trailing entries may
    be omitted; missing durations default to ``default_duration`` and missing
    idle intervals default to zero.
    """

    durations: tuple[float, ...] = ()
    idle_after: tuple[float, ...] = ()
    default_duration: float = 1.0
    name: str = "schedule"

    def __post_init__(self) -> None:
        if self.default_duration < 0.0:
            raise ValueError("default_duration must be nonnegative")
        for value in self.durations:
            if float(value) < 0.0:
                raise ValueError("operation durations must be nonnegative")
        for value in self.idle_after:
            if float(value) < 0.0:
                raise ValueError("idle intervals must be nonnegative")

    def normalized(self, num_operations: int) -> "Schedule":
        if len(self.durations) > num_operations or len(self.idle_after) > num_operations:
            raise ValueError("schedule has more entries than circuit operations")
        durations = tuple(self.durations) + tuple(
            self.default_duration
            for _ in range(num_operations - len(self.durations))
        )
        idle_after = tuple(self.idle_after) + tuple(
            0.0 for _ in range(num_operations - len(self.idle_after))
        )
        return Schedule(durations, idle_after, self.default_duration, self.name)

    def time_before(self, operation_index: int) -> float:
        """Absolute time at which operation ``operation_index`` starts."""
        return float(
            sum(self.durations[:operation_index])
            + sum(self.idle_after[:operation_index])
        )

    def site_time(self, operation_index: int) -> float:
        """Absolute time just after operation ``operation_index`` finishes.

        Correlated-noise sites are inserted ``after_gate`` an operation, so the
        environment is observed at the end of that operation, before the idle
        wait that follows it.
        """
        return self.time_before(operation_index) + float(self.durations[operation_index])

    @property
    def total_duration(self) -> float:
        return float(sum(self.durations) + sum(self.idle_after))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "durations": list(self.durations),
            "idle_after": list(self.idle_after),
            "default_duration": self.default_duration,
        }

    @staticmethod
    def from_dict(data: Optional[dict[str, Any]]) -> Optional["Schedule"]:
        if data is None:
            return None
        return Schedule(
            durations=tuple(float(v) for v in data.get("durations", ())),
            idle_after=tuple(float(v) for v in data.get("idle_after", ())),
            default_duration=float(data.get("default_duration", 1.0)),
            name=data.get("name", "schedule"),
        )

    @staticmethod
    def uniform(num_operations: int, idle_after: Sequence[float] | None = None) -> "Schedule":
        """Unit-duration operations with optional per-operation idle waits."""
        waits = tuple(idle_after or ())
        return Schedule((1.0,) * num_operations, waits, 1.0, "uniform")
