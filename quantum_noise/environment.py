"""Continuous-time ambient noise and finite external disturbances.

An :class:`~quantum_noise.noise.AmbientNoise` process acts at a constant rate
per unit physical time.  A :class:`~quantum_noise.noise.DisturbancePlan`
overrides those rates inside finite, possibly overlapping time windows; once
the last window ends the ambient baseline is restored.

The channel of every constant-rate segment is computed from the exact
Lindblad generator superoperator, so splitting a homogeneous time interval
(or an operation that crosses it) never changes the resulting channel.  The
discrete Kraus channels are therefore never concatenated with rates linear
in the segment length, which would break time-semigroup composition.
"""

from __future__ import annotations

import cmath

import numpy as np

from .linalg import Matrix
from .noise import AMBIENT_CHANNEL_NAMES, AmbientNoise, DisturbancePlan

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)
_A0 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # |0><1|


def ambient_rates(ambient: AmbientNoise) -> dict[str, float]:
    return {name: float(getattr(ambient, name)) for name in AMBIENT_CHANNEL_NAMES}


def generator(rates: dict[str, float]) -> np.ndarray:
    """Return the 4x4 Lindblad generator in the vec(rho) row-flat basis."""

    def lindblad(rho: np.ndarray) -> np.ndarray:
        total = np.zeros((2, 2), dtype=complex)
        bit_flip = rates.get("bit_flip", 0.0)
        if bit_flip:
            total += bit_flip * (_X @ rho @ _X - rho)
        phase_flip = rates.get("phase_flip", 0.0)
        if phase_flip:
            total += phase_flip * (_Z @ rho @ _Z - rho)
        depolarizing = rates.get("depolarizing", 0.0)
        if depolarizing:
            # X, Y, Z jumps at rate r/4 generate the depolarizing generator
            # (r/2) * (I/2 - rho); the Bloch vector contracts by exp(-r t).
            for pauli in (_X, _Y, _Z):
                total += (depolarizing / 4.0) * (pauli @ rho @ pauli - rho)
        damping = rates.get("amplitude_damping", 0.0)
        if damping:
            jump = _A0 @ rho @ _A0.conj().T
            anticommutator = _A0.conj().T @ _A0 @ rho + rho @ _A0.conj().T @ _A0
            total += damping * (jump - 0.5 * anticommutator)
        return total

    superoperator = np.zeros((4, 4), dtype=complex)
    for column in range(4):
        rho = np.zeros((2, 2), dtype=complex)
        rho.flat[column] = 1.0
        superoperator[:, column] = lindblad(rho).flat
    return superoperator


def segment_kraus(rates: dict[str, float], duration: float) -> list[Matrix]:
    """Kraus operators of exp(L*duration), obtained from the channel Choi."""
    if duration <= 0.0 or not any(rates.get(name, 0.0) > 0.0 for name in AMBIENT_CHANNEL_NAMES):
        return []
    superop = generator(rates)
    eigenvalues, eigenvectors = np.linalg.eig(superop)
    channel = eigenvectors @ np.diag(np.exp(eigenvalues * duration)) @ np.linalg.inv(
        eigenvectors
    )
    # The permuted channel is the Choi matrix; eigenvectors in row-major vec
    # order give the Kraus operators with eigenvalue as their weight.
    choi = channel.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    choi = 0.5 * (choi + choi.conj().T)
    eigvals, eigvecs = np.linalg.eigh(choi)
    kraus: list[Matrix] = []
    for eigenvalue, eigenvector in zip(eigvals, eigvecs.T):
        if eigenvalue <= 1e-12:
            continue
        weight = max(0.0, eigenvalue.real)
        operator = np.reshape(np.sqrt(weight) * eigenvector, (2, 2))
        kraus.append([[complex(value) for value in row] for row in operator])
    return kraus


def fractional_unitary(unitary: Matrix, fraction: float) -> Matrix:
    """Return ``U ** fraction`` from the principal-branch matrix logarithm.

    Gate pieces therefore compose exactly when the fractions sum to one, so
    splitting a gate at a disturbance boundary preserves the operation.
    """
    size = len(unitary)
    if fraction <= 0.0:
        return [
            [complex(1 if row == col else 0) for col in range(size)]
            for row in range(size)
        ]
    if fraction >= 1.0:
        return [[complex(value) for value in row] for row in unitary]
    matrix = np.array(unitary, dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    powered = np.diag([cmath.exp(fraction * cmath.log(value)) for value in eigenvalues])
    result = eigenvectors @ powered @ np.linalg.inv(eigenvectors)
    return [[complex(value) for value in row] for row in result]


def resolve_segments(
    durations: tuple[float, ...],
    idle_after: tuple[float, ...],
    ambient: AmbientNoise,
    plan: DisturbancePlan | None,
) -> list[dict[str, object]]:
    """Resolve the physical timeline into constant-rate segments.

    Windows use half-open intervals ``[start, end)``: an instantaneous
    window with ``start == end`` changes nothing, and a window ending at a
    boundary has already restored the baseline there.
    """
    horizon = sum(durations) + sum(idle_after)
    windows = list(plan.disturbances) if plan is not None else []
    for window in windows:
        if window.start < 0.0 or window.end > horizon + 1e-12:
            raise ValueError("disturbance window lies outside the schedule horizon")
    boundaries = {0.0}
    for window in windows:
        boundaries.add(float(window.start))
        boundaries.add(float(window.end))
    points = _merge_times(sorted(boundaries | {max(horizon, 0.0)}))
    baseline = ambient_rates(ambient)
    overlap = plan.overlap if plan is not None else "reject"
    segments: list[dict[str, object]] = []
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        active = [window for window in windows if window.start <= midpoint < window.end]
        rates = dict(baseline)
        if len(active) > 1:
            if overlap == "reject":
                raise ValueError(
                    f"disturbance windows overlap near t={midpoint:g}; "
                    "choose overlap='latest' or 'max'"
                )
            if overlap == "latest":
                active.sort(key=lambda window: (window.start, windows.index(window)))
                chosen = active[-1]
                for name in AMBIENT_CHANNEL_NAMES:
                    value = getattr(chosen, name)
                    if value is not None:
                        rates[name] = float(value)
                active = [chosen]
            elif overlap == "max":
                for name in AMBIENT_CHANNEL_NAMES:
                    values = [
                        float(getattr(window, name))
                        for window in active
                        if getattr(window, name) is not None
                    ]
                    if values:
                        rates[name] = max(rates[name], max(values))
            else:
                raise ValueError(f"unknown overlap policy: {overlap!r}")
        elif active:
            window = active[0]
            for name in AMBIENT_CHANNEL_NAMES:
                value = getattr(window, name)
                if value is not None:
                    rates[name] = float(value)
        names = [window.name for window in active]
        segments.append(
            {
                "time_start": float(left),
                "time_end": float(right),
                "duration": float(right - left),
                "rates": rates,
                "disturbances": names,
                "restored": not names,
            }
        )
    return segments


def _merge_times(times: list[float], tolerance: float = 1e-10) -> list[float]:
    merged: list[float] = []
    for time in times:
        if not merged or abs(time - merged[-1]) > tolerance:
            merged.append(float(time))
        else:
            merged[-1] = 0.5 * (merged[-1] + float(time))
    return merged


def operation_intervals(
    durations: tuple[float, ...], idle_after: tuple[float, ...]
) -> list[tuple[float, float, int]]:
    """Return (start, end, operation index); idle intervals fill the gaps."""
    cursor = 0.0
    intervals: list[tuple[float, float, int]] = []
    for index, duration in enumerate(durations):
        end = cursor + duration
        intervals.append((cursor, end, index))
        cursor = end + idle_after[index]
    return intervals
