"""Density-matrix simulation and computational-basis measurement."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from typing import Optional, Sequence

from .gates import Circuit
from .linalg import Matrix, dagger, embedding, matmul, trace
from .noise import NoiseModel, NoiseRule


@dataclass(frozen=True)
class SimulationResult:
    state: Matrix
    probabilities: dict[str, float]

    @property
    def trace(self) -> complex:
        return trace(self.state)


def zero_state(num_qubits: int) -> Matrix:
    size = 2**num_qubits
    state = [[0.0j for _ in range(size)] for _ in range(size)]
    state[0][0] = 1.0 + 0j
    return state


def simulate(circuit: Circuit, noise_model: Optional[NoiseModel] = None) -> SimulationResult:
    rules = list(noise_model.rules if noise_model is not None else [])
    _validate_rules(circuit, rules)
    state = zero_state(circuit.num_qubits)

    for index, operation in enumerate(circuit.operations):
        unitary = embedding(operation.matrix, operation.qubits, circuit.num_qubits)
        state = _apply_unitary(state, unitary)
        for rule in rules:
            if rule.after_gate == index:
                state = _apply_rule(state, rule, circuit.num_qubits)

    for rule in rules:
        if rule.after_gate is None:
            state = _apply_rule(state, rule, circuit.num_qubits)

    probabilities = measurement_probabilities(state, circuit.num_qubits)
    result = SimulationResult(state, probabilities)
    validate_density_matrix(result.state)
    return result


def compare_ideal_and_noisy(
    circuit: Circuit, noise_model: NoiseModel
) -> tuple[SimulationResult, SimulationResult]:
    return simulate(circuit), simulate(circuit, noise_model)


def _apply_unitary(state: Matrix, unitary: Matrix) -> Matrix:
    return matmul(matmul(unitary, state), dagger(unitary))


def _apply_rule(state: Matrix, rule: NoiseRule, num_qubits: int) -> Matrix:
    if len(rule.qubits) != 1:
        raise ValueError("this version applies local noise to exactly one qubit per rule")
    size = len(state)
    output = [[0.0j for _ in range(size)] for _ in range(size)]
    for kraus in rule.kraus_operators():
        operator = embedding(kraus, rule.qubits, num_qubits)
        component = matmul(matmul(operator, state), dagger(operator))
        for row in range(size):
            for col in range(size):
                output[row][col] += component[row][col]
    return output


def _validate_rules(circuit: Circuit, rules: Sequence[NoiseRule]) -> None:
    for rule in rules:
        if rule.after_gate is not None and not 0 <= rule.after_gate < len(circuit.operations):
            raise ValueError("noise rule points outside circuit operation indices")
        if len(rule.qubits) != 1:
            raise ValueError("this version applies local noise to exactly one qubit per rule")
        qubit = rule.qubits[0]
        if not 0 <= qubit < circuit.num_qubits:
            raise ValueError("noise rule targets a qubit outside circuit")


def measurement_probabilities(state: Matrix, num_qubits: int) -> dict[str, float]:
    raw = [max(0.0, state[index][index].real) for index in range(len(state))]
    total = sum(raw)
    if total <= 0:
        raise ValueError("measurement probabilities have zero total mass")
    return {
        format(index, f"0{num_qubits}b"): value / total
        for index, value in enumerate(raw)
    }


def sample_counts(
    probabilities: dict[str, float], shots: int, seed: Optional[int] = None
) -> dict[str, int]:
    if shots <= 0:
        raise ValueError("shots must be positive")
    rng = random.Random(seed)
    outcomes = list(probabilities)
    cumulative = []
    running = 0.0
    for outcome in outcomes:
        running += probabilities[outcome]
        cumulative.append(running)
    counts = {outcome: 0 for outcome in outcomes}
    for _ in range(shots):
        point = rng.random()
        for outcome, threshold in zip(outcomes, cumulative):
            if point <= threshold:
                counts[outcome] += 1
                break
        else:
            counts[outcomes[-1]] += 1
    return counts


def validate_density_matrix(state: Matrix, tolerance: float = 1e-9) -> None:
    size = len(state)
    if any(len(row) != size for row in state):
        raise ValueError("state must be a square matrix")
    state_trace = trace(state)
    if abs(state_trace - 1.0) > tolerance:
        raise ValueError(f"state trace is {state_trace!r}, not 1")
    for row in range(size):
        for col in range(size):
            difference = abs(state[row][col] - state[col][row].conjugate())
            if difference > tolerance:
                raise ValueError("state is not Hermitian")
    if not _is_positive_semidefinite(state, tolerance):
        raise ValueError("state is not positive semidefinite")


def _is_positive_semidefinite(state: Matrix, tolerance: float) -> bool:
    """Cholesky-style PSD test for small dense density matrices."""
    size = len(state)
    working = [row[:] for row in state]
    for pivot_index in range(size):
        pivot = working[pivot_index][pivot_index].real
        if pivot < -tolerance:
            return False
        pivot_value = max(pivot, 0.0)
        if pivot_value <= tolerance:
            for row in range(pivot_index + 1, size):
                if abs(working[row][pivot_index]) > tolerance:
                    return False
            continue
        root = sqrt(pivot_value)
        for col in range(pivot_index + 1, size):
            working[pivot_index][col] /= root
        for row in range(pivot_index + 1, size):
            for col in range(row, size):
                reduction = working[row][pivot_index] * working[pivot_index][col].conjugate()
                working[row][col] -= reduction
                if row != col:
                    working[col][row] -= reduction.conjugate()
    return True
