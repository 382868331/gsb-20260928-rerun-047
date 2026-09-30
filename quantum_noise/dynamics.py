"""Trajectory-aware dynamics with mid-circuit measurement and conditional gates.

Exact probabilities sum over every classical measurement branch and every
correlated-noise path.  Per-shot traces draw the noise trajectory and each
measurement result from separate random streams, so noise randomness and
measurement randomness can be audited independently.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Optional

from .correlated import CorrelatedNoiseModel, sample_trajectory, trajectory_paths
from .gates import Circuit, ConditionalOperation, Measurement, Operation
from .linalg import Matrix, dagger, embedding, matmul, trace
from .simulator import validate_density_matrix, zero_state


@dataclass(frozen=True)
class ShotTrace:
    final_outcome: str
    measurement_outcomes: dict[int, int]
    branch_path: tuple[int, ...]
    noise_errors: dict[str, tuple[int, ...]]


def unitary_gate_count(circuit: Circuit) -> int:
    """Count chronological gate slots: plain gates and conditional gates."""

    return sum(1 for op in circuit.operations if isinstance(op, (Operation, ConditionalOperation)))


def exact_probabilities(
    circuit: Circuit,
    noise_model: Optional[CorrelatedNoiseModel] = None,
) -> dict[str, float]:
    """Exact final-bit probabilities with correlated noise and branches."""

    events_by_gate, path_weights = _resolve_noise(circuit, noise_model)
    aggregate: dict[str, float] = {}
    for path, noise_weight in path_weights:
        branch_distribution = _exact_branches(circuit, events_by_gate, path)
        for outcome, branch_weight in branch_distribution.items():
            aggregate[outcome] = aggregate.get(outcome, 0.0) + noise_weight * branch_weight
    total = sum(aggregate.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"exact branch weights total {total!r}, not 1")
    return {
        format(index, f"0{circuit.num_qubits}b"): aggregate.get(format(index, f"0{circuit.num_qubits}b"), 0.0)
        for index in range(2**circuit.num_qubits)
    }


def simulate_shot(
    circuit: Circuit,
    noise_model: Optional[CorrelatedNoiseModel],
    noise_rng: random.Random,
    measurement_rng: random.Random,
) -> ShotTrace:
    """Run one physical shot; noise and measurement use distinct RNGs."""

    events_by_gate, ordered_by_key, sampled_errors = _sample_noise(
        circuit, noise_model, noise_rng
    )
    state = zero_state(circuit.num_qubits)
    bits: dict[int, int] = {}
    branch_path: list[int] = []
    gate_index = 0
    for instruction in circuit.operations:
        if isinstance(instruction, Operation):
            state = _unitary(state, instruction, circuit.num_qubits)
            gate_index += 1
            for event, error in events_by_gate.get(gate_index - 1, []):
                if error:
                    state = _pauli(state, event.qubit, event.error_pauli(), circuit.num_qubits)
        elif isinstance(instruction, Measurement):
            value = _project_and_draw(state, instruction.qubit, circuit.num_qubits, measurement_rng)
            state = value.state
            bits[instruction.bit] = value.outcome
            branch_path.append(value.outcome)
        elif isinstance(instruction, ConditionalOperation):
            if bits.get(instruction.bit) == 1:
                state = _unitary(state, instruction.operation, circuit.num_qubits)
            slot = gate_index
            gate_index += 1
            for event, error in events_by_gate.get(slot, []):
                if error:
                    state = _pauli(state, event.qubit, event.error_pauli(), circuit.num_qubits)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown circuit instruction {instruction!r}")
    final_index = _draw_final(state, circuit.num_qubits, measurement_rng)
    noise_errors = {
        key: tuple(sampled_errors[id(event)] for event in events)
        for key, events in ordered_by_key.items()
    }
    return ShotTrace(
        final_outcome=format(final_index, f"0{circuit.num_qubits}b"),
        measurement_outcomes=dict(bits),
        branch_path=tuple(branch_path),
        noise_errors=noise_errors,
    )


def _resolve_noise(
    circuit: Circuit,
    noise_model: Optional[CorrelatedNoiseModel],
) -> tuple[dict[int, list], list[tuple[dict, float]]]:
    gate_count = unitary_gate_count(circuit)
    ordered = noise_model.ordered_events() if noise_model is not None else []
    if noise_model is not None:
        noise_model.validate(gate_count)
        for event in ordered:
            if event.qubit >= circuit.num_qubits:
                raise ValueError("noise event targets a qubit outside circuit")
    by_key: dict[str, list] = {}
    for event in ordered:
        by_key.setdefault(event.key(), []).append(event)
    chain_options = []
    for key, events in by_key.items():
        paths = trajectory_paths(events)
        chain_options.append([(key, events, values, weight) for values, weight in paths])
    combined_paths: list[tuple[dict, float]] = []
    for selection in itertools.product(*chain_options) if chain_options else [()]:
        errors: dict[int, int] = {}
        weight = 1.0
        for key, events, values, path_weight in selection:
            weight *= path_weight
            for event, error in zip(events, values):
                errors[id(event)] = error
        combined_paths.append((errors, weight))
    events_by_gate: dict[int, list] = {}
    for event in ordered:
        events_by_gate.setdefault(event.after_gate, []).append(event)
    return events_by_gate, combined_paths


def _sample_noise(circuit, noise_model, noise_rng):
    gate_count = unitary_gate_count(circuit)
    ordered = noise_model.ordered_events() if noise_model is not None else []
    if noise_model is not None:
        noise_model.validate(gate_count)
        for event in ordered:
            if event.qubit >= circuit.num_qubits:
                raise ValueError("noise event targets a qubit outside circuit")
    by_key: dict[str, list] = {}
    for event in ordered:
        by_key.setdefault(event.key(), []).append(event)
    sampled_errors: dict[int, int] = {}
    for events in by_key.values():
        drawn = sample_trajectory(events, noise_rng)
        sampled_errors.update(drawn)
    events_by_gate = {}
    for event in ordered:
        events_by_gate.setdefault(event.after_gate, []).append((event, sampled_errors[id(event)]))
    return events_by_gate, by_key, sampled_errors


def _exact_branches(
    circuit: Circuit,
    events_by_gate: dict[int, list],
    noise_errors: dict[int, int],
) -> dict[str, float]:
    state = zero_state(circuit.num_qubits)
    distribution: dict[str, float] = {}

    def evolve(step: int, current: Matrix, gate_index: int, bits: dict[int, int], weight: float) -> None:
        for cursor in range(step, len(circuit.operations)):
            instruction = circuit.operations[cursor]
            if isinstance(instruction, Operation):
                current = _unitary(current, instruction, circuit.num_qubits)
                gate_index += 1
                for event in events_by_gate.get(gate_index - 1, []):
                    if noise_errors.get(id(event), 0):
                        current = _pauli(current, event.qubit, event.error_pauli(), circuit.num_qubits)
            elif isinstance(instruction, ConditionalOperation):
                if bits.get(instruction.bit) == 1:
                    current = _unitary(current, instruction.operation, circuit.num_qubits)
                slot = gate_index
                gate_index += 1
                for event in events_by_gate.get(slot, []):
                    if noise_errors.get(id(event), 0):
                        current = _pauli(current, event.qubit, event.error_pauli(), circuit.num_qubits)
            elif isinstance(instruction, Measurement):
                for outcome in (0, 1):
                    projected, probability = _project(current, instruction.qubit, outcome, circuit.num_qubits)
                    if probability <= 1e-15:
                        continue
                    next_bits = dict(bits)
                    next_bits[instruction.bit] = outcome
                    evolve(cursor + 1, projected, gate_index, next_bits, weight * probability)
                return
        validate_density_matrix(current)
        for index in range(2**circuit.num_qubits):
            probability = max(0.0, current[index][index].real) * weight
            if probability > 0.0:
                outcome = format(index, f"0{circuit.num_qubits}b")
                distribution[outcome] = distribution.get(outcome, 0.0) + probability

    evolve(0, state, 0, {}, 1.0)
    return distribution


def _unitary(state: Matrix, operation: Operation, num_qubits: int) -> Matrix:
    unitary = embedding(operation.matrix, operation.qubits, num_qubits)
    return matmul(matmul(unitary, state), dagger(unitary))


def _pauli(state: Matrix, qubit: int, pauli: Matrix, num_qubits: int) -> Matrix:
    return _unitary(state, Operation("error", pauli, (qubit,)), num_qubits)


def _projector(qubit: int, outcome: int, num_qubits: int) -> Matrix:
    operator = [[1.0 + 0j, 0j], [0j, 0j]] if outcome == 0 else [[0j, 0j], [0j, 1.0 + 0j]]
    return embedding(operator, (qubit,), num_qubits)


def _project(state: Matrix, qubit: int, outcome: int, num_qubits: int) -> tuple[Matrix, float]:
    projector = _projector(qubit, outcome, num_qubits)
    projected = matmul(matmul(projector, state), projector)
    probability = trace(projected).real
    if probability <= 1e-15:
        return projected, probability
    normalized = [[value / probability for value in row] for row in projected]
    return normalized, probability


@dataclass(frozen=True)
class _Projection:
    state: Matrix
    outcome: int


def _project_and_draw(
    state: Matrix, qubit: int, num_qubits: int, rng: random.Random
) -> _Projection:
    probability_one = trace(matmul(_projector(qubit, 1, num_qubits), state)).real
    outcome = 1 if rng.random() < probability_one else 0
    projected, _ = _project(state, qubit, outcome, num_qubits)
    return _Projection(projected, outcome)


def _draw_final(state: Matrix, num_qubits: int, rng: random.Random) -> int:
    point = rng.random()
    cumulative = 0.0
    for index in range(2**num_qubits):
        cumulative += max(0.0, state[index][index].real)
        if point <= cumulative:
            return index
    return 2**num_qubits - 1
