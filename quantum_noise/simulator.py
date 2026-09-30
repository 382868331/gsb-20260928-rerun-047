"""Density-matrix simulation and computational-basis measurement."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Optional, Sequence

from .gates import Circuit
from .linalg import Matrix, dagger, embedding, matmul, trace, zeros
from .noise import CorrelatedNoiseGroup, NoiseModel, NoiseRule


@dataclass(frozen=True)
class SimulationResult:
    state: Matrix
    probabilities: dict[str, float]
    classical_probabilities: dict[str, float] | None = None
    noise_trajectories: tuple["NoiseTrajectory", ...] | None = None

    @property
    def trace(self) -> complex:
        return trace(self.state)


@dataclass(frozen=True)
class NoiseTrajectory:
    """One joint noise realization of the correlated groups.

    "probability" is the exact joint probability of the trajectory;
    "outcome_probabilities" is the measurement distribution conditioned on
    this trajectory.  Sampling uses the two separately so noise trajectories
    and measurement randomness stay traceable to different seeds.
    """

    label: str
    probability: float
    outcome_probabilities: dict[str, float]


def zero_state(num_qubits: int) -> Matrix:
    size = 2**num_qubits
    state = [[0.0j for _ in range(size)] for _ in range(size)]
    state[0][0] = 1.0 + 0j
    return state


def simulate(circuit: Circuit, noise_model: Optional[NoiseModel] = None) -> SimulationResult:
    rules = list(noise_model.rules if noise_model is not None else [])
    groups = list(noise_model.groups if noise_model is not None else [])
    _validate_rules(circuit, rules)
    _validate_groups(circuit, groups)

    if not groups:
        state, classical = _evolve(circuit, rules, {})
        result = SimulationResult(
            state=state,
            probabilities=measurement_probabilities(state, circuit.num_qubits),
            classical_probabilities=classical,
        )
        validate_density_matrix(result.state)
        return result

    # Exact mixture over the joint correlated-noise trajectories.  Each
    # trajectory applies its error pattern deterministically; the final state
    # is the probability-weighted sum, so correlations are never replaced by
    # independent per-site sampling.
    combined = _combined_trajectories(groups)
    size = 2**circuit.num_qubits
    state = zeros(size, size)
    classical_total: dict[str, float] = {}
    trajectories: list[NoiseTrajectory] = []
    for pattern, weight, events in combined:
        branch_state, classical = _evolve(circuit, rules, events)
        for row in range(size):
            for col in range(size):
                state[row][col] += weight * branch_state[row][col]
        for key, value in classical.items():
            classical_total[key] = classical_total.get(key, 0.0) + weight * value
        trajectories.append(
            NoiseTrajectory(
                label="".join(str(bit) for bit in pattern),
                probability=weight,
                outcome_probabilities=measurement_probabilities(
                    branch_state, circuit.num_qubits
                ),
            )
        )

    result = SimulationResult(
        state=state,
        probabilities=measurement_probabilities(state, circuit.num_qubits),
        classical_probabilities=classical_total or None,
        noise_trajectories=tuple(trajectories),
    )
    validate_density_matrix(result.state)
    return result


def _combined_trajectories(
    groups: Sequence[CorrelatedNoiseGroup],
) -> list[tuple[tuple[int, ...], float, dict[int, list[tuple[Matrix, tuple[int, ...]]]]]]:
    """Combine per-group Markov trajectories into joint error events.

    Groups are mutually independent; sites inside one group are correlated.
    Returns (pattern, probability, events) triples where events maps a gate
    index to the deterministic error operators applied after that gate.
    """

    per_group = []
    for group in groups:
        entries = []
        for pattern, weight in group.trajectories():
            events: dict[int, list[tuple[Matrix, tuple[int, ...]]]] = {}
            for site, bit in zip(group.sites, pattern):
                if bit:
                    after_gate, qubit = site
                    events.setdefault(after_gate, []).append(
                        (group.error_operator, (qubit,))
                    )
            entries.append((pattern, weight, events))
        per_group.append(entries)

    combined = []
    for choices in product(*per_group):
        pattern: tuple[int, ...] = ()
        weight = 1.0
        events: dict[int, list[tuple[Matrix, tuple[int, ...]]]] = {}
        for choice_pattern, choice_weight, choice_events in choices:
            pattern += choice_pattern
            weight *= choice_weight
            for index, operators in choice_events.items():
                events.setdefault(index, []).extend(operators)
        combined.append((pattern, weight, events))
    return combined


def _evolve(
    circuit: Circuit,
    rules: Sequence[NoiseRule],
    error_events: dict[int, list[tuple[Matrix, tuple[int, ...]]]],
) -> tuple[Matrix, dict[str, float]]:
    """Propagate branch states through gates, measurements and noise.

    Mid-circuit measurements split the evolution into branches keyed by the
    classical register; conditional gates only act on matching branches.
    Returns the summed final density matrix and the classical-register
    distribution (empty when the circuit has no measurement).
    """

    branches: list[tuple[Matrix, tuple[int, ...]]] = [
        (zero_state(circuit.num_qubits), ())
    ]

    def apply_noise(state: Matrix, index: Optional[int]) -> Matrix:
        for rule in rules:
            if rule.after_gate == index:
                state = _apply_rule(state, rule, circuit.num_qubits)
        for operator, qubits in error_events.get(index, ()):  # type: ignore[arg-type]
            unitary = embedding(operator, qubits, circuit.num_qubits)
            state = _apply_unitary(state, unitary)
        return state

    for index, operation in enumerate(circuit.operations):
        next_branches: list[tuple[Matrix, tuple[int, ...]]] = []
        for state, register in branches:
            if operation.name == "measure":
                next_branches.extend(
                    _measure(state, operation.qubits[0], operation.cbit, register,
                             circuit.num_qubits)
                )
                continue
            if operation.condition is not None:
                cbit, value = operation.condition
                if cbit >= len(register) or register[cbit] != value:
                    next_branches.append((state, register))
                    continue
            unitary = embedding(operation.matrix, operation.qubits, circuit.num_qubits)
            next_branches.append((_apply_unitary(state, unitary), register))
        branches = [
            (apply_noise(state, index), register) for state, register in next_branches
        ]

    branches = [(apply_noise(state, None), register) for state, register in branches]

    size = 2**circuit.num_qubits
    final = zeros(size, size)
    classical: dict[str, float] = {}
    for state, register in branches:
        weight = trace(state).real
        for row in range(size):
            for col in range(size):
                final[row][col] += state[row][col]
        if register:
            key = "".join(str(bit) for bit in register)
            classical[key] = classical.get(key, 0.0) + weight
    return final, classical


def _measure(
    state: Matrix, qubit: int, cbit: Optional[int], register: tuple[int, ...],
    num_qubits: int,
) -> list[tuple[Matrix, tuple[int, ...]]]:
    if cbit is None:
        raise ValueError("measure operation is missing its classical bit")
    outcomes = []
    for value in (0, 1):
        projector = [[1.0 + 0j if row == col == value else 0.0j for col in range(2)]
                     for row in range(2)]
        embedded = embedding(projector, (qubit,), num_qubits)
        projected = matmul(matmul(embedded, state), embedded)
        if trace(projected).real <= 1e-15:
            continue
        new_register = list(register)
        while len(new_register) <= cbit:
            new_register.append(0)
        new_register[cbit] = value
        outcomes.append((projected, tuple(new_register)))
    return outcomes


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


def _validate_groups(circuit: Circuit, groups: Sequence[CorrelatedNoiseGroup]) -> None:
    for group in groups:
        for after_gate, qubit in group.sites:
            if not 0 <= after_gate < len(circuit.operations):
                raise ValueError("correlated noise site points outside circuit operations")
            if not 0 <= qubit < circuit.num_qubits:
                raise ValueError("correlated noise site targets a qubit outside circuit")


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


def sample_counts_trajectory(
    trajectories: Sequence[NoiseTrajectory],
    shots: int,
    noise_seed: Optional[int] = None,
    measure_seed: Optional[int] = None,
) -> dict[str, int]:
    """Sample shots by first drawing a noise trajectory, then a measurement.

    The two draws use independent RNG streams seeded by "noise_seed" and
    "measure_seed", so the correlated noise realization and the measurement
    randomness of every shot stay separately traceable and reproducible.
    The marginal outcome distribution equals sampling from the exact mixed
    probabilities, but the correlated joint process is preserved per shot.
    """

    if shots <= 0:
        raise ValueError("shots must be positive")
    if not trajectories:
        raise ValueError("at least one noise trajectory is required")
    noise_rng = random.Random(noise_seed)
    measure_rng = random.Random(measure_seed)
    weights = [trajectory.probability for trajectory in trajectories]
    total = sum(weights)
    if total <= 0:
        raise ValueError("noise trajectories have zero total probability")
    cumulative = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cumulative.append(running)

    counts: dict[str, int] = {}
    for trajectory in trajectories:
        for outcome in trajectory.outcome_probabilities:
            counts.setdefault(outcome, 0)
    for _ in range(shots):
        point = noise_rng.random()
        for trajectory, threshold in zip(trajectories, cumulative):
            if point <= threshold:
                break
        else:
            trajectory = trajectories[-1]
        outcome = _draw_outcome(trajectory.outcome_probabilities, measure_rng)
        counts[outcome] += 1
    return counts


def _draw_outcome(probabilities: dict[str, float], rng: random.Random) -> str:
    outcomes = list(probabilities)
    point = rng.random()
    running = 0.0
    for outcome in outcomes:
        running += probabilities[outcome]
        if point <= running:
            return outcome
    return outcomes[-1]


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
