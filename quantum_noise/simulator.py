"""Density-matrix simulation and computational-basis measurement."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from math import sqrt
from typing import Optional, Sequence

from .gates import Circuit
from .linalg import Matrix, dagger, embedding, matmul, trace, zeros
from .noise import CorrelatedNoiseGroup, NoiseModel, NoiseRule, TimeSchedule


@dataclass(frozen=True)
class SimulationResult:
    state: Matrix
    probabilities: dict[str, float]
    classical_probabilities: dict[str, float] | None = None
    noise_trajectories: tuple["NoiseTrajectory", ...] | None = None
    environment_records: tuple[dict[str, object], ...] | None = None
    measurement_conditions: tuple[dict[str, object], ...] | None = None
    schedule: TimeSchedule | None = None

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
    branch: str = ""
    environment_states: dict[str, list[int]] = field(default_factory=dict)


def zero_state(num_qubits: int) -> Matrix:
    size = 2**num_qubits
    state = [[0.0j for _ in range(size)] for _ in range(size)]
    state[0][0] = 1.0 + 0j
    return state


def simulate(
    circuit: Circuit,
    noise_model: Optional[NoiseModel] = None,
    schedule: Optional[TimeSchedule] = None,
) -> SimulationResult:
    rules = list(noise_model.rules if noise_model is not None else [])
    groups = list(noise_model.groups if noise_model is not None else [])
    if schedule is not None:
        schedule = schedule.normalized(len(circuit.operations))
    _validate_rules(circuit, rules)
    _validate_groups(circuit, groups)

    if not groups:
        state, classical = _evolve(circuit, rules, {}, schedule=schedule)
        environment_records = None
        measurement_conditions = None
        if schedule is not None or any(op.name == "measure" for op in circuit.operations):
            registers = _possible_registers(circuit)
            measurement_conditions = tuple(
                _measurement_conditions(circuit, register) for register in registers
            ) or None
            environment_records = tuple(
                {
                    "branch": "".join(str(bit) for bit in register),
                    "probability": 1.0 if not register else None,
                    "measurement_conditions": _measurement_conditions(circuit, register),
                    "schedule": schedule.to_dict() if schedule is not None else None,
                    "events": _rule_environment_events(circuit, rules, schedule),
                }
                for register in _possible_registers(circuit)
            )
        result = SimulationResult(
            state=state,
            probabilities=measurement_probabilities(state, circuit.num_qubits),
            classical_probabilities=classical,
            environment_records=environment_records,
            measurement_conditions=measurement_conditions,
            schedule=schedule,
        )
        validate_density_matrix(result.state)
        return result

    # Exact mixture over the joint correlated-noise trajectories.  Each
    # trajectory applies its error pattern deterministically; the final state
    # is the probability-weighted sum, so correlations are never replaced by
    # independent per-site sampling.
    registers = [
        register
        for register in _possible_registers(circuit)
        if _branch_probability(circuit, register, schedule) > 1e-15
    ]
    measurement_conditions = tuple(
        _measurement_conditions(circuit, register) for register in registers
    )
    combined = _combined_branch_trajectories(circuit, groups, schedule)
    size = 2**circuit.num_qubits
    state = zeros(size, size)
    classical_total: dict[str, float] = {}
    trajectories: list[NoiseTrajectory] = []
    environment_records: list[dict[str, object]] = []
    for branch, pattern, weight, events, environment_states, record in combined:
        branch_state, classical = _evolve(
            circuit,
            rules,
            {branch: events},
            schedule=schedule,
            only_branch=branch,
        )
        for row in range(size):
            for col in range(size):
                state[row][col] += weight * branch_state[row][col]
        for key, value in classical.items():
            classical_total[key] = classical_total.get(key, 0.0) + weight * value
        record_with_probability = dict(record)
        record_with_probability["probability"] = weight
        environment_records.append(record_with_probability)
        trajectories.append(
            NoiseTrajectory(
                label=f"{branch}:{pattern}" if branch else pattern,
                probability=weight,
                outcome_probabilities=measurement_probabilities(
                    branch_state, circuit.num_qubits
                )
                if trace(branch_state).real > 1e-15
                else {},
                branch=branch,
                environment_states=environment_states,
            )
        )

    result = SimulationResult(
        state=state,
        probabilities=measurement_probabilities(state, circuit.num_qubits),
        classical_probabilities=classical_total or None,
        noise_trajectories=tuple(trajectories),
        environment_records=tuple(environment_records),
        measurement_conditions=measurement_conditions or None,
        schedule=schedule,
    )
    validate_density_matrix(result.state)
    return result


def _possible_registers(circuit: Circuit) -> list[tuple[int, ...]]:
    cbit_count = 0
    for operation in circuit.operations:
        if operation.name == "measure" and operation.cbit is not None:
            cbit_count = max(cbit_count, operation.cbit + 1)
    if cbit_count == 0:
        return [()]
    return [
        tuple(int(bit) for bit in format(value, f"0{cbit_count}b"))
        for value in range(2**cbit_count)
    ]


def _branch_probability(
    circuit: Circuit, register: tuple[int, ...], schedule: Optional[TimeSchedule]
) -> float:
    if not register:
        return 1.0
    state, _classical = _evolve(
        circuit,
        [],
        {},
        schedule=schedule,
        only_branch="".join(str(bit) for bit in register),
    )
    return trace(state).real


def _register_matches(register: tuple[int, ...], condition: tuple[int, int]) -> bool:
    cbit, value = condition
    return cbit < len(register) and register[cbit] == value


def _experienced_indices(
    circuit: Circuit,
    register: tuple[int, ...],
    groups: Sequence[CorrelatedNoiseGroup],
    schedule: Optional[TimeSchedule],
) -> list[list[int]]:
    active = set(range(len(circuit.operations)))
    for index, operation in enumerate(circuit.operations):
        if (
            operation.condition is not None
            and not _register_matches(register, operation.condition)
        ):
            active.discard(index)
    result = []
    for group in groups:
        indices = [
            index
            for index, (after_gate, _qubit) in enumerate(group.sites)
            if after_gate in active
        ]
        result.append(indices)
    return result


def _measurement_conditions(
    circuit: Circuit, register: tuple[int, ...]
) -> dict[str, object]:
    values = {str(index): int(value) for index, value in enumerate(register)}
    operations = []
    for index, operation in enumerate(circuit.operations):
        if operation.name == "measure":
            value = int(register[operation.cbit]) if operation.cbit < len(register) else 0
            operations.append(
                {
                    "operation": index,
                    "qubit": operation.qubits[0],
                    "cbit": operation.cbit,
                    "value": value,
                }
            )
    return {"register": values, "operations": operations}


def _rule_environment_events(
    circuit: Circuit,
    rules: Sequence[NoiseRule],
    schedule: Optional[TimeSchedule],
) -> list[dict[str, object]]:
    events = []
    for rule in rules:
        if rule.after_gate is None:
            continue
        events.append(
            {
                "site": [rule.after_gate, rule.qubits[0]],
                "time": schedule.event_time(rule.after_gate) if schedule is not None else None,
                "channel": rule.channel,
                "probability": rule.probability,
            }
        )
    return events


def _combined_branch_trajectories(
    circuit: Circuit,
    groups: Sequence[CorrelatedNoiseGroup],
    schedule: Optional[TimeSchedule],
) -> list[
    tuple[
        str,
        str,
        float,
        dict[int, list[tuple[Matrix, tuple[int, ...]]]],
        dict[str, list[int]],
        dict[str, object],
    ]
]:
    combined = []
    for branch in _possible_registers(circuit):
        branch_key = "".join(str(bit) for bit in branch)
        experienced = _experienced_indices(circuit, branch, groups, schedule)
        for pattern, weight, events, states, records in _combined_trajectories(
            groups, experienced, schedule
        ):
            combined.append(
                (
                    branch_key,
                    pattern,
                    weight,
                    events,
                    states,
                    {
                        "branch": branch_key,
                        "pattern": pattern,
                        "environment_states": states,
                        "measurement_conditions": _measurement_conditions(circuit, branch),
                        "events": records,
                    },
                )
            )
    return combined


def _combined_trajectories(
    groups: Sequence[CorrelatedNoiseGroup],
    experienced_by_group: list[list[int]],
    schedule: Optional[TimeSchedule],
) -> list[
    tuple[
        str,
        float,
        dict[int, list[tuple[Matrix, tuple[int, ...]]]],
        dict[str, list[int]],
        dict[str, list[dict]],
    ]
]:
    """Combine per-group Markov trajectories into joint error events.

    Groups are mutually independent; sites inside one group are correlated.
    Returns (pattern, probability, events) triples where events maps a gate
    index to the deterministic error operators applied after that gate.
    """

    per_group = []
    for group_index, group in enumerate(groups):
        entries = []
        experienced_indices = experienced_by_group[group_index]
        for pattern, weight, transitions in _group_trajectories(
            group, experienced_indices, schedule
        ):
            events: dict[int, list[tuple[Matrix, tuple[int, ...]]]] = {}
            for site_index, bit in zip(experienced_indices, pattern):
                site = group.sites[site_index]
                if bit:
                    after_gate, qubit = site
                    events.setdefault(after_gate, []).append(
                        (group.error_operator, (qubit,))
                    )
            entries.append((pattern, weight, events, transitions))
        per_group.append(entries)

    combined = []
    for choices in product(*per_group):
        pattern_parts: list[str] = []
        weight = 1.0
        events: dict[int, list[tuple[Matrix, tuple[int, ...]]]] = {}
        environment_states: dict[str, list[int]] = {}
        event_records: dict[str, list[dict]] = {}
        for group, choice in zip(groups, choices):
            choice_pattern, choice_weight, choice_events, transitions = choice
            pattern_parts.append("".join(str(bit) for bit in choice_pattern))
            weight *= choice_weight
            for index, operators in choice_events.items():
                events.setdefault(index, []).extend(operators)
            environment_states[group.name] = list(choice_pattern)
            event_records[group.name] = transitions
        combined.append(
            (
                "-".join(pattern_parts),
                weight,
                events,
                environment_states,
                event_records,
            )
        )
    return combined


def _group_trajectories(
    group: CorrelatedNoiseGroup,
    experienced_indices: Sequence[int],
    schedule: Optional[TimeSchedule],
) -> list[tuple[tuple[int, ...], float, list[dict]]]:
    p = group.probability
    results: list[tuple[tuple[int, ...], float, list[dict]]] = [((), 1.0, [])]
    previous_time: Optional[float] = None
    previous_site: Optional[tuple[int, int]] = None
    for site_index in experienced_indices:
        after_gate, qubit = group.sites[site_index]
        current_time = None if schedule is None else schedule.event_time(after_gate)
        elapsed = None if previous_time is None or current_time is None else current_time - previous_time
        correlation = group.correlation_after(elapsed)
        stay_error = min(1.0, max(0.0, p + correlation * (1.0 - p)))
        become_error = min(1.0, max(0.0, p * (1.0 - correlation)))
        next_results = []
        for pattern, old_weight, history in results:
            p_error = p if not pattern else (stay_error if pattern[-1] else become_error)
            for bit, transition_probability in ((1, p_error), (0, 1.0 - p_error)):
                record = {
                    "site": [after_gate, qubit],
                    "time": current_time,
                    "elapsed": elapsed,
                    "correlation": correlation,
                    "state": bit,
                    "previous_site": list(previous_site) if previous_site is not None else None,
                }
                next_results.append(
                    (
                        pattern + (bit,),
                    old_weight * transition_probability,
                    history + [record],
                    )
                )
        results = next_results
        previous_time = current_time
        previous_site = (after_gate, qubit)
    return [(pattern, weight, history) for pattern, weight, history in results if weight > 0.0]


def _evolve(
    circuit: Circuit,
    rules: Sequence[NoiseRule],
    error_events: dict[str, dict[int, list[tuple[Matrix, tuple[int, ...]]]]] | dict[int, list[tuple[Matrix, tuple[int, ...]]]],
    schedule: Optional[TimeSchedule] = None,
    only_branch: str = "",
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

    def apply_noise(state: Matrix, index: Optional[int], register: tuple[int, ...]) -> Matrix:
        for rule in rules:
            if rule.after_gate == index:
                state = _apply_rule(state, rule, circuit.num_qubits)
        if error_events and not isinstance(next(iter(error_events)), int):
            branch_key = "".join(str(bit) for bit in register)
            branch_events = error_events.get(branch_key, {})  # type: ignore[union-attr]
        else:
            branch_events = error_events  # type: ignore[assignment]
        for operator, qubits in branch_events.get(index, ()):  # type: ignore[union-attr]
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
                if not _register_matches(register, (cbit, value)):
                    next_branches.append((state, register))
                    continue
            unitary = embedding(operation.matrix, operation.qubits, circuit.num_qubits)
            next_branches.append((_apply_unitary(state, unitary), register))
        branches = [
            (apply_noise(state, index, register), register) for state, register in next_branches
        ]

    branches = [
        (apply_noise(state, None, register), register) for state, register in branches
    ]

    size = 2**circuit.num_qubits
    final = zeros(size, size)
    classical: dict[str, float] = {}
    for state, register in branches:
        branch_key = "".join(str(bit) for bit in register)
        if only_branch and branch_key != only_branch:
            continue
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
    circuit: Circuit, noise_model: NoiseModel, schedule: Optional[TimeSchedule] = None
) -> tuple[SimulationResult, SimulationResult]:
    return simulate(circuit, schedule=schedule), simulate(
        circuit, noise_model, schedule=schedule
    )


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
