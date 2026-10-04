"""Density-matrix simulation and computational-basis measurement."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from math import sqrt
from typing import Optional, Sequence

from .gates import Circuit
from .linalg import Matrix, dagger, embedding, matmul, trace, zeros
from .noise import (
    CorrelatedNoiseGroup,
    EnvironmentTimeline,
    NoiseModel,
    NoiseRule,
    TimeSchedule,
)
from .environment import (
    EnvironmentEvent,
    EventCancellation,
    event_joint_probabilities,
    firing_plan,
)


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
    timeline: Optional[EnvironmentTimeline] = None,
    environment_events: Optional[Sequence[EnvironmentEvent]] = None,
    event_cancellations: Optional[Sequence[EventCancellation]] = None,
) -> SimulationResult:
    rules = list(noise_model.rules if noise_model is not None else [])
    groups = list(noise_model.groups if noise_model is not None else [])
    environment_events = tuple(environment_events or ())
    event_cancellations = tuple(event_cancellations or ())
    if schedule is not None:
        schedule = schedule.normalized(len(circuit.operations))
    timeline = timeline or EnvironmentTimeline()
    if timeline.perturbations and schedule is None:
        raise ValueError("perturbations require a TimeSchedule to identify physical times")
    if environment_events and schedule is None:
        raise ValueError("environment events require a TimeSchedule to identify physical times")
    _validate_events(circuit, environment_events)
    cancellation_names = {event.name for event in environment_events}
    for cancellation in event_cancellations:
        if cancellation.event_name not in cancellation_names:
            raise ValueError(f"cancellation targets unknown event {cancellation.event_name!r}")
    _validate_rules(circuit, rules)
    _validate_groups(circuit, groups)

    if not groups and not environment_events:
        state, classical = _evolve(circuit, rules, {}, schedule=schedule, timeline=timeline)
        environment_records = None
        measurement_conditions = None
        if schedule is not None or any(op.name == "measure" for op in circuit.operations):
            registers = _possible_registers(circuit)
            measurement_conditions = tuple(
                _measurement_conditions(circuit, register) for register in registers
            ) or None
            timeline_records = timeline.change_records() if timeline.perturbations else None
            environment_records = tuple(
                {
                    "branch": "".join(str(bit) for bit in register),
                    "probability": 1.0 if not register else None,
                    "measurement_conditions": _measurement_conditions(circuit, register),
                    "schedule": schedule.to_dict() if schedule is not None else None,
                    "events": _rule_environment_events(circuit, rules, schedule, timeline),
                    "timeline": timeline_records,
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

    # Exact mixture over joint correlated-noise and common-environment
    # trajectories.  An environment event is drawn once for the whole
    # circuit, so pre-measurement firings are shared across the classical
    # branches splitting off afterward (never re-applied per branch).
    registers = list(_possible_registers(circuit))
    measurement_conditions = tuple(
        _measurement_conditions(circuit, register) for register in registers
    )
    size = 2**circuit.num_qubits
    state = zeros(size, size)
    classical_total: dict[str, float] = {}
    trajectories: list[NoiseTrajectory] = []
    environment_records: list[dict[str, object]] = []
    if environment_events:
        measurement_indices = [
            index
            for index, operation in enumerate(circuit.operations)
            if operation.name == "measure"
        ]
        plan = firing_plan(
            environment_events,
            schedule,
            event_cancellations,
            measurement_indices,
        )
        trajectory_entries = _event_trajectory_entries(
            circuit, groups, environment_events, plan, schedule, timeline
        )
        for pattern, weight, events, environment_states, records in trajectory_entries:
            evolved, classical = _evolve(
                circuit, rules, events, schedule=schedule, timeline=timeline
            )
            for row in range(size):
                for col in range(size):
                    state[row][col] += weight * evolved[row][col]
            for key, value in classical.items():
                classical_total[key] = classical_total.get(key, 0.0) + weight * value
            environment_records.append(
                {
                    "pattern": pattern,
                    "probability": weight,
                    "environment_states": environment_states,
                    "events": records,
                    "timeline": timeline.change_records()
                    if timeline.perturbations
                    else None,
                }
            )
            trajectories.append(
                NoiseTrajectory(
                    label=pattern,
                    probability=weight,
                    outcome_probabilities=measurement_probabilities(
                        evolved, circuit.num_qubits
                    )
                    if trace(evolved).real > 1e-15
                    else {},
                    branch="",
                    environment_states=environment_states,
                )
            )
    else:
        combined = _combined_branch_trajectories(
            circuit, groups, schedule, timeline
        )
        for branch, pattern, weight, events, environment_states, record in combined:
            branch_state, classical = _evolve(
                circuit,
                rules,
                {branch: events},
                schedule=schedule,
                timeline=timeline,
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
    circuit: Circuit,
    register: tuple[int, ...],
    schedule: Optional[TimeSchedule],
    timeline: Optional[EnvironmentTimeline] = None,
) -> float:
    if not register:
        return 1.0
    state, _classical = _evolve(
        circuit,
        [],
        {},
        schedule=schedule,
        timeline=timeline or EnvironmentTimeline(),
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
    timeline: EnvironmentTimeline,
) -> list[dict[str, object]]:
    events = []
    for rule in rules:
        if rule.after_gate is None:
            continue
        event_time = schedule.event_time(rule.after_gate) if schedule is not None else None
        perturbation, effective_probability = (
            timeline.effective_at(event_time, rule.channel)
            if event_time is not None
            else (None, None)
        )
        events.append(
            {
                "site": [rule.after_gate, rule.qubits[0]],
                "time": event_time,
                "channel": rule.channel,
                "probability": rule.probability,
                "effective_probability": rule.probability
                if effective_probability is None
                else effective_probability,
                "perturbation": perturbation.name if perturbation is not None else None,
            }
        )
    return events


def _combined_branch_trajectories(
    circuit: Circuit,
    groups: Sequence[CorrelatedNoiseGroup],
    schedule: Optional[TimeSchedule],
    timeline: EnvironmentTimeline,
    environment_events: Sequence[EnvironmentEvent] = (),
    event_cancellations: Sequence[EventCancellation] = (),
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
    plan = (
        firing_plan(environment_events, schedule, event_cancellations)
        if environment_events and schedule is not None
        else {}
    )
    event_joint = (
        event_joint_probabilities(environment_events) if environment_events else []
    )
    for branch in _possible_registers(circuit):
        branch_key = "".join(str(bit) for bit in branch)
        experienced = _experienced_indices(circuit, branch, groups, schedule)
        branch_active = set(range(len(circuit.operations)))
        for index, operation in enumerate(circuit.operations):
            if (
                operation.condition is not None
                and not _register_matches(branch, operation.condition)
            ):
                branch_active.discard(index)
        noise_choices = _combined_trajectories(
            groups, experienced, schedule, timeline
        ) if groups else [("", 1.0, {}, {}, {})]
        event_choices = event_joint or [([], 1.0)]
        for noise_choice, event_choice in product(noise_choices, event_choices):
            pattern, weight, events, states, records = noise_choice
            event_bits, event_weight = event_choice
            events = {index: list(operators) for index, operators in events.items()}
            states = dict(states)
            records = dict(records)
            event_events, event_states, event_records = _events_to_error_map(
                environment_events,
                event_bits,
                plan,
                schedule,
                branch_active,
            )
            for index, operators in event_events.items():
                events.setdefault(index, []).extend(operators)
            if event_states:
                states.update(event_states)
            if event_records:
                records["environment_events"] = event_records
            combined.append(
                (
                    branch_key,
                    pattern,
                    weight * event_weight,
                    events,
                    states,
                    {
                        "branch": branch_key,
                        "pattern": pattern,
                        "environment_states": states,
                        "measurement_conditions": _measurement_conditions(circuit, branch),
                        "events": records,
                        "timeline": timeline.change_records()
                        if timeline.perturbations
                        else None,
                    },
                )
            )
    return combined


def _event_trajectory_entries(
    circuit: Circuit,
    groups: Sequence[CorrelatedNoiseGroup],
    environment_events: Sequence[EnvironmentEvent],
    plan: dict[int, list[tuple[int, EnvironmentEvent]]],
    schedule: Optional[TimeSchedule],
    timeline: EnvironmentTimeline,
) -> list[
    tuple[
        str,
        float,
        dict[int, list[tuple[Matrix, tuple[int, ...]]]],
        dict[str, list[int]],
        dict[str, object],
    ]
]:
    """Enumerate one joint environment draw for the whole circuit.

    Measurements split each realization into branches inside evolution, so
    a firing before a measurement is never redrawn per branch.  When
    correlated groups are also present their sites must all be reached
    unconditionally (a site behind a conditional gate would require a
    branch-specific draw, which would no longer be one shared realization).
    """
    has_measurement = any(
        operation.name == "measure" for operation in circuit.operations
    )
    if groups and has_measurement:
        unconditional = set(range(len(circuit.operations)))
        for index, operation in enumerate(circuit.operations):
            if operation.condition is not None:
                unconditional.discard(index)
        for group in groups:
            for after_gate, _qubit in group.sites:
                if after_gate not in unconditional:
                    raise ValueError(
                        "combining environment events with a correlated group site "
                        "behind a conditional gate is not supported; split the study"
                    )
    event_choices = event_joint_probabilities(environment_events)
    full_active = set(range(len(circuit.operations)))
    entries: list[
        tuple[
            str,
            float,
            dict[int, list[tuple[Matrix, tuple[int, ...]]]],
            dict[str, list[int]],
            dict[str, object],
        ]
    ] = []
    experienced = _experienced_indices(circuit, (), groups, schedule)
    group_choices = (
        list(_combined_trajectories(groups, experienced, schedule, timeline))
        if groups
        else [("", 1.0, {}, {}, {})]
    )
    for event_bits, event_weight in event_choices:
        event_events, event_states, event_records = _events_to_error_map(
            environment_events, event_bits, plan, schedule, full_active
        )
        for pattern, group_weight, group_events, group_states, group_records in group_choices:
            events = {index: list(operators) for index, operators in event_events.items()}
            for index, operators in group_events.items():
                events.setdefault(index, []).extend(operators)
            states = dict(event_states)
            states.update(group_states)
            records = dict(group_records)
            if event_records:
                records["environment_events"] = event_records
            entries.append(
                (
                    pattern,
                    event_weight * group_weight,
                    events,
                    states,
                    records,
                )
            )
    return entries


def _events_to_error_map(
    environment_events: Sequence[EnvironmentEvent],
    event_bits: Sequence[tuple[int, ...]],
    plan: dict[int, list[tuple[int, EnvironmentEvent]]],
    schedule: Optional[TimeSchedule],
    branch_active: set[int],
) -> tuple[
    dict[int, list[tuple[Matrix, tuple[int, ...]]]],
    dict[str, list[int]],
    list[dict],
]:
    """Expand one shared realization per event into per-boundary operators."""
    events: dict[int, list[tuple[Matrix, tuple[int, ...]]]] = {}
    states: dict[str, list[int]] = {}
    records: list[dict] = []
    for event_index, event in enumerate(environment_events):
        bits = event_bits[event_index]
        flips = event.realization_flips(bits)
        states[event.name] = list(flips)
        for operation_index, ordered in plan.items():
            if operation_index not in branch_active:
                continue
            if not any(index == event_index for index, _entry in ordered):
                continue
            firing_time = (
                schedule.event_time(operation_index) if schedule is not None else None
            )
            records.append(
                {
                    "event": event.name,
                    "operation": operation_index,
                    "time": firing_time,
                    "common_bit": bits[0],
                    "independent_bits": list(bits[1:]),
                    "qubit_flips": list(flips),
                    "qubits": list(event.qubits),
                }
            )
            for qubit, flip in zip(event.qubits, flips):
                if flip:
                    events.setdefault(operation_index, []).append(
                        (event.error_operator, (qubit,))
                    )
    records.sort(key=lambda record: (record["operation"], record["event"]))
    return events, states, records


def _combined_trajectories(
    groups: Sequence[CorrelatedNoiseGroup],
    experienced_by_group: list[list[int]],
    schedule: Optional[TimeSchedule],
    timeline: EnvironmentTimeline,
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
            group, experienced_indices, schedule, timeline
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
    timeline: EnvironmentTimeline,
) -> list[tuple[tuple[int, ...], float, list[dict]]]:
    results: list[tuple[tuple[int, ...], float, list[dict]]] = [((), 1.0, [])]
    previous_time: Optional[float] = None
    previous_site: Optional[tuple[int, int]] = None
    for site_index in experienced_indices:
        after_gate, qubit = group.sites[site_index]
        current_time = None if schedule is None else schedule.event_time(after_gate)
        elapsed = None if previous_time is None or current_time is None else current_time - previous_time
        perturbation, effective_probability = (
            timeline.effective_at(current_time, group.channel)
            if current_time is not None
            else (None, None)
        )
        p = group.probability if effective_probability is None else effective_probability
        if elapsed is None:
            correlation = group.correlation
        else:
            correlation = timeline.correlation_over(
                group.correlation,
                previous_time,
                current_time,
                group.correlation_time,
                group.channel,
            )
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
                    "probability": p,
                    "state": bit,
                    "perturbation": perturbation.name if perturbation is not None else None,
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
    timeline: Optional[EnvironmentTimeline] = None,
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
    timeline = timeline or EnvironmentTimeline()

    def apply_noise(state: Matrix, index: Optional[int], register: tuple[int, ...]) -> Matrix:
        for rule in rules:
            if rule.after_gate == index:
                event_time = (
                    schedule.event_time(index)
                    if schedule is not None and index is not None
                    else None
                )
                _perturbation, probability = (
                    timeline.effective_at(event_time, rule.channel)
                    if event_time is not None
                    else (None, None)
                )
                effective_rule = (
                    rule
                    if probability is None
                    else NoiseRule(
                        channel=rule.channel,
                        probability=probability,
                        qubits=rule.qubits,
                        after_gate=rule.after_gate,
                        params=dict(rule.params),
                    )
                )
                state = _apply_rule(state, effective_rule, circuit.num_qubits)
        branch_events = {}
        if error_events and not isinstance(next(iter(error_events)), int):
            branch_key = "".join(str(bit) for bit in register)
            branch_events = error_events.get(branch_key, {})  # type: ignore[union-attr]
            global_events = error_events.get("__shared__", {})  # type: ignore[union-attr]
        else:
            global_events = error_events  # type: ignore[assignment]
        for operator, qubits in global_events.get(index, ()):  # type: ignore[union-attr]
            unitary = embedding(operator, qubits, circuit.num_qubits)
            state = _apply_unitary(state, unitary)
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
    circuit: Circuit,
    noise_model: NoiseModel,
    schedule: Optional[TimeSchedule] = None,
    timeline: Optional[EnvironmentTimeline] = None,
    environment_events: Optional[Sequence[EnvironmentEvent]] = None,
    event_cancellations: Optional[Sequence[EventCancellation]] = None,
) -> tuple[SimulationResult, SimulationResult]:
    return simulate(circuit, schedule=schedule, timeline=timeline), simulate(
        circuit,
        noise_model,
        schedule=schedule,
        timeline=timeline,
        environment_events=environment_events,
        event_cancellations=event_cancellations,
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


def _validate_events(
    circuit: Circuit, events: Sequence[EnvironmentEvent]
) -> None:
    for event in events:
        for qubit in event.qubits:
            if not 0 <= qubit < circuit.num_qubits:
                raise ValueError("environment event targets a qubit outside circuit")


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
    weights = [max(0.0, trajectory.probability) for trajectory in trajectories]
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
        if not trajectory.outcome_probabilities:
            continue
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
