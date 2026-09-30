"""Density-matrix simulation and computational-basis measurement."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Optional, Sequence

from .gates import Circuit
from .schedule import Schedule
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
    register: Optional[str] = None
    events: Optional[dict[int, int]] = None


def zero_state(num_qubits: int) -> Matrix:
    size = 2**num_qubits
    state = [[0.0j for _ in range(size)] for _ in range(size)]
    state[0][0] = 1.0 + 0j
    return state


def simulate(
    circuit: Circuit,
    noise_model: Optional[NoiseModel] = None,
    schedule: Optional[Schedule] = None,
) -> SimulationResult:
    rules = list(noise_model.rules if noise_model is not None else [])
    groups = list(noise_model.groups if noise_model is not None else [])
    _validate_rules(circuit, rules)
    _validate_groups(circuit, groups)

    timed = schedule is not None or any(group.uses_time for group in groups)
    if not groups:
        state, classical = _evolve(circuit, rules, {})
        result = SimulationResult(
            state=state,
            probabilities=measurement_probabilities(state, circuit.num_qubits),
            classical_probabilities=classical,
        )
        validate_density_matrix(result.state)
        return result

    # Exact mixture over the joint correlated-noise trajectories.  When an
    # explicit schedule (or time-decaying correlation) is used, or when the
    # circuit contains a mid-circuit measurement, the environment is evolved
    # *per branch*: after a measurement each branch carries on its own
    # environment state, so two branches with the same number of later
    # operations do not experience the same disturbance.  Without schedules
    # and measurements the legacy global-trajectory enumeration is identical
    # and kept for backwards compatibility.
    has_measurement = any(op.name == "measure" for op in circuit.operations)
    if timed or has_measurement:
        normalized_schedule = (
            schedule.normalized(len(circuit.operations)) if schedule is not None else None
        )
        return _simulate_scheduled(circuit, rules, groups, normalized_schedule)

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


# ---------------------------------------------------------------------------
# Time-scheduled, branch-independent environment evolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GroupSite:
    after_gate: int
    qubit: int
    site_index: int  # position within the group


@dataclass
class _EnvComponent:
    """One joint realization with weight (classical path probability)."""

    state: Matrix
    register: tuple[int, ...]
    weight: float
    # Per group: (last site index observed on this branch, last env state).
    env: dict[int, tuple[int, int]]
    label: str  # event bit string in group order (site order), for archiving


def _group_site_table(
    groups: Sequence[CorrelatedNoiseGroup],
) -> tuple[dict[int, list[tuple[int, _GroupSite]]], list[int]]:
    table: dict[int, list[tuple[int, _GroupSite]]] = {}
    sizes: list[int] = []
    for group_index, group in enumerate(groups):
        sizes.append(len(group.sites))
        for site_index, site in enumerate(group.sites):
            after_gate, qubit = site
            table.setdefault(int(after_gate), []).append(
                (group_index, _GroupSite(int(after_gate), int(qubit), site_index))
            )
    return table, sizes


def _simulate_scheduled(
    circuit: Circuit,
    rules: Sequence[NoiseRule],
    groups: Sequence[CorrelatedNoiseGroup],
    schedule: Optional[Schedule],
) -> SimulationResult:
    if schedule is None:
        default_schedule = Schedule(
            tuple(1.0 for _ in circuit.operations),
            tuple(0.0 for _ in circuit.operations),
            1.0,
            "unit-step",
        )
    else:
        default_schedule = schedule
    site_table, group_sizes = _group_site_table(groups)
    site_times: list[Optional[tuple[float, ...]]] = []
    for group in groups:
        if group.times is not None:
            site_times.append(tuple(float(t) for t in group.times))
        elif group.time_correlation is not None:
            site_times.append(
                tuple(default_schedule.site_time(int(s[0])) for s in group.sites)
            )
        else:
            site_times.append(None)

    components = [
        _EnvComponent(zero_state(circuit.num_qubits), (), 1.0, {}, "")
    ]
    for index, operation in enumerate(circuit.operations):
        next_components: list[_EnvComponent] = []
        for component in components:
            next_components.extend(
                _scheduled_operation(
                    component,
                    operation,
                    index,
                    circuit.num_qubits,
                    rules,
                    groups,
                    site_table,
                    site_times,
                )
            )
        components = next_components

    # Rules inserted after the complete circuit (after_gate is None).
    final_components: list[_EnvComponent] = []
    for component in components:
        state = component.state
        for rule in rules:
            if rule.after_gate is None:
                state = _apply_rule(state, rule, circuit.num_qubits)
        final_components.append(
            _EnvComponent(
                state,
                component.register,
                component.weight,
                dict(component.env),
                component.label,
            )
        )
    components = final_components

    size = 2**circuit.num_qubits
    final = zeros(size, size)
    classical: dict[str, float] = {}
    paths: dict[tuple[str, str], dict[str, float]] = {}
    path_weights: dict[tuple[str, str], float] = {}
    for component in components:
        norm = trace(component.state).real
        if norm <= 1e-15:
            continue
        register_key = "".join(str(bit) for bit in component.register)
        normalized = [[component.state[r][c] / norm for c in range(size)] for r in range(size)]
        for row in range(size):
            for col in range(size):
                final[row][col] += component.weight * normalized[row][col]
        if component.register:
            classical[register_key] = classical.get(register_key, 0.0) + component.weight
        probs = measurement_probabilities(normalized, circuit.num_qubits)
        key = (component.label, register_key)
        path_weights[key] = path_weights.get(key, 0.0) + component.weight
        bucket = paths.setdefault(key, {})
        for outcome, probability in probs.items():
            bucket[outcome] = bucket.get(outcome, 0.0) + component.weight * probability

    trajectories: list[NoiseTrajectory] = []
    for (label, register_key), weighted in sorted(paths.items()):
        path_weight = path_weights[(label, register_key)]
        if path_weight <= 1e-15:
            continue
        conditional = {
            outcome: value / path_weight for outcome, value in weighted.items()
        }
        events: dict[int, int] = {}
        for group_index, group in enumerate(groups):
            offset = sum(group_sizes[:group_index])
            for site_index in range(group_sizes[group_index]):
                if label[offset + site_index] == "1":
                    after_gate = int(group.sites[site_index][0])
                    events[(group_index, after_gate)] = 1
        trajectories.append(
            NoiseTrajectory(
                label=label,
                probability=path_weight,
                outcome_probabilities=conditional,
                register=register_key or None,
                events=events,
            )
        )

    result = SimulationResult(
        state=final,
        probabilities=measurement_probabilities(final, circuit.num_qubits),
        classical_probabilities=classical or None,
        noise_trajectories=tuple(trajectories),
    )
    validate_density_matrix(result.state)
    return result


def _scheduled_operation(
    component: _EnvComponent,
    operation,
    index: int,
    num_qubits: int,
    rules: Sequence[NoiseRule],
    groups: Sequence[CorrelatedNoiseGroup],
    site_table: dict[int, list[tuple[int, _GroupSite]]],
    site_times: Sequence[Optional[tuple[float, ...]]],
) -> list[_EnvComponent]:
    """Propagate one component through one operation and the noise sites after it."""

    if operation.name == "measure":
        branches = _measure(
            component.state, operation.qubits[0], operation.cbit, component.register,
            num_qubits,
        )
        measured: list[_EnvComponent] = []
        for projected, register in branches:
            branch_weight = trace(projected).real
            projected = _normalize(projected, branch_weight)
            measured.append(
                _EnvComponent(
                    projected,
                    register,
                    component.weight * branch_weight,
                    dict(component.env),
                    component.label,
                )
            )
    else:
        state = component.state
        if operation.condition is None or _condition_matches(operation.condition, component.register):
            unitary = embedding(operation.matrix, operation.qubits, num_qubits)
            state = _apply_unitary(state, unitary)
        measured = [
            _EnvComponent(
                state,
                component.register,
                component.weight,
                dict(component.env),
                component.label,
            )
        ]

    results = measured
    for rule in rules:
        if rule.after_gate == index:
            for current in results:
                current.state = _apply_rule(current.state, rule, num_qubits)

    entries = sorted(
        site_table.get(index, ()), key=lambda item: (item[0], item[1].site_index)
    )
    for group_index, site in entries:
        group = groups[group_index]
        times = site_times[group_index]
        next_results: list[_EnvComponent] = []
        for current in results:
            previous = current.env.get(group_index)
            if previous is None:
                p_error = group.probability
            else:
                previous_site, previous_env = previous
                correlation = _correlation_to_site(
                    group, times, site.site_index, previous_site
                )
                stay_error, become_error = group.transition_probabilities(
                    group.probability, correlation
                )
                p_error = stay_error if previous_env == 1 else become_error
            for error_bit in (1, 0):
                probability = p_error if error_bit == 1 else 1.0 - p_error
                if probability <= 0.0:
                    continue
                state = current.state
                if error_bit:
                    unitary = embedding(group.error_operator, (site.qubit,), num_qubits)
                    state = _apply_unitary(state, unitary)
                env = dict(current.env)
                env[group_index] = (site.site_index, error_bit)
                next_results.append(
                    _EnvComponent(
                        state,
                        current.register,
                        current.weight * probability,
                        env,
                        current.label + str(error_bit),
                    )
                )
        results = next_results
    return results


def _condition_matches(condition, register: tuple[int, ...]) -> bool:
    cbit, value = condition
    return cbit < len(register) and register[cbit] == value


def _normalize(state: Matrix, norm: float) -> Matrix:
    size = len(state)
    return [[state[r][c] / norm for c in range(size)] for r in range(size)]


def _correlation_to_site(
    group: CorrelatedNoiseGroup,
    times: Optional[tuple[float, ...]],
    site_index: int,
    previous_site: int,
) -> float:
    if times is not None and group.time_correlation is not None:
        delta = max(0.0, times[site_index] - times[previous_site])
        return float(group.time_correlation) ** delta
    # Without a time model, adjacent sites on the same branch share the
    # configured Pearson correlation; conditioned-out sites belong to a
    # different branch and must not be counted as shared history.
    return group.correlation


def compare_ideal_and_noisy(
    circuit: Circuit, noise_model: NoiseModel
) -> tuple[SimulationResult, SimulationResult]:
    return _compare(circuit, noise_model, None)


def compare_ideal_and_noisy_scheduled(
    circuit: Circuit, noise_model: NoiseModel, schedule: Schedule
) -> tuple[SimulationResult, SimulationResult]:
    return _compare(circuit, noise_model, schedule)


def _compare(
    circuit: Circuit,
    noise_model: NoiseModel,
    schedule: Optional[Schedule],
) -> tuple[SimulationResult, SimulationResult]:
    return simulate(circuit, schedule=schedule), simulate(circuit, noise_model, schedule)


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


@dataclass(frozen=True)
class ShotRecord:
    """Traceability record for one shot of a scheduled batch."""

    outcome: str
    register: str
    noise_label: str


class ScheduledShotSampler:
    """Shot-by-shot sampler for scheduled correlated-noise experiments.

    Each shot first walks the two-state environment(s) and the mid-circuit
    measurement branches (noise stream), then draws the final measurement
    (measure stream).  RNG states are part of the sampler state, so a batch
    can be paused after any shot, serialized, restored and replayed to give
    exactly the same counts as running it in one call.  Interleaving two
    batches does not change either result because every sampler owns its
    independent RNG streams.
    """

    def __init__(
        self,
        circuit: Circuit,
        groups: Sequence[CorrelatedNoiseGroup],
        schedule: Optional[Schedule],
        noise_seed: Optional[int],
        measure_seed: Optional[int],
        noise_rng_state=None,
        measure_rng_state=None,
    ) -> None:
        self.circuit = circuit
        self.groups = list(groups)
        self.schedule = (
            schedule.normalized(len(circuit.operations)) if schedule is not None else None
        )
        self.noise_rng = random.Random(noise_seed)
        self.measure_rng = random.Random(measure_seed)
        if noise_rng_state is not None:
            self.noise_rng.setstate(noise_rng_state)
        if measure_rng_state is not None:
            self.measure_rng.setstate(measure_rng_state)
        self.site_table, self.group_sizes = _group_site_table(self.groups)
        self.site_times = []
        for group in self.groups:
            if group.times is not None:
                self.site_times.append(tuple(float(t) for t in group.times))
            elif group.time_correlation is not None:
                effective_schedule = self.schedule or Schedule(
                    tuple(1.0 for _ in circuit.operations),
                    tuple(0.0 for _ in circuit.operations),
                    1.0,
                    "unit-step",
                )
                self.site_times.append(
                    tuple(effective_schedule.site_time(int(s[0])) for s in group.sites)
                )
            else:
                self.site_times.append(None)

    def state(self) -> dict:
        """JSON-serializable state for pause/resume replay."""
        return {
            "noise_rng": list(self.noise_rng.getstate()[1]),
            "noise_rng_index": self.noise_rng.getstate()[0],
            "measure_rng": list(self.measure_rng.getstate()[1]),
            "measure_rng_index": self.measure_rng.getstate()[0],
        }

    @staticmethod
    def restore_state(data: Optional[dict]):
        if not data:
            return None, None
        noise_state = (int(data["noise_rng_index"]), tuple(data["noise_rng"]), None)
        measure_state = (int(data["measure_rng_index"]), tuple(data["measure_rng"]), None)
        return noise_state, measure_state

    def draw_shots(self, shots: int) -> list[ShotRecord]:
        return [self.draw_shot() for _ in range(shots)]

    def draw_shot(self) -> ShotRecord:
        state = zero_state(self.circuit.num_qubits)
        register: tuple[int, ...] = ()
        env: dict[int, tuple[int, int]] = {}
        labels = ["" for _ in self.groups]
        for index, operation in enumerate(self.circuit.operations):
            if operation.name == "measure":
                probabilities = {
                    str(value): _measurement_probability(
                        state, operation.qubits[0], value, self.circuit.num_qubits
                    )
                    for value in (0, 1)
                }
                value = int(self._draw(probabilities, self.noise_rng))
                projector = [
                    [1.0 + 0j if row == col == value else 0.0j for col in range(2)]
                    for row in range(2)
                ]
                embedded = embedding(projector, (operation.qubits[0],), self.circuit.num_qubits)
                state = matmul(matmul(embedded, state), embedded)
                new_register = list(register)
                while len(new_register) <= operation.cbit:
                    new_register.append(0)
                new_register[operation.cbit] = value
                register = tuple(new_register)
            elif operation.condition is None or _condition_matches(operation.condition, register):
                unitary = embedding(
                    operation.matrix, operation.qubits, self.circuit.num_qubits
                )
                state = _apply_unitary(state, unitary)

            entries = sorted(
                self.site_table.get(index, ()),
                key=lambda item: (item[0], item[1].site_index),
            )
            for group_index, site in entries:
                group = self.groups[group_index]
                previous = env.get(group_index)
                if previous is None:
                    p_error = group.probability
                else:
                    previous_site, previous_env = previous
                    times = self.site_times[group_index]
                    if times is not None and group.time_correlation is not None:
                        delta = max(0.0, times[site.site_index] - times[previous_site])
                        correlation = float(group.time_correlation) ** delta
                    else:
                        correlation = group.correlation
                    stay_error, become_error = group.transition_probabilities(
                        group.probability, correlation
                    )
                    p_error = stay_error if previous_env == 1 else become_error
                error_bit = int(self._draw({"1": p_error, "0": 1.0 - p_error}, self.noise_rng))
                labels[group_index] += str(error_bit)
                if error_bit:
                    unitary = embedding(group.error_operator, (site.qubit,), self.circuit.num_qubits)
                    state = _apply_unitary(state, unitary)
                env[group_index] = (site.site_index, error_bit)

        probabilities = measurement_probabilities(state, self.circuit.num_qubits)
        outcome = self._draw(probabilities, self.measure_rng)
        return ShotRecord(
            outcome=outcome,
            register="".join(str(bit) for bit in register),
            noise_label="".join(labels),
        )

    @staticmethod
    def _draw(probabilities: dict[str, float], rng: random.Random) -> str:
        point = rng.random()
        running = 0.0
        for outcome, probability in probabilities.items():
            running += probability
            if point <= running:
                return outcome
        return next(reversed(probabilities))


def _measurement_probability(
    state: Matrix, qubit: int, value: int, num_qubits: int
) -> float:
    projector = [
        [1.0 + 0j if row == col == value else 0.0j for col in range(2)]
        for row in range(2)
    ]
    embedded = embedding(projector, (qubit,), num_qubits)
    return trace(matmul(embedded, state)).real


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
