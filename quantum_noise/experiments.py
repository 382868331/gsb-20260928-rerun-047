"""Repeatable experiment batches and JSON serialization."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .gates import Circuit, CNOT, CZ, H, S, SWAP, X, Y, Z, rx, ry, rz
from .noise import NoiseModel
from .schedule import Schedule
from .simulator import (
    ScheduledShotSampler,
    ShotRecord,
    compare_ideal_and_noisy,
    compare_ideal_and_noisy_scheduled,
    sample_counts,
    sample_counts_trajectory,
)
from .stats import DifferenceTest, ProportionEstimate, compare_counts, proportions


ARCHIVE_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ExperimentBatch:
    circuit: Circuit
    noise_model: NoiseModel
    shots: int
    base_seed: int
    name: str = "experiment"
    schedule: Optional[Schedule] = None

    def seeds(self, repeats: int) -> list[int]:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        digest = hashlib.sha256(f"{self.base_seed}:{self.name}".encode("utf-8")).digest()
        root = int.from_bytes(digest[:8], "little")
        rng = random.Random(root)
        return [rng.getrandbits(64) for _ in range(repeats)]


@dataclass(frozen=True)
class BatchRun:
    batch_name: str
    shots: int
    base_seed: int
    repeat_index: int
    seed: int
    ideal_probabilities: dict[str, float]
    noisy_probabilities: dict[str, float]
    ideal_counts: dict[str, int]
    noisy_counts: dict[str, int]
    ideal_estimates: dict[str, ProportionEstimate]
    noisy_estimates: dict[str, ProportionEstimate]
    difference_tests: dict[str, DifferenceTest]
    noise_seed: Optional[int] = None
    measure_seed: Optional[int] = None
    noise_trajectories: Optional[list[dict[str, Any]]] = None
    measurement_conditions: Optional[dict[str, int]] = None
    noise_paths: Optional[dict[str, int]] = None
    schedule_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "shots": self.shots,
            "base_seed": self.base_seed,
            "repeat_index": self.repeat_index,
            "seed": self.seed,
            "ideal_probabilities": self.ideal_probabilities,
            "noisy_probabilities": self.noisy_probabilities,
            "ideal_counts": self.ideal_counts,
            "noisy_counts": self.noisy_counts,
            "difference_tests": {
                outcome: {
                    "observed_difference": test.observed_difference,
                    "standard_error": test.standard_error,
                    "confidence_95": list(test.confidence_95),
                    "significant": test.significant,
                    "exact_difference": test.exact_difference,
                    "has_exact_effect": test.has_exact_effect,
                    "minimum_expected_count": test.minimum_expected_count,
                    "enough_samples": test.enough_samples,
                    "interpretation": test.interpretation,
                }
                for outcome, test in self.difference_tests.items()
            },
            "noise_seed": self.noise_seed,
            "measure_seed": self.measure_seed,
            "noise_trajectories": self.noise_trajectories,
            "measurement_conditions": self.measurement_conditions,
            "noise_paths": self.noise_paths,
            "schedule_name": self.schedule_name,
        }


def run_batch(
    batch: ExperimentBatch, repeats: int = 1, outcomes: Optional[list[str]] = None
) -> list[BatchRun]:
    scheduled = batch.schedule is not None
    if scheduled:
        return _run_scheduled_batch(batch, repeats, outcomes)
    ideal_result, noisy_result = compare_ideal_and_noisy(batch.circuit, batch.noise_model)
    correlated = bool(batch.noise_model.groups)
    has_measurement = any(op.name == "measure" for op in batch.circuit.operations)
    correlated = correlated and not has_measurement
    trajectory_records = None
    if correlated:
        trajectory_records = [
            {
                "label": trajectory.label,
                "probability": trajectory.probability,
                "outcome_probabilities": trajectory.outcome_probabilities,
            }
            for trajectory in noisy_result.noise_trajectories or ()
        ]
    runs: list[BatchRun] = []
    for index, seed in enumerate(batch.seeds(repeats)):
        ideal_rng_seed = _derive_seed(seed, b"ideal")
        ideal_counts = sample_counts(ideal_result.probabilities, batch.shots, ideal_rng_seed)
        if correlated:
            # Correlated batches sample the joint noise trajectory and the
            # measurement outcome from separately seeded streams, keeping the
            # two sources of randomness individually reproducible.
            noise_seed = _derive_seed(seed, b"noise")
            measure_seed = _derive_seed(seed, b"measure")
            noisy_counts = sample_counts_trajectory(
                noisy_result.noise_trajectories or (),
                batch.shots,
                noise_seed=noise_seed,
                measure_seed=measure_seed,
            )
        else:
            noise_seed = None
            measure_seed = None
            noisy_rng_seed = _derive_seed(seed, b"noisy")
            noisy_counts = sample_counts(
                noisy_result.probabilities, batch.shots, noisy_rng_seed
            )
        tests = compare_counts(
            ideal_counts,
            noisy_counts,
            ideal_result.probabilities,
            noisy_result.probabilities,
            outcomes=outcomes,
        )
        runs.append(
            BatchRun(
                batch_name=batch.name,
                shots=batch.shots,
                base_seed=batch.base_seed,
                repeat_index=index,
                seed=seed,
                ideal_probabilities=ideal_result.probabilities,
                noisy_probabilities=noisy_result.probabilities,
                ideal_counts=ideal_counts,
                noisy_counts=noisy_counts,
                ideal_estimates=proportions(ideal_counts, ideal_result.probabilities),
                noisy_estimates=proportions(noisy_counts, noisy_result.probabilities),
                difference_tests=tests,
                noise_seed=noise_seed,
                measure_seed=measure_seed,
                noise_trajectories=trajectory_records,
            )
        )
    return runs


def _derive_seed(seed: int, label: bytes) -> int:
    digest = hashlib.sha256(seed.to_bytes(8, "little", signed=False) + label).digest()
    return int.from_bytes(digest[:8], "little")


def _run_scheduled_batch(
    batch: ExperimentBatch, repeats: int, outcomes: Optional[list[str]]
) -> list[BatchRun]:
    assert batch.schedule is not None
    schedule = batch.schedule.normalized(len(batch.circuit.operations))
    ideal_result, noisy_result = compare_ideal_and_noisy_scheduled(
        batch.circuit, batch.noise_model, schedule
    )
    environment_records = [
        {
            "label": trajectory.label,
            "register": trajectory.register,
            "probability": trajectory.probability,
            "outcome_probabilities": trajectory.outcome_probabilities,
            "events": {
                f"{group_index}:{after_gate}": 1
                for (group_index, after_gate) in (trajectory.events or {})
            },
        }
        for trajectory in noisy_result.noise_trajectories or ()
    ]
    runs: list[BatchRun] = []
    for index, seed in enumerate(batch.seeds(repeats)):
        ideal_rng_seed = _derive_seed(seed, b"ideal")
        ideal_counts = sample_counts(ideal_result.probabilities, batch.shots, ideal_rng_seed)
        noise_seed = _derive_seed(seed, b"noise")
        measure_seed = _derive_seed(seed, b"measure")
        sampler = ScheduledShotSampler(
            batch.circuit,
            batch.noise_model.groups,
            schedule,
            noise_seed=noise_seed,
            measure_seed=measure_seed,
        )
        records = sampler.draw_shots(batch.shots)
        noisy_counts, conditions, paths = _summarize_shot_records(records)
        tests = compare_counts(
            ideal_counts,
            noisy_counts,
            ideal_result.probabilities,
            noisy_result.probabilities,
            outcomes=outcomes,
        )
        runs.append(
            BatchRun(
                batch_name=batch.name,
                shots=batch.shots,
                base_seed=batch.base_seed,
                repeat_index=index,
                seed=seed,
                ideal_probabilities=ideal_result.probabilities,
                noisy_probabilities=noisy_result.probabilities,
                ideal_counts=ideal_counts,
                noisy_counts=noisy_counts,
                ideal_estimates=proportions(ideal_counts, ideal_result.probabilities),
                noisy_estimates=proportions(noisy_counts, noisy_result.probabilities),
                difference_tests=tests,
                noise_seed=noise_seed,
                measure_seed=measure_seed,
                noise_trajectories=environment_records,
                measurement_conditions=conditions,
                noise_paths=paths,
                schedule_name=schedule.name,
            )
        )
    return runs


def _summarize_shot_records(
    records: Sequence[ShotRecord],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    conditions: dict[str, int] = {}
    paths: dict[str, int] = {}
    for record in records:
        counts[record.outcome] = counts.get(record.outcome, 0) + 1
        if record.register:
            conditions[record.register] = conditions.get(record.register, 0) + 1
        if record.noise_label:
            paths[record.noise_label] = paths.get(record.noise_label, 0) + 1
    return counts, conditions or None, paths or None


@dataclass(frozen=True)
class BatchCheckpoint:
    """Serializable progress marker for a resumable scheduled batch run.

    Pausing after a mid-circuit measurement or an idle interval simply means
    stopping after whole shots have completed; restoring the checkpoint hands
    back exactly the same sampler RNG states, so the resumed batch is
    identical to an uninterrupted one.
    """

    batch_name: str
    repeat_index: int
    completed_shots: int
    total_shots: int
    seed: int
    noise_seed: int
    measure_seed: int
    counts: dict[str, int]
    measurement_conditions: dict[str, int]
    noise_paths: dict[str, int]
    sampler_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "checkpoint": {
                "batch_name": self.batch_name,
                "repeat_index": self.repeat_index,
                "completed_shots": self.completed_shots,
                "total_shots": self.total_shots,
                "seed": self.seed,
                "noise_seed": self.noise_seed,
                "measure_seed": self.measure_seed,
                "counts": dict(self.counts),
                "measurement_conditions": dict(self.measurement_conditions),
                "noise_paths": dict(self.noise_paths),
                "sampler_state": self.sampler_state,
            },
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "BatchCheckpoint":
        point = data["checkpoint"]
        return BatchCheckpoint(
            batch_name=point["batch_name"],
            repeat_index=int(point["repeat_index"]),
            completed_shots=int(point["completed_shots"]),
            total_shots=int(point["total_shots"]),
            seed=int(point["seed"]),
            noise_seed=int(point["noise_seed"]),
            measure_seed=int(point["measure_seed"]),
            counts={k: int(v) for k, v in point["counts"].items()},
            measurement_conditions={
                k: int(v) for k, v in point["measurement_conditions"].items()
            },
            noise_paths={k: int(v) for k, v in point["noise_paths"].items()},
            sampler_state=point["sampler_state"],
        )


class ResumableBatchRun:
    """Run one repeat of a scheduled batch in segments with replay support."""

    def __init__(self, batch: ExperimentBatch, repeat_index: int = 0) -> None:
        if batch.schedule is None:
            raise ValueError("resumable runs require an explicit time schedule")
        self.batch = batch
        self.repeat_index = repeat_index
        self.schedule = batch.schedule.normalized(len(batch.circuit.operations))
        seed = batch.seeds(repeat_index + 1)[repeat_index]
        self.seed = seed
        self.noise_seed = _derive_seed(seed, b"noise")
        self.measure_seed = _derive_seed(seed, b"measure")
        self.counts: dict[str, int] = {}
        self.conditions: dict[str, int] = {}
        self.paths: dict[str, int] = {}
        self.completed = 0
        self.sampler = ScheduledShotSampler(
            batch.circuit,
            batch.noise_model.groups,
            self.schedule,
            noise_seed=self.noise_seed,
            measure_seed=self.measure_seed,
        )

    def advance(self, shots: int) -> bool:
        if shots < 0 or self.completed + shots > self.batch.shots:
            raise ValueError("segment would exceed the configured number of shots")
        records = self.sampler.draw_shots(shots)
        segment_counts, segment_conditions, segment_paths = _summarize_shot_records(records)
        for outcome, value in segment_counts.items():
            self.counts[outcome] = self.counts.get(outcome, 0) + value
        for condition, value in (segment_conditions or {}).items():
            self.conditions[condition] = self.conditions.get(condition, 0) + value
        for path, value in (segment_paths or {}).items():
            self.paths[path] = self.paths.get(path, 0) + value
        self.completed += shots
        return self.completed == self.batch.shots

    def checkpoint(self) -> BatchCheckpoint:
        return BatchCheckpoint(
            batch_name=self.batch.name,
            repeat_index=self.repeat_index,
            completed_shots=self.completed,
            total_shots=self.batch.shots,
            seed=self.seed,
            noise_seed=self.noise_seed,
            measure_seed=self.measure_seed,
            counts=dict(self.counts),
            measurement_conditions=dict(self.conditions),
            noise_paths=dict(self.paths),
            sampler_state=self.sampler.state(),
        )

    @staticmethod
    def restore(batch: ExperimentBatch, checkpoint: BatchCheckpoint) -> "ResumableBatchRun":
        if checkpoint.batch_name != batch.name:
            raise ValueError("checkpoint belongs to a different batch")
        runner = ResumableBatchRun(batch, checkpoint.repeat_index)
        if runner.seed != checkpoint.seed:
            raise ValueError("checkpoint seed does not match batch specification")
        noise_state, measure_state = ScheduledShotSampler.restore_state(
            checkpoint.sampler_state
        )
        runner.sampler = ScheduledShotSampler(
            batch.circuit,
            batch.noise_model.groups,
            runner.schedule,
            noise_seed=checkpoint.noise_seed,
            measure_seed=checkpoint.measure_seed,
            noise_rng_state=noise_state,
            measure_rng_state=measure_state,
        )
        runner.counts = dict(checkpoint.counts)
        runner.conditions = dict(checkpoint.measurement_conditions)
        runner.paths = dict(checkpoint.noise_paths)
        runner.completed = checkpoint.completed_shots
        return runner

    def finish(self) -> BatchRun:
        if self.completed != self.batch.shots:
            raise ValueError("batch run is not complete yet")
        ideal_result, noisy_result = compare_ideal_and_noisy_scheduled(
            self.batch.circuit, self.batch.noise_model, self.schedule
        )
        ideal_counts = sample_counts(
            ideal_result.probabilities,
            self.batch.shots,
            _derive_seed(self.seed, b"ideal"),
        )
        tests = compare_counts(
            ideal_counts,
            self.counts,
            ideal_result.probabilities,
            noisy_result.probabilities,
        )
        environment_records = [
            {
                "label": trajectory.label,
                "register": trajectory.register,
                "probability": trajectory.probability,
                "outcome_probabilities": trajectory.outcome_probabilities,
                "events": {
                    f"{group_index}:{after_gate}": 1
                    for (group_index, after_gate) in (trajectory.events or {})
                },
            }
            for trajectory in noisy_result.noise_trajectories or ()
        ]
        return BatchRun(
            batch_name=self.batch.name,
            shots=self.batch.shots,
            base_seed=self.batch.base_seed,
            repeat_index=self.repeat_index,
            seed=self.seed,
            ideal_probabilities=ideal_result.probabilities,
            noisy_probabilities=noisy_result.probabilities,
            ideal_counts=ideal_counts,
            noisy_counts=dict(self.counts),
            ideal_estimates=proportions(ideal_counts, ideal_result.probabilities),
            noisy_estimates=proportions(self.counts, noisy_result.probabilities),
            difference_tests=tests,
            noise_seed=self.noise_seed,
            measure_seed=self.measure_seed,
            noise_trajectories=environment_records,
            measurement_conditions=dict(self.conditions) or None,
            noise_paths=dict(self.paths) or None,
            schedule_name=self.schedule.name,
        )


def save_checkpoint(path: str | Path, checkpoint: BatchCheckpoint) -> None:
    Path(path).write_text(
        json.dumps(checkpoint.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_checkpoint(path: str | Path) -> BatchCheckpoint:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BatchCheckpoint.from_dict(payload)


def save_batch_archive(
    path: str | Path,
    batch: ExperimentBatch,
    runs: list[BatchRun],
) -> None:
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "batch": {
            "name": batch.name,
            "shots": batch.shots,
            "base_seed": batch.base_seed,
            "circuit": circuit_to_dict(batch.circuit),
            "noise_model": batch.noise_model.to_dict(),
            "schedule": batch.schedule.to_dict() if batch.schedule is not None else None,
        },
        "runs": [run.to_dict() for run in runs],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_batch(path: str | Path) -> tuple[ExperimentBatch, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") not in (1, 2, ARCHIVE_SCHEMA_VERSION):
        raise ValueError("unsupported batch archive schema")
    data = payload["batch"]
    batch = ExperimentBatch(
        circuit=circuit_from_dict(data["circuit"]),
        noise_model=NoiseModel.from_dict(data["noise_model"]),
        shots=int(data["shots"]),
        base_seed=int(data["base_seed"]),
        name=data["name"],
        schedule=Schedule.from_dict(data.get("schedule")),
    )
    return batch, payload["runs"]


def rerun_archive(path: str | Path) -> list[BatchRun]:
    batch, saved_runs = load_batch(path)
    rerun = run_batch(batch, repeats=len(saved_runs))
    for saved, current in zip(saved_runs, rerun):
        if (
            saved["seed"] != current.seed
            or saved["ideal_counts"] != current.ideal_counts
            or saved["noisy_counts"] != current.noisy_counts
            or saved.get("measurement_conditions") != current.measurement_conditions
            or saved.get("noise_paths") != current.noise_paths
        ):
            raise ValueError("archive could not be reproduced with its stored seed and specification")
    return rerun


def circuit_to_dict(circuit: Circuit) -> dict[str, Any]:
    operations = []
    for operation in circuit.operations:
        if operation.name == "measure":
            operations.append(
                {
                    "name": "measure",
                    "qubits": list(operation.qubits),
                    "angle": None,
                    "cbit": operation.cbit,
                }
            )
            continue
        kind, angle = _operation_kind_and_angle(operation.name)
        entry: dict[str, Any] = {
            "name": kind,
            "qubits": list(operation.qubits),
            "angle": angle,
        }
        if operation.condition is not None:
            entry["condition"] = list(operation.condition)
        operations.append(entry)
    return {"num_qubits": circuit.num_qubits, "operations": operations}


def circuit_from_dict(data: dict[str, Any]) -> Circuit:
    circuit = Circuit(int(data["num_qubits"]))
    for item in data["operations"]:
        name = item["name"]
        qubits = tuple(item["qubits"])
        if name == "measure":
            circuit.measure(qubits[0], int(item["cbit"]))
            continue
        angle = item.get("angle")
        condition = item.get("condition")
        matrix = _gate_matrix(name, angle)
        operation_name = f"{name}({angle:.12g})" if angle is not None else name
        circuit.apply(
            operation_name,
            matrix,
            qubits,
            condition=tuple(condition) if condition is not None else None,
        )
    return circuit


def _operation_kind_and_angle(name: str) -> tuple[str, Optional[float]]:
    if name.endswith(")") and "(" in name:
        kind, value = name.split("(", 1)
        return kind, float(value[:-1])
    return name, None


def _gate_matrix(name: str, angle: Optional[float]) -> Any:
    table = {
        "h": H,
        "x": X,
        "y": Y,
        "z": Z,
        "s": S,
        "cnot": CNOT,
        "cz": CZ,
        "swap": SWAP,
    }
    if name in table:
        return table[name]
    if angle is None:
        raise ValueError(f"unknown gate {name!r}")
    if name == "rx":
        return rx(angle)
    if name == "ry":
        return ry(angle)
    if name == "rz":
        return rz(angle)
    raise ValueError(f"unknown gate {name!r}")
