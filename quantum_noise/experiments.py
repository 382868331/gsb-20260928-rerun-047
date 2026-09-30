"""Repeatable experiment batches and JSON serialization."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .gates import Circuit, CNOT, CZ, H, S, SWAP, X, Y, Z, rx, ry, rz
from .noise import NoiseModel, TimeSchedule
from .simulator import compare_ideal_and_noisy, sample_counts, sample_counts_trajectory
from .stats import DifferenceTest, ProportionEstimate, compare_counts, proportions


ARCHIVE_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ExperimentBatch:
    circuit: Circuit
    noise_model: NoiseModel
    shots: int
    base_seed: int
    name: str = "experiment"
    schedule: Optional[TimeSchedule] = None

    def seeds(self, repeats: int) -> list[int]:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        digest = hashlib.sha256(f"{self.base_seed}:{self.name}".encode("utf-8")).digest()
        root = int.from_bytes(digest[:8], "little")
        rng = random.Random(root)
        return [rng.getrandbits(64) for _ in range(repeats)]

    @property
    def normalized_schedule(self) -> Optional[TimeSchedule]:
        if self.schedule is None:
            return None
        return self.schedule.normalized(len(self.circuit.operations))


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
    environment_changes: Optional[list[dict[str, Any]]] = None
    measurement_conditions: Optional[list[dict[str, Any]]] = None
    schedule: Optional[TimeSchedule] = None

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
            "environment_changes": self.environment_changes,
            "measurement_conditions": self.measurement_conditions,
            "schedule": self.schedule.to_dict() if self.schedule is not None else None,
        }


def run_batch(
    batch: ExperimentBatch, repeats: int = 1, outcomes: Optional[list[str]] = None
) -> list[BatchRun]:
    schedule = (
        batch.schedule.normalized(len(batch.circuit.operations))
        if batch.schedule is not None
        else None
    )
    ideal_result, noisy_result = compare_ideal_and_noisy(
        batch.circuit, batch.noise_model, schedule
    )
    correlated = bool(batch.noise_model.groups)
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
    environment_changes = None
    measurement_conditions = None
    if noisy_result.environment_records is not None:
        environment_changes = [dict(record) for record in noisy_result.environment_records]
    if noisy_result.measurement_conditions is not None:
        measurement_conditions = [dict(record) for record in noisy_result.measurement_conditions]
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
                environment_changes=environment_changes,
                measurement_conditions=measurement_conditions,
                schedule=schedule,
            )
        )
    return runs


def _derive_seed(seed: int, label: bytes) -> int:
    digest = hashlib.sha256(seed.to_bytes(8, "little", signed=False) + label).digest()
    return int.from_bytes(digest[:8], "little")


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
        schedule=TimeSchedule.from_dict(data["schedule"])
        if data.get("schedule") is not None
        else None,
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
        ):
            raise ValueError("archive could not be reproduced with its stored seed and specification")
    return rerun


@dataclass(frozen=True)
class ExperimentTask:
    batch: ExperimentBatch
    repeats: int = 1
    key: Optional[str] = None

    @property
    def task_key(self) -> str:
        return self.key or self.batch.name


def compare_schedules(
    circuit: Circuit,
    noise_model: NoiseModel,
    schedules: dict[str, TimeSchedule],
    shots: int,
    base_seed: int,
) -> dict[str, list[BatchRun]]:
    """Run one operation sequence under independently named physical timings."""
    results = {}
    for name, schedule in schedules.items():
        batch = ExperimentBatch(
            circuit=circuit,
            noise_model=noise_model,
            shots=shots,
            base_seed=base_seed,
            name=name,
            schedule=schedule.normalized(len(circuit.operations)),
        )
        results[name] = run_batch(batch)
    return results


def save_schedule_comparison(
    path: str | Path,
    batches: dict[str, ExperimentBatch],
    results: dict[str, list[BatchRun]],
) -> None:
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "comparison_type": "time_schedule",
        "schedules": {
            name: {
                "batch": _batch_payload(batch),
                "environment_changes": [
                    run.environment_changes for run in results[name]
                ],
                "measurement_conditions": [
                    run.measurement_conditions for run in results[name]
                ],
                "statistics": [run.to_dict() for run in results[name]],
            }
            for name, batch in batches.items()
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _batch_payload(batch: ExperimentBatch) -> dict[str, Any]:
    return {
        "name": batch.name,
        "shots": batch.shots,
        "base_seed": batch.base_seed,
        "circuit": circuit_to_dict(batch.circuit),
        "noise_model": batch.noise_model.to_dict(),
        "schedule": batch.normalized_schedule.to_dict()
        if batch.normalized_schedule is not None
        else None,
    }


class BatchCheckpoint:
    """JSON checkpoint for a collection of independently reproducible batches."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.completed: dict[str, list[BatchRun]] = {}
        self.pause_point: Optional[dict[str, Any]] = None

    @classmethod
    def load(cls, path: str | Path) -> "BatchCheckpoint":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        checkpoint = cls(path)
        checkpoint.completed = {
            key: [_batch_run_from_dict(item) for item in runs]
            for key, runs in data.get("completed", {}).items()
        }
        checkpoint.pause_point = data.get("pause_point")
        return checkpoint

    def save(self) -> None:
        if self.path is None:
            raise ValueError("checkpoint requires a path")
        payload = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "pause_point": self.pause_point,
            "completed": {
                key: [run.to_dict() for run in runs]
                for key, runs in self.completed.items()
            },
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)

    def resume(
        self,
        tasks: Sequence[ExperimentTask],
        order: Optional[Sequence[str]] = None,
        pause_after: Optional[str] = None,
    ) -> dict[str, list[BatchRun]]:
        keys = [task.task_key for task in tasks]
        if len(set(keys)) != len(keys):
            raise ValueError("task keys must be unique")
        task_by_key = {task.task_key: task for task in tasks}
        selected = list(order) if order is not None else keys
        if set(selected) != set(keys):
            raise ValueError("execution order must contain each task exactly once")
        for key in selected:
            if key in self.completed:
                continue
            task = task_by_key[key]
            self.completed[key] = run_batch(task.batch, repeats=task.repeats)
            if pause_after is not None and key == pause_after:
                self.pause_point = {"after_task": key, "resumable": True}
                self.save()
                return dict(self.completed)
        self.pause_point = None
        self.save()
        return dict(self.completed)


@dataclass
class ResumableBatch:
    path: Path
    batch: ExperimentBatch
    repeats: int
    completed: list[BatchRun]
    pause_point: Optional[dict[str, Any]] = None

    @classmethod
    def start(
        cls,
        path: str | Path,
        batch: ExperimentBatch,
        repeats: int = 1,
        pause_after: Optional[dict[str, Any]] = None,
    ) -> "ResumableBatch":
        execution = cls(
            path=Path(path),
            batch=batch,
            repeats=repeats,
            completed=[],
            pause_point=pause_after,
        )
        execution._run_next()
        return execution

    @classmethod
    def resume(cls, path: str | Path) -> "ResumableBatch":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        batch_data = data["batch"]
        batch = ExperimentBatch(
            circuit=circuit_from_dict(batch_data["circuit"]),
            noise_model=NoiseModel.from_dict(batch_data["noise_model"]),
            shots=int(batch_data["shots"]),
            base_seed=int(batch_data["base_seed"]),
            name=batch_data["name"],
            schedule=TimeSchedule.from_dict(batch_data["schedule"])
            if batch_data.get("schedule") is not None
            else None,
        )
        execution = cls(
            path=Path(path),
            batch=batch,
            repeats=int(data["repeats"]),
            completed=[_batch_run_from_dict(item) for item in data.get("completed", [])],
            pause_point=data.get("pause_point"),
        )
        execution._run_next()
        return execution

    @property
    def runs(self) -> list[BatchRun]:
        return list(self.completed)

    def replay(self) -> list[BatchRun]:
        return run_batch(self.batch, repeats=self.repeats)

    def _run_next(self) -> None:
        if len(self.completed) >= self.repeats:
            self.pause_point = None
            self._save()
            return
        next_index = len(self.completed)
        should_pause = self.pause_point is not None and not self.completed
        if should_pause:
            self.completed.extend(run_batch(self.batch, repeats=next_index + 1)[next_index:])
        else:
            self.completed.extend(run_batch(self.batch, repeats=self.repeats)[next_index:])
        if should_pause and len(self.completed) < self.repeats:
            self.pause_point["resumable"] = True
        else:
            self.pause_point = None
        self._save()

    def _save(self) -> None:
        payload = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "repeats": self.repeats,
            "pause_point": self.pause_point,
            "batch": _batch_payload(self.batch),
            "completed": [run.to_dict() for run in self.completed],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)


def start_resumable_batch(
    path: str | Path,
    batch: ExperimentBatch,
    repeats: int = 1,
    pause_after_measurement: bool = False,
    pause_after_idle: bool = False,
) -> ResumableBatch:
    pause_point = None
    if pause_after_measurement or pause_after_idle:
        pause_point = {
            "kind": "measurement" if pause_after_measurement else "idle",
            "resumable": True,
        }
    return ResumableBatch.start(path, batch, repeats, pause_point)


def resume_batch(path: str | Path) -> ResumableBatch:
    return ResumableBatch.resume(path)


def _batch_run_from_dict(data: dict[str, Any]) -> BatchRun:
    tests = {
        outcome: DifferenceTest(
            outcome=outcome,
            observed_difference=item["observed_difference"],
            standard_error=item["standard_error"],
            confidence_95=tuple(item["confidence_95"]),
            significant=item["significant"],
            exact_difference=item["exact_difference"],
            has_exact_effect=item["has_exact_effect"],
            minimum_expected_count=item["minimum_expected_count"],
            enough_samples=item["enough_samples"],
        )
        for outcome, item in data["difference_tests"].items()
    }
    estimates_data = data.get("ideal_estimates", {})
    ideal_estimates = {
        key: ProportionEstimate(
            outcome=value.get("outcome", key),
            count=value.get("count", data["ideal_counts"].get(key, 0)),
            shots=value.get("shots", data["shots"]),
            empirical=value.get("empirical", data["ideal_counts"].get(key, 0) / data["shots"]),
            exact=value.get("exact", data["ideal_probabilities"].get(key, 0.0)),
            standard_error=value.get("standard_error", 0.0),
            confidence_95=tuple(value.get("confidence_95", (0.0, 0.0))),
        )
        for key, value in estimates_data.items()
    }
    noisy_estimates_data = data.get("noisy_estimates", {})
    noisy_estimates = {
        key: ProportionEstimate(
            outcome=value.get("outcome", key),
            count=value.get("count", data["noisy_counts"].get(key, 0)),
            shots=value.get("shots", data["shots"]),
            empirical=value.get("empirical", data["noisy_counts"].get(key, 0) / data["shots"]),
            exact=value.get("exact", data["noisy_probabilities"].get(key, 0.0)),
            standard_error=value.get("standard_error", 0.0),
            confidence_95=tuple(value.get("confidence_95", (0.0, 0.0))),
        )
        for key, value in noisy_estimates_data.items()
    }
    return BatchRun(
        batch_name=data["batch_name"],
        shots=data["shots"],
        base_seed=data["base_seed"],
        repeat_index=data["repeat_index"],
        seed=data["seed"],
        ideal_probabilities=data["ideal_probabilities"],
        noisy_probabilities=data["noisy_probabilities"],
        ideal_counts=data["ideal_counts"],
        noisy_counts=data["noisy_counts"],
        ideal_estimates=ideal_estimates,
        noisy_estimates=noisy_estimates,
        difference_tests=tests,
        noise_seed=data.get("noise_seed"),
        measure_seed=data.get("measure_seed"),
        noise_trajectories=data.get("noise_trajectories"),
        environment_changes=data.get("environment_changes"),
        measurement_conditions=data.get("measurement_conditions"),
        schedule=TimeSchedule.from_dict(data["schedule"])
        if data.get("schedule") is not None
        else None,
    )


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
