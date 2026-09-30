"""Correlated-noise experiment batches and correlation-strength sweeps."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .correlated import CorrelatedNoiseEvent, CorrelatedNoiseModel
from .dynamics import exact_probabilities, simulate_shot, unitary_gate_count
from .experiments import rerun_archive
from .gates import Circuit
from .serialization import circuit_from_dict, circuit_to_dict
from .stats import DifferenceTest, compare_counts


def repeat_seeds(base_seed: int, name: str, repeats: int) -> list[int]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    root = int.from_bytes(digest[:8], "little")
    rng = random.Random(root)
    return [rng.getrandbits(64) for _ in range(repeats)]


def _derive_seed(seed: int, label: bytes) -> int:
    digest = hashlib.sha256(seed.to_bytes(8, "little", signed=False) + label).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class TrajectoryBatch:
    circuit: Circuit
    noise_model: CorrelatedNoiseModel
    shots: int
    base_seed: int
    name: str = "trajectory"

    def validate(self) -> None:
        self.noise_model.validate(unitary_gate_count(self.circuit))
        for event in self.noise_model.events:
            if event.qubit >= self.circuit.num_qubits:
                raise ValueError("noise event targets a qubit outside circuit")
        if self.shots <= 0:
            raise ValueError("shots must be positive")

    def seeds(self, repeats: int) -> list[int]:
        return repeat_seeds(self.base_seed, self.name, repeats)


@dataclass(frozen=True)
class TrajectoryRun:
    batch_name: str
    shots: int
    base_seed: int
    repeat_index: int
    seed: int
    noise_seed: int
    measurement_seed: int
    ideal_probabilities: dict[str, float]
    noisy_probabilities: dict[str, float]
    ideal_counts: dict[str, int]
    noisy_counts: dict[str, int]
    difference_tests: dict[str, DifferenceTest]
    traces: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "shots": self.shots,
            "base_seed": self.base_seed,
            "repeat_index": self.repeat_index,
            "seed": self.seed,
            "noise_seed": self.noise_seed,
            "measurement_seed": self.measurement_seed,
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
            "traces": list(self.traces),
        }


def _empty_counts(probabilities: dict[str, float]) -> dict[str, int]:
    return {outcome: 0 for outcome in probabilities}


def run_trajectory_batch(
    batch: TrajectoryBatch,
    repeats: int = 1,
    outcomes: Optional[list[str]] = None,
    record_traces: bool = True,
) -> list[TrajectoryRun]:
    batch.validate()
    ideal_probabilities = exact_probabilities(batch.circuit, None)
    noisy_probabilities = exact_probabilities(batch.circuit, batch.noise_model)
    runs: list[TrajectoryRun] = []
    for index, seed in enumerate(batch.seeds(repeats)):
        noise_seed = _derive_seed(seed, b"correlated-noise")
        measurement_seed = _derive_seed(seed, b"measurement")
        ideal_seed = _derive_seed(seed, b"ideal-measurement")

        noise_rng = random.Random(noise_seed)
        measurement_rng = random.Random(measurement_seed)
        ideal_rng = random.Random(ideal_seed)

        noisy_counts = _empty_counts(noisy_probabilities)
        ideal_counts = _empty_counts(ideal_probabilities)
        traces: list[dict[str, Any]] = []
        for shot_index in range(batch.shots):
            noisy_trace = simulate_shot(
                batch.circuit, batch.noise_model, noise_rng, measurement_rng
            )
            noisy_counts[noisy_trace.final_outcome] += 1
            ideal_trace = simulate_shot(batch.circuit, None, random.Random(), ideal_rng)
            ideal_counts[ideal_trace.final_outcome] += 1
            if record_traces:
                traces.append(
                    {
                        "shot": shot_index,
                        "final_outcome": noisy_trace.final_outcome,
                        "branch_path": list(noisy_trace.branch_path),
                        "mid_measurements": {
                            str(bit): value for bit, value in noisy_trace.measurement_outcomes.items()
                        },
                        "noise_errors": {
                            key: list(values) for key, values in noisy_trace.noise_errors.items()
                        },
                    }
                )
        tests = compare_counts(
            ideal_counts,
            noisy_counts,
            ideal_probabilities,
            noisy_probabilities,
            outcomes=outcomes,
        )
        runs.append(
            TrajectoryRun(
                batch_name=batch.name,
                shots=batch.shots,
                base_seed=batch.base_seed,
                repeat_index=index,
                seed=seed,
                noise_seed=noise_seed,
                measurement_seed=measurement_seed,
                ideal_probabilities=ideal_probabilities,
                noisy_probabilities=noisy_probabilities,
                ideal_counts=ideal_counts,
                noisy_counts=noisy_counts,
                difference_tests=tests,
                traces=tuple(traces),
            )
        )
    return runs


@dataclass(frozen=True)
class CorrelationSweepPoint:
    correlation: float
    noisy_probabilities: dict[str, float]
    run: TrajectoryRun


def with_correlation(model: CorrelatedNoiseModel, correlation: float, name: Optional[str] = None):
    events = tuple(
        CorrelatedNoiseEvent(
            channel=event.channel,
            after_gate=event.after_gate,
            qubit=event.qubit,
            probability=event.probability,
            trajectory=event.trajectory,
            correlation=correlation,
            params=dict(event.params),
        )
        for event in model.events
    )
    return CorrelatedNoiseModel(name or model.name, events)


def sweep_correlation(
    batch: TrajectoryBatch,
    correlations: Sequence[float],
) -> list[CorrelationSweepPoint]:
    """Run the identical circuit at each correlation strength."""

    points: list[CorrelationSweepPoint] = []
    for correlation in correlations:
        model = with_correlation(batch.noise_model, float(correlation))
        point_batch = TrajectoryBatch(
            circuit=batch.circuit,
            noise_model=model,
            shots=batch.shots,
            base_seed=batch.base_seed,
            name=f"{batch.name}:rho={float(correlation):.6g}",
        )
        run = run_trajectory_batch(point_batch, record_traces=False)[0]
        points.append(
            CorrelationSweepPoint(
                correlation=float(correlation),
                noisy_probabilities=dict(run.noisy_probabilities),
                run=run,
            )
        )
    return points


def save_trajectory_archive(
    path: str | Path,
    batch: TrajectoryBatch,
    runs: list[TrajectoryRun],
) -> None:
    payload = {
        "schema_version": 2,
        "archive_kind": "correlated",
        "batch": {
            "name": batch.name,
            "shots": batch.shots,
            "base_seed": batch.base_seed,
            "circuit": circuit_to_dict(batch.circuit),
            "noise_model": batch.noise_model.to_dict(),
        },
        "runs": [run.to_dict() for run in runs],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_trajectory_batch(path: str | Path) -> tuple[TrajectoryBatch, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("archive_kind") != "correlated" or payload.get("schema_version") != 2:
        raise ValueError("file is not a correlated-noise trajectory archive")
    data = payload["batch"]
    batch = TrajectoryBatch(
        circuit=circuit_from_dict(data["circuit"]),
        noise_model=CorrelatedNoiseModel.from_dict(data["noise_model"]),
        shots=int(data["shots"]),
        base_seed=int(data["base_seed"]),
        name=data["name"],
    )
    return batch, payload["runs"]


def rerun_trajectory_archive(path: str | Path) -> list[TrajectoryRun]:
    batch, saved_runs = load_trajectory_batch(path)
    reruns = run_trajectory_batch(batch, repeats=len(saved_runs))
    for saved, current in zip(saved_runs, reruns):
        if (
            saved["seed"] != current.seed
            or saved["noise_seed"] != current.noise_seed
            or saved["measurement_seed"] != current.measurement_seed
            or saved["noisy_counts"] != current.noisy_counts
            or saved["ideal_counts"] != current.ideal_counts
        ):
            raise ValueError("correlated archive could not be reproduced with its stored seeds")
    return reruns


def archive_kind(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("archive_kind") == "correlated":
        return "correlated"
    return "independent"


def replay_archive(path: str | Path):
    """Replay either a legacy independent archive or a correlated archive."""

    if archive_kind(path) == "correlated":
        return rerun_trajectory_archive(path)
    return rerun_archive(path)
