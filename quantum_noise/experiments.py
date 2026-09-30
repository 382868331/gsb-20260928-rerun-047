"""Repeatable experiment batches and JSON serialization."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .noise import NoiseModel
from .serialization import circuit_from_dict, circuit_to_dict
from .simulator import compare_ideal_and_noisy, sample_counts
from .stats import DifferenceTest, ProportionEstimate, compare_counts, proportions


@dataclass(frozen=True)
class ExperimentBatch:
    circuit: Circuit
    noise_model: NoiseModel
    shots: int
    base_seed: int
    name: str = "experiment"

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
        }


def run_batch(
    batch: ExperimentBatch, repeats: int = 1, outcomes: Optional[list[str]] = None
) -> list[BatchRun]:
    ideal_result, noisy_result = compare_ideal_and_noisy(batch.circuit, batch.noise_model)
    runs: list[BatchRun] = []
    for index, seed in enumerate(batch.seeds(repeats)):
        ideal_rng_seed = _derive_seed(seed, b"ideal")
        noisy_rng_seed = _derive_seed(seed, b"noisy")
        ideal_counts = sample_counts(ideal_result.probabilities, batch.shots, ideal_rng_seed)
        noisy_counts = sample_counts(noisy_result.probabilities, batch.shots, noisy_rng_seed)
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
        "schema_version": 2,
        "archive_kind": "independent",
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


def load_batch(path: str | Path) -> tuple[ExperimentBatch, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("archive_kind") not in (None, "independent") or payload.get("schema_version") not in (1, 2):
        raise ValueError("file is not an independent-noise batch archive")
    data = payload["batch"]
    batch = ExperimentBatch(
        circuit=circuit_from_dict(data["circuit"]),
        noise_model=NoiseModel.from_dict(data["noise_model"]),
        shots=int(data["shots"]),
        base_seed=int(data["base_seed"]),
        name=data["name"],
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
