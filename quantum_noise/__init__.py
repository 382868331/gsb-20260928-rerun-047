"""A small repeatable quantum-noise research toolkit."""

from .experiments import (
    BatchCheckpoint,
    BatchRun,
    ExperimentBatch,
    ResumableBatchRun,
    load_checkpoint,
    rerun_archive,
    run_batch,
    save_batch_archive,
    save_checkpoint,
)
from .gates import Circuit, bell_circuit, ghz_circuit
from .noise import CorrelatedNoiseGroup, NoiseModel, NoiseRule
from .schedule import Schedule
from .simulator import (
    NoiseTrajectory,
    SimulationResult,
    compare_ideal_and_noisy,
    sample_counts,
    sample_counts_trajectory,
    simulate,
)
from .stats import DifferenceTest, ProportionEstimate, compare_counts, proportions

__all__ = [
    "BatchRun",
    "BatchCheckpoint",
    "Circuit",
    "CorrelatedNoiseGroup",
    "DifferenceTest",
    "ExperimentBatch",
    "NoiseModel",
    "NoiseRule",
    "NoiseTrajectory",
    "ProportionEstimate",
    "ResumableBatchRun",
    "Schedule",
    "bell_circuit",
    "compare_counts",
    "compare_ideal_and_noisy",
    "ghz_circuit",
    "proportions",
    "run_batch",
    "rerun_archive",
    "load_checkpoint",
    "save_checkpoint",
    "sample_counts",
    "sample_counts_trajectory",
    "save_batch_archive",
    "simulate",
    "SimulationResult",
]
