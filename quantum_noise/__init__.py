"""A small repeatable quantum-noise research toolkit."""

from .experiments import BatchRun, ExperimentBatch, run_batch, save_batch_archive
from .gates import Circuit, bell_circuit, ghz_circuit
from .noise import CorrelatedNoiseGroup, NoiseModel, NoiseRule
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
    "Circuit",
    "CorrelatedNoiseGroup",
    "DifferenceTest",
    "ExperimentBatch",
    "NoiseModel",
    "NoiseRule",
    "NoiseTrajectory",
    "ProportionEstimate",
    "bell_circuit",
    "compare_counts",
    "compare_ideal_and_noisy",
    "ghz_circuit",
    "proportions",
    "run_batch",
    "sample_counts",
    "sample_counts_trajectory",
    "save_batch_archive",
    "simulate",
    "SimulationResult",
]
