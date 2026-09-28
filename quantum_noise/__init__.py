"""A small repeatable quantum-noise research toolkit."""

from .experiments import BatchRun, ExperimentBatch, run_batch, save_batch_archive
from .gates import Circuit, bell_circuit, ghz_circuit
from .noise import NoiseModel, NoiseRule
from .simulator import SimulationResult, compare_ideal_and_noisy, sample_counts, simulate
from .stats import DifferenceTest, ProportionEstimate, compare_counts, proportions

__all__ = [
    "BatchRun",
    "Circuit",
    "DifferenceTest",
    "ExperimentBatch",
    "NoiseModel",
    "NoiseRule",
    "ProportionEstimate",
    "bell_circuit",
    "compare_counts",
    "compare_ideal_and_noisy",
    "ghz_circuit",
    "proportions",
    "run_batch",
    "sample_counts",
    "save_batch_archive",
    "simulate",
    "SimulationResult",
]
