"""A small repeatable quantum-noise research toolkit."""

from .correlated import CorrelatedNoiseEvent, CorrelatedNoiseModel
from .dynamics import ShotTrace, exact_probabilities, simulate_shot
from .experiments import (
    BatchRun,
    ExperimentBatch,
    load_batch,
    rerun_archive,
    run_batch,
    save_batch_archive,
)
from .gates import (
    Circuit,
    ConditionalOperation,
    Measurement,
    bell_circuit,
    ghz_circuit,
)
from .noise import NoiseModel, NoiseRule
from .simulator import SimulationResult, compare_ideal_and_noisy, sample_counts, simulate
from .stats import DifferenceTest, ProportionEstimate, compare_counts, proportions
from .trajectories import (
    CorrelationSweepPoint,
    TrajectoryBatch,
    TrajectoryRun,
    archive_kind,
    replay_archive,
    rerun_trajectory_archive,
    run_trajectory_batch,
    save_trajectory_archive,
    sweep_correlation,
    with_correlation,
)

__all__ = [
    "BatchRun",
    "Circuit",
    "ConditionalOperation",
    "CorrelatedNoiseEvent",
    "CorrelatedNoiseModel",
    "CorrelationSweepPoint",
    "DifferenceTest",
    "ExperimentBatch",
    "Measurement",
    "NoiseModel",
    "NoiseRule",
    "ProportionEstimate",
    "ShotTrace",
    "SimulationResult",
    "TrajectoryBatch",
    "TrajectoryRun",
    "archive_kind",
    "bell_circuit",
    "compare_counts",
    "compare_ideal_and_noisy",
    "exact_probabilities",
    "ghz_circuit",
    "load_batch",
    "proportions",
    "replay_archive",
    "rerun_archive",
    "rerun_trajectory_archive",
    "run_batch",
    "run_trajectory_batch",
    "sample_counts",
    "save_batch_archive",
    "save_trajectory_archive",
    "simulate",
    "simulate_shot",
    "sweep_correlation",
    "with_correlation",
]
