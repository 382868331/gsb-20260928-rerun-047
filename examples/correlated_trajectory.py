"""Temporal correlation sweep plus a mid-circuit conditional branch.

Run with ``python examples/correlated_trajectory.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    Circuit,
    CorrelatedNoiseEvent,
    CorrelatedNoiseModel,
    TrajectoryBatch,
    exact_probabilities,
    run_trajectory_batch,
    save_trajectory_archive,
    sweep_correlation,
)
from quantum_noise.report import format_correlation_sweep, format_trajectory_run_report


def correlated_flip_batch(rho: float) -> TrajectoryBatch:
    # Two X gates with a correlated bit flip inserted after each one.
    circuit = Circuit(1).x(0).x(0)
    model = CorrelatedNoiseModel(
        f"two-flips-rho{rho:g}",
        [
            CorrelatedNoiseEvent("bit_flip", after_gate=0, qubit=0, probability=0.2, correlation=rho),
            CorrelatedNoiseEvent("bit_flip", after_gate=1, qubit=0, probability=0.2, correlation=rho),
        ],
    )
    return TrajectoryBatch(circuit, model, shots=2000, base_seed=20260930, name="two-flips")


def main() -> None:
    base = correlated_flip_batch(0.0)
    print(format_correlation_sweep(sweep_correlation(base, [0.0, 0.5, 1.0])))

    run = run_trajectory_batch(correlated_flip_batch(0.8))[0]
    print(format_trajectory_run_report(run))
    save_trajectory_archive(Path("correlated_batch.json"), correlated_flip_batch(0.8), [run])

    # Mid-circuit measurement followed by a conditional X: the exact
    # calculation enumerates both classical branches.
    branched = Circuit(2).h(0).measure(0, 0).cx_conditional(0, 1)
    branched_model = CorrelatedNoiseModel(
        "branch-noise",
        [
            CorrelatedNoiseEvent("bit_flip", after_gate=0, qubit=0, probability=0.1, correlation=0.0),
            CorrelatedNoiseEvent("bit_flip", after_gate=1, qubit=0, probability=0.1, correlation=0.0),
        ],
    )
    print("ideal branches :", exact_probabilities(branched))
    print("noisy branches :", exact_probabilities(branched, branched_model))


if __name__ == "__main__":
    main()
