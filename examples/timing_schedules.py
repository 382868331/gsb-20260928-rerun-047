"""Compare one operation sequence under different time schedules.

The circuit and the noise sites are identical in both experiments; only the
duration of the operations and the idle wait between them change.  Because the
environment correlation decays with elapsed time, the two experiments sample a
different joint error distribution.  For each experiment the archive stores
separately:

* the environment evolution (joint noise paths and their exact probabilities),
* the measurement conditions (mid-circuit branch frequencies),
* the measured statistics (counts, estimates and difference tests).

The script also runs one batch in two segments, pausing after the first
segment, and shows the resumed replay matches the uninterrupted run.  It then
runs the two batches interleaved and checks the execution order does not
change either result.
"""

import sys
from math import pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    Circuit,
    CorrelatedNoiseGroup,
    ExperimentBatch,
    NoiseModel,
    ResumableBatchRun,
    run_batch,
)
from quantum_noise.schedule import Schedule
from quantum_noise.simulator import simulate


def make_batch(label: str, idle: float, shots: int, seed: int) -> ExperimentBatch:
    circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
    schedule = Schedule((1.0, 1.0), (idle, 0.0), name=label)
    group = CorrelatedNoiseGroup(
        channel="bit_flip",
        sites=((0, 0), (1, 0)),
        probability=0.2,
        time_correlation=0.5,
        name=f"time-correlated-{label}",
    )
    return ExperimentBatch(
        circuit=circuit,
        noise_model=NoiseModel(f"timing-{label}", (), (group,)),
        shots=shots,
        base_seed=seed,
        name=label,
        schedule=schedule,
    )


def main() -> None:
    tight = make_batch("tight", idle=0.0, shots=2000, seed=20261001)
    spaced = make_batch("spaced", idle=8.0, shots=2000, seed=20261001)

    print("same operation sequence rx(pi/2) rx(pi/2), different time schedules")
    for batch in (tight, spaced):
        exact = simulate(batch.circuit, batch.noise_model, batch.schedule)
        run = run_batch(batch)[0]
        print(
            f"schedule={batch.schedule.name:6s} total_time={batch.schedule.total_duration:5.1f} "
            f"exact P(0)={exact.probabilities['0']:.6f} counts={run.noisy_counts}"
        )
        print("  environment paths (separately archived):")
        for path in run.noise_trajectories:
            print(
                f"    label={path['label']} register={path['register']} "
                f"probability={path['probability']:.4f} "
                f"conditioned_outcomes={path['outcome_probabilities']}"
            )
        print(f"  sampled noise-path frequencies: {run.noise_paths}")
        print(f"  measurement conditions: {run.measurement_conditions}")

    # Pause after the first 300 shots, resume, and compare to one full run.
    interrupted = ResumableBatchRun(tight)
    interrupted.advance(300)
    checkpoint = interrupted.checkpoint()
    resumed = ResumableBatchRun.restore(tight, checkpoint)
    resumed.advance(tight.shots - 300)
    full = run_batch(tight)[0]
    assert resumed.finish().noisy_counts == full.noisy_counts
    print()
    print("paused after shot 300 and resumed: counts identical to one full run")

    # Interleave the two batches; each must keep its own environment stream.
    tight_runner = ResumableBatchRun(tight)
    spaced_runner = ResumableBatchRun(spaced)
    tight_runner.advance(500)
    spaced_runner.advance(900)
    tight_runner.advance(tight.shots - 500)
    spaced_runner.advance(spaced.shots - 900)
    assert tight_runner.finish().noisy_counts == full.noisy_counts
    assert spaced_runner.finish().noisy_counts == run_batch(spaced)[0].noisy_counts
    print("interleaving the two batches leaves both results unchanged")


if __name__ == "__main__":
    main()
