"""Compare the same experiment under disturbances placed at different times.

Siblings are forked from one experiment prefix: circuit, ambient noise,
schedule, shots and base seed are shared; only the finite disturbance window
moves.  The pulse is integrated at its real time, including windows that fall
in the middle of an operation, and the ambient baseline is restored once the
window ends.
"""

import sys
from math import pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    AmbientNoise,
    Circuit,
    Disturbance,
    DisturbancePlan,
    ExperimentBatch,
    NoiseModel,
    TimeSchedule,
    run_batch,
)


def main() -> None:
    circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
    ambient = AmbientNoise(qubits=(0,), bit_flip=0.05, name="background")
    model = NoiseModel("ambient-background", (), (), ambient)
    prefix = ExperimentBatch(
        circuit=circuit,
        noise_model=model,
        shots=4000,
        base_seed=20261002,
        name="disturbance-prefix",
        # Gate 1 during [0, 1], idle (1, 3], gate 2 during [3, 4].
        schedule=TimeSchedule((1.0, 1.0), (0.0, 2.0)),
    )

    variants = {
        "no-disturbance": prefix,
        "pulse-mid-gate": prefix.fork_disturbance(
            DisturbancePlan((Disturbance(0.25, 0.9, bit_flip=1.5),)),
            name="pulse-mid-gate",
        ),
        "pulse-mid-idle": prefix.fork_disturbance(
            DisturbancePlan((Disturbance(1.5, 2.1, bit_flip=1.5),)),
            name="pulse-mid-idle",
        ),
        "persistent-pulse": prefix.fork_disturbance(
            DisturbancePlan((Disturbance(0.0, 4.0, bit_flip=0.8),)),
            name="persistent-pulse",
        ),
        "instant-boundary": prefix.fork_disturbance(
            DisturbancePlan((Disturbance(1.0, 1.0, bit_flip=10.0),)),
            name="instant-boundary",
        ),
    }

    for name, batch in variants.items():
        run = run_batch(batch)[0]
        print(name, run.noisy_probabilities, run.noisy_counts)


if __name__ == "__main__":
    main()
