"""Compare identical circuits under perturbations placed at different times."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    Circuit,
    ExperimentBatch,
    ExperimentPrefix,
    NoiseModel,
    NoiseRule,
    Perturbation,
    run_batch,
    save_batch_archive,
    TimeSchedule,
)


def main() -> None:
    circuit = Circuit(1).h(0).h(0)
    schedule = TimeSchedule((1.0, 1.0), (0.0, 5.0), "long-idle")
    model = NoiseModel("idle-bit-flip", [NoiseRule("bit_flip", 0.1, (0,), after_gate=1)])
    base = ExperimentBatch(
        circuit,
        model,
        shots=2000,
        base_seed=20261002,
        name="prefix",
        schedule=schedule,
    )
    prefix = ExperimentPrefix(base, anchor_time=2.0)
    variants = {
        "unperturbed": base,
        "mid-idle": prefix.fork("mid-idle", [Perturbation(4.0, 8.0, 0.4, name="pulse")]),
        "after-event": prefix.fork("after-event", [Perturbation(4.0, 6.5, 0.4, name="pulse")]),
        "boundary": prefix.fork("boundary", [Perturbation(4.0, 7.0, 0.4, name="pulse")]),
        "overlap": prefix.fork(
            "overlap",
            [
                Perturbation(4.0, 8.0, 0.2, name="background", priority=0),
                Perturbation(5.0, 7.5, 0.5, name="spike", priority=10),
            ],
        ),
    }
    for name, batch in variants.items():
        run = run_batch(batch)[0]
        print(name, run.noisy_probabilities)
        save_batch_archive(Path(f"perturbation_{name}.json"), batch, [run])


if __name__ == "__main__":
    main()
