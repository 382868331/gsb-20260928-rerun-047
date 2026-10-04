"""Compare common, independent and mixed finite-duration environment events."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    Circuit,
    ExperimentBatch,
    JointEnvironmentEvent,
    NoiseModel,
    TimeSchedule,
    run_batch,
    save_batch_archive,
)


def batch(name: str, common_probability: float, independent_probability: float, seed: int):
    circuit = Circuit(2).rx(0, 0.0).rx(1, 0.0)
    event = JointEnvironmentEvent(
        name="shared-pulse",
        start=1.0,
        end=3.0,
        after_gate=1,
        qubits=(0, 1),
        common_probability=common_probability,
        independent_probability=independent_probability,
    )
    schedule = TimeSchedule((1.0, 1.0), name="two-event-points")
    return ExperimentBatch(
        circuit,
        NoiseModel(name, (), (), (event,)),
        shots=4096,
        base_seed=seed,
        name=name,
        schedule=schedule,
    )


def main() -> None:
    variants = {
        "common": batch("common", common_probability=0.2, independent_probability=0.0, seed=1),
        "independent": batch("independent", common_probability=0.0, independent_probability=0.1, seed=2),
        "mixed": batch("mixed", common_probability=0.2, independent_probability=0.1, seed=3),
    }
    output_dir = Path("joint_environment_results")
    output_dir.mkdir(exist_ok=True)
    for name, experiment in variants.items():
        run = run_batch(experiment)[0]
        print(name, run.noisy_probabilities, run.noisy_counts)
        save_batch_archive(output_dir / f"{name}.json", experiment, [run])


if __name__ == "__main__":
    main()
