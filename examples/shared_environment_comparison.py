"""Joint-result controls for independent, common and mixed environmental noise."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    ExperimentBatch,
    NoiseModel,
    SharedEnvironmentGroup,
    TimeSchedule,
    run_batch,
)
from quantum_noise.gates import Circuit, I


def make_model(name: str, common_probability: float, independent_probability: float):
    group = SharedEnvironmentGroup(
        channel="bit_flip",
        sites=((0, 0), (0, 1)),
        common_probability=common_probability,
        independent_probability=independent_probability,
        name=name,
    )
    return NoiseModel(name, (), (), (group,))


def main() -> None:
    circuit = Circuit(2).apply("i", I, 0).apply("i", I, 1)
    overlap = TimeSchedule((0.0, 0.0), (0.0, 0.0), "overlap")
    nonoverlap = TimeSchedule((0.0, 0.0), (0.0, 20.0), "non-overlap")
    variants = {
        "no-common-effect": make_model("no-common-effect", 0.0, 0.25),
        "fully-common": make_model("fully-common", 0.25, 0.0),
        "common-plus-independent": make_model("common-plus-independent", 0.2, 0.1),
    }
    for name, model in variants.items():
        batch = ExperimentBatch(
            circuit,
            model,
            shots=2000,
            base_seed=20261003,
            name=name,
            schedule=overlap,
        )
        run = run_batch(batch)[0]
        print(name, "joint exact:", run.noisy_probabilities, "counts:", run.noisy_counts)

    persistent_group = SharedEnvironmentGroup(
        "bit_flip",
        ((0, 0), (1, 1)),
        common_probability=0.25,
        independent_probability=0.0,
        common_correlation=1.0,
        correlation_time=1.0,
        name="finite-duration-common",
    )
    separated = ExperimentBatch(
        circuit,
        NoiseModel("non-overlap", (), (), (persistent_group,)),
        shots=2000,
        base_seed=20261003,
        name="non-overlap",
        schedule=nonoverlap,
    )
    run = run_batch(separated)[0]
    print("non-overlap", "joint exact:", run.noisy_probabilities, "counts:", run.noisy_counts)


if __name__ == "__main__":
    main()
