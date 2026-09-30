"""Compare one circuit under tight and long-idle time schedules."""

import sys
from math import pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    Circuit,
    CorrelatedNoiseGroup,
    ExperimentBatch,
    NoiseModel,
    TimeSchedule,
    compare_schedules,
    save_schedule_comparison,
)


def main() -> None:
    circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
    group = CorrelatedNoiseGroup(
        channel="bit_flip",
        sites=((0, 0), (1, 0)),
        probability=0.2,
        correlation=1.0,
        correlation_time=1.0,
        name="finite-duration-disturbance",
    )
    model = NoiseModel("time-correlated", (), (group,))
    schedules = {
        "tight": TimeSchedule((0.0, 0.0), (0.0, 0.0), "tight"),
        "long-idle": TimeSchedule((0.0, 0.0), (0.0, 20.0), "long-idle"),
    }
    results = compare_schedules(
        circuit=circuit,
        noise_model=model,
        schedules=schedules,
        shots=2000,
        base_seed=20261001,
    )
    batches = {
        name: ExperimentBatch(
            circuit,
            model,
            shots=2000,
            base_seed=20261001,
            name=name,
            schedule=schedules[name],
        )
        for name in schedules
    }
    save_schedule_comparison(Path("schedule_comparison.json"), batches, results)
    for name, runs in results.items():
        run = runs[0]
        print(name, run.noisy_probabilities, run.noisy_counts)


if __name__ == "__main__":
    main()
