"""Compare common, independent and mixed environmental events.

This is a runnable research workflow: every joint distribution below can
also be computed by hand, and the study can be saved at the mid-circuit
measurement point and continued either with the observed environmental
conditions or with freshly drawn ones.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    Circuit,
    EnvironmentEvent,
    EventCancellation,
    ExperimentBatch,
    MeasurementStudy,
    NoiseModel,
    TimeSchedule,
    run_batch,
    save_batch_archive,
)


def two_qubit_identity():
    """Two identity operations, event fires once at the final boundary."""
    circuit = Circuit(2).rx(0, 0.0).rx(1, 0.0)
    schedule = TimeSchedule((1.0, 1.0), (0.0, 0.0), "two-boundaries")
    return circuit, schedule


def main() -> None:
    circuit, schedule = two_qubit_identity()
    shots = 4000
    seed = 20261004

    variants = {
        "fully-common": EnvironmentEvent(
            1.0 + 1e-9, 3.0, (0, 1),
            common_probability=0.5, name="storm",
        ),
        "fully-independent": EnvironmentEvent(
            1.0 + 1e-9, 3.0, (0, 1),
            independent_probability=0.5, name="storm",
        ),
        "mixed": EnvironmentEvent(
            1.0 + 1e-9, 3.0, (0, 1),
            common_probability=0.25, independent_probability=0.25, name="storm",
        ),
    }

    print("== two-qubit joint measurement distribution ==")
    print("hand values:")
    print("  fully-common      P(00)=P(11)=1/2, P(01)=P(10)=0")
    print("  fully-independent P(00)=P(01)=P(10)=P(11)=1/4")
    print("  mixed             P(00)=7/16, other outcomes=3/16")
    for name, event in variants.items():
        batch = ExperimentBatch(
            circuit, NoiseModel("events"), shots, seed,
            name=name, schedule=schedule, environment_events=(event,),
        )
        run = run_batch(batch)[0]
        print(name, {key: round(value, 4) for key, value in run.noisy_probabilities.items()})
        save_batch_archive(Path(f"environment_{name}.json"), batch, [run])

    print()
    print("== three qubits: partially overlapping events ==")
    three = Circuit(3).rx(0, 0.0).rx(1, 0.0).rx(2, 0.0).rx(0, 0.0)
    three_schedule = TimeSchedule((1.0,) * 4, (0.0,) * 4, "four-boundaries")
    # A targets q0,q1 and B targets q1,q2; both are common flips and their
    # windows overlap on the boundary t=3 but neither window is nested.
    event_a = EnvironmentEvent(1.0 + 1e-9, 2.5, (0, 1),
                               common_probability=1.0, name="A")
    event_b = EnvironmentEvent(2.0 + 1e-9, 3.5, (1, 2),
                               common_probability=1.0, name="B", priority=10)
    batch = ExperimentBatch(
        three, NoiseModel("events"), shots, seed,
        name="overlapping", schedule=three_schedule,
        environment_events=(event_a, event_b),
    )
    run = run_batch(batch)[0]
    print("overlap:", run.noisy_probabilities)
    print("hand value: A flips q0q1 once and B flips q1q2 once -> 101")
    save_batch_archive(Path("environment_overlapping.json"), batch, [run])

    print()
    print("== edge cases ==")
    zero = EnvironmentEvent(2.0, 2.0, (0, 1), common_probability=1.0, name="instant")
    zero_batch = ExperimentBatch(
        three, NoiseModel("events"), shots, seed,
        name="zero-duration", schedule=three_schedule, environment_events=(zero,),
    )
    print("zero duration:", run_batch(zero_batch)[0].noisy_probabilities)
    lasting = EnvironmentEvent(1.0 + 1e-9, 3.5, (0, 1),
                               common_probability=1.0, name="lasting")
    cancelled = ExperimentBatch(
        three, NoiseModel("events"), shots, seed,
        name="cancelled", schedule=three_schedule,
        environment_events=(lasting,),
        event_cancellations=(EventCancellation("lasting", 3.0),),
    )
    print("cancelled at t=3:", run_batch(cancelled)[0].noisy_probabilities)
    try:
        EnvironmentEvent(0.0, 1.0, (0,), common_probability=1.5)
    except ValueError as error:
        print("invalid probability rejected:", error)

    print()
    print("== save at mid-circuit measurement, then continue ==")
    mid_circuit = Circuit(2).rx(0, 0.0).measure(0, 0).rx(1, 0.0)
    mid_schedule = TimeSchedule((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), "with-measure")
    storm = EnvironmentEvent(0.5, 3.5, (0, 1),
                             common_probability=0.5, name="storm")
    study = MeasurementStudy(
        mid_circuit, NoiseModel("events"), mid_schedule,
        measurement_time=2.0, shots=shots, base_seed=seed,
        name="mid-measurement", events=(storm,),
    )
    observed = study.continue_observed()
    resampled = study.continue_resampled()
    observed_run = run_batch(observed)[0]
    resampled_run = run_batch(resampled)[0]
    print("observed conditions :",
          {key: round(value, 3) for key, value in observed_run.noisy_probabilities.items()})
    print("fresh environment   :",
          {key: round(value, 3) for key, value in resampled_run.noisy_probabilities.items()})
    print("hand values: observed -> same draw twice, always 00;")
    print("             resampled -> independent draws, P(00)=P(11)=1/2")
    save_batch_archive(Path("environment_continue_observed.json"), observed, [observed_run])
    save_batch_archive(Path("environment_continue_resampled.json"), resampled, [resampled_run])


if __name__ == "__main__":
    main()
