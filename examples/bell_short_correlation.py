import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import (
    ExperimentBatch,
    NoiseModel,
    NoiseRule,
    bell_circuit,
    run_batch,
    save_batch_archive,
)
from quantum_noise.report import format_run_report


def main() -> None:
    for position, gate_index in (("early", 0), ("late", 1)):
        batch = ExperimentBatch(
            circuit=bell_circuit(),
            noise_model=NoiseModel(
                f"bell-{position}",
                [NoiseRule("bit_flip", 0.10, (0,), after_gate=gate_index)],
            ),
            shots=2000,
            base_seed=20260929,
            name=f"bell-{position}",
        )
        run = run_batch(batch)[0]
        print(format_run_report(run))

    low_shots = ExperimentBatch(
        circuit=bell_circuit(),
        noise_model=NoiseModel("tiny", [NoiseRule("bit_flip", 0.10, (0,), after_gate=1)]),
        shots=10,
        base_seed=7,
        name="bell-low-shots",
    )
    print(format_run_report(run_batch(low_shots)[0]))
    save_batch_archive(Path("bell_batch.json"), batch, [run])


if __name__ == "__main__":
    main()
