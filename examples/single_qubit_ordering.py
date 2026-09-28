import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import Circuit, ExperimentBatch, NoiseModel, NoiseRule, run_batch
from quantum_noise.report import format_run_report


def main() -> None:
    circuit = Circuit(1).h(0).h(0).h(0)
    for ordering, rules in (
        (
            "ad-bf",
            [
                NoiseRule("amplitude_damping", 0.15, (0,), after_gate=0),
                NoiseRule("bit_flip", 0.15, (0,), after_gate=0),
            ],
        ),
        (
            "bf-ad",
            [
                NoiseRule("bit_flip", 0.15, (0,), after_gate=0),
                NoiseRule("amplitude_damping", 0.15, (0,), after_gate=0),
            ],
        ),
    ):
        batch = ExperimentBatch(
            circuit=circuit,
            noise_model=NoiseModel(ordering, rules),
            shots=2000,
            base_seed=20260929,
            name=f"single-{ordering}",
        )
        print(format_run_report(run_batch(batch)[0]))


if __name__ == "__main__":
    main()
