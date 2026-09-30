"""Compare one circuit under several time-correlation strengths.

The same rx(pi/2)-rx(pi/2) circuit suffers two possibly correlated bit
flips; the exact P(0) = 2 p (1-p) (1-c) depends on the correlation
strength c.  For each strength the example prints exact probabilities and
the finite-sampling verdict against the independent (c = 0) baseline; with
too few shots the comparison stays inconclusive instead of over-claiming.
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
    compare_counts,
    run_batch,
)
from quantum_noise.simulator import simulate


def correlated_model(probability: float, correlation: float) -> NoiseModel:
    group = CorrelatedNoiseGroup(
        channel="bit_flip",
        sites=((0, 0), (1, 0)),
        probability=probability,
        correlation=correlation,
        name="time-correlated-bit-flip",
    )
    return NoiseModel(f"corr-p{probability:g}-c{correlation:g}", (), (group,))


def main() -> None:
    circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
    probability = 0.2
    strengths = [0.0, 0.5, 1.0]
    shots = 2000

    print(f"circuit: rx(pi/2) rx(pi/2) with two bit-flip sites, p={probability:g}, shots={shots}")
    runs = {}
    for correlation in strengths:
        model = correlated_model(probability, correlation)
        exact = simulate(circuit, model)
        batch = ExperimentBatch(circuit, model, shots=shots, base_seed=20260930,
                                name=f"corr-c{correlation:g}")
        run = run_batch(batch)[0]
        runs[correlation] = (exact, run)
        print(
            f"c={correlation:4.1f}  exact P(0)={exact.probabilities['0']:.6f}  "
            f"sampled counts={run.noisy_counts}"
        )

    baseline_exact, baseline_run = runs[0.0]
    print()
    print("finite-sample comparison against the independent baseline (c=0):")
    for correlation in strengths[1:]:
        exact, run = runs[correlation]
        tests = compare_counts(
            baseline_run.noisy_counts,
            run.noisy_counts,
            baseline_exact.probabilities,
            exact.probabilities,
        )
        for outcome, test in sorted(tests.items()):
            print(f"  c={correlation:4.1f} outcome {outcome}: {test.interpretation}")

    print()
    print("same comparison with only 20 shots (uncertainty is preserved):")
    weak = correlated_model(probability, 1.0)
    small_base = run_batch(
        ExperimentBatch(circuit, correlated_model(probability, 0.0), 20, 7, "small-0")
    )[0]
    small_corr = run_batch(ExperimentBatch(circuit, weak, 20, 7, "small-1"))[0]
    tests = compare_counts(
        small_base.noisy_counts,
        small_corr.noisy_counts,
        correlated_model(probability, 0.0) and runs[0.0][0].probabilities,
        runs[1.0][0].probabilities,
    )
    for outcome, test in sorted(tests.items()):
        print(f"  outcome {outcome}: {test.interpretation}")


if __name__ == "__main__":
    main()
