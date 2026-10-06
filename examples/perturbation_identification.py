"""Identify common versus independent perturbations from joint counts.

The workflow: fit candidate event windows against observed counts, compare
the fully common, fully independent, and mixed model classes, then carry a
selected candidate to a new circuit and check it with fresh independent
samples.  Follow-up evidence gathered while continuing observed conditions
is kept separate from resampled evidence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_noise import Circuit, TimeSchedule
from quantum_noise.identification import (
    CandidateWindow,
    PerturbationStudy,
    predict_joint_distribution,
    test_prediction,
)


def main() -> None:
    circuit = Circuit(2).rx(0, 0.0).rx(1, 0.0)
    schedule = TimeSchedule((1.0, 1.0), name="unit-gap")
    windows = (
        CandidateWindow("early-pulse", 1.0, 3.0, 1, (0, 1)),
        CandidateWindow("late-pulse", 3.0, 5.0, 1, (0, 1)),
    )
    observed_counts = {"00": 1600, "11": 400}

    study = PerturbationStudy(
        study_id="pulse-study",
        circuit=circuit,
        schedule=schedule,
        windows=windows,
        counts=observed_counts,
    )
    result = study.identify()
    print("status:", result.status)
    for reason in result.reasons:
        print("  reason:", reason)
    for model_class, fit in result.best_by_class.items():
        print(
            f"  best {model_class}: c={fit.common_probability:.2f} "
            f"q={fit.independent_probability:.2f} ll={fit.log_likelihood:.1f}"
        )

    candidate = result.best_by_class["common"]
    new_circuit = Circuit(2).h(0).h(1)
    new_schedule = TimeSchedule((1.0, 1.0), name="follow-up")
    predicted = predict_joint_distribution(candidate, windows, new_circuit, new_schedule)
    print("predicted joint distribution on the new circuit:", predicted)

    fresh_counts = {outcome: round(p * 4000) for outcome, p in predicted.items()}
    study.add_followup(fresh_counts, "resample", note="fresh samples on new circuit")
    test = test_prediction(predicted, study.pooled_counts("resample"))
    print("follow-up test:", test.interpretation)


if __name__ == "__main__":
    main()

