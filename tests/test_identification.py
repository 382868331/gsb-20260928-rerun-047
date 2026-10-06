"""Tests for common- versus independent-perturbation identification."""

import unittest

from quantum_noise import Circuit, TimeSchedule
from quantum_noise.identification import (
    CandidateWindow,
    PerturbationStudy,
    fit_event_models,
    predict_joint_distribution,
    test_prediction,
)


def two_qubit_circuit() -> Circuit:
    return Circuit(2).rx(0, 0.0).rx(1, 0.0)


def schedule() -> TimeSchedule:
    return TimeSchedule((1.0, 1.0), name="unit-gap")


def window(name: str = "pulse") -> CandidateWindow:
    return CandidateWindow(name, 1.0, 3.0, 1, (0, 1))


class FitEventModelsTests(unittest.TestCase):
    def test_identifies_fully_common_model(self):
        counts = {"00": 1600, "11": 400}
        result = fit_event_models(two_qubit_circuit(), schedule(), counts, [window()])
        self.assertEqual(result.status, "identified")
        self.assertTrue(result.enough_samples)
        best = result.best_by_class["common"]
        self.assertAlmostEqual(best.common_probability, 0.2, places=6)
        self.assertEqual(best.independent_probability, 0.0)
        self.assertEqual({f.model_class for f in result.plausible}, {"common"})

    def test_identifies_fully_independent_model(self):
        counts = {"00": 8100, "01": 900, "10": 900, "11": 100}
        result = fit_event_models(two_qubit_circuit(), schedule(), counts, [window()])
        self.assertEqual(result.status, "identified")
        best = result.best_by_class["independent"]
        self.assertAlmostEqual(best.independent_probability, 0.1, places=6)
        self.assertEqual(best.common_probability, 0.0)
        self.assertEqual({f.model_class for f in result.plausible}, {"independent"})

    def test_all_zero_counts_keep_common_and_independent(self):
        result = fit_event_models(
            two_qubit_circuit(), schedule(), {"00": 400}, [window()]
        )
        self.assertEqual(result.status, "inconclusive")
        classes = {f.model_class for f in result.plausible}
        self.assertIn("common", classes)
        self.assertIn("independent", classes)
        self.assertTrue(any("not separated" in r for r in result.reasons))

    def test_indistinguishable_windows_are_all_kept(self):
        windows = [window("pulse-a"), window("pulse-b")]
        result = fit_event_models(
            two_qubit_circuit(), schedule(), {"00": 1600, "11": 400}, windows
        )
        self.assertEqual(result.status, "inconclusive")
        self.assertTrue(any("not identifiable" in r for r in result.reasons))
        self.assertEqual({f.window for f in result.plausible}, {"pulse-a", "pulse-b"})

    def test_small_sample_is_inconclusive(self):
        result = fit_event_models(
            two_qubit_circuit(), schedule(), {"00": 9, "11": 1}, [window()]
        )
        self.assertEqual(result.status, "inconclusive")
        self.assertFalse(result.enough_samples)
        self.assertTrue(any("sample too small" in r for r in result.reasons))

    def test_equal_marginals_do_not_imply_equivalence(self):
        result = fit_event_models(
            two_qubit_circuit(), schedule(), {"00": 90, "11": 10}, [window()]
        )
        self.assertIsNotNone(result.marginals_warning)
        self.assertIn("marginals", result.marginals_warning)
        common = result.best_by_class["common"].distribution
        independent = result.best_by_class["independent"].distribution
        joint_gap = 0.5 * sum(
            abs(common.get(k, 0.0) - independent.get(k, 0.0))
            for k in set(common) | set(independent)
        )
        self.assertGreater(joint_gap, 0.1)

    def test_mixed_data_keeps_mixed_explanation(self):
        counts = {"00": 6480, "01": 720, "10": 720, "11": 2080}
        result = fit_event_models(two_qubit_circuit(), schedule(), counts, [window()])
        best = result.best_by_class["mixed"]
        self.assertAlmostEqual(best.common_probability, 0.2, places=6)
        self.assertAlmostEqual(best.independent_probability, 0.1, places=6)
        self.assertIn("mixed", {f.model_class for f in result.plausible})

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fit_event_models(
                Circuit(4).rx(0, 0.0), TimeSchedule((1.0,)), {"0000": 10}, [window()]
            )
        with self.assertRaises(ValueError):
            fit_event_models(
                two_qubit_circuit(), schedule(), {"00": 5, "000": 5}, [window()]
            )
        with self.assertRaises(ValueError):
            fit_event_models(two_qubit_circuit(), schedule(), {}, [window()])
        with self.assertRaises(ValueError):
            fit_event_models(
                two_qubit_circuit(),
                schedule(),
                {"00": 10},
                [window("dup"), window("dup")],
            )
        with self.assertRaises(ValueError):
            fit_event_models(two_qubit_circuit(), schedule(), {"00": 10}, [])


class PredictionTests(unittest.TestCase):
    def test_prediction_on_new_circuit_matches_fresh_counts(self):
        windows = [window()]
        result = fit_event_models(
            two_qubit_circuit(), schedule(), {"00": 1600, "11": 400}, windows
        )
        best = result.best_by_class["common"]
        new_circuit = Circuit(2).h(0).h(1)
        new_schedule = TimeSchedule((1.0, 1.0), name="new")
        predicted = predict_joint_distribution(best, windows, new_circuit, new_schedule)
        self.assertAlmostEqual(sum(predicted.values()), 1.0, places=9)
        counts = {outcome: round(p * 4000) for outcome, p in predicted.items()}
        test = test_prediction(predicted, counts)
        self.assertTrue(test.enough_samples)
        self.assertTrue(test.consistent, test.interpretation)

    def test_fresh_counts_reject_wrong_model(self):
        windows = [window()]
        result = fit_event_models(
            two_qubit_circuit(), schedule(), {"00": 1600, "11": 400}, windows
        )
        common_best = result.best_by_class["common"]
        predicted = predict_joint_distribution(
            common_best, windows, two_qubit_circuit(), schedule()
        )
        independent_counts = {"00": 25600, "01": 6400, "10": 6400, "11": 1600}
        test = test_prediction(predicted, independent_counts)
        self.assertFalse(test.consistent)
        self.assertIn("reject", test.interpretation)

    def test_prediction_with_too_few_samples_is_inconclusive(self):
        predicted = {"00": 0.8, "11": 0.2}
        test = test_prediction(predicted, {"00": 3, "11": 1})
        self.assertFalse(test.enough_samples)
        self.assertIn("inconclusive", test.interpretation)

    def test_prediction_rejects_unknown_outcome(self):
        with self.assertRaises(ValueError):
            test_prediction({"00": 1.0}, {"00": 5, "01": 1})


class PerturbationStudyTests(unittest.TestCase):
    def make_study(self) -> PerturbationStudy:
        return PerturbationStudy(
            study_id="study-1",
            circuit=two_qubit_circuit(),
            schedule=schedule(),
            windows=(window(),),
            counts={"00": 1600, "11": 400},
        )

    def test_identify_uses_only_original_counts(self):
        study = self.make_study()
        before = study.identify()
        study.add_followup({"00": 50, "11": 50}, "observed")
        after = study.identify()
        self.assertEqual(before.status, after.status)
        self.assertEqual(before.shots, after.shots)
        self.assertEqual(before.shots, 2000)

    def test_followup_modes_are_validated(self):
        study = self.make_study()
        with self.assertRaises(ValueError):
            study.add_followup({"00": 10}, "pooled")

    def test_pooling_never_mixes_observed_and_resample(self):
        study = self.make_study()
        study.add_followup({"00": 30, "11": 10}, "observed")
        study.add_followup({"00": 90, "11": 10}, "resample")
        observed = study.pooled_counts("observed")
        resample = study.pooled_counts("resample")
        self.assertEqual(observed, {"00": 30, "11": 10})
        self.assertEqual(resample, {"00": 90, "11": 10})
        with self.assertRaises(ValueError):
            study.pooled_counts("everything")

    def test_evidence_records_keep_their_mode(self):
        study = self.make_study()
        study.add_followup({"00": 30, "11": 10}, "observed", note="continued")
        study.add_followup({"00": 90, "11": 10}, "resample")
        modes = [record.mode for record in study.evidence()]
        self.assertEqual(modes, ["observed", "resample"])
        self.assertEqual(len(study.evidence("observed")), 1)
        self.assertEqual(study.evidence("observed")[0].note, "continued")

    def test_study_requires_valid_counts(self):
        with self.assertRaises(ValueError):
            PerturbationStudy(
                study_id="bad",
                circuit=two_qubit_circuit(),
                schedule=schedule(),
                windows=(window(),),
                counts={"000": 5},
            )


if __name__ == "__main__":
    unittest.main()

