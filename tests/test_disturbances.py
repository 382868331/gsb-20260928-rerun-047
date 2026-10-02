import json
import tempfile
import unittest
from math import exp, pi
from math import sqrt
from pathlib import Path

from quantum_noise import (
    AmbientNoise,
    Circuit,
    Disturbance,
    DisturbancePlan,
    ExperimentBatch,
    NoiseModel,
    NoiseRule,
    TimeSchedule,
    disturbance_pause_times,
    resume_batch,
    run_batch,
    save_batch_archive,
    start_resumable_batch,
)
from quantum_noise.experiments import load_batch
from quantum_noise.simulator import simulate, validate_density_matrix


def _short_circuit():
    return Circuit(1).rx(0, pi / 2).rx(0, pi / 2)


class DisturbancePhysicsTests(unittest.TestCase):
    def test_persistent_disturbance_matches_direct_short_circuit(self):
        # Bit-flip rate r integrated over total time T gives the exact
        # bit-flip channel with p = (1 - exp(-2 r T)) / 2.  The gates are
        # instantaneous and the only physical interval sits between them.
        rate, total_time = 0.3, 2.0
        schedule = TimeSchedule((0.0, 0.0), (total_time, 0.0))
        ambient = AmbientNoise(qubits=(0,), bit_flip=rate)
        model = NoiseModel("ambient", (), (), ambient)
        result = simulate(_short_circuit(), model, schedule)
        probability = (1 - exp(-2 * rate * total_time)) / 2
        direct = simulate(
            Circuit(1).rx(0, pi / 2).rx(0, pi / 2),
            NoiseModel("direct", [NoiseRule("bit_flip", probability, (0,), after_gate=0)]),
        )
        validate_density_matrix(result.state)
        for outcome in ("0", "1"):
            self.assertAlmostEqual(
                result.probabilities[outcome], direct.probabilities[outcome], places=12
            )

    def test_instantaneous_boundary_window_changes_nothing(self):
        ambient = AmbientNoise(qubits=(0,), bit_flip=0.3)
        model = NoiseModel("ambient", (), (), ambient)
        schedule = TimeSchedule((1.0, 1.0))
        instantaneous = DisturbancePlan((Disturbance(1.0, 1.0, bit_flip=50.0),))
        disturbed = simulate(_short_circuit(), model, schedule, instantaneous)
        baseline = simulate(_short_circuit(), model, schedule)
        self.assertEqual(disturbed.probabilities, baseline.probabilities)

    def test_segmentation_representation_does_not_change_meaning(self):
        ambient = AmbientNoise(qubits=(0,), bit_flip=0.2)
        model = NoiseModel("ambient", (), (), ambient)
        plan = DisturbancePlan((Disturbance(0.3, 1.7, bit_flip=1.5, name="pulse"),))
        representations = [
            TimeSchedule((0.5, 0.5), (0.5, 0.5)),
            TimeSchedule((0.3, 0.7), (0.5, 0.5)),
            TimeSchedule((1.0, 1.0), (0.0, 0.0)),
            TimeSchedule((0.2, 0.8), (0.1, 0.9)),
        ]
        probabilities = [
            simulate(_short_circuit(), model, schedule, plan).probabilities
            for schedule in representations
        ]
        for other in probabilities[1:]:
            for outcome in ("0", "1"):
                self.assertAlmostEqual(
                    probabilities[0][outcome], other[outcome], places=12
                )

    def test_disturbance_in_middle_of_operation_counts_by_actual_time(self):
        ambient = AmbientNoise(qubits=(0,), bit_flip=0.05)
        model = NoiseModel("ambient", (), (), ambient)
        schedule = TimeSchedule((1.0, 1.0), (0.0, 2.0))
        mid_operation = DisturbancePlan((Disturbance(0.25, 0.75, bit_flip=2.0),))
        mid_idle = DisturbancePlan((Disturbance(2.25, 2.75, bit_flip=2.0),))
        during_gate = simulate(_short_circuit(), model, schedule, mid_operation)
        during_idle = simulate(_short_circuit(), model, schedule, mid_idle)
        self.assertNotEqual(
            during_gate.probabilities["0"], during_idle.probabilities["0"]
        )
        # The pulse is not silently attached to the following operation.
        records = during_gate.environment_records[0]["segments"]
        pulsed = [entry for entry in records if entry["disturbances"]]
        self.assertTrue(any(entry["time_end"] <= 1.0 for entry in pulsed))

    def test_overlap_resolution_and_restore(self):
        ambient = AmbientNoise(qubits=(0,), bit_flip=0.1)
        model = NoiseModel("ambient", (), (), ambient)
        schedule = TimeSchedule((0.5, 0.5), (0.0, 1.0))
        windows = (
            Disturbance(0.0, 1.0, bit_flip=0.4, name="first"),
            Disturbance(0.5, 1.5, bit_flip=1.2, name="second"),
        )
        with self.assertRaises(ValueError):
            simulate(_short_circuit(), model, schedule, DisturbancePlan(windows))
        latest = simulate(
            _short_circuit(), model, schedule, DisturbancePlan(windows, overlap="latest")
        )
        strongest = simulate(
            _short_circuit(), model, schedule, DisturbancePlan(windows, overlap="max")
        )
        validate_density_matrix(latest.state)
        validate_density_matrix(strongest.state)
        records = latest.environment_records[0]["segments"]
        restored = [entry for entry in records if entry["time_start"] >= 1.5]
        self.assertTrue(restored)
        self.assertTrue(all(entry["restored"] for entry in restored))
        self.assertTrue(
            all(entry["rates"]["bit_flip"] == 0.1 for entry in restored)
        )

    def test_multi_qubit_ambient_and_depolarizing_semigroup(self):
        circuit = Circuit(2).h(0).cnot(0, 1)
        ambient = AmbientNoise(
            qubits=(0, 1), depolarizing=0.1, amplitude_damping=0.05
        )
        model = NoiseModel("two-qubit", (), (), ambient)
        plan = DisturbancePlan((Disturbance(0.2, 0.8, depolarizing=1.0),))
        result = simulate(
            circuit, model, TimeSchedule((1.0, 1.0), (0.0, 1.0)), plan
        )
        validate_density_matrix(result.state)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)

    def test_depolarizing_channel_contracts_bloch_vector_exactly(self):
        import numpy as np

        from quantum_noise.environment import ambient_rates, segment_kraus

        rate, duration = 0.1, 2.0
        kraus = segment_kraus(ambient_rates(AmbientNoise(depolarizing=rate)), duration)
        excited = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
        output = sum(
            np.array(operator) @ excited @ np.array(operator).conj().T
            for operator in kraus
        )
        contraction = exp(-rate * duration)
        self.assertAlmostEqual(output[1][1].real, 0.5 + 0.5 * contraction, places=9)
        self.assertAlmostEqual(output[0][0].real, 0.5 - 0.5 * contraction, places=9)

    def test_disturbance_works_with_mid_circuit_measurement_branches(self):
        circuit = Circuit(1).h(0).measure(0, 0)
        circuit.apply("rx(1.5707963267948966)", __import__(
            "quantum_noise.gates", fromlist=["rx"]
        ).rx(pi / 2), 0, condition=(0, 0))
        ambient = AmbientNoise(qubits=(0,), amplitude_damping=0.2)
        model = NoiseModel("ambient", (), (), ambient)
        schedule = TimeSchedule((1.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        plan = DisturbancePlan((Disturbance(1.2, 1.8, amplitude_damping=1.5),))
        result = simulate(circuit, model, schedule, plan)
        validate_density_matrix(result.state)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)


class DisturbanceBatchTests(unittest.TestCase):
    def _batch(self, plan=None, name="prefix"):
        ambient = AmbientNoise(qubits=(0,), bit_flip=0.15)
        model = NoiseModel("ambient", (), (), ambient)
        return ExperimentBatch(
            circuit=_short_circuit(),
            noise_model=model,
            shots=256,
            base_seed=4242,
            name=name,
            schedule=TimeSchedule((1.0, 1.0), (0.0, 2.0)),
            disturbances=plan,
        )

    def test_variants_share_prefix_but_keep_separate_results(self):
        prefix = self._batch()
        early = prefix.fork_disturbance(
            DisturbancePlan((Disturbance(0.2, 0.8, bit_flip=1.0),)), name="early"
        )
        late = prefix.fork_disturbance(
            DisturbancePlan((Disturbance(3.2, 3.8, bit_flip=1.0),)), name="late"
        )
        self.assertEqual(early.prefix, "prefix")
        self.assertEqual(late.prefix, "prefix")
        early_run = run_batch(early)[0]
        late_run = run_batch(late)[0]
        self.assertNotEqual(
            early_run.noisy_probabilities, late_run.noisy_probabilities
        )

    def test_past_disturbance_cannot_be_rewritten(self):
        prefix = self._batch(
            DisturbancePlan((Disturbance(0.2, 0.8, bit_flip=1.0),))
        )
        with self.assertRaises(ValueError):
            prefix.fork_disturbance(
                DisturbancePlan((Disturbance(0.2, 0.8, bit_flip=2.0),)),
                name="changed",
                as_of=1.0,
            )
        revised = prefix.revise_future_disturbances(
            1.0,
            [Disturbance(0.2, 0.8, bit_flip=1.0), Disturbance(3.0, 3.5, bit_flip=2.0)],
        )
        starts = sorted(item.start for item in revised.disturbances.disturbances)
        self.assertEqual(starts, [0.2, 3.0])

    def test_archive_round_trips_disturbance_plan(self):
        plan = DisturbancePlan((Disturbance(0.2, 0.8, bit_flip=1.0),))
        batch = self._batch(plan, name="archived")
        runs = run_batch(batch, repeats=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "disturbance.json"
            save_batch_archive(path, batch, runs)
            loaded, saved = load_batch(path)
            self.assertEqual(loaded.disturbances, batch.disturbances)
            replayed = run_batch(loaded, repeats=2)
            self.assertEqual(
                [run.noisy_counts for run in runs],
                [run.noisy_counts for run in replayed],
            )

    def test_pause_near_disturbance_replays_same_batch(self):
        plan = DisturbancePlan(
            (
                Disturbance(0.5, 1.5, bit_flip=1.0, name="first"),
                Disturbance(1.2, 2.0, bit_flip=2.0, name="second"),
            ),
            overlap="latest",
        )
        self.assertEqual(disturbance_pause_times(plan), [0.5, 1.2, 1.5, 2.0])
        batch = self._batch(plan, name="paused")
        expected = run_batch(batch, repeats=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paused.json"
            paused = start_resumable_batch(path, batch, repeats=3, pause_at=1.2)
            self.assertEqual(len(paused.runs), 1)
            resumed = resume_batch(path)
            self.assertEqual(len(resumed.runs), 3)
            self.assertEqual(
                [run.noisy_counts for run in resumed.runs],
                [run.noisy_counts for run in expected],
            )


if __name__ == "__main__":
    unittest.main()
