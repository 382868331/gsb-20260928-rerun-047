import tempfile
import unittest
from pathlib import Path

from quantum_noise.experiments import ExperimentBatch, rerun_archive, run_batch, save_batch_archive
from quantum_noise.gates import Circuit, bell_circuit
from quantum_noise.noise import NoiseModel, NoiseRule
from quantum_noise.simulator import simulate, validate_density_matrix


class SimulatorTests(unittest.TestCase):
    def test_bell_state(self):
        result = simulate(bell_circuit())
        self.assertAlmostEqual(result.probabilities["00"], 0.5)
        self.assertAlmostEqual(result.probabilities["11"], 0.5)
        self.assertAlmostEqual(result.probabilities["01"], 0.0)
        validate_density_matrix(result.state)

    def test_noise_position_changes_bell_calculation(self):
        early = simulate(
            bell_circuit(),
            NoiseModel("early", [NoiseRule("bit_flip", 0.1, (0,), after_gate=0)]),
        )
        late = simulate(
            bell_circuit(),
            NoiseModel("late", [NoiseRule("bit_flip", 0.1, (0,), after_gate=1)]),
        )
        validate_density_matrix(early.state)
        validate_density_matrix(late.state)
        self.assertNotAlmostEqual(early.probabilities["00"], late.probabilities["00"])
        self.assertAlmostEqual(sum(early.probabilities.values()), 1.0)

    def test_channel_order_changes_single_qubit_calculation(self):
        circuit = Circuit(1).h(0).h(0).h(0)
        ad_first = simulate(
            circuit,
            NoiseModel(
                "ad-bf",
                [
                    NoiseRule("amplitude_damping", 0.2, (0,), after_gate=0),
                    NoiseRule("bit_flip", 0.2, (0,), after_gate=0),
                ],
            ),
        )
        bf_first = simulate(
            circuit,
            NoiseModel(
                "bf-ad",
                [
                    NoiseRule("bit_flip", 0.2, (0,), after_gate=0),
                    NoiseRule("amplitude_damping", 0.2, (0,), after_gate=0),
                ],
            ),
        )
        self.assertNotAlmostEqual(ad_first.probabilities["0"], bf_first.probabilities["0"])


class ExperimentTests(unittest.TestCase):
    def make_batch(self, shots=1000):
        return ExperimentBatch(
            circuit=bell_circuit(),
            noise_model=NoiseModel("late", [NoiseRule("bit_flip", 0.1, (0,), after_gate=1)]),
            shots=shots,
            base_seed=12345,
            name="bell",
        )

    def test_seeded_batch_is_reproducible(self):
        batch = self.make_batch()
        first = run_batch(batch, repeats=3)
        second = run_batch(batch, repeats=3)
        self.assertEqual([run.seed for run in first], [run.seed for run in second])
        self.assertEqual([run.noisy_counts for run in first], [run.noisy_counts for run in second])

    def test_archive_round_trip_reproduces_counts(self):
        batch = self.make_batch(shots=256)
        runs = run_batch(batch, repeats=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            save_batch_archive(path, batch, runs)
            reproduced = rerun_archive(path)
        self.assertEqual([run.noisy_counts for run in runs], [run.noisy_counts for run in reproduced])

    def test_low_shots_are_not_called_significant(self):
        run = run_batch(self.make_batch(shots=10))[0]
        self.assertTrue(all(not test.significant for test in run.difference_tests.values()))
        self.assertIn("inconclusive", run.difference_tests["00"].interpretation)


if __name__ == "__main__":
    unittest.main()
