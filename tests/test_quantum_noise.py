import tempfile
import unittest
from math import pi
from pathlib import Path

from quantum_noise.experiments import (
    BatchCheckpoint,
    ExperimentBatch,
    ExperimentTask,
    compare_schedules,
    load_batch,
    rerun_archive,
    resume_batch,
    run_batch,
    save_batch_archive,
    save_schedule_comparison,
    start_resumable_batch,
)
from quantum_noise.gates import Circuit, X, Z, bell_circuit
from quantum_noise.noise import CorrelatedNoiseGroup, NoiseModel, NoiseRule, TimeSchedule
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


class CorrelatedNoiseTests(unittest.TestCase):
    def make_group(self, probability=0.2, correlation=0.5):
        return CorrelatedNoiseGroup(
            channel="bit_flip",
            sites=((0, 0), (1, 0)),
            probability=probability,
            correlation=correlation,
        )

    def test_trajectory_distribution_matches_markov_chain(self):
        group = self.make_group(probability=0.3, correlation=0.5)
        distribution = dict(group.trajectories())
        stay = 0.3 + 0.5 * 0.7
        become = 0.3 * 0.5
        self.assertAlmostEqual(distribution[(1, 1)], 0.3 * stay)
        self.assertAlmostEqual(distribution[(1, 0)], 0.3 * (1 - stay))
        self.assertAlmostEqual(distribution[(0, 1)], 0.7 * become)
        self.assertAlmostEqual(distribution[(0, 0)], 0.7 * (1 - become))
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_correlation_bounds_are_enforced(self):
        with self.assertRaises(ValueError):
            self.make_group(probability=0.2, correlation=1.5)
        with self.assertRaises(ValueError):
            self.make_group(probability=0.2, correlation=-0.5)

    def test_exact_probabilities_follow_correlation_strength(self):
        # rx(pi/2) twice with two bit-flip sites: P(0) = 2 p (1-p) (1-c).
        circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
        for correlation, expected in ((0.0, 0.32), (0.5, 0.16), (1.0, 0.0)):
            model = NoiseModel(
                "corr", (), (self.make_group(0.2, correlation),)
            )
            result = simulate(circuit, model)
            validate_density_matrix(result.state)
            self.assertAlmostEqual(result.probabilities["0"], expected, places=9)
            self.assertIsNotNone(result.noise_trajectories)
            weights = sum(t.probability for t in result.noise_trajectories)
            self.assertAlmostEqual(weights, 1.0)

    def test_correlated_mixture_matches_direct_manual_sum(self):
        # Direct computation: enumerate the joint patterns by hand and mix
        # the per-pattern unitary evolutions with their Markov weights.
        circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
        p, c = 0.25, 0.6
        model = NoiseModel("corr", (), (self.make_group(p, c),))
        result = simulate(circuit, model)
        stay = p + c * (1 - p)
        become = p * (1 - c)
        weights = {
            (0, 0): (1 - p) * (1 - become),
            (0, 1): (1 - p) * become,
            (1, 0): p * (1 - stay),
            (1, 1): p * stay,
        }
        expected_p0 = 0.0
        for (e1, e2), weight in weights.items():
            pattern_circuit = Circuit(1).rx(0, pi / 2)
            if e1:
                pattern_circuit.x(0)
            pattern_circuit.rx(0, pi / 2)
            if e2:
                pattern_circuit.x(0)
            expected_p0 += weight * simulate(pattern_circuit).probabilities["0"]
        self.assertAlmostEqual(result.probabilities["0"], expected_p0, places=12)

    def test_zero_correlation_matches_independent_channels(self):
        circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
        correlated = simulate(
            circuit, NoiseModel("corr", (), (self.make_group(0.2, 0.0),))
        )
        independent = simulate(
            circuit,
            NoiseModel(
                "indep",
                [
                    NoiseRule("bit_flip", 0.2, (0,), after_gate=0),
                    NoiseRule("bit_flip", 0.2, (0,), after_gate=1),
                ],
            ),
        )
        for outcome in ("0", "1"):
            self.assertAlmostEqual(
                correlated.probabilities[outcome],
                independent.probabilities[outcome],
                places=12,
            )


class MidCircuitMeasurementTests(unittest.TestCase):
    def test_conditional_gate_direct_computation(self):
        # H, measure, X conditioned on outcome 0: every branch ends in |1>.
        circuit = Circuit(1).h(0).measure(0, 0)
        circuit.apply("x", X, 0, condition=(0, 0))
        result = simulate(circuit)
        validate_density_matrix(result.state)
        self.assertAlmostEqual(result.probabilities["1"], 1.0)
        self.assertAlmostEqual(result.classical_probabilities["0"], 0.5)
        self.assertAlmostEqual(result.classical_probabilities["1"], 0.5)

    def test_conditional_gate_on_other_outcome(self):
        # Condition on outcome 1 instead: branches end in |0> or stay |1>.
        circuit = Circuit(1).h(0).measure(0, 0)
        circuit.apply("x", X, 0, condition=(0, 1))
        result = simulate(circuit)
        validate_density_matrix(result.state)
        self.assertAlmostEqual(result.probabilities["0"], 1.0)

    def test_unconditioned_circuit_is_unchanged(self):
        circuit = Circuit(1).h(0).measure(0, 0)
        result = simulate(circuit)
        self.assertAlmostEqual(result.probabilities["0"], 0.5)
        self.assertAlmostEqual(result.probabilities["1"], 0.5)

    def test_conditional_branch_archive_round_trip(self):
        circuit = Circuit(1).h(0).measure(0, 0)
        circuit.apply("x", X, 0, condition=(0, 0))
        batch = ExperimentBatch(
            circuit=circuit,
            noise_model=NoiseModel("none"),
            shots=128,
            base_seed=99,
            name="conditional",
        )
        runs = run_batch(batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            save_batch_archive(path, batch, runs)
            reproduced = rerun_archive(path)
        self.assertEqual(runs[0].noisy_counts, reproduced[0].noisy_counts)


class CorrelatedBatchTests(unittest.TestCase):
    def make_batch(self, shots=500, correlation=0.5):
        circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
        group = CorrelatedNoiseGroup(
            channel="bit_flip",
            sites=((0, 0), (1, 0)),
            probability=0.2,
            correlation=correlation,
        )
        return ExperimentBatch(
            circuit=circuit,
            noise_model=NoiseModel("corr", (), (group,)),
            shots=shots,
            base_seed=777,
            name="corr",
        )

    def test_noise_and_measure_seeds_are_separate_and_reproducible(self):
        batch = self.make_batch()
        first = run_batch(batch, repeats=2)
        second = run_batch(batch, repeats=2)
        for run in first:
            self.assertIsNotNone(run.noise_seed)
            self.assertIsNotNone(run.measure_seed)
            self.assertNotEqual(run.noise_seed, run.measure_seed)
            self.assertTrue(run.noise_trajectories)
        self.assertEqual(
            [(r.noise_seed, r.measure_seed, r.noisy_counts) for r in first],
            [(r.noise_seed, r.measure_seed, r.noisy_counts) for r in second],
        )

    def test_correlated_archive_round_trip(self):
        batch = self.make_batch(correlation=0.8)
        runs = run_batch(batch, repeats=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corr.json"
            save_batch_archive(path, batch, runs)
            reproduced = rerun_archive(path)
        self.assertEqual(
            [run.noisy_counts for run in runs],
            [run.noisy_counts for run in reproduced],
        )

    def test_low_shots_stay_inconclusive(self):
        run = run_batch(self.make_batch(shots=10))[0]
        self.assertTrue(all(not t.significant for t in run.difference_tests.values()))
        self.assertIn("inconclusive", run.difference_tests["0"].interpretation)

    def test_schema1_archive_still_loads(self):
        batch = ExperimentBatch(
            circuit=bell_circuit(),
            noise_model=NoiseModel("late", [NoiseRule("bit_flip", 0.1, (0,), after_gate=1)]),
            shots=64,
            base_seed=5,
            name="bell",
        )
        runs = run_batch(batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            save_batch_archive(path, batch, runs)
            # Rewrite as a schema-1 archive without the new fields.
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["schema_version"] = 1
            del payload["batch"]["noise_model"]["groups"]
            for saved in payload["runs"]:
                for key in ("noise_seed", "measure_seed", "noise_trajectories"):
                    saved.pop(key, None)
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded_batch, saved_runs = load_batch(path)
            self.assertEqual(loaded_batch.noise_model.groups, ())
            self.assertEqual(len(saved_runs), 1)
            reproduced = rerun_archive(path)
            self.assertEqual(runs[0].noisy_counts, reproduced[0].noisy_counts)


class TimeScheduleTests(unittest.TestCase):
    def test_same_operations_different_timings_change_correlation(self):
        circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
        group = CorrelatedNoiseGroup(
            "bit_flip",
            ((0, 0), (1, 0)),
            probability=0.2,
            correlation=1.0,
            correlation_time=1.0,
        )
        model = NoiseModel("time", (), (group,))
        tight = TimeSchedule((0.0, 0.0), (0.0, 0.0), "tight")
        relaxed = TimeSchedule((1.0, 20.0), (0.0, 0.0), "relaxed")
        tight_result = simulate(circuit, model, tight)
        relaxed_result = simulate(circuit, model, relaxed)
        self.assertAlmostEqual(tight_result.probabilities["0"], 0.0)
        self.assertAlmostEqual(relaxed_result.probabilities["0"], 0.32, places=6)
        effective = tight_result.environment_records[0]["events"][group.name][1]["correlation"]
        self.assertAlmostEqual(effective, 1.0)
        relaxed_effective = relaxed_result.environment_records[0]["events"][group.name][1]["correlation"]
        self.assertAlmostEqual(relaxed_effective, 0.0, places=5)

    def test_short_circuit_direct_extreme_correlation_and_independence(self):
        circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
        p = 0.25
        group = CorrelatedNoiseGroup(
            "bit_flip",
            ((0, 0), (1, 0)),
            probability=p,
            correlation=1.0,
            correlation_time=1.0,
        )
        model = NoiseModel("extremes", (), (group,))
        fully_correlated = simulate(circuit, model, TimeSchedule((0.0, 0.0), name="tight"))
        independent = simulate(circuit, model, TimeSchedule((1.0, 100.0), name="long-gap"))
        direct_correlated_p0 = 0.0
        direct_independent_p0 = 2.0 * p * (1.0 - p)
        self.assertAlmostEqual(fully_correlated.probabilities["0"], direct_correlated_p0)
        self.assertAlmostEqual(independent.probabilities["0"], direct_independent_p0, places=5)

    def test_schedule_comparison_is_saved_with_separate_records_and_statistics(self):
        circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
        group = CorrelatedNoiseGroup(
            "bit_flip",
            ((0, 0), (1, 0)),
            probability=0.2,
            correlation=1.0,
            correlation_time=1.0,
        )
        model = NoiseModel("time", (), (group,))
        schedules = {
            "tight": TimeSchedule((1.0, 1.0), name="tight"),
            "long-idle": TimeSchedule((1.0, 1.0), (0.0, 20.0), "long-idle"),
        }
        results = compare_schedules(circuit, model, schedules, shots=128, base_seed=31)
        batches = {name: run[0].schedule and ExperimentBatch(
            circuit=circuit,
            noise_model=model,
            shots=128,
            base_seed=31,
            name=name,
            schedule=schedules[name],
        ) for name, run in results.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            save_schedule_comparison(path, batches, results)
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload["schedules"]), {"tight", "long-idle"})
        for name in schedules:
            entry = payload["schedules"][name]
            self.assertTrue(entry["environment_changes"][0])
            self.assertTrue(entry["measurement_conditions"] or entry["statistics"])
            self.assertTrue(entry["statistics"][0]["noisy_counts"])
        self.assertNotEqual(
            results["tight"][0].noisy_probabilities,
            results["long-idle"][0].noisy_probabilities,
        )


class BranchEnvironmentTests(unittest.TestCase):
    def test_measurement_branches_continue_environments_independently(self):
        # Each post-measurement branch experiences its own later site, but
        # both have the same number of environment events.
        circuit = Circuit(1).h(0).measure(0, 0)
        circuit.apply("x", X, 0, condition=(0, 0))
        circuit.apply("z", Z, 0, condition=(0, 1))
        group = CorrelatedNoiseGroup(
            "bit_flip",
            ((0, 0), (2, 0), (3, 0)),
            probability=0.5,
            correlation=1.0,
            name="branch-env",
        )
        result = simulate(circuit, NoiseModel("branches", (), (group,)))
        branches = {record["branch"]: record for record in result.environment_records}
        self.assertEqual(set(branches), {"0", "1"})
        self.assertEqual(len(branches["0"]["events"]["branch-env"]), 2)
        self.assertEqual(len(branches["1"]["events"]["branch-env"]), 2)
        self.assertAlmostEqual(result.probabilities["1"], 0.5)


class CheckpointTests(unittest.TestCase):
    def test_batch_resumes_and_replays_same_batch(self):
        circuit = Circuit(1).h(0).h(0)
        batch = ExperimentBatch(
            circuit,
            NoiseModel("simple", [NoiseRule("bit_flip", 0.1, (0,), after_gate=1)]),
            shots=64,
            base_seed=88,
            name="checkpointed",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch-checkpoint.json"
            paused = start_resumable_batch(path, batch, repeats=3, pause_after_idle=True)
            self.assertEqual(len(paused.runs), 1)
            resumed = resume_batch(path)
            self.assertEqual(len(resumed.runs), 3)
            expected = run_batch(batch, repeats=3)
            self.assertEqual([r.noisy_counts for r in resumed.runs],
                             [r.noisy_counts for r in expected])

    def test_checkpoint_execution_order_does_not_change_results(self):
        def make_batch(name: str, noise: float) -> ExperimentBatch:
            return ExperimentBatch(
                Circuit(1).h(0).h(0),
                NoiseModel(name, [NoiseRule("bit_flip", noise, (0,), after_gate=1)]),
                shots=96,
                base_seed=44,
                name=name,
            )

        tasks = [
            ExperimentTask(make_batch("a", 0.1), 2, "a"),
            ExperimentTask(make_batch("b", 0.2), 1, "b"),
        ]
        direct = {task.task_key: run_batch(task.batch, task.repeats) for task in tasks}
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            first = BatchCheckpoint(first_path).resume(tasks, order=["a", "b"], pause_after="a")
            first = BatchCheckpoint.load(first_path).resume(tasks, order=["a", "b"])
            second_path = Path(directory) / "second.json"
            second = BatchCheckpoint(second_path).resume(tasks, order=["b", "a"])
        for key in ("a", "b"):
            self.assertEqual(
                [r.noisy_counts for r in first[key]],
                [r.noisy_counts for r in direct[key]],
            )
            self.assertEqual(
                [r.noisy_counts for r in second[key]],
                [r.noisy_counts for r in direct[key]],
            )


if __name__ == "__main__":
    unittest.main()
