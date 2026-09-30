import tempfile
import unittest
from math import pi
from pathlib import Path

from quantum_noise.experiments import (
    ExperimentBatch,
    ResumableBatchRun,
    load_checkpoint,
    rerun_archive,
    run_batch,
    save_batch_archive,
    save_checkpoint,
)
from quantum_noise.gates import Circuit, X
from quantum_noise.noise import CorrelatedNoiseGroup, NoiseModel
from quantum_noise.schedule import Schedule
from quantum_noise.simulator import simulate, validate_density_matrix


def short_circuit():
    return Circuit(1).rx(0, pi / 2).rx(0, pi / 2)


def short_model(time_correlation=0.5):
    group = CorrelatedNoiseGroup(
        channel="bit_flip",
        sites=((0, 0), (1, 0)),
        probability=0.2,
        time_correlation=time_correlation,
        name="time-correlated",
    )
    return NoiseModel("timing", (), (group,))


class TimeScheduleTests(unittest.TestCase):
    def test_site_times_follow_durations_and_idle_waits(self):
        schedule = Schedule((2.0, 3.0), (4.0, 0.0)).normalized(2)
        self.assertEqual(schedule.time_before(0), 0.0)
        self.assertEqual(schedule.site_time(0), 2.0)
        self.assertEqual(schedule.time_before(1), 6.0)
        self.assertEqual(schedule.site_time(1), 9.0)
        self.assertEqual(schedule.total_duration, 9.0)

    def test_same_operations_different_schedules_differ(self):
        circuit = short_circuit()
        model = short_model()
        tight = simulate(circuit, model, Schedule((1, 1), (0, 0), name="tight"))
        spaced = simulate(circuit, model, Schedule((1, 1), (9, 0), name="spaced"))
        self.assertNotAlmostEqual(
            tight.probabilities["0"], spaced.probabilities["0"]
        )
        validate_density_matrix(tight.state)
        validate_density_matrix(spaced.state)

    def test_extreme_correlations_match_direct_short_circuit(self):
        circuit = short_circuit()
        # P(0) = 2 p (1 - p) (1 - corr) for the rx-rx short circuit.
        for correlation, expected in ((0.0, 2 * 0.2 * 0.8), (1.0, 0.0)):
            group = CorrelatedNoiseGroup(
                channel="bit_flip",
                sites=((0, 0), (1, 0)),
                probability=0.2,
                time_correlation=correlation,
                times=(0.0, 1.0),
            )
            result = simulate(circuit, NoiseModel("extreme", (), (group,)))
            self.assertAlmostEqual(result.probabilities["0"], expected, places=12)

    def test_explicit_times_match_schedule_based_times(self):
        circuit = short_circuit()
        group_times = CorrelatedNoiseGroup(
            channel="bit_flip",
            sites=((0, 0), (1, 0)),
            probability=0.2,
            time_correlation=0.5,
            times=(2.0, 11.0),
        )
        group_schedule = CorrelatedNoiseGroup(
            channel="bit_flip",
            sites=((0, 0), (1, 0)),
            probability=0.2,
            time_correlation=0.5,
        )
        schedule = Schedule((2.0, 9.0), (0.0, 0.0))
        direct = simulate(circuit, NoiseModel("times", (), (group_times,)))
        scheduled = simulate(
            circuit, NoiseModel("schedule", (), (group_schedule,)), schedule
        )
        for outcome in ("0", "1"):
            self.assertAlmostEqual(
                direct.probabilities[outcome],
                scheduled.probabilities[outcome],
                places=12,
            )


class BranchIndependenceTests(unittest.TestCase):
    def branch_circuit(self):
        circuit = Circuit(1).h(0).measure(0, 0)
        circuit.apply("x", X, 0, condition=(0, 0))
        return circuit

    def test_environment_continues_independently_per_branch(self):
        circuit = self.branch_circuit()
        # Two sites: right after the measurement and right after the final
        # (possibly conditioned) operation.  With fully correlated errors each
        # branch flips either twice or never, so both branches end in |1>:
        # branch 0 is flipped back by the conditioned X before the two flips,
        # branch 1 simply stays |1>.
        group = CorrelatedNoiseGroup(
            channel="bit_flip",
            sites=((1, 0), (2, 0)),
            probability=0.25,
            correlation=1.0,
        )
        model = NoiseModel("branched", (), (group,))
        schedule = Schedule((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        result = simulate(circuit, model, schedule)
        self.assertAlmostEqual(result.probabilities["1"], 1.0)
        registers: dict[str, float] = {}
        for trajectory in result.noise_trajectories:
            registers[trajectory.register] = (
                registers.get(trajectory.register, 0.0) + trajectory.probability
            )
        self.assertAlmostEqual(registers["0"], 0.5)
        self.assertAlmostEqual(registers["1"], 0.5)

    def test_equal_operation_counts_do_not_share_disturbance(self):
        # c = 0 on a branch crossing the measurement: marginal P(0) must match
        # independent errors on both branches (checked against the rule-based
        # independent channel on an equivalent branch circuit).
        circuit = self.branch_circuit()
        group = CorrelatedNoiseGroup(
            channel="bit_flip",
            sites=((1, 0), (2, 0)),
            probability=0.2,
            correlation=0.0,
        )
        model = NoiseModel("branched-independent", (), (group,))
        schedule = Schedule((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        result = simulate(circuit, model, schedule)
        validate_density_matrix(result.state)
        # Both branches end in |1> even with any independent bit flips only
        # when flips coincide; a single flip on a branch leaves |0>.
        self.assertTrue(0.0 < result.probabilities["0"] < 1.0)
        # Exact value: P(0) = single-flip probability on either branch
        # = 2 p (1 - p) = 0.32.
        self.assertAlmostEqual(result.probabilities["0"], 0.32, places=9)


class ScheduledBatchTests(unittest.TestCase):
    def make_batch(self, shots=500, idle=0.0, name="tight", seed=77):
        return ExperimentBatch(
            circuit=short_circuit(),
            noise_model=short_model(),
            shots=shots,
            base_seed=seed,
            name=name,
            schedule=Schedule((1, 1), (idle, 0), name=name),
        )

    def test_environment_conditions_and_statistics_are_separately_saved(self):
        run = run_batch(self.make_batch())[0]
        self.assertIsNotNone(run.schedule_name)
        self.assertTrue(run.noise_trajectories)
        self.assertTrue(run.noise_paths)
        self.assertEqual(
            sum(run.noise_paths.values()),
            run.shots,
        )
        labels = {entry["label"] for entry in run.noise_trajectories}
        self.assertTrue(set(run.noise_paths) <= labels)

    def test_different_schedules_produce_different_batch_results(self):
        tight = run_batch(self.make_batch(shots=4000, idle=0.0, name="tight"))[0]
        spaced = run_batch(self.make_batch(shots=4000, idle=8.0, name="spaced"))[0]
        self.assertNotAlmostEqual(
            tight.noisy_probabilities["0"], spaced.noisy_probabilities["0"]
        )

    def test_pause_resume_replays_same_batch(self):
        batch = self.make_batch(shots=300)
        full = run_batch(batch)[0]
        runner = ResumableBatchRun(batch)
        runner.advance(73)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_checkpoint(path, runner.checkpoint())
            restored = ResumableBatchRun.restore(batch, load_checkpoint(path))
        self.assertEqual(restored.completed, 73)
        restored.advance(227)
        finished = restored.finish()
        self.assertEqual(finished.noisy_counts, full.noisy_counts)
        self.assertEqual(finished.noise_paths, full.noise_paths)

    def test_changing_batch_execution_order_leaves_results_unchanged(self):
        first = self.make_batch(shots=260, idle=0.0, name="first", seed=31)
        second = self.make_batch(shots=180, idle=6.0, name="second", seed=32)
        straight_first = run_batch(first)[0].noisy_counts
        straight_second = run_batch(second)[0].noisy_counts

        runner_first = ResumableBatchRun(first)
        runner_second = ResumableBatchRun(second)
        runner_first.advance(100)
        runner_second.advance(90)
        runner_second.advance(90)
        runner_first.advance(160)
        self.assertEqual(runner_first.finish().noisy_counts, straight_first)
        self.assertEqual(runner_second.finish().noisy_counts, straight_second)

    def test_scheduled_archive_round_trip(self):
        batch = self.make_batch(shots=128, name="archived")
        runs = run_batch(batch, repeats=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scheduled.json"
            save_batch_archive(path, batch, runs)
            reproduced = rerun_archive(path)
        self.assertEqual(
            [run.noisy_counts for run in runs],
            [run.noisy_counts for run in reproduced],
        )
        self.assertEqual(
            [run.noise_paths for run in runs],
            [run.noise_paths for run in reproduced],
        )


if __name__ == "__main__":
    unittest.main()
