import json
import random
import tempfile
import unittest
from itertools import product
from pathlib import Path

from quantum_noise.correlated import (
    CorrelatedNoiseEvent,
    CorrelatedNoiseModel,
    sample_trajectory,
    transition_probability,
)
from quantum_noise.dynamics import exact_probabilities, simulate_shot
from quantum_noise.gates import Circuit
from quantum_noise.experiments import ExperimentBatch, run_batch, save_batch_archive
from quantum_noise.noise import NoiseModel, NoiseRule
from quantum_noise.simulator import validate_density_matrix, zero_state
from quantum_noise.trajectories import (
    TrajectoryBatch,
    archive_kind,
    replay_archive,
    rerun_trajectory_archive,
    run_trajectory_batch,
    save_trajectory_archive,
    sweep_correlation,
)


def two_flip_model(rho: float) -> CorrelatedNoiseModel:
    return CorrelatedNoiseModel(
        f"two-flips-{rho}",
        [
            CorrelatedNoiseEvent("bit_flip", 0, 0, 0.2, correlation=rho),
            CorrelatedNoiseEvent("bit_flip", 1, 0, 0.2, correlation=rho),
        ],
    )


class CorrelatedNoiseTests(unittest.TestCase):
    def test_correlation_changes_exact_probabilities(self):
        circuit = Circuit(1).x(0).x(0)
        independent = exact_probabilities(circuit, two_flip_model(0.0))
        perfect = exact_probabilities(circuit, two_flip_model(1.0))
        # Independent flips: P(odd) = 2 p (1-p) = 0.32.
        self.assertAlmostEqual(independent["1"], 0.32)
        self.assertAlmostEqual(independent["0"], 0.68)
        # Perfectly correlated flips always come in pairs and cancel.
        self.assertAlmostEqual(perfect["1"], 0.0)
        self.assertAlmostEqual(perfect["0"], 1.0)
        self.assertNotAlmostEqual(independent["1"], perfect["1"])

    def test_direct_enumeration_matches_markov_paths(self):
        p, rho = 0.2, 0.6
        events = list(two_flip_model(rho).ordered_events())
        direct = {}
        for e1, e2 in product((0, 1), repeat=2):
            prior = p if e1 else 1.0 - p
            rate = transition_probability(p, rho, e1)
            joint = prior * (rate if e2 else 1.0 - rate)
            odd = (e1 + e2) % 2
            direct[odd] = direct.get(odd, 0.0) + joint
        exact = exact_probabilities(Circuit(1).x(0).x(0), two_flip_model(rho))
        self.assertAlmostEqual(direct[1], exact["1"])
        self.assertAlmostEqual(direct[0], exact["0"])
        self.assertAlmostEqual(direct[0] + direct[1], 1.0)

    def test_markov_samples_are_serially_correlated(self):
        events = list(two_flip_model(0.75).ordered_events())
        draws = 200_000
        counts = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
        rng = random.Random(123)
        for _ in range(draws):
            indicators = sample_trajectory(events, rng)
            pair = tuple(indicators[id(event)] for event in events)
            counts[pair] += 1
        p11 = counts[(1, 1)] / draws
        # Cov(e1, e2) = rho p (1-p).
        self.assertAlmostEqual(p11, 0.2**2 + 0.75 * 0.2 * 0.8, delta=2e-3)

    def test_infeasible_correlation_is_rejected(self):
        with self.assertRaises(ValueError):
            CorrelatedNoiseEvent("bit_flip", 0, 0, 0.2, correlation=-0.99).validate()


class BranchTests(unittest.TestCase):
    def test_mid_measurement_conditional_branch_exact(self):
        # Measure |+> then conditionally flip the second qubit.
        circuit = Circuit(2).h(0).measure(0, 0).cx_conditional(0, 1)
        probabilities = exact_probabilities(circuit)
        validate_density_matrix(zero_state(2))  # module import sanity helper used below
        self.assertAlmostEqual(probabilities["00"], 0.5)
        self.assertAlmostEqual(probabilities["11"], 0.5)
        self.assertAlmostEqual(probabilities["01"], 0.0)
        self.assertAlmostEqual(probabilities["10"], 0.0)

    def test_branched_shots_follow_their_measurement(self):
        circuit = Circuit(2).h(0).measure(0, 0).cx_conditional(0, 1)
        rng = random.Random(9)
        for _ in range(50):
            trace = simulate_shot(circuit, None, random.Random(), rng)
            measured = trace.measurement_outcomes[0]
            self.assertEqual(int(trace.final_outcome[1]), measured)
            self.assertEqual(int(trace.final_outcome[0]), measured)
            self.assertEqual(list(trace.branch_path), [measured])

    def test_conditional_noise_probabilities_are_valid_states(self):
        circuit = Circuit(2).h(0).measure(0, 0).cx_conditional(0, 1)
        model = CorrelatedNoiseModel(
            "branched",
            [
                CorrelatedNoiseEvent("bit_flip", 0, 0, 0.15, correlation=0.3),
                CorrelatedNoiseEvent("bit_flip", 1, 0, 0.15, correlation=0.3),
            ],
        )
        probabilities = exact_probabilities(circuit, model)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        for value in probabilities.values():
            self.assertGreaterEqual(value, 0.0)


class RandomnessSeparationTests(unittest.TestCase):
    def batch(self):
        circuit = Circuit(1).h(0).measure(0, 0).x(0)
        model = CorrelatedNoiseModel(
            "sep",
            [CorrelatedNoiseEvent("bit_flip", 0, 0, 0.4, correlation=0.8)],
        )
        return TrajectoryBatch(circuit, model, shots=40, base_seed=5, name="sep")

    def test_noise_stream_controls_only_trajectory(self):
        circuit = self.batch().circuit
        model = self.batch().noise_model
        noise_rng = random.Random(11)
        measured_rng_a = random.Random(22)
        measured_rng_b = random.Random(77)
        first = [simulate_shot(circuit, model, noise_rng, measured_rng_a) for _ in range(40)]
        noise_rng = random.Random(11)
        measured_rng_c = random.Random(99)
        second = [simulate_shot(circuit, model, noise_rng, measured_rng_c) for _ in range(40)]
        self.assertEqual(
            [trace.noise_errors for trace in first],
            [trace.noise_errors for trace in second],
        )
        noise_rng = random.Random(11)
        again = [simulate_shot(circuit, model, noise_rng, measured_rng_b) for _ in range(40)]
        self.assertNotEqual(
            [trace.branch_path for trace in first],
            [trace.branch_path for trace in again],
        )

    def test_batch_is_seeded_and_reproducible(self):
        batch = self.batch()
        first = run_trajectory_batch(batch, repeats=2)
        second = run_trajectory_batch(batch, repeats=2)
        self.assertEqual([run.noisy_counts for run in first], [run.noisy_counts for run in second])
        self.assertEqual([run.traces for run in first], [run.traces for run in second])

    def test_low_shots_remain_inconclusive(self):
        tiny = TrajectoryBatch(
            Circuit(1).x(0).x(0), two_flip_model(0.8), shots=8, base_seed=1, name="tiny"
        )
        run = run_trajectory_batch(tiny)[0]
        self.assertTrue(all(not test.significant for test in run.difference_tests.values()))


class SweepAndArchiveTests(unittest.TestCase):
    def test_sweep_keeps_circuit_fixed_and_changes_rho(self):
        batch = TrajectoryBatch(
            Circuit(1).x(0).x(0), two_flip_model(0.0), shots=400, base_seed=4, name="sweep"
        )
        points = sweep_correlation(batch, [0.0, 0.5, 1.0])
        self.assertEqual([point.correlation for point in points], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(points[0].noisy_probabilities["1"], 0.32)
        self.assertAlmostEqual(points[1].noisy_probabilities["1"], 0.16)
        self.assertAlmostEqual(points[2].noisy_probabilities["1"], 0.0)

    def test_correlated_archive_round_trip(self):
        batch = TrajectoryBatch(
            Circuit(1).x(0).x(0), two_flip_model(0.6), shots=128, base_seed=77, name="arch"
        )
        runs = run_trajectory_batch(batch, repeats=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corr.json"
            save_trajectory_archive(path, batch, runs)
            self.assertEqual(archive_kind(path), "correlated")
            reproduced = rerun_trajectory_archive(path)
            generic = replay_archive(path)
        self.assertEqual([run.noisy_counts for run in runs], [run.noisy_counts for run in reproduced])
        self.assertEqual([run.noise_seed for run in runs], [run.noise_seed for run in generic])

    def test_legacy_independent_archive_still_reads(self):
        legacy_batch = ExperimentBatch(
            circuit=Circuit(2).h(0).cnot(0, 1),
            noise_model=NoiseModel("old", [NoiseRule("bit_flip", 0.1, (0,), after_gate=1)]),
            shots=128,
            base_seed=31,
            name="legacy",
        )
        runs = run_batch(legacy_batch, repeats=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            save_batch_archive(path, legacy_batch, runs)
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Downgrade to the original first-turn schema version.
            payload["schema_version"] = 1
            del payload["archive_kind"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(archive_kind(path), "independent")
            replayed = replay_archive(path)
        self.assertEqual(replayed[0].noisy_counts, runs[0].noisy_counts)


if __name__ == "__main__":
    unittest.main()
