"""Human-readable side-by-side summaries."""

from __future__ import annotations

from .experiments import BatchRun
from .trajectories import CorrelationSweepPoint, TrajectoryRun


def format_run_report(run: BatchRun) -> str:
    outcomes = sorted(set(run.ideal_probabilities) | set(run.noisy_probabilities))
    lines = [
        f"batch: {run.batch_name} repeat={run.repeat_index} shots={run.shots}",
        f"base_seed={run.base_seed} sampling_seed={run.seed}",
        "outcome | ideal exact | noisy exact | ideal count | noisy count | 95% test",
    ]
    for outcome in outcomes:
        test = run.difference_tests.get(outcome)
        verdict = test.interpretation if test else "not tested"
        lines.append(
            " | ".join(
                [
                    outcome,
                    f"{run.ideal_probabilities.get(outcome, 0.0):.6f}",
                    f"{run.noisy_probabilities.get(outcome, 0.0):.6f}",
                    str(run.ideal_counts.get(outcome, 0)),
                    str(run.noisy_counts.get(outcome, 0)),
                    verdict,
                ]
            )
        )
    return "\n".join(lines)


def format_batch_report(runs: list[BatchRun]) -> str:
    return "\n\n".join(format_run_report(run) for run in runs)


def format_trajectory_run_report(run: TrajectoryRun) -> str:
    outcomes = sorted(set(run.ideal_probabilities) | set(run.noisy_probabilities))
    lines = [
        f"correlated batch: {run.batch_name} repeat={run.repeat_index} shots={run.shots}",
        (
            f"base_seed={run.base_seed} noise_seed={run.noise_seed} "
            f"measurement_seed={run.measurement_seed}"
        ),
        "outcome | ideal exact | noisy exact | ideal count | noisy count | 95% test",
    ]
    for outcome in outcomes:
        test = run.difference_tests.get(outcome)
        verdict = test.interpretation if test else "not tested"
        lines.append(
            " | ".join(
                [
                    outcome,
                    f"{run.ideal_probabilities.get(outcome, 0.0):.6f}",
                    f"{run.noisy_probabilities.get(outcome, 0.0):.6f}",
                    str(run.ideal_counts.get(outcome, 0)),
                    str(run.noisy_counts.get(outcome, 0)),
                    verdict,
                ]
            )
        )
    if run.traces:
        first = run.traces[0]
        lines.append(
            "trace audit: first shot "
            f"noise_errors={first['noise_errors']} mid_measurements={first['mid_measurements']} "
            f"final={first['final_outcome']}"
        )
    return "\n".join(lines)


def format_trajectory_batch_report(runs: list[TrajectoryRun]) -> str:
    return "\n\n".join(format_trajectory_run_report(run) for run in runs)


def format_correlation_sweep(points: list[CorrelationSweepPoint]) -> str:
    outcomes = sorted({outcome for point in points for outcome in point.noisy_probabilities})
    lines = ["rho | " + " | ".join(f"p({outcome})" for outcome in outcomes) + " | counts"]
    for point in points:
        lines.append(
            f"{point.correlation:.3f} | "
            + " | ".join(f"{point.noisy_probabilities.get(outcome, 0.0):.6f}" for outcome in outcomes)
            + f" | {point.run.noisy_counts}"
        )
    return "\n".join(lines)
