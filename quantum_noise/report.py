"""Human-readable side-by-side summaries."""

from __future__ import annotations

from .experiments import BatchRun


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
