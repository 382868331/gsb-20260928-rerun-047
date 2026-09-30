"""Command-line entry point for the bundled small experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .correlated import CorrelatedNoiseEvent, CorrelatedNoiseModel
from .experiments import ExperimentBatch, rerun_archive, run_batch, save_batch_archive
from .gates import Circuit
from .noise import NoiseModel, NoiseRule
from .report import (
    format_batch_report,
    format_correlation_sweep,
    format_trajectory_batch_report,
)
from .trajectories import (
    TrajectoryBatch,
    archive_kind,
    replay_archive,
    run_trajectory_batch,
    save_trajectory_archive,
    sweep_correlation,
)


def build_batch(name: str, shots: int, seed: int, noise: float, ordering: str) -> ExperimentBatch:
    if name == "single":
        circuit = Circuit(1).h(0).h(0).h(0)
        damping = NoiseRule("amplitude_damping", noise, (0,), after_gate=0)
        flip = NoiseRule("bit_flip", noise, (0,), after_gate=0)
        if ordering == "after":
            ordering = "ad-bf"
        if ordering == "ad-bf":
            rules = [damping, flip]
        elif ordering == "bf-ad":
            rules = [flip, damping]
        else:
            raise ValueError("single ordering must be 'ad-bf' or 'bf-ad'")
    elif name == "bell":
        circuit = Circuit(2).h(0).cnot(0, 1)
        if ordering == "early":
            rules = [NoiseRule("bit_flip", noise, (0,), after_gate=0)]
        elif ordering == "late":
            rules = [NoiseRule("bit_flip", noise, (0,), after_gate=1)]
        else:
            raise ValueError("bell ordering must be 'early' or 'late'")
    else:
        raise ValueError("preset must be 'single', 'bell' or 'corr'")
    return ExperimentBatch(
        circuit=circuit,
        noise_model=NoiseModel(f"{name}-{ordering}-p{noise:g}", rules),
        shots=shots,
        base_seed=seed,
        name=name,
    )


def build_correlated_batch(
    shots: int, seed: int, probability: float, correlation: float
) -> TrajectoryBatch:
    # Two X gates with one correlated bit-flip perturbation after each: two
    # flips cancel when perfectly correlated, so the outcome distribution
    # changes continuously with the correlation strength.
    circuit = Circuit(1).x(0).x(0)
    model = CorrelatedNoiseModel(
        f"corr-p{probability:g}-rho{correlation:g}",
        [
            CorrelatedNoiseEvent("bit_flip", 0, 0, probability, correlation=correlation),
            CorrelatedNoiseEvent("bit_flip", 1, 0, probability, correlation=correlation),
        ],
    )
    return TrajectoryBatch(circuit, model, shots=shots, base_seed=seed, name="corr")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small quantum noise research experiments")
    parser.add_argument("--preset", choices=("single", "bell", "corr"), default="single")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--noise", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=0.0, help="temporal noise correlation")
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="comma-separated correlations, e.g. 0,0.5,1 (corr preset only)",
    )
    parser.add_argument(
        "--ordering",
        choices=("early", "late", "after", "ad-bf", "bf-ad"),
        default="after",
        help="noise insertion position; early/late differ for bell",
    )
    parser.add_argument("--output", type=Path, help="JSON archive path")
    parser.add_argument("--verify", type=Path, help="verify and replay a saved archive")
    args = parser.parse_args(argv)

    if args.verify is not None:
        replayed = replay_archive(args.verify)
        if archive_kind(args.verify) == "correlated":
            print(format_trajectory_batch_report(replayed))
        else:
            print(format_batch_report(replayed))
        return 0

    if args.preset == "corr":
        batch = build_correlated_batch(args.shots, args.seed, args.noise, args.rho)
        if args.sweep is not None:
            correlations = [float(value) for value in args.sweep.split(",") if value.strip()]
            points = sweep_correlation(batch, correlations)
            print(format_correlation_sweep(points))
            if args.output is not None:
                save_trajectory_archive(args.output, batch, [point.run for point in points])
            return 0
        runs = run_trajectory_batch(batch, repeats=args.repeats)
        print(format_trajectory_batch_report(runs))
        if args.output is not None:
            save_trajectory_archive(args.output, batch, runs)
        return 0

    ordering = (
        "early"
        if args.ordering == "after" and args.preset == "bell"
        else args.ordering
    )
    batch = build_batch(args.preset, args.shots, args.seed, args.noise, ordering)
    runs = run_batch(batch, repeats=args.repeats)
    if args.output is not None:
        save_batch_archive(args.output, batch, runs)
    print(format_batch_report(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
