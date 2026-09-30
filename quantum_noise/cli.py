"""Command-line entry point for the bundled small experiments."""

from __future__ import annotations

import argparse
from math import pi
from pathlib import Path

from .experiments import ExperimentBatch, rerun_archive, run_batch, save_batch_archive
from .gates import Circuit
from .noise import CorrelatedNoiseGroup, NoiseModel, NoiseRule
from .report import format_batch_report


def build_batch(
    name: str, shots: int, seed: int, noise: float, ordering: str,
    correlation: float = 0.0,
) -> ExperimentBatch:
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
    elif name == "corr":
        # rx(pi/2) twice: P(0) = 2 p (1-p) (1-c), so the correlation
        # strength is directly visible in the exact probabilities.
        circuit = Circuit(1).rx(0, pi / 2).rx(0, pi / 2)
        group = CorrelatedNoiseGroup(
            channel="bit_flip",
            sites=((0, 0), (1, 0)),
            probability=noise,
            correlation=correlation,
            name="time-correlated-bit-flip",
        )
        return ExperimentBatch(
            circuit=circuit,
            noise_model=NoiseModel(f"corr-p{noise:g}-c{correlation:g}", (), (group,)),
            shots=shots,
            base_seed=seed,
            name=name,
        )
    else:
        raise ValueError("preset must be 'single', 'bell' or 'corr'")
    return ExperimentBatch(
        circuit=circuit,
        noise_model=NoiseModel(f"{name}-{ordering}-p{noise:g}", rules),
        shots=shots,
        base_seed=seed,
        name=name,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small quantum noise research experiments")
    parser.add_argument("--preset", choices=("single", "bell", "corr"), default="single")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--noise", type=float, default=0.10)
    parser.add_argument(
        "--correlation",
        type=float,
        default=0.0,
        help="time-correlation strength for the 'corr' preset (0 = independent)",
    )
    parser.add_argument(
        "--ordering",
        choices=("early", "late", "after", "ad-bf", "bf-ad"),
        default="after",
        help="noise insertion position; early/late differ for bell",
    )
    parser.add_argument("--output", type=Path, help="JSON archive path")
    parser.add_argument("--verify", type=Path, help="verify and report a saved archive")
    args = parser.parse_args(argv)

    if args.verify is not None:
        runs = rerun_archive(args.verify)
    else:
        ordering = (
            "early"
            if args.ordering == "after" and args.preset == "bell"
            else args.ordering
        )
        batch = build_batch(
            args.preset, args.shots, args.seed, args.noise, ordering, args.correlation
        )
        runs = run_batch(batch, repeats=args.repeats)
        if args.output is not None:
            save_batch_archive(args.output, batch, runs)
    print(format_batch_report(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
