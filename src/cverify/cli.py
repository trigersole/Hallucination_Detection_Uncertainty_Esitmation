"""Command-line entry point."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cverify", description="Controlled verification hallucination detector")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Generate answers and extract controlled features")
    extract.add_argument("--data", required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--samples", type=int, default=5)
    extract.add_argument("--temperature", type=float, default=0.7)
    extract.add_argument("--max-new-tokens", type=int, default=32)
    extract.add_argument("--layers", type=int, nargs="+", default=[-1, -4, -8])
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--dtype", default="auto")
    extract.add_argument("--trust-remote-code", action="store_true")
    train = sub.add_parser("train", help="Fit calibrated probes and evaluate on a held-out split")
    train.add_argument("--run", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--dev-size", type=float, default=0.25, help="Fraction of the non-test data")
    train.add_argument("--c", type=float, default=0.1)
    train.add_argument("--bootstrap", type=int, default=1000)
    train.add_argument("--confidence-logprob", type=float, default=-0.5)
    train.add_argument("--consistency", type=float, default=0.8)
    train.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        from .extract import run_extraction
        run_extraction(args)
    else:
        from .train import run_training
        run_training(args)


if __name__ == "__main__":
    main()
