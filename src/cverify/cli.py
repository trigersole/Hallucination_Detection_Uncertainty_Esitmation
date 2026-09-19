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
    extract.add_argument("--offset", type=int, default=0)
    extract.add_argument("--samples", type=int, default=5)
    extract.add_argument("--temperature", type=float, default=0.7)
    extract.add_argument("--max-new-tokens", type=int, default=32)
    extract.add_argument("--layers", type=int, nargs="+", default=[-1, -4, -8])
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--dtype", default="auto")
    extract.add_argument("--trust-remote-code", action="store_true")
    extract.add_argument("--labeler", choices=["exact", "bleurt"], default="exact")
    extract.add_argument("--label-threshold", type=float, default=0.5)
    extract.add_argument("--bleurt-model")
    extract.add_argument("--labeler-device", default="cuda")
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
    prepare = sub.add_parser("prepare-truthfulqa", help="Download and export all 817 TruthfulQA questions")
    prepare.add_argument("--output", default="data/truthfulqa.jsonl")
    merge = sub.add_parser("merge", help="Merge feature extraction shards")
    merge.add_argument("--runs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        from .extract import run_extraction
        run_extraction(args)
    elif args.command == "train":
        from .train import run_training
        run_training(args)
    elif args.command == "prepare-truthfulqa":
        from .data import export_truthfulqa
        count = export_truthfulqa(args.output)
        print(f"Wrote {count} TruthfulQA questions to {args.output}")
    else:
        from .merge import run_merge
        run_merge(args)


if __name__ == "__main__":
    main()
