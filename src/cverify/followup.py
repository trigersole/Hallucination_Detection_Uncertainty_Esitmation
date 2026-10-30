"""Explicit follow-up tests on completed reviewed TruthfulQA and TriviaQA runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import attribution
from .combined import make_splits, paired_comparison, probabilities, uncertainty_metrics
from .data import normalize_answer
from .features import load_run_features
from .pass_geometry import analyze as analyze_geometry
from .review import read_records
from .storage import atomic_json, digest, file_lock, sha256_file
from .subspaces import analyze as analyze_subspaces
from .train import select_layers
from .transfer import analyze as analyze_transfer, fit_source
from .triviaqa_workflow import external_predictions


def resolve_run(path):
    if (path / "manifest.json").exists():
        return path.resolve()
    status = json.loads((path / "finalize_status.json").read_text())
    if status.get("phase") != "complete":
        raise ValueError(f"Complete the reviewed dataset workflow first: {path}")
    return Path(status["results"]).resolve()


def frozen_transfer(source, target, output, bootstrap=1000):
    a, am, ar, saved, splits = attribution.load_context(source)
    b, bm, br, _, _ = attribution.load_context(target)
    for key in ("model", "layers", "generation_prompt", "evaluation", "chat_format", "chat_template", "runtime",
                "generation_state_position", "resolved_layers"):
        if am.get(key) != bm.get(key):
            raise ValueError(f"Frozen transfer representation mismatch: {key}")
    if am.get("labeler") != "audited_labels" or bm.get("labeler") != "audited_labels":
        raise ValueError("Both datasets must have reviewed factual labels")
    raw = Path(am["feature_view"]["source_run"]) if am.get("feature_storage") == "indexed_view_v1" else source
    all_source = read_records(raw / "metadata.jsonl")
    source_questions = {normalize_answer(row["question"]) for row in all_source}
    if any(normalize_answer(row["question"]) in source_questions for row in br):
        raise ValueError("Source/target question overlap detected; do not silently evaluate duplicates")
    provenance = {"source": attribution.identity(source), "target": attribution.identity(target),
                  "source_metrics_sha256": sha256_file(source / "results_seed_42_all_layers" / "metrics.json"),
                  "code": {**attribution.implementation(), "followup.py": sha256_file(Path(__file__)),
                           "triviaqa_workflow.py": sha256_file(Path(__file__).with_name("triviaqa_workflow.py"))}}
    output.mkdir(parents=True, exist_ok=True)
    p, checks = external_predictions(a, b, splits, saved["configuration"]["c"], output / "frozen_models",
                                     digest(provenance), saved)
    y = b["y_error"]
    report = {"provenance": provenance, "bootstrap": bootstrap, "source_split_seed": saved["configuration"]["seed"],
              "target_counts": {"all_reviewed": len(y), "correct": int(np.sum(y == 0)), "incorrect": int(np.sum(y == 1))},
              "protocol": "Scaler/probe fit only on TruthfulQA train; calibration only on TruthfulQA dev. Evaluate every included reviewed TriviaQA answer, with no TriviaQA fitting, calibration or selection. Reproduce original source-test baselines first.",
              "limitations": "Retrospective frozen transfer: target data and results were already inspected in prior research. This is not fresh prospective validation. Dataset class priors differ; fixed-threshold accuracy and probability calibration may shift. Conditional paired intervals are unadjusted; assessment methods add compute.",
              "source_reproduction": checks,
              "methods": {name: uncertainty_metrics(y, values) for name, values in p.items()},
              "second_minus_first": {name: paired_comparison(y, p[name], p["first_pass_state"], bootstrap, 42)
                                     for name in ("native_none", "native_verify", "native_control", "control_none_concat", "ensemble_mean")}}
    atomic_json(output / "report.json", report)
    atomic_json(output / "predictions.json", {"question_ids": [row["id"] for row in br], "y_error": y.tolist(),
                                              "p_error": {name: values.tolist() for name, values in p.items()}})
    lines = ["FROZEN TRUTHFULQA -> REVIEWED TRIVIAQA", report["protocol"], report["limitations"], f"Target counts: {report['target_counts']}"]
    for name, m in report["methods"].items():
        lines.append(f"{name}: AUROC={m['auroc']:.4f}, Brier={m['brier']:.4f}, accuracy05={m['accuracy_at_05']:.4f}")
    for name, pair in report["second_minus_first"].items():
        lines.append(f"{name} minus first: delta AUC={pair['auroc']['difference']:+.4f} CI={pair['auroc']['paired_95_ci']}; delta Brier={pair['brier']['difference']:+.4f} CI={pair['brier']['paired_95_ci']}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def detail_matrices(data):
    matrices = {"first_pass_state": data["generation_state"], **{f"native_{name}": data[f"{name}_state"] for name in attribution.NATIVE},
                "verify_minus_control": data["verify_state"] - data["control_state"],
                "verify_minus_none": data["verify_state"] - data["none_state"],
                "control_minus_none": data["control_state"] - data["none_state"],
                "control_none_concat": np.c_[data["control_state"], data["none_state"]]}
    if any(not np.isfinite(x).all() for x in matrices.values()):
        raise ValueError("Nonfinite saved state")
    return matrices


def detailed_detection(run, output, seeds=(42, 43, 44, 45, 46), bootstrap=1000):
    data, manifest, rows, saved, _ = attribution.load_context(run)
    y = data["y_error"]
    groups = {f"layer_{layer}": [layer] for layer in manifest["layers"]}
    groups.update(original_layers=[-1, -4, -8], all_layers=manifest["layers"])
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"source": attribution.identity(run), "code": attribution.implementation(),
                  "followup_sha256": sha256_file(Path(__file__)), "bootstrap": bootstrap,
                  "metrics_sha256": sha256_file(run / "results_seed_42_all_layers" / "metrics.json")}
    report = {"provenance": provenance, "runs": {},
              "protocol": "Each layer/group is fitted independently with original per-question splits, train-only scaling/probe and dev-only calibration; probability subtraction is not used. Subtractions are of saved L2-normalised hidden-state vectors.",
              "limitations": "Descriptive follow-up on inspected test data. Many layer/method comparisons, unadjusted conditional intervals. Do not select a new winner or claim a unique mechanism from this sweep."}
    lines = ["DETAILED LAYER / VERIFICATION / SUBTRACTION DETECTION", report["protocol"], report["limitations"]]
    for group, layers in groups.items():
        x = detail_matrices(select_layers(data, manifest["layers"], layers))
        for seed in seeds:
            key = f"{group}_seed_{seed}"
            path = output / f"{key}.json"
            fingerprint = digest({"provenance": provenance, "seed": seed, "layers": layers})
            previous = json.loads(path.read_text()) if path.exists() else {}
            if previous.get("fingerprint") == fingerprint:
                result = previous
            else:
                splits = make_splits(y, seed, saved["configuration"], saved["split"])
                p = {}
                for name, values in x.items():
                    model, cal = fit_source(values, y, splits["train"], splits["dev"], saved["configuration"]["c"])
                    p[name] = probabilities(model, cal, values[splits["test"]])
                p["ensemble_mean"] = np.mean([p[f"native_{name}"] for name in attribution.NATIVE], axis=0)
                result = {"fingerprint": fingerprint, "layers": layers, "seed": seed,
                          "methods": {name: uncertainty_metrics(y[splits["test"]], values) for name, values in p.items()},
                          "minus_first": {name: paired_comparison(y[splits["test"]], values, p["first_pass_state"], bootstrap, seed)
                                           for name, values in p.items() if name != "first_pass_state"},
                          "split": {name: indices.tolist() for name, indices in splits.items()},
                          "y_error": y[splits["test"]].tolist(), "p_error": {name: values.tolist() for name, values in p.items()}}
                # The original seed42 all-layer single probes must reproduce the completed run.
                if seed == saved["configuration"]["seed"] and layers == manifest["layers"]:
                    keys = {"first_pass_state": "generation_state", "native_verify": "verify_state", "native_control": "control_state", "native_none": "none_state", "control_none_concat": "control_none_concat_state"}
                    for name, original in keys.items():
                        for metric in ("auroc", "brier"):
                            if not np.isclose(result["methods"][name][metric], saved["models"][original]["metrics"][metric], rtol=0, atol=1e-6):
                                raise ValueError(f"Original all-layer baseline failed reproduction: {name}/{metric}")
                atomic_json(path, result)
            report["runs"][key] = {k: v for k, v in result.items() if k not in ("p_error", "y_error")}
            print(f"Completed details: {key}", flush=True)
        lines.append(f"\n{group}: mean across {len(seeds)} overlapping split seeds")
        for method in list(x) + ["ensemble_mean"]:
            values = [report["runs"][f"{group}_seed_{seed}"]["methods"][method] for seed in seeds]
            lines.append(f"{method}: mean AUC={np.mean([m['auroc'] for m in values]):.4f}, mean Brier={np.mean([m['brier'] for m in values]):.4f}")
    atomic_json(output / "report.json", report)
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["paths", "cpu", "question", "question-score", "pairs-export", "pairs-extract", "pairs-score"])
    parser.add_argument("--truth", type=Path, default=Path("runs/truthfulqa_concise_chat_audit_l15"))
    parser.add_argument("--trivia", type=Path, default=Path("runs/triviaqa_validation500_replication_reviewed"))
    parser.add_argument("--dataset", choices=["truth", "trivia", "both"], default="both")
    parser.add_argument("--output-name", default="followup", help="Separate follow-up directory name for a changed protocol")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    if not args.output_name or Path(args.output_name).name != args.output_name or args.output_name in {".", ".."}:
        parser.error("output-name must be one directory name")
    runs = {"truth": resolve_run(args.truth), "trivia": resolve_run(args.trivia)}
    if args.command == "paths":
        for name, run in runs.items():
            print(f"{name}: {run}")
            for path in ("reviewed_pairs.json", "question_detection/summary.txt", "pair_detection/summary.txt",
                         "frozen_transfer/summary.txt", "detailed_detection/summary.txt", "geometry/summary.txt"):
                print(f"  {run / args.output_name / path}")
        return
    if args.command == "cpu":
        target = runs["trivia"]
        output = target / args.output_name
        output.mkdir(exist_ok=True)
        with file_lock(output / "cpu.lock"):
            atomic_json(output / "cpu_status.json", {"phase": "running"})
            frozen_transfer(runs["truth"], target, output / "frozen_transfer", args.bootstrap)
            detailed_detection(target, output / "detailed_detection", bootstrap=args.bootstrap)
            metrics = target / "results_seed_42_all_layers" / "metrics.json"
            for name, function, extra in (("geometry", analyze_geometry, 10), ("subspaces", analyze_subspaces, 10), ("condition_transfer", analyze_transfer, args.bootstrap)):
                destination = output / name
                destination.mkdir(exist_ok=True)
                with file_lock(destination / "analysis.lock"):
                    function(target, metrics, destination / "subspace_top10_train.json" if name == "subspaces" else destination, extra)
            atomic_json(output / "cpu_status.json", {"phase": "complete"})
        print(f"CPU follow-up summaries: {output}", flush=True)
        return
    for name, run in runs.items():
        if args.dataset not in ("both", name):
            continue
        output = run / args.output_name
        output.mkdir(exist_ok=True)
        pairs = output / "reviewed_pairs.json"
        with file_lock(output / f"{args.command}.lock"):
            if args.command == "question":
                attribution.extract_question_features(run, output / "question_features")
            elif args.command == "question-score":
                attribution.score_questions(run, output / "question_features", output / "question_detection", bootstrap=args.bootstrap)
            elif args.command == "pairs-export":
                attribution.export_pairs(run, pairs)
                print(f"Review candidate answer pairs in: {pairs}", flush=True)
            elif args.command == "pairs-extract":
                attribution.extract_pairs(run, pairs, output / "pair_features")
            else:
                attribution.score_pairs(run, pairs, output / "pair_features", output / "pair_detection", args.bootstrap)
        print(f"Completed {args.command}: {run}", flush=True)


if __name__ == "__main__":
    main()
