"""Final matched first-answer versus second-pass detector comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from .combined import make_splits, paired_comparison, probabilities, uncertainty_metrics
from .features import load_run_features
from .finish import wait_lock
from .storage import atomic_json, digest, sha256_file
from .transfer import fit_source

SECOND = ("native_none", "control_none_concat", "ensemble_mean")


def metrics(y, p):
    result = uncertainty_metrics(y, p)
    result["confusion_matrix_correct_incorrect"] = confusion_matrix(y, p >= .5, labels=[0, 1]).tolist()
    return result


def validate_second(checkpoint, summary_entry, expected_fingerprint, seed, layers, y, splits):
    if (checkpoint.get("fingerprint") != expected_fingerprint
            or summary_entry.get("fingerprint") != expected_fingerprint
            or checkpoint.get("seed") != seed or checkpoint.get("layers") != layers):
        raise ValueError("Second-pass checkpoint provenance does not match features/settings")
    predictions = {}
    for split in ("dev", "test"):
        saved = checkpoint["predictions"][split]
        if (saved["indices"] != splits[split].tolist()
                or saved["y_error"] != y[splits[split]].tolist()):
            raise ValueError("Second-pass questions, ordering or labels differ from matched split")
        predictions[split] = {}
        for name in SECOND:
            p = np.asarray(saved["p_error"][name], dtype=float)
            if p.shape != (len(splits[split]),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("Invalid cached second-pass probabilities")
            computed = metrics(y[splits[split]], p)
            for record in (checkpoint, summary_entry):
                for key in ("auroc", "brier"):
                    if not np.isclose(record["methods"][name][split][key], computed[key], rtol=0, atol=1e-10):
                        raise ValueError("Cached second-pass metrics do not match predictions")
            predictions[split][name] = p
    return predictions


def evaluate_first(first_features, y, splits, cached, c, repetitions, seed):
    # Only this model is newly fitted. Every second-pass prediction is reused.
    base, calibrator = fit_source(first_features, y, splits["train"], splits["dev"], c)
    predictions = {split: dict(values) for split, values in cached.items()}
    for split in predictions:
        predictions[split]["first_pass_state"] = probabilities(base, calibrator, first_features[splits[split]])
    methods = {name: {split: metrics(y[splits[split]], predictions[split][name]) for split in predictions}
               for name in ("first_pass_state",) + SECOND}
    comparisons = {name: paired_comparison(y[splits["test"]], predictions["test"][name],
                                           predictions["test"]["first_pass_state"], repetitions, seed)
                   for name in SECOND}
    return {"methods": methods, "second_minus_first": comparisons,
            "first_pass_calibration_slope": float(calibrator.coef_[0, 0]),
            "split_class_counts": {split: {"correct": int(np.sum(y[rows] == 0)), "incorrect": int(np.sum(y[rows] == 1))}
                                   for split, rows in splits.items()},
            "predictions": {split: {"indices": splits[split].tolist(), "y_error": y[splits[split]].tolist(),
                                     "p_error": {name: p.tolist() for name, p in values.items()}}
                            for split, values in predictions.items()}}


def summarize(report):
    lines = ["MATCHED FIRST-PASS / SECOND-PASS DETECTION", "Fixed all-layer configuration; same questions, labels, splits and probe/calibration protocol.",
             "AUROC is ranking performance, not classification accuracy. Accuracy uses the fixed probability threshold 0.5.",
             "Deltas=second minus first: +AUROC, -Brier and -risk80 favour second pass.",
             "Paired intervals are conditional, exploratory and unadjusted; overlapping split seeds are not independent validation."]
    for seed, result in report["seeds"].items():
        lines.append(f"\nSEED {seed}")
        for name, splits in result["methods"].items():
            m = splits["test"]
            lines.append(f"{name}: AUC={m['auroc']:.4f}, Brier={m['brier']:.4f}, risk80={m['error_rate_at_80pct_coverage']:.4f}, accuracy05={m['accuracy_at_05']:.4f}, balanced_accuracy05={m['balanced_accuracy_at_05']:.4f}")
        for name, comparison in result["second_minus_first"].items():
            a, b = comparison["auroc"], comparison["brier"]
            lines.append(f"  {name} minus first: delta AUC={a['difference']:+.4f}, CI={a['paired_95_ci']}; delta Brier={b['difference']:+.4f}, CI={b['paired_95_ci']}")
    lines.append("\nIMPACT ACROSS SPLIT SEEDS")
    n = len(report["seeds"])
    for name in SECOND:
        pairs = [result["second_minus_first"][name] for result in report["seeds"].values()]
        fields = []
        for key in ("auroc", "brier", "risk80"):
            values = np.array([pair[key]["difference"] for pair in pairs])
            fields.append(f"delta {key}={values.mean():+.4f} +/- {values.std(ddof=1) if n>1 else 0:.4f} SD")
        auc_wins = sum(pair["auroc"]["difference"] > 0 for pair in pairs)
        brier_wins = sum(pair["brier"]["difference"] < 0 for pair in pairs)
        auc_intervals = sum(pair["auroc"]["paired_95_ci"] is not None and pair["auroc"]["paired_95_ci"][0] > 0 for pair in pairs)
        brier_intervals = sum(pair["brier"]["paired_95_ci"] is not None and pair["brier"]["paired_95_ci"][1] < 0 for pair in pairs)
        lines.append(name + ": " + "; ".join(fields))
        lines.append(f"  higher AUC in {auc_wins}/{n}; lower Brier in {brier_wins}/{n}; paired CI wholly favourable: AUC {auc_intervals}/{n}, Brier {brier_intervals}/{n}.")
        ranking = "consistently higher observed AUROC" if auc_wins == n else "no consistent AUROC improvement across every split"
        probability = "consistently lower observed Brier" if brier_wins == n else "no consistent Brier improvement across every split"
        lines.append(f"  Bounded conclusion: {ranking}; {probability}. This is not an independent generalisation claim.")
    lines.extend(["\nINTERPRETATION", "One second-pass none representation adds one assessment forward pass; control+none uses two and the native ensemble uses three, beyond original answer generation. These are stage comparisons, not equal-compute experiments.",
                  "This closes the matched baseline gap. Context and token position differ between passes; question-versus-answer attribution and fresh-data validation remain future work."])
    return "\n".join(lines) + "\n"


def thesis_impact(report):
    n = len(report["seeds"])
    lines = ["# Matched first-pass and second-pass detection: experimental closure", "",
             "## Experiment", "",
             f"The final comparison used the same reviewed answers and {n} stratified question-split seeds ({', '.join(report['seeds'])}). "
             f"All saved layers {report['layers']} were fixed in advance for this comparison. "
             f"The first-pass answer-state probe used balanced logistic regression (C={report['provenance']['c']}), training-only scaling and development-only Platt calibration. "
             "Second-pass probabilities were reused from the completed, identically aligned analysis. No new answers, labels, method selection or test-threshold optimisation were introduced.", "",
             "## Observed results", "",
             "Values are means +/- sample standard deviations across overlapping split seeds, not independent-replication confidence intervals.", "",
             "| Fixed detector | AUROC | Brier | Error rate at 80% coverage | Accuracy at 0.5 |",
             "|---|---:|---:|---:|---:|"]
    for name in ("first_pass_state",) + SECOND:
        cells = []
        for key in ("auroc", "brier", "error_rate_at_80pct_coverage", "accuracy_at_05"):
            values = np.array([r["methods"][name]["test"][key] for r in report["seeds"].values()])
            cells.append(f"{values.mean():.4f} +/- {values.std(ddof=1) if n>1 else 0:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Impact relative to the first-pass baseline", ""])
    for name in SECOND:
        comparisons = [r["second_minus_first"][name] for r in report["seeds"].values()]
        delta_auc = float(np.mean([p["auroc"]["difference"] for p in comparisons]))
        delta_brier = float(np.mean([p["brier"]["difference"] for p in comparisons]))
        auc_wins = sum(p["auroc"]["difference"] > 0 for p in comparisons)
        brier_wins = sum(p["brier"]["difference"] < 0 for p in comparisons)
        lines.append(f"- **{name}:** mean second-minus-first AUROC difference {delta_auc:+.4f} ({auc_wins}/{n} splits higher), "
                     f"and Brier difference {delta_brier:+.4f} ({brier_wins}/{n} splits lower). Per-seed paired intervals are reported in `summary.txt` and the detailed JSON.")
    lines.extend(["", "## Conclusion boundary", "",
                  "This completes the matched comparison missing from the earlier repeated-split analysis. "
                  "The direction, magnitude and consistency of the observed changes must be reported rather than assuming a two-pass advantage. "
                  "AUROC is not classification accuracy, and lower Brier does not isolate calibration from discrimination. "
                  "Paired intervals use question resampling conditional on the fitted models, without multiplicity correction. "
                  "Repeated split seeds overlap, and the examples were inspected during research development; the results are exploratory rather than independent validation.", "",
                  "Native none adds one assessment forward pass, control+none adds two, and the ensemble adds three beyond answer generation. "
                  "Consequently, this is not an equal-compute comparison. Different token positions and contexts, "
                  "and unresolved question-versus-answer contributions, prevent a causal or verification-specific interpretation. "
                  "These limitations are retained as future work rather than motivating more experiments for the current submission."])
    return "\n".join(lines) + "\n"


def analyze(run, output, seeds, repetitions):
    if repetitions < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use positive bootstrap repetitions and unique split seeds")
    print("Loading verified saved features and completed second-pass outputs", flush=True)
    data, manifest = load_run_features(run)
    y = data["y_error"]
    first = data["generation_state"]
    if not np.isin(y, [0, 1]).all() or first.ndim != 2 or len(first) != len(y) or not np.isfinite(first).all():
        raise ValueError("Valid audited labels and finite first-pass features required")
    metrics_path = run / "results_seed_42_original_layers" / "metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    combined_dir = run / "combined_detectors"
    summary_path = combined_dir / "share_combined_results.json"
    if not summary_path.exists():
        raise ValueError("Completed combined-detector outputs are required; run finish_analysis.sbatch first")
    combined = json.loads(summary_path.read_text(encoding="utf-8"))
    config, original = saved["configuration"], combined["provenance"]
    if config.get("audit") != manifest.get("audit") or combined.get("audit") != manifest.get("audit"):
        raise ValueError("Correctness audits differ")
    expected = {"metrics_sha256": sha256_file(metrics_path), "manifest_sha256": sha256_file(run / "manifest.json"),
                "labels_or_features_sha256": sha256_file(run / ("labels.npz" if manifest.get("feature_storage") == "indexed_view_v1" else "features.npz"))}
    if any(original.get(key) != value for key, value in expected.items()) or original["c"] != config["c"]:
        raise ValueError("Completed second-pass features, labels or classifier settings differ")
    for name, value in original["code"].items():
        if sha256_file(Path(__file__).with_name(name)) != value:
            raise ValueError(f"Comparison implementation changed since second-pass fitting: {name}")
    report = {"provenance": {**expected, "combined_summary_sha256": sha256_file(summary_path),
                              "code": {name: sha256_file(Path(__file__).with_name(name))
                                       for name in ("matched_passes.py", "combined.py", "transfer.py", "train.py", "features.py")},
                              "seeds": seeds, "bootstrap": repetitions, "c": config["c"]},
              "layers": manifest["layers"], "audit": manifest.get("audit"),
              "protocol": "Fixed all saved layers. Source train scaler and balanced logistic regression; source dev Platt calibration. Reuse exactly aligned completed second-pass probabilities. No method/layer/threshold selection in this comparison.",
              "limitations": "Previously inspected data and overlapping split seeds; unadjusted conditional paired-bootstrap intervals. Calibrated-probability AUROC, Brier and risk80 are compared consistently. Probability saturation may affect ranking. Different contexts/token positions and extra assessment forwards prevent causal or equal-compute interpretations.",
              "seeds": {}}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", {"phase": "running"})
    for seed in seeds:
        splits = make_splits(y, seed, config, saved["split"])
        key = f"seed_{seed}_all_layers"
        source = combined_dir / f"{key}.json"
        checkpoint = json.loads(source.read_text(encoding="utf-8"))
        source_fingerprint = digest({"provenance": original, "seed": seed, "layers": manifest["layers"]})
        cached = validate_second(checkpoint, combined["runs"][key], source_fingerprint, seed, manifest["layers"], y, splits)
        baseline_path = run / "results_seed_42_all_layers" / "metrics.json"
        fingerprint = digest({"provenance": report["provenance"], "seed": seed, "second_checkpoint_sha256": sha256_file(source),
                              "original_baseline_sha256": sha256_file(baseline_path) if baseline_path.exists() else None})
        target = output / f"seed_{seed}.json"
        result = json.loads(target.read_text()) if target.exists() else {}
        if result.get("fingerprint") != fingerprint:
            print(f"Seed {seed}: fitting one first-pass probe and paired comparisons", flush=True)
            result = evaluate_first(first, y, splits, cached, config["c"], repetitions, seed)
            result.update(seed=seed, fingerprint=fingerprint)
            # Reproduce the original all-layer first-pass baseline where available.
            if seed == config["seed"] and baseline_path.exists():
                baseline = json.loads(baseline_path.read_text())
                if (baseline["configuration"].get("selected_layers") == manifest["layers"]
                        and baseline["split"] == saved["split"] and baseline["configuration"].get("audit") == manifest.get("audit")):
                    for metric in ("auroc", "brier"):
                        if not np.isclose(result["methods"]["first_pass_state"]["test"][metric], baseline["models"]["generation_state"]["metrics"][metric], atol=1e-6, rtol=0):
                            raise ValueError("New first-pass fit does not reproduce original-split baseline")
                    result["original_baseline_reproduced"] = True
            atomic_json(target, result)
        report["seeds"][str(seed)] = {name: value for name, value in result.items() if name != "predictions"}
        print(f"Seed {seed} complete", flush=True)
    atomic_json(output / "share_matched_passes.json", report)
    text = summarize(report)
    (output / "summary.txt").write_text(text, encoding="utf-8")
    (output / "thesis_impact.md").write_text(thesis_impact(report), encoding="utf-8")
    atomic_json(output / "status.json", {"phase": "complete", "summary": str((output / "summary.txt").resolve())})
    print(text, flush=True)
    print(f"Share: {(output / 'summary.txt').resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    output = args.output or args.run / "matched_pass_detection"
    output.mkdir(parents=True, exist_ok=True)
    with wait_lock(output / "analysis.lock"):
        analyze(args.run, output, args.seeds, args.bootstrap)


if __name__ == "__main__":
    main()
