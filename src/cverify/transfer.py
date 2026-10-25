"""Source-only correctness probes transferred between assessment conditions."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import load_run_features
from .storage import atomic_json, digest, sha256_file
from .train import _metrics, select_layers

CONDITIONS = ("none", "verify", "control")


def fit_source(values, y, train, dev, c):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
    model.fit(values[train], y[train])
    calibrator = LogisticRegression(C=1e6, max_iter=2000)
    calibrator.fit(model.decision_function(values[dev]).reshape(-1, 1), y[dev])
    return model, calibrator


def paired_auc_difference(y, first, second, repetitions, seed):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        rows = rng.integers(0, len(y), len(y))
        if len(np.unique(y[rows])) == 2:
            differences.append(roc_auc_score(y[rows], first[rows]) - roc_auc_score(y[rows], second[rows]))
    return {
        "auroc_difference": float(roc_auc_score(y, first) - roc_auc_score(y, second)),
        "paired_bootstrap_95_ci": np.quantile(differences, [0.025, 0.975]).tolist() if differences else None,
        "valid_bootstrap_draws": len(differences),
    }


def evaluate_group(features, y, splits, c, repetitions, seed):
    train, dev, test = (splits[name] for name in ("train", "dev", "test"))
    predictions, decisions, cells, gradients = {}, {}, {}, {}
    for source in CONDITIONS:
        model, calibrator = fit_source(features[source], y, train, dev, c)
        # Express fitted discriminant directions in common raw hidden coordinates.
        gradients[source] = model[-1].coef_[0] / model[0].scale_
        for target in CONDITIONS:
            key = f"{source}_to_{target}"
            decision = model.decision_function(features[target][test])
            probability = calibrator.predict_proba(decision.reshape(-1, 1))[:, 1]
            predictions[key] = probability
            decisions[key] = decision
            cells[key] = {
                "source": source, "target": target,
                "metrics": _metrics(y[test], probability),
                "uncalibrated_decision_auroc": float(roc_auc_score(y[test], decision)),
                "calibration_slope": float(calibrator.coef_[0, 0]),
            }
    for source, target in itertools.product(CONDITIONS, repeat=2):
        key = f"{source}_to_{target}"
        if source != target:
            cells[key]["versus_target_native"] = paired_auc_difference(
                y[test], decisions[key], decisions[f"{target}_to_{target}"], repetitions, seed)
            cells[key]["versus_source_native"] = paired_auc_difference(
                y[test], decisions[key], decisions[f"{source}_to_{source}"], repetitions, seed)
    angles = {}
    for left, right in itertools.combinations(CONDITIONS, 2):
        a, b = gradients[left], gradients[right]
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        angles[f"{left}_vs_{right}"] = float(a @ b / denominator) if denominator else None
    return {"cells": cells, "raw_probe_direction_signed_cosines": angles,
            "test_predictions": {"source_indices": test.tolist(), "y_error": y[test].tolist(),
                                 "raw_decision_scores": {key: value.tolist() for key, value in decisions.items()},
                                 "error_probabilities": {key: value.tolist() for key, value in predictions.items()}}}


def analyze(run, metrics_path, output, repetitions=1000):
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    data, manifest = load_run_features(run)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = saved["configuration"]
    if config.get("audit") != manifest.get("audit"):
        raise ValueError("Metrics and features have different correctness audits")
    y = data["y_error"].astype(int)
    splits = {name: np.asarray(saved["split"][name]) for name in ("train", "dev", "test")}
    rows = np.concatenate(list(splits.values()))
    if (any(v.ndim != 1 for v in splits.values()) or not np.issubdtype(rows.dtype, np.integer)
            or not np.array_equal(np.sort(rows), np.arange(len(y)))):
        raise ValueError("Saved splits must partition the feature rows exactly once")
    for name, indices in splits.items():
        if not np.isin(y[indices], [0, 1]).all() or len(np.unique(y[indices])) != 2:
            raise ValueError(f"Both audited classes are required in {name}")
    available = manifest["layers"]
    groups = {f"layer_{layer}": [layer] for layer in available}
    if all(layer in available for layer in [-1, -4, -8]):
        groups["original_layers"] = [-1, -4, -8]
    groups["all_layers"] = available
    provenance = {"metrics_sha256": sha256_file(metrics_path), "manifest_sha256": sha256_file(run / "manifest.json"),
                  "local_feature_or_label_sha256": sha256_file(run / ("labels.npz" if manifest.get("feature_storage") == "indexed_view_v1" else "features.npz")),
                  "code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ("transfer.py", "train.py", "features.py")},
                  "c": config["c"], "seed": config["seed"], "bootstrap": repetitions}
    report = {"run": str(run.resolve()), "provenance": provenance, "audit": manifest.get("audit"),
              "split_sizes": {name: len(rows) for name, rows in splits.items()},
              "protocol": "Train scaler and balanced linear probe on source train only; Platt calibration on source dev only; frozen transfer to target test. No target adaptation.",
              "primary_measure": "Raw decision AUROC and paired differences use uncalibrated source probe scores. Cell metrics additionally describe source-calibrated probabilities; probability saturation can lose ranking resolution.",
              "scope": "Exploratory analysis of previously inspected held-out questions. Bootstrap intervals condition on these fitted probes and this split, with no multiple-comparison correction.",
              "groups": {}}
    output.mkdir(parents=True, exist_ok=True)
    for name, layers in groups.items():
        fingerprint = digest({"provenance": provenance, "layers": layers})
        checkpoint = output / f"{name}.json"
        result = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        if result.get("fingerprint") != fingerprint:
            selected = select_layers(data, available, layers)
            features = {condition: selected[f"{condition}_state"] for condition in CONDITIONS}
            if any(not np.isfinite(values).all() for values in features.values()):
                raise ValueError("Non-finite representation values")
            result = evaluate_group(features, y, splits, config["c"], repetitions, config["seed"])
            result.update(fingerprint=fingerprint, layers=layers)
            atomic_json(checkpoint, result)
        report["groups"][name] = {key: value for key, value in result.items() if key != "test_predictions"}
        print(f"\n{name}: raw decision AUROC; rows=train condition, columns=test condition", flush=True)
        print("source    " + " ".join(f"{condition:>9}" for condition in CONDITIONS), flush=True)
        for source in CONDITIONS:
            print(f"{source:9} " + " ".join(f"{result['cells'][f'{source}_to_{target}']['uncalibrated_decision_auroc']:9.4f}"
                                          for target in CONDITIONS), flush=True)
    atomic_json(output / "share_transfer_results.json", report)
    print(f"Share: {(output / 'share_transfer_results.json').resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", default=1000, type=int)
    args = parser.parse_args()
    analyze(args.run, args.metrics or args.run / "results_seed_42_original_layers" / "metrics.json",
            args.output or args.run / "cross_condition_transfer", args.bootstrap)


if __name__ == "__main__":
    main()
