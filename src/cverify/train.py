"""Train and evaluate regularized, calibrated linear error detectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    keep = np.argsort(p)[: max(1, int(np.ceil(0.8 * len(p))))]
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc_error_positive": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "error_rate_at_80pct_coverage": float(np.mean(y[keep])),
    }


def _bootstrap(y: np.ndarray, p: np.ndarray, repetitions: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "auroc": [], "auprc_error_positive": [], "brier": [], "error_rate_at_80pct_coverage": []
    }
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        result = _metrics(y[idx], p[idx])
        for key in values:
            values[key].append(result[key])
    return {key: [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] for key, vals in values.items() if vals}


def run_training(args) -> None:
    run = Path(args.run)
    data = np.load(run / "features.npz")
    y = data["y_error"].astype(int)
    if len(y) < 20 or np.min(np.bincount(y)) < 5:
        raise ValueError("Need at least 20 examples and 5 examples of each class to split and calibrate reliably")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, stratify=y)
    train_idx, dev_idx = train_test_split(train_idx, test_size=args.dev_size, random_state=args.seed + 1, stratify=y[train_idx])
    feature_sets = {
        "generation_only": data["generation"],
        "generation_state": data["generation_state"],
        "generation_behavior_plus_state": np.c_[data["generation"], data["generation_state"]],
        "p_true_ensemble": data["verification_scores"],
        "generation_plus_verification": np.c_[data["generation"], data["verification_scores"]],
        "verify_state": data["verify_state"],
        "control_state": data["control_state"],
        "concat_state": data["concat_state"],
        "difference_state": data["difference_state"],
        "full_controlled": np.c_[
            data["generation"], data["generation_state"],
            data["verification_scores"], data["difference_state"]
        ],
    }
    difficult_mask = (
        (data["generation"][:, 0] >= args.confidence_logprob)
        & (data["generation"][:, 1] >= args.consistency)
    )
    report = {
        "split": {"train": train_idx.tolist(), "dev": dev_idx.tolist(), "test": test_idx.tolist()},
        "difficult_subset_definition": {
            "generation_mean_log_probability_gte": args.confidence_logprob,
            "consistency_gte": args.consistency,
        },
        "models": {},
    }
    prediction_rows = []
    for name, features in feature_sets.items():
        base = make_pipeline(StandardScaler(), LogisticRegression(C=args.c, max_iter=2000, class_weight="balanced"))
        base.fit(features[train_idx], y[train_idx])
        # Platt scaling is fit only on development predictions. Doing this
        # explicitly avoids refitting the feature pipeline on held-out data.
        calibrator = LogisticRegression(C=1e6, max_iter=2000)
        calibrator.fit(base.decision_function(features[dev_idx]).reshape(-1, 1), y[dev_idx])
        probability = calibrator.predict_proba(base.decision_function(features[test_idx]).reshape(-1, 1))[:, 1]
        model_report = {
            "metrics": _metrics(y[test_idx], probability),
            "bootstrap_95_ci": _bootstrap(y[test_idx], probability, args.bootstrap, args.seed),
        }
        subgroup = difficult_mask[test_idx]
        subgroup_y = y[test_idx][subgroup]
        if subgroup.sum() >= 2 and len(np.unique(subgroup_y)) == 2:
            model_report["confident_consistent_subset"] = {
                "n": int(subgroup.sum()),
                "errors": int(subgroup_y.sum()),
                "metrics": _metrics(subgroup_y, probability[subgroup]),
            }
        else:
            model_report["confident_consistent_subset"] = {
                "n": int(subgroup.sum()),
                "errors": int(subgroup_y.sum()),
                "metrics": None,
                "reason": "AUROC requires both correct and incorrect answers in the subset",
            }
        report["models"][name] = model_report
        prediction_rows.append((name, probability))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for position, source_index in enumerate(test_idx):
            handle.write(json.dumps({
                "source_index": int(source_index),
                "y_error": int(y[source_index]),
                "confident_consistent": bool(difficult_mask[source_index]),
                "error_probabilities": {name: float(probability[position]) for name, probability in prediction_rows},
            }) + "\n")
    print(json.dumps(report["models"], indent=2))
