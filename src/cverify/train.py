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


def select_layers(data: dict[str, np.ndarray], available: list[int], requested: list[int]) -> dict[str, np.ndarray]:
    """Slice [prompt pair, layer, hidden coordinate] without mixing pair blocks."""
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Select at least one layer, with no duplicates")
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(f"Requested layers were not extracted: {sorted(missing)}")
    positions = [available.index(layer) for layer in requested]
    hidden_size, remainder = divmod(data["generation_state"].shape[1], len(available))
    if remainder or hidden_size == 0:
        raise ValueError("Generation-state width does not match manifest layers")
    selected = dict(data)
    for name in (
        "generation_state", "verify_state", "control_state", "difference_state",
        "none_state", "verify_none_difference_state", "control_none_difference_state",
    ):
        if name not in data:
            continue
        values = data[name]
        blocks, remainder = divmod(values.shape[1], len(available) * hidden_size)
        if remainder or blocks == 0:
            raise ValueError(f"{name} width does not match manifest layers")
        shaped = values.reshape(len(values), blocks, len(available), hidden_size)
        selected[name] = shaped[:, :, positions, :].reshape(len(values), -1)
    selected["concat_state"] = np.c_[selected["verify_state"], selected["control_state"]]
    return selected


def feature_sets_for(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build score-based or representation-only comparisons from their real inputs."""
    baseline = np.c_[data["generation"], data["generation_state"]]
    feature_sets = {
        "generation_only": data["generation"],
        "generation_state": data["generation_state"],
        "generation_behavior_plus_state": baseline,
        "verify_state": data["verify_state"],
        "control_state": data["control_state"],
        "concat_state": data["concat_state"],
        "difference_state": data["difference_state"],
    }
    if "verification_scores" in data:
        feature_sets.update({
            "p_true_ensemble": data["verification_scores"],
            "generation_plus_verification": np.c_[data["generation"], data["verification_scores"]],
            "full_controlled": np.c_[baseline, data["verification_scores"], data["difference_state"]],
        })
    else:
        feature_sets.update({
            "none_state": data["none_state"],
            "verify_none_concat_state": np.c_[data["verify_state"], data["none_state"]],
            "control_none_concat_state": np.c_[data["control_state"], data["none_state"]],
            "verify_none_difference_state": data["verify_none_difference_state"],
            "control_none_difference_state": data["control_none_difference_state"],
            "generation_behavior_state_plus_verify_control_difference": np.c_[baseline, data["difference_state"]],
            "generation_behavior_state_plus_verify_none_difference": np.c_[baseline, data["verify_none_difference_state"]],
        })
    return feature_sets


def run_training(args) -> None:
    run = Path(args.run)
    with np.load(run / "features.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    selected = getattr(args, "layers", None)
    if selected is not None:
        if "layers" not in manifest:
            raise ValueError("Layer selection requires a manifest with extracted layer indices")
        data = select_layers(data, manifest["layers"], selected)
    y = data["y_error"].astype(int)
    if len(y) < 20 or np.min(np.bincount(y)) < 5:
        raise ValueError("Need at least 20 examples and 5 examples of each class to split and calibrate reliably")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, stratify=y)
    train_idx, dev_idx = train_test_split(train_idx, test_size=args.dev_size, random_state=args.seed + 1, stratify=y[train_idx])
    feature_sets = feature_sets_for(data)
    difficult_mask = (
        (data["generation"][:, 0] >= args.confidence_logprob)
        & (data["generation"][:, 1] >= args.consistency)
    )
    report = {
        "configuration": {
            "seed": args.seed, "c": args.c, "test_size": args.test_size,
            "dev_size": args.dev_size, "bootstrap": args.bootstrap,
            "extracted_layers": manifest.get("layers"),
            "selected_layers": selected if selected is not None else manifest.get("layers"),
            "evaluation": manifest.get("evaluation"),
            "generation_prompt": manifest.get("generation_prompt"),
            "feature_dimensions": {name: int(values.shape[1]) for name, values in feature_sets.items()},
        },
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
