"""Bounded, question-grouped comparisons of pooled and combined error detectors."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import load_run_features
from .storage import atomic_json, digest, file_lock, sha256_file
from .train import _metrics, select_layers
from .transfer import CONDITIONS, fit_source


def uncertainty_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    result = _metrics(y, p)
    order = np.argsort(p, kind="stable")
    risk = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    result.update(log_loss=float(log_loss(y, p, labels=[0, 1])),
                  accuracy_at_05=float(np.mean((p >= .5) == y)),
                  balanced_accuracy_at_05=float(balanced_accuracy_score(y, p >= .5)),
                  aurc=float(np.mean(risk)))
    result["error_rate_at_80pct_coverage"] = float(risk[int(np.ceil(.8 * len(y))) - 1])
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = []
    for b in range(10):
        mask = bins == b
        if mask.any():
            calibration.append({"bin": b, "n": int(mask.sum()), "mean_p_error": float(p[mask].mean()),
                                "observed_error_rate": float(y[mask].mean())})
    result["ece_10_bins"] = float(sum(row["n"] * abs(row["mean_p_error"] - row["observed_error_rate"])
                                            for row in calibration) / len(y))
    result["calibration_bins"] = calibration
    result["selective_risk"] = {str(coverage): float(risk[int(np.ceil(coverage * len(y))) - 1])
                                for coverage in (.5, .6, .7, .8, .9, 1.)}
    return result


def fit_pooled(features, conditions, y, splits, c):
    """Give each unique question total weight one, independent of condition count."""
    train, dev = splits["train"], splits["dev"]
    count = len(conditions)
    x = np.concatenate([features[name][train] for name in conditions])
    labels = np.tile(y[train], count)
    weights = np.full(len(labels), 1 / count)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
    model.fit(x, labels, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    scores = np.concatenate([model.decision_function(features[name][dev]) for name in conditions])
    calibrator = LogisticRegression(C=1e6, max_iter=2000)
    calibrator.fit(scores.reshape(-1, 1), np.tile(y[dev], count),
                   sample_weight=np.full(len(scores), 1 / count))
    return model, calibrator


def probabilities(model, calibrator, x):
    return calibrator.predict_proba(model.decision_function(x).reshape(-1, 1))[:, 1]


def fit_predictions(data, y, splits, c):
    """Return dev/test probabilities; test labels never enter any fitting step."""
    features = {name: data[f"{name}_state"] for name in CONDITIONS}
    predictions = {"dev": {}, "test": {}}
    matrices = {f"native_{name}": values for name, values in features.items()}
    matrices.update(generation_only=data["generation"],
                    control_none_concat=np.c_[features["control"], features["none"]],
                    verify_none_difference=features["verify"] - features["none"])
    for name, x in matrices.items():
        model, calibrator = fit_source(x, y, splits["train"], splits["dev"], c)
        for split in predictions:
            predictions[split][name] = probabilities(model, calibrator, x[splits[split]])
    pooled, calibrator = fit_pooled(features, CONDITIONS, y, splits, c)
    for split, values in predictions.items():
        for name in CONDITIONS:
            values[f"pooled_{name}"] = probabilities(pooled, calibrator, features[name][splits[split]])
        values["pooled_mean"] = np.mean([values[f"pooled_{name}"] for name in CONDITIONS], axis=0)
        values["ensemble_mean"] = np.mean([values[f"native_{name}"] for name in CONDITIONS], axis=0)
        values["train_prevalence"] = np.full(len(splits[split]), y[splits["train"]].mean())
    for target in CONDITIONS:
        sources = tuple(name for name in CONDITIONS if name != target)
        model, calibrator = fit_pooled(features, sources, y, splits, c)
        for split in predictions:
            predictions[split][f"leave_out_{target}"] = probabilities(
                model, calibrator, features[target][splits[split]])
    return predictions


def select_by_development(y, predictions, names):
    return min(names, key=lambda name: (float(np.mean((predictions[name] - y) ** 2)), name))


def scalar_comparison(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "risk80": float(y[np.argsort(p, kind="stable")[:int(np.ceil(.8 * len(y)))]].mean())}


def paired_comparison(y, p, baseline, repetitions, seed):
    first, second = scalar_comparison(y, p), scalar_comparison(y, baseline)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in first}
    for _ in range(repetitions):
        rows = rng.integers(0, len(y), len(y))
        if len(np.unique(y[rows])) < 2:
            continue
        a, b = scalar_comparison(y[rows], p[rows]), scalar_comparison(y[rows], baseline[rows])
        for key in samples:
            samples[key].append(a[key] - b[key])
    return {key: {"difference": first[key] - second[key],
                  "paired_95_ci": np.quantile(values, [.025, .975]).tolist() if values else None}
            for key, values in samples.items()}


def complementarity(y, predictions):
    result = {}
    for left, right in itertools.combinations(CONDITIONS, 2):
        a, b = (predictions[f"native_{name}"] for name in (left, right))
        mistakes_a, mistakes_b = (a >= .5) != y, (b >= .5) != y
        result[f"{left}_vs_{right}"] = {
            "prediction_disagreement_rate": float(np.mean((a >= .5) != (b >= .5))),
            "both_detectors_wrong": int(np.sum(mistakes_a & mistakes_b)),
            "only_first_wrong": int(np.sum(mistakes_a & ~mistakes_b)),
            "only_second_wrong": int(np.sum(~mistakes_a & mistakes_b)),
        }
    return result


def evaluate(data, y, splits, c, bootstrap, seed):
    p = fit_predictions(data, y, splits, c)
    candidates = [name for name in p["dev"] if not name.startswith("leave_out_") and name != "train_prevalence"]
    selected = select_by_development(y[splits["dev"]], p["dev"], candidates)
    native = select_by_development(y[splits["dev"]], p["dev"], [f"native_{name}" for name in CONDITIONS])
    result = {"selected_by_dev_brier": selected, "native_selected_by_dev_brier": native,
              "split_class_counts": {name: {"correct": int(np.sum(y[indices] == 0)), "incorrect": int(np.sum(y[indices] == 1))}
                                     for name, indices in splits.items()},
              "methods": {name: {"dev": uncertainty_metrics(y[splits["dev"]], p["dev"][name]),
                                  "test": uncertainty_metrics(y[splits["test"]], values)}
                          for name, values in p["test"].items()},
              "dev_complementarity": complementarity(y[splits["dev"]], p["dev"]),
              "test_complementarity": complementarity(y[splits["test"]], p["test"]),
              "paired_comparisons": {}}
    for candidate in sorted(set(("pooled_mean", "ensemble_mean", selected))):
        for baseline in sorted(set((native, "control_none_concat", "verify_none_difference", "generation_only"))):
            if candidate != baseline:
                result["paired_comparisons"][f"{candidate}_minus_{baseline}"] = paired_comparison(
                    y[splits["test"]], p["test"][candidate], p["test"][baseline], bootstrap, seed)
    result["predictions"] = {split: {"indices": splits[split].tolist(), "y_error": y[splits[split]].tolist(),
                                    "p_error": {name: values.tolist() for name, values in predictions.items()}}
                             for split, predictions in p.items()}
    return result


def make_splits(y, seed, config, saved):
    if seed == config["seed"]:
        splits = {name: np.asarray(saved[name]) for name in ("train", "dev", "test")}
    else:
        train, test = train_test_split(np.arange(len(y)), test_size=config["test_size"], random_state=seed, stratify=y)
        train, dev = train_test_split(train, test_size=config["dev_size"], random_state=seed + 1, stratify=y[train])
        splits = {"train": train, "dev": dev, "test": test}
    rows = np.concatenate(list(splits.values()))
    if (any(v.ndim != 1 for v in splits.values()) or not np.issubdtype(rows.dtype, np.integer)
            or not np.array_equal(np.sort(rows), np.arange(len(y)))):
        raise ValueError("Question splits must partition rows exactly once")
    for name, indices in splits.items():
        if len(np.unique(y[indices])) != 2:
            raise ValueError(f"Both classes required in {name}")
    return splits


def analyze(run, metrics_path, output, seeds, bootstrap):
    if bootstrap < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use positive bootstrap repetitions and unique seeds")
    data, manifest = load_run_features(run)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = saved["configuration"]
    if config.get("audit") != manifest.get("audit"):
        raise ValueError("Metrics and features have different audits")
    y = data["y_error"]
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary audited labels required")
    groups = {"original_layers": [-1, -4, -8], "layer15": [15], "all_layers": manifest["layers"]}
    groups = {name: layers for name, layers in groups.items() if set(layers) <= set(manifest["layers"])}
    provenance = {"metrics_sha256": sha256_file(metrics_path), "manifest_sha256": sha256_file(run / "manifest.json"),
                  "labels_or_features_sha256": sha256_file(run / ("labels.npz" if manifest.get("feature_storage") == "indexed_view_v1" else "features.npz")),
                  "code": {name: sha256_file(Path(__file__).with_name(name))
                           for name in ("combined.py", "transfer.py", "train.py", "features.py")},
                  "bootstrap": bootstrap, "seeds": seeds, "c": config["c"]}
    report = {"provenance": provenance, "audit": manifest.get("audit"), "runs": {},
              "protocol": "All question representations stay in one split. Weighted pooled fitting gives each unique question total weight one. Source-dev Platt calibration; fixed equal-weight probability ensembles. Selection minimises development Brier score.",
              "limitations": "Exploratory reuse of inspected data; repeated splits overlap and are not independent replications. Bootstrap conditions on fitted models, with no multiplicity correction. ECE is descriptive with small samples. Fresh external validation and label reliability assessment remain required.",
              "selection": {}}
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        splits = make_splits(y, seed, config, saved["split"])
        for group, layers in groups.items():
            key = f"seed_{seed}_{group}"
            fingerprint = digest({"provenance": provenance, "seed": seed, "layers": layers})
            checkpoint = output / f"{key}.json"
            result = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            if result.get("fingerprint") != fingerprint:
                print(f"Starting {key}: fitting detectors and paired bootstrap comparisons", flush=True)
                selected = select_layers(data, manifest["layers"], layers)
                if any(not np.isfinite(selected[name]).all() for name in ("generation", "none_state", "verify_state", "control_state")):
                    raise ValueError("Non-finite features")
                result = evaluate(selected, y, splits, config["c"], bootstrap, seed)
                result.update(fingerprint=fingerprint, seed=seed, layers=layers)
                atomic_json(checkpoint, result)
            report["runs"][key] = {name: values for name, values in result.items() if name != "predictions"}
            chosen = result["selected_by_dev_brier"]
            print(f"{key}: dev-selected={chosen}; test AUROC={result['methods'][chosen]['test']['auroc']:.4f}, Brier={result['methods'][chosen]['test']['brier']:.4f}", flush=True)
        choices = [(key, result["selected_by_dev_brier"]) for key, result in report["runs"].items() if result["seed"] == seed]
        group, method = min(choices, key=lambda pair: (report["runs"][pair[0]]["methods"][pair[1]]["dev"]["brier"], pair))
        report["selection"][str(seed)] = {"group": group, "method": method,
                                          "test": report["runs"][group]["methods"][method]["test"]}
    atomic_json(output / "share_combined_results.json", report)
    lines = ["EXPLORATORY COMBINED DETECTORS", "Selected by development Brier; test data never used for selection.",
             "Overlapping split seeds are stability checks, not independent validation."]
    for seed, result in report["selection"].items():
        m = result["test"]
        lines.append(f"seed={seed} {result['group']} {result['method']}: AUROC={m['auroc']:.4f} Brier={m['brier']:.4f} risk80={m['error_rate_at_80pct_coverage']:.4f}")
    for group in groups:
        lines.append(f"\n{group}: test mean +/- sample SD across split seeds")
        for method in ("train_prevalence", "generation_only", "native_none", "native_verify", "native_control", "control_none_concat", "verify_none_difference", "pooled_mean", "ensemble_mean"):
            values = [report["runs"][f"seed_{seed}_{group}"]["methods"][method]["test"] for seed in seeds]
            fields = []
            for metric in ("auroc", "brier", "error_rate_at_80pct_coverage"):
                a = np.array([value[metric] for value in values])
                fields.append(f"{metric}={a.mean():.4f}+/-{a.std(ddof=1) if len(a)>1 else 0:.4f}")
            lines.append(method + ": " + " ".join(fields))
        if 42 in seeds:
            lines.append("Seed 42 paired deltas versus control+none (candidate minus baseline):")
            comparisons = report["runs"][f"seed_42_{group}"]["paired_comparisons"]
            for method in ("pooled_mean", "ensemble_mean"):
                comparison = comparisons[f"{method}_minus_control_none_concat"]
                fields = []
                for metric in ("auroc", "brier"):
                    value = comparison[metric]
                    fields.append(f"{metric}: delta={value['difference']:+.4f}, CI={value['paired_95_ci']}")
                lines.append(method + ": " + "; ".join(fields))
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"Share: {(output / 'share_combined_results.json').resolve()}\nShort summary: {(output / 'summary.txt').resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    output = args.output or args.run / "combined_detectors"
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / "analysis.lock"):
        analyze(args.run, args.metrics or args.run / "results_seed_42_original_layers" / "metrics.json",
                output, args.seeds, args.bootstrap)


if __name__ == "__main__":
    main()
