"""First-answer versus second-pass geometry using existing paired hidden states."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .features import load_run_features
from .storage import atomic_json, digest, file_lock, sha256_file
from .subspaces import compare_spaces, principal_space
from .train import select_layers

CONDITIONS = ("generation", "none", "verify", "control")


def direction_correctness(scores, y, train, test):
    """Choose each axis orientation on train only; held-out AUROC is descriptive."""
    result = []
    for column in scores.T:
        train_auc = float(roc_auc_score(y[train], column[train]))
        sign = 1 if train_auc >= .5 else -1
        scale = float(column[train].std())
        effect = float(column[train][y[train] == 1].mean() - column[train][y[train] == 0].mean())
        result.append({"train_auroc_oriented": max(train_auc, 1 - train_auc),
                       "orientation_chosen_on_train": sign,
                       "test_auroc_same_orientation": float(roc_auc_score(y[test], sign * column[test])),
                       "train_class_mean_difference_in_train_sd": effect / scale if scale else None})
    return result


def pair_geometry(left, right, a, b, train):
    x, z = left[train], right[train]
    delta = z.mean(axis=0) - x.mean(axis=0)
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(z, axis=1)
    cosine = np.divide(np.sum(x * z, axis=1), norms, out=np.zeros(len(x)), where=norms > 0)
    centered_x, centered_z = x - x.mean(axis=0), z - z.mean(axis=0)
    def captured(values, basis):
        total = float(np.sum(values ** 2))
        return float(np.sum((values @ basis) ** 2) / total) if total else None
    return {**compare_spaces(a, b), "train_mean_shift_norm": float(np.linalg.norm(delta)),
            "train_mean_paired_cosine": float(cosine.mean()),
            "train_paired_cosine_quantiles_05_50_95": np.quantile(cosine, [.05, .5, .95]).tolist(),
            "train_mean_paired_distance": float(np.linalg.norm(z - x, axis=1).mean()),
            "right_centered_variance_captured_by_left_top_k": captured(centered_z, a),
            "left_centered_variance_captured_by_right_top_k": captured(centered_x, b)}


def group_analysis(features, y, splits, k, csv_path):
    train, test = splits["train"], splits["test"]
    bases, centered, descriptions = {}, {}, {}
    for name, x in features.items():
        bases[name], descriptions[name] = principal_space(x[train], k)
        centered[name] = np.asarray(x, dtype=float) - x[train].mean(axis=0)
        descriptions[name]["correctness_along_own_components"] = direction_correctness(
            centered[name] @ bases[name], y, train, test)
    pairs = {f"{left}_vs_{right}": pair_geometry(features[left], features[right], bases[left], bases[right], train)
             for left, right in itertools.combinations(features, 2)}
    shared_projections = {name: x @ bases["generation"] for name, x in centered.items()}
    transition = {}
    for name in CONDITIONS[1:]:
        scores = shared_projections[name] - shared_projections["generation"]
        transition[f"generation_to_{name}"] = {
            "description": "Paired second-minus-first changes on first-pass PCs after subtracting each condition's training mean",
            "correctness_along_transition_components": direction_correctness(scores, y, train, test)}
    # PC numbers across different own bases are not shared coordinates. Export
    # first-pass-basis coordinates as well so paired movement can be plotted fairly.
    split_names = np.empty(len(y), dtype=object)
    for name, rows in splits.items():
        split_names[rows] = name
    temporary = csv_path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_index", "split", "y_error", "condition"] +
                            [f"own_pc_{i+1}" for i in range(k)] + [f"first_pass_pc_{i+1}" for i in range(k)])
            for name in features:
                own = centered[name] @ bases[name]
                for row in range(len(y)):
                    writer.writerow([row, split_names[row], int(y[row]), name] + own[row].tolist() + shared_projections[name][row].tolist())
        temporary.replace(csv_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"conditions": descriptions, "pairs": pairs, "paired_transitions": transition,
            "projections_csv": csv_path.name}


def analyze(run, metrics_path, output, k=10):
    data, manifest = load_run_features(run)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    if saved["configuration"].get("audit") != manifest.get("audit"):
        raise ValueError("Metrics and feature manifest have different audits")
    y = data["y_error"]
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Audited binary labels required")
    splits = {name: np.asarray(saved["split"][name]) for name in ("train", "dev", "test")}
    rows = np.concatenate(list(splits.values()))
    if (any(a.ndim != 1 for a in splits.values()) or not np.issubdtype(rows.dtype, np.integer)
            or not np.array_equal(np.sort(rows), np.arange(len(y)))):
        raise ValueError("Saved splits must partition source rows exactly once")
    if any(len(np.unique(y[splits[name]])) != 2 for name in ("train", "test")):
        raise ValueError("Both correctness classes required in train and test")
    layers = manifest["layers"]
    groups = {f"layer_{layer}": [layer] for layer in layers}
    if set([-1, -4, -8]) <= set(layers):
        groups["original_layers"] = [-1, -4, -8]
    groups["all_layers"] = layers
    provenance = {"metrics_sha256": sha256_file(metrics_path), "manifest_sha256": sha256_file(run / "manifest.json"),
                  "local_feature_or_label_sha256": sha256_file(run / ("labels.npz" if manifest.get("feature_storage") == "indexed_view_v1" else "features.npz")),
                  "code": {name: sha256_file(Path(__file__).with_name(name))
                           for name in ("pass_geometry.py", "subspaces.py", "train.py", "features.py")}, "k": k}
    report = {"provenance": provenance, "audit": manifest.get("audit"),
              "evaluation": manifest.get("evaluation"), "generation_prompt_template": manifest.get("generation_prompt_template"),
              "first_pass_state_position": manifest.get("generation_state_position", "unspecified in manifest"),
              "second_pass_state_position": manifest.get("evaluation", {}).get("hidden_position", "unspecified in manifest"),
              "split_sizes": {name: len(rows) for name, rows in splits.items()},
              "method": "Train-only condition means and covariance eigenvectors; shared saved feature coordinates, without additional coordinate scaling. Existing extractor L2-normalises each layer state before storage.",
              "limits": "First and second pass use different token positions and contexts. Mean shifts and overlaps are descriptive, not causal verification effects. Correctness projections are exploratory univariate diagnostics, not a trained/calibrated hallucination detector. Eigenvector signs are arbitrary; close eigenvalues permit rotations. No bootstrap stability or multiple-testing correction in this report.",
              "groups": {}}
    output.mkdir(parents=True, exist_ok=True)
    lines = ["FIRST-PASS / SECOND-PASS GEOMETRY", f"Top {k} covariance components; fitted on training rows only.",
             "generation=original answer state; none/verify/control=second-pass states at recorded manifest positions.",
             "Overlap: 1=same subspace, 0=orthogonal. Feature vectors were already layer-normalised by the extractor (L2 norm, not model LayerNorm)."]
    for group, selected_layers in groups.items():
        fingerprint = digest({"provenance": provenance, "layers": selected_layers})
        checkpoint, csv_path = output / f"{group}.json", output / f"{group}_projections.csv"
        result = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        if result.get("fingerprint") != fingerprint or not csv_path.exists():
            selected = select_layers(data, layers, selected_layers)
            features = {name: selected[f"{name}_state"] for name in CONDITIONS}
            if any(not np.isfinite(x).all() for x in features.values()):
                raise ValueError("Non-finite saved features")
            if len({x.shape for x in features.values()}) != 1:
                raise ValueError("Conditions must share row count and hidden coordinates; multiple prompt blocks require explicit alignment")
            result = group_analysis(features, y, splits, k, csv_path)
            result.update(fingerprint=fingerprint, layers=selected_layers)
            atomic_json(checkpoint, result)
        report["groups"][group] = result
        lines.append(f"\n{group}")
        for condition, description in result["conditions"].items():
            lines.append(f"{condition}: top{k} variance={description['top_k_explained_variance']:.4f}; cutoff eigengap={description['boundary_eigengap_relative']}")
        for condition in CONDITIONS[1:]:
            pair = result["pairs"][f"generation_vs_{condition}"]
            lines.append(f"generation vs {condition}: overlap={pair['subspace_overlap']:.4f}; mean paired cosine={pair['train_mean_paired_cosine']:.4f}; mean shift norm={pair['train_mean_shift_norm']:.4f}")
        print("\n".join(lines[-8:]), flush=True)
    atomic_json(output / "share_pass_geometry.json", report)
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Share: {(output / 'summary.txt').resolve()}\nDetail: {(output / 'share_pass_geometry.json').resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    output = args.output or args.run / "first_second_pass_geometry"
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / "analysis.lock"):
        analyze(args.run, args.metrics or args.run / "results_seed_42_original_layers" / "metrics.json", output, args.k)


if __name__ == "__main__":
    main()
