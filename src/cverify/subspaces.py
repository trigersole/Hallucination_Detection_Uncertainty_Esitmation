"""Compare training-only covariance eigenspaces in a shared hidden coordinate system."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .features import load_run_features
from .storage import atomic_json, sha256_file
from .train import select_layers


def principal_space(values, k):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite example-by-feature matrix")
    centered = values - values.mean(axis=0)
    # The sample Gram matrix avoids constructing a hidden-width squared covariance.
    eigenvalues, vectors = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    tolerance = max(centered.shape) * np.finfo(float).eps * max(float(eigenvalues[0]), 0)
    rank = int(np.sum(eigenvalues > tolerance))
    if k < 1 or k > rank:
        raise ValueError(f"Requested {k} components but centered numerical rank is {rank}")
    basis = centered.T @ vectors[:, :k] / np.sqrt(eigenvalues[:k])
    return basis, {
        "numerical_rank": rank,
        "covariance_eigenvalues": (eigenvalues[:k] / (len(values) - 1)).tolist(),
        "explained_variance_ratio": (eigenvalues[:k] / np.maximum(eigenvalues, 0).sum()).tolist(),
        "top_k_explained_variance": float(eigenvalues[:k].sum() / np.maximum(eigenvalues, 0).sum()),
        "boundary_eigengap_relative": float((eigenvalues[k-1] - eigenvalues[k]) / eigenvalues[k-1])
            if k < len(eigenvalues) else None,
    }


def compare_spaces(left, right):
    cross = left.T @ right
    cosines = np.clip(np.linalg.svd(cross, compute_uv=False), 0, 1)
    return {
        "principal_angle_cosines": cosines.tolist(),
        "principal_angles_degrees": np.degrees(np.arccos(cosines)).tolist(),
        "subspace_overlap": float(np.mean(cosines ** 2)),
        "absolute_eigenvector_cosines": np.abs(cross).tolist(),
    }


def analyze(run, metrics_path, output, k=10):
    data, manifest = load_run_features(run)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["configuration"].get("audit") != manifest.get("audit"):
        raise ValueError("Metrics and feature manifest refer to different correctness audits")
    n = len(data["y_error"])
    splits = {name: np.asarray(metrics["split"][name]) for name in ("train", "dev", "test")}
    combined = np.concatenate(list(splits.values()))
    if (not np.issubdtype(combined.dtype, np.integer) or len(combined) != n
            or not np.array_equal(np.sort(combined), np.arange(n))):
        raise ValueError("Saved splits must partition the feature rows exactly once")
    train = splits["train"]
    available = manifest["layers"]
    groups = {f"layer_{layer}": [layer] for layer in available}
    if all(layer in available for layer in [-1, -4, -8]):
        groups["original_layers"] = [-1, -4, -8]
    groups["all_layers"] = available
    report = {
        "run": str(run.resolve()), "metrics_file": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path), "audit": manifest.get("audit"),
        "k": k, "train_n": len(train), "train_indices": train.tolist(),
        "method": "Separate train-mean centering, raw shared hidden coordinates; no per-condition scaling",
        "interpretation": "Overlap is mean squared cosine of principal angles: 1=same subspace, 0=orthogonal. This measures variance geometry, not correctness information.",
        "groups": {},
    }
    for name, layers in groups.items():
        selected = select_layers(data, available, layers)
        bases, summaries = {}, {}
        for condition in ("none", "verify", "control"):
            bases[condition], summaries[condition] = principal_space(selected[f"{condition}_state"][train], k)
        pairs = {f"{left}_vs_{right}": compare_spaces(bases[left], bases[right])
                 for left, right in itertools.combinations(bases, 2)}
        report["groups"][name] = {"layers": layers, "conditions": summaries, "pairs": pairs}
        print(name + ": " + ", ".join(f"{pair}={value['subspace_overlap']:.4f}" for pair, value in pairs.items()), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(f"Share: {output.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    analyze(args.run, args.metrics or args.run / "results_seed_42_original_layers" / "metrics.json",
            args.output or args.run / "subspace_top10_train.json", args.k)


if __name__ == "__main__":
    main()
