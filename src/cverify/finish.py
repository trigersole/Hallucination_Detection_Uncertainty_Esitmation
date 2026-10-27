"""Finish existing CPU comparisons and inspect first-pass inheritance of transitions."""
import argparse
from contextlib import contextmanager, ExitStack
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from .combined import analyze as analyze_combined
from .pass_geometry import analyze as analyze_geometry
from .storage import LockBusyError, atomic_json, file_lock, sha256_file


@contextmanager
def wait_lock(path):
    """Let an already running analysis finish instead of submitting duplicate work."""
    announced = False
    with ExitStack() as stack:
        while True:
            try:
                stack.enter_context(file_lock(path))
                break
            except LockBusyError:
                if not announced:
                    print(f"Waiting for the existing analysis holding {path}", flush=True)
                    announced = True
                time.sleep(30)
        yield


def oriented_auc(y, score, train, test):
    auc = float(roc_auc_score(y[train], score[train]))
    sign = 1 if auc >= .5 else -1
    return {"train_auroc": max(auc, 1 - auc), "train_orientation": sign,
            "test_auroc": float(roc_auc_score(y[test], sign * score[test]))}


def transition_component(first, second, y, train, test):
    delta = second - first
    x, d = -first[train], delta[train]
    variance = float(np.var(x))
    covariance = float(np.mean((x - x.mean()) * (d - d.mean())))
    slope = covariance / variance if variance else 0.
    intercept = float(d.mean() - slope * x.mean())
    residual = delta - (intercept - slope * first)
    denominator = float(np.std(x) * np.std(d))
    correlation = covariance / denominator if denominator else None
    return {
        "train_second_to_first_variance_ratio": float(np.var(second[train]) / variance) if variance else None,
        "train_corr_transition_with_negative_first": correlation,
        "train_linear_r2_from_negative_first": correlation ** 2 if correlation is not None else None,
        "train_regression_slope": slope,
        "first": oriented_auc(y, first, train, test),
        "second_on_first_basis": oriented_auc(y, second, train, test),
        "transition": oriented_auc(y, delta, train, test),
        "transition_residual_after_train_only_regression": oriented_auc(y, residual, train, test),
    }


def read_projection_table(path, k):
    tables = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition, index = row["condition"], int(row["source_index"])
            condition_rows = tables.setdefault(condition, {})
            if index in condition_rows:
                raise ValueError("Duplicate question-condition projection row")
            condition_rows[index] = (row["split"], int(row["y_error"]),
                                     [float(row[f"first_pass_pc_{i+1}"]) for i in range(k)])
    if set(tables) != {"generation", "none", "verify", "control"}:
        raise ValueError("Expected generation/none/verify/control projection conditions")
    indices = sorted(tables["generation"])
    if indices != list(range(len(indices))):
        raise ValueError("Projection rows must cover source indices consecutively")
    reference = [(tables["generation"][i][0], tables["generation"][i][1]) for i in indices]
    features = {}
    for condition, rows in tables.items():
        if sorted(rows) != indices or [(rows[i][0], rows[i][1]) for i in indices] != reference:
            raise ValueError("Condition projections are not question/label/split aligned")
        features[condition] = np.asarray([rows[i][2] for i in indices])
        if not np.isfinite(features[condition]).all():
            raise ValueError("Non-finite projection coordinates")
    split_names, y = zip(*reference)
    y, split_names = np.asarray(y), np.asarray(split_names)
    if not np.isin(y, [0, 1]).all() or not np.isin(split_names, ["train", "dev", "test"]).all():
        raise ValueError("Invalid correctness labels or split names")
    train, test = np.flatnonzero(split_names == "train"), np.flatnonzero(split_names == "test")
    if any(len(np.unique(y[rows])) != 2 for rows in (train, test)):
        raise ValueError("Both classes required in training and test projections")
    return features, y, train, test


def transition_diagnostics(folder, geometry):
    groups = {}
    for group, description in geometry["groups"].items():
        path = folder / description["projections_csv"]
        features, y, train, test = read_projection_table(path, geometry["provenance"]["k"])
        k = features["generation"].shape[1]
        first_auc = [oriented_auc(y, features["generation"][:, i], train, test)["train_auroc"] for i in range(k)]
        selected = int(np.argmax(first_auc))
        comparisons = {condition: [transition_component(features["generation"][:, i], features[condition][:, i], y, train, test)
                                    for i in range(k)] for condition in ("none", "verify", "control")}
        groups[group] = {"projection_csv_sha256": sha256_file(path), "pc_selected_by_train_first_pass_auroc": selected + 1,
                         "components": comparisons}
    return {"groups": groups,
            "method": "Transition=second-first in the common first-pass PCA basis. Regress transition on negative-first using training rows only; evaluate the residual with training-fixed label orientation. Every component is retained; summary component selection uses training AUROC only.",
            "limits": "High positive correlation and small second/first variance ratio support arithmetic inheritance of first-pass scores. This is a descriptive diagnosis, not a causal decomposition or an independent validation. Low residual AUROC does not rule out information in other directions or nonlinear relationships."}


def write_reports(output, combined, transitions):
    lines = ["FINAL EXPLORATORY ANALYSIS", "Existing answers and reviewed labels; no new generation.",
             "Model selection uses development Brier. Split seeds overlap; fresh-data validation remains outstanding.",
             "\nDEVELOPMENT-SELECTED DETECTORS"]
    for seed, chosen in combined["selection"].items():
        m = chosen["test"]
        lines.append(f"seed {seed}: {chosen['group']} / {chosen['method']}; test AUC={m['auroc']:.4f}, Brier={m['brier']:.4f}, risk80={m['error_rate_at_80pct_coverage']:.4f}")
    runs = combined["runs"]
    group_names = sorted({key.split("_", 2)[2] for key in runs})
    lines.append("\nCOMBINATION PERFORMANCE VERSUS CONTROL+NONE")
    lines.append("Mean differences across split seeds: +AUROC is better; -Brier and -risk80 are better. These are not independent-replication confidence intervals.")
    for group in group_names:
        relevant = [result for key, result in runs.items() if key.split("_", 2)[2] == group]
        for candidate in ("pooled_mean", "ensemble_mean"):
            differences = {metric: [result["methods"][candidate]["test"][metric] - result["methods"]["control_none_concat"]["test"][metric]
                                    for result in relevant]
                           for metric in ("auroc", "brier", "error_rate_at_80pct_coverage")}
            better = sum(value < 0 for value in differences["brier"])
            lines.append(f"{group} / {candidate}: mean delta AUC={np.mean(differences['auroc']):+.4f}, Brier={np.mean(differences['brier']):+.4f}, risk80={np.mean(differences['error_rate_at_80pct_coverage']):+.4f}; lower Brier in {better}/{len(relevant)} splits")
            seed42 = runs.get(f"seed_42_{group}")
            if seed42:
                comparison = seed42["paired_comparisons"][f"{candidate}_minus_control_none_concat"]
                lines.append(f"  seed42 paired 95% CIs: delta AUC={comparison['auroc']['paired_95_ci']}; delta Brier={comparison['brier']['paired_95_ci']}")
    lines.extend(["\nTRANSITION INHERITANCE CHECK", "One PC per group chosen by first-pass TRAIN AUROC, not test performance.",
                  "corr = training correlation(delta, -first); ratio = Var(second)/Var(first). Residual removes a training-fitted linear prediction of delta from first."])
    for group, result in transitions["groups"].items():
        pc = result["pc_selected_by_train_first_pass_auroc"]
        lines.append(f"{group}: train-selected PC{pc}")
        for condition, components in result["components"].items():
            r = components[pc-1]
            def number(value):
                return f"{value:.4f}" if value is not None else "undefined"
            lines.append(f"  {condition}: corr={number(r['train_corr_transition_with_negative_first'])}, variance ratio={number(r['train_second_to_first_variance_ratio'])}; test AUC first={r['first']['test_auroc']:.4f}, transition={r['transition']['test_auroc']:.4f}, residual={r['transition_residual_after_train_only_regression']['test_auroc']:.4f}")
    lines.extend(["\nCLAIM LIMIT", "These results can establish exploratory detection/calibration behaviour on the reviewed subset. They cannot establish a new validated uncertainty method without independent data and a documented label-reliability check."])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "transition_diagnostics.json", transitions)
    atomic_json(output / "share_final_analysis.json", {"combined": combined, "transition_diagnostics": transitions})
    text = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"Share: {(output / 'summary.txt').resolve()}", flush=True)


def finish(run, seeds, bootstrap):
    metrics = run / "results_seed_42_original_layers" / "metrics.json"
    combined_dir, geometry_dir = run / "combined_detectors", run / "first_second_pass_geometry"
    combined_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    output = run / "final_analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", {"phase": "combined_comparison"})
    print("Stage 1/3: resume/check pooled and combined comparisons", flush=True)
    with wait_lock(combined_dir / "analysis.lock"):
        combined = analyze_combined(run, metrics, combined_dir, seeds, bootstrap)
    print("Stage 2/3: resume/check saved first/second-pass PCA projections", flush=True)
    atomic_json(output / "status.json", {"phase": "pass_geometry"})
    with wait_lock(geometry_dir / "analysis.lock"):
        geometry = analyze_geometry(run, metrics, geometry_dir, 10)
    for left, right in (("metrics_sha256", "metrics_sha256"), ("manifest_sha256", "manifest_sha256"),
                        ("labels_or_features_sha256", "local_feature_or_label_sha256")):
        if combined["provenance"][left] != geometry["provenance"][right]:
            raise ValueError("Feature/label or split provenance changed between analysis stages")
    print("Stage 3/3: transition inheritance diagnostics and final summary", flush=True)
    transitions = transition_diagnostics(geometry_dir, geometry)
    write_reports(output, combined, transitions)
    atomic_json(output / "status.json", {"phase": "complete", "summary": str((output / "summary.txt").resolve()),
                                         "report_sha256": sha256_file(output / "share_final_analysis.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    output = args.run / "final_analysis"
    output.mkdir(parents=True, exist_ok=True)
    with wait_lock(output / "analysis.lock"):
        finish(args.run, args.seeds, args.bootstrap)


if __name__ == "__main__":
    main()
