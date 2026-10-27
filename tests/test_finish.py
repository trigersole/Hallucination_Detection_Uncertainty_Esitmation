from contextlib import nullcontext, contextmanager
import csv
import json

import numpy as np
import pytest

from cverify.finish import transition_component, read_projection_table, transition_diagnostics, finish, wait_lock
from cverify.storage import LockBusyError


def test_transition_inherits_first_when_second_projection_is_small():
    rng = np.random.default_rng(11)
    y = np.arange(1000) % 2
    first = y + rng.normal(scale=.5, size=len(y))
    second = rng.normal(scale=.005, size=len(y))
    train, test = np.arange(600), np.arange(600, 1000)
    result = transition_component(first, second, y, train, test)
    assert result["train_corr_transition_with_negative_first"] > .999
    assert result["train_second_to_first_variance_ratio"] < .001
    assert result["first"]["test_auroc"] > .85
    assert result["transition"]["test_auroc"] > .85
    assert .35 < result["transition_residual_after_train_only_regression"]["test_auroc"] < .65


def test_residual_can_retain_additional_correctness_information():
    rng = np.random.default_rng(9)
    y = np.arange(1000) % 2
    first = rng.normal(size=len(y))
    second = y + rng.normal(scale=.2, size=len(y))
    train, test = np.arange(600), np.arange(600, 1000)
    result = transition_component(first, second, y, train, test)
    assert result["transition_residual_after_train_only_regression"]["test_auroc"] > .95
    changed = second.copy()
    changed[test] += rng.normal(size=len(test)) * 50
    other = transition_component(first, changed, y, train, test)
    for key in ("train_corr_transition_with_negative_first", "train_regression_slope", "train_second_to_first_variance_ratio"):
        assert result[key] == other[key]


def write_csv(path, misalign=False):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_index", "condition", "split", "y_error", "first_pass_pc_1", "first_pass_pc_2"])
        rng = np.random.default_rng(5)
        for condition in ("generation", "none", "verify", "control"):
            for i in range(40):
                split = "train" if i < 24 else "test"
                label = i % 2
                if misalign and condition == "none" and i == 0:
                    label = 1 - label
                writer.writerow([i, condition, split, label, i % 2 + rng.normal(), rng.normal()])


def test_projection_alignment_and_train_selected_summary(tmp_path):
    path = tmp_path / "original_projections.csv"
    write_csv(path)
    features, y, train, test = read_projection_table(path, 2)
    assert features["generation"].shape == (40, 2)
    result = transition_diagnostics(tmp_path, {"provenance": {"k": 2}, "groups": {"original_layers": {"projections_csv": path.name}}})
    assert result["groups"]["original_layers"]["pc_selected_by_train_first_pass_auroc"] in (1, 2)
    assert len(result["groups"]["original_layers"]["components"]["none"]) == 2
    write_csv(path, misalign=True)
    with pytest.raises(ValueError, match="aligned"):
        read_projection_table(path, 2)


def test_finish_reuses_existing_analyses_and_reports_completion(tmp_path, monkeypatch):
    metrics = {"auroc": .8, "brier": .17, "error_rate_at_80pct_coverage": .6}
    comparison = {"auroc": {"paired_95_ci": [-.1, .1]}, "brier": {"paired_95_ci": [-.02, .02]}}
    combined = {"provenance": {"metrics_sha256": "m", "manifest_sha256": "f", "labels_or_features_sha256": "l"},
                "selection": {"42": {"group": "seed_42_original_layers", "method": "pooled_mean", "test": metrics}},
                "runs": {"seed_42_original_layers": {
                    "methods": {name: {"test": metrics} for name in ("pooled_mean", "ensemble_mean", "control_none_concat")},
                    "paired_comparisons": {f"{name}_minus_control_none_concat": comparison for name in ("pooled_mean", "ensemble_mean")}}}}
    geometry = {"provenance": {"metrics_sha256": "m", "manifest_sha256": "f", "local_feature_or_label_sha256": "l"}}
    calls = []
    def get_combined(*args):
        calls.append("combined")
        assert args[-2:] == ([42], 3)
        return combined
    def get_geometry(*args):
        calls.append("geometry")
        return geometry
    monkeypatch.setattr("cverify.finish.file_lock", lambda path: nullcontext())
    monkeypatch.setattr("cverify.finish.analyze_combined", get_combined)
    monkeypatch.setattr("cverify.finish.analyze_geometry", get_geometry)
    monkeypatch.setattr("cverify.finish.transition_diagnostics", lambda *args: {"groups": {}})
    finish(tmp_path, [42], 3)
    assert calls == ["combined", "geometry"]
    output = tmp_path / "final_analysis"
    assert json.loads((output / "status.json").read_text())["phase"] == "complete"
    assert "COMBINATION PERFORMANCE" in (output / "summary.txt").read_text()
    geometry["provenance"]["local_feature_or_label_sha256"] = "changed"
    with pytest.raises(ValueError, match="provenance"):
        finish(tmp_path, [42], 3)
    assert json.loads((output / "status.json").read_text())["phase"] != "complete"


def test_waits_for_active_analysis_without_removing_lock(tmp_path, monkeypatch):
    attempts, waits = [], []
    @contextmanager
    def busy_then_available(path):
        attempts.append(path)
        if len(attempts) < 3:
            raise LockBusyError("active")
        yield
    monkeypatch.setattr("cverify.finish.file_lock", busy_then_available)
    monkeypatch.setattr("cverify.finish.time.sleep", lambda duration: waits.append(duration))
    path = tmp_path / "analysis.lock"
    path.write_text("preserved")
    with wait_lock(path):
        assert path.read_text() == "preserved"
    assert len(attempts) == 3 and waits == [30, 30]
