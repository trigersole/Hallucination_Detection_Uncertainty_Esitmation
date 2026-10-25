import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cverify.transfer import evaluate_group, paired_auc_difference, analyze


def fixture_data():
    rng = np.random.default_rng(82)
    y = np.arange(120) % 2
    x = rng.normal(size=(120, 8))
    x[:, 0] += 5 * y - 2.5
    splits = {"train": np.arange(80), "dev": np.arange(80, 100), "test": np.arange(100, 120)}
    return x, y, splits


def test_identical_conditions_transfer_identically():
    x, y, splits = fixture_data()
    result = evaluate_group({condition: x.copy() for condition in ("none", "verify", "control")}, y, splits, .1, 3, 42)
    predictions = result["test_predictions"]["error_probabilities"]
    for values in predictions.values():
        np.testing.assert_allclose(values, predictions["none_to_none"])
    assert result["raw_probe_direction_signed_cosines"]["none_vs_verify"] == pytest.approx(1)
    assert result["cells"]["none_to_control"]["versus_target_native"]["paired_bootstrap_95_ci"] == [0, 0]


def test_changed_encoding_can_predict_well_without_transfer():
    x, y, splits = fixture_data()
    flipped = -x
    result = evaluate_group({"none": x, "verify": flipped, "control": x}, y, splits, .1, 3, 42)
    assert result["cells"]["none_to_none"]["metrics"]["auroc"] > .9
    assert result["cells"]["verify_to_verify"]["metrics"]["auroc"] > .9
    assert result["cells"]["none_to_verify"]["metrics"]["auroc"] < .1
    assert result["raw_probe_direction_signed_cosines"]["none_vs_verify"] == pytest.approx(-1)


def test_target_shift_does_not_change_source_native_predictions():
    x, y, splits = fixture_data()
    first = evaluate_group({"none": x, "verify": x, "control": x}, y, splits, .1, 1, 42)
    shifted = x.copy()
    shifted[:, 0] += 100
    second = evaluate_group({"none": x, "verify": shifted, "control": x}, y, splits, .1, 1, 42)
    a, b = (r["test_predictions"]["error_probabilities"] for r in (first, second))
    np.testing.assert_allclose(a["none_to_none"], b["none_to_none"])
    assert np.mean(b["none_to_verify"]) > .99
    assert second["cells"]["none_to_verify"]["uncalibrated_decision_auroc"] == pytest.approx(
        first["cells"]["none_to_verify"]["uncalibrated_decision_auroc"])


def test_paired_bootstrap_uses_same_rows_for_each_predictor():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .3, .5, .8])
    result = paired_auc_difference(y, p, p, 30, 42)
    assert result["paired_bootstrap_95_ci"] == [0, 0]
    result = paired_auc_difference(y, p, 1 - p, 30, 42)
    assert result["auroc_difference"] == roc_auc_score(y, p) - roc_auc_score(y, 1-p)
    assert result["paired_bootstrap_95_ci"] == [1, 1]


def test_analysis_resume_provenance_and_split_validation(tmp_path, monkeypatch):
    x, y, splits = fixture_data()
    data = {key: x for key in ("generation_state", "none_state", "verify_state", "control_state")}
    np.savez(tmp_path / "features.npz", y_error=y, **data)
    (tmp_path / "manifest.json").write_text(json.dumps({"layers": [15]}))
    metrics = tmp_path / "metrics.json"
    saved = {"configuration": {"c": .1, "seed": 42}, "split": {key: value.tolist() for key, value in splits.items()}}
    metrics.write_text(json.dumps(saved))
    first = analyze(tmp_path, metrics, tmp_path / "transfer", 2)
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Valid checkpoint must not refit")
    monkeypatch.setattr("cverify.transfer.fit_source", unexpected_fit)
    second = analyze(tmp_path, metrics, tmp_path / "transfer", 2)
    assert first == second
    with pytest.raises(AssertionError, match="refit"):
        analyze(tmp_path, metrics, tmp_path / "transfer", 3)
    saved["split"]["test"][0] = 0
    metrics.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="partition"):
        analyze(tmp_path, metrics, tmp_path / "transfer", 2)
