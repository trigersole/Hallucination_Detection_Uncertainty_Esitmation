import json

import numpy as np
import pytest

from cverify.combined import (fit_pooled, probabilities, fit_predictions, make_splits,
                              select_by_development, uncertainty_metrics, paired_comparison, analyze)


def data_fixture():
    rng = np.random.default_rng(33)
    y = np.arange(60) % 2
    x = rng.normal(size=(60, 8))
    x[:, 0] += 2 * y
    data = {key: x.copy() for key in ("generation_state", "none_state", "verify_state", "control_state")}
    data["generation"] = rng.normal(size=(60, 3))
    data["y_error"] = y
    splits = {"train": np.arange(40), "dev": np.arange(40, 50), "test": np.arange(50, 60)}
    return data, y, splits


def test_pooled_replicates_have_same_regularization_and_calibration():
    data, y, splits = data_fixture()
    features = {name: data[f"{name}_state"] for name in ("none", "verify", "control")}
    single = fit_pooled(features, ("none",), y, splits, .1)
    pooled = fit_pooled(features, ("none", "verify", "control"), y, splits, .1)
    np.testing.assert_allclose(single[0][0].mean_, pooled[0][0].mean_, atol=1e-10)
    np.testing.assert_allclose(single[0][-1].coef_, pooled[0][-1].coef_, atol=1e-8)
    np.testing.assert_allclose(probabilities(*single, features["none"]), probabilities(*pooled, features["none"]), atol=1e-7)


def test_identical_conditions_ensemble_equals_native_and_test_labels_do_not_fit():
    data, y, splits = data_fixture()
    first = fit_predictions(data, y, splits, .1)
    np.testing.assert_allclose(first["test"]["ensemble_mean"], first["test"]["native_none"])
    np.testing.assert_allclose(first["test"]["pooled_mean"], first["test"]["pooled_none"])
    changed = y.copy()
    changed[splits["test"]] = 1 - changed[splits["test"]]
    second = fit_predictions(data, changed, splits, .1)
    for split in first:
        for name in first[split]:
            np.testing.assert_array_equal(first[split][name], second[split][name])


def test_saved_and_repeated_splits_partition_unique_questions():
    _, y, splits = data_fixture()
    config = {"seed": 42, "test_size": .25, "dev_size": .2}
    saved = {key: value.tolist() for key, value in splits.items()}
    result = make_splits(y, 42, config, saved)
    np.testing.assert_array_equal(result["train"], splits["train"])
    new = make_splits(y, 43, config, saved)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(new.values()))), np.arange(60))
    saved["test"][0] = 0
    with pytest.raises(ValueError, match="partition"):
        make_splits(y, 42, config, saved)


def test_development_selection_and_uncertainty_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .2, .8, .9])
    assert select_by_development(y, {"good": p, "bad": 1-p}, ["bad", "good"]) == "good"
    metrics = uncertainty_metrics(y, p)
    assert metrics["auroc"] == 1
    assert metrics["brier"] == pytest.approx(.025)
    assert metrics["ece_10_bins"] == pytest.approx(.15)
    assert metrics["selective_risk"]["0.5"] == 0
    assert sum(row["n"] for row in metrics["calibration_bins"]) == 4
    result = paired_comparison(y, p, p, 5, 42)
    for metric in result.values():
        assert metric["difference"] == 0
        assert metric["paired_95_ci"] == [0, 0]


def test_analysis_resumes_and_writes_compact_summary(tmp_path, monkeypatch):
    data, _, splits = data_fixture()
    np.savez(tmp_path / "features.npz", **data)
    (tmp_path / "manifest.json").write_text(json.dumps({"layers": [-1, -4, -8, 15]}))
    saved = {"configuration": {"seed": 42, "c": .1, "test_size": .25, "dev_size": .2},
             "split": {key: value.tolist() for key, value in splits.items()}}
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(saved))
    first = analyze(tmp_path, metrics, tmp_path / "output", [42], 1)
    assert len(first["runs"]) == 3
    assert (tmp_path / "output" / "summary.txt").exists()
    summary = (tmp_path / "output" / "summary.txt").read_text()
    assert "paired deltas versus control+none" in summary
    assert "CI=" in summary
    for result in first["runs"].values():
        assert len(result["methods"]) == 15
        assert "predictions" not in result
    def fail(*args, **kwargs):
        raise AssertionError("Unexpected recomputation")
    monkeypatch.setattr("cverify.combined.evaluate", fail)
    assert analyze(tmp_path, metrics, tmp_path / "output", [42], 1) == first
    with pytest.raises(AssertionError, match="recomputation"):
        analyze(tmp_path, metrics, tmp_path / "output", [42], 2)
