import json
from pathlib import Path

import numpy as np
import pytest

import cverify.combined as combined_module
import cverify.matched_passes as matched
from cverify.combined import make_splits
from cverify.storage import digest, sha256_file


def make_completed_fixture(folder):
    rng = np.random.default_rng(91)
    y = np.arange(60) % 2
    first = rng.normal(size=(60, 8))
    first[:, 0] += 2 * y
    np.savez(folder / "features.npz", generation_state=first, y_error=y)
    (folder / "manifest.json").write_text(json.dumps({"layers": [15]}))
    saved = {"configuration": {"seed": 42, "c": .1, "test_size": .25, "dev_size": .2},
             "split": {"train": list(range(40)), "dev": list(range(40, 50)), "test": list(range(50, 60))}}
    metrics_path = folder / "results_seed_42_original_layers" / "metrics.json"
    metrics_path.parent.mkdir()
    metrics_path.write_text(json.dumps(saved))
    source = folder / "combined_detectors"
    source.mkdir()
    code_dir = Path(combined_module.__file__).parent
    provenance = {"metrics_sha256": sha256_file(metrics_path), "manifest_sha256": sha256_file(folder / "manifest.json"),
                  "labels_or_features_sha256": sha256_file(folder / "features.npz"),
                  "code": {name: sha256_file(code_dir / name) for name in ("combined.py", "transfer.py", "train.py", "features.py")},
                  "c": .1, "bootstrap": 2, "seeds": [42, 43]}
    summary = {"provenance": provenance, "runs": {}}
    for seed in (42, 43):
        splits = make_splits(y, seed, saved["configuration"], saved["split"])
        probabilities = {"native_none": 1 / (1 + np.exp(-first[:, 0])),
                         "control_none_concat": 1 / (1 + np.exp(-(.8 * first[:, 0] + .2 * first[:, 1]))),
                         "ensemble_mean": 1 / (1 + np.exp(-(.9 * first[:, 0] + .1 * first[:, 2])))}
        record = {"seed": seed, "layers": [15], "methods": {}, "predictions": {},
                  "fingerprint": digest({"provenance": provenance, "seed": seed, "layers": [15]})}
        for split in ("dev", "test"):
            record["predictions"][split] = {"indices": splits[split].tolist(), "y_error": y[splits[split]].tolist(),
                                             "p_error": {name: p[splits[split]].tolist() for name, p in probabilities.items()}}
            for name, p in probabilities.items():
                record["methods"].setdefault(name, {})[split] = matched.metrics(y[splits[split]], p[splits[split]])
        key = f"seed_{seed}_all_layers"
        (source / f"{key}.json").write_text(json.dumps(record))
        summary["runs"][key] = {name: value for name, value in record.items() if name != "predictions"}
    (source / "share_combined_results.json").write_text(json.dumps(summary))
    return y, first, saved, summary


def test_only_first_pass_is_fitted_and_test_labels_do_not_enter_fit():
    rng = np.random.default_rng(4)
    y = np.arange(60) % 2
    first = rng.normal(size=(60, 6))
    first[:, 0] += y
    splits = {"train": np.arange(40), "dev": np.arange(40, 50), "test": np.arange(50, 60)}
    cached = {split: {name: np.full(len(rows), .65) for name in matched.SECOND}
              for split, rows in splits.items() if split != "train"}
    original = matched.evaluate_first(first, y, splits, cached, .1, 2, 42)
    changed = y.copy()
    changed[splits["test"]] = 1 - changed[splits["test"]]
    other = matched.evaluate_first(first, changed, splits, cached, .1, 2, 42)
    for split in ("dev", "test"):
        np.testing.assert_array_equal(original["predictions"][split]["p_error"]["first_pass_state"],
                                      other["predictions"][split]["p_error"]["first_pass_state"])
        for name in matched.SECOND:
            np.testing.assert_array_equal(original["predictions"][split]["p_error"][name], cached[split][name])


def test_matched_analysis_reuses_probabilities_and_resumes(tmp_path, monkeypatch):
    y, first, saved, combined = make_completed_fixture(tmp_path)
    original_fit, calls = matched.fit_source, []
    def counting_fit(*args):
        calls.append(1)
        return original_fit(*args)
    monkeypatch.setattr(matched, "fit_source", counting_fit)
    report = matched.analyze(tmp_path, tmp_path / "out", [42, 43], 2)
    assert len(calls) == 2
    assert set(report["seeds"]) == {"42", "43"}
    for seed, result in report["seeds"].items():
        assert result["methods"]["native_none"] == combined["runs"][f"seed_{seed}_all_layers"]["methods"]["native_none"]
    assert "accuracy05=" in (tmp_path / "out" / "summary.txt").read_text()
    assert "IMPACT ACROSS SPLIT SEEDS" in (tmp_path / "out" / "summary.txt").read_text()
    impact = (tmp_path / "out" / "thesis_impact.md").read_text()
    assert "| first_pass_state |" in impact
    assert "not an equal-compute comparison" in impact
    assert json.loads((tmp_path / "out" / "status.json").read_text())["phase"] == "complete"
    again = matched.analyze(tmp_path, tmp_path / "out", [42, 43], 2)
    assert again == report and len(calls) == 2
    matched.analyze(tmp_path, tmp_path / "out", [42, 43], 3)
    assert len(calls) == 4


@pytest.mark.parametrize("change,match", [("order", "ordering"), ("probability", "metrics"), ("fingerprint", "provenance")])
def test_rejects_misaligned_or_changed_cached_outputs(tmp_path, change, match):
    make_completed_fixture(tmp_path)
    path = tmp_path / "combined_detectors" / "seed_42_all_layers.json"
    record = json.loads(path.read_text())
    if change == "order":
        record["predictions"]["test"]["indices"].reverse()
    elif change == "probability":
        record["predictions"]["test"]["p_error"]["native_none"] = [.01] * 10
    else:
        record["fingerprint"] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=match):
        matched.analyze(tmp_path, tmp_path / "out", [42], 2)


def test_original_seed_first_pass_baseline_is_reproduced(tmp_path):
    y, first, saved, _ = make_completed_fixture(tmp_path)
    splits = make_splits(y, 42, saved["configuration"], saved["split"])
    base, calibrator = matched.fit_source(first, y, splits["train"], splits["dev"], .1)
    p = matched.probabilities(base, calibrator, first[splits["test"]])
    original = {"configuration": {"selected_layers": [15]}, "split": saved["split"],
                "models": {"generation_state": {"metrics": matched.metrics(y[splits["test"]], p)}}}
    path = tmp_path / "results_seed_42_all_layers" / "metrics.json"
    path.parent.mkdir()
    path.write_text(json.dumps(original))
    report = matched.analyze(tmp_path, tmp_path / "out", [42], 2)
    assert report["seeds"]["42"]["original_baseline_reproduced"]
    original["models"]["generation_state"]["metrics"]["auroc"] = -.5
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="reproduce"):
        matched.analyze(tmp_path, tmp_path / "out", [42], 2)
