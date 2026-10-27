import csv
import json

import numpy as np
import pytest

from cverify.pass_geometry import analyze, direction_correctness, pair_geometry
from cverify.subspaces import principal_space


def test_mean_shift_is_separate_from_centered_subspace():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(40, 12))
    shift = np.arange(12) / 10
    z = x + shift
    train = np.arange(30)
    a, _ = principal_space(x[train], 5)
    b, _ = principal_space(z[train], 5)
    result = pair_geometry(x, z, a, b, train)
    assert result["subspace_overlap"] == pytest.approx(1)
    assert result["train_mean_shift_norm"] == pytest.approx(np.linalg.norm(shift))
    assert result["train_mean_paired_distance"] == pytest.approx(np.linalg.norm(shift))


def test_correctness_orientation_uses_training_only():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    scores = np.array([-.2, -.1, .1, .2, .2, .1, -.1, -.2]).reshape(-1, 1)
    result = direction_correctness(scores, y, np.arange(4), np.arange(4, 8))[0]
    assert result["orientation_chosen_on_train"] == 1
    assert result["train_auroc_oriented"] == 1
    assert result["test_auroc_same_orientation"] == 0


def test_first_pass_included_and_held_out_does_not_change_directions(tmp_path):
    rng = np.random.default_rng(10)
    arrays = {f"{name}_state": rng.normal(size=(40, 12)) for name in ("generation", "none", "verify", "control")}
    arrays["y_error"] = np.arange(40) % 2
    np.savez(tmp_path / "features.npz", **arrays)
    (tmp_path / "manifest.json").write_text(json.dumps({"layers": [15], "evaluation": {"hidden_position": "Assessment:"},
                                                        "generation_state_position": "last_non_eos_generated_token"}))
    saved = {"configuration": {}, "split": {"train": list(range(24)), "dev": list(range(24, 30)), "test": list(range(30, 40))}}
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(saved))
    first = analyze(tmp_path, metrics, tmp_path / "first", 3)
    group = first["groups"]["layer_15"]
    assert len(group["pairs"]) == 6
    assert "generation_vs_none" in group["pairs"]
    assert len(group["conditions"]["generation"]["correctness_along_own_components"]) == 3
    with (tmp_path / "first" / "layer_15_projections.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40 * 4
    assert "own_pc_1" in rows[0] and "first_pass_pc_1" in rows[0]
    assert len([row for row in rows if row["split"] == "train"]) == 24 * 4
    for key, array in arrays.items():
        if key != "y_error":
            array[24:] += rng.normal(size=(16, 12)) * 100
    np.savez(tmp_path / "features.npz", **arrays)
    second = analyze(tmp_path, metrics, tmp_path / "second", 3)
    assert group["pairs"] == second["groups"]["layer_15"]["pairs"]
    assert group["conditions"]["generation"]["covariance_eigenvalues"] == second["groups"]["layer_15"]["conditions"]["generation"]["covariance_eigenvalues"]
    # A matching checkpoint must also have its projections; regenerate a missing CSV.
    (tmp_path / "second" / "layer_15_projections.csv").unlink()
    analyze(tmp_path, metrics, tmp_path / "second", 3)
    assert (tmp_path / "second" / "layer_15_projections.csv").exists()
    saved["split"]["test"][0] = 0
    metrics.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="partition"):
        analyze(tmp_path, metrics, tmp_path / "bad", 3)
