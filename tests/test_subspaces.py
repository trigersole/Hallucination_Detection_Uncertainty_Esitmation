import json

import numpy as np
import pytest

from cverify.subspaces import principal_space, compare_spaces, analyze


def test_covariance_eigenvectors_match_svd():
    values = np.random.default_rng(7).normal(size=(24, 40))
    basis, summary = principal_space(values, 10)
    _, singular, vt = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)
    np.testing.assert_allclose(basis.T @ basis, np.eye(10), atol=1e-12)
    assert compare_spaces(basis, vt[:10].T)["subspace_overlap"] == pytest.approx(1)
    np.testing.assert_allclose(summary["covariance_eigenvalues"], singular[:10] ** 2 / 23)


def test_subspace_comparison_ignores_sign_and_rotation():
    basis = np.eye(20)[:, :10]
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(10, 10)))
    assert compare_spaces(basis, basis @ rotation)["subspace_overlap"] == pytest.approx(1)
    assert compare_spaces(basis, -basis)["subspace_overlap"] == pytest.approx(1)
    assert compare_spaces(basis, np.eye(20)[:, 10:])["subspace_overlap"] == pytest.approx(0)


def test_pca_centers_each_condition_and_rejects_insufficient_rank():
    values = np.random.default_rng(6).normal(size=(20, 30))
    basis, _ = principal_space(values, 10)
    shifted, _ = principal_space(values + 100, 10)
    assert compare_spaces(basis, shifted)["subspace_overlap"] == pytest.approx(1)
    with pytest.raises(ValueError, match="rank"):
        principal_space(np.ones((20, 30)), 10)


def test_analysis_uses_only_saved_training_rows(tmp_path):
    rng = np.random.default_rng(8)
    data = {name: rng.normal(size=(30, 12)) for name in
            ("generation_state", "none_state", "verify_state", "control_state")}
    data["y_error"] = np.arange(30) % 2
    np.savez(tmp_path / "features.npz", **data)
    (tmp_path / "manifest.json").write_text(json.dumps({"layers": [15]}))
    metrics = tmp_path / "metrics.json"
    saved = {"configuration": {}, "split": {"train": list(range(20)), "dev": list(range(20, 25)), "test": list(range(25, 30))}}
    metrics.write_text(json.dumps(saved))
    original = analyze(tmp_path, metrics, tmp_path / "first.json", 3)
    for key in data:
        if key != "y_error":
            data[key][20:] = np.nan
    np.savez(tmp_path / "features.npz", **data)
    changed = analyze(tmp_path, metrics, tmp_path / "second.json", 3)
    assert original["groups"] == changed["groups"]
    saved["split"]["test"][0] = 0
    metrics.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="partition"):
        analyze(tmp_path, metrics, tmp_path / "invalid.json", 3)
