"""Read physical feature archives or audited row views without duplicating disk data."""
import json
from pathlib import Path

import numpy as np

from .storage import sha256_file


def load_run_features(run: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("feature_storage") != "indexed_view_v1":
        with np.load(run / "features.npz", allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}, manifest
    view = manifest["feature_view"]
    source = Path(view["source_run"])
    if source.resolve() == run.resolve():
        raise ValueError("An indexed feature view cannot reference itself")
    for name, checksum in view["source_sha256"].items():
        if sha256_file(source / name) != checksum:
            raise ValueError(f"Source {name} changed since correctness review; refusing mismatched features")
    with np.load(run / "labels.npz", allow_pickle=False) as labels:
        indices = labels["source_indices"]
        y = labels["y_error"]
    n = view["source_examples"]
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n)
            or y.shape != indices.shape or not np.isin(y, [0, 1]).all() or len(indices) != manifest["examples"]):
        raise ValueError("Invalid row indices or correctness labels in the reviewed feature view")
    with np.load(source / "features.npz", allow_pickle=False) as archive:
        data = {}
        for key in archive.files:
            if key == "y_error":
                continue
            values = archive[key]
            if values.ndim < 1 or len(values) != n:
                raise ValueError(f"Source row count differs for feature {key}")
            data[key] = values[indices]
    data["y_error"] = y.astype(np.int64)
    return data, manifest
