"""Merge independently extracted SLURM shards."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def run_merge(args) -> None:
    runs = [Path(path) for path in args.runs]
    if len(runs) < 2:
        raise ValueError("Provide at least two shard directories")
    manifests = [json.loads((run / "manifest.json").read_text(encoding="utf-8")) for run in runs]
    comparable = ("model", "data", "samples", "temperature", "layers", "labeler", "label_threshold")
    for key in comparable:
        values = [manifest.get(key) for manifest in manifests]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Shard mismatch for {key}: {values}")
    loaded = [np.load(run / "features.npz") for run in runs]
    keys = set(loaded[0].files)
    if any(set(item.files) != keys for item in loaded[1:]):
        raise ValueError("Feature files do not contain identical array names")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    np.savez_compressed(output / "features.npz", **arrays)
    seen_ids: set[str] = set()
    metadata = []
    for run in runs:
        with (run / "metadata.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] in seen_ids:
                    raise ValueError(f"Duplicate question id across shards: {row['id']}")
                seen_ids.add(row["id"])
                metadata.append(row)
    if len(metadata) != len(arrays["y_error"]):
        raise ValueError("Metadata and feature row counts differ")
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = dict(manifests[0])
    manifest.update({
        "examples": len(metadata), "offset": 0,
        "merged_from": [os.fspath(run.resolve()) for run in runs],
        "approximate_processed_label_and_generated_tokens": sum(
            item.get("approximate_processed_label_and_generated_tokens", 0) for item in manifests
        ),
    })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Merged {len(runs)} shards and {len(metadata)} examples into {output}")
