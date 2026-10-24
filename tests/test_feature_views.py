import errno
import json
from pathlib import Path

import numpy as np
import pytest

from cverify.cli import build_parser
from cverify.features import load_run_features
from cverify.review import apply_review, export_review, read_records
from cverify.storage import atomic_json, atomic_npz
from cverify.train import run_training


def make_reviewed_view(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    n = 40
    rng = np.random.default_rng(12)
    verify, control, none = (rng.normal(size=(n, 8)) for _ in range(3))
    np.savez_compressed(source / "features.npz", y_error=np.full(n, -1), generation=rng.normal(size=(n, 3)),
                        generation_state=rng.normal(size=(n, 8)), verify_state=verify, control_state=control,
                        none_state=none, concat_state=np.c_[verify, control], difference_state=verify - control,
                        verify_none_difference_state=verify - none, control_none_difference_state=control - none)
    rows = [{"id": f"q{i}", "question": f"Q{i}?", "answer": "A.", "aliases": ["A."],
             "generation_details": [{"stop_reason": "eos"}]} for i in range(n)]
    (source / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    atomic_json(source / "manifest.json", {"examples": n, "layers": [-1, -4, -8, 15]})
    review = tmp_path / "review.jsonl"
    export_review(source, review)
    judgments = read_records(review)
    for i, row in enumerate(judgments):
        row.update(judgment="correct" if i % 2 else "incorrect", reviewer="fixture")
    judgments[0]["judgment"] = "ambiguous"
    review.write_text("\n".join(json.dumps(row) for row in judgments))
    output = tmp_path / "audited"
    apply_review(source, review, output)
    return source, output


def test_indexed_review_reuses_source_and_matches_physical_archive_training(tmp_path):
    source, view = make_reviewed_view(tmp_path)
    assert not (view / "features.npz").exists()
    assert (view / "labels.npz").stat().st_size < 2048
    data, manifest = load_run_features(view)
    assert manifest["feature_storage"] == "indexed_view_v1"
    with np.load(source / "features.npz") as raw:
        for key in raw.files:
            if key != "y_error":
                np.testing.assert_array_equal(data[key], raw[key][1:])
    np.testing.assert_array_equal(data["y_error"], [i % 2 == 0 for i in range(1, 40)])
    physical = tmp_path / "physical"
    physical.mkdir()
    np.savez(physical / "features.npz", **data)
    atomic_json(physical / "manifest.json", {"examples": 39, "layers": [-1, -4, -8, 15]})
    reports = []
    for name, run in (("view", view), ("physical", physical)):
        output = tmp_path / f"results_{name}"
        run_training(build_parser().parse_args([
            "train", "--run", str(run), "--output", str(output), "--layers", "15", "--bootstrap", "3",
        ]))
        reports.append(json.loads((output / "metrics.json").read_text()))
    assert reports[0]["split"] == reports[1]["split"]
    assert reports[0]["models"] == reports[1]["models"]


def test_view_rejects_modified_source_archive(tmp_path):
    source, view = make_reviewed_view(tmp_path)
    with (source / "features.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Source features.npz changed"):
        load_run_features(view)


def test_atomic_npz_quota_failure_discards_partial_and_preserves_target(tmp_path, monkeypatch):
    from cverify import storage
    target = tmp_path / "features.npz"
    atomic_npz(target, values=np.arange(4))
    original = target.read_bytes()
    def fail(handle, **kwargs):
        handle.write(b"partial-feature-copy")
        raise OSError(122, "Disk quota exceeded")
    monkeypatch.setattr(storage.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="Disk quota"):
        atomic_npz(target, values=np.arange(5))
    assert target.read_bytes() == original
    assert not (tmp_path / "features.npz.tmp").exists()


def test_atomic_json_space_failure_preserves_existing_file(tmp_path, monkeypatch):
    from cverify import storage
    target = tmp_path / "state.json"
    atomic_json(target, {"original": True})
    def fail(fd):
        raise OSError(errno.ENOSPC, "No space left")
    monkeypatch.setattr(storage.os, "fsync", fail)
    with pytest.raises(OSError, match="No space"):
        atomic_json(target, {"original": False})
    assert json.loads(target.read_text()) == {"original": True}
    assert not (tmp_path / "state.json.tmp").exists()


@pytest.mark.parametrize("code", [errno.EDQUOT, errno.ENOSPC])
def test_quota_failure_reports_storage_blocked_and_exit_75(tmp_path, monkeypatch, code):
    from cverify import workflow
    (tmp_path / "review.jsonl").write_text("review")
    def fail(config):
        raise OSError(code, "Disk quota exceeded")
    monkeypatch.setattr(workflow, "finalize_worker", fail)
    with pytest.raises(SystemExit) as error:
        workflow.finalize_once({"run": str(tmp_path)})
    assert error.value.code == 75
    assert json.loads((tmp_path / "finalize_status.json").read_text())["phase"] == "storage_blocked"


def test_controller_stops_after_new_quota_failure_instead_of_retrying(tmp_path, monkeypatch):
    from cverify import workflow
    monkeypatch.setattr(workflow, "shard_complete", lambda config, index: True)
    class Scheduler:
        submitted = 0
        def live_jobs(self): return set()
        def state(self, job): return "FAILED"
        def exit_code(self, job): return 75
        def submit(self, config, path, kind, index):
            self.submitted += 1
            return str(self.submitted)
    scheduler = Scheduler()
    result = workflow.controller(tmp_path / "config.json", {"run": str(tmp_path), "examples": 1, "shard_size": 1},
                                 scheduler=scheduler, poll_seconds=0, max_cycles=3)
    assert result == 75 and scheduler.submitted == 1
    assert json.loads((tmp_path / "workflow_state.json").read_text())["phase"] == "storage_blocked"
