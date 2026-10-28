import json

import numpy as np

from cverify import triviaqa_replication as replication
from cverify.features import load_run_features
from cverify.review import read_records


def test_replication_fits_target_with_source_settings_and_resumes(tmp_path, monkeypatch):
    full = tmp_path / "full"
    full.mkdir()
    np.savez(full / "features.npz", y_error=np.arange(40) % 2)
    (full / "manifest.json").write_text(json.dumps({"layers": [-1, -4, -8, 15]}))
    (full / "metadata.jsonl").write_text("{}\n" * 40)
    training, combined, matched = [], [], []
    def train(args):
        assert args.run == str(full) and args.c == .17 and args.seed == 42
        assert args.test_size == .25 and args.dev_size == .1633986928
        assert args.bootstrap == 1000
        training.append(args.layers)
        from pathlib import Path
        output = Path(args.output)
        output.mkdir(parents=True)
        (output / "metrics.json").write_text("{}")
    def combine(run, metrics, output, seeds, repetitions):
        assert run == full and seeds == [42, 43, 44, 45, 46] and repetitions == 1000
        combined.append(run)
    def match(run, output, seeds, repetitions):
        assert run == full and seeds == [42, 43, 44, 45, 46] and repetitions == 1000
        output.mkdir(exist_ok=True)
        (output / "summary.txt").write_text("matched target results")
        matched.append(run)
        return {"target": True}
    monkeypatch.setattr(replication, "run_training", train)
    monkeypatch.setattr(replication, "analyze_combined", combine)
    monkeypatch.setattr(replication, "analyze_matched", match)
    config = {"run": str(tmp_path), "external_protocol": "within dataset", "dataset_export": {"n": 40}}
    settings = {"c": .17, "test_size": .25, "dev_size": .1633986928}
    for _ in range(2):
        replication.finalize_replication(config, full, settings, [{"answer": "Paris"}] * 40)
    assert training == [[-1, -4, -8], [15], [-1, -4, -8, 15]]
    assert len(combined) == len(matched) == 2
    status = json.loads((tmp_path / "finalize_status.json").read_text())
    assert status["phase"] == "complete" and status["analysis_mode"] == "within_dataset"
    report = json.loads((tmp_path / "dataset_replication" / "report.json").read_text())
    assert report["counts"]["evaluated"] == 40 and report["counts"]["quality_excluded"] == 0
    assert report["matched_comparison"] == {"target": True}


def test_review_gate_preserves_judgments_and_keeps_em_supplementary(tmp_path, monkeypatch):
    full = tmp_path / "full"
    full.mkdir()
    np.savez(full / "features.npz", y_error=np.full(26, -1), generation_state=np.arange(52).reshape(26, 2))
    (full / "manifest.json").write_text(json.dumps({"layers": [15], "labeler": "deferred"}))
    rows = [{"id": str(i), "question": f"Question {i}?", "aliases": ["paris"],
             "answer": "The capital is Paris." if i < 12 else "Rome",
             "is_correct": None, "generation_details": [{"stop_reason": "eos"}]} for i in range(26)]
    rows[19]["generation_details"][0]["stop_reason"] = "length"
    (full / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    config = {"run": str(tmp_path), "label_protocol": "reviewed_factual_correctness"}
    calls = []
    def analyze(received, audited, settings, kept_rows):
        data, manifest = load_run_features(audited)
        assert data["y_error"].tolist() == [0] * 12 + [1] * 9
        assert not (audited / "features.npz").exists()  # Indexed view, no large copy.
        assert manifest["audit"]["counts"]["quality_excluded"] == 1
        assert len(kept_rows) == 21
        calls.append(audited)
    monkeypatch.setattr(replication, "finalize_replication", analyze)
    replication.finalize_reviewed_replication(config, full, {}, rows)
    status = json.loads((tmp_path / "finalize_status.json").read_text())
    assert status["phase"] == "needs_review" and not calls
    review = tmp_path / "review.jsonl"
    reviews = read_records(review)
    assert all(row["judgment"] is None and "exact_match" not in row for row in reviews)
    supplementary = json.loads((tmp_path / "supplementary_exact_match.json").read_text())
    assert supplementary["exact_match_count"] == 0  # Correct prose fails EM, but remains correct after review.
    for i, row in enumerate(reviews):
        row.update(judgment="correct" if i < 12 else "incorrect", reviewer="reviewer1")
    for i, label in zip(range(22, 26), ["ambiguous", "ambiguous", "refusal", "exclude"]):
        reviews[i]["judgment"] = label
    review.write_text("\n".join(json.dumps(row) for row in reviews))
    complete_text = review.read_text()
    replication.finalize_reviewed_replication(config, full, {}, rows)
    assert len(calls) == 1 and review.read_text() == complete_text
    # A changed answer invalidates the old review rather than reusing its label.
    rows[0]["answer"] = "changed answer"
    (full / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    replication.finalize_reviewed_replication(config, full, {}, rows)
    assert len(calls) == 1
    assert json.loads((tmp_path / "finalize_status.json").read_text())["phase"] == "needs_review"


def test_reviewed_replication_runs_real_training_through_matched_results(tmp_path, monkeypatch):
    from cverify import combined, matched_passes, train
    # Exercise actual fitting and provenance; reduce only resampling cost.
    original_comparison = combined.paired_comparison
    def fast_comparison(y, p, baseline, repetitions, seed):
        return original_comparison(y, p, baseline, 2, seed)
    monkeypatch.setattr(combined, "paired_comparison", fast_comparison)
    monkeypatch.setattr(matched_passes, "paired_comparison", fast_comparison)
    monkeypatch.setattr(train, "_bootstrap", lambda *args: {})
    full = tmp_path / "full"
    full.mkdir()
    rng = np.random.default_rng(42)
    y = np.arange(48) % 2
    states = {name: rng.normal(size=(48, 8)) + y[:, None] for name in
              ("generation_state", "none_state", "verify_state", "control_state")}
    states.update(generation=rng.normal(size=(48, 3)), y_error=np.full(48, -1),
                  difference_state=states["verify_state"] - states["control_state"],
                  verify_none_difference_state=states["verify_state"] - states["none_state"],
                  control_none_difference_state=states["control_state"] - states["none_state"])
    np.savez(full / "features.npz", **states)
    (full / "manifest.json").write_text(json.dumps({"layers": [-1, -4, -8, 15], "labeler": "deferred"}))
    rows = [{"id": str(i), "question": f"Question {i}", "answer": "Paris is the capital." if label == 0 else "Rome",
             "aliases": ["paris"], "is_correct": None, "generation_details": [{"stop_reason": "eos"}]}
            for i, label in enumerate(y)]
    (full / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    config = {"run": str(tmp_path), "label_protocol": "reviewed_factual_correctness",
              "external_protocol": "reviewed replication", "dataset_export": {"n": 48}}
    settings = {"c": .1, "test_size": .25, "dev_size": .1633986928}
    replication.finalize_reviewed_replication(config, full, settings, rows)
    review = tmp_path / "review.jsonl"
    reviews = read_records(review)
    for row, label in zip(reviews, y):
        row.update(judgment="correct" if label == 0 else "incorrect", reviewer="reviewer1")
    review.write_text("\n".join(json.dumps(row) for row in reviews))
    replication.finalize_reviewed_replication(config, full, settings, rows)
    status = json.loads((tmp_path / "finalize_status.json").read_text())
    assert status["phase"] == "complete" and status["review_hash"]
    assert status["label_protocol"] == "reviewed_factual_correctness"
    report = json.loads((tmp_path / "dataset_replication" / "report.json").read_text())
    assert report["counts"]["correct"] == report["counts"]["incorrect"] == 24
    assert len(report["matched_comparison"]["seeds"]) == 5
    assert report["matched_comparison"]["seeds"]["42"]["original_baseline_reproduced"]
