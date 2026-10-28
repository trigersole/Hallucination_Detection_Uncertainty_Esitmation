import json

import numpy as np

from cverify import triviaqa_replication as replication


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
