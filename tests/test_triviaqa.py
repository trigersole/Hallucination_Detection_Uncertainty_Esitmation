import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cverify.labeling import build_labeler
from cverify.model import CausalLMAdapter
from cverify.storage import sha256_file
from cverify.triviaqa_data import normalize_triviaqa, select_subset, export_subset
from cverify.triviaqa_workflow import (prepare_workflow, model_matrices, external_predictions, finalize_external,
                                       source_snapshot, validate_target_protocol, safe_metrics)
from cverify.transfer import fit_source
from cverify.combined import probabilities


def test_official_normalization_is_whole_answer_not_factual_substring():
    scorer = build_labeler("triviaqa-exact", bleurt_model=None, device="cpu")
    assert normalize_triviaqa("The_United ‘States’!!!") == "united states"
    assert scorer.score("The Paris!", ("paris",)) == 1
    assert scorer.score("Paris is not the answer.", ("paris",)) == 0
    assert scorer.score("The answer is Paris.", ("paris",)) == 0
    assert scorer.score("Straße", ("strasse",)) == 0  # Official lower(), not casefold().
    assert scorer.score("Beyoncé", ("beyonce",)) == 0  # No extra Unicode accent removal.


def dataset_rows(n=30):
    return [{"question_id": str(i), "question": f"Trivia question number {i}?",
             "answer": {"normalized_aliases": ["paris"], "aliases": ["Paris"], "value": "Paris"}}
            for i in range(n)]


def test_subset_is_deterministic_deduplicated_and_excludes_source_questions():
    rows = dataset_rows()
    rows.append({**rows[0], "question_id": "duplicate-text"})
    first, counts = select_subset(rows, 10, 42, [rows[1]["question"]])
    second, _ = select_subset(list(reversed(rows)), 10, 42, [rows[1]["question"]])
    assert first == second
    assert counts["overlap_questions_removed"] == 1
    assert counts["duplicate_question_texts_removed"] == 1
    assert rows[1]["question"] not in [row["question"] for row in first]
    with pytest.raises(ValueError, match="available"):
        select_subset(rows, 100, 42, [])


def test_export_pins_dataset_revision_and_preserves_selected_ids(tmp_path, monkeypatch):
    class Dataset(list):
        _fingerprint = "fixture-fingerprint"
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return Dataset(dataset_rows())
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=lambda name: SimpleNamespace(sha="immutable-sha"))))
    path = tmp_path / "triviaqa.jsonl"
    metadata = export_subset(path, 10, 42, [])
    assert metadata["dataset_revision"] == "immutable-sha"
    assert calls[0][1] == {"split": "validation", "revision": "immutable-sha"}
    assert export_subset(path, 10, 42, []) == metadata
    assert len(calls) == 1
    with pytest.raises(ValueError, match="changed"):
        export_subset(path, 11, 42, [])


def test_model_and_tokenizer_receive_same_pinned_revision(monkeypatch):
    calls = []
    tokenizer = SimpleNamespace(chat_template="fixture", pad_token_id=1)
    model = SimpleNamespace(eval=lambda: None)
    def token_load(name, **kwargs):
        calls.append(kwargs)
        return tokenizer
    def model_load(name, **kwargs):
        calls.append(kwargs)
        return model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=token_load), AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_load)))
    CausalLMAdapter("fixture", revision="pinned", chat_format="tokenizer")
    assert [call["revision"] for call in calls] == ["pinned", "pinned"]


def source_fixture(tmp_path):
    rng = np.random.default_rng(19)
    source = tmp_path / "source"
    source.mkdir()
    y = np.arange(80) % 2
    data = {key: rng.normal(size=(80, 8)) for key in ("generation_state", "none_state", "verify_state", "control_state")}
    for values in data.values():
        values[:, 0] += 2 * y
    data["y_error"] = y
    np.savez(source / "features.npz", **data)
    from cverify.prompts import evaluation_spec
    manifest = {"model": "fixture-model", "layers": [-1, -4, -8, 15], "generation_prompt": "concise",
                "evaluation": evaluation_spec("assessment-v3"), "chat_format": "tokenizer", "chat_template": "fixture",
                "samples": 5, "temperature": .7, "seed": 42, "max_new_tokens": 256, "max_retry_tokens": 1024,
                "generation_state_position": "last_non_eos_generated_token", "resolved_layers": [32, 29, 25, 15],
                "runtime": {"model_commit": "pinned-source"}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    splits = {"train": np.arange(50), "dev": np.arange(50, 64), "test": np.arange(64, 80)}
    saved = {"configuration": {"seed": 42, "c": .1, "test_size": .2, "dev_size": .2, "selected_layers": manifest["layers"]},
             "split": {name: rows.tolist() for name, rows in splits.items()}, "models": {}}
    for name, x in model_matrices(data).items():
        base, cal = fit_source(x, y, splits["train"], splits["dev"], .1)
        p = probabilities(base, cal, x[splits["test"]])
        from cverify.triviaqa_workflow import MODELS
        saved["models"][MODELS[name]] = {"metrics": safe_metrics(y[splits["test"]], p)}
    for group in ("original_layers", "all_layers"):
        path = source / f"results_seed_42_{group}"
        path.mkdir()
        (path / "metrics.json").write_text(json.dumps(saved))
    return source, data, manifest, saved, splits


def test_external_detectors_fit_only_source_and_reuse_caches(tmp_path, monkeypatch):
    source, data, manifest, saved, splits = source_fixture(tmp_path)
    target = {key: values[:20].copy() for key, values in data.items()}
    from cverify import triviaqa_workflow as external
    original, calls = external.fit_source, []
    def spy(values, labels, train, dev, c):
        assert len(values) == 80 and len(labels) == 80
        assert np.array_equal(labels, data["y_error"])
        calls.append(1)
        return original(values, labels, train, dev, c)
    monkeypatch.setattr(external, "fit_source", spy)
    p, checks = external_predictions(data, target, splits, .1, tmp_path / "models", "same", saved)
    assert len(calls) == 5 and len(p) == 6
    target["y_error"] = 1 - target["y_error"]
    other, _ = external_predictions(data, target, splits, .1, tmp_path / "models", "same", saved)
    assert len(calls) == 5
    for name in p:
        np.testing.assert_array_equal(p[name], other[name])
    validate_target_protocol(manifest, {**manifest, "labeler": "triviaqa-exact"})
    with pytest.raises(ValueError, match="runtime"):
        validate_target_protocol(manifest, {**manifest, "runtime": {"model_commit": "changed"}, "labeler": "triviaqa-exact"})


@pytest.mark.parametrize("mode", ["frozen", "within_dataset"])
def test_external_finalization_uses_all_official_scores_without_manual_gate(tmp_path, monkeypatch, mode):
    source, data, manifest, saved, splits = source_fixture(tmp_path)
    run, full = tmp_path / "external", tmp_path / "external" / "full"
    full.mkdir(parents=True)
    target = {key: values[:20].copy() for key, values in data.items()}
    np.savez(full / "features.npz", **target)
    (full / "manifest.json").write_text(json.dumps({**manifest, "labeler": "triviaqa-exact"}))
    metadata, questions = [], []
    for i, label in enumerate(target["y_error"]):
        metadata.append({"id": str(i), "question": f"Target question {i}", "aliases": ["paris"],
                         "answer": "Paris" if label == 0 else "Rome", "is_correct": bool(label == 0),
                         "generation_details": [{"stop_reason": "eos"}]})
        questions.append({"id": str(i), "question": f"Target question {i}", "answers": ["paris"]})
    (full / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in metadata))
    dataset = run / "triviaqa.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in questions))
    config = {"run": str(run), "source_run": str(source), "source_snapshot": source_snapshot(source), "examples": 20,
              "data": str(dataset), "dataset_export": {"label_protocol": "official_whole_answer_normalized_exact_match"},
              "code_sha256": "fixture", "external_protocol": "frozen fixture"}
    import cverify.triviaqa_workflow as external
    if mode == "within_dataset":
        config["analysis_mode"] = mode
        calls = []
        def replicate(received, received_full, settings, received_rows):
            assert received_full == full and settings == saved["configuration"]
            assert len(received_rows) == 20
            calls.append(1)
            from cverify.storage import atomic_json
            atomic_json(run / "finalize_status.json", {"phase": "complete"})
        monkeypatch.setattr("cverify.triviaqa_replication.finalize_replication", replicate)
    actual = external.paired_comparison
    monkeypatch.setattr(external, "paired_comparison", lambda y, a, b, reps, seed: actual(y, a, b, 2, seed))
    finalize_external(config, full)
    assert json.loads((run / "finalize_status.json").read_text())["phase"] == "complete"
    if mode == "frozen":
        report = json.loads((run / "external_validation" / "report.json").read_text())
        assert report["counts"]["evaluated"] == 20 and report["counts"]["quality_excluded"] == 0
        assert len(report["methods"]) == 6
    else:
        assert calls == [1] and not (run / "external_validation").exists()
    assert not (run / "review.jsonl").exists()
    metadata[0]["answer"] = "Definitely not Paris"
    (full / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in metadata))
    with pytest.raises(ValueError, match="official scores"):
        finalize_external(config, full)
    assert json.loads((run / "finalize_status.json").read_text())["phase"] != "complete"


def test_safe_metrics_explains_single_class_without_fabricating_auc():
    result = safe_metrics(np.ones(10), np.full(10, .7))
    assert result["auroc"] is None
    assert result["brier"] == pytest.approx(.09)


def test_deferred_triviaqa_finalization_stops_for_factual_review(tmp_path):
    source, data, manifest, saved, splits = source_fixture(tmp_path)
    run, full = tmp_path / "reviewed", tmp_path / "reviewed" / "full"
    full.mkdir(parents=True)
    target = {key: values[:20].copy() for key, values in data.items()}
    target["y_error"] = np.full(20, -1)
    np.savez(full / "features.npz", **target)
    (full / "manifest.json").write_text(json.dumps({**manifest, "labeler": "deferred"}))
    rows = [{"id": str(i), "question": f"Question {i}", "answer": "The capital is Paris.",
             "aliases": ["paris"], "is_correct": None, "generation_details": [{"stop_reason": "eos"}]}
            for i in range(20)]
    (full / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    dataset = run / "triviaqa.jsonl"
    dataset.write_text("\n".join(json.dumps({"id": row["id"], "question": row["question"], "answers": row["aliases"]})
                                 for row in rows))
    config = {"run": str(run), "source_run": str(source), "source_snapshot": source_snapshot(source),
              "examples": 20, "data": str(dataset), "analysis_mode": "within_dataset",
              "label_protocol": "reviewed_factual_correctness"}
    finalize_external(config, full)
    status = json.loads((run / "finalize_status.json").read_text())
    assert status["phase"] == "needs_review" and status["review_hash"]
    reviews = [json.loads(line) for line in (run / "review.jsonl").read_text().splitlines()]
    assert len(reviews) == 20 and all(row["judgment"] is None for row in reviews)
    assert not (run / "dataset_replication" / "summary.txt").exists()


@pytest.mark.parametrize("mode", ["frozen", "within_dataset", "within_reviewed"])
def test_workflow_prepares_separate_run_and_uses_automatic_pinned_extraction(tmp_path, monkeypatch, mode):
    source, data, manifest, saved, splits = source_fixture(tmp_path)
    project = tmp_path / "project"
    script = project / "scripts" / "run_triviaqa.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    truth = tmp_path / "truthfulqa"
    truth.mkdir()
    truth_data = truth / "questions.jsonl"
    truth_data.write_text("\n".join(json.dumps({"id": str(i), "question": f"Source question {i}", "answers": ["Paris"]}) for i in range(80)))
    (truth / "workflow_config.json").write_text(json.dumps({"data": str(truth_data), "data_sha256": sha256_file(truth_data)}))
    (truth / "finalize_status.json").write_text(json.dumps({"phase": "complete", "results": str(source)}))
    monkeypatch.setenv("CVERIFY_TRUTHFULQA_RUN", str(truth))
    monkeypatch.setenv("CVERIFY_BENCHMARK", "triviaqa")
    monkeypatch.setenv("CVERIFY_PIPELINE_RUN", "triviaqa_test")
    monkeypatch.setenv("CVERIFY_TRIVIAQA_SIZE", "6")
    analysis = "within_dataset" if mode == "within_reviewed" else mode
    labeler = "deferred" if mode == "within_reviewed" else "triviaqa-exact"
    monkeypatch.setenv("CVERIFY_TRIVIAQA_ANALYSIS", analysis)
    monkeypatch.setenv("CVERIFY_TRIVIAQA_LABELS", "reviewed" if mode == "within_reviewed" else "exact_match")
    def export(path, size, seed, excluded):
        assert size == 6 and seed == 42 and len(excluded) == 80
        path.write_text("\n".join(json.dumps({"id": f"t{i}", "question": f"Target question {i}", "answers": ["paris"]}) for i in range(size)))
        return {"n": size, "seed": seed, "jsonl_sha256": sha256_file(path)}
    monkeypatch.setattr("cverify.triviaqa_workflow.export_subset", export)
    from cverify import workflow
    path, config = workflow.prepare_config(script)
    assert config["model_revision"] == "pinned-source" and config["labeler"] == labeler
    assert config["source_run"] == str(source)
    assert path.parent == project / "runs" / "triviaqa_test"
    assert config["cpus_per_task"] == 1 and config["max_submitted_jobs"] == 2
    assert config.get("analysis_mode", "frozen") == analysis
    if mode == "within_reviewed":
        assert config["label_protocol"] == "reviewed_factual_correctness"
    captured = []
    monkeypatch.setattr(workflow, "run_extraction", lambda args: captured.append(args))
    workflow.extract_worker(config, 0)
    assert captured[0].model_revision == "pinned-source"
    assert captured[0].labeler == labeler
    assert captured[0].layers == [-1, -4, -8, 15]
    assert captured[0].expected_runtime == manifest["runtime"]


def test_runtime_mismatch_stops_before_any_target_generation(tmp_path, monkeypatch):
    from cverify import extract
    from cverify.cli import build_parser
    class Adapter:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="actual-model", num_hidden_layers=32))
            self.tokenizer = SimpleNamespace(chat_template="fixture")
        def generate_complete(self, *args, **kwargs):
            raise AssertionError("Target generation must not start")
    monkeypatch.setattr(extract, "CausalLMAdapter", Adapter)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"id": "q", "question": "Q?", "answers": ["a"]}))
    args = build_parser().parse_args(["extract", "--data", str(dataset), "--model", "fixture", "--output", str(tmp_path / "out"),
                                     "--generation-prompt", "concise", "--evaluation-profile", "assessment-v3", "--chat-format", "tokenizer",
                                     "--labeler", "triviaqa-exact", "--resume"])
    args.expected_runtime = {"model_commit": "different"}
    with pytest.raises(ValueError, match="before generating"):
        extract.run_extraction(args)
