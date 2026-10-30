import json

import numpy as np
import pytest

from cverify import attribution, followup
from cverify.combined import probabilities, uncertainty_metrics
from cverify.prompts import evaluation_spec
from cverify.storage import atomic_json
from cverify.transfer import fit_source


def fixture_run(folder, prefix="source"):
    folder.mkdir()
    rng = np.random.default_rng(81)
    y = np.arange(80) % 2
    data = {name: rng.normal(size=(80, 8)).astype(np.float32) + y[:, None] for name in
            ("generation_state", "none_state", "verify_state", "control_state")}
    data.update(y_error=y, generation=rng.normal(size=(80, 3)))
    np.savez(folder / "features.npz", **data)
    manifest = {"layers": [-1, -4, -8, 15], "model": "fixture", "chat_format": "tokenizer", "generation_prompt": "concise",
                "labeler": "audited_labels", "audit": {"reviewed": True}, "evaluation": evaluation_spec("assessment-v3"),
                "runtime": {"model_commit": "pinned"}}
    atomic_json(folder / "manifest.json", manifest)
    rows = [{"id": f"{prefix}-{i}", "question": f"{prefix} question number {i}?", "answer": f"original answer {i}",
             "aliases": [f"reference answer {i}"], "incorrect_references": [], "is_correct": bool(label == 0)}
            for i, label in enumerate(y)]
    (folder / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    saved = {"configuration": {"seed": 42, "c": .1, "test_size": .25, "dev_size": .1633986928,
                               "selected_layers": manifest["layers"], "audit": manifest["audit"]},
             "split": {"train": list(range(50)), "dev": list(range(50, 60)), "test": list(range(60, 80))}, "models": {}}
    x = followup.detail_matrices(data)
    for name, key in (("first_pass_state", "generation_state"), ("native_none", "none_state"),
                      ("native_verify", "verify_state"), ("native_control", "control_state"),
                      ("control_none_concat", "control_none_concat_state")):
        base, cal = fit_source(x[name], y, np.arange(50), np.arange(50, 60), .1)
        p = probabilities(base, cal, x[name][60:])
        saved["models"][key] = {"metrics": uncertainty_metrics(y[60:], p)}
    output = folder / "results_seed_42_all_layers"
    output.mkdir()
    atomic_json(output / "metrics.json", saved)
    return data, manifest, rows, saved


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def prompt_hidden(self, prompt, layers):
        self.calls.append(prompt)
        return {layer: np.array([len(prompt), sum(map(ord, prompt)) % 97 + j], dtype=np.float32)
                for j, layer in enumerate(layers)}, len(prompt)


def test_question_prompts_omit_answer_and_keep_tasks_and_cue():
    prompts = attribution.question_prompts("Question text?")
    assert set(prompts) == {"first", "none", "verify", "control"}
    assert prompts["first"].startswith("Answer the question concisely.")
    assert all("Proposed answer:" not in prompt for prompt in prompts.values())
    for name in ("none", "verify", "control"):
        assert prompts[name].endswith("Assessment:")
    assert "factual correctness" in prompts["verify"] and "formatting clarity" in prompts["control"]


def test_question_extraction_resumes_and_scores_aligned_features(tmp_path, monkeypatch):
    run = tmp_path / "run"
    data, _, rows, _ = fixture_run(run)
    fake = FakeAdapter()
    monkeypatch.setattr(attribution, "adapter_for", lambda manifest: fake)
    extracted = tmp_path / "question"
    attribution.extract_question_features(run, extracted)
    assert len(fake.calls) == len(rows) * 4
    assert all("original answer" not in p and "reference answer" not in p for p in fake.calls)
    attribution.extract_question_features(run, extracted)
    assert len(fake.calls) == len(rows) * 4
    report = attribution.score_questions(run, extracted, tmp_path / "scores", seeds=[42], bootstrap=2)
    assert len(report["seeds"]["42"]["conditions"]) == 4
    assert report["seeds"]["42"]["split"]["test"] == list(range(60, 80))
    marker = json.loads((extracted / "complete.json").read_text())
    marker["ids"].reverse()
    atomic_json(extracted / "complete.json", marker)
    with pytest.raises(ValueError, match="provenance"):
        attribution.score_questions(run, extracted, tmp_path / "bad", seeds=[42], bootstrap=2)


def reviewed_pair_file(run, destination):
    attribution.export_pairs(run, destination)
    document = json.loads(destination.read_text())
    for pair in document["pairs"]:
        pair.update(correct_answer="A confirmed correct factual answer.", incorrect_answer="A confirmed incorrect factual answer.", reviewer="reviewer1")
    atomic_json(destination, document)
    return document


def test_pairs_require_review_heldout_membership_and_distinct_answers(tmp_path):
    run = tmp_path / "run"
    fixture_run(run)
    destination = tmp_path / "pairs.json"
    attribution.export_pairs(run, destination)
    with pytest.raises(ValueError, match="reviewed distinct"):
        attribution.validated_pairs(run, destination)
    document = reviewed_pair_file(run, destination)
    before = destination.read_text()
    attribution.export_pairs(run, destination)
    assert destination.read_text() == before
    kept, _ = attribution.validated_pairs(run, destination)
    assert len(kept) == 20 and all(int(p["question_id"].split("-")[-1]) >= 60 for p in kept)
    document["pairs"][0]["question_id"] = "source-1"
    atomic_json(destination, document)
    with pytest.raises(ValueError, match="membership"):
        attribution.validated_pairs(run, destination)


def test_pair_statistics_keep_question_pair_unit_and_ties_half():
    p = np.array([[.1, .9], [.2, .8], [.5, .5], [.9, .1]])
    result = attribution.pair_statistics(p, repetitions=20)
    assert result["pair_order_accuracy_ties_half"] == .625
    assert result["questions"] == 4
    assert "question pair" in result["resampling_unit"]
    assert attribution.pair_statistics(np.full((10, 2), .4), 10)["pair_order_accuracy_ties_half"] == .5


def test_pair_extraction_and_frozen_scoring_use_no_test_fitting(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _, _, _, saved = fixture_run(run)
    pair_file = tmp_path / "pairs.json"
    document = reviewed_pair_file(run, pair_file)
    fake = FakeAdapter()
    monkeypatch.setattr(attribution, "adapter_for", lambda manifest: fake)
    extracted = tmp_path / "extracted"
    attribution.extract_pairs(run, pair_file, extracted)
    assert len(fake.calls) == 20 * 2 * 3
    original, calls = attribution.fit_source, []
    def spy(x, y, train, dev, c):
        assert list(train) == list(range(50)) and list(dev) == list(range(50, 60))
        calls.append(1)
        return original(x, y, train, dev, c)
    monkeypatch.setattr(attribution, "fit_source", spy)
    report = attribution.score_pairs(run, pair_file, extracted, tmp_path / "scored", bootstrap=2)
    assert len(calls) == 3 and len(report["methods"]) == 4
    assert report["methods"]["ensemble_mean"]["questions"] == 20
    document["pairs"][0]["incorrect_answer"] += " edited"
    atomic_json(pair_file, document)
    with pytest.raises(ValueError, match="provenance"):
        attribution.score_pairs(run, pair_file, extracted, tmp_path / "bad", bootstrap=2)


def test_frozen_transfer_never_fits_triviaqa_and_reuses_source_models(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    data, _, _, _ = fixture_run(source, "source")
    fixture_run(target, "target")
    import cverify.triviaqa_workflow as workflow
    actual, calls = workflow.fit_source, []
    def spy(x, y, train, dev, c):
        np.testing.assert_array_equal(y, data["y_error"])
        assert list(train) == list(range(50)) and list(dev) == list(range(50, 60))
        calls.append(1)
        return actual(x, y, train, dev, c)
    monkeypatch.setattr(workflow, "fit_source", spy)
    report = followup.frozen_transfer(source, target, tmp_path / "frozen", bootstrap=2)
    assert len(calls) == 5 and report["target_counts"]["all_reviewed"] == 80
    followup.frozen_transfer(source, target, tmp_path / "frozen", bootstrap=2)
    assert len(calls) == 5
    # Normalized question overlap must be rejected, not silently kept.
    rows = attribution.read_records(target / "metadata.jsonl")
    rows[0]["question"] = "SOURCE QUESTION NUMBER 0!"
    (target / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="overlap"):
        followup.frozen_transfer(source, target, tmp_path / "bad", bootstrap=2)


def test_detailed_sweep_includes_layers_verification_and_subtractions(tmp_path):
    run = tmp_path / "run"
    fixture_run(run)
    report = followup.detailed_detection(run, tmp_path / "detail", seeds=[42], bootstrap=2)
    assert len(report["runs"]) == 6
    for record in report["runs"].values():
        assert {"native_verify", "verify_minus_none", "verify_minus_control", "control_minus_none"} <= set(record["methods"])
        assert record["split"]["test"] == list(range(60, 80))


def test_cpu_runner_passes_subspace_output_as_file_and_finishes(tmp_path, monkeypatch):
    import sys
    source, target = tmp_path / "source", tmp_path / "target"
    fixture_run(source, "source")
    fixture_run(target, "target")
    calls = []
    def existing_analysis(run, metrics, output, extra):
        assert run == target and metrics == target / "results_seed_42_all_layers" / "metrics.json"
        if output.suffix == ".json":
            assert not output.is_dir()
            atomic_json(output, {"ok": True})
        else:
            assert output.is_dir()
        calls.append(output)
    monkeypatch.setattr(followup, "frozen_transfer", lambda *args, **kwargs: None)
    monkeypatch.setattr(followup, "detailed_detection", lambda *args, **kwargs: None)
    for name in ("analyze_geometry", "analyze_subspaces", "analyze_transfer"):
        monkeypatch.setattr(followup, name, existing_analysis)
    monkeypatch.setattr(sys, "argv", ["followup", "cpu", "--truth", str(source), "--trivia", str(target)])
    followup.main()
    assert len(calls) == 3
    assert json.loads((target / "followup" / "cpu_status.json").read_text())["phase"] == "complete"
