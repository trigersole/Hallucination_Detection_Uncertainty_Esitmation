import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from cverify.cli import build_parser
from cverify.model import CausalLMAdapter, GeneratedAnswer
from cverify.prompts import EVALUATION_PROFILES, evaluation_prompt, evaluation_spec, generation_prompt
from cverify.train import feature_sets_for, run_training, select_layers


def test_exact_concise_and_three_assessment_templates():
    assert generation_prompt("Q?", "concise") == "Answer the question concisely.\nQuestion: Q?\nAnswer:"
    verify, control, labels = EVALUATION_PROFILES["assessment-v3"]
    assert labels == ()
    assert len(verify) == len(control) == 1
    prefix = "Question: Q?\nProposed answer: A.\n"
    assert evaluation_prompt("Q?", "A.", verify[0], "assessment-v3") == (
        prefix + "Task: Assess the factual correctness of the proposed answer for this question.\nAssessment:"
    )
    assert evaluation_prompt("Q?", "A.", control[0], "assessment-v3") == (
        prefix + "Task: Assess the formatting clarity of the proposed answer.\nAssessment:"
    )
    assert evaluation_prompt("Q?", "A.", None, "assessment-v3") == prefix + "Assessment:"
    spec = evaluation_spec("assessment-v3")
    assert spec["mode"] == "representation_only"
    assert spec["labels"] == []
    assert "Task:" not in spec["none_prompt_template"]
    assert "Yes" not in spec["prompt_template"]
    with pytest.raises(ValueError, match="only in assessment-v3"):
        evaluation_prompt("Q?", "A.", None, "yes-no-v2")


def test_prompt_hidden_reads_final_prompt_position_without_generation_or_labels():
    class Tensor(np.ndarray):
        def to(self, device):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self)

    class Batch(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            return self

    calls = []

    class Model:
        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])

        def __call__(self, *, input_ids, output_hidden_states, use_cache):
            calls.append(input_ids.copy())
            assert output_hidden_states and not use_cache
            return SimpleNamespace(hidden_states=tuple(
                (np.arange(6).reshape(1, 3, 2) + 100 * layer).view(Tensor)
                for layer in range(33)
            ))

    adapter = object.__new__(CausalLMAdapter)
    adapter.torch = SimpleNamespace(inference_mode=nullcontext)
    adapter.model = Model()
    adapter.tokenizer = lambda prompt, **kwargs: Batch(input_ids=np.array([[7, 8, 9]]).view(Tensor))
    hidden, tokens = adapter.prompt_hidden("Question: Q?\nProposed answer: A.\nAssessment:", [-1, 15])
    assert len(calls) == 1 and tokens == 3
    np.testing.assert_array_equal(calls[0], [[7, 8, 9]])
    np.testing.assert_array_equal(hidden[-1], [3204, 3205])
    np.testing.assert_array_equal(hidden[15], [1504, 1505])


def test_assessment_extraction_merge_layer_selection_and_training(tmp_path, monkeypatch):
    from cverify import extract
    from cverify.merge import run_merge

    dataset = tmp_path / "input.jsonl"
    dataset.write_text("".join(json.dumps({
        "id": f"q{i}", "question": f"Question {i}?", "answers": ["Paris" if i % 2 else "London"],
    }) + "\n" for i in range(40)), encoding="utf-8")
    generated_prompts, assessment_calls = [], []

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32))

        _resolve_layers = CausalLMAdapter._resolve_layers

        def generate(self, prompt, layers, **kwargs):
            generated_prompts.append(prompt)
            return GeneratedAnswer("Paris", -0.2, {layer: np.array([1., 2.]) for layer in layers}, 1)

        def label_probability_and_hidden(self, *args, **kwargs):
            pytest.fail("Representation-only extraction must not score response labels")

        def prompt_hidden(self, prompt, layers):
            assessment_calls.append(prompt)
            kind = 1 if "factual correctness" in prompt else 2 if "formatting clarity" in prompt else 3
            hidden = {layer: np.array([float(kind), float(layer + 10)]) for layer in layers}
            return hidden, len(prompt.split())

    monkeypatch.setattr(extract, "CausalLMAdapter", FakeAdapter)
    shards = []
    for index in range(2):
        shard = tmp_path / f"shard_{index}"
        args = build_parser().parse_args([
            "extract", "--data", str(dataset), "--model", "test-model", "--output", str(shard),
            "--generation-prompt", "concise", "--evaluation-profile", "assessment-v3",
            "--layers", "-1", "-4", "-8", "15", "--samples", "2",
            "--offset", str(index * 20), "--limit", "20",
        ])
        extract.run_extraction(args)
        shards.append(str(shard))
        records = [json.loads(line) for line in (shard / "metadata.jsonl").read_text().splitlines()]
        assert len(records) == 20
        assert all("verify_probabilities" not in row and "control_probabilities" not in row for row in records)
    assert len(generated_prompts) == 80
    assert all(prompt.startswith("Answer the question concisely.\n") for prompt in generated_prompts)
    assert len(assessment_calls) == 120
    for i in range(0, 120, 3):
        verify, control, none = assessment_calls[i:i + 3]
        assert "Task:" in verify and "Task:" in control and "Task:" not in none
        assert all(prompt.endswith("Assessment:") for prompt in (verify, control, none))
        assert all("Proposed answer: Paris\n" in prompt for prompt in (verify, control, none))
    merged = tmp_path / "full"
    run_merge(SimpleNamespace(runs=shards, output=str(merged)))
    manifest = json.loads((merged / "manifest.json").read_text())
    assert manifest["evaluation"]["none_prompt_template"].endswith("Assessment:")
    assert manifest["assessment_prompt_input_tokens"] == sum(len(prompt.split()) for prompt in assessment_calls)
    assert manifest["resolved_layers"] == [32, 29, 25, 15]
    with np.load(merged / "features.npz") as archive:
        data = dict(archive)
    assert "verification_scores" not in data
    for key in ("verify_state", "control_state", "none_state"):
        assert data[key].shape == (40, 8)
        np.testing.assert_allclose(np.linalg.norm(data[key].reshape(40, 4, 2), axis=2), 1., atol=1e-6)
    for key, left, right in (
        ("difference_state", "verify_state", "control_state"),
        ("verify_none_difference_state", "verify_state", "none_state"),
        ("control_none_difference_state", "control_state", "none_state"),
    ):
        np.testing.assert_allclose(data[key], data[left] - data[right], atol=1e-6)
    selected = select_layers(data, [-1, -4, -8, 15], [15])
    for key in ("none_state", "verify_none_difference_state", "control_none_difference_state"):
        np.testing.assert_array_equal(selected[key], data[key][:, 6:8])
    assert len(feature_sets_for(selected)) == 14
    output = tmp_path / "results"
    run_training(build_parser().parse_args([
        "train", "--run", str(merged), "--output", str(output),
        "--layers", "15", "--bootstrap", "3", "--seed", "42",
    ]))
    report = json.loads((output / "metrics.json").read_text())
    assert len(report["models"]) == 14
    assert not {"p_true_ensemble", "generation_plus_verification", "full_controlled"} & report["models"].keys()
    assert report["configuration"]["feature_dimensions"]["none_state"] == 2
    assert report["configuration"]["feature_dimensions"]["verify_none_concat_state"] == 4
    predictions = (output / "test_predictions.jsonl").read_text().splitlines()
    assert len(predictions) == 8


def test_merge_rejects_assessment_and_label_profiles(tmp_path):
    from cverify.merge import run_merge

    runs = []
    for profile in ("yes-no-v2", "assessment-v3"):
        path = tmp_path / profile
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"evaluation": evaluation_spec(profile)}))
        runs.append(str(path))
    with pytest.raises(ValueError, match="Shard mismatch for evaluation"):
        run_merge(SimpleNamespace(runs=runs, output=str(tmp_path / "invalid")))
