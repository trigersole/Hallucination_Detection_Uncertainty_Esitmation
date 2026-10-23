import json
from types import SimpleNamespace

import numpy as np
import pytest

from cverify.cli import build_parser
from cverify.model import CausalLMAdapter, GeneratedAnswer
from cverify.prompts import EVALUATION_PROFILES, evaluation_prompt, evaluation_spec
from cverify.train import select_layers


def test_yes_no_instructions_and_scored_labels_agree():
    verify, control, labels = EVALUATION_PROFILES["yes-no-v2"]
    assert labels == (" Yes", " No")
    for instruction in (*verify, *control):
        assert instruction.endswith("?")
        prompt = evaluation_prompt("Question?", "Answer.", instruction, "yes-no-v2")
        assert prompt.endswith("Respond with exactly one label: Yes or No.\nAnswer:")
        assert "Correct or Incorrect" not in prompt
    assert evaluation_spec("legacy") != evaluation_spec("yes-no-v2")


def test_layer15_is_fifteenth_block_not_embedding_or_sixteenth_block():
    adapter = object.__new__(CausalLMAdapter)
    assert adapter._resolve_layers([-1, -4, -8, 15], 33) == [32, 29, 25, 15]
    with pytest.raises(ValueError):
        adapter._resolve_layers([33], 33)


def test_select_layers_preserves_prompt_and_concat_layout():
    generation = np.arange(16).reshape(2, 8)
    verify = np.arange(48).reshape(2, 24)
    control = verify + 100
    data = {
        "generation_state": generation,
        "verify_state": verify,
        "control_state": control,
        "difference_state": verify - control,
        "concat_state": np.c_[verify, control],
        "generation": np.ones((2, 3)),
    }
    selected = select_layers(data, [-1, -4, -8, 15], [15])
    np.testing.assert_array_equal(selected["generation_state"], generation[:, [6, 7]])
    np.testing.assert_array_equal(selected["verify_state"], verify[:, [6, 7, 14, 15, 22, 23]])
    np.testing.assert_array_equal(selected["concat_state"], np.c_[selected["verify_state"], selected["control_state"]])
    np.testing.assert_array_equal(selected["difference_state"], selected["verify_state"] - selected["control_state"])
    assert selected["generation"] is data["generation"]
    with pytest.raises(ValueError, match="not extracted"):
        select_layers(data, [-1, -4, -8, 15], [16])


def test_extraction_records_corrected_protocol_and_checks_reference(tmp_path, monkeypatch):
    from cverify import extract

    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"id":"q1","question":"Capital of France?","answers":["Paris"]}\n', encoding="utf-8")
    calls = []

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32))

        _resolve_layers = CausalLMAdapter._resolve_layers

        def generate(self, prompt, layers, **kwargs):
            return GeneratedAnswer("Paris", -0.2, {layer: np.array([1., 2.]) for layer in layers}, 1)

        def label_probability_and_hidden(self, prompt, layers, *, labels):
            calls.append((prompt, labels))
            return 0.8, {layer: np.array([2., 1.]) for layer in layers}, 2

    monkeypatch.setattr(extract, "CausalLMAdapter", FakeAdapter)
    parser = build_parser()

    def arguments(output, *extra):
        return parser.parse_args([
            "extract", "--data", str(dataset), "--model", "test-model", "--output", str(output),
            "--evaluation-profile", "yes-no-v2", "--layers", "-1", "-4", "-8", "15",
            "--samples", "2", *extra,
        ])

    original = tmp_path / "original"
    extract.run_extraction(arguments(original))
    assert len(calls) == 6
    assert all(labels == (" Yes", " No") and prompt.endswith("Answer:") for prompt, labels in calls)
    manifest = json.loads((original / "manifest.json").read_text())
    assert manifest["evaluation"]["profile"] == "yes-no-v2"
    assert manifest["resolved_layers"] == [32, 29, 25, 15]
    with np.load(original / "features.npz") as features:
        assert features["generation_state"].shape == (1, 8)
        assert features["difference_state"].shape == (1, 24)
    extract.run_extraction(arguments(tmp_path / "paired", "--reference-run", str(original)))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        extract.run_extraction(arguments(original))
    metadata_path = original / "metadata.jsonl"
    metadata = json.loads(metadata_path.read_text())
    metadata["answer"] = "London"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="answers differ"):
        extract.run_extraction(arguments(tmp_path / "mismatched", "--reference-run", str(original)))


def test_merge_rejects_mixed_evaluation_profiles_before_loading_features(tmp_path):
    from cverify.merge import run_merge

    runs = []
    for profile in ("legacy", "yes-no-v2"):
        shard = tmp_path / profile
        shard.mkdir()
        (shard / "manifest.json").write_text(json.dumps({"evaluation": evaluation_spec(profile)}))
        runs.append(str(shard))
    with pytest.raises(ValueError, match="Shard mismatch for evaluation"):
        run_merge(SimpleNamespace(runs=runs, output=str(tmp_path / "merged")))


def test_three_layer_conditions_share_split_and_generation_baseline(tmp_path):
    from cverify.train import run_training

    rng = np.random.default_rng(17)
    run = tmp_path / "run"
    run.mkdir()
    n = 817
    verify = rng.normal(size=(n, 24))
    control = rng.normal(size=(n, 24))
    np.savez_compressed(
        run / "features.npz", y_error=np.arange(n) % 2,
        generation=rng.normal(size=(n, 3)),
        verification_scores=rng.uniform(size=(n, 4)),
        generation_state=rng.normal(size=(n, 8)),
        verify_state=verify, control_state=control,
        concat_state=np.c_[verify, control], difference_state=verify - control,
    )
    (run / "manifest.json").write_text(json.dumps({
        "layers": [-1, -4, -8, 15], "evaluation": evaluation_spec("yes-no-v2"),
    }))
    reports, baseline_probabilities = [], []
    for name, layers in (("original", [-1, -4, -8]), ("layer15", [15]), ("all", [-1, -4, -8, 15])):
        output = tmp_path / name
        args = build_parser().parse_args([
            "train", "--run", str(run), "--output", str(output),
            "--layers", *map(str, layers), "--seed", "42", "--bootstrap", "3",
            "--test-size", "0.25", "--dev-size", "0.1633986928",
        ])
        run_training(args)
        report = json.loads((output / "metrics.json").read_text())
        assert report["configuration"]["selected_layers"] == layers
        assert {key: len(value) for key, value in report["split"].items()} == {"train": 512, "dev": 100, "test": 205}
        assert len(report["models"]) == 10
        reports.append(report)
        predictions = [json.loads(line) for line in (output / "test_predictions.jsonl").read_text().splitlines()]
        baseline_probabilities.append([row["error_probabilities"]["generation_only"] for row in predictions])
    assert reports[0]["split"] == reports[1]["split"] == reports[2]["split"]
    np.testing.assert_allclose(baseline_probabilities[0], baseline_probabilities[1])
    np.testing.assert_allclose(baseline_probabilities[0], baseline_probabilities[2])
