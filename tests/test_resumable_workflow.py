import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cverify.cli import build_parser
from cverify.model import CausalLMAdapter, GeneratedAnswer
from cverify.storage import atomic_json


def test_chat_template_and_assistant_assessment_prefix():
    calls = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "<s>[INST] " + messages[0]["content"] + " [/INST] "
    adapter = object.__new__(CausalLMAdapter)
    adapter.chat_format = "tokenizer"
    adapter.tokenizer = Tokenizer()
    rendered = adapter.render_prompt("Question: Q?\nAnswer:")
    assert rendered == "<s>[INST] Question: Q?\nAnswer: [/INST] "
    rendered = adapter.render_prompt("Question: Q?\nProposed answer: A.\nAssessment:", "Assessment:")
    assert rendered.endswith("[/INST] Assessment:")
    assert "Assessment:" not in calls[-1][0][0]["content"]
    assert calls[-1][1] == {"tokenize": False, "add_generation_prompt": True}


def test_missing_template_uses_only_explicit_official_llama2_fallback():
    from cverify.model import ensure_chat_template
    tokenizer = SimpleNamespace(chat_template=None, bos_token="<s>")
    assert ensure_chat_template(tokenizer, "meta-llama/Llama-2-7b-chat-hf") == "meta_llama2_single_user_fallback"
    assert "[INST]" in tokenizer.chat_template and "[/INST]" in tokenizer.chat_template
    with pytest.raises(ValueError, match="no chat template"):
        ensure_chat_template(SimpleNamespace(chat_template=None, bos_token="<s>"), "unknown/model")


@pytest.mark.parametrize("stop_at_ceiling", [False, True])
def test_retry_preserves_prompt_seed_and_records_each_budget(stop_at_ceiling):
    calls = []
    adapter = object.__new__(CausalLMAdapter)
    def generate(prompt, layers, **kwargs):
        calls.append((prompt, layers, kwargs))
        reason = "length" if stop_at_ceiling or kwargs["max_new_tokens"] < 1024 else "eos"
        return GeneratedAnswer("answer", -0.1, {}, kwargs["max_new_tokens"], stop_reason=reason)
    adapter.generate = generate
    result = adapter.generate_complete("Q?", [15], temperature=0.7, max_new_tokens=256, max_retry_tokens=1024, seed=42)
    assert [call[2]["max_new_tokens"] for call in calls] == [256, 512, 1024]
    assert all(call[0] == "Q?" and call[2]["seed"] == 42 for call in calls)
    assert len(result.attempts) == 3
    assert result.stop_reason == ("length" if stop_at_ceiling else "eos")


def test_generation_records_eos_ids_and_last_content_state_without_duplicate_bos():
    class Tensor(np.ndarray):
        def to(self, device): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    class Batch(dict):
        @property
        def input_ids(self): return self["input_ids"]
        def to(self, device): return self
    tokenizer_calls, replay_calls = [], []
    class Tokenizer:
        eos_token_id = pad_token_id = 2
        def apply_chat_template(self, messages, **kwargs): return "<s>[INST] Q? [/INST]"
        def __call__(self, text, **kwargs):
            tokenizer_calls.append(kwargs)
            return Batch(input_ids=np.array([[1, 99]]).view(Tensor))
        def decode(self, ids, **kwargs): return "Full answer."
    class Model:
        generation_config = SimpleNamespace(eos_token_id=2)
        def parameters(self): return iter([SimpleNamespace(device="cpu")])
        def generate(self, **kwargs):
            return SimpleNamespace(sequences=np.array([[1, 99, 10, 11, 2]]).view(Tensor),
                                   scores=[np.zeros((1, 12)).view(Tensor) for _ in range(3)])
        def __call__(self, ids, **kwargs):
            replay_calls.append(ids)
            return SimpleNamespace(hidden_states=tuple(
                (np.arange(10).reshape(1, 5, 2) + layer * 100).view(Tensor) for layer in range(33)))
    adapter = object.__new__(CausalLMAdapter)
    adapter.chat_format, adapter.model, adapter.tokenizer = "tokenizer", Model(), Tokenizer()
    adapter.torch = SimpleNamespace(manual_seed=lambda seed: None, inference_mode=nullcontext,
                                    log_softmax=lambda values, dim: (values - np.log(np.exp(values).sum(axis=dim, keepdims=True))).view(Tensor))
    result = adapter.generate("Q?", [15], temperature=0.0, max_new_tokens=3, seed=42)
    assert result.generated_token_ids == [10, 11, 2]
    assert result.stop_reason == "eos"  # EOS at the cap is completion, not truncation.
    assert result.prompt_tokens == 2
    np.testing.assert_array_equal(result.hidden_by_layer[15], [1506, 1507])
    assert tokenizer_calls[0]["add_special_tokens"] is False
    adapter.generate("Q?", [], temperature=0.7, max_new_tokens=3, seed=42)
    assert len(replay_calls) == 1  # Sample-only generation avoids hidden-state replay.


class FakeAdapter:
    calls = []
    fail_question = None
    initializations = 0
    def __init__(self, *args, **kwargs):
        type(self).initializations += 1
        self.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32))
        self.tokenizer = SimpleNamespace(chat_template="fixture-template")
    _resolve_layers = CausalLMAdapter._resolve_layers
    def generate_complete(self, prompt, layers, **kwargs):
        type(self).calls.append((prompt, kwargs["seed"]))
        if type(self).fail_question and type(self).fail_question in prompt:
            type(self).fail_question = None
            raise RuntimeError("simulated interruption")
        return GeneratedAnswer("Paris", -0.2, {layer: np.array([1., 2.]) for layer in layers}, 2,
                               [7, 2], "eos", 8, [{"limit": 256, "generated_tokens": 2, "stop_reason": "eos"}])
    def prompt_hidden(self, prompt, layers):
        return {layer: np.array([float(layer + 10), 2.]) for layer in layers}, len(prompt.split())


def make_dataset(path, count):
    path.write_text("".join(json.dumps({"id": f"q{i}", "question": f"Question {i}?", "answers": ["Paris"]}) + "\n"
                            for i in range(count)), encoding="utf-8")


def test_question_checkpoints_resume_and_reject_changed_configuration(tmp_path, monkeypatch):
    from cverify import extract
    monkeypatch.setattr(extract, "CausalLMAdapter", FakeAdapter)
    FakeAdapter.calls, FakeAdapter.fail_question, FakeAdapter.initializations = [], "Question 1?", 0
    dataset = tmp_path / "input.jsonl"
    make_dataset(dataset, 3)
    output = tmp_path / "run"
    argv = ["extract", "--data", str(dataset), "--model", "fixture", "--output", str(output),
            "--generation-prompt", "concise", "--evaluation-profile", "assessment-v3", "--chat-format", "tokenizer",
            "--labeler", "deferred", "--max-new-tokens", "256", "--max-retry-tokens", "1024",
            "--layers", "15", "--samples", "2", "--seed-scheme", "global", "--resume"]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        extract.run_extraction(build_parser().parse_args(argv))
    assert (output / "checkpoints/000000.npz").exists()
    assert not (output / "complete.json").exists()
    calls_before = len(FakeAdapter.calls)
    extract.run_extraction(build_parser().parse_args(argv))
    assert all("Question 0?" not in prompt for prompt, seed in FakeAdapter.calls[calls_before:])
    metadata = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
    assert len(metadata) == 3 and all(row["is_correct"] is None for row in metadata)
    assert all(row["generation_details"][0]["stop_reason"] == "eos" for row in metadata)
    with np.load(output / "features.npz") as data:
        np.testing.assert_array_equal(data["y_error"], [-1, -1, -1])
    initialized = FakeAdapter.initializations
    extract.run_extraction(build_parser().parse_args(argv))
    assert FakeAdapter.initializations == initialized  # Finished run does not load another GPU model.
    changed = list(argv)
    changed[changed.index("256")] = "128"
    with pytest.raises(ValueError, match="configuration"):
        extract.run_extraction(build_parser().parse_args(changed))


def test_unreviewed_labels_cannot_reach_classifier(tmp_path):
    from cverify.train import run_training
    np.savez(tmp_path / "features.npz", y_error=np.full(20, -1))
    args = build_parser().parse_args(["train", "--run", str(tmp_path), "--output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="audited labels"):
        run_training(args)


def test_full_workflow_waits_for_review_then_resumes_training(tmp_path, monkeypatch):
    from cverify import extract, workflow
    from cverify.review import read_records
    monkeypatch.setattr(extract, "CausalLMAdapter", FakeAdapter)
    FakeAdapter.fail_question, FakeAdapter.calls = None, []
    script = tmp_path / "scripts/run_assessment.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\n")
    dataset = tmp_path / "input.jsonl"
    make_dataset(dataset, 40)
    monkeypatch.setenv("CVERIFY_PIPELINE_DATA", str(dataset))
    monkeypatch.setenv("CVERIFY_PIPELINE_RUN", "fixture")
    path, config = workflow.prepare_config(script)
    workflow.extract_worker(config, 0)
    workflow.extract_worker(config, 1)
    workflow.finalize_worker(config)
    root = Path(config["run"])
    assert json.loads((root / "finalize_status.json").read_text())["phase"] == "needs_review"
    reviews = read_records(root / "review.jsonl")
    assert len(reviews) == 40 and all(row["judgment"] is None for row in reviews)
    reviews[0]["sample_hash"] = "incorrect-hash"
    (root / "review.jsonl").write_text("\n".join(json.dumps(row) for row in reviews))
    workflow.finalize_worker(config)
    assert "different question/answer" in json.loads((root / "finalize_status.json").read_text())["message"]
    from cverify.review import sample_hash
    raw = read_records(root / "full/metadata.jsonl")
    reviews[0]["sample_hash"] = sample_hash(raw[0])
    for i, row in enumerate(reviews):
        row.update(judgment="correct" if i % 2 else "incorrect", reviewer="fixture-reviewer")
    reviews[0]["judgment"] = "ambiguous"
    (root / "review.jsonl").write_text("\n".join(json.dumps(row) for row in reviews))
    training = []
    def train(args):
        training.append(args.layers)
        from cverify.features import load_run_features
        data, manifest = load_run_features(Path(args.run))
        assert len(data["y_error"]) == 39
        assert set(data["y_error"]) == {0, 1}
        Path(args.output).mkdir(parents=True)
    monkeypatch.setattr(workflow, "run_training", train)
    workflow.finalize_worker(config)
    assert training == [[-1, -4, -8], [15], [-1, -4, -8, 15]]
    assert json.loads((root / "finalize_status.json").read_text())["phase"] == "complete"
    workflow.finalize_worker(config)
    assert len(training) == 3


class FakeSlurm:
    def __init__(self):
        self.live, self.states, self.submissions = set(), {}, []
    def live_jobs(self): return set(self.live)
    def state(self, job): return self.states.get(job, "")
    def submit(self, config, path, kind, index):
        job = str(100 + len(self.submissions))
        self.submissions.append((kind, index, job))
        self.live.add(job)
        self.states[job] = "RUNNING"
        return job


def controller_fixture(tmp_path, monkeypatch):
    from cverify import workflow
    done = set()
    monkeypatch.setattr(workflow, "shard_complete", lambda config, i: i in done)
    config = {"run": str(tmp_path), "examples": 3, "shard_size": 1}
    return workflow, config, done


def test_scheduler_limits_resume_and_failed_job_retry(tmp_path, monkeypatch):
    workflow, config, done = controller_fixture(tmp_path, monkeypatch)
    scheduler = FakeSlurm()
    path = tmp_path / "config.json"
    workflow.controller(path, config, scheduler, max_cycles=1)
    assert [(kind, i) for kind, i, job in scheduler.submissions] == [("extract", 0), ("extract", 1)]
    workflow.controller(path, config, scheduler, max_cycles=1)
    assert len(scheduler.submissions) == 2  # Resume cannot duplicate live jobs.
    scheduler.live.remove("100")
    scheduler.states["100"] = "TIMEOUT"
    workflow.controller(path, config, scheduler, max_cycles=1)
    assert scheduler.submissions[-1][:2] == ("extract", 0)
    done.add(0)
    scheduler.live.remove("102")
    scheduler.states["102"] = "COMPLETED"
    workflow.controller(path, config, scheduler, max_cycles=1)
    assert scheduler.submissions[-1][:2] == ("extract", 2)


def test_scheduler_reserves_unobserved_jobs_and_other_user_jobs(tmp_path, monkeypatch):
    workflow, config, done = controller_fixture(tmp_path, monkeypatch)
    scheduler = FakeSlurm()
    scheduler.live.add("other-job")
    workflow.controller(tmp_path / "config.json", config, scheduler, max_cycles=1)
    assert len(scheduler.submissions) == 1
    scheduler.live.clear()
    scheduler.states["100"] = ""
    # Submit one other shard, not two: the unobserved job still reserves a slot.
    workflow.controller(tmp_path / "config.json", config, scheduler, max_cycles=1)
    assert len(scheduler.submissions) == 2


def test_controller_review_gate_and_changed_review_resubmission(tmp_path, monkeypatch):
    workflow, config, done = controller_fixture(tmp_path, monkeypatch)
    done.update([0, 1, 2])
    review = tmp_path / "review.jsonl"
    review.write_text("pending")
    atomic_json(tmp_path / "finalize_status.json", {"phase": "needs_review", "review_hash": workflow.file_digest(review)})
    scheduler = FakeSlurm()
    assert workflow.controller(tmp_path / "config.json", config, scheduler, max_cycles=1) == 2
    assert not scheduler.submissions
    review.write_text("completed")
    workflow.controller(tmp_path / "config.json", config, scheduler, max_cycles=1)
    assert scheduler.submissions[0][:2] == ("finalize", None)


def test_controller_recovers_submission_missing_from_state_file(tmp_path, monkeypatch):
    workflow, config, done = controller_fixture(tmp_path, monkeypatch)
    scheduler = FakeSlurm()
    scheduler.live.add("orphan")
    scheduler.find = lambda config, kind, i: "orphan" if i == 0 else None
    workflow.controller(tmp_path / "config.json", config, scheduler, max_cycles=1)
    assert scheduler.submissions[0][:2] == ("extract", 1)
    state = json.loads((tmp_path / "workflow_state.json").read_text())
    assert state["jobs"]["extract:0"]["recovered"] is True


def test_capped_samples_are_excluded_even_after_correctness_review(tmp_path):
    from cverify.review import apply_review, export_review, read_records
    source = tmp_path / "source"
    source.mkdir()
    rows = [{"id": f"q{i}", "question": "Q?", "answer": "A.", "aliases": ["A."],
             "generation_details": [{"stop_reason": "eos"}, {"stop_reason": "length" if i == 0 else "eos"}]}
            for i in range(2)]
    (source / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (source / "manifest.json").write_text('{"examples":2}')
    np.savez(source / "features.npz", y_error=np.array([-1, -1]), generation=np.zeros((2, 3)))
    path = tmp_path / "review.jsonl"
    export_review(source, path)
    reviews = read_records(path)
    for row in reviews:
        row.update(judgment="correct", reviewer="fixture")
    path.write_text("\n".join(json.dumps(row) for row in reviews))
    summary = apply_review(source, path, tmp_path / "audited")
    assert summary["included"] == 1 and summary["counts"]["quality_excluded"] == 1
    from cverify.features import load_run_features
    data, manifest = load_run_features(tmp_path / "audited")
    np.testing.assert_array_equal(data["y_error"], [0])
    assert not (tmp_path / "audited/features.npz").exists()


def test_all_batch_scripts_request_one_cpu():
    root = Path(__file__).parent.parent
    scripts = list((root / "scripts").glob("*.sbatch")) + [root / "scripts/run_assessment.sh"]
    for script in scripts:
        assert "#SBATCH --cpus-per-task=1" in script.read_text()
        assert "#SBATCH --cpus-per-task=8" not in script.read_text()


def test_duplicate_finalizer_exits_cleanly_and_keeps_live_lock(tmp_path, monkeypatch):
    from cverify import workflow
    from cverify.storage import file_lock, lock_is_busy
    called = []
    monkeypatch.setattr(workflow, "finalize_worker", lambda config: called.append(config))
    config = {"run": str(tmp_path)}
    path = tmp_path / "finalization.lock"
    with file_lock(path):
        assert lock_is_busy(path)
        assert workflow.finalize_once(config) is False
        assert not called
        assert lock_is_busy(path)
    assert not lock_is_busy(path)
    assert workflow.finalize_once(config) is True
    assert len(called) == 1


def test_controller_waits_for_finalization_lock_instead_of_resubmitting(tmp_path, monkeypatch):
    from cverify.storage import file_lock
    workflow, config, done = controller_fixture(tmp_path, monkeypatch)
    done.update([0, 1, 2])
    scheduler = FakeSlurm()
    with file_lock(tmp_path / "finalization.lock"):
        workflow.controller(tmp_path / "config.json", config, scheduler, max_cycles=1)
    assert not scheduler.submissions
    assert json.loads((tmp_path / "workflow_state.json").read_text())["phase"] == "finalizing"


def test_finalization_upgrade_preserves_original_extraction_configuration(tmp_path, monkeypatch):
    from cverify import extract, workflow
    monkeypatch.setattr(extract, "CausalLMAdapter", FakeAdapter)
    FakeAdapter.fail_question = None
    script = tmp_path / "scripts/run_assessment.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\n")
    dataset = tmp_path / "input.jsonl"
    make_dataset(dataset, 3)
    monkeypatch.setenv("CVERIFY_PIPELINE_DATA", str(dataset))
    monkeypatch.setenv("CVERIFY_PIPELINE_RUN", "upgrade")
    original_path, original = workflow.prepare_config(script)
    original_bytes = original_path.read_bytes()
    workflow.extract_worker(original, 0)
    script.write_text("#!/bin/bash\n# orchestration fix\n")
    with pytest.raises(ValueError, match="changed"):
        workflow.prepare_config(script)
    final_path, final = workflow.prepare_finalization(script)
    assert original_path.read_bytes() == original_bytes
    assert final["code_sha256"] == original["code_sha256"]
    assert final["finalization_code_sha256"] != original["code_sha256"]
    workflow.validate_worker(final, "finalize")
    with pytest.raises(ValueError, match="cannot launch GPU"):
        workflow.validate_worker(final, "extract")
    (Path(original["run"]) / "shard_0/complete.json").unlink()
    with pytest.raises(ValueError, match="every extraction shard"):
        workflow.prepare_finalization(script)


def test_finalize_submission_enforces_one_task_and_rejects_gpu_for_upgrade(tmp_path):
    from cverify.workflow import Slurm
    scheduler = Slurm()
    calls = []
    scheduler.call = lambda *args: calls.append(args) or "123"
    config = {"project": str(tmp_path), "run": str(tmp_path), "script": "run.sh", "finalize_existing": True}
    assert scheduler.submit(config, tmp_path / "config.json", "finalize", None) == "123"
    assert "--ntasks=1" in calls[0] and "--nodes=1" in calls[0] and "--cpus-per-task=1" in calls[0]
    assert "--gres=gpu:1" not in calls[0]
    with pytest.raises(ValueError, match="cannot submit extraction"):
        scheduler.submit(config, tmp_path / "config.json", "extract", 0)
