"""One resumable controller: two submitted jobs maximum, checkpoints, and audited targets."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from .cli import build_parser
from .data import export_truthfulqa, load_jsonl
from .extract import run_extraction
from .merge import run_merge
from .review import apply_review, export_review
from .storage import LockBusyError, atomic_json, digest, file_lock, lock_is_busy
from .train import run_training


def code_digest(script: Path) -> str:
    files = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(path.read_bytes() for path in files) + script.read_bytes()).hexdigest()


def file_digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def prepare_config(script: Path) -> tuple[Path, dict]:
    project = script.parent.parent.resolve()
    name = os.environ.get("CVERIFY_PIPELINE_RUN", "truthfulqa_concise_chat_audit_l15")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("CVERIFY_PIPELINE_RUN must be a single directory name")
    run = project / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    path = run / "workflow_config.json"
    dataset = Path(os.environ.get("CVERIFY_PIPELINE_DATA", str(run / "truthfulqa.jsonl"))).resolve()
    if not dataset.exists():
        export_truthfulqa(dataset)
    examples = load_jsonl(dataset)
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("Duplicate question IDs in dataset")
    config = {
        "project": str(project), "run": str(run), "script": str(script), "data": str(dataset),
        "data_sha256": file_digest(dataset), "code_sha256": code_digest(script),
        "model": os.environ.get("CVERIFY_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
        "initial_tokens": int(os.environ.get("CVERIFY_PIPELINE_INITIAL_TOKENS", "256")),
        "retry_tokens": int(os.environ.get("CVERIFY_PIPELINE_MAX_TOKENS", "1024")),
        "shard_size": int(os.environ.get("CVERIFY_PIPELINE_SHARD_SIZE", "25")),
        "layers": [-1, -4, -8, 15], "samples": 5, "temperature": 0.7, "seed": 42,
        "examples": len(examples), "chat_format": "tokenizer", "labeler": "deferred",
        "evaluation_profile": "assessment-v3", "generation_prompt": "concise",
        "cpus_per_task": 1, "max_submitted_jobs": 2,
    }
    if config["initial_tokens"] < 1 or config["retry_tokens"] < config["initial_tokens"] or config["shard_size"] < 1:
        raise ValueError("Invalid token ceiling or shard size")
    previous = load_json(path)
    if previous is not None and previous != config:
        raise ValueError("Code, dataset or settings changed since this run began. Restore them or set a NEW CVERIFY_PIPELINE_RUN.")
    atomic_json(path, config)
    return path, config


def prepare_finalization(script: Path) -> tuple[Path, dict]:
    """Use immutable extracted data after an orchestration-only code update."""
    project = script.parent.parent.resolve()
    name = os.environ.get("CVERIFY_PIPELINE_RUN", "truthfulqa_concise_chat_audit_l15")
    original_path = project / "runs" / name / "workflow_config.json"
    original = load_json(original_path)
    if original is None:
        raise ValueError("No existing workflow configuration; start a normal extraction run first")
    if Path(original["run"]).resolve() != original_path.parent.resolve() or Path(original["project"]).resolve() != project:
        raise ValueError("Existing configuration belongs to a different project/run")
    if file_digest(Path(original["data"])) != original["data_sha256"]:
        raise ValueError("Original extraction dataset changed; cannot finalize existing features")
    count = math.ceil(original["examples"] / original["shard_size"])
    if not all(shard_complete(original, index) for index in range(count)):
        raise ValueError("Finalization-only resume requires every extraction shard to be complete")
    for index in range(count):
        manifest = load_json(shard_path(original, index) / "manifest.json")
        expected = {"model": original["model"], "layers": original["layers"], "chat_format": original["chat_format"],
                    "samples": original["samples"], "temperature": original["temperature"],
                    "max_new_tokens": original["initial_tokens"], "max_retry_tokens": original["retry_tokens"],
                    "labeler": original["labeler"], "seed": original["seed"],
                    "generation_prompt": original["generation_prompt"]}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Extracted settings differ from the original configuration in shard {index}")
        if manifest.get("evaluation", {}).get("profile") != original["evaluation_profile"]:
            raise ValueError(f"Extracted evaluation profile differs in shard {index}")
    current_code = code_digest(script)
    config = {**original, "finalize_existing": True, "finalization_code_sha256": current_code,
              "original_config_sha256": file_digest(original_path)}
    path = original_path.parent / f"finalize_config_{current_code[:12]}.json"
    atomic_json(path, config)
    return path, config


def validate_worker(config: dict, kind: str = "extract") -> None:
    if config.get("finalize_existing") and kind != "finalize":
        raise ValueError("Finalization-only configuration cannot launch GPU extraction")
    expected_code = config.get("finalization_code_sha256", config["code_sha256"]) if kind == "finalize" else config["code_sha256"]
    if file_digest(Path(config["data"])) != config["data_sha256"] or code_digest(Path(config["script"])) != expected_code:
        raise ValueError("Code/dataset changed after submission; refusing to mix experiments")
    if config.get("finalize_existing"):
        original_path = Path(config["run"]) / "workflow_config.json"
        if file_digest(original_path) != config["original_config_sha256"]:
            raise ValueError("Original extraction configuration changed after finalization was scheduled")


def shard_path(config: dict, index: int) -> Path:
    return Path(config["run"]) / f"shard_{index}"


def shard_complete(config: dict, index: int) -> bool:
    path = shard_path(config, index)
    marker = load_json(path / "complete.json")
    if not marker:
        return False
    expected = min(config["shard_size"], config["examples"] - index * config["shard_size"])
    if marker["examples"] != expected:
        raise ValueError(f"Wrong completed shard size: {path}")
    if not all((path / name).exists() for name in ("features.npz", "metadata.jsonl", "manifest.json")):
        return False
    manifest = load_json(path / "manifest.json")
    if manifest.get("checkpoint_fingerprint") != marker["fingerprint"]:
        raise ValueError(f"Shard completion/manifest mismatch: {path}")
    return True


def extract_worker(config: dict, index: int) -> None:
    args = build_parser().parse_args([
        "extract", "--data", config["data"], "--model", config["model"], "--output", str(shard_path(config, index)),
        "--offset", str(index * config["shard_size"]), "--limit", str(config["shard_size"]),
        "--generation-prompt", "concise", "--evaluation-profile", "assessment-v3", "--chat-format", "tokenizer",
        "--max-new-tokens", str(config["initial_tokens"]), "--max-retry-tokens", str(config["retry_tokens"]),
        "--samples", "5", "--temperature", "0.7", "--layers", "-1", "-4", "-8", "15",
        "--seed", "42", "--seed-scheme", "global", "--dtype", "float16", "--labeler", "deferred", "--resume",
    ])
    run_extraction(args)


def finalize_worker(config: dict) -> None:
    run = Path(config["run"])
    status_path = run / "finalize_status.json"
    count = math.ceil(config["examples"] / config["shard_size"])
    if not all(shard_complete(config, i) for i in range(count)):
        raise ValueError("Cannot finalize: extraction is incomplete")
    full = run / "full"
    signature = digest([load_json(shard_path(config, i) / "complete.json") for i in range(count)])
    if load_json(run / "merge_complete.json") != {"signature": signature}:
        run_merge(SimpleNamespace(runs=[str(shard_path(config, i)) for i in range(count)], output=str(full)))
        atomic_json(run / "merge_complete.json", {"signature": signature})
    review = run / "review.jsonl"
    export_review(full, review)
    review_hash = file_digest(review)
    audited = run / "audited" / review_hash[:16]
    try:
        summary = apply_review(full, review, audited)
    except ValueError as error:
        atomic_json(status_path, {"phase": "needs_review", "review_hash": review_hash, "message": str(error)})
        return
    if summary["counts"]["pending"]:
        atomic_json(status_path, {"phase": "needs_review", "review_hash": review_hash,
                                  "message": f"Review {summary['counts']['pending']} answers in {review}; fill judgment and reviewer."})
        return
    with np.load(audited / "features.npz") as data:
        y = data["y_error"]
        if len(y) < 20 or len(np.unique(y)) != 2 or np.min(np.bincount(y)) < 5:
            atomic_json(status_path, {"phase": "needs_review", "review_hash": review_hash,
                                      "message": "Need at least 20 usable answers and five of each class after audited exclusions."})
            return
    for condition, layers in (("original_layers", [-1, -4, -8]), ("layer15", [15]), ("all_layers", [-1, -4, -8, 15])):
        output = audited / f"results_seed_42_{condition}"
        done = output / "training_complete.json"
        if load_json(done) == {"review_hash": review_hash, "layers": layers}:
            continue
        args = build_parser().parse_args([
            "train", "--run", str(audited), "--output", str(output), "--layers", *map(str, layers),
            "--seed", "42", "--test-size", "0.25", "--dev-size", "0.1633986928", "--bootstrap", "1000",
        ])
        run_training(args)
        atomic_json(done, {"review_hash": review_hash, "layers": layers})
    atomic_json(status_path, {"phase": "complete", "review_hash": review_hash, "results": str(audited)})


def finalize_once(config: dict) -> bool:
    """A duplicate worker exits successfully without touching another worker's files."""
    try:
        with file_lock(Path(config["run"]) / "finalization.lock"):
            finalize_worker(config)
        return True
    except LockBusyError:
        print("Finalization is already running in another process; this duplicate worker is exiting without changes.", flush=True)
        return False


class Slurm:
    def call(self, *args) -> str:
        return subprocess.check_output(args, text=True).strip()

    def live_jobs(self) -> set[str]:
        # --array expands array tasks so other jobs count against the submission limit.
        rows = self.call("squeue", "--array", "--noheader", "--user", os.environ["USER"], "--format", "%i|%j")
        self.names = {}
        ids = set()
        for line in rows.splitlines():
            if line.strip():
                job, name = (value.strip() for value in line.split("|", 1))
                ids.add(job)
                self.names[name] = job
        return ids

    @staticmethod
    def job_name(config: dict, kind: str, index: int | None) -> str:
        return f"cv-{digest(config['run'])[:10]}-{kind}-{index if index is not None else 'all'}"

    def find(self, config: dict, kind: str, index: int | None):
        return self.names.get(self.job_name(config, kind, index))

    def state(self, job: str) -> str:
        value = self.call("sacct", "-n", "-X", "-j", job, "--format=State", "--parsable2")
        return value.splitlines()[0].split("|")[0].split()[0].rstrip("+") if value else ""

    def submit(self, config: dict, config_path: Path, kind: str, index: int | None) -> str:
        if config.get("finalize_existing") and kind != "finalize":
            raise ValueError("Finalization-only resume cannot submit extraction jobs")
        command = ["sbatch", "--parsable", "--export=ALL", "--cpus-per-task=1", "--ntasks=1", "--nodes=1", "--chdir", config["project"],
                   "--job-name", self.job_name(config, kind, index)]
        if kind == "extract":
            command.append("--gres=gpu:1")
        command += [config["script"], "--worker", kind, "--config", str(config_path)]
        if index is not None:
            command += ["--shard", str(index)]
        response = self.call(*command)
        job = response.split(";")[0]
        if not job.isdigit():
            raise ValueError(f"Unexpected sbatch response: {response}")
        return job


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def controller(config_path: Path, config: dict, scheduler=None, poll_seconds=30, max_cycles=None) -> int:
    scheduler = scheduler or Slurm()
    run = Path(config["run"])
    state_path = run / "workflow_state.json"
    state = load_json(state_path, {"jobs": {}, "phase": "extracting"})
    attempts = {}
    count = math.ceil(config["examples"] / config["shard_size"])
    cycle = 0
    while True:
        live = scheduler.live_jobs()
        completed = [shard_complete(config, i) for i in range(count)]
        if all(completed) and lock_is_busy(run / "finalization.lock"):
            state.update(phase="finalizing", message="Waiting for the active finalization process; no duplicate job submitted")
            atomic_json(state_path, state)
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                return 0
            time.sleep(poll_seconds)
            continue
        state.pop("message", None)
        terminal_states = {}
        for key, info in state["jobs"].items():
            if info["id"] in live:
                continue
            if key.startswith("extract:") and completed[int(key.split(":")[1])]:
                status = "COMPLETED"
            else:
                status = info.get("terminal_state") or scheduler.state(info["id"])
            terminal_states[info["id"]] = status
            if status in TERMINAL:
                info["terminal_state"] = status
        atomic_json(state_path, state)
        # Reserve slots for submitted jobs not yet visible in squeue/accounting.
        unobserved = {job for job, status in terminal_states.items() if status not in TERMINAL}
        active = live | unobserved
        # Never submit the final CPU job until every extraction job has ended.
        if all(completed) and not any(info["id"] in active for key, info in state["jobs"].items() if key.startswith("extract:")):
            final = load_json(run / "finalize_status.json")
            active_finalizer = any(info["id"] in active for key, info in state["jobs"].items() if key == "finalize")
            if final and final.get("review_hash") == file_digest(run / "review.jsonl") and not active_finalizer:
                state["phase"] = final["phase"]
                atomic_json(state_path, state)
                print(json.dumps(final, indent=2), flush=True)
                return 0 if final["phase"] == "complete" else 2
            needed = [("finalize", None)]
        else:
            if config.get("finalize_existing") and not all(completed):
                raise ValueError("An extraction shard is no longer complete; finalization-only resume will not regenerate it")
            needed = [("extract", i) for i in range(count) if not completed[i]]
        slots = max(0, min(2 - len(active), 2))
        for kind, index in needed:
            key = f"{kind}:{index}" if index is not None else kind
            previous = state["jobs"].get(key)
            orphan = getattr(scheduler, "find", lambda *args: None)(config, kind, index)
            if orphan and (previous is None or previous["id"] != orphan):
                # Recover a submission made just before a controller crash, before its ID was saved.
                state["jobs"][key] = {"id": orphan, "submitted_at": time.time(),
                                       "total_submissions": (previous or {}).get("total_submissions", 0) + 1, "recovered": True}
                atomic_json(state_path, state)
                continue
            if previous:
                if previous["id"] in live:
                    continue
                terminal = terminal_states[previous["id"]]
                if terminal not in TERMINAL:
                    if time.time() - previous["submitted_at"] > 600:
                        raise RuntimeError(f"Cannot establish terminal state of job {previous['id']}; refusing duplicate submission")
                    continue
                if kind == "finalize" and terminal == "COMPLETED":
                    # Final status may have changed during this loop; reload at the top.
                    final = load_json(run / "finalize_status.json")
                    if final and final.get("review_hash") == file_digest(run / "review.jsonl"):
                        break
            if slots == 0:
                continue
            if attempts.get(key, 0) >= 3:
                state["phase"] = "failed"
                atomic_json(state_path, state)
                raise RuntimeError(f"{key} failed three attempts this invocation. Inspect job logs, then rerun the same command to resume.")
            try:
                job = scheduler.submit(config, config_path, kind, index)
            except subprocess.CalledProcessError as error:
                # Scheduler limits can change between querying and submitting. Do not record a nonexistent job.
                state["last_submission_error"] = str(error)
                atomic_json(state_path, state)
                return 1
            attempts[key] = attempts.get(key, 0) + 1
            state["jobs"][key] = {"id": job, "submitted_at": time.time(),
                                   "total_submissions": (previous or {}).get("total_submissions", 0) + 1}
            state["phase"] = "extracting" if kind == "extract" else "finalizing"
            atomic_json(state_path, state)
            print(f"Submitted {key}: {job}; {sum(completed)}/{count} shards complete", flush=True)
            slots -= 1
        cycle += 1
        if max_cycles is not None and cycle >= max_cycles:
            return 0
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--script", type=Path, required=True)
    run.add_argument("--status", action="store_true")
    run.add_argument("--finalize-existing", action="store_true",
                     help="Resume only merge/review/training from completed extraction after a controller update")
    worker = commands.add_parser("worker")
    worker.add_argument("kind", choices=["extract", "finalize"])
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.command == "run":
        name = os.environ.get("CVERIFY_PIPELINE_RUN", "truthfulqa_concise_chat_audit_l15")
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("CVERIFY_PIPELINE_RUN must be a single directory name")
        run_path = args.script.resolve().parent.parent / "runs" / name
        if args.status:
            print(json.dumps({"workflow": load_json(run_path / "workflow_state.json", {"phase": "not_started"}),
                              "finalization": load_json(run_path / "finalize_status.json")}, indent=2))
            return
        try:
            with file_lock(run_path / "controller.lock"):
                prepare = prepare_finalization if args.finalize_existing else prepare_config
                config_path, config = prepare(args.script.resolve())
                raise SystemExit(controller(config_path, config))
        except LockBusyError:
            print("A controller is already running for this run; no second controller was started.", flush=True)
    else:
        config = load_json(args.config)
        validate_worker(config, args.kind)
        if args.kind == "extract":
            if args.shard is None or not 0 <= args.shard < math.ceil(config["examples"] / config["shard_size"]):
                raise ValueError("Invalid shard index")
            extract_worker(config, args.shard)
        else:
            finalize_once(config)


if __name__ == "__main__":
    main()
