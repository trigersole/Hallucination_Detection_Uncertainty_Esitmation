"""Feature extraction for paired verification and control conditions."""

from __future__ import annotations

import json
import hashlib
import re
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import load_jsonl
from .labeling import build_labeler
from .model import CausalLMAdapter
from .prompts import EVALUATION_PROFILES, evaluation_prompt, evaluation_spec, generation_prompt
from .signals import lexical_cluster_statistics, rejection_and_disagreement
from .storage import atomic_json, atomic_npz, digest, file_lock


def _unit(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), eps)


def run_extraction(args) -> None:
    with file_lock(Path(args.output) / ".extract.lock"):
        _run_extraction(args)


def _run_extraction(args) -> None:
    output = Path(args.output)
    existing = [path for path in output.iterdir() if path.name != ".extract.lock"]
    if existing and not args.resume:
        raise ValueError(f"Extraction output must be empty; refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.max_new_tokens < 1 or (args.max_retry_tokens is not None and args.max_retry_tokens < args.max_new_tokens):
        raise ValueError("Token limits must be positive and retry ceiling cannot be below the initial limit")
    configuration = {key: value for key, value in vars(args).items() if key not in ("resume", "output")}
    configuration["data_sha256"] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    implementation_files = ["extract.py", "model.py", "prompts.py", "labeling.py"]
    if args.labeler == "triviaqa-exact":
        implementation_files.append("triviaqa_data.py")
    configuration["implementation_sha256"] = hashlib.sha256(b"".join(
        Path(__file__).with_name(name).read_bytes() for name in implementation_files
    )).hexdigest()
    fingerprint = digest(configuration)
    progress_path = output / "progress.json"
    if progress_path.exists():
        if json.loads(progress_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("Resume configuration, code or dataset changed; choose a new run name")
    elif existing:
        raise ValueError("Existing directory is not a resumable run; choose a new run name")
    atomic_json(progress_path, {"fingerprint": fingerprint, "configuration": configuration})
    if args.resume and (output / "complete.json").exists():
        marker = json.loads((output / "complete.json").read_text())
        if marker["fingerprint"] != fingerprint:
            raise ValueError("Completion marker does not match this configuration")
        if all((output / name).exists() for name in ("features.npz", "metadata.jsonl", "manifest.json")):
            print(f"Already complete: {output}")
            return
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    verify_templates, control_templates, labels = EVALUATION_PROFILES[args.evaluation_profile]
    spec = evaluation_spec(args.evaluation_profile)
    examples = load_jsonl(args.data, args.limit, args.offset)
    reference = {}
    if args.reference_run:
        reference_path = Path(args.reference_run)
        reference_manifest = json.loads((reference_path / "manifest.json").read_text(encoding="utf-8"))
        for key in ("model", "generation_prompt", "max_new_tokens", "samples", "temperature", "seed", "label_threshold"):
            if reference_manifest.get(key) != getattr(args, key):
                raise ValueError(f"Reference run has different {key}; cannot treat this as a paired correction")
        expected_labeler = "bleurt" if args.labeler == "bleurt" else "exact_match"
        if reference_manifest.get("labeler") != expected_labeler:
            raise ValueError("Reference run uses a different correctness labeler")
        with (reference_path / "metadata.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] in reference:
                    raise ValueError(f"Duplicate id in reference run: {row['id']}")
                reference[row["id"]] = row
        for example in examples:
            row = reference.get(example.id)
            if row is None or row["question"] != example.question or tuple(row["aliases"]) != example.answers:
                raise ValueError(f"Reference question or aliases differ for {example.id}")
    adapter = CausalLMAdapter(args.model, device=args.device, dtype=args.dtype,
                              trust_remote_code=args.trust_remote_code, chat_format=args.chat_format,
                              revision=getattr(args, "model_revision", None))
    runtime = {
        "model_commit": getattr(adapter.model.config, "_commit_hash", None),
        "chat_template": getattr(getattr(adapter, "tokenizer", None), "chat_template", None),
        "chat_template_source": getattr(adapter, "chat_template_source", None),
        "dtype": args.dtype,
    }
    for package in ("torch", "transformers", "numpy"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = None
    if getattr(args, "expected_runtime", None) is not None and args.expected_runtime != runtime:
        raise ValueError("Frozen source model/tokenizer/library runtime differs; stop before generating target answers")
    runtime_path = output / "runtime.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text(encoding="utf-8")) != runtime:
        raise ValueError("Model revision, chat template or library versions changed; refuse mixed checkpoints")
    atomic_json(runtime_path, runtime)
    labeler = build_labeler(args.labeler, bleurt_model=args.bleurt_model, device=args.labeler_device)
    rows: dict[str, list[np.ndarray | float | int]] = {
        "y_error": [], "generation": [],
        "generation_state": [],
        "verify_state": [], "control_state": [], "concat_state": [], "difference_state": [],
    }
    if labels:
        rows["verification_scores"] = []
    else:
        rows.update({"none_state": [], "verify_none_difference_state": [], "control_none_difference_state": []})
    metadata_path = output / "metadata.jsonl"
    total_tokens = 0
    assessment_prompt_tokens = 0
    temporary_metadata = output / "metadata.jsonl.tmp"
    with temporary_metadata.open("w", encoding="utf-8") as meta:
        for index, example in enumerate(tqdm(examples, desc="Extracting")):
            checkpoint = checkpoints / f"{index:06d}.npz"
            if args.resume and checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    record = json.loads(str(saved["record"].item()))
                    if record["id"] != example.id or set(saved.files) != set(rows) | {"record"}:
                        raise ValueError(f"Checkpoint identity or feature schema mismatch: {checkpoint}")
                    for key in rows:
                        rows[key].append(saved[key].copy())
                total_tokens += record["resource_usage"]["generation_and_label_tokens"]
                assessment_prompt_tokens += record["resource_usage"]["assessment_prompt_input_tokens"]
                meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            token_start, assessment_start = total_tokens, assessment_prompt_tokens
            prompt = generation_prompt(example.question, args.generation_prompt)
            source_index = args.offset + index if args.seed_scheme == "global" else index
            def generate(temperature, seed, layers):
                kwargs = dict(temperature=temperature, max_new_tokens=args.max_new_tokens, seed=seed)
                if args.max_retry_tokens is not None:
                    return adapter.generate_complete(prompt, layers, max_retry_tokens=args.max_retry_tokens, **kwargs)
                return adapter.generate(prompt, layers, **kwargs)
            original = generate(0.0, args.seed + source_index, args.layers)
            samples = [original]
            for sample_index in range(1, args.samples):
                samples.append(generate(args.temperature, args.seed + source_index * 1009 + sample_index,
                                        [] if args.max_retry_tokens is not None else args.layers))
            if reference:
                old = reference[example.id]
                if old["answer"] != original.text or old["sampled_answers"] != [item.text for item in samples]:
                    raise ValueError(f"Regenerated answers differ from reference for {example.id}; stop and inspect environment/settings")
            total_tokens += sum(sum(attempt["generated_tokens"] for attempt in item.attempts)
                                if item.attempts else item.generated_tokens for item in samples)
            consistency, lexical_entropy = lexical_cluster_statistics([item.text for item in samples])
            verify_probs, control_probs = [], []
            verify_states, control_states, differences = [], [], []
            for verify_instruction, control_instruction in zip(verify_templates, control_templates):
                v_prompt = evaluation_prompt(example.question, original.text, verify_instruction, args.evaluation_profile)
                c_prompt = evaluation_prompt(example.question, original.text, control_instruction, args.evaluation_profile)
                if labels:
                    v_prob, v_hidden, v_tokens = adapter.label_probability_and_hidden(v_prompt, args.layers, labels=labels)
                    c_prob, c_hidden, c_tokens = adapter.label_probability_and_hidden(c_prompt, args.layers, labels=labels)
                    total_tokens += v_tokens + c_tokens
                    verify_probs.append(v_prob)
                    control_probs.append(c_prob)
                else:
                    v_hidden, v_tokens = adapter.prompt_hidden(v_prompt, args.layers)
                    c_hidden, c_tokens = adapter.prompt_hidden(c_prompt, args.layers)
                    assessment_prompt_tokens += v_tokens + c_tokens
                v_vec = np.concatenate([_unit(v_hidden[layer]) for layer in args.layers])
                c_vec = np.concatenate([_unit(c_hidden[layer]) for layer in args.layers])
                verify_states.append(v_vec)
                control_states.append(c_vec)
                differences.append(v_vec - c_vec)
            gen_features = np.array([original.mean_log_probability, consistency, lexical_entropy], dtype=np.float32)
            generation_state = np.concatenate([_unit(original.hidden_by_layer[layer]) for layer in args.layers]).astype(np.float32)
            verify_flat = np.concatenate(verify_states).astype(np.float32)
            control_flat = np.concatenate(control_states).astype(np.float32)
            difference_flat = np.concatenate(differences).astype(np.float32)
            if not labels:
                n_prompt = evaluation_prompt(example.question, original.text, None, args.evaluation_profile)
                n_hidden, n_tokens = adapter.prompt_hidden(n_prompt, args.layers)
                assessment_prompt_tokens += n_tokens
                none_flat = np.concatenate([_unit(n_hidden[layer]) for layer in args.layers]).astype(np.float32)
                rows["none_state"].append(none_flat)
                rows["verify_none_difference_state"].append(verify_flat - none_flat)
                rows["control_none_difference_state"].append(control_flat - none_flat)
            label_score = labeler.score(original.text, example.answers)
            is_correct = label_score >= args.label_threshold if label_score is not None else None
            if reference and bool(reference[example.id]["is_correct"]) != is_correct:
                raise ValueError(f"Correctness label changed for {example.id}; cannot compare on the same split")
            rows["y_error"].append(int(not is_correct) if is_correct is not None else -1)
            rows["generation"].append(gen_features)
            if labels:
                rejection, disagreement = rejection_and_disagreement(verify_probs)
                rows["verification_scores"].append(np.array([
                    np.mean(verify_probs), rejection, disagreement, np.std(verify_probs)
                ], dtype=np.float32))
            rows["generation_state"].append(generation_state)
            rows["verify_state"].append(verify_flat)
            rows["control_state"].append(control_flat)
            rows["concat_state"].append(np.concatenate([verify_flat, control_flat]))
            rows["difference_state"].append(difference_flat)
            record = {
                "id": example.id, "question": example.question, "aliases": example.answers,
                "incorrect_references": example.incorrect_answers,
                "generation_prompt": args.generation_prompt,
                "evaluation_profile": args.evaluation_profile,
                "answer": original.text, "sampled_answers": [item.text for item in samples],
                "is_correct": is_correct, "labeler": labeler.name, "label_score": label_score,
                "label_threshold": args.label_threshold,
                "generation_mean_log_probability": original.mean_log_probability,
                "consistency": consistency, "lexical_cluster_entropy": lexical_entropy,
                "generation_details": [{
                    "generated_tokens": item.generated_tokens, "generated_token_ids": item.generated_token_ids,
                    "stop_reason": item.stop_reason, "hit_token_limit": item.stop_reason == "length",
                    "prompt_tokens": item.prompt_tokens, "attempts": item.attempts,
                    "contains_new_question": bool(re.search(r"(?m)^\s*Question:", item.text)),
                } for item in samples],
                "resource_usage": {
                    "generation_and_label_tokens": total_tokens - token_start,
                    "assessment_prompt_input_tokens": assessment_prompt_tokens - assessment_start,
                },
            }
            if labels:
                record.update({"verify_probabilities": verify_probs, "control_probabilities": control_probs})
            atomic_npz(checkpoint, record=np.array(json.dumps(record, ensure_ascii=False)),
                       **{key: np.asarray(values[-1]) for key, values in rows.items()})
            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    atomic_npz(output / "features.npz", **arrays)
    temporary_metadata.replace(metadata_path)
    atomic_json(output / "manifest.json", {
        "model": args.model, "data": str(Path(args.data).resolve()), "examples": len(examples),
        "generation_prompt": args.generation_prompt, "max_new_tokens": args.max_new_tokens,
        "generation_prompt_template": generation_prompt("{question}", args.generation_prompt),
        "evaluation": spec,
        "reference_run": str(Path(args.reference_run).resolve()) if args.reference_run else None,
        "checkpoint_fingerprint": fingerprint,
        "chat_format": args.chat_format, "max_retry_tokens": args.max_retry_tokens,
        "seed_scheme": args.seed_scheme,
        "chat_template": getattr(getattr(adapter, "tokenizer", None), "chat_template", None),
        "runtime": runtime,
        "generation_state_position": "last_non_eos_generated_token" if args.chat_format == "tokenizer" else "last_generated_token_including_eos",
        "layer_index_convention": "hidden_states tuple: 0=embedding; n>0=output after n blocks; negatives index from end",
        "resolved_layers": adapter._resolve_layers(args.layers, adapter.model.config.num_hidden_layers + 1),
        "offset": args.offset, "labeler": labeler.name, "label_threshold": args.label_threshold,
        "samples": args.samples, "temperature": args.temperature, "layers": args.layers,
        "seed": args.seed, "approximate_processed_label_and_generated_tokens": total_tokens,
        "assessment_prompt_input_tokens": assessment_prompt_tokens,
        "entropy_kind": "lexical_normalized-answer clusters",
    })
    atomic_json(output / "complete.json", {"fingerprint": fingerprint, "examples": len(examples)})
