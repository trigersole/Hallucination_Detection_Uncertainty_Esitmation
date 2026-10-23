"""Feature extraction for paired verification and control conditions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import load_jsonl
from .labeling import build_labeler
from .model import CausalLMAdapter
from .prompts import EVALUATION_PROFILES, evaluation_prompt, evaluation_spec, generation_prompt
from .signals import lexical_cluster_statistics, rejection_and_disagreement


def _unit(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), eps)


def run_extraction(args) -> None:
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Extraction output must be empty; refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
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
    adapter = CausalLMAdapter(args.model, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
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
    with metadata_path.open("w", encoding="utf-8") as meta:
        for index, example in enumerate(tqdm(examples, desc="Extracting")):
            prompt = generation_prompt(example.question, args.generation_prompt)
            original = adapter.generate(prompt, args.layers, temperature=0.0, max_new_tokens=args.max_new_tokens, seed=args.seed + index)
            samples = [original]
            for sample_index in range(1, args.samples):
                samples.append(adapter.generate(prompt, args.layers, temperature=args.temperature, max_new_tokens=args.max_new_tokens, seed=args.seed + index * 1009 + sample_index))
            if reference:
                old = reference[example.id]
                if old["answer"] != original.text or old["sampled_answers"] != [item.text for item in samples]:
                    raise ValueError(f"Regenerated answers differ from reference for {example.id}; stop and inspect environment/settings")
            total_tokens += sum(item.generated_tokens for item in samples)
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
            is_correct = label_score >= args.label_threshold
            if reference and bool(reference[example.id]["is_correct"]) != is_correct:
                raise ValueError(f"Correctness label changed for {example.id}; cannot compare on the same split")
            rows["y_error"].append(int(not is_correct))
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
                "generation_prompt": args.generation_prompt,
                "evaluation_profile": args.evaluation_profile,
                "answer": original.text, "sampled_answers": [item.text for item in samples],
                "is_correct": is_correct, "labeler": labeler.name, "label_score": label_score,
                "label_threshold": args.label_threshold,
                "generation_mean_log_probability": original.mean_log_probability,
                "consistency": consistency, "lexical_cluster_entropy": lexical_entropy,
            }
            if labels:
                record.update({"verify_probabilities": verify_probs, "control_probabilities": control_probs})
            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    np.savez_compressed(output / "features.npz", **arrays)
    (output / "manifest.json").write_text(json.dumps({
        "model": args.model, "data": str(Path(args.data).resolve()), "examples": len(examples),
        "generation_prompt": args.generation_prompt, "max_new_tokens": args.max_new_tokens,
        "generation_prompt_template": generation_prompt("{question}", args.generation_prompt),
        "evaluation": spec,
        "reference_run": str(Path(args.reference_run).resolve()) if args.reference_run else None,
        "layer_index_convention": "hidden_states tuple: 0=embedding; n>0=output after n blocks; negatives index from end",
        "resolved_layers": adapter._resolve_layers(args.layers, adapter.model.config.num_hidden_layers + 1),
        "offset": args.offset, "labeler": labeler.name, "label_threshold": args.label_threshold,
        "samples": args.samples, "temperature": args.temperature, "layers": args.layers,
        "seed": args.seed, "approximate_processed_label_and_generated_tokens": total_tokens,
        "assessment_prompt_input_tokens": assessment_prompt_tokens,
        "entropy_kind": "lexical_normalized-answer clusters",
    }, indent=2), encoding="utf-8")
