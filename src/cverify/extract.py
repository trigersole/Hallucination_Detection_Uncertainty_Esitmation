"""Feature extraction for paired verification and control conditions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import exact_match, load_jsonl
from .model import CausalLMAdapter
from .prompts import CONTROL_TEMPLATES, VERIFY_TEMPLATES, evaluation_prompt, generation_prompt
from .signals import lexical_cluster_statistics, rejection_and_disagreement


def _unit(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), eps)


def run_extraction(args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    examples = load_jsonl(args.data, args.limit)
    adapter = CausalLMAdapter(args.model, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    rows: dict[str, list[np.ndarray | float | int]] = {
        "y_error": [], "generation": [], "verification_scores": [],
        "generation_state": [],
        "verify_state": [], "control_state": [], "concat_state": [], "difference_state": [],
    }
    metadata_path = output / "metadata.jsonl"
    total_tokens = 0
    with metadata_path.open("w", encoding="utf-8") as meta:
        for index, example in enumerate(tqdm(examples, desc="Extracting")):
            prompt = generation_prompt(example.question)
            original = adapter.generate(prompt, args.layers, temperature=0.0, max_new_tokens=args.max_new_tokens, seed=args.seed + index)
            samples = [original]
            for sample_index in range(1, args.samples):
                samples.append(adapter.generate(prompt, args.layers, temperature=args.temperature, max_new_tokens=args.max_new_tokens, seed=args.seed + index * 1009 + sample_index))
            total_tokens += sum(item.generated_tokens for item in samples)
            consistency, lexical_entropy = lexical_cluster_statistics([item.text for item in samples])
            verify_probs, control_probs = [], []
            verify_states, control_states, differences = [], [], []
            for verify_instruction, control_instruction in zip(VERIFY_TEMPLATES, CONTROL_TEMPLATES):
                v_prompt = evaluation_prompt(example.question, original.text, verify_instruction)
                c_prompt = evaluation_prompt(example.question, original.text, control_instruction)
                v_prob, v_hidden, v_tokens = adapter.label_probability_and_hidden(v_prompt, args.layers)
                c_prob, c_hidden, c_tokens = adapter.label_probability_and_hidden(c_prompt, args.layers)
                total_tokens += v_tokens + c_tokens
                verify_probs.append(v_prob)
                control_probs.append(c_prob)
                v_vec = np.concatenate([_unit(v_hidden[layer]) for layer in args.layers])
                c_vec = np.concatenate([_unit(c_hidden[layer]) for layer in args.layers])
                verify_states.append(v_vec)
                control_states.append(c_vec)
                differences.append(v_vec - c_vec)
            rejection, disagreement = rejection_and_disagreement(verify_probs)
            gen_features = np.array([original.mean_log_probability, consistency, lexical_entropy], dtype=np.float32)
            verification_features = np.array([np.mean(verify_probs), rejection, disagreement, np.std(verify_probs)], dtype=np.float32)
            generation_state = np.concatenate([_unit(original.hidden_by_layer[layer]) for layer in args.layers]).astype(np.float32)
            verify_flat = np.concatenate(verify_states).astype(np.float32)
            control_flat = np.concatenate(control_states).astype(np.float32)
            difference_flat = np.concatenate(differences).astype(np.float32)
            is_correct = exact_match(original.text, example.answers)
            rows["y_error"].append(int(not is_correct))
            rows["generation"].append(gen_features)
            rows["verification_scores"].append(verification_features)
            rows["generation_state"].append(generation_state)
            rows["verify_state"].append(verify_flat)
            rows["control_state"].append(control_flat)
            rows["concat_state"].append(np.concatenate([verify_flat, control_flat]))
            rows["difference_state"].append(difference_flat)
            meta.write(json.dumps({
                "id": example.id, "question": example.question, "aliases": example.answers,
                "answer": original.text, "sampled_answers": [item.text for item in samples],
                "is_correct": is_correct, "generation_mean_log_probability": original.mean_log_probability,
                "consistency": consistency, "lexical_cluster_entropy": lexical_entropy,
                "verify_probabilities": verify_probs, "control_probabilities": control_probs,
            }, ensure_ascii=False) + "\n")
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    np.savez_compressed(output / "features.npz", **arrays)
    (output / "manifest.json").write_text(json.dumps({
        "model": args.model, "data": str(Path(args.data).resolve()), "examples": len(examples),
        "samples": args.samples, "temperature": args.temperature, "layers": args.layers,
        "seed": args.seed, "approximate_processed_label_and_generated_tokens": total_tokens,
        "entropy_kind": "lexical_normalized-answer clusters",
    }, indent=2), encoding="utf-8")
