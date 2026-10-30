"""Answer-free baselines and reviewed same-question answer substitutions."""
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .combined import make_splits, paired_comparison, probabilities, uncertainty_metrics
from .extract import _unit
from .features import load_run_features
from .model import CausalLMAdapter
from .prompts import EVALUATION_PROFILES, evaluation_prompt, generation_prompt
from .review import read_records
from .storage import atomic_json, atomic_npz, digest, sha256_file
from .transfer import fit_source

NATIVE = ("none", "verify", "control")


def identity(run):
    manifest = json.loads((run / "manifest.json").read_text())
    names = ["manifest.json", "metadata.jsonl", "labels.npz" if manifest.get("feature_storage") == "indexed_view_v1" else "features.npz"]
    return {name: sha256_file(run / name) for name in names}


def implementation():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in
            ("attribution.py", "model.py", "prompts.py", "train.py", "transfer.py", "combined.py", "features.py")}


def load_context(run):
    data, manifest = load_run_features(run)
    rows = read_records(run / "metadata.jsonl")
    metrics_path = run / "results_seed_42_all_layers" / "metrics.json"
    saved = json.loads(metrics_path.read_text())
    if (len(rows) != len(data["y_error"]) or not np.isin(data["y_error"], [0, 1]).all()
            or saved["configuration"].get("audit") != manifest.get("audit")
            or saved["configuration"].get("selected_layers") != manifest["layers"]
            or manifest.get("generation_prompt") != "concise"
            or manifest.get("evaluation", {}).get("profile") != "assessment-v3"):
        raise ValueError("Completed aligned assessment-v3 all-layer metrics and binary labels required")
    splits = make_splits(data["y_error"], saved["configuration"]["seed"], saved["configuration"], saved["split"])
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    if any(type(row.get("is_correct")) is not bool for row in rows) or [int(not row["is_correct"]) for row in rows] != data["y_error"].tolist():
        raise ValueError("Reviewed metadata and feature labels differ")
    return data, manifest, rows, saved, splits


def question_prompts(question):
    verify, control, _ = EVALUATION_PROFILES["assessment-v3"]
    return {"first": generation_prompt(question, "concise"),
            "none": f"Question: {question}\nAssessment:",
            "verify": f"Question: {question}\nTask: {verify[0]}\nAssessment:",
            "control": f"Question: {question}\nTask: {control[0]}\nAssessment:"}


def adapter_for(manifest):
    runtime = manifest.get("runtime", {})
    if not runtime.get("model_commit"):
        raise ValueError("Saved model revision required")
    adapter = CausalLMAdapter(manifest["model"], chat_format=manifest["chat_format"],
                              dtype=runtime["dtype"], revision=runtime["model_commit"])
    actual = {"model_commit": getattr(adapter.model.config, "_commit_hash", None),
              "chat_template": adapter.tokenizer.chat_template,
              "chat_template_source": adapter.chat_template_source, "dtype": runtime["dtype"]}
    for package in ("torch", "transformers", "numpy"):
        try:
            actual[package] = version(package)
        except PackageNotFoundError:
            actual[package] = None
    if runtime != actual:
        raise ValueError("Model/tokenizer/library runtime differs from original extraction")
    return adapter


def extract_prompts(adapter, prompts, layers):
    result = {}
    for name, prompt in prompts.items():
        hidden, _ = adapter.prompt_hidden(prompt, layers)
        result[name] = np.concatenate([_unit(hidden[layer]) for layer in layers]).astype(np.float32)
        if not np.isfinite(result[name]).all():
            raise ValueError("Nonfinite extracted state")
    return result


def extract_question_features(run, output):
    _, manifest, rows, _, _ = load_context(run)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = digest({"source": identity(run), "code": implementation(), "question_prompts": question_prompts("{question}")})
    progress = output / "progress.json"
    if progress.exists() and json.loads(progress.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Question extraction inputs/code changed; choose a separate output directory")
    atomic_json(progress, {"fingerprint": fingerprint, "source": identity(run)})
    if (output / "complete.json").exists() and (output / "features.npz").exists():
        return
    adapter = adapter_for(manifest)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    features = {name: [] for name in question_prompts("")}
    for i, row in enumerate(rows):
        path = checkpoint_dir / f"{i}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                states = {name: archive[name] for name in features}
        else:
            states = extract_prompts(adapter, question_prompts(row["question"]), manifest["layers"])
            atomic_npz(path, **states)
        for name in features:
            features[name].append(states[name])
        print(f"Question-only {i+1}/{len(rows)}", flush=True)
    atomic_npz(output / "features.npz", **{name: np.asarray(values) for name, values in features.items()})
    atomic_json(output / "complete.json", {"fingerprint": fingerprint, "ids": [row["id"] for row in rows],
                                          "features_sha256": sha256_file(output / "features.npz")})


def score_questions(run, extracted, output, seeds=(42, 43, 44, 45, 46), bootstrap=1000):
    data, manifest, rows, saved, _ = load_context(run)
    marker = json.loads((extracted / "complete.json").read_text())
    expected = digest({"source": identity(run), "code": implementation(), "question_prompts": question_prompts("{question}")})
    if (marker["fingerprint"] != expected or marker["ids"] != [row["id"] for row in rows]
            or marker["features_sha256"] != sha256_file(extracted / "features.npz")):
        raise ValueError("Question-only provenance/order/features changed")
    with np.load(extracted / "features.npz") as archive:
        q = {name: archive[name] for name in ("first",) + NATIVE}
    y = data["y_error"]
    pairs = {"first": "generation_state", **{name: f"{name}_state" for name in NATIVE}}
    if any(x.shape != data["generation_state"].shape or not np.isfinite(x).all() for x in q.values()):
        raise ValueError("Invalid question-only state matrices")
    output.mkdir(parents=True, exist_ok=True)
    report = {"source": identity(run), "code": implementation(), "seeds": {},
              "protocol": "Identical per-question splits, training-only scaling/probe fitting and dev-only calibration. Full-minus-question-only effects. No answers or references enter question-only prompts.",
              "limitations": "Question-only predicts correctness of one observed answer per question and can capture question difficulty. Omitting the answer changes context and position; this is an ablation, not causal attribution. Splits overlap; inspected data; unadjusted conditional intervals."}
    fingerprint = digest({"report": report, "extraction": marker, "bootstrap": bootstrap, "seeds": list(seeds)})
    lines = ["QUESTION-ONLY VERSUS ANSWER-CONDITIONED DETECTION", report["protocol"], report["limitations"]]
    for seed in seeds:
        path = output / f"seed_{seed}.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        if previous.get("fingerprint") == fingerprint:
            result = previous
        else:
            splits = make_splits(y, seed, saved["configuration"], saved["split"])
            result = {"fingerprint": fingerprint, "conditions": {}, "split": {k: v.tolist() for k, v in splits.items()}}
            for name, key in pairs.items():
                probabilities_by_mode = {}
                for mode, x in (("question_only", q[name]), ("full", data[key])):
                    base, cal = fit_source(x, y, splits["train"], splits["dev"], saved["configuration"]["c"])
                    probabilities_by_mode[mode] = probabilities(base, cal, x[splits["test"]])
                p = probabilities_by_mode
                result["conditions"][name] = {"metrics": {mode: uncertainty_metrics(y[splits["test"]], values) for mode, values in p.items()},
                    "full_minus_question_only": paired_comparison(y[splits["test"]], p["full"], p["question_only"], bootstrap, seed),
                    "y_error": y[splits["test"]].tolist(), "p_error": {mode: values.tolist() for mode, values in p.items()}}
            atomic_json(path, result)
        report["seeds"][str(seed)] = result
        for name, values in result["conditions"].items():
            comparison = values["full_minus_question_only"]
            lines.append(f"seed={seed} {name}: question AUC={values['metrics']['question_only']['auroc']:.4f}, full AUC={values['metrics']['full']['auroc']:.4f}; delta={comparison['auroc']['difference']:+.4f} CI={comparison['auroc']['paired_95_ci']}; delta Brier={comparison['brier']['difference']:+.4f}")
    atomic_json(output / "report.json", report)
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def export_pairs(run, destination):
    _, _, rows, _, splits = load_context(run)
    if destination.exists():
        return  # Preserve edits. Validation on extraction rejects changed questions.
    pairs = []
    for i in splits["test"]:
        row = rows[i]
        correct = row["answer"] if row["is_correct"] else next(iter(row["aliases"]), "")
        incorrect = row["answer"] if not row["is_correct"] else next(iter(row.get("incorrect_references", [])), "")
        pairs.append({"question_id": row["id"], "question": row["question"], "correct_answer": correct,
                      "incorrect_answer": incorrect, "reviewer": "", "notes": "", "skip_reason": "",
                      "instructions": "Candidates are unverified. Confirm both answers, comparable style/length and factual labels; fill reviewer. Or record skip_reason. Do not use model scores to choose pairs."})
    atomic_json(destination, {"source": identity(run), "split_seed": 42, "pairs": pairs})


def validated_pairs(run, path):
    _, _, rows, _, splits = load_context(run)
    document = json.loads(path.read_text())
    test = {rows[i]["id"]: rows[i] for i in splits["test"]}
    candidates = document["pairs"]
    if (document["source"] != identity(run) or document.get("split_seed") != 42
            or len({p["question_id"] for p in candidates}) != len(candidates)
            or {p["question_id"] for p in candidates} != set(test)):
        raise ValueError("Pair document provenance or test-question membership changed")
    kept = []
    for pair in candidates:
        if pair["question"] != test[pair["question_id"]]["question"]:
            raise ValueError("Pair question changed")
        if pair.get("skip_reason", "").strip():
            continue
        if (not pair.get("reviewer", "").strip() or not pair["correct_answer"].strip()
                or not pair["incorrect_answer"].strip() or pair["correct_answer"].strip() == pair["incorrect_answer"].strip()):
            raise ValueError("Every non-skipped pair needs reviewed distinct correct/incorrect answers and reviewer")
        kept.append(pair)
    if len(kept) < 10:
        raise ValueError("Need at least 10 reviewed held-out question pairs; report selection/skips")
    return kept, document


def extract_pairs(run, pair_file, output):
    pairs, document = validated_pairs(run, pair_file)
    _, manifest, _, _, _ = load_context(run)
    fingerprint = digest({"document": document, "code": implementation()})
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    if progress.exists() and json.loads(progress.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Reviewed pair/code changed; choose a separate output directory")
    atomic_json(progress, {"fingerprint": fingerprint})
    if (output / "complete.json").exists() and (output / "features.npz").exists():
        return
    adapter = adapter_for(manifest)
    verify, control, _ = EVALUATION_PROFILES["assessment-v3"]
    tasks = {"none": None, "verify": verify[0], "control": control[0]}
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    arrays = {name: [] for name in NATIVE}
    for i, pair in enumerate(pairs):
        path = checkpoints / f"{i}.npz"
        if path.exists():
            with np.load(path) as archive:
                states = {name: archive[name] for name in NATIVE}
        else:
            alternatives = [extract_prompts(adapter, {name: evaluation_prompt(pair["question"], pair[key], task, "assessment-v3")
                            for name, task in tasks.items()}, manifest["layers"])
                            for key in ("correct_answer", "incorrect_answer")]
            states = {name: np.stack([answer[name] for answer in alternatives]) for name in NATIVE}
            atomic_npz(path, **states)
        for name in NATIVE:
            arrays[name].append(states[name])
        print(f"Reviewed pair {i+1}/{len(pairs)}", flush=True)
    atomic_npz(output / "features.npz", **{name: np.asarray(values) for name, values in arrays.items()})
    atomic_json(output / "complete.json", {"fingerprint": fingerprint, "ids": [pair["question_id"] for pair in pairs],
                                          "features_sha256": sha256_file(output / "features.npz")})


def pair_statistics(p, repetitions=1000, seed=42):
    if repetitions < 1 or len(p) < 1 or p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite per-question [correct, incorrect] probability pairs")
    delta = p[:, 1] - p[:, 0]
    ordering = (delta > 0).astype(float) + .5 * (delta == 0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(p), size=(repetitions, len(p)))
    return {"questions": len(p), "mean_incorrect_minus_correct_p_error": float(delta.mean()),
            "mean_difference_95_ci": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
            "pair_order_accuracy_ties_half": float(ordering.mean()),
            "pair_order_accuracy_95_ci": np.quantile(ordering[draws].mean(axis=1), [.025, .975]).tolist(),
            "pooled_answer_auroc": float(roc_auc_score(np.tile([0, 1], len(p)), p.ravel())),
            "resampling_unit": "question pair; both answers retained in every bootstrap draw"}


def score_pairs(run, pair_file, extracted, output, bootstrap=1000):
    pairs, document = validated_pairs(run, pair_file)
    data, _, _, saved, splits = load_context(run)
    marker = json.loads((extracted / "complete.json").read_text())
    if (marker["fingerprint"] != digest({"document": document, "code": implementation()})
            or marker["ids"] != [pair["question_id"] for pair in pairs]
            or marker["features_sha256"] != sha256_file(extracted / "features.npz")):
        raise ValueError("Pair feature provenance mismatch")
    with np.load(extracted / "features.npz") as archive:
        states = {name: archive[name] for name in NATIVE}
    prediction = {}
    for name in NATIVE:
        x = states[name]
        if x.shape != (len(pairs), 2, data[f"{name}_state"].shape[1]):
            raise ValueError("Pair state shape mismatch")
        base, cal = fit_source(data[f"{name}_state"], data["y_error"], splits["train"], splits["dev"], saved["configuration"]["c"])
        source_p = probabilities(base, cal, data[f"{name}_state"][splits["test"]])
        source_metrics = uncertainty_metrics(data["y_error"][splits["test"]], source_p)
        for metric in ("auroc", "brier"):
            if not np.isclose(source_metrics[metric], saved["models"][f"{name}_state"]["metrics"][metric], rtol=0, atol=1e-6):
                raise ValueError(f"Original pair-test probe failed reproduction: {name}/{metric}")
        prediction[name] = probabilities(base, cal, x.reshape(-1, x.shape[-1])).reshape(len(pairs), 2)
    prediction["ensemble_mean"] = np.mean(list(prediction.values()), axis=0)
    report = {"source": identity(run), "pair_file_sha256": sha256_file(pair_file), "code": implementation(),
              "protocol": "Freeze original seed42 train/dev native probes. Evaluate only original test questions, each with reviewed correct and incorrect alternatives. No fitting on substituted answers.",
              "limitations": "Exploratory test on previously inspected questions. Reviewer/selection, answer length, style and lexical differences can affect scores. This controls question identity, not all answer confounds; it does not prove factual reasoning. No generated-answer accuracy claim.",
              "question_only_pair_order_reference": .5,
              "exported_questions": len(document["pairs"]), "skipped_questions": len(document["pairs"]) - len(pairs),
              "methods": {name: pair_statistics(p, bootstrap) for name, p in prediction.items()}}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    atomic_json(output / "predictions.json", {"question_ids": marker["ids"], "answer_order": ["correct", "incorrect"],
                                              "p_error": {name: p.tolist() for name, p in prediction.items()}})
    lines = ["REVIEWED SAME-QUESTION ANSWER SUBSTITUTIONS", report["protocol"], report["limitations"]]
    for name, m in report["methods"].items():
        lines.append(f"{name}: pair-order accuracy={m['pair_order_accuracy_ties_half']:.4f} CI={m['pair_order_accuracy_95_ci']}; mean p(incorrect)-p(correct)={m['mean_incorrect_minus_correct_p_error']:+.4f} CI={m['mean_difference_95_ci']}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
