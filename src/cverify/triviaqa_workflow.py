"""Frozen TruthfulQA-trained detectors evaluated on a fixed TriviaQA subset."""
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from .combined import make_splits, paired_comparison, probabilities, uncertainty_metrics
from .data import load_jsonl
from .features import load_run_features
from .labeling import TriviaQAExactMatchLabeler
from .review import generation_flags, read_records
from .storage import atomic_json, atomic_npz, digest, sha256_file
from .transfer import fit_source
from .triviaqa_data import export_subset

MODELS = {"first_pass_state": "generation_state", "native_none": "none_state",
          "native_verify": "verify_state", "native_control": "control_state",
          "control_none_concat": "control_none_concat_state"}


def source_snapshot(source):
    names = ["manifest.json", "results_seed_42_original_layers/metrics.json", "results_seed_42_all_layers/metrics.json"]
    manifest = json.loads((source / "manifest.json").read_text())
    names.append("labels.npz" if manifest.get("feature_storage") == "indexed_view_v1" else "features.npz")
    return {name: sha256_file(source / name) for name in names}


def prepare_workflow(script):
    from .workflow import code_digest, load_json
    project = script.parent.parent.resolve()
    name = os.environ.get("CVERIFY_PIPELINE_RUN", "triviaqa_validation500_frozen_em")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Use a single new TriviaQA run directory name")
    run = project / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    path = run / "workflow_config.json"
    previous = load_json(path)
    if previous and previous.get("benchmark") != "triviaqa":
        raise ValueError("Refusing to reuse a non-TriviaQA run directory")
    truth = Path(os.environ.get("CVERIFY_TRUTHFULQA_RUN", str(project / "runs" / "truthfulqa_concise_chat_audit_l15"))).resolve()
    status = load_json(truth / "finalize_status.json")
    if not status or status.get("phase") != "complete":
        raise ValueError("A completed TruthfulQA source run is required")
    source = Path(status["results"])
    manifest = load_json(source / "manifest.json")
    if (manifest.get("layers") != [-1, -4, -8, 15] or manifest.get("generation_prompt") != "concise"
            or manifest.get("evaluation", {}).get("profile") != "assessment-v3"
            or manifest.get("samples") != 5 or manifest.get("temperature") != .7
            or manifest.get("chat_format") != "tokenizer" or manifest.get("seed") != 42):
        raise ValueError("TruthfulQA source does not match the frozen assessment-v3 protocol")
    revision = manifest.get("runtime", {}).get("model_commit")
    if not revision:
        raise ValueError("Source model revision is unavailable; cannot pin identical model weights")
    truth_config = load_json(truth / "workflow_config.json")
    if sha256_file(Path(truth_config["data"])) != truth_config.get("data_sha256"):
        raise ValueError("Source TruthfulQA question file changed; cannot verify question overlap")
    excluded = [example.question for example in load_jsonl(truth_config["data"])]
    size = int(os.environ.get("CVERIFY_TRIVIAQA_SIZE", "500"))
    dataset = run / "triviaqa.jsonl"
    exported = export_subset(dataset, size, 42, excluded)
    config = {"project": str(project), "run": str(run), "script": str(script), "data": str(dataset),
              "data_sha256": sha256_file(dataset), "code_sha256": code_digest(script), "benchmark": "triviaqa",
              "source_run": str(source), "source_snapshot": source_snapshot(source), "dataset_export": exported,
              "model": manifest["model"], "model_revision": revision,
              "expected_runtime": manifest["runtime"],
              "initial_tokens": manifest["max_new_tokens"], "retry_tokens": manifest["max_retry_tokens"],
              "shard_size": 25, "layers": manifest["layers"], "samples": 5, "temperature": .7, "seed": 42,
              "examples": size, "chat_format": "tokenizer", "labeler": "triviaqa-exact",
              "evaluation_profile": "assessment-v3", "generation_prompt": "concise",
              "cpus_per_task": 1, "max_submitted_jobs": 2,
              "external_protocol": "All-layer probes fitted on TruthfulQA seed42 train, calibrated on TruthfulQA dev; no TriviaQA fitting, selection or recalibration; official TriviaQA EM labels."}
    analysis = os.environ.get("CVERIFY_TRIVIAQA_ANALYSIS", "frozen")
    if analysis not in {"frozen", "within_dataset"}:
        raise ValueError("Unknown TriviaQA analysis mode")
    if analysis == "within_dataset":
        config["analysis_mode"] = analysis
        config["external_protocol"] = (
            "Independent TriviaQA train/dev/test experiment with the TruthfulQA representation and detector protocol; "
            "five split seeds 42-46, fixed all-layer matched comparison, official exact-match labels. "
            "No TruthfulQA-fitted detector is used for this comparison.")
    labels = os.environ.get("CVERIFY_TRIVIAQA_LABELS", "exact_match")
    if labels not in {"reviewed", "exact_match"} or (labels == "reviewed" and analysis != "within_dataset"):
        raise ValueError("Reviewed TriviaQA labels require within-dataset replication")
    if labels == "reviewed":
        config["label_protocol"] = "reviewed_factual_correctness"
        config["labeler"] = "deferred"
        config["external_protocol"] = (
            "Independent TriviaQA train/dev/test experiment with reviewed factual-correctness labels and "
            "the same review/exclusion and representation/detector protocol as TruthfulQA; "
            "five split seeds 42-46, fixed all-layer comparison. Official EM is supplementary only.")
    if previous is not None and previous != config:
        raise ValueError("Code/source/subset settings changed since TriviaQA started; restore them or choose a new TriviaQA run name")
    atomic_json(path, config)
    return path, config


def validate_target_protocol(source, target, expected_labeler="triviaqa-exact"):
    keys = ("model", "layers", "generation_prompt", "evaluation", "chat_format", "chat_template", "samples", "temperature",
            "max_new_tokens", "max_retry_tokens", "generation_state_position", "resolved_layers", "runtime")
    for key in keys:
        if source.get(key) != target.get(key):
            raise ValueError(f"TriviaQA/source representation protocol mismatch: {key}")
    if target.get("labeler") != expected_labeler:
        raise ValueError(f"TriviaQA target labeler must be {expected_labeler}")


def safe_metrics(y, p):
    if len(np.unique(y)) == 2:
        return uncertainty_metrics(y, p)
    return {"auroc": None, "auprc_error_positive": None, "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "accuracy_at_05": float(np.mean((p >= .5) == y)),
            "error_rate_at_80pct_coverage": float(y[np.argsort(p, kind="stable")[:int(math.ceil(.8 * len(y)))]].mean()),
            "limitation": "Only one label class: ranking discrimination cannot be evaluated"}


def model_matrices(data):
    return {"first_pass_state": data["generation_state"], "native_none": data["none_state"],
            "native_verify": data["verify_state"], "native_control": data["control_state"],
            "control_none_concat": np.c_[data["control_state"], data["none_state"]]}


def external_predictions(source_data, target_data, splits, c, output, fingerprint, original_metrics):
    source_x, target_x = model_matrices(source_data), model_matrices(target_data)
    y = source_data["y_error"]
    predictions, checks = {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        cache, parameters = output / f"{name}.json", output / f"{name}_parameters.npz"
        previous = json.loads(cache.read_text()) if cache.exists() else {}
        identity = digest({"run": fingerprint, "model": name})
        if previous.get("fingerprint") == identity and parameters.exists():
            p = np.asarray(previous["target_p_error"])
            if p.shape != (len(target_data["y_error"]),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("Invalid cached external probabilities")
            predictions[name], checks[name] = p, previous["source_test_metrics"]
            continue
        print(f"Fitting frozen source-only probe: {name}", flush=True)
        model, calibrator = fit_source(source_x[name], y, splits["train"], splits["dev"], c)
        source_p = probabilities(model, calibrator, source_x[name][splits["test"]])
        source_metrics = safe_metrics(y[splits["test"]], source_p)
        reference = original_metrics["models"][MODELS[name]]["metrics"]
        for key in ("auroc", "brier"):
            if not np.isclose(source_metrics[key], reference[key], atol=1e-6, rtol=0):
                raise ValueError(f"Frozen source baseline failed reproduction: {name}/{key}")
        p = probabilities(model, calibrator, target_x[name])
        atomic_npz(parameters, scaler_mean=model[0].mean_, scaler_scale=model[0].scale_,
                   classifier_coef=model[-1].coef_, classifier_intercept=model[-1].intercept_,
                   calibration_coef=calibrator.coef_, calibration_intercept=calibrator.intercept_)
        atomic_json(cache, {"fingerprint": identity, "target_p_error": p.tolist(), "source_test_metrics": source_metrics})
        predictions[name], checks[name] = p, source_metrics
    predictions["ensemble_mean"] = np.mean([predictions[f"native_{name}"] for name in ("none", "verify", "control")], axis=0)
    return predictions, checks


def finalize_external(config, full):
    run, source = Path(config["run"]), Path(config["source_run"])
    atomic_json(run / "finalize_status.json", {"phase": "evaluating", "review_hash": None})
    if source_snapshot(source) != config["source_snapshot"]:
        raise ValueError("Frozen TruthfulQA source artifacts changed")
    source_data, source_manifest = load_run_features(source)
    target_data, target_manifest = load_run_features(full)
    reviewed = config.get("label_protocol") == "reviewed_factual_correctness"
    validate_target_protocol(source_manifest, target_manifest, "deferred" if reviewed else "triviaqa-exact")
    saved = json.loads((source / "results_seed_42_original_layers" / "metrics.json").read_text())
    original = json.loads((source / "results_seed_42_all_layers" / "metrics.json").read_text())
    if original["configuration"]["selected_layers"] != source_manifest["layers"] or original["split"] != saved["split"]:
        raise ValueError("Source all-layer metrics do not match the declared split/layers")
    y = target_data["y_error"]
    if len(y) != config["examples"] or not np.isin(y, [-1] if reviewed else [0, 1]).all():
        raise ValueError("Incomplete or invalid TriviaQA extraction labels")
    rows = read_records(full / "metadata.jsonl")
    questions = load_jsonl(config["data"])
    scorer = TriviaQAExactMatchLabeler()
    if (len(rows) != len(y) or len(questions) != len(y)
            or (not reviewed and [int(not row["is_correct"]) for row in rows] != y.tolist())):
        raise ValueError("Target metadata labels and features differ")
    for row, question, label in zip(rows, questions, y):
        if (row["id"] != question.id or row["question"] != question.question or tuple(row["aliases"]) != question.answers
                or (not reviewed and int(not scorer.score(row["answer"], question.answers)) != label)):
            raise ValueError("Target references, sample order or official scores differ from exported subset")
    if config.get("analysis_mode") == "within_dataset":
        if reviewed:
            from .triviaqa_replication import finalize_reviewed_replication
            finalize_reviewed_replication(config, full, saved["configuration"], rows)
            return
        from .triviaqa_replication import finalize_replication
        finalize_replication(config, full, saved["configuration"], rows)
        return
    splits = make_splits(source_data["y_error"], 42, saved["configuration"], saved["split"])
    output = run / "external_validation"
    output.mkdir(parents=True, exist_ok=True)
    target_hashes = {name: sha256_file(full / name) for name in ("manifest.json", "features.npz", "metadata.jsonl")}
    fingerprint = digest({"config": config, "target_files": target_hashes,
                          "finalization_code": config.get("finalization_code_sha256", config["code_sha256"])})
    predictions, checks = external_predictions(source_data, target_data, splits, saved["configuration"]["c"],
                                               output, fingerprint, original)
    report = {"protocol": config["external_protocol"], "source_run": str(source), "source_artifacts": config["source_snapshot"],
              "dataset": config["dataset_export"], "fingerprint": fingerprint, "target_files_sha256": target_hashes,
              "counts": {"evaluated": len(y), "exact_match": int(np.sum(y == 0)), "non_match": int(np.sum(y == 1)),
                         "generation_flagged": sum(bool(generation_flags(row)) for row in rows), "quality_excluded": 0},
              "source_test_reproduction": checks, "methods": {name: safe_metrics(y, p) for name, p in predictions.items()},
              "paired_second_minus_first": {},
              "limitations": "Fixed subset of a different benchmark, unseen by detector fitting, not necessarily unseen during LLM pretraining. Target labels measure official exact answer matching, not reviewed factual correctness; source/target differences combine dataset and label-protocol shifts. No target recalibration or model/layer selection. Multiple assessment passes add compute. Bootstrap intervals condition on frozen source probes and are unadjusted."}
    if len(np.unique(y)) == 2:
        for name in ("native_none", "control_none_concat", "ensemble_mean"):
            report["paired_second_minus_first"][name] = paired_comparison(y, predictions[name], predictions["first_pass_state"], 1000, 42)
    atomic_json(output / "report.json", report)
    atomic_json(output / "predictions.json", {"ids": [row["id"] for row in rows], "y_em_error": y.tolist(),
                                              "p_error": {name: p.tolist() for name, p in predictions.items()}})
    lines = ["TRIVIAQA: FROZEN CROSS-DATASET / LABEL-PROTOCOL TRANSFER", config["external_protocol"],
             f"Evaluated={len(y)}, exact matches={np.sum(y == 0)}, nonmatches={np.sum(y == 1)}; all exported questions retained.",
             "Correct prose may fail whole-answer exact match; these are benchmark-match labels, not a new manual factual audit."]
    for name, m in report["methods"].items():
        auc = f"{m['auroc']:.4f}" if m["auroc"] is not None else "undefined"
        lines.append(f"{name}: AUROC={auc}, Brier={m['brier']:.4f}, risk80={m['error_rate_at_80pct_coverage']:.4f}, accuracy05={m['accuracy_at_05']:.4f}")
    for name, pair in report["paired_second_minus_first"].items():
        lines.append(f"{name} minus first: delta AUC={pair['auroc']['difference']:+.4f} CI={pair['auroc']['paired_95_ci']}; delta Brier={pair['brier']['difference']:+.4f} CI={pair['brier']['paired_95_ci']}")
    lines.append("Interpret as transfer of the frozen source protocol to benchmark-matching errors; do not claim equivalent factual label definitions or an equal-compute comparison.")
    text = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(text, encoding="utf-8")
    atomic_json(run / "finalize_status.json", {"phase": "complete", "review_hash": None, "results": str(output),
                                               "label_protocol": "official_exact_match", "summary": str(output / "summary.txt")})
    print(text, flush=True)
