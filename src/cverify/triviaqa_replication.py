"""Independent TriviaQA detector fitting using the completed source protocol."""
import json
from pathlib import Path

import numpy as np

from .cli import build_parser
from .combined import analyze as analyze_combined
from .features import load_run_features
from .matched_passes import analyze as analyze_matched
from .labeling import TriviaQAExactMatchLabeler
from .review import apply_review, export_review, generation_flags, read_records
from .storage import atomic_json, digest, sha256_file
from .train import run_training


def finalize_reviewed_replication(config, full, source_settings, rows):
    """Pause for the same answer-bound review and exclusion rules as TruthfulQA."""
    run, full = Path(config["run"]), Path(full)
    review = run / "review.jsonl"
    export_review(full, review)
    review_hash = sha256_file(review)
    audited = run / "audited" / review_hash[:16]
    # Keep EM away from the primary training labels and exported review fields.
    scorer = TriviaQAExactMatchLabeler()
    matches = [bool(scorer.score(row["answer"], row["aliases"])) for row in rows]
    atomic_json(run / "supplementary_exact_match.json", {
        "role": "Supplementary generator benchmark score only; never used as factual training labels",
        "metadata_sha256": sha256_file(full / "metadata.jsonl"),
        "extracted": len(rows), "exact_match_count": sum(matches),
        "exact_match_rate": sum(matches) / len(rows),
        "answers": [{"id": row["id"], "exact_match": match} for row, match in zip(rows, matches)]})
    try:
        summary = apply_review(full, review, audited)
    except ValueError as error:
        atomic_json(run / "finalize_status.json", {"phase": "needs_review", "review_hash": review_hash,
                    "message": str(error)})
        return
    if summary["counts"]["pending"]:
        atomic_json(run / "finalize_status.json", {"phase": "needs_review", "review_hash": review_hash,
                    "message": f"Review {summary['counts']['pending']} answers in {review}; fill judgment and reviewer. "
                               "Use factual correctness, not exact match. Then resume the same controller command."})
        return
    with np.load(audited / "labels.npz") as labels:
        y = labels["y_error"]
    if len(y) < 20 or len(np.unique(y)) != 2 or np.min(np.bincount(y.astype(int))) < 5:
        atomic_json(run / "finalize_status.json", {"phase": "needs_review", "review_hash": review_hash,
                    "message": "Need at least 20 usable reviewed answers and five of each class after exclusions. "
                               "Check the review summary; do not alter valid judgments to force a split."})
        return
    finalize_replication({**config, "applied_review_sha256": review_hash}, audited, source_settings,
                         read_records(audited / "metadata.jsonl"))


def finalize_replication(config, full, source_settings, rows):
    """Use source hyperparameters, but fit every probe on target train only."""
    run, full = Path(config["run"]), Path(full)
    data, manifest = load_run_features(full)
    y = data["y_error"]
    if len(y) < 20 or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2 or np.min(np.bincount(y.astype(int))) < 5:
        raise ValueError("TriviaQA replication needs at least 20 answers and five of each label class; do not change labels to force a split")
    settings = {key: source_settings[key] for key in ("c", "test_size", "dev_size")}
    code = {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    archive = "labels.npz" if manifest.get("feature_storage") == "indexed_view_v1" else "features.npz"
    identity = digest({"settings": settings, "files": {name: sha256_file(full / name)
                      for name in (archive, "manifest.json", "metadata.jsonl")}, "code": code})
    # Same original three-layer, middle-layer and all-layer analyses as TruthfulQA.
    for group, layers in (("original_layers", [-1, -4, -8]), ("layer15", [15]), ("all_layers", manifest["layers"])):
        output = full / f"results_seed_42_{group}"
        marker = output / "replication_training_complete.json"
        expected = {"fingerprint": identity, "layers": layers}
        previous = json.loads(marker.read_text()) if marker.exists() else None
        if previous != expected or not (output / "metrics.json").exists():
            args = build_parser().parse_args([
                "train", "--run", str(full), "--output", str(output), "--layers", *map(str, layers),
                "--seed", "42", "--c", str(settings["c"]), "--test-size", str(settings["test_size"]),
                "--dev-size", str(settings["dev_size"]), "--bootstrap", "1000",
            ])
            run_training(args)
            atomic_json(marker, expected)
    # Reuse exactly the implementations used for the completed TruthfulQA study.
    analyze_combined(full, full / "results_seed_42_original_layers" / "metrics.json",
                     full / "combined_detectors", [42, 43, 44, 45, 46], 1000)
    report = analyze_matched(full, full / "matched_pass_detection", [42, 43, 44, 45, 46], 1000)
    output = run / "dataset_replication"
    output.mkdir(parents=True, exist_ok=True)
    counts = {"evaluated": len(y), "exact_match": int(np.sum(y == 0)), "non_match": int(np.sum(y == 1)),
              "generation_flagged": sum(bool(generation_flags(row)) for row in rows), "quality_excluded": 0}
    reviewed = config.get("label_protocol") == "reviewed_factual_correctness"
    if reviewed:
        counts = {"evaluated": len(y), "correct": int(np.sum(y == 0)), "incorrect": int(np.sum(y == 1)),
                  "review_summary": manifest["audit"]}
    labeling_note = ("Primary labels are reviewed factual correctness. Ambiguous/refusal/exclude and quality flags "
                     "are handled by the same review code as TruthfulQA. Official EM is supplementary only."
                     if reviewed else "All exported questions retained. EM nonmatches can include factually correct prose.")
    supplementary = None
    if reviewed:
        em = json.loads((run / "supplementary_exact_match.json").read_text())
        supplementary = {key: value for key, value in em.items() if key != "answers"}
        labeling_note += (f" Supplementary generator EM: {em['exact_match_count']}/{em['extracted']} "
                          f"({em['exact_match_rate']:.4f}), measured on all extracted answers.")
    lines = ["TRIVIAQA: INDEPENDENT WITHIN-DATASET REPLICATION", config["external_protocol"],
             f"Counts: {counts}", "Detectors fitted on TriviaQA train; calibration on TriviaQA dev; evaluation on TriviaQA test.",
             labeling_note,
             "Repeated split seeds overlap. No test-based choice of method, layer or threshold in the fixed all-layer comparison.",
             "A shared pattern across these datasets requires checking the actual direction and uncertainty of the paired effects.",
             ("Both datasets use the same review protocol; reviewer reliability and dataset-specific effects remain limitations."
              if reviewed else "Different label definitions prevent a claim of identical factual-error measurement across datasets."), "",
             (full / "matched_pass_detection" / "summary.txt").read_text(encoding="utf-8")]
    text = "\n".join(lines)
    (output / "summary.txt").write_text(text, encoding="utf-8")
    atomic_json(output / "report.json", {"protocol": config["external_protocol"], "counts": counts,
                "dataset": config["dataset_export"], "training_settings": settings, "fingerprint": identity,
                "applied_review_sha256": config.get("applied_review_sha256"), "supplementary_exact_match": supplementary,
                "matched_comparison": report})
    atomic_json(run / "finalize_status.json", {"phase": "complete", "review_hash": config.get("applied_review_sha256") if reviewed else None,
                "results": str(full), "label_protocol": "reviewed_factual_correctness" if reviewed else "official_exact_match",
                "analysis_mode": "within_dataset", "summary": str(output / "summary.txt")})
    print(text, flush=True)
