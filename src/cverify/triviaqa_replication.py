"""Independent TriviaQA detector fitting using the completed source protocol."""
import json
from pathlib import Path

import numpy as np

from .cli import build_parser
from .combined import analyze as analyze_combined
from .features import load_run_features
from .matched_passes import analyze as analyze_matched
from .review import generation_flags
from .storage import atomic_json, digest, sha256_file
from .train import run_training


def finalize_replication(config, full, source_settings, rows):
    """Use source hyperparameters, but fit every probe on target train only."""
    run, full = Path(config["run"]), Path(full)
    data, manifest = load_run_features(full)
    y = data["y_error"]
    if len(y) < 20 or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2 or np.min(np.bincount(y.astype(int))) < 5:
        raise ValueError("TriviaQA replication needs at least 20 answers and five of each EM class; do not change labels to force a split")
    settings = {key: source_settings[key] for key in ("c", "test_size", "dev_size")}
    code = {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    identity = digest({"settings": settings, "files": {name: sha256_file(full / name)
                      for name in ("features.npz", "manifest.json", "metadata.jsonl")}, "code": code})
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
    lines = ["TRIVIAQA: INDEPENDENT WITHIN-DATASET REPLICATION", config["external_protocol"],
             f"Counts: {counts}", "Detectors fitted on TriviaQA train; calibration on TriviaQA dev; evaluation on TriviaQA test.",
             "All exported questions retained. EM nonmatches can include factually correct prose.",
             "Repeated split seeds overlap. No test-based choice of method, layer or threshold in the fixed all-layer comparison.",
             "A shared pattern across these datasets requires checking the actual direction and uncertainty of the paired effects.",
             "Different label definitions prevent a claim of identical factual-error measurement across datasets.", "",
             (full / "matched_pass_detection" / "summary.txt").read_text(encoding="utf-8")]
    text = "\n".join(lines)
    (output / "summary.txt").write_text(text, encoding="utf-8")
    atomic_json(output / "report.json", {"protocol": config["external_protocol"], "counts": counts,
                "dataset": config["dataset_export"], "training_settings": settings, "fingerprint": identity,
                "matched_comparison": report})
    atomic_json(run / "finalize_status.json", {"phase": "complete", "review_hash": None,
                "results": str(full), "label_protocol": "official_exact_match",
                "analysis_mode": "within_dataset", "summary": str(output / "summary.txt")})
    print(text, flush=True)
