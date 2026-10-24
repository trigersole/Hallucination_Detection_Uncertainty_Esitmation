"""Export answer review and bind external correctness judgments to exact answers."""
import json
from pathlib import Path

import numpy as np

from .storage import atomic_json, atomic_npz, digest, sha256_file


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_hash(row: dict) -> str:
    return digest({key: row[key] for key in ("id", "question", "answer", "aliases")})


def generation_flags(row: dict) -> list[str]:
    flags = []
    details = row.get("generation_details", [])
    if not details or any(item["stop_reason"] != "eos" for item in details):
        flags.append("one_or_more_generations_did_not_stop_with_eos")
    if any(item.get("contains_new_question") for item in details):
        flags.append("generated_new_question")
    if not row["answer"].strip():
        flags.append("empty_answer")
    return flags


def export_review(run: Path, destination: Path) -> None:
    if destination.exists():
        return  # Never overwrite the user's judgments on resume.
    rows = read_records(run / "metadata.jsonl")
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "id": row["id"], "sample_hash": sample_hash(row),
                "question": row["question"], "answer": row["answer"],
                "references": row["aliases"], "incorrect_references": row.get("incorrect_references", []),
                "generation_flags": generation_flags(row),
                "judgment": None, "reviewer": "", "notes": "",
            }, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def apply_review(run: Path, review_path: Path, output: Path) -> dict:
    rows = read_records(run / "metadata.jsonl")
    reviews = read_records(review_path)
    by_id = {row["id"]: row for row in reviews}
    if len(by_id) != len(reviews) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Review must contain each extracted question ID exactly once, with no extra IDs")
    kept, labels, audited = [], [], []
    counts = {key: 0 for key in ("pending", "correct", "incorrect", "ambiguous", "refusal", "exclude", "quality_excluded")}
    for index, row in enumerate(rows):
        review = by_id[row["id"]]
        if review.get("sample_hash") != sample_hash(row) or review.get("answer") != row["answer"] or review.get("question") != row["question"]:
            raise ValueError(f"Review belongs to different question/answer text: {row['id']}")
        judgment = review.get("judgment")
        if judgment is None or judgment == "":
            counts["pending"] += 1
            continue
        if judgment not in {"correct", "incorrect", "ambiguous", "refusal", "exclude"}:
            raise ValueError(f"Unknown judgment {judgment!r} for {row['id']}")
        if not review.get("reviewer", "").strip():
            raise ValueError(f"Record the reviewer/source of the judgment for {row['id']}")
        counts[judgment] += 1
        if generation_flags(row):
            counts["quality_excluded"] += 1
            continue
        if judgment in {"correct", "incorrect"}:
            kept.append(index)
            labels.append(int(judgment == "incorrect"))
            audited.append({**row, "is_correct": judgment == "correct", "labeler": "audited_labels",
                            "label_score": None, "label_threshold": None,
                            "review": review, "source_index_before_review": index})
    summary = {"counts": counts, "extracted": len(rows), "included": len(kept),
               "excluded": len(rows) - len(kept) - counts["pending"],
               "review_sha256": digest(reviews), "source_answer_hashes": digest([sample_hash(row) for row in rows])}
    atomic_json(review_path.with_suffix(".summary.json"), summary)
    if counts["pending"]:
        return summary
    output.mkdir(parents=True, exist_ok=True)
    with np.load(run / "features.npz", allow_pickle=False) as archive:
        if len(archive["y_error"]) != len(rows):
            raise ValueError("Source metadata and feature row counts differ")
    # Keep the large archive in place; audited versions store only indices and labels.
    atomic_npz(output / "labels.npz", source_indices=np.asarray(kept, dtype=np.int64),
               y_error=np.asarray(labels, dtype=np.int64))
    temporary = output / "metadata.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output / "metadata.jsonl")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"examples": len(kept), "labeler": "audited_labels", "label_threshold": None,
                     "audit": summary, "source_run": str(run.resolve()),
                     "feature_storage": "indexed_view_v1",
                     "feature_view": {"source_run": str(run.resolve()), "source_examples": len(rows),
                                      "source_sha256": {name: sha256_file(run / name)
                                                        for name in ("features.npz", "metadata.jsonl", "manifest.json")}}})
    atomic_json(output / "manifest.json", manifest)
    return summary
