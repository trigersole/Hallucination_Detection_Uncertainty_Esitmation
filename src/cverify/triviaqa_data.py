"""Pinned TriviaQA export and official normalization (no evidence in prompts)."""
import hashlib
import json
from pathlib import Path
import re
import string

from .data import normalize_answer
from .storage import atomic_json, digest, sha256_file

DATASET = "mandarjoshi/trivia_qa"
CONFIG = "unfiltered.nocontext"
SPLIT = "validation"
EVALUATOR_URL = "https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py"


def normalize_triviaqa(text):
    """Match the authors' evaluator: lower, underscores/punctuation, articles, spaces."""
    text = text.lower().replace("_", " ")
    punctuation = set(string.punctuation + "‘’´`")
    text = "".join(" " if character in punctuation else character for character in text)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def select_subset(rows, n, seed, excluded_questions):
    if n < 1:
        raise ValueError("TriviaQA subset size must be positive")
    excluded = {normalize_answer(question) for question in excluded_questions}
    by_id = {}
    for row in rows:
        qid, question = str(row["question_id"]), row["question"]
        aliases = row["answer"].get("normalized_aliases")
        if not isinstance(aliases, list) or not aliases or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError(f"Missing official answer aliases for TriviaQA question {qid}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Empty TriviaQA question")
        record = {"id": f"triviaqa-{qid}", "question": question, "answers": list(dict.fromkeys(aliases)),
                  "source_question_id": qid, "answer_value": row["answer"].get("value"),
                  "raw_aliases": list(row["answer"].get("aliases", []))}
        if qid in by_id and by_id[qid] != record:
            raise ValueError(f"Conflicting duplicate TriviaQA question ID {qid}")
        by_id[qid] = record
    ranked = sorted(by_id.values(), key=lambda row: (hashlib.sha256(f"{seed}|{row['source_question_id']}".encode()).hexdigest(), row["id"]))
    selected, seen = [], set()
    overlap = duplicates = 0
    for row in ranked:
        key = normalize_answer(row["question"])
        if key in excluded:
            overlap += 1
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        selected.append(row)
    if len(selected) < n:
        raise ValueError(f"Only {len(selected)} unique non-overlapping questions available, requested {n}")
    return selected[:n], {"eligible_unique_questions": len(selected), "overlap_questions_removed": overlap,
                          "duplicate_question_texts_removed": duplicates, "source_unique_ids": len(by_id)}


def export_subset(destination, n, seed, excluded_questions):
    destination = Path(destination)
    metadata_path = destination.with_suffix(".manifest.json")
    settings = {"dataset": DATASET, "configuration": CONFIG, "split": SPLIT, "n": n, "seed": seed,
                "excluded_questions_sha256": digest(sorted({normalize_answer(q) for q in excluded_questions}))}
    if destination.exists() or metadata_path.exists():
        if not destination.exists() or not metadata_path.exists():
            raise ValueError("Incomplete TriviaQA export; preserve it and inspect before restarting")
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in settings.items()) or metadata["jsonl_sha256"] != sha256_file(destination):
            raise ValueError("TriviaQA subset settings or exported data changed")
        return metadata
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(DATASET).sha
    if not revision:
        raise ValueError("Dataset revision is unavailable; refusing an unpinned export")
    dataset = load_dataset(DATASET, CONFIG, split=SPLIT, revision=revision)
    selected, counts = select_subset(dataset, n, seed, excluded_questions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    metadata = {**settings, **counts, "dataset_revision": revision, "dataset_fingerprint": dataset._fingerprint,
                "jsonl_sha256": sha256_file(destination), "selected_ids_sha256": digest([row["id"] for row in selected]),
                "label_protocol": "official_whole_answer_normalized_exact_match", "evaluator": EVALUATOR_URL}
    atomic_json(metadata_path, metadata)
    return metadata
