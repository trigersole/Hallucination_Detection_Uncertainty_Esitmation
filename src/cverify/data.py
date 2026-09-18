"""Dataset loading and conservative short-answer labeling."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answers: tuple[str, ...]


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, aliases: tuple[str, ...]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(alias) for alias in aliases)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[Example]:
    rows: list[Example] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            answers = raw.get("answers")
            if not isinstance(answers, list) or not answers:
                raise ValueError(f"Line {line_number}: 'answers' must be a non-empty list")
            rows.append(Example(str(raw.get("id", line_number)), raw["question"], tuple(map(str, answers))))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Dataset contains no examples")
    return rows

