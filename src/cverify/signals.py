"""Small, model-independent uncertainty computations."""

from __future__ import annotations

import math
from collections import Counter

from .data import normalize_answer


def binary_entropy(p: float, eps: float = 1e-12) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def rejection_and_disagreement(probabilities: list[float]) -> tuple[float, float]:
    if not probabilities:
        raise ValueError("At least one probability is required")
    mean = sum(probabilities) / len(probabilities)
    disagreement = binary_entropy(mean) - sum(map(binary_entropy, probabilities)) / len(probabilities)
    return 1.0 - mean, max(0.0, disagreement)


def lexical_cluster_statistics(answers: list[str]) -> tuple[float, float]:
    """Return normalized modal consistency and entropy over exact normalized clusters."""
    if not answers:
        raise ValueError("At least one answer is required")
    counts = Counter(normalize_answer(answer) for answer in answers)
    total = len(answers)
    consistency = max(counts.values()) / total
    entropy = -sum((n / total) * math.log(n / total) for n in counts.values())
    return consistency, entropy

