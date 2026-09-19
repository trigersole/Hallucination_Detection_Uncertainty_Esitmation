"""Reference-based correctness labelers."""

from __future__ import annotations

from .data import exact_match


class ExactMatchLabeler:
    name = "exact_match"

    def score(self, answer: str, aliases: tuple[str, ...]) -> float:
        return float(exact_match(answer, aliases))


class BleurtLabeler:
    """Maximum BLEURT score against acceptable reference answers."""

    name = "bleurt"

    def __init__(self, model_path: str, device: str = "cuda"):
        try:
            import torch
            from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BLEURT labeling requires bleurt_pytorch. Point PYTHONPATH at the "
                "HaloScope dependency overlay or install that package."
            ) from exc
        self.torch = torch
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = BleurtTokenizer.from_pretrained(model_path)
        self.model = BleurtForSequenceClassification.from_pretrained(model_path).to(self.device).eval()

    def score(self, answer: str, aliases: tuple[str, ...]) -> float:
        candidates = [answer] * len(aliases)
        encoded = self.tokenizer(list(aliases), candidates, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            scores = self.model(**encoded).logits.reshape(-1)
        return float(scores.max().item())


def build_labeler(kind: str, *, bleurt_model: str | None, device: str):
    if kind == "exact":
        return ExactMatchLabeler()
    if not bleurt_model:
        raise ValueError("--bleurt-model is required when --labeler bleurt is selected")
    return BleurtLabeler(bleurt_model, device=device)
