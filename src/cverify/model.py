"""Hugging Face causal-LM adapter used by the extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class GeneratedAnswer:
    text: str
    mean_log_probability: float
    hidden_by_layer: dict[int, "object"]
    generated_tokens: int


class CausalLMAdapter:
    def __init__(self, model_name: str, *, device: str = "auto", dtype: str = "auto", trust_remote_code: bool = False):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with: pip install -e .") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": dtype}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _resolve_layers(self, requested: Sequence[int], hidden_count: int) -> list[int]:
        resolved = []
        for layer in requested:
            idx = layer if layer >= 0 else hidden_count + layer
            if idx < 0 or idx >= hidden_count:
                raise ValueError(f"Layer {layer} is invalid for {hidden_count} hidden-state tensors")
            resolved.append(idx)
        return resolved

    def generate(self, prompt: str, layers: Sequence[int], *, temperature: float, max_new_tokens: int, seed: int) -> GeneratedAnswer:
        torch = self.torch
        torch.manual_seed(seed)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = encoded.input_ids.shape[1]
        generated_ids = output.sequences[:, prompt_len:]
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        token_logps = []
        for score, token_id in zip(output.scores, generated_ids[0]):
            token_logps.append(torch.log_softmax(score[0].float(), dim=-1)[token_id].item())
        with torch.inference_mode():
            replay = self.model(output.sequences, output_hidden_states=True, use_cache=False)
        resolved = self._resolve_layers(layers, len(replay.hidden_states))
        hidden = {
            requested: replay.hidden_states[actual][0, -1].float().cpu().numpy()
            for requested, actual in zip(layers, resolved)
        }
        return GeneratedAnswer(text, sum(token_logps) / max(1, len(token_logps)), hidden, len(token_logps))

    def label_probability_and_hidden(self, prompt: str, layers: Sequence[int]) -> tuple[float, dict[int, "object"], int]:
        """Score complete label strings and return hidden states at the shared cue."""
        torch = self.torch
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        label_scores = []
        total_scored = 0
        hidden = None
        for label in (" Correct", " Incorrect"):
            label_ids = self.tokenizer(label, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            ids = torch.cat([prompt_ids, label_ids], dim=1)
            with torch.inference_mode():
                out = self.model(ids, output_hidden_states=True, use_cache=False)
            start = prompt_ids.shape[1] - 1
            logits = out.logits[:, start:-1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            score = log_probs.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1).sum()
            label_scores.append(score)
            total_scored += label_ids.numel()
            if hidden is None:
                resolved = self._resolve_layers(layers, len(out.hidden_states))
                hidden = {
                    requested: out.hidden_states[actual][0, prompt_ids.shape[1] - 1].float().cpu().numpy()
                    for requested, actual in zip(layers, resolved)
                }
        probability = torch.softmax(torch.stack(label_scores), dim=0)[0].item()
        return probability, hidden or {}, total_scored

