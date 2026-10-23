"""Hugging Face causal-LM adapter used by the extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


def ensure_chat_template(tokenizer, model_name: str) -> str:
    """Support legacy Llama-2 caches using Meta's single-user [INST] format."""
    if tokenizer.chat_template:
        return "tokenizer"
    if model_name not in {"meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"}:
        raise ValueError("Tokenizer has no chat template; supply the correct chat-model tokenizer")
    if not tokenizer.bos_token:
        raise ValueError("Llama-2 chat formatting requires its BOS token")
    # Meta reference: llama/generation.py chat_completion, one user message, no system message.
    tokenizer.chat_template = (
        "{% if messages|length != 1 or messages[0]['role'] != 'user' %}"
        "{{ raise_exception('This fallback supports exactly one user message') }}{% endif %}"
        "{{ bos_token + '[INST] ' + messages[0]['content'].strip() + ' [/INST]' }}"
    )
    return "meta_llama2_single_user_fallback"


@dataclass
class GeneratedAnswer:
    text: str
    mean_log_probability: float
    hidden_by_layer: dict[int, "object"]
    generated_tokens: int
    generated_token_ids: list[int] = field(default_factory=list)
    stop_reason: str = "unknown"
    prompt_tokens: int = 0
    attempts: list[dict] = field(default_factory=list)


class CausalLMAdapter:
    def __init__(self, model_name: str, *, device: str = "auto", dtype: str = "auto", trust_remote_code: bool = False, chat_format: str = "raw"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with: pip install -e .") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.chat_format = chat_format
        self.chat_template_source = ensure_chat_template(self.tokenizer, model_name) if chat_format == "tokenizer" else "raw"
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

    def render_prompt(self, prompt: str, response_prefix: str | None = None) -> str:
        if getattr(self, "chat_format", "raw") == "raw":
            return prompt
        content = prompt
        if response_prefix is not None:
            if not prompt.endswith(response_prefix):
                raise ValueError("Response prefix must be the final text of the prompt")
            content = prompt[:-len(response_prefix)].rstrip()
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        # The cue is an assistant prefill after the template's user terminator.
        separator = " " if response_prefix and not rendered.endswith((" ", "\n")) else ""
        return rendered + separator + (response_prefix or "")

    def _encode_prompt(self, prompt: str, response_prefix: str | None = None):
        rendered = self.render_prompt(prompt, response_prefix)
        kwargs = {"add_special_tokens": False} if getattr(self, "chat_format", "raw") == "tokenizer" else {}
        return self.tokenizer(rendered, return_tensors="pt", **kwargs).to(self.device)

    def generate(self, prompt: str, layers: Sequence[int], *, temperature: float, max_new_tokens: int, seed: int) -> GeneratedAnswer:
        torch = self.torch
        torch.manual_seed(seed)
        encoded = self._encode_prompt(prompt)
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
        ids = generated_ids[0].tolist()
        eos_ids = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
        ended = bool(ids and ids[-1] in eos_ids)
        reason = "eos" if ended else "length" if len(ids) >= max_new_tokens else "unknown"
        hidden = {}
        if layers:
            with torch.inference_mode():
                replay = self.model(output.sequences, output_hidden_states=True, use_cache=False)
            resolved = self._resolve_layers(layers, len(replay.hidden_states))
            position = -2 if ended and len(ids) > 1 and self.chat_format == "tokenizer" else -1
            hidden = {
                requested: replay.hidden_states[actual][0, position].float().cpu().numpy()
                for requested, actual in zip(layers, resolved)
            }
        return GeneratedAnswer(text, sum(token_logps) / max(1, len(token_logps)), hidden, len(token_logps),
                               ids, reason, int(prompt_len))

    def generate_complete(self, prompt: str, layers: Sequence[int], *, temperature: float,
                          max_new_tokens: int, max_retry_tokens: int, seed: int) -> GeneratedAnswer:
        """Retry length-stopped generations from the same prompt and seed, without editing text."""
        if max_retry_tokens < max_new_tokens:
            raise ValueError("Retry token ceiling must be at least the initial token budget")
        limit, attempts = max_new_tokens, []
        while True:
            result = self.generate(prompt, layers, temperature=temperature, max_new_tokens=limit, seed=seed)
            attempts.append({"limit": limit, "generated_tokens": result.generated_tokens, "stop_reason": result.stop_reason})
            result.attempts = list(attempts)
            if result.stop_reason != "length" or limit >= max_retry_tokens:
                return result
            limit = min(limit * 2, max_retry_tokens)

    def prompt_hidden(self, prompt: str, layers: Sequence[int]) -> tuple[dict[int, "object"], int]:
        """Read the final prompt position in one forward pass; generate/score no labels."""
        prefix = "Assessment:" if prompt.endswith("Assessment:") else None
        encoded = self._encode_prompt(prompt, prefix)
        with self.torch.inference_mode():
            out = self.model(**encoded, output_hidden_states=True, use_cache=False)
        resolved = self._resolve_layers(layers, len(out.hidden_states))
        hidden = {
            requested: out.hidden_states[actual][0, -1].float().cpu().numpy()
            for requested, actual in zip(layers, resolved)
        }
        return hidden, int(encoded.input_ids.shape[1])

    def label_probability_and_hidden(
        self, prompt: str, layers: Sequence[int],
        labels: tuple[str, str] = (" Correct", " Incorrect"),
    ) -> tuple[float, dict[int, "object"], int]:
        """Score complete label strings and return hidden states at the shared cue."""
        torch = self.torch
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        label_scores = []
        total_scored = 0
        hidden = None
        for label in labels:
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

