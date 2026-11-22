"""A loaded language model: weights, tokenizer, chat template and stop tokens.

tinygrad runs on one device and its global state (JIT, BEAM settings) is not thread-safe,
so callers must hold the inference lock in serving.py while they use a LoadedModel.
"""
from __future__ import annotations

import gc
import json
import time
from collections.abc import Iterator
from pathlib import Path

import jinja2
from tokenizers import Tokenizer

from llm_loader import load_transformer

STOP_TOKEN_NAMES = ("<|im_end|>", "<|endoftext|>", "</s>", "<|end_of_text|>")


class ContextTooLong(ValueError):
    pass


class LoadedModel:
    kind = "chat"

    def __init__(self, model_id: str, model_dir: Path, max_context: int):
        self.model_id = model_id
        self.model_dir = model_dir
        self.max_context = max_context
        self.model, self.config = load_transformer(model_dir, max_context)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

        tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        env = jinja2.Environment()
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(tok_cfg["chat_template"])

        self.stop_ids = self._stop_ids(model_dir)

    def _stop_ids(self, model_dir: Path) -> set[int]:
        ids: set[int] = set()
        gen_path = model_dir / "generation_config.json"
        if gen_path.exists():
            eos = json.loads(gen_path.read_text()).get("eos_token_id")
            ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        for name in STOP_TOKEN_NAMES:
            tid = self.tokenizer.token_to_id(name)
            if tid is not None:
                ids.add(tid)
        return ids

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.template.render(messages=messages, add_generation_prompt=True)
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) >= self.max_context:
            raise ContextTooLong(
                f"prompt has {len(ids)} tokens, but the model context is {self.max_context}")
        return ids

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def generate(self, ids: list[int], max_new_tokens: int, temperature: float, info: dict) -> Iterator[int]:
        """Yield generated token ids. `info` receives finish_reason and timings when done."""
        info.update(finish_reason="length", completion_tokens=0, prefill_s=None, decode_s=None)
        start = time.perf_counter()
        first_at = None
        produced = 0
        for token in self.model.generate(list(ids), temperature=temperature):
            now = time.perf_counter()
            if first_at is None:
                first_at = now
                info["prefill_s"] = now - start
            if token in self.stop_ids:
                info["finish_reason"] = "stop"
                break
            yield token
            produced += 1
            info["completion_tokens"] = produced
            if produced >= max_new_tokens:
                break
        if first_at is not None:
            info["decode_s"] = time.perf_counter() - first_at
        info["total_s"] = time.perf_counter() - start

    def close(self) -> None:
        self.model = None
        gc.collect()


class TextStreamer:
    """Turns a stream of token ids into text deltas, holding back incomplete UTF-8."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.emitted = ""

    def push(self, token: int, final: bool = False) -> str:
        self.ids.append(token)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if not final and text.endswith("�"):
            return ""
        if not text.startswith(self.emitted):
            return ""
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta

    def flush(self) -> str:
        """Emit whatever is still held back at the end of generation."""
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if not text.startswith(self.emitted):
            return ""
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta


def _raise(message: str):
    raise jinja2.TemplateError(message)
