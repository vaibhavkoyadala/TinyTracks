"""A loaded Whisper model: speech in, text out.

Callers must run this on the GpuWorker thread, like the LLM runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tinygrad import Tensor
from tokenizers import Tokenizer

from audio import log_mel, split_chunks
from whisper_model import load_whisper

MAX_TARGET = 448
DEFAULT_LANGUAGE = "en"


class UnsupportedLanguage(ValueError):
    pass


class LoadedWhisper:
    kind = "transcription"

    def __init__(self, model_id: str, model_dir: Path):
        self.model_id = model_id
        self.model, self.config = load_whisper(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        gen = json.loads((model_dir / "generation_config.json").read_text())
        self.sot = gen["decoder_start_token_id"]
        self.transcribe = gen["task_to_id"]["transcribe"]
        self.no_timestamps = gen["no_timestamps_token_id"]
        self.eot = gen["eos_token_id"]
        self.lang_to_id: dict[str, int] = gen["lang_to_id"]
        self.suppress = list(gen.get("suppress_tokens", []))
        self.begin_suppress = list(gen.get("begin_suppress_tokens", []))
        self.max_target = min(gen.get("max_length", MAX_TARGET), MAX_TARGET)

    def language_token(self, language: str | None) -> int:
        code = (language or DEFAULT_LANGUAGE).strip().lower()
        token = f"<|{code}|>"
        if token not in self.lang_to_id:
            raise UnsupportedLanguage(f"unsupported language '{language}'")
        return self.lang_to_id[token]

    def transcribe_samples(self, samples: np.ndarray, language: str | None = None) -> dict:
        """Transcribe mono 16 kHz samples. Audio longer than 30 s is split into consecutive windows."""
        lang = self.language_token(language)
        texts = []
        steps = 0
        for chunk in split_chunks(samples):
            text, n = self._decode_window(log_mel(chunk), lang)
            steps += n
            if text:
                texts.append(text)
        return {
            "text": " ".join(texts).strip(),
            "language": (language or DEFAULT_LANGUAGE).lower(),
            "tokens": steps,
            "duration": round(len(samples) / 16000, 3),
        }

    def _decode_window(self, features: np.ndarray, lang: int) -> tuple[str, int]:
        mel = Tensor(features[None]).realize()
        enc = self.model.encode(mel).realize()

        prompt = [self.sot, lang, self.transcribe, self.no_timestamps]
        generated: list[int] = []
        ids = list(prompt)
        while len(ids) < self.max_target:
            logits = self.model.logits(Tensor([ids]), enc)[:, -1, :].realize().numpy()[0].astype(np.float32)
            for t in self.suppress:
                logits[t] = -np.inf
            if not generated:
                for t in self.begin_suppress:
                    logits[t] = -np.inf
            token = int(np.argmax(logits))
            if token == self.eot:
                break
            generated.append(token)
            ids.append(token)
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, len(generated)

    def close(self) -> None:
        self.model = None
