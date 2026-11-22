"""Model lifecycle for live serving: download -> load -> play (live) <-> pause, and eject.

    downloaded   weights on disk, not in memory; the API does not serve it
    loaded       weights in memory, paused; the API returns 503 for it
    live         loaded and playing; the OpenAI-compatible API serves it

Every tinygrad call runs on the GpuWorker thread (see gpu_worker.py).
"""
from __future__ import annotations

import gc
import secrets
import threading
import time
from collections.abc import Iterator

from catalog import CATALOG_BY_ID
from downloads import Busy, DownloadManager
from gpu_worker import GpuWorker
from llm_loader import SUPPORTED_ARCHITECTURES
from llm_runtime import ContextTooLong, LoadedModel, TextStreamer
from store import Store
from whisper_runtime import LoadedWhisper

AUDIO_ARCHITECTURES = {"WhisperForConditionalGeneration"}
SERVABLE_ARCHITECTURES = SUPPORTED_ARCHITECTURES | AUDIO_ARCHITECTURES

DEFAULT_MAX_CONTEXT = 2048
API_KEY_SETTING = "openai_api_key"


class NotServable(RuntimeError):
    """The model cannot be loaded: not downloaded, or its architecture is not supported."""


class NotLive(RuntimeError):
    """The model exists but is not playing, so the API will not serve it."""


class WrongModelKind(ValueError):
    """The model is live, but this endpoint does not serve its kind (chat vs. transcription)."""


class Prepared:
    __slots__ = ("model_id", "model", "ids", "max_new_tokens")

    def __init__(self, model_id: str, model: LoadedModel, ids: list[int], max_new_tokens: int):
        self.model_id, self.model, self.ids, self.max_new_tokens = model_id, model, ids, max_new_tokens


class Serving:
    def __init__(self, store: Store, downloads: DownloadManager, gpu: GpuWorker,
                 max_context: int = DEFAULT_MAX_CONTEXT):
        self.store = store
        self.downloads = downloads
        self.gpu = gpu
        self.max_context = max_context
        self._state_lock = threading.Lock()  # guards the dicts below (plain data, never tinygrad)
        self._loaded: dict[str, LoadedModel] = {}
        self._live: set[str] = set()
        self._busy: set[str] = set()         # models in use by a benchmark (cannot be ejected)
        self._stats: dict[str, dict] = {}
        self._model_locks: dict[str, threading.Lock] = {}  # one generation at a time per model

    # ----- lifecycle --------------------------------------------------------

    def check_servable(self, model_id: str) -> None:
        model = self.downloads.get(model_id)  # raises UnknownModel for ids outside the catalog
        if model["status"] != "ready":
            raise NotServable(f"download {model['name']} before loading it")
        arch = CATALOG_BY_ID[model_id]["architecture"]
        if arch not in SERVABLE_ARCHITECTURES:
            raise NotServable(f"{model['name']} ({arch}) cannot be served yet; only Llama, Qwen2 and Whisper models can")

    def model_dir(self, model_id: str):
        return self.downloads.weights_dir / model_id

    def load(self, model_id: str) -> None:
        self.check_servable(model_id)

        def job() -> None:
            with self._state_lock:
                if model_id in self._loaded:
                    return
            model = self._open(model_id)
            with self._state_lock:
                self._loaded[model_id] = model

        self.gpu.call(job)

    def _open(self, model_id: str):
        """Build the runtime that matches the model's architecture. Runs on the GpuWorker thread."""
        if CATALOG_BY_ID[model_id]["architecture"] in AUDIO_ARCHITECTURES:
            return LoadedWhisper(model_id, self.model_dir(model_id))
        return LoadedModel(model_id, self.model_dir(model_id), self.max_context)

    def _lock_for(self, model_id: str) -> threading.Lock:
        with self._state_lock:
            return self._model_locks.setdefault(model_id, threading.Lock())

    def eject(self, model_id: str) -> None:
        with self._state_lock:
            if model_id in self._busy:
                raise Busy("a benchmark is using this model; wait for it to finish or cancel it")

        def job() -> None:
            with self._state_lock:
                self._live.discard(model_id)
                model = self._loaded.pop(model_id, None)
            if model is not None:
                model.close()
            del model
            gc.collect()

        # Wait for any generation on this model to finish before unloading it.
        with self._lock_for(model_id):
            self.gpu.call(job)

    def play(self, model_id: str) -> None:
        self.load(model_id)
        with self._state_lock:
            self._live.add(model_id)

    def pause(self, model_id: str) -> None:
        with self._state_lock:
            self._live.discard(model_id)

    def state_of(self, model_id: str) -> str:
        with self._state_lock:
            if model_id in self._live:
                return "live"
            if model_id in self._loaded:
                return "loaded"
        return "downloaded"

    def live_ids(self) -> list[str]:
        with self._state_lock:
            return sorted(self._live)

    def loaded_model(self, model_id: str) -> LoadedModel | None:
        with self._state_lock:
            return self._loaded.get(model_id)

    def mark_busy(self, model_id: str, busy: bool) -> None:
        with self._state_lock:
            (self._busy.add if busy else self._busy.discard)(model_id)

    # ----- api key ----------------------------------------------------------

    def api_key(self) -> str:
        key = self.store.get_setting(API_KEY_SETTING)
        if key is None:
            key = self.rotate_api_key()
        return key

    def rotate_api_key(self) -> str:
        key = "tt-" + secrets.token_urlsafe(24)
        self.store.set_setting(API_KEY_SETTING, key)
        return key

    # ----- inference --------------------------------------------------------

    def prepare(self, model_id: str, messages: list[dict], max_new_tokens: int | None) -> Prepared:
        """Validate and tokenize a request on the request thread (tokenizers does not touch tinygrad)."""
        with self._state_lock:
            model = self._loaded.get(model_id) if model_id in self._live else None
        if model is None:
            raise NotLive(f"{model_id} is not live; press Play in TinyTracks to serve it")
        if model.kind != "chat":
            raise WrongModelKind(f"{model_id} is a speech model; send audio to /v1/audio/transcriptions")
        ids = model.prompt_ids(messages)
        room = model.max_context - len(ids)
        budget = room if max_new_tokens is None else min(max_new_tokens, room)
        return Prepared(model_id, model, ids, max(1, budget))

    def run(self, prepared: Prepared, temperature: float) -> Iterator[tuple[str, object]]:
        """Yield ("delta", text) events as tokens arrive, then ("done", info).

        Other models keep running between this model's tokens. Only one generation per model runs at a time,
        because the model's KV cache belongs to the generation in progress.
        """
        with self._lock_for(prepared.model_id):
            yield from self.gpu.stream(lambda: self._generate(prepared, temperature))

    def _generate(self, prepared: Prepared, temperature: float):
        """Generator run one step at a time on the GpuWorker thread. Yields an event per token, or None."""
        with self._state_lock:
            still_live = prepared.model_id in self._live and self._loaded.get(prepared.model_id) is prepared.model
        if not still_live:
            raise NotLive(f"{prepared.model_id} was paused or ejected before generation started")

        streamer = TextStreamer(prepared.model.tokenizer)
        info: dict = {}
        completed = False
        try:
            for token in prepared.model.generate(prepared.ids, prepared.max_new_tokens, temperature, info):
                delta = streamer.push(token)
                yield ("delta", delta) if delta else None
            tail = streamer.flush()
            if tail:
                yield ("delta", tail)
            completed = True
        finally:  # also runs when the consumer disconnects and the stream is closed
            if not completed:
                info["finish_reason"] = "cancelled"
            self._record(prepared.model_id, info, len(prepared.ids))
        yield ("done", info)

    def _record(self, model_id: str, info: dict, prompt_tokens: int) -> None:
        completion = info.get("completion_tokens", 0)
        decode_s = info.get("decode_s") or 0.0
        with self._state_lock:
            st = self._stats.setdefault(model_id, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                    "last_tok_s": None, "since": time.time()})
            st["requests"] += 1
            st["prompt_tokens"] += prompt_tokens
            st["completion_tokens"] += completion
            if completion and decode_s > 0:
                st["last_tok_s"] = round(completion / decode_s, 2)

    def stats(self, model_id: str) -> dict:
        with self._state_lock:
            st = self._stats.get(model_id)
            return dict(st) if st else {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                        "last_tok_s": None, "since": None}

    def transcribe(self, model_id: str, samples, language: str | None) -> dict:
        """Run speech recognition on a live Whisper model. Returns {text, language, tokens, duration}."""
        with self._state_lock:
            model = self._loaded.get(model_id) if model_id in self._live else None
        if model is None:
            raise NotLive(f"{model_id} is not live; press Play in TinyTracks to serve it")
        if model.kind != "transcription":
            raise WrongModelKind(f"{model_id} is a chat model; send messages to /v1/chat/completions")

        def job() -> dict:
            with self._state_lock:
                if self._loaded.get(model_id) is not model or model_id not in self._live:
                    raise NotLive(f"{model_id} was paused or ejected before transcription started")
            out = model.transcribe_samples(samples, language)
            self._record(model_id, {"completion_tokens": out["tokens"]}, prompt_tokens=0)
            return out

        with self._lock_for(model_id):
            return self.gpu.call(job)

