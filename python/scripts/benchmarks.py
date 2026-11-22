"""Beam-search benchmarks: how fast a model decodes at each BEAM width, and which kernels won.

tinygrad's BEAM=<n> searches for faster kernels by compiling and timing candidates; the
winning kernel options are cached in tinygrad's cache.db (table `beam_search_<version>`).
A job runs one measurement per width:
  - BEAM=0 is the baseline (no kernel search).
  - Each width gets a fresh model instance and ignores the existing cache for reads, so the
    search really runs and `search_s` is its cost.
  - Then prefill latency and decode ms/token are timed on the same device.

Auto-optimize: an auto job keeps measuring the next width (0, 1, 2, ...) while `auto` is on, up
to AUTO_MAX_WIDTH. Turning it off lets the current width finish, and the job then ends as done.

Jobs run one at a time, because they share the GPU and their timings would interfere otherwise.
Pause and resume take effect between widths. A search already in progress finishes first.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import pickle
import sqlite3
import threading
import time
import uuid

from tinygrad import Device
from tinygrad.helpers import CACHEDB, VERSION, Context

from catalog import CATALOG_BY_ID
from downloads import Busy
from gpu_worker import GpuWorker
from llm_loader import SUPPORTED_ARCHITECTURES
from llm_runtime import LoadedModel
from serving import Serving
from store import Store

DEFAULT_PROMPT = "Write a short poem about the ocean."
MAX_WIDTH = 16           # manual widths
AUTO_MAX_WIDTH = 32      # auto-optimize stops here even if left on
MAX_GEN_TOKENS = 512
WARMUP_TOKENS = 3
MAX_CONTEXT = 1024
ACTIVE = ("queued", "running", "pausing")


class BenchmarkManager:
    def __init__(self, store: Store, serving: Serving, gpu: GpuWorker):
        self.store = store
        self.serving = serving
        self.gpu = gpu
        self._lock = threading.Lock()
        self._active: str | None = None
        self._resume: dict[str, threading.Event] = {}
        self._cancel: dict[str, threading.Event] = {}
        # Jobs that were running when the app last stopped come back as paused.
        for job in store.list_jobs():
            if job["status"] in ACTIVE:
                store.update_job(job["id"], status="paused")

    # ----- api --------------------------------------------------------------

    def create(self, model_id: str, widths: list[int], gen_tokens: int, prompt: str | None,
               auto: bool = False) -> dict:
        if model_id not in CATALOG_BY_ID:
            raise KeyError(model_id)
        self.serving.check_servable(model_id)
        if CATALOG_BY_ID[model_id]["architecture"] not in SUPPORTED_ARCHITECTURES:
            raise ValueError("beam search benchmarks are for language models")
        if not 8 <= gen_tokens <= MAX_GEN_TOKENS:
            raise ValueError(f"gen_tokens must be between 8 and {MAX_GEN_TOKENS}")
        if auto:
            widths = [0]
        else:
            widths = sorted(set(widths))
            if not widths or any(not isinstance(w, int) or w < 0 or w > MAX_WIDTH for w in widths):
                raise ValueError(f"widths must be integers from 0 (baseline) to {MAX_WIDTH}")

        with self._lock:
            if self._active is not None:
                raise Busy("another benchmark is running; pause or cancel it first")
            job_id = uuid.uuid4().hex[:12]
            self.store.create_job({
                "id": job_id, "model_id": model_id, "widths": ",".join(map(str, widths)),
                "gen_tokens": gen_tokens, "prompt": (prompt or DEFAULT_PROMPT).strip(), "status": "queued",
                "auto": auto,
            })
            self._active = job_id
        self._start(job_id)
        return self.get(job_id)

    def pause(self, job_id: str) -> dict:
        job = self._require(job_id)
        if job_id in self._resume and job["status"] == "running":
            self._resume[job_id].clear()
            self.store.update_job(job_id, status="pausing")  # becomes "paused" after the current width
        return self.get(job_id)

    def resume(self, job_id: str) -> dict:
        job = self._require(job_id)
        with self._lock:
            if job_id in self._resume and self._active == job_id:
                self._resume[job_id].set()
                if job["status"] in ("pausing", "paused"):
                    self.store.update_job(job_id, status="running")
                return self.get(job_id)
            if self._active is not None:
                raise Busy("another benchmark is running")
            if job["status"] not in ("paused", "failed"):
                raise ValueError(f"job is {job['status']}; only paused jobs can be resumed")
            self._active = job_id
        self._start(job_id)
        return self.get(job_id)

    def set_auto(self, job_id: str, enabled: bool) -> dict:
        """Turn auto-optimize on or off. Turning it on for a finished job continues it."""
        self._require(job_id)
        with self._lock:
            self.store.update_job(job_id, auto=int(enabled))
            if enabled and job_id not in self._cancel:
                if self._active is not None:
                    raise Busy("another benchmark is running; it continues after you turn auto on")
                self._active = job_id
                self.store.update_job(job_id, status="queued")
                self._start(job_id)
        return self.get(job_id)

    def cancel(self, job_id: str) -> dict:
        self._require(job_id)
        if job_id in self._cancel:
            self._cancel[job_id].set()
            if job_id in self._resume:
                self._resume[job_id].set()  # wake a paused job so it can stop
        else:
            self.store.update_job(job_id, status="cancelled")
        return self.get(job_id)

    def list(self) -> list[dict]:
        return [self._present(j) for j in self.store.list_jobs()]

    def get(self, job_id: str) -> dict:
        return self._present(self._require(job_id))

    def kernels(self, model_id: str | None = None) -> list[dict]:
        return self.store.list_kernels(model_id)

    def stop(self) -> None:
        """Called at shutdown: ask the active job to stop. Its status is saved as paused on the next start."""
        for event in list(self._cancel.values()):
            event.set()
        for event in list(self._resume.values()):
            event.set()

    # ----- internals --------------------------------------------------------

    def _require(self, job_id: str) -> dict:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _present(self, job: dict) -> dict:
        results = self.store.list_results(job["id"])
        best = min((r for r in results if r["decode_ms_per_token"]), key=lambda r: r["decode_ms_per_token"], default=None)
        for r in results:
            r["best"] = best is not None and r["beam"] == best["beam"]
        return {
            "id": job["id"],
            "model_id": job["model_id"],
            "widths": [int(w) for w in job["widths"].split(",")],
            "auto": bool(job["auto"]),
            "auto_max_width": AUTO_MAX_WIDTH,
            "gen_tokens": job["gen_tokens"],
            "prompt": job["prompt"],
            "status": job["status"],
            "current_width": job["current_width"],
            "error": job["error"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "results": results,
            "best_beam": best["beam"] if best else None,
        }

    def _start(self, job_id: str) -> None:
        resume = threading.Event()
        resume.set()
        self._resume[job_id] = resume
        self._cancel[job_id] = threading.Event()
        threading.Thread(target=self._run, args=(job_id,), name=f"bench-{job_id}", daemon=True).start()

    @staticmethod
    def _next_width(job: dict, done: set[int]) -> int | None:
        """The next width to measure, or None when the job has nothing left to do."""
        if job["auto"]:
            return next((w for w in range(0, AUTO_MAX_WIDTH + 1) if w not in done), None)
        pending = [int(w) for w in job["widths"].split(",") if int(w) not in done]
        return pending[0] if pending else None

    def _run(self, job_id: str) -> None:
        model_id = None
        try:
            model_id = self._require(job_id)["model_id"]
            self.serving.mark_busy(model_id, True)
            self.store.update_job(job_id, status="running")
            while not self._cancel[job_id].is_set():
                job = self._require(job_id)  # re-read every width, so toggling auto takes effect
                done = {r["beam"] for r in self.store.list_results(job_id)}
                width = self._next_width(job, done)
                if width is None:
                    break  # manual list finished, or auto reached AUTO_MAX_WIDTH
                if not self._resume[job_id].is_set():
                    self.store.update_job(job_id, status="paused")
                    self._resume[job_id].wait()
                    if self._cancel[job_id].is_set():
                        break
                    self.store.update_job(job_id, status="running")
                self.store.update_job(job_id, current_width=width)
                result, kernels, device = self.gpu.call(
                    lambda w=width, j=job: self._measure(model_id, w, j["gen_tokens"], j["prompt"]))
                self.store.add_result(job_id, result)
                if kernels:
                    self.store.add_kernels(job_id, model_id, width, device, kernels)
            final = "cancelled" if self._cancel[job_id].is_set() else "done"
            self.store.update_job(job_id, status=final, current_width=None)
        except Exception as exc:
            self.store.update_job(job_id, status="failed", error=str(exc)[:500], current_width=None)
        finally:
            if model_id is not None:
                self.serving.mark_busy(model_id, False)
            with self._lock:
                self._active = None
                self._resume.pop(job_id, None)
                self._cancel.pop(job_id, None)

    def _measure(self, model_id: str, beam: int, gen_tokens: int, prompt: str):
        """Runs on the GpuWorker thread. Returns (result, kernels, device)."""
        device = Device.DEFAULT
        before = _beam_rows() if beam else {}
        with Context(BEAM=beam, IGNORE_BEAM_CACHE=1):
            lm = LoadedModel(model_id, self.serving.model_dir(model_id), MAX_CONTEXT)
            try:
                ids = lm.prompt_ids([{"role": "user", "content": prompt}])

                # Warm-up: compiles every kernel, which is where the BEAM search runs.
                t0 = time.perf_counter()
                list(itertools.islice(lm.model.generate(list(ids), temperature=0.0), WARMUP_TOKENS))
                search_s = time.perf_counter() - t0

                # Prefill: drop the KV cache so the whole prompt is processed again.
                lm.model._cached_tokens = []
                gen = lm.model.generate(list(ids), temperature=0.0)
                t1 = time.perf_counter()
                next(gen)
                prefill_ms = (time.perf_counter() - t1) * 1000

                # Decode: time the remaining tokens. Stop tokens are ignored so every run
                # produces the same number of tokens.
                produced = 0
                t2 = time.perf_counter()
                for _ in range(gen_tokens - 1):
                    next(gen)
                    produced += 1
                decode_s = time.perf_counter() - t2
                del gen
            finally:
                lm.close()
                del lm
                gc.collect()

        decode_ms = (decode_s * 1000 / produced) if produced else None
        result = {
            "beam": beam,
            "search_s": round(search_s, 3),
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms_per_token": round(decode_ms, 3) if decode_ms else None,
            "decode_tok_s": round(1000 / decode_ms, 2) if decode_ms else None,
            "tokens": produced + 1,
        }
        kernels = []
        if beam:
            after = _beam_rows()
            for key, val in after.items():
                if key in before and before[key] == val:
                    continue
                if key[1] != beam or key[2] != device:  # key = (ast, amt, device, suffix)
                    continue
                kernels.append((hashlib.sha1(repr(key[0]).encode()).hexdigest()[:12], _describe_opts(val)))
        return result, kernels, device


def _beam_rows() -> dict:
    """Snapshot tinygrad's beam cache: {(ast, amt, device, suffix): pickled opts}."""
    table = f"beam_search_{VERSION}"
    try:
        conn = sqlite3.connect(f"file:{CACHEDB}?mode=ro", uri=True, timeout=10)
    except sqlite3.OperationalError:
        return {}
    try:
        rows = conn.execute(f"SELECT ast, amt, device, suffix, val FROM '{table}'").fetchall()
    except sqlite3.OperationalError:
        return {}  # the table does not exist yet
    finally:
        conn.close()
    return {(ast, amt, device, suffix): val for ast, amt, device, suffix, val in rows}


def _describe_opts(val: bytes) -> str:
    try:
        opts = pickle.loads(val)
    except Exception:
        return "?"
    return "; ".join(str(o) for o in opts) or "(no changes)"
