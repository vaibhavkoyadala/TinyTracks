"""Internal HTTP API used by the Electron window (token-gated). Add routes here.

The OpenAI-compatible API is a separate app on its own port; see openai_api.py.
"""
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from tinygrad import Device, Tensor

from benchmarks import BenchmarkManager
from catalog import CATALOG_BY_ID
from downloads import Busy, DownloadManager, NoSpace, UnknownModel
from gpu_worker import GpuWorker
import gpu_stats
from request_log import RequestLogMiddleware
from serving import AUDIO_ARCHITECTURES, NotServable, Serving

app = FastAPI(title="TinyTracks API", version="0.2.0")

_TOKEN = os.environ.get("TINYTRACKS_TOKEN", "")

# Electron sets TINYTRACKS_DATA_DIR. Without it (plain `python worker.py`), use python/data.
DATA_DIR = Path(os.environ.get("TINYTRACKS_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")
OPENAI_HOST = "127.0.0.1"
OPENAI_PORT = int(os.environ.get("TINYTRACKS_OPENAI_PORT", "8080"))
openai_error: str | None = None  # set by worker.py if the OpenAI-compatible server could not start

gpu = GpuWorker()
manager = DownloadManager.open(DATA_DIR)
serving = Serving(manager.store, manager, gpu)
benchmarks = BenchmarkManager(manager.store, serving, gpu)
app.add_middleware(RequestLogMiddleware, server="window", store=manager.store)


def require_token(x_tinytracks_token: str = Header(default="")) -> None:
    # Fail closed: if Electron didn't pass a token, nothing is allowed.
    if not _TOKEN or not secrets.compare_digest(x_tinytracks_token, _TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing token")


@contextmanager
def _errors():
    """Map domain errors to HTTP status codes."""
    try:
        yield
    except UnknownModel:
        raise HTTPException(status_code=404, detail="Unknown model")
    except KeyError:
        raise HTTPException(status_code=404, detail="Not found")
    except Busy as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except NotServable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except NoSpace as exc:
        raise HTTPException(status_code=507, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class ComputeRequest(BaseModel):
    values: list[float] = Field(min_length=1, max_length=1_000_000)


@app.get("/api/health", dependencies=[Depends(require_token)])
def health() -> dict:
    import tinygrad

    return {
        "status": "ok",
        "tinygrad_version": getattr(tinygrad, "__version__", "unknown"),
        "device": Device.DEFAULT,
    }


@app.post("/api/compute", dependencies=[Depends(require_token)])
def compute(req: ComputeRequest) -> dict:
    # Example tinygrad computation: mean and sum of squares. Runs on the tinygrad thread.
    def job():
        t = Tensor(req.values)
        return {
            "count": len(req.values),
            "mean": t.mean().item(),
            "sum_of_squares": t.square().sum().item(),
        }

    return gpu.call(job)


# ----- models: download ------------------------------------------------------

@app.get("/api/models", dependencies=[Depends(require_token)])
def list_models() -> dict:
    """All catalog models with download state, serving state and progress."""
    return {"models": [_model_view(m) for m in manager.list()]}


@app.get("/api/models/{model_id}", dependencies=[Depends(require_token)])
def get_model(model_id: str) -> dict:
    with _errors():
        return _model_view(manager.get(model_id))


@app.post("/api/models/{model_id}/download", status_code=202, dependencies=[Depends(require_token)])
def download_model(model_id: str) -> dict:
    """Start or resume a download. Returns immediately; poll GET to track progress."""
    with _errors():
        return _model_view(manager.start(model_id))


@app.post("/api/models/{model_id}/download/pause", dependencies=[Depends(require_token)])
def pause_download(model_id: str) -> dict:
    with _errors():
        return _model_view(manager.pause(model_id))


@app.delete("/api/models/{model_id}", dependencies=[Depends(require_token)])
def delete_model(model_id: str) -> dict:
    """Remove the downloaded weights. Partial downloads are removed too."""
    with _errors():
        if serving.state_of(model_id) != "downloaded":
            raise Busy("eject the model before deleting it")
        return _model_view(manager.delete(model_id))


# ----- models: serving (load / eject / play / pause) --------------------------

@app.post("/api/models/{model_id}/load", dependencies=[Depends(require_token)])
def load_model(model_id: str) -> dict:
    with _errors():
        manager.get(model_id)
        serving.load(model_id)
        return _model_view(manager.get(model_id))


@app.post("/api/models/{model_id}/eject", dependencies=[Depends(require_token)])
def eject_model(model_id: str) -> dict:
    with _errors():
        manager.get(model_id)
        serving.eject(model_id)
        return _model_view(manager.get(model_id))


@app.post("/api/models/{model_id}/play", dependencies=[Depends(require_token)])
def play_model(model_id: str) -> dict:
    """Load if needed and go live: the OpenAI-compatible API serves this model."""
    with _errors():
        manager.get(model_id)
        serving.play(model_id)
        return _model_view(manager.get(model_id))


@app.post("/api/models/{model_id}/pause", dependencies=[Depends(require_token)])
def pause_model(model_id: str) -> dict:
    """Stop serving. Stays in memory, so Play resumes immediately."""
    with _errors():
        manager.get(model_id)
        serving.pause(model_id)
        return _model_view(manager.get(model_id))


# ----- serving / dashboard ----------------------------------------------------

@app.get("/api/serving", dependencies=[Depends(require_token)])
def serving_dashboard() -> dict:
    key = serving.api_key()
    base = f"http://{OPENAI_HOST}:{OPENAI_PORT}/v1"
    models = []
    for m in manager.list():
        mid = m["id"]
        state = serving.state_of(mid)
        try:
            serving.check_servable(mid)
            servable, reason = True, None
        except NotServable as exc:
            servable, reason = False, str(exc)
        live = state == "live"
        audio = CATALOG_BY_ID[mid]["architecture"] in AUDIO_ARCHITECTURES
        endpoint = f"{base}/audio/transcriptions" if audio else f"{base}/chat/completions"
        models.append({
            "id": mid,
            "name": m["name"],
            "architecture": CATALOG_BY_ID[mid]["architecture"],
            "kind": "transcription" if audio else "chat",
            "state": state,
            "live": live,
            "servable": servable,
            "reason": reason,
            "url": endpoint if live else None,
            "curl": (_curl_audio_sample(base, key, mid) if audio else _curl_sample(base, key, mid)) if live else None,
            "stats": serving.stats(mid),
        })
    return {
        "base_url": base,
        "api_key": key,
        "openai": {"running": openai_error is None, "error": openai_error},
        "models": models,
    }


@app.post("/api/serving/rotate-key", dependencies=[Depends(require_token)])
def rotate_key() -> dict:
    return {"api_key": serving.rotate_api_key()}


def _curl_audio_sample(base: str, key: str, model_id: str) -> str:
    return (
        f"curl {base}/audio/transcriptions \\\n"
        f"  -H \"Authorization: Bearer {key}\" \\\n"
        f"  -F model={model_id} \\\n"
        f"  -F file=@audio.wav"
    )


def _curl_sample(base: str, key: str, model_id: str) -> str:
    return (
        f"curl {base}/chat/completions \\\n"
        f"  -H \"Authorization: Bearer {key}\" \\\n"
        f"  -H \"Content-Type: application/json\" \\\n"
        f"  -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'"
    )


def _model_view(m: dict) -> dict:
    """Download fields plus serving state and the architecture."""
    return {**m, "state": serving.state_of(m["id"]),
            "architecture": CATALOG_BY_ID[m["id"]]["architecture"],
            "servable": _servable(m["id"])}


def _servable(model_id: str) -> bool:
    try:
        serving.check_servable(model_id)
        return True
    except NotServable:
        return False


# ----- benchmarks (beam search) ----------------------------------------------

class BenchmarkRequest(BaseModel):
    model_id: str
    widths: list[int] = Field(default=[0, 1, 2, 4], max_length=17)
    gen_tokens: int = 64
    prompt: str | None = Field(default=None, max_length=2000)
    auto: bool = False  # auto-optimize: keep raising the beam width until turned off


class AutoRequest(BaseModel):
    enabled: bool


@app.get("/api/benchmarks", dependencies=[Depends(require_token)])
def list_benchmarks() -> dict:
    return {"jobs": benchmarks.list()}


@app.post("/api/benchmarks", status_code=202, dependencies=[Depends(require_token)])
def create_benchmark(req: BenchmarkRequest) -> dict:
    with _errors():
        manager.get(req.model_id)
        return benchmarks.create(req.model_id, req.widths, req.gen_tokens, req.prompt, auto=req.auto)


@app.get("/api/benchmarks/{job_id}", dependencies=[Depends(require_token)])
def get_benchmark(job_id: str) -> dict:
    with _errors():
        return benchmarks.get(job_id)


@app.post("/api/benchmarks/{job_id}/pause", dependencies=[Depends(require_token)])
def pause_benchmark(job_id: str) -> dict:
    with _errors():
        return benchmarks.pause(job_id)


@app.post("/api/benchmarks/{job_id}/resume", dependencies=[Depends(require_token)])
def resume_benchmark(job_id: str) -> dict:
    with _errors():
        return benchmarks.resume(job_id)


@app.post("/api/benchmarks/{job_id}/auto", dependencies=[Depends(require_token)])
def auto_benchmark(job_id: str, req: AutoRequest) -> dict:
    """Turn auto-optimize on (keeps raising the beam width) or off (stops after the current width)."""
    with _errors():
        return benchmarks.set_auto(job_id, req.enabled)


@app.post("/api/benchmarks/{job_id}/cancel", dependencies=[Depends(require_token)])
def cancel_benchmark(job_id: str) -> dict:
    with _errors():
        return benchmarks.cancel(job_id)


# ----- kernels ----------------------------------------------------------------

@app.get("/api/kernels", dependencies=[Depends(require_token)])
def list_kernels(model_id: str | None = None) -> dict:
    """Kernel options that BEAM search found, grouped by the job that found them."""
    return {"kernels": benchmarks.kernels(model_id)}


# ----- request log and monitor ---------------------------------------------------

@app.get("/api/requests", dependencies=[Depends(require_token)])
def list_requests(limit: int = 200, before: int | None = None, server: str | None = None,
                  errors_only: bool = False) -> dict:
    """Recent requests to both APIs, newest first. Pass the last id as `before` to page back."""
    limit = max(1, min(limit, 1000))
    return {"requests": manager.store.list_requests(limit, before, server, errors_only)}


@app.get("/api/requests/summary", dependencies=[Depends(require_token)])
def request_summary() -> dict:
    return manager.store.request_summary(time.time() - 3600)


@app.get("/api/gpu", dependencies=[Depends(require_token)])
def gpu_status() -> dict:
    return gpu_stats.read()
