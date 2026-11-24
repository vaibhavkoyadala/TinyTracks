"""OpenAI-compatible API for live models.

Modeled on the photon-tap gateway: the model id goes in the request body, auth is
`Authorization: Bearer <key>` (or `x-api-key`), and `/v1/chat/completions` and `/v1/models`
behave like OpenAI's. Errors use OpenAI's `{"error": {...}}` shape so SDKs can parse them.

This app listens on its own loopback port with a stable key, so the URL and curl command
shown in the dashboard keep working across restarts.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from audio import AudioError, read_wav
from vl_runtime import ImageNotSupported
from vl_image import ImageError
from request_log import RequestLogMiddleware
from catalog import CATALOG_BY_ID
from llm_runtime import ContextTooLong
from serving import NotLive, Serving, WrongModelKind
from whisper_runtime import UnsupportedLanguage

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # OpenAI's documented upload limit


class OpenAIError(Exception):
    def __init__(self, status: int, message: str, error_type: str, code: str | None = None, param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.error_type, self.code, self.param = status, message, error_type, code, param


def _error_body(err: OpenAIError) -> dict:
    return {"error": {"message": err.message, "type": err.error_type, "param": err.param, "code": err.code}}


def build_app(serving: Serving) -> FastAPI:
    app = FastAPI(title="TinyTracks OpenAI-compatible API", version="1.0")
    app.add_middleware(RequestLogMiddleware, server="openai", store=serving.store)

    @app.exception_handler(OpenAIError)
    async def _handle(_request: Request, err: OpenAIError):
        return JSONResponse(status_code=err.status, content=_error_body(err))

    def check_auth(request: Request) -> None:
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if not token or not secrets.compare_digest(token, serving.api_key()):
            raise OpenAIError(401, "Incorrect API key provided.", "invalid_request_error", "invalid_api_key")

    @app.get("/v1/models")
    def list_models(request: Request) -> dict:
        check_auth(request)
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "created": now, "owned_by": "tinytracks"}
                for mid in serving.live_ids()
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        check_auth(request)
        body = await _json_body(request)
        model_id = body.get("model")
        request.state.model_id = model_id if isinstance(model_id, str) else None
        messages = body.get("messages")
        if not isinstance(model_id, str) or not model_id:
            raise OpenAIError(400, "You must provide a model parameter.", "invalid_request_error", "missing_model", "model")
        if not isinstance(messages, list) or not messages:
            raise OpenAIError(400, "messages must be a non-empty array.", "invalid_request_error", "invalid_messages", "messages")
        if model_id not in CATALOG_BY_ID:
            raise OpenAIError(404, f"The model '{model_id}' does not exist.", "invalid_request_error", "model_not_found", "model")

        temperature = _number(body, "temperature", 1.0, 0.0, 2.0)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        if max_tokens is not None:
            max_tokens = int(max_tokens)
            if max_tokens < 1:
                raise OpenAIError(400, "max_tokens must be at least 1.", "invalid_request_error", None, "max_tokens")
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))

        messages, images = _normalize_messages(messages, serving.accepts_images(model_id))
        try:
            prepared = serving.prepare(model_id, messages, max_tokens, images)
        except NotLive as exc:
            raise OpenAIError(503, str(exc), "server_error", "model_not_live")
        except ImageNotSupported as exc:
            raise OpenAIError(400, str(exc), "invalid_request_error", "image_not_supported", "messages")
        except ImageError as exc:
            raise OpenAIError(400, str(exc), "invalid_request_error", "invalid_image", "messages")
        except WrongModelKind as exc:
            raise OpenAIError(400, str(exc), "invalid_request_error", "wrong_endpoint", "model")
        except ContextTooLong as exc:
            raise OpenAIError(400, str(exc), "invalid_request_error", "context_length_exceeded", "messages")
        except ValueError as exc:  # template or message shape problems
            raise OpenAIError(400, f"Could not build the prompt: {exc}", "invalid_request_error", "invalid_messages", "messages")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def chunk(delta: dict, finish: str | None = None) -> str:
            payload = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(payload)}\n\n"

        def usage_of(info: dict) -> dict:
            completion = info.get("completion_tokens", 0)
            return {"prompt_tokens": prepared.n_prompt, "completion_tokens": completion,
                    "total_tokens": prepared.n_prompt + completion}

        if stream:
            def events():
                try:
                    yield chunk({"role": "assistant", "content": ""})
                    info: dict = {}
                    for kind, value in serving.run(prepared, temperature):
                        if kind == "delta":
                            yield chunk({"content": value})
                        else:
                            info = value
                    yield chunk({}, finish=info.get("finish_reason", "stop"))
                    if include_usage:
                        final = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                                 "model": model_id, "choices": [], "usage": usage_of(info)}
                        yield f"data: {json.dumps(final)}\n\n"
                except NotLive as exc:
                    yield f"data: {json.dumps(_error_body(OpenAIError(503, str(exc), 'server_error', 'model_not_live')))}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        try:
            # Generation blocks on the inference lock, so run it off the event loop.
            text, info = await asyncio.to_thread(_collect, serving, prepared, temperature)
        except NotLive as exc:
            raise OpenAIError(503, str(exc), "server_error", "model_not_live")
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": info.get("finish_reason", "stop")}],
            "usage": usage_of(info),
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(None),
        response_format: str = Form("json"),
    ):
        check_auth(request)
        request.state.model_id = model
        if model not in CATALOG_BY_ID:
            raise OpenAIError(404, f"The model '{model}' does not exist.", "invalid_request_error", "model_not_found", "model")
        if response_format not in ("json", "text", "verbose_json"):
            raise OpenAIError(400, "response_format must be json, text or verbose_json.", "invalid_request_error", None, "response_format")
        data = await file.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise OpenAIError(413, "The audio file is larger than 25 MB.", "invalid_request_error", "file_too_large", "file")
        if not data:
            raise OpenAIError(400, "The audio file is empty.", "invalid_request_error", "empty_file", "file")
        try:
            samples = read_wav(data)
        except AudioError as exc:
            raise OpenAIError(400, str(exc), "invalid_request_error", "invalid_audio", "file")

        try:
            # Transcription runs on the GPU worker, so keep it off the event loop.
            out = await asyncio.to_thread(serving.transcribe, model, samples, language or None)
        except NotLive as exc:
            raise OpenAIError(503, str(exc), "server_error", "model_not_live")
        except WrongModelKind as exc:
            raise OpenAIError(400, str(exc), "invalid_request_error", "wrong_endpoint", "model")
        except UnsupportedLanguage as exc:
            raise OpenAIError(400, str(exc), "invalid_request_error", "unsupported_language", "language")

        if response_format == "text":
            return PlainTextResponse(out["text"])
        body = {"text": out["text"]}
        if response_format == "verbose_json":
            body = {"task": "transcribe", "language": out["language"], "duration": out["duration"], "text": out["text"]}
        return body

    return app


def _normalize_messages(messages: list, allow_images: bool) -> tuple[list, list[str]]:
    """OpenAI content parts -> plain text when there are no images, or -> parts with {"type": "image"} placeholders.

    Returns the rewritten messages and the image data URLs, in order. Only data: URLs are accepted,
    so the server never fetches anything.
    """
    out: list = []
    images: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise OpenAIError(400, "each message must be an object", "invalid_request_error", "invalid_messages", "messages")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            texts = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                    texts.append(str(part.get("text", "")))
                elif kind == "image_url":
                    if not allow_images:
                        raise OpenAIError(400, "this model does not accept images; use a vision model",
                                          "invalid_request_error", "image_not_supported", "messages")
                    url = (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else None
                    if not isinstance(url, str):
                        raise OpenAIError(400, "image_url.url must be a data: URL", "invalid_request_error", "invalid_image", "messages")
                    images.append(url)
                    parts.append({"type": "image"})
                else:
                    raise OpenAIError(400, f"unsupported content part type {kind!r}", "invalid_request_error",
                                      "unsupported_content", "messages")
            message = {**message, "content": parts if images else "\n".join(texts)}
        out.append(message)
    return out, images


def _collect(serving: Serving, prepared, temperature: float) -> tuple[str, dict]:
    pieces: list[str] = []
    info: dict = {}
    for kind, value in serving.run(prepared, temperature):
        if kind == "delta":
            pieces.append(value)
        else:
            info = value
    return "".join(pieces), info


async def _json_body(request: Request) -> dict:
    try:
        body = json.loads(await request.body() or b"{}")
    except json.JSONDecodeError:
        raise OpenAIError(400, "We could not parse the JSON body of your request.", "invalid_request_error")
    if not isinstance(body, dict):
        raise OpenAIError(400, "The JSON body must be an object.", "invalid_request_error")
    return body


def _number(body: dict, key: str, default: float, low: float, high: float) -> float:
    value = body.get(key, default)
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise OpenAIError(400, f"{key} must be a number.", "invalid_request_error", None, key)
    if not low <= number <= high:
        raise OpenAIError(400, f"{key} must be between {low} and {high}.", "invalid_request_error", None, key)
    return number


class OpenAIServer:
    """Runs the OpenAI-compatible app in a background thread on a fixed loopback port."""

    def __init__(self, serving: Serving, host: str, port: int):
        self.host, self.port = host, port
        config = uvicorn.Config(build_app(serving), host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None  # not the main thread; the worker owns signals
        self._thread = threading.Thread(target=self._server.run, name="openai-api", daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            if self._server.started:
                return
            if not self._thread.is_alive():
                break
            time.sleep(0.05)
        raise RuntimeError(f"OpenAI-compatible API could not start on {self.host}:{self.port} (port in use?)")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
