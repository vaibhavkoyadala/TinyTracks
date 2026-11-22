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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from catalog import CATALOG_BY_ID
from llm_runtime import ContextTooLong
from serving import NotLive, Serving


class OpenAIError(Exception):
    def __init__(self, status: int, message: str, error_type: str, code: str | None = None, param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.error_type, self.code, self.param = status, message, error_type, code, param


def _error_body(err: OpenAIError) -> dict:
    return {"error": {"message": err.message, "type": err.error_type, "param": err.param, "code": err.code}}


def build_app(serving: Serving) -> FastAPI:
    app = FastAPI(title="TinyTracks OpenAI-compatible API", version="1.0")

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

        try:
            prepared = serving.prepare(model_id, messages, max_tokens)
        except NotLive as exc:
            raise OpenAIError(503, str(exc), "server_error", "model_not_live")
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
            return {"prompt_tokens": len(prepared.ids), "completion_tokens": completion,
                    "total_tokens": len(prepared.ids) + completion}

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

    return app


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
