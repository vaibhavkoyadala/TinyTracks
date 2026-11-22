"""Records every HTTP request to the window API and the OpenAI-compatible API.

Written as pure ASGI middleware, so a streamed response is timed until its last byte.
Only metadata is stored: method, path, status, timings, the model id if the handler set one, the
client address and byte counts. Request bodies, headers and API keys are never stored.
"""
from __future__ import annotations

import re
import time

from store import Store

MODEL_PATH = re.compile(r"^/api/models/([^/]+)")

KEEP_ROWS = 20000
PRUNE_EVERY = 200


class RequestLogMiddleware:
    def __init__(self, app, server: str, store: Store):
        self.app = app
        self.server = server
        self.store = store
        self._writes = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        ts = time.time()
        status = {"code": 500}
        response_bytes = 0

        async def send_wrapper(message):
            nonlocal response_bytes
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            elif message["type"] == "http.response.body":
                response_bytes += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self._record(scope, ts, started, status["code"], response_bytes)

    def _record(self, scope, ts: float, started: float, status: int, response_bytes: int) -> None:
        headers = dict(scope.get("headers") or [])
        try:
            request_bytes = int(headers.get(b"content-length", b"0"))
        except ValueError:
            request_bytes = 0
        client = scope.get("client")
        state = scope.get("state") or {}
        model_id = state.get("model_id")
        if model_id is None:  # window routes carry the model id in the path
            match = MODEL_PATH.match(scope.get("path", ""))
            model_id = match.group(1) if match else None
        self.store.add_request({
            "ts": ts,
            "server": self.server,
            "method": scope.get("method", ""),
            "path": scope.get("path", ""),
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "model_id": model_id,
            "client": client[0] if client else None,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
        })
        self._writes += 1
        if self._writes % PRUNE_EVERY == 0:
            self.store.prune_requests(KEEP_ROWS)
