"""Demo safety limits: request body size and a per-client rate cap.

Both are harmless for local use. ``RATE_LIMIT_PER_MINUTE=0`` turns the
limiter off. ``/health``, ``/``, ``/ui``, and OpenAPI stay unlimited.
``/ask``, ``/query``, and ``/ingest`` count toward the cap.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque

from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import get_settings
from app.core.logging import get_request_id

_LIMITED = {"/ask", "/query", "/ingest"}
_windows: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def reset_rate_limits() -> None:
    """Drop in-memory counters. Tests call this before asserting a low cap."""
    with _lock:
        _windows.clear()


def _client_key(scope: Scope) -> str:
    client = scope.get("client")
    if client and client[0]:
        return str(client[0])
    return "unknown"


def _content_length(scope: Scope) -> int | None:
    for key, value in scope.get("headers") or []:
        if key.lower() == b"content-length":
            try:
                return int(value.decode("latin1"))
            except ValueError:
                return None
    return None


async def _reject(
    send: Send,
    status: int,
    message: str,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps({"error": message, "request_id": get_request_id()}).encode()
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method") in {"POST", "PUT", "PATCH"}:
            limit = get_settings().max_body_bytes
            size = _content_length(scope)
            if limit > 0 and size is not None and size > limit:
                await _reject(send, 413, f"Request body exceeds {limit} bytes")
                return
        await self.app(scope, receive, send)


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        limit = get_settings().rate_limit_per_minute
        if limit <= 0 or path not in _LIMITED:
            await self.app(scope, receive, send)
            return
        now = time.monotonic()
        key = _client_key(scope)
        with _lock:
            bucket = _windows[key]
            cutoff = now - 60.0
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry = 60 if not bucket else max(1, int(60 - (now - bucket[0])))
                await _reject(
                    send,
                    429,
                    "Rate limit exceeded. Try again shortly.",
                    extra_headers=[(b"retry-after", str(retry).encode())],
                )
                return
            bucket.append(now)
        await self.app(scope, receive, send)
