"""Structured logging with request_id via contextvars."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return request_id_ctx.get()


class RequestIdFilter(logging.Filter):
    """Inject request_id from the context var so format strings always work."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s %(levelname)s request_id=%(request_id)s %(name)s %(message)s"
    root = logging.getLogger()
    root.setLevel(level.upper())

    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_rag_fmt", False)
        for h in root.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        handler.addFilter(RequestIdFilter())
        handler._rag_fmt = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    rid_filter = RequestIdFilter()
    for handler in root.handlers:
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(rid_filter)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = request_id_ctx.set(rid)
        request.state.request_id = rid
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["x-request-id"] = rid
        return response
