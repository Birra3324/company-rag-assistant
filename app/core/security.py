"""Require a configured service key for data and model endpoints."""
from secrets import compare_digest
from fastapi import Header, HTTPException
from app.core.config import get_settings

async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = get_settings().api_key
    if not expected or not expected.strip():
        raise HTTPException(503, "API authentication is not configured")
    if not x_api_key or not compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "Invalid or missing API key")
