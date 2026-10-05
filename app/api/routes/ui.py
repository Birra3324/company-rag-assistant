"""Browser demo: one static page for asking and inspecting citations."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["ui"])

_PAGE = Path(__file__).resolve().parents[2] / "static" / "index.html"


@router.get("/ui", include_in_schema=False)
def ui_page() -> FileResponse:
    return FileResponse(_PAGE, media_type="text/html; charset=utf-8")
