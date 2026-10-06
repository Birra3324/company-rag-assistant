#!/usr/bin/env python3
"""Capture docs/ui-*.png from the local demo. Requires Google Chrome. Not used in CI.

Usage (from the repo root, after pip install -r requirements.txt):

    python3 scripts/capture_ui.py
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CHROME = os.environ.get("CHROME_BIN", "/usr/local/bin/google-chrome")
API_KEY = "local-demo-key"
PORT = int(os.environ.get("CAPTURE_PORT", "8788"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Cdp:
    def __init__(self, ws_url: str) -> None:
        import websockets.sync.client

        self.ws = websockets.sync.client.connect(
            ws_url, max_size=32 * 1024 * 1024, legacy=True
        )
        self._id = 0

    def call(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        self._id += 1
        msg_id = self._id
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.ws.recv(timeout=max(0.1, deadline - time.time()))
            msg = json.loads(raw)
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method} failed: {msg['error']}")
                return msg.get("result") or {}
        raise TimeoutError(method)

    def close(self) -> None:
        self.ws.close()


def _wait_http(url: str, proc: subprocess.Popen[str], timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited before {url} was up")
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status < 500:
                    return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(url)


def _shot(cdp: Cdp, path: Path) -> None:
    metrics = cdp.call(
        "Runtime.evaluate",
        {
            "expression": "({width: 1200, height: Math.max(900, document.documentElement.scrollHeight)})",
            "returnByValue": True,
        },
    )
    size = metrics["result"]["value"]
    cdp.call(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": int(size["width"]),
            "height": int(size["height"]),
            "deviceScaleFactor": 1,
            "mobile": False,
        },
    )
    png = cdp.call(
        "Page.captureScreenshot",
        {"format": "png", "captureBeyondViewport": True, "fromSurface": True},
    )
    path.write_bytes(base64.b64decode(png["data"]))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> int:
    if not Path(CHROME).exists():
        print(f"Chrome not found at {CHROME}", file=sys.stderr)
        return 1
    DOCS.mkdir(parents=True, exist_ok=True)
    debug_port = _free_port()
    store = Path("/tmp/rag-ui-capture")
    store.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "APP_ENV": "development",
            "API_KEY": API_KEY,
            "EMBEDDING_PROVIDER": "local",
            "LLM_PROVIDER": "extractive",
            "VECTOR_BACKEND": "sqlite",
            "VECTOR_STORE_PATH": str(store / "rag.sqlite3"),
            "AUTO_INGEST_ON_STARTUP": "true",
            "DOCS_PATH": str(ROOT / "data" / "sample_docs"),
        }
    )
    server_log = open("/tmp/rag-ui-capture-server.log", "w", encoding="utf-8")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        cwd=ROOT,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    profile = Path(f"/tmp/chrome-rag-capture-{debug_port}")
    chrome = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            f"--remote-debugging-port={debug_port}",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http(f"http://127.0.0.1:{PORT}/health", server)
        _wait_http(f"http://127.0.0.1:{debug_port}/json/version", chrome)
        new_target = urllib.request.Request(
            f"http://127.0.0.1:{debug_port}/json/new?http://127.0.0.1:{PORT}/ui",
            method="PUT",
        )
        with urllib.request.urlopen(new_target) as resp:
            target = json.load(resp)
        cdp = Cdp(target["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1200, "height": 900, "deviceScaleFactor": 1, "mobile": False},
        )
        cdp.call("Page.navigate", {"url": f"http://127.0.0.1:{PORT}/ui"})
        time.sleep(0.6)
        typed = cdp.call(
            "Runtime.evaluate",
            {
                "expression": """
                (() => {
                  document.querySelector('#api-key').value = 'local-demo-key';
                  document.querySelector('#question').value = 'How many PTO days do employees receive?';
                  return document.querySelector('#question').value;
                })()
                """,
                "returnByValue": True,
            },
        )
        if "PTO" not in str(typed.get("result", {}).get("value")):
            raise RuntimeError(f"failed to fill the form: {typed}")
        _shot(cdp, DOCS / "ui-ask.png")
        answered = cdp.call(
            "Runtime.evaluate",
            {
                "expression": """
                (async () => {
                  document.querySelector('#ask-form').requestSubmit();
                  const start = Date.now();
                  while (Date.now() - start < 15000) {
                    const result = document.querySelector('#result');
                    const err = document.querySelector('#error');
                    if (err && !err.classList.contains('hidden') && err.textContent.trim()) {
                      return 'error:' + err.textContent.trim();
                    }
                    if (result && !result.classList.contains('hidden') && /20/.test(result.innerText)) {
                      return 'ok';
                    }
                    await new Promise((r) => setTimeout(r, 100));
                  }
                  return 'timeout:' + document.body.innerText.slice(0, 400);
                })()
                """,
                "awaitPromise": True,
                "returnByValue": True,
            },
            timeout=20,
        )
        status = str(answered.get("result", {}).get("value"))
        if status != "ok":
            raise RuntimeError(f"ask did not render: {status}")
        _shot(cdp, DOCS / "ui-answer.png")
        cdp.close()
    finally:
        chrome.terminate()
        server.terminate()
        try:
            chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome.kill()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
