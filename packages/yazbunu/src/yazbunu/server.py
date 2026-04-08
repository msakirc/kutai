"""
Yazbunu log viewer server.

Lightweight aiohttp app serving the log viewer PWA and a minimal API
for reading JSONL log files.

Usage:
    python -m yazbunu.server --log-dir ./logs --port 9880
"""

import argparse
import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path

import aiohttp
from aiohttp import web

from yazbunu.auth import (
    build_url,
    get_or_create_token,
    render_qr_page,
    validate_token,
)

__version__ = "0.2.0"

STATIC_DIR = Path(__file__).parent / "static"

_start_time = time.monotonic()


def _safe_filename(log_dir: str, filename: str) -> Path | None:
    """Validate filename — no path traversal, must be .jsonl, must exist."""
    if ".." in filename or "/" in filename or "\\" in filename:
        return None
    if not filename.endswith(".jsonl"):
        return None
    path = Path(log_dir) / filename
    if not path.is_file():
        return None
    return path


def _extract_ts(line: str) -> str:
    """Fast substring extraction of 'ts' value without JSON parsing.

    Looks for "ts":"..." or "ts": "..." patterns.
    """
    idx = line.find('"ts"')
    if idx == -1:
        return ""
    # Skip past "ts" and find the colon, then the opening quote
    rest = line[idx + 4:]
    colon = rest.find(":")
    if colon == -1:
        return ""
    rest = rest[colon + 1:].lstrip()
    if not rest or rest[0] != '"':
        return ""
    end = rest.find('"', 1)
    if end == -1:
        return ""
    return rest[1:end]


def _extract_field(line: str, field: str) -> str:
    """Fast substring extraction of a JSON string field value."""
    # Try "field":"value" and "field": "value"
    for pattern in (f'"{field}":', f'"{field}" :'):
        idx = line.find(pattern)
        if idx == -1:
            continue
        rest = line[idx + len(pattern):].lstrip()
        if not rest or rest[0] != '"':
            continue
        end = rest.find('"', 1)
        if end == -1:
            continue
        return rest[1:end]
    return ""


def create_app(
    log_dir: str,
    *,
    auth_dir: str | None = None,
    require_auth: bool = False,
    host: str = "0.0.0.0",
    port: int = 9880,
) -> web.Application:
    # --- Auth setup ---
    token: str | None = None
    viewer_url: str | None = None
    if auth_dir:
        token = get_or_create_token(auth_dir)
        viewer_url = build_url(host, port, token)

    # --- Auth middleware ---
    _OPEN_PATHS = frozenset({"/health", "/auth/qr"})

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        path = request.path

        # Always open paths
        if path in _OPEN_PATHS or path.startswith("/static/"):
            return await handler(request)

        # Root with token param — set cookie
        if path == "/" and request.query.get("token") and token:
            if validate_token(request.query["token"], token):
                resp = await handler(request)
                resp.set_cookie(
                    "_yz_token",
                    token,
                    max_age=30 * 24 * 3600,
                    httponly=True,
                    samesite="Lax",
                )
                return resp

        if not require_auth:
            return await handler(request)

        # Check auth: query param, header, cookie
        req_token = (
            request.query.get("token")
            or _bearer_token(request)
            or request.cookies.get("_yz_token", "")
        )
        if token and validate_token(req_token, token):
            return await handler(request)

        return web.json_response({"error": "unauthorized"}, status=401)

    def _bearer_token(request: web.Request) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return ""

    app = web.Application(middlewares=[auth_middleware])
    app["log_dir"] = log_dir
    app["_token"] = token
    app["_viewer_url"] = viewer_url

    # --- Route handlers ---

    async def handle_index(request: web.Request) -> web.Response:
        viewer_path = STATIC_DIR / "viewer.html"
        return web.FileResponse(viewer_path)

    async def handle_list_files(request: web.Request) -> web.Response:
        ld = request.app["log_dir"]
        files = []
        for f in sorted(Path(ld).glob("*.jsonl")):
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
        return web.json_response({"files": files})

    async def handle_get_logs(request: web.Request) -> web.Response:
        ld = request.app["log_dir"]
        filename = request.query.get("file", "")
        lines_count = int(request.query.get("lines", "1000"))

        if ".." in filename or "/" in filename or "\\" in filename:
            return web.json_response({"error": "invalid filename"}, status=400)

        path = _safe_filename(ld, filename)
        if path is None:
            return web.json_response({"error": "file not found"}, status=404)

        # Ring buffer — iterate lines, keep last N without loading entire file
        ring: deque[str] = deque(maxlen=lines_count)
        total = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n\r")
                if stripped:
                    ring.append(stripped)
                    total += 1

        return web.json_response({"lines": list(ring), "total": total})

    async def handle_tail(request: web.Request) -> web.Response:
        ld = request.app["log_dir"]
        filename = request.query.get("file", "")
        after = request.query.get("after", "")

        if ".." in filename or "/" in filename or "\\" in filename:
            return web.json_response({"error": "invalid filename"}, status=400)

        path = _safe_filename(ld, filename)
        if path is None:
            return web.json_response({"error": "file not found"}, status=404)

        result = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n\r")
                if not stripped:
                    continue
                ts = _extract_ts(stripped)
                if ts > after:
                    result.append(stripped)

        return web.json_response({"lines": result})

    async def handle_health(request: web.Request) -> web.Response:
        uptime = round(time.monotonic() - _start_time, 1)
        data = {
            "status": "ok",
            "version": __version__,
            "uptime_sec": uptime,
            "token": token or "",
            "url": viewer_url or "",
        }
        return web.json_response(data)

    async def handle_stats(request: web.Request) -> web.Response:
        ld = request.app["log_dir"]
        filename = request.query.get("file", "")

        path = _safe_filename(ld, filename)
        if path is None:
            return web.json_response({"error": "file not found"}, status=404)

        total = 0
        levels: dict[str, int] = {}
        sources: dict[str, int] = {}
        first_ts = ""
        last_ts = ""

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n\r")
                if not stripped:
                    continue
                total += 1

                # Extract level via substring
                level = _extract_field(stripped, "level")
                if level:
                    levels[level] = levels.get(level, 0) + 1

                # Extract source
                src = _extract_field(stripped, "src")
                if src:
                    sources[src] = sources.get(src, 0) + 1

                # Extract timestamp
                ts = _extract_ts(stripped)
                if ts:
                    if not first_ts:
                        first_ts = ts
                    last_ts = ts

        # Top sources sorted by count descending
        top_sources = sorted(sources.items(), key=lambda x: -x[1])[:10]

        return web.json_response({
            "total": total,
            "levels": levels,
            "time_range": {"first": first_ts, "last": last_ts},
            "top_sources": top_sources,
            "indexed": False,
        })

    async def handle_context(request: web.Request) -> web.Response:
        ld = request.app["log_dir"]
        filename = request.query.get("file", "")
        field = request.query.get("field", "")
        value = request.query.get("value", "")

        if not field or not value:
            return web.json_response({"error": "field and value required"}, status=400)

        path = _safe_filename(ld, filename)
        if path is None:
            return web.json_response({"error": "file not found"}, status=404)

        # Substring patterns: "field":"value" and "field": "value"
        patterns = [f'"{field}":"{value}"', f'"{field}": "{value}"']
        result = []

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n\r")
                if not stripped:
                    continue
                for pat in patterns:
                    if pat in stripped:
                        result.append(stripped)
                        break

        return web.json_response({"lines": result})

    async def handle_ws_tail(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        ld = request.app["log_dir"]
        filename = request.query.get("file", "")
        path = _safe_filename(ld, filename)
        if path is None:
            await ws.send_json({"type": "error", "error": "file not found"})
            await ws.close()
            return ws

        # Start at current end of file
        try:
            offset = os.stat(path).st_size
        except OSError:
            await ws.send_json({"type": "error", "error": "cannot stat file"})
            await ws.close()
            return ws

        try:
            while not ws.closed:
                await asyncio.sleep(2)
                try:
                    new_size = os.stat(path).st_size
                except OSError:
                    break
                if new_size > offset:
                    with open(path, encoding="utf-8") as fh:
                        fh.seek(offset)
                        new_data = fh.read(new_size - offset)
                    offset = new_size
                    lines = [
                        l for l in new_data.split("\n") if l.strip()
                    ]
                    if lines:
                        await ws.send_json({"type": "lines", "lines": lines})
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            if not ws.closed:
                await ws.close()

        return ws

    async def handle_qr(request: web.Request) -> web.Response:
        url = viewer_url or build_url(host, port, token or "", tls=False)
        html = render_qr_page(url)
        return web.Response(text=html, content_type="text/html")

    # --- Routes ---
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/auth/qr", handle_qr)
    app.router.add_get("/api/files", handle_list_files)
    app.router.add_get("/api/logs", handle_get_logs)
    app.router.add_get("/api/tail", handle_tail)
    app.router.add_get("/api/stats", handle_stats)
    app.router.add_get("/api/context", handle_context)
    app.router.add_get("/ws/tail", handle_ws_tail)

    # Serve static files (manifest.json, etc.)
    if STATIC_DIR.is_dir():
        app.router.add_static("/static/", STATIC_DIR)

    return app


def main():
    parser = argparse.ArgumentParser(description="Yazbunu log viewer server")
    parser.add_argument("--log-dir", default="./logs", help="Directory containing .jsonl log files")
    parser.add_argument("--port", type=int, default=9880, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--auth-dir", default=None, help="Directory for auth token storage")
    parser.add_argument("--require-auth", action="store_true", help="Require token auth")
    args = parser.parse_args()

    print(f"Yazbunu server starting on http://{args.host}:{args.port}")
    print(f"Log directory: {os.path.abspath(args.log_dir)}")
    app = create_app(
        args.log_dir,
        auth_dir=args.auth_dir,
        require_auth=args.require_auth,
        host=args.host,
        port=args.port,
    )
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
