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
from pathlib import Path

from aiohttp import web

STATIC_DIR = Path(__file__).parent / "static"


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


DEFAULT_TRACE_KEYS = ["task", "mission_id", "agent_type"]


def create_app(log_dir: str, trace_keys: list[str] | None = None) -> web.Application:
    app = web.Application()
    app["log_dir"] = log_dir
    app["trace_keys"] = trace_keys if trace_keys is not None else DEFAULT_TRACE_KEYS

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

        # Read last N lines efficiently
        all_lines = path.read_text(encoding="utf-8").strip().split("\n")
        tail = all_lines[-lines_count:] if lines_count < len(all_lines) else all_lines

        return web.json_response({"lines": tail, "total": len(all_lines)})

    async def handle_get_all_logs(request: web.Request) -> web.Response:
        """Load last N lines merged from all JSONL files, sorted by timestamp."""
        ld = request.app["log_dir"]
        lines_count = int(request.query.get("lines", "2000"))
        all_lines = []
        for f in Path(ld).glob("*.jsonl"):
            raw = f.read_text(encoding="utf-8").strip()
            if raw:
                all_lines.extend(raw.split("\n"))
        # Sort by timestamp (ts field is first in JSONL, string sort works for ISO)
        def ts_key(line):
            try:
                return json.loads(line).get("ts", "")
            except (json.JSONDecodeError, AttributeError):
                return ""
        all_lines.sort(key=ts_key)
        tail = all_lines[-lines_count:] if lines_count < len(all_lines) else all_lines
        return web.json_response({"lines": tail, "total": len(all_lines)})

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
        for raw_line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not raw_line:
                continue
            try:
                doc = json.loads(raw_line)
                if doc.get("ts", "") > after:
                    result.append(raw_line)
            except json.JSONDecodeError:
                continue

        return web.json_response({"lines": result})

    async def handle_health(request: web.Request) -> web.Response:
        ld = request.app["log_dir"]
        file_count = len(list(Path(ld).glob("*.jsonl")))
        return web.json_response({"status": "ok", "files": file_count})

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
                    lines = [l for l in new_data.split("\n") if l.strip()]
                    if lines:
                        await ws.send_json({"type": "lines", "lines": lines})
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            if not ws.closed:
                await ws.close()

        return ws

    async def handle_config(request: web.Request) -> web.Response:
        return web.json_response({"trace_keys": request.app["trace_keys"]})

    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/config", handle_config)
    app.router.add_get("/api/files", handle_list_files)
    app.router.add_get("/api/logs", handle_get_logs)
    app.router.add_get("/api/logs/all", handle_get_all_logs)
    app.router.add_get("/api/tail", handle_tail)
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
    parser.add_argument("--trace-keys", default=None,
                        help="Comma-separated context keys for cross-component tracing (default: task,mission_id,agent_type)")
    args = parser.parse_args()

    trace_keys = args.trace_keys.split(",") if args.trace_keys else None
    print(f"Yazbunu server starting on http://{args.host}:{args.port}")
    print(f"Log directory: {os.path.abspath(args.log_dir)}")
    app = create_app(args.log_dir, trace_keys=trace_keys)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
