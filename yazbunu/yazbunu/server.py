"""
Yazbunu log viewer server.

Lightweight aiohttp app serving the log viewer PWA and a minimal API
for reading JSONL log files.

Usage:
    python -m yazbunu.server --log-dir ./logs --port 9880
"""

import argparse
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


def create_app(log_dir: str) -> web.Application:
    app = web.Application()
    app["log_dir"] = log_dir

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

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/files", handle_list_files)
    app.router.add_get("/api/logs", handle_get_logs)
    app.router.add_get("/api/tail", handle_tail)

    # Serve static files (manifest.json, etc.)
    if STATIC_DIR.is_dir():
        app.router.add_static("/static/", STATIC_DIR)

    return app


def main():
    parser = argparse.ArgumentParser(description="Yazbunu log viewer server")
    parser.add_argument("--log-dir", default="./logs", help="Directory containing .jsonl log files")
    parser.add_argument("--port", type=int, default=9880, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"Yazbunu server starting on http://{args.host}:{args.port}")
    print(f"Log directory: {os.path.abspath(args.log_dir)}")
    app = create_app(args.log_dir)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
