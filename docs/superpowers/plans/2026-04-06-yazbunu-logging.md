# Yazbunu Logging System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `yazbunu`, a standalone centralized logging package with a mobile-friendly web viewer, plug-and-play for any submodule inside or outside the kutai package.

**Architecture:** Standalone Python package (`yazbunu/`) at repo root. Provides `get_logger()` + `init_logging()` with JSONL output (Loki-compatible schema). Includes an aiohttp-based log viewer server (~20-30MB) serving a single-file PWA that supports live tail, filtering, and search over Tailscale on mobile.

**Tech Stack:** Python stdlib `logging`, `aiohttp` (already a dependency), vanilla HTML/CSS/JS (no framework)

**Spec:** `docs/superpowers/specs/2026-04-06-yazbunu-logging-design.md`

---

## File Structure

```
yazbunu/                        # package root (repo root level)
  pyproject.toml                # minimal packaging
  yazbunu/
    __init__.py                 # get_logger(), init_logging(), public API
    formatter.py                # _JsonFormatter — new JSONL schema
    server.py                   # aiohttp log viewer server
    static/
      viewer.html               # single-file PWA log viewer
      manifest.json             # PWA manifest
tests/
  test_yazbunu.py               # unit tests for logging lib
  test_yazbunu_server.py        # server API tests
src/infra/logging_config.py     # modified — becomes re-export shim
```

---

### Task 1: Package Skeleton + JSONL Formatter

**Files:**
- Create: `yazbunu/pyproject.toml`
- Create: `yazbunu/yazbunu/__init__.py`
- Create: `yazbunu/yazbunu/formatter.py`
- Test: `tests/test_yazbunu.py`

- [ ] **Step 1: Write failing test for the JSONL formatter**

Create `tests/test_yazbunu.py`:

```python
"""Tests for yazbunu logging library."""
import json
import logging
import sys
import os

# yazbunu is a sibling directory at repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yazbunu"))

from yazbunu.formatter import YazFormatter


def test_formatter_required_fields():
    """Formatter output contains ts, level, src, msg."""
    fmt = YazFormatter()
    record = logging.LogRecord(
        name="kutai.core.orchestrator",
        level=logging.INFO,
        pathname="orchestrator.py",
        lineno=42,
        msg="task dispatched",
        args=(),
        exc_info=None,
    )
    line = fmt.format(record)
    doc = json.loads(line)
    assert "ts" in doc
    assert doc["level"] == "INFO"
    assert doc["src"] == "kutai.core.orchestrator"
    assert doc["msg"] == "task dispatched"
    # INFO should NOT have fn/ln
    assert "fn" not in doc
    assert "ln" not in doc


def test_formatter_warning_includes_fn_ln():
    """WARNING+ records include fn and ln fields."""
    fmt = YazFormatter()
    record = logging.LogRecord(
        name="kutai.agents.base",
        level=logging.WARNING,
        pathname="base.py",
        lineno=284,
        msg="tool exec failed",
        args=(),
        exc_info=None,
    )
    record.funcName = "_run_tool"
    line = fmt.format(record)
    doc = json.loads(line)
    assert doc["fn"] == "_run_tool"
    assert doc["ln"] == 284


def test_formatter_context_fields():
    """Extra context fields (task, mission, agent, model) appear in output."""
    fmt = YazFormatter()
    record = logging.LogRecord(
        name="kutai.core.orchestrator",
        level=logging.INFO,
        pathname="orchestrator.py",
        lineno=42,
        msg="task dispatched",
        args=(),
        exc_info=None,
    )
    record.task = "42"
    record.mission = "m-7"
    record.agent = "coder"
    record.model = "qwen-32b"
    line = fmt.format(record)
    doc = json.loads(line)
    assert doc["task"] == "42"
    assert doc["mission"] == "m-7"
    assert doc["agent"] == "coder"
    assert doc["model"] == "qwen-32b"


def test_formatter_exception():
    """Exception info is captured in exc field."""
    fmt = YazFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="kutai.core.orchestrator",
        level=logging.ERROR,
        pathname="orchestrator.py",
        lineno=42,
        msg="something failed",
        args=(),
        exc_info=exc_info,
    )
    line = fmt.format(record)
    doc = json.loads(line)
    assert "exc" in doc
    assert "ValueError" in doc["exc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yazbunu.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'yazbunu'`

- [ ] **Step 3: Create package skeleton**

Create `yazbunu/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "yazbunu"
version = "0.1.0"
description = "Lightweight structured JSONL logging with a mobile-friendly web viewer"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
server = ["aiohttp>=3.9.0"]

[project.scripts]
yazbunu-server = "yazbunu.server:main"
```

Create `yazbunu/yazbunu/__init__.py`:

```python
"""
Yazbunu — structured JSONL logging for the KutAI ecosystem.

Usage:
    from yazbunu import get_logger, init_logging

    init_logging(log_dir="./logs", project="kutai")
    logger = get_logger("core.orchestrator")
    logger.info("task dispatched", task="42", mission="m-7")
"""

from yazbunu.formatter import YazFormatter

__all__ = ["get_logger", "init_logging", "YazFormatter"]
```

(The `get_logger` and `init_logging` functions will be added in the next task. For now this file just re-exports the formatter.)

Create `yazbunu/yazbunu/formatter.py`:

```python
"""JSONL formatter — the schema contract for all yazbunu logs."""

import json
import logging
from datetime import datetime, timezone

# Context fields that get promoted to top-level JSON keys
CONTEXT_FIELDS = frozenset({
    "task", "mission", "agent", "model", "duration_ms",
})


class YazFormatter(logging.Formatter):
    """
    Outputs one JSON object per line.

    Required: ts, level, src, msg
    WARNING+: fn, ln (auto-populated from LogRecord)
    Optional: any extra context fields set on the record
    """

    def format(self, record: logging.LogRecord) -> str:
        doc: dict = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "src": record.name,
            "msg": record.getMessage(),
        }

        # fn/ln only on WARNING+
        if record.levelno >= logging.WARNING:
            doc["fn"] = record.funcName
            doc["ln"] = record.lineno

        # Known context fields
        for field in CONTEXT_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                doc[field] = val

        # Arbitrary extra context (set via _ContextLogger)
        for key, val in getattr(record, "_yaz_extra", {}).items():
            if key not in doc:
                doc[key] = val

        # Exception
        if record.exc_info and record.exc_info[0] is not None:
            doc["exc"] = self.formatException(record.exc_info)

        return json.dumps(doc, ensure_ascii=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_yazbunu.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yazbunu/ tests/test_yazbunu.py
git commit -m "feat(yazbunu): package skeleton + JSONL formatter with new schema"
```

---

### Task 2: ContextLogger + init_logging

**Files:**
- Modify: `yazbunu/yazbunu/__init__.py`
- Modify: `tests/test_yazbunu.py`

- [ ] **Step 1: Write failing tests for get_logger and init_logging**

Append to `tests/test_yazbunu.py`:

```python
import tempfile
from pathlib import Path
from yazbunu import get_logger, init_logging


def test_get_logger_info(tmp_path):
    """get_logger returns a logger that writes structured JSONL."""
    init_logging(log_dir=str(tmp_path), project="testproj", console=False)
    logger = get_logger("core.thing")
    logger.info("hello", task="1")

    log_file = tmp_path / "testproj.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    doc = json.loads(lines[-1])
    assert doc["src"] == "testproj.core.thing"
    assert doc["msg"] == "hello"
    assert doc["task"] == "1"


def test_get_logger_bind(tmp_path):
    """Bound loggers carry context across calls."""
    init_logging(log_dir=str(tmp_path), project="testproj2", console=False)
    logger = get_logger("agents.base").bind(task="99", mission="m-3")
    logger.info("step done")

    log_file = tmp_path / "testproj2.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    doc = json.loads(lines[-1])
    assert doc["task"] == "99"
    assert doc["mission"] == "m-3"


def test_init_logging_rotation(tmp_path):
    """Rotating file handler is created with correct params."""
    init_logging(log_dir=str(tmp_path), project="rottest", console=False,
                 max_bytes=1000, backup_count=2)
    logger = get_logger("x")
    # Write enough to trigger rotation
    for i in range(200):
        logger.info(f"line {i}" + "x" * 100)

    files = list(tmp_path.glob("rottest.jsonl*"))
    assert len(files) >= 2  # at least one backup
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yazbunu.py::test_get_logger_info -v`
Expected: FAIL — `ImportError: cannot import name 'get_logger' from 'yazbunu'`

- [ ] **Step 3: Implement get_logger and init_logging**

Replace `yazbunu/yazbunu/__init__.py` with:

```python
"""
Yazbunu — structured JSONL logging for the KutAI ecosystem.

Usage:
    from yazbunu import get_logger, init_logging

    init_logging(log_dir="./logs", project="kutai")
    logger = get_logger("core.orchestrator")
    logger.info("task dispatched", task="42", mission="m-7")
"""

import logging
import logging.handlers
import os
import sys

from yazbunu.formatter import YazFormatter

__all__ = ["get_logger", "init_logging", "YazFormatter"]

_project_prefix: str = ""
_initialized_projects: set[str] = set()

# ─── Reserved LogRecord attributes ───────────────────────────────────────────
_RESERVED = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "process", "processName", "message",
})


class _ContextLogger:
    """
    Thin wrapper around stdlib logger supporting keyword-arg context fields.

    Usage:
        logger.info("msg", task="5", duration_ms=120)
    """

    def __init__(self, name: str):
        self._log = logging.getLogger(name)
        self.name = name

    def _emit(self, level: int, msg: str, ctx: dict):
        safe = {k: v for k, v in ctx.items() if k not in _RESERVED}
        extra = {**safe, "_yaz_extra": safe}
        self._log.log(level, msg, extra=extra)

    def debug(self, msg: str, **ctx):
        self._emit(logging.DEBUG, msg, ctx)

    def info(self, msg: str, **ctx):
        self._emit(logging.INFO, msg, ctx)

    def warning(self, msg: str, **ctx):
        self._emit(logging.WARNING, msg, ctx)

    def error(self, msg: str, **ctx):
        self._emit(logging.ERROR, msg, ctx)

    def critical(self, msg: str, **ctx):
        self._emit(logging.CRITICAL, msg, ctx)

    def exception(self, msg: str, **ctx):
        safe = {k: v for k, v in ctx.items() if k not in _RESERVED}
        extra = {**safe, "_yaz_extra": safe}
        self._log.exception(msg, extra=extra)

    def bind(self, **ctx) -> "_BoundLogger":
        return _BoundLogger(self, ctx)


class _BoundLogger:
    """Logger with pre-bound context fields."""

    def __init__(self, parent: _ContextLogger, bound: dict):
        self._parent = parent
        self._bound = bound

    def _merge(self, ctx: dict) -> dict:
        return {**self._bound, **ctx}

    def debug(self, msg, **ctx): self._parent.debug(msg, **self._merge(ctx))
    def info(self, msg, **ctx): self._parent.info(msg, **self._merge(ctx))
    def warning(self, msg, **ctx): self._parent.warning(msg, **self._merge(ctx))
    def error(self, msg, **ctx): self._parent.error(msg, **self._merge(ctx))
    def critical(self, msg, **ctx): self._parent.critical(msg, **self._merge(ctx))
    def exception(self, msg, **ctx): self._parent.exception(msg, **self._merge(ctx))
    def bind(self, **ctx): return _BoundLogger(self._parent, self._merge(ctx))


def get_logger(component: str) -> _ContextLogger:
    """
    Return a structured logger for the given component name.

    If init_logging(project="foo") was called, the logger name becomes
    "foo.component". Otherwise it's just "component".
    """
    name = f"{_project_prefix}.{component}" if _project_prefix else component
    return _ContextLogger(name)


def init_logging(
    log_dir: str = "./logs",
    project: str = "app",
    console: bool = True,
    level: str = "DEBUG",
    max_bytes: int = 50_000_000,
    backup_count: int = 5,
) -> None:
    """
    Configure logging sinks for a project.

    Args:
        log_dir: Directory for JSONL log files.
        project: Project name prefix — becomes the log filename and logger prefix.
        console: Enable console (stdout) output.
        level: Minimum log level.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated backup files to keep.
    """
    global _project_prefix

    if project in _initialized_projects:
        return
    _initialized_projects.add(project)
    _project_prefix = project

    os.makedirs(log_dir, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(log_level)

    # JSONL file sink — rotating
    log_path = os.path.join(log_dir, f"{project}.jsonl")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(YazFormatter())
    root.addHandler(file_handler)

    # Console sink — human-readable
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(console_handler)

    # Quiet noisy libraries
    for lib in ("httpcore", "httpx", "aiosqlite", "asyncio", "urllib3",
                "telegram.ext", "aiohttp.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)
```

- [ ] **Step 4: Run all yazbunu tests**

Run: `pytest tests/test_yazbunu.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yazbunu/yazbunu/__init__.py tests/test_yazbunu.py
git commit -m "feat(yazbunu): ContextLogger, BoundLogger, init_logging with rotation"
```

---

### Task 3: Log Viewer Server (API)

**Files:**
- Create: `yazbunu/yazbunu/server.py`
- Create: `tests/test_yazbunu_server.py`

- [ ] **Step 1: Write failing tests for the server API**

Create `tests/test_yazbunu_server.py`:

```python
"""Tests for yazbunu log viewer server API."""
import json
import os
import sys
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yazbunu"))

from yazbunu.server import create_app


@pytest.fixture
def log_dir(tmp_path):
    """Create a temp log dir with sample JSONL data."""
    ts_base = "2026-04-06T12:00:0"
    lines = []
    for i in range(50):
        doc = {
            "ts": f"{ts_base}{i % 10}.000Z",
            "level": "ERROR" if i == 49 else "INFO",
            "src": "kutai.core.orchestrator",
            "msg": f"line {i}",
            "task": str(i),
        }
        lines.append(json.dumps(doc))
    (tmp_path / "kutai.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "shopping.jsonl").write_text(
        json.dumps({"ts": "2026-04-06T13:00:00.000Z", "level": "INFO",
                     "src": "shopping.scraper", "msg": "scrape done"}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
async def client(log_dir, aiohttp_client):
    app = create_app(str(log_dir))
    return await aiohttp_client(app)


@pytest.mark.asyncio
async def test_list_files(client):
    resp = await client.get("/api/files")
    assert resp.status == 200
    data = await resp.json()
    names = [f["name"] for f in data["files"]]
    assert "kutai.jsonl" in names
    assert "shopping.jsonl" in names


@pytest.mark.asyncio
async def test_get_logs_last_n(client):
    resp = await client.get("/api/logs?file=kutai.jsonl&lines=5")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["lines"]) == 5
    # Should be the LAST 5 lines
    assert json.loads(data["lines"][-1])["msg"] == "line 49"


@pytest.mark.asyncio
async def test_get_logs_default_1000(client):
    resp = await client.get("/api/logs?file=kutai.jsonl")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["lines"]) == 50  # all lines (fewer than 1000)


@pytest.mark.asyncio
async def test_tail_after_timestamp(client):
    resp = await client.get("/api/tail?file=kutai.jsonl&after=2026-04-06T12:00:05.000Z")
    assert resp.status == 200
    data = await resp.json()
    # Should return lines with ts > the given timestamp
    for line_str in data["lines"]:
        doc = json.loads(line_str)
        assert doc["ts"] >= "2026-04-06T12:00:05.000Z"


@pytest.mark.asyncio
async def test_file_not_found(client):
    resp = await client.get("/api/logs?file=nonexistent.jsonl")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_path_traversal_blocked(client):
    resp = await client.get("/api/logs?file=../../../etc/passwd")
    assert resp.status == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yazbunu_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'yazbunu.server'`

- [ ] **Step 3: Implement the server**

Create `yazbunu/yazbunu/server.py`:

```python
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
```

Also create `yazbunu/yazbunu/__main__.py` for `python -m yazbunu.server`:

```python
from yazbunu.server import main
main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_yazbunu_server.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yazbunu/yazbunu/server.py yazbunu/yazbunu/__main__.py tests/test_yazbunu_server.py
git commit -m "feat(yazbunu): aiohttp log viewer server with file list, tail, and path traversal protection"
```

---

### Task 4: HTML Viewer (PWA)

**Files:**
- Create: `yazbunu/yazbunu/static/viewer.html`
- Create: `yazbunu/yazbunu/static/manifest.json`

- [ ] **Step 1: Create PWA manifest**

Create `yazbunu/yazbunu/static/manifest.json`:

```json
{
  "name": "Yazbunu Log Viewer",
  "short_name": "Yazbunu",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#1a1a2e",
  "theme_color": "#16213e",
  "icons": []
}
```

- [ ] **Step 2: Create the viewer HTML**

Create `yazbunu/yazbunu/static/viewer.html`. This is a single-file PWA with inline CSS and JS. Key sections:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#1a1a2e">
<link rel="manifest" href="/static/manifest.json">
<title>Yazbunu</title>
<style>
  :root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --text: #e0e0e0;
    --text-dim: #888;
    --border: #2a2a4a;
    --accent: #4fc3f7;
    --error: #ef5350;
    --warn: #ffa726;
    --info: #4fc3f7;
    --debug: #888;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 13px;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ─── Toolbar ─── */
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px 12px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    align-items: center;
  }
  .toolbar select, .toolbar input, .toolbar button {
    font-family: inherit;
    font-size: 13px;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 10px;
  }
  .toolbar button {
    cursor: pointer;
    background: var(--accent);
    color: #000;
    border: none;
    font-weight: bold;
  }
  .toolbar button.active {
    background: var(--error);
  }
  .toolbar input[type="text"] {
    flex: 1;
    min-width: 120px;
  }
  .status {
    color: var(--text-dim);
    font-size: 11px;
    margin-left: auto;
  }

  /* ─── Log area ─── */
  .log-area {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
  }
  .log-line {
    display: flex;
    padding: 2px 12px;
    border-bottom: 1px solid var(--border);
    gap: 8px;
    align-items: baseline;
    word-break: break-word;
  }
  .log-line:hover { background: rgba(255,255,255,0.03); }
  .log-line.level-ERROR { background: rgba(239,83,80,0.1); }
  .log-line.level-WARNING { background: rgba(255,167,38,0.1); }
  .log-line.level-CRITICAL { background: rgba(239,83,80,0.2); }
  .log-line.level-DEBUG { opacity: 0.6; }

  .ts { color: var(--text-dim); white-space: nowrap; font-size: 11px; }
  .level { font-weight: bold; width: 50px; font-size: 11px; }
  .level-ERROR .level { color: var(--error); }
  .level-WARNING .level { color: var(--warn); }
  .level-INFO .level { color: var(--info); }
  .level-DEBUG .level { color: var(--debug); }
  .level-CRITICAL .level { color: var(--error); }
  .src { color: var(--accent); white-space: nowrap; }
  .msg { flex: 1; }
  .pill {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    background: var(--border);
    color: var(--text);
    cursor: pointer;
    margin-left: 4px;
  }
  .pill:hover { background: var(--accent); color: #000; }
  .fn-info { color: var(--text-dim); font-size: 11px; }

  /* ─── Mobile ─── */
  @media (max-width: 600px) {
    .ts { display: none; }
    .src { max-width: 100px; overflow: hidden; text-overflow: ellipsis; }
    .toolbar { padding: 6px 8px; gap: 4px; }
    .toolbar select, .toolbar input, .toolbar button { padding: 8px; font-size: 14px; }
    .log-line { padding: 4px 8px; font-size: 14px; }
  }
</style>
</head>
<body>

<div class="toolbar">
  <select id="fileSelect"><option value="">Loading...</option></select>
  <select id="levelFilter">
    <option value="">All levels</option>
    <option value="DEBUG">DEBUG</option>
    <option value="INFO">INFO</option>
    <option value="WARNING">WARNING</option>
    <option value="ERROR">ERROR</option>
    <option value="CRITICAL">CRITICAL</option>
  </select>
  <input type="text" id="searchBox" placeholder="Search src/msg (regex)...">
  <button id="tailBtn">Live Tail</button>
  <span class="status" id="status">Ready</span>
</div>

<div class="log-area" id="logArea"></div>

<script>
const LEVEL_ORDER = {DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4};
const POLL_MS = 2000;
const CONTEXT_KEYS = new Set(["ts", "level", "src", "msg", "fn", "ln", "exc"]);

let allLines = [];
let tailing = false;
let tailTimer = null;
let lastTs = "";
let currentFile = "";

const $file = document.getElementById("fileSelect");
const $level = document.getElementById("levelFilter");
const $search = document.getElementById("searchBox");
const $tailBtn = document.getElementById("tailBtn");
const $area = document.getElementById("logArea");
const $status = document.getElementById("status");

// ─── State from URL hash ───
function loadHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("file")) currentFile = p.get("file");
  if (p.get("level")) $level.value = p.get("level");
  if (p.get("q")) $search.value = p.get("q");
}
function saveHash() {
  const p = new URLSearchParams();
  if (currentFile) p.set("file", currentFile);
  if ($level.value) p.set("level", $level.value);
  if ($search.value) p.set("q", $search.value);
  location.hash = p.toString();
}

// ─── API ───
async function fetchFiles() {
  const r = await fetch("/api/files");
  const d = await r.json();
  $file.innerHTML = d.files.map(f =>
    `<option value="${f.name}">${f.name} (${(f.size/1024).toFixed(0)}KB)</option>`
  ).join("");
  if (currentFile) $file.value = currentFile;
  else currentFile = $file.value;
}

async function fetchLogs(file, lines = 2000) {
  $status.textContent = "Loading...";
  const r = await fetch(`/api/logs?file=${encodeURIComponent(file)}&lines=${lines}`);
  if (!r.ok) { $status.textContent = "Error " + r.status; return; }
  const d = await r.json();
  allLines = d.lines.map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  lastTs = allLines.length ? allLines[allLines.length - 1].ts : "";
  $status.textContent = `${allLines.length} / ${d.total} lines`;
  render();
}

async function fetchTail() {
  if (!currentFile || !lastTs) return;
  const r = await fetch(`/api/tail?file=${encodeURIComponent(currentFile)}&after=${encodeURIComponent(lastTs)}`);
  if (!r.ok) return;
  const d = await r.json();
  if (!d.lines.length) return;
  const newDocs = d.lines.map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  allLines.push(...newDocs);
  lastTs = allLines[allLines.length - 1].ts;
  $status.textContent = `${allLines.length} lines (live)`;
  render();
  $area.scrollTop = $area.scrollHeight;
}

// ─── Rendering ───
function render() {
  const levelMin = LEVEL_ORDER[$level.value] ?? -1;
  let regex = null;
  if ($search.value) {
    try { regex = new RegExp($search.value, "i"); } catch { regex = null; }
  }

  const frag = document.createDocumentFragment();

  for (const doc of allLines) {
    if ((LEVEL_ORDER[doc.level] ?? 0) < levelMin) continue;
    if (regex && !regex.test(doc.src || "") && !regex.test(doc.msg || "")) continue;

    const row = document.createElement("div");
    row.className = `log-line level-${doc.level}`;

    // Timestamp
    const tsEl = document.createElement("span");
    tsEl.className = "ts";
    tsEl.textContent = doc.ts ? doc.ts.slice(11, 23) : "";
    row.appendChild(tsEl);

    // Level
    const lvl = document.createElement("span");
    lvl.className = "level";
    lvl.textContent = doc.level;
    row.appendChild(lvl);

    // Source
    const src = document.createElement("span");
    src.className = "src";
    src.textContent = doc.src || "";
    row.appendChild(src);

    // Message
    const msg = document.createElement("span");
    msg.className = "msg";
    msg.textContent = doc.msg || "";

    // fn/ln info
    if (doc.fn) {
      const fnSpan = document.createElement("span");
      fnSpan.className = "fn-info";
      fnSpan.textContent = ` (${doc.fn}:${doc.ln})`;
      msg.appendChild(fnSpan);
    }

    // Context pills
    for (const [k, v] of Object.entries(doc)) {
      if (CONTEXT_KEYS.has(k)) continue;
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = `${k}=${v}`;
      pill.onclick = () => { $search.value = String(v); saveHash(); render(); };
      msg.appendChild(pill);
    }

    row.appendChild(msg);

    // Exception (collapsible)
    if (doc.exc) {
      const exc = document.createElement("details");
      exc.style.cssText = "color:var(--error);font-size:11px;margin-top:2px;width:100%";
      exc.innerHTML = `<summary>Exception</summary><pre style="white-space:pre-wrap;margin:4px 0">${doc.exc.replace(/</g,"&lt;")}</pre>`;
      row.appendChild(exc);
    }

    frag.appendChild(row);
  }

  $area.innerHTML = "";
  $area.appendChild(frag);
}

// ─── Event handlers ───
$file.onchange = () => {
  currentFile = $file.value;
  saveHash();
  fetchLogs(currentFile);
};

$level.onchange = () => { saveHash(); render(); };
$search.oninput = () => { saveHash(); render(); };

$tailBtn.onclick = () => {
  tailing = !tailing;
  $tailBtn.textContent = tailing ? "Pause" : "Live Tail";
  $tailBtn.classList.toggle("active", tailing);
  if (tailing) {
    tailTimer = setInterval(fetchTail, POLL_MS);
    fetchTail();
  } else {
    clearInterval(tailTimer);
  }
};

// ─── Init ───
loadHash();
fetchFiles().then(() => {
  if ($file.value) fetchLogs($file.value);
});
</script>
</body>
</html>
```

- [ ] **Step 3: Manually verify the viewer**

Run: `python -m yazbunu.server --log-dir ./logs --port 9880`
Open `http://localhost:9880` in a browser. Verify:
- File dropdown shows `orchestrator.jsonl`
- Logs render with colors and pills
- Search and level filter work
- Live tail toggles and streams new lines

- [ ] **Step 4: Commit**

```bash
git add yazbunu/yazbunu/static/
git commit -m "feat(yazbunu): single-file PWA log viewer with live tail, filters, mobile layout"
```

---

### Task 5: Migrate KutAI to Yazbunu

**Files:**
- Modify: `src/infra/logging_config.py`
- Modify: `src/app/run.py`

- [ ] **Step 1: Write failing test for the migration shim**

Append to `tests/test_yazbunu.py`:

```python
def test_kutai_shim_reexports():
    """src.infra.logging_config re-exports yazbunu's public API."""
    from src.infra.logging_config import get_logger, init_logging
    # Verify they are the yazbunu versions
    import yazbunu
    assert get_logger is yazbunu.get_logger
    assert init_logging is yazbunu.init_logging
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yazbunu.py::test_kutai_shim_reexports -v`
Expected: FAIL — `AssertionError` (currently returns the old ContextLogger)

- [ ] **Step 3: Replace logging_config.py with re-export shim**

Replace `src/infra/logging_config.py` with:

```python
# logging_config.py
"""
Structured logging configuration — thin re-export from yazbunu.

All modules continue to import from here:
    from src.infra.logging_config import get_logger
    logger = get_logger("core.orchestrator")

The implementation now lives in the yazbunu package.
"""

import logging
import sys

from yazbunu import get_logger, init_logging, YazFormatter  # noqa: F401

# ─── Fallback before init_logging() runs ─────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiosqlite").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.INFO)
```

- [ ] **Step 4: Update run.py init_logging call**

In `src/app/run.py`, change the `init_logging()` call (line 30) to:

```python
init_logging(log_dir="logs", project="kutai")
```

- [ ] **Step 5: Preserve TelegramAlertHandler attachment**

Add to `src/app/run.py` right after the `init_logging()` call:

```python
# Attach Telegram alert handler (ERROR+) to root logger
try:
    from src.infra.notifications import TelegramAlertHandler
    logging.getLogger().addHandler(TelegramAlertHandler())
except Exception as e:
    _log = get_logger("app.run")
    _log.warning("Could not attach TelegramAlertHandler", error=str(e))
```

And fix the `_log` assignment to come after `init_logging`:

```python
init_logging(log_dir="logs", project="kutai")
_log = get_logger("app.run")
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_yazbunu.py -v && pytest tests/ -x --ignore=tests/integration -q`
Expected: All yazbunu tests pass. Existing tests should not break since the `get_logger` API is preserved.

- [ ] **Step 7: Verify import works for all modules**

Run: `python -c "from src.infra.logging_config import get_logger, init_logging; print('OK')"` 
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add src/infra/logging_config.py src/app/run.py
git commit -m "refactor: migrate logging_config.py to yazbunu re-export shim"
```

---

### Task 6: Install yazbunu as editable + update requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install yazbunu as editable dependency**

Run: `pip install -e ./yazbunu`

- [ ] **Step 2: Add yazbunu to requirements.txt**

Add to `requirements.txt`:

```
# Local packages
-e ./yazbunu
```

- [ ] **Step 3: Verify full import chain works**

Run: `python -c "from yazbunu import get_logger, init_logging; from src.infra.logging_config import get_logger; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/test_yazbunu.py tests/test_yazbunu_server.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "chore: add yazbunu as editable dependency"
```

---

### Task 7: Integration Smoke Test

**Files:** None (verification only)

- [ ] **Step 1: Start the yazbunu server against real logs**

Run: `python -m yazbunu.server --log-dir ./logs --port 9880`

- [ ] **Step 2: Verify API endpoints**

In a second terminal:

```bash
curl http://localhost:9880/api/files
curl "http://localhost:9880/api/logs?file=orchestrator.jsonl&lines=5"
curl "http://localhost:9880/api/tail?file=orchestrator.jsonl&after=2026-04-06T00:00:00Z"
```

Note: existing `orchestrator.jsonl` uses the OLD schema (`timestamp`, `component`). The viewer should still render these — it just won't have nice pills for the old field names. New logs from KutAI will use the new schema after restarting.

- [ ] **Step 3: Verify mobile access**

Open `http://<tailscale-ip>:9880` on the mobile device. Verify:
- Responsive layout works
- Filters work
- Live tail works
- "Add to Home Screen" installs the PWA

- [ ] **Step 4: Stop the server, verify KutAI still starts**

Stop the yazbunu server. Run KutAI normally — it should start and log to `logs/kutai.jsonl` with the new schema.
