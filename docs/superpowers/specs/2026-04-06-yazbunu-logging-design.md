# Yazbunu — Centralized Logging System

**Date**: 2026-04-06
**Status**: Approved

## Problem

KutAI is being modularized into a hub of submodules. Needs a centralized logging system that is:
- Plug-and-play for any submodule (inside or outside the kutai package)
- Browsable on mobile via Tailscale (real-time tail, search, filter)
- Near-zero RAM on the host (VRAM/RAM is the system bottleneck)
- Loki-compatible schema so Grafana dashboards can be added later

Previous notification system (ntfy) was removed — no current way to browse logs remotely.

## Constraints

- RAM: log viewer process budget ~20-30MB, host RAM hits 100% under LLM load
- Mobile: Android phone connected via Tailscale, must work in browser
- Volume: ~30k lines/day at 24/7 operation, bursts to ~2700 lines/hour during active tasks
- Submodules: may live inside `kutai.*` or as standalone packages — library must work for both

## Design

### 1. JSONL Schema (the contract)

Every submodule writes one JSON object per line:

```json
{"ts":"2026-04-06T12:24:31.930Z","level":"INFO","src":"kutai.core.orchestrator","msg":"task dispatched","task":"42","agent":"coder","model":"qwen-32b","mission":"m-7"}
```

**Required fields**:
| Field | Type | Description |
|-------|------|-------------|
| `ts` | string | ISO 8601 UTC timestamp |
| `level` | string | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `src` | string | Dotted logger name: `{project}.{component}.{sub}` |
| `msg` | string | Human-readable log message |

**Auto-populated on WARNING+**:
| Field | Type | Description |
|-------|------|-------------|
| `fn` | string | Function name (from stack frame) |
| `ln` | int | Line number (from stack frame) |

**Common optional context fields**:
| Field | Type | Description |
|-------|------|-------------|
| `task` | string | Task ID |
| `mission` | string | Mission ID |
| `agent` | string | Agent name |
| `model` | string | LLM model name |
| `exc` | string | Exception traceback (on error) |

Any additional key-value pairs are allowed — the viewer renders them as-is.

**File naming**: each submodule writes to `{log_dir}/{project_name}.jsonl` (e.g. `logs/kutai.jsonl`, `logs/shopping.jsonl`).

**Rotation**: 5 files x 50MB = 250MB max per submodule (configurable).

### 2. Yazbunu Package

Standalone Python package at repo root: `yazbunu/`

```
yazbunu/
  pyproject.toml
  yazbunu/
    __init__.py       # get_logger(), init_logging()
    formatter.py      # _JsonFormatter with new schema
    server.py         # aiohttp log viewer server
    viewer.html       # single-file PWA log viewer
    manifest.json     # PWA manifest for "Add to Home Screen"
```

**Dependencies**: stdlib + `aiohttp` (for server only; logging itself is stdlib-only)

#### Public API

```python
from yazbunu import get_logger, init_logging

# Called once at startup
init_logging(
    log_dir="./logs",           # directory for JSONL files
    project="kutai",            # prefix for log file name
    console=True,               # enable console output
    level="DEBUG",              # minimum log level
)

# Used everywhere
logger = get_logger("core.orchestrator")  # becomes "kutai.core.orchestrator" via project prefix
logger.info("task dispatched", task="42", mission="m-7")
logger.error("tool failed", task="42", fn_override=True)  # fn/ln auto-populated on WARNING+

# Bound loggers for long-running contexts
ctx_logger = logger.bind(task="42", mission="m-7")
ctx_logger.info("step completed")  # task and mission auto-included
```

#### Submodule Integration

**Inside kutai package**: `src/infra/logging_config.py` becomes a thin re-export:
```python
from yazbunu import get_logger, init_logging
```

**Outside kutai package**: `pip install -e ../yazbunu` or add as dependency, then import directly:
```python
from yazbunu import get_logger, init_logging
init_logging(log_dir="../logs", project="mysubmodule")
```

**Non-Python submodules**: just write the same JSONL format to the `logs/` directory. Yazbunu is a convenience, not a requirement.

### 3. Log Viewer Server

Standalone aiohttp process: `python -m yazbunu.server --log-dir ./logs --port 9880`

**RAM**: ~20-30MB, runs independently of KutAI (available during development, crashes, restarts).

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the HTML viewer |
| GET | `/api/files` | Lists all `.jsonl` files in log directory |
| GET | `/api/logs?file=kutai.jsonl&lines=1000` | Returns last N lines from a file |
| GET | `/api/tail?file=kutai.jsonl&after=<iso-ts>` | Returns lines after a timestamp (for live polling) |

All API responses are JSON. The `/api/tail` endpoint enables efficient live tail — the viewer only fetches new lines, not the entire file.

**Binding**: `0.0.0.0:9880` — accessible over Tailscale from mobile.

### 4. HTML Viewer (PWA)

Single HTML file, no build step, no framework, no npm.

**Features**:
- **File picker**: dropdown of all discovered `.jsonl` files, plus "All" to merge
- **Live tail**: polls `/api/tail` every 2 seconds, auto-scrolls, pause button
- **Level filter**: dropdown (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- **Text search**: regex search across `src` and `msg` fields
- **Time range**: date-time picker for historical browsing
- **Context pills**: extra fields (`task`, `mission`, `agent`, `model`) rendered as colored clickable pills — click to filter by value
- **Color coding**: row background by level (red=ERROR, yellow=WARNING, gray=DEBUG)
- **Mobile-first**: responsive layout, touch-friendly controls, readable on small screens
- **Permalink**: filter state encoded in URL hash for sharing
- **PWA**: manifest + service worker for "Add to Home Screen" on Android — launches fullscreen, own icon

### 5. Migration from Current Logging

Current `logging_config.py` field mapping:
| Current | New |
|---------|-----|
| `timestamp` | `ts` |
| `component` | `src` |
| `message` | `msg` |
| `task_id` | `task` |
| `mission_id` | `mission` |
| `agent_type` | `agent` |
| `duration_ms` | `duration_ms` (unchanged) |

The existing `_ContextLogger` and `_BoundLogger` APIs are preserved — only the internal formatter changes. No call-site modifications needed beyond the import path.

### 6. Future: Loki Integration

The JSONL schema is designed to be Loki-compatible. When ready:
1. Add Promtail config to watch `logs/*.jsonl`
2. Labels: `project`, `level` (from file name and field)
3. Grafana datasource already exists — just add log dashboards
4. Yazbunu viewer continues to work as a lightweight alternative

No schema changes needed for Loki adoption.
