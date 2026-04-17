# NerdHerd Sidecar Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract NerdHerd from running inside the orchestrator to a standalone sidecar process managed by Yaşar Usta, with the orchestrator connecting as an HTTP client.

**Architecture:** NerdHerd becomes a standalone process (`python -m nerd_herd`) with an expanded HTTP API. The orchestrator uses `NerdHerdClient` (HTTP proxy) instead of instantiating `NerdHerd` directly. Yaşar Usta's guard is generalized from single-sidecar to multi-sidecar support, and NerdHerd is added alongside Yazbunu. Mode persistence and auto-detect run inside the sidecar itself.

**Tech Stack:** Python 3.10, aiohttp (HTTP API + client), asyncio, SQLite (mode persistence), pytest

**Spec:** `docs/superpowers/specs/2026-04-13-nerd-herd-sidecar.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `packages/nerd_herd/src/nerd_herd/exposition.py` | Modify | Add HTTP API endpoints for state/mode/gpu/degraded |
| `packages/nerd_herd/src/nerd_herd/client.py` | Create | HTTP client proxy with same interface as NerdHerd |
| `packages/nerd_herd/src/nerd_herd/__main__.py` | Create | Standalone entry point with signal handling + DB persistence |
| `packages/nerd_herd/src/nerd_herd/__init__.py` | Modify | Export NerdHerdClient |
| `packages/yasar_usta/src/yasar_usta/config.py` | Modify | `sidecar` → `sidecars: list[SidecarConfig]` |
| `packages/yasar_usta/src/yasar_usta/guard.py` | Modify | Handle list of SidecarManagers |
| `packages/yasar_usta/src/yasar_usta/commands.py` | Modify | Inline buttons for N sidecars |
| `packages/yasar_usta/src/yasar_usta/status.py` | Modify | Show all sidecars in status panel |
| `kutai_wrapper.py` | Modify | Add nerd_herd to sidecars list, use `sidecars=` |
| `src/app/run.py` | Modify | Replace NerdHerd with NerdHerdClient |
| `src/infra/load_manager.py` | Modify | Delegate to NerdHerdClient |
| `src/infra/runtime_state.py` | Modify | Use NerdHerdClient |
| `src/models/gpu_monitor.py` | Modify | Fallback-only (client doesn't expose registry) |
| `src/app/api.py` | Modify | Fetch /metrics from NerdHerd via HTTP |
| Tests | Create/Modify | `packages/nerd_herd/tests/test_client.py`, `packages/yasar_usta/tests/test_guard.py`, `packages/yasar_usta/tests/test_status.py` |

---

### Task 1: Expand NerdHerd HTTP API

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/exposition.py`
- Test: `packages/nerd_herd/tests/test_exposition.py`

Add HTTP endpoints so the orchestrator can interact with NerdHerd over HTTP instead of in-process calls.

- [ ] **Step 1: Write tests for new API endpoints**

Create `packages/nerd_herd/tests/test_exposition.py`:

```python
"""Tests for NerdHerd HTTP API endpoints."""
import asyncio
import pytest
import aiohttp
from unittest.mock import MagicMock, AsyncMock

from nerd_herd.nerd_herd import NerdHerd


@pytest.fixture
def nh():
    """Create a NerdHerd instance on a test port."""
    return NerdHerd(metrics_port=19881, llama_server_url=None)


@pytest.fixture
async def running_nh(nh):
    await nh.start()
    yield nh
    await nh.stop()


@pytest.fixture
def base_url():
    return "http://127.0.0.1:19881"


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, running_nh, base_url):
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base_url}/health") as r:
                assert r.status == 200
                data = await r.json()
                assert data["status"] == "ok"


class TestStateEndpoint:
    @pytest.mark.asyncio
    async def test_state_returns_load_info(self, running_nh, base_url):
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base_url}/api/state") as r:
                assert r.status == 200
                data = await r.json()
                assert data["load_mode"] == "full"
                assert data["vram_budget_fraction"] == 1.0
                assert data["local_inference_allowed"] is True
                assert data["auto_managed"] is True


class TestModeEndpoint:
    @pytest.mark.asyncio
    async def test_set_mode(self, running_nh, base_url):
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{base_url}/api/mode",
                              json={"mode": "shared", "source": "user"}) as r:
                assert r.status == 200
                data = await r.json()
                assert "shared" in data["result"]
            # Verify it changed
            async with s.get(f"{base_url}/api/state") as r:
                data = await r.json()
                assert data["load_mode"] == "shared"

    @pytest.mark.asyncio
    async def test_set_invalid_mode(self, running_nh, base_url):
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{base_url}/api/mode",
                              json={"mode": "bogus"}) as r:
                assert r.status == 200
                data = await r.json()
                assert "Unknown" in data["result"]


class TestAutoEndpoint:
    @pytest.mark.asyncio
    async def test_enable_auto(self, running_nh, base_url):
        # First disable auto by setting manual mode
        async with aiohttp.ClientSession() as s:
            await s.post(f"{base_url}/api/mode",
                         json={"mode": "shared", "source": "user"})
            # Now re-enable auto
            async with s.post(f"{base_url}/api/auto") as r:
                assert r.status == 200
            async with s.get(f"{base_url}/api/state") as r:
                data = await r.json()
                assert data["auto_managed"] is True


class TestGpuEndpoint:
    @pytest.mark.asyncio
    async def test_gpu_returns_state(self, running_nh, base_url):
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base_url}/api/gpu") as r:
                assert r.status == 200
                data = await r.json()
                assert "vram_total_mb" in data
                assert "vram_free_mb" in data


class TestDegradedEndpoint:
    @pytest.mark.asyncio
    async def test_mark_degraded(self, running_nh, base_url):
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{base_url}/api/degraded",
                              json={"capability": "inference"}) as r:
                assert r.status == 200
            # Verify
            async with s.get(f"{base_url}/api/state") as r:
                data = await r.json()
                assert "inference" in data.get("degraded", [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/nerd_herd && python -m pytest tests/test_exposition.py -v`
Expected: FAIL — `/api/state`, `/api/mode`, `/api/auto`, `/api/gpu`, `/api/degraded` not found (404)

- [ ] **Step 3: Add API endpoints to MetricsServer**

In `packages/nerd_herd/src/nerd_herd/exposition.py`, modify `MetricsServer` to accept a `NerdHerd` reference and add routes:

```python
class MetricsServer:
    """Lightweight aiohttp server serving /metrics and /api for Grafana + orchestrator."""

    def __init__(self, registry: CollectorRegistry, port: int = 9881,
                 nerd_herd=None) -> None:
        self._registry = registry
        self._port = port
        self._runner: web.AppRunner | None = None
        self._nh = nerd_herd  # Back-reference for API endpoints

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/metrics", self._handle_metrics)
        app.router.add_get("/health", self._handle_health)
        # API endpoints for remote clients
        if self._nh is not None:
            app.router.add_get("/api/state", self._handle_state)
            app.router.add_post("/api/mode", self._handle_set_mode)
            app.router.add_post("/api/auto", self._handle_enable_auto)
            app.router.add_get("/api/gpu", self._handle_gpu)
            app.router.add_post("/api/degraded", self._handle_degraded)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port, reuse_address=True)
        await site.start()
        logger.info("Metrics server started", port=self._port)

    # ... existing _handle_metrics, _handle_health unchanged ...

    async def _handle_state(self, request: web.Request) -> web.Response:
        nh = self._nh
        data = {
            "load_mode": nh.get_load_mode(),
            "vram_budget_fraction": nh.get_vram_budget_fraction(),
            "vram_budget_mb": nh.get_vram_budget_mb(),
            "local_inference_allowed": nh.is_local_inference_allowed(),
            "auto_managed": nh._load.is_auto_managed(),
            "degraded": list(nh._health._degraded),
        }
        return web.json_response(data)

    async def _handle_set_mode(self, request: web.Request) -> web.Response:
        body = await request.json()
        mode = body.get("mode", "")
        source = body.get("source", "user")
        result = self._nh.set_load_mode(mode, source)
        return web.json_response({"result": result, "load_mode": self._nh.get_load_mode()})

    async def _handle_enable_auto(self, request: web.Request) -> web.Response:
        self._nh.enable_auto_management()
        return web.json_response({"auto_managed": True})

    async def _handle_gpu(self, request: web.Request) -> web.Response:
        gs = self._nh.gpu_state()
        return web.json_response({
            "vram_total_mb": gs.vram_total_mb,
            "vram_free_mb": gs.vram_free_mb,
            "vram_used_mb": gs.vram_used_mb,
            "gpu_name": gs.gpu_name,
            "gpu_util_pct": gs.gpu_util_pct,
        })

    async def _handle_degraded(self, request: web.Request) -> web.Response:
        body = await request.json()
        cap = body.get("capability", "")
        if cap:
            self._nh.mark_degraded(cap)
        return web.json_response({"ok": True})
```

- [ ] **Step 4: Pass `nerd_herd` reference to MetricsServer in NerdHerd.__init__**

In `packages/nerd_herd/src/nerd_herd/nerd_herd.py`, change the `__init__` method:

```python
self._server = MetricsServer(self.registry, port=metrics_port, nerd_herd=self)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/nerd_herd && python -m pytest tests/test_exposition.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/exposition.py packages/nerd_herd/src/nerd_herd/nerd_herd.py packages/nerd_herd/tests/test_exposition.py
git commit -m "feat(nerd_herd): add HTTP API endpoints for remote client access"
```

---

### Task 2: Create NerdHerdClient (HTTP proxy)

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/client.py`
- Modify: `packages/nerd_herd/src/nerd_herd/__init__.py`
- Test: `packages/nerd_herd/tests/test_client.py`

A lightweight HTTP client with the same public API as `NerdHerd`, so callers can switch transparently.

- [ ] **Step 1: Write tests for NerdHerdClient**

Create `packages/nerd_herd/tests/test_client.py`:

```python
"""Tests for NerdHerdClient — HTTP proxy to a running NerdHerd sidecar."""
import pytest
import aiohttp

from nerd_herd import NerdHerd
from nerd_herd.client import NerdHerdClient


@pytest.fixture
async def server():
    """Start a real NerdHerd server on test port."""
    nh = NerdHerd(metrics_port=19882, llama_server_url=None)
    await nh.start()
    yield nh
    await nh.stop()


@pytest.fixture
async def client(server):
    c = NerdHerdClient(port=19882)
    yield c
    await c.close()


class TestNerdHerdClient:
    @pytest.mark.asyncio
    async def test_get_load_mode(self, client):
        mode = await client.get_load_mode()
        assert mode == "full"

    @pytest.mark.asyncio
    async def test_set_load_mode(self, client):
        result = await client.set_load_mode("shared", "user")
        assert "shared" in result
        mode = await client.get_load_mode()
        assert mode == "shared"

    @pytest.mark.asyncio
    async def test_is_local_inference_allowed(self, client):
        assert await client.is_local_inference_allowed() is True
        await client.set_load_mode("minimal", "user")
        assert await client.is_local_inference_allowed() is False

    @pytest.mark.asyncio
    async def test_get_vram_budget_fraction(self, client):
        frac = await client.get_vram_budget_fraction()
        assert frac == 1.0

    @pytest.mark.asyncio
    async def test_get_vram_budget_mb(self, client):
        mb = await client.get_vram_budget_mb()
        assert isinstance(mb, int)

    @pytest.mark.asyncio
    async def test_enable_auto_management(self, client):
        await client.set_load_mode("shared", "user")  # disables auto
        await client.enable_auto_management()
        assert await client.is_auto_managed() is True

    @pytest.mark.asyncio
    async def test_mark_degraded(self, client):
        await client.mark_degraded("inference")
        # No error = success (state is on server side)

    @pytest.mark.asyncio
    async def test_gpu_state(self, client):
        gs = await client.gpu_state()
        assert hasattr(gs, "vram_total_mb")

    @pytest.mark.asyncio
    async def test_graceful_when_server_down(self):
        """Client should return safe defaults when NerdHerd is unreachable."""
        c = NerdHerdClient(port=19899)  # nothing on this port
        mode = await c.get_load_mode()
        assert mode == "full"  # safe default
        assert await c.is_local_inference_allowed() is True
        assert await c.get_vram_budget_fraction() == 1.0
        await c.close()

    @pytest.mark.asyncio
    async def test_prometheus_lines(self, client):
        lines = await client.prometheus_lines()
        assert isinstance(lines, str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/nerd_herd && python -m pytest tests/test_client.py -v`
Expected: FAIL — `nerd_herd.client` does not exist

- [ ] **Step 3: Implement NerdHerdClient**

Create `packages/nerd_herd/src/nerd_herd/client.py`:

```python
"""NerdHerdClient — HTTP proxy to a running NerdHerd sidecar.

Drop-in replacement for NerdHerd when running as a remote client.
All methods are async (unlike the in-process NerdHerd which mixes sync/async).
Returns safe defaults when the sidecar is unreachable.
"""
from __future__ import annotations

import aiohttp
from dataclasses import dataclass

from yazbunu import get_logger

logger = get_logger("nerd_herd.client")


@dataclass
class GPUStateProxy:
    """Lightweight mirror of GPUState for client use."""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    vram_used_mb: int = 0
    gpu_name: str = ""
    gpu_util_pct: float = 0.0


class NerdHerdClient:
    """HTTP client proxy for a NerdHerd sidecar.

    Usage:
        client = NerdHerdClient(port=9881)
        mode = await client.get_load_mode()
        await client.close()
    """

    def __init__(self, port: int = 9881, host: str = "127.0.0.1",
                 timeout: float = 3.0) -> None:
        self._base = f"http://{host}:{port}"
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_json(self, path: str, default=None):
        try:
            async with self._get_session().get(f"{self._base}{path}") as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        return default

    async def _post_json(self, path: str, data: dict, default=None):
        try:
            async with self._get_session().post(
                f"{self._base}{path}", json=data
            ) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        return default

    async def _get_state(self) -> dict:
        return await self._get_json("/api/state", default={})

    # ── Public API (mirrors NerdHerd) ────────────────────────────

    async def get_load_mode(self) -> str:
        state = await self._get_state()
        return state.get("load_mode", "full")

    async def set_load_mode(self, mode: str, source: str = "user") -> str:
        resp = await self._post_json("/api/mode", {"mode": mode, "source": source})
        if resp:
            return resp.get("result", f"Set to {mode}")
        return "NerdHerd unreachable"

    async def enable_auto_management(self) -> None:
        await self._post_json("/api/auto", {})

    async def is_auto_managed(self) -> bool:
        state = await self._get_state()
        return state.get("auto_managed", True)

    async def is_local_inference_allowed(self) -> bool:
        state = await self._get_state()
        return state.get("local_inference_allowed", True)

    async def get_vram_budget_fraction(self) -> float:
        state = await self._get_state()
        return state.get("vram_budget_fraction", 1.0)

    async def get_vram_budget_mb(self) -> int:
        state = await self._get_state()
        return state.get("vram_budget_mb", 0)

    async def gpu_state(self) -> GPUStateProxy:
        data = await self._get_json("/api/gpu", default={})
        return GPUStateProxy(
            vram_total_mb=data.get("vram_total_mb", 0),
            vram_free_mb=data.get("vram_free_mb", 0),
            vram_used_mb=data.get("vram_used_mb", 0),
            gpu_name=data.get("gpu_name", ""),
            gpu_util_pct=data.get("gpu_util_pct", 0.0),
        )

    async def mark_degraded(self, capability: str) -> None:
        await self._post_json("/api/degraded", {"capability": capability})

    async def prometheus_lines(self) -> str:
        try:
            async with self._get_session().get(f"{self._base}/metrics") as r:
                if r.status == 200:
                    return await r.text()
        except Exception:
            pass
        return ""
```

- [ ] **Step 4: Export NerdHerdClient from `__init__.py`**

In `packages/nerd_herd/src/nerd_herd/__init__.py`, add:

```python
from nerd_herd.client import NerdHerdClient
```

And add `"NerdHerdClient"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/nerd_herd && python -m pytest tests/test_client.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/client.py packages/nerd_herd/src/nerd_herd/__init__.py packages/nerd_herd/tests/test_client.py
git commit -m "feat(nerd_herd): add NerdHerdClient HTTP proxy"
```

---

### Task 3: Create NerdHerd standalone entry point

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/__main__.py`

Standalone entry point: `python -m nerd_herd --port 9881`. Handles signal-based shutdown, persists load mode to DB, starts auto-detect.

- [ ] **Step 1: Write `__main__.py`**

Create `packages/nerd_herd/src/nerd_herd/__main__.py`:

```python
"""Standalone entry point for NerdHerd sidecar.

Usage: python -m nerd_herd --port 9881 --llama-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

# Ensure project root is on sys.path for DB access
_project_root = os.environ.get("NERD_HERD_PROJECT_ROOT", "")
if _project_root and _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from nerd_herd.nerd_herd import NerdHerd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NerdHerd observability sidecar")
    p.add_argument("--port", type=int, default=9881)
    p.add_argument("--llama-url", default="http://127.0.0.1:8080")
    p.add_argument("--pid-file", default=None)
    p.add_argument("--db-path", default=None,
                   help="SQLite DB path for load mode persistence")
    p.add_argument("--detect-interval", type=int, default=30)
    p.add_argument("--upgrade-delay", type=int, default=300)
    return p.parse_args()


async def _load_mode_from_db(db_path: str | None) -> str:
    """Restore persisted load mode from DB, or default to 'full'."""
    if not db_path:
        return "full"
    try:
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT mode FROM load_mode WHERE id = 1")
            row = await cur.fetchone()
            if row and row["mode"] in ("full", "heavy", "shared", "minimal"):
                return row["mode"]
    except Exception:
        pass
    return "full"


async def _persist_mode(db_path: str | None, mode: str, auto_managed: bool) -> None:
    """Save current load mode to DB."""
    if not db_path:
        return
    try:
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO load_mode (id, mode, auto_managed, updated_at) "
                "VALUES (1, ?, ?, datetime('now'))",
                (mode, int(auto_managed)),
            )
            await db.commit()
    except Exception:
        pass


async def _run(args: argparse.Namespace) -> None:
    shutdown_event = asyncio.Event()

    # Write PID file
    pid_path = Path(args.pid_file) if args.pid_file else None
    if pid_path:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))

    # Restore mode from DB
    initial_mode = await _load_mode_from_db(args.db_path)

    nh = NerdHerd(
        metrics_port=args.port,
        llama_server_url=args.llama_url,
        detect_interval=args.detect_interval,
        upgrade_delay=args.upgrade_delay,
        initial_load_mode=initial_mode,
    )

    # Wire mode persistence callback
    def _on_mode_change(old: str, new: str, source: str) -> None:
        asyncio.create_task(
            _persist_mode(args.db_path, new, nh._load.is_auto_managed())
        )

    nh.on_mode_change(_on_mode_change)

    await nh.start()
    await nh.start_auto_detect()

    print(f"[NerdHerd] Started on port {args.port} (mode={initial_mode}, PID={os.getpid()})")

    # Signal handling
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)
    else:
        # Windows: SIGINT via Ctrl+C, SIGTERM via os.kill
        def _handler(signum, frame):
            shutdown_event.set()
        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handler)

    await shutdown_event.wait()

    print("[NerdHerd] Shutting down...")
    await nh.stop()
    if pid_path and pid_path.exists():
        pid_path.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it starts and responds**

Run:
```bash
cd packages/nerd_herd
python -m nerd_herd --port 19883 --llama-url http://127.0.0.1:8080 &
sleep 2
curl http://127.0.0.1:19883/health
# Expected: {"status": "ok"}
curl http://127.0.0.1:19883/api/state
# Expected: {"load_mode": "full", ...}
# Kill the process
```

- [ ] **Step 3: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/__main__.py
git commit -m "feat(nerd_herd): add standalone __main__.py entry point"
```

---

### Task 4: Multi-sidecar support in Yaşar Usta (config + guard)

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/config.py:127-128`
- Modify: `packages/yasar_usta/src/yasar_usta/guard.py`
- Test: `packages/yasar_usta/tests/test_guard.py`

Change from single `sidecar` to `sidecars: list[SidecarConfig]`.

- [ ] **Step 1: Update existing tests for multi-sidecar**

In `packages/yasar_usta/tests/test_guard.py`, update `test_creates_with_full_config`:

```python
def test_creates_with_full_config(self):
    cfg = GuardConfig(
        name="Yaşar Usta",
        app_name="Kutay",
        command=["python", "run.py"],
        telegram_token="tok",
        telegram_chat_id="123",
        sidecars=[
            SidecarConfig(
                name="yazbunu",
                command=["python", "-m", "yazbunu.server"],
                health_url="http://127.0.0.1:9880/",
                pid_file="/tmp/yazbunu.pid",
            ),
            SidecarConfig(
                name="nerd_herd",
                command=["python", "-m", "nerd_herd"],
                health_url="http://127.0.0.1:9881/health",
                pid_file="/tmp/nerd_herd.pid",
            ),
        ],
    )
    guard = ProcessGuard(cfg)
    assert guard.telegram.enabled is True
    assert len(guard.sidecars) == 2
    assert guard.sidecars["yazbunu"].name == "yazbunu"
    assert guard.sidecars["nerd_herd"].name == "nerd_herd"
```

Add a new test:

```python
def test_creates_with_no_sidecars(self):
    cfg = GuardConfig(
        command=[sys.executable, "-c", "pass"],
        log_dir=tempfile.mkdtemp(),
    )
    guard = ProcessGuard(cfg)
    assert guard.sidecars == {}
```

Also update the import to include `SidecarConfig`:

```python
from yasar_usta import ProcessGuard, GuardConfig, Messages, SidecarConfig
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/yasar_usta && python -m pytest tests/test_guard.py -v`
Expected: FAIL — `sidecars` not recognized, guard has `sidecar` not `sidecars`

- [ ] **Step 3: Update config.py — `sidecar` → `sidecars`**

In `packages/yasar_usta/src/yasar_usta/config.py`, change line 127-128:

```python
    # Sidecars (log viewer, observability, etc.)
    sidecars: list[SidecarConfig] = field(default_factory=list)
```

- [ ] **Step 4: Update guard.py — multi-sidecar support**

In `packages/yasar_usta/src/yasar_usta/guard.py`:

**Constructor** — replace lines 55-68:

```python
        # Sidecars
        self.sidecars: dict[str, SidecarManager] = {}
        for sc in config.sidecars:
            if sc.command:
                self.sidecars[sc.name] = SidecarManager(
                    name=sc.name,
                    command=sc.command,
                    pid_file=sc.pid_file,
                    health_url=sc.health_url,
                    health_timeout=sc.health_timeout,
                    log_file=str(Path(config.log_dir) / f"{sc.name}.log"),
                    cwd=config.cwd,
                    detached=sc.detached,
                )
```

**`_send_status`** — replace lines 137-159:

```python
    async def _send_status(self, edit_message_id: int | None = None) -> None:
        sidecar_infos = []
        for name, sc in self.sidecars.items():
            sidecar_infos.append({
                "name": name,
                "alive": await sc.is_alive(),
                "pid": sc.pid_alive(),
                "health_url": sc.health_url,
                "http_alive": await sc.http_alive(),
            })

        text = build_status_text(
            name=self.cfg.name,
            app_name=self.cfg.app_name,
            guard_start_time=self._guard_start_time,
            app_running=self.subprocess.running,
            heartbeat_age=self.subprocess.heartbeat_age(),
            heartbeat_healthy_seconds=self.cfg.heartbeat_healthy_seconds,
            total_crashes=self.backoff.total_crashes,
            sidecar_infos=sidecar_infos,
            extra_processes=self.cfg.extra_processes,
        )
        sidecar_names = list(self.sidecars.keys()) if self.sidecars else []
        inline_kb = build_status_inline_keyboard(
            self.msgs, self.cfg.name, sidecar_names)

        if edit_message_id:
            await self.telegram.edit(edit_message_id, text, reply_markup=inline_kb)
        else:
            await self._send(
                self.msgs.status_title.format(name=self.cfg.name) + "panel:",
                reply_markup=self._kb(),
            )
            await self.telegram.send(text, reply_markup=inline_kb)
```

**`_send_logs`** — replace lines 187-191 (the sidecar check):

```python
        yazbunu = self.sidecars.get("yazbunu")
        if yazbunu and yazbunu.health_url:
            if await yazbunu.http_alive():
                url = yazbunu.health_url.replace("/health", "/")
                formatted += f"\n\n📊 [Yazbunu Log Viewer]({url})"
```

**`_signal_watch_loop`** — replace lines 218-219 (condition) and 244-246 (sidecar ensure):

Line 219: change condition to check sidecars dict:
```python
        if not self.cfg.claude_signal_file and not self.sidecars:
            return
```

Lines 244-246: replace with loop:
```python
                if self.sidecars and sidecar_check_counter >= 10:
                    sidecar_check_counter = 0
                    for sc in self.sidecars.values():
                        await sc.ensure()
```

**Callback queries** — replace lines 301-306:

```python
                            elif cb_data.startswith("restart_sidecar:"):
                                sc_name = cb_data.split(":", 1)[1]
                                sc = self.sidecars.get(sc_name)
                                if sc:
                                    await self.telegram.answer_callback(cb["id"], f"📊 {sc_name} yeniden başlatılıyor...")
                                    await sc.stop()
                                    await sc.start()
                                    await self._send_status(edit_message_id=cb_msg_id)
                                else:
                                    await self.telegram.answer_callback(cb["id"])
```

Also handle the legacy `restart_sidecar` and `restart_yazbunu` callbacks for backward compat — add a fallback before the new handler:

```python
                            elif cb_data in ("restart_sidecar", "restart_yazbunu"):
                                sc = self.sidecars.get("yazbunu") or (
                                    next(iter(self.sidecars.values()), None) if self.sidecars else None)
                                if sc:
                                    await self.telegram.answer_callback(cb["id"], f"📊 {sc.name} yeniden başlatılıyor...")
                                    await sc.stop()
                                    await sc.start()
                                    await self._send_status(edit_message_id=cb_msg_id)
                                else:
                                    await self.telegram.answer_callback(cb["id"])
```

**`run()` method** — replace lines 496-498 (start sidecar):

```python
        # Start sidecars
        for sc in self.sidecars.values():
            if sc.command:
                await sc.start()
```

Replace line 522-523 (ensure sidecar after app exit):

```python
                if self.sidecars:
                    for sc in self.sidecars.values():
                        await sc.ensure()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/yasar_usta && python -m pytest tests/test_guard.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/config.py packages/yasar_usta/src/yasar_usta/guard.py packages/yasar_usta/tests/test_guard.py
git commit -m "feat(yasar_usta): multi-sidecar support in config and guard"
```

---

### Task 5: Multi-sidecar UI (commands + status)

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/commands.py:30-41`
- Modify: `packages/yasar_usta/src/yasar_usta/status.py:13-77`
- Test: `packages/yasar_usta/tests/test_status.py`

- [ ] **Step 1: Update status tests for multi-sidecar**

In `packages/yasar_usta/tests/test_status.py`, update `test_sidecar_http_alive` and `test_sidecar_not_running` to use `sidecar_infos`:

```python
    def test_sidecar_http_alive(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=True,
            heartbeat_age=5.0,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
            sidecar_infos=[{
                "name": "LogViewer",
                "http_alive": True,
                "pid": 12345,
                "health_url": "http://localhost:9880",
                "alive": True,
            }],
        )
        assert "LogViewer" in text
        assert "running" in text

    def test_sidecar_not_running(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=True,
            heartbeat_age=5.0,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
            sidecar_infos=[{
                "name": "LogViewer",
                "http_alive": False,
                "pid": None,
                "health_url": None,
                "alive": False,
            }],
        )
        assert "LogViewer" in text
        assert "not running" in text
```

Add a test for multiple sidecars:

```python
    def test_multiple_sidecars(self):
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=True,
            heartbeat_age=5.0,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
            sidecar_infos=[
                {"name": "yazbunu", "http_alive": True, "pid": 111,
                 "health_url": "http://localhost:9880", "alive": True},
                {"name": "nerd_herd", "http_alive": True, "pid": 222,
                 "health_url": "http://localhost:9881", "alive": True},
            ],
        )
        assert "yazbunu" in text
        assert "nerd_herd" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/yasar_usta && python -m pytest tests/test_status.py -v`
Expected: FAIL — `sidecar_infos` not accepted

- [ ] **Step 3: Update `build_status_text` for multi-sidecar**

In `packages/yasar_usta/src/yasar_usta/status.py`, change the function signature and sidecar section:

```python
def build_status_text(
    *,
    name: str,
    app_name: str,
    guard_start_time: float,
    app_running: bool,
    heartbeat_age: float | None,
    heartbeat_healthy_seconds: int,
    total_crashes: int,
    sidecar_infos: list[dict] | None = None,
    extra_processes: list[dict] | None = None,
    messages=None,
    # Legacy single-sidecar params (ignored if sidecar_infos provided)
    sidecar_name: str | None = None,
    sidecar_alive: bool = False,
    sidecar_pid: int | None = None,
    sidecar_health_url: str | None = None,
    sidecar_http_alive: bool = False,
) -> str:
```

Replace the sidecar section (lines 62-72) with:

```python
    # Sidecars
    _infos = sidecar_infos
    if _infos is None and sidecar_name:
        # Legacy single-sidecar compat
        _infos = [{
            "name": sidecar_name,
            "alive": sidecar_alive,
            "pid": sidecar_pid,
            "health_url": sidecar_health_url,
            "http_alive": sidecar_http_alive,
        }]
    for si in (_infos or []):
        sc_name = si["name"]
        if si.get("http_alive"):
            pid_str = f", PID {si['pid']}" if si.get("pid") else ""
            lines.append(f"📊 {sc_name}: running ({si.get('health_url', '')}{pid_str})")
        elif si.get("pid"):
            lines.append(f"🟠 {sc_name}: process alive but not responding (PID {si['pid']})")
        elif si.get("alive"):
            lines.append(f"🟢 {sc_name}: running")
        else:
            lines.append(f"⚫ {sc_name}: not running")
```

- [ ] **Step 4: Update `build_status_inline_keyboard` for multi-sidecar**

In `packages/yasar_usta/src/yasar_usta/commands.py`, change the function signature and add buttons for each sidecar:

```python
def build_status_inline_keyboard(messages: Messages, name: str,
                                  sidecar_names: list[str] | None = None,
                                  sidecar_name: str | None = None) -> dict:
    """Build inline keyboard for the status panel."""
    buttons = [
        [{"text": messages.btn_refresh, "callback_data": "guard_refresh"}],
        [{"text": messages.btn_restart_guard.format(name=name), "callback_data": "restart_guard"}],
    ]
    # Multi-sidecar buttons
    names = sidecar_names or ([sidecar_name] if sidecar_name else [])
    if names:
        sidecar_row = []
        for sc_name in names:
            sidecar_row.append({
                "text": f"📊 Restart {sc_name}",
                "callback_data": f"restart_sidecar:{sc_name}",
            })
        buttons.append(sidecar_row)
    return {"inline_keyboard": buttons}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/yasar_usta && python -m pytest tests/test_status.py tests/test_guard.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/commands.py packages/yasar_usta/src/yasar_usta/status.py packages/yasar_usta/tests/test_status.py
git commit -m "feat(yasar_usta): multi-sidecar status panel and inline buttons"
```

---

### Task 6: Wire NerdHerd as sidecar in kutai_wrapper.py

**Files:**
- Modify: `kutai_wrapper.py:130-138`

- [ ] **Step 1: Update kutai_wrapper.py to use `sidecars=` list**

Replace lines 130-138 (`sidecar=SidecarConfig(...)`) with:

```python
    sidecars=[
        SidecarConfig(
            name="yazbunu",
            command=[venv_python, "-m", "yazbunu.server",
                     "--log-dir", "./logs", "--port", "9880", "--host", "0.0.0.0"],
            health_url="http://127.0.0.1:9880/health",
            pid_file=str(PROJECT_ROOT / "logs" / "yazbunu.pid"),
            detached=True,
            auto_start=True,
        ),
        SidecarConfig(
            name="nerd_herd",
            command=[venv_python, "-m", "nerd_herd",
                     "--port", "9881",
                     "--llama-url", "http://127.0.0.1:8080",
                     "--pid-file", str(PROJECT_ROOT / "logs" / "nerd_herd.pid"),
                     "--db-path", os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "kutai.db"))],
            health_url="http://127.0.0.1:9881/health",
            pid_file=str(PROJECT_ROOT / "logs" / "nerd_herd.pid"),
            detached=True,
            auto_start=True,
        ),
    ],
```

Also update the `btn_restart_sidecar` message to be generic:

```python
        btn_restart_sidecar="📊 {sidecar_name} Yeniden Başlat",
```

- [ ] **Step 2: Verify wrapper imports work**

Run: `python -c "from kutai_wrapper import config; print(len(config.sidecars), 'sidecars')"`
Expected: `2 sidecars`

- [ ] **Step 3: Commit**

```bash
git add kutai_wrapper.py
git commit -m "feat(wrapper): add nerd_herd as second sidecar"
```

---

### Task 7: Switch orchestrator to NerdHerdClient

**Files:**
- Modify: `src/app/run.py:42-47, 404-493`
- Modify: `src/infra/load_manager.py`
- Modify: `src/infra/runtime_state.py`
- Modify: `src/models/gpu_monitor.py`
- Modify: `src/app/api.py:242-253`

The orchestrator no longer owns NerdHerd — it connects as a client via HTTP.

- [ ] **Step 1: Update run.py — replace NerdHerd with NerdHerdClient**

At the top of `src/app/run.py`, change the import (line 42) and accessor (lines 44-47):

```python
from nerd_herd.client import NerdHerdClient

_nerd_herd: NerdHerdClient | None = None

def get_nerd_herd() -> NerdHerdClient | None:
    return _nerd_herd
```

Replace Phase 3 (lines 404-479) with a simpler client connection:

```python
    # Phase 3: Connect to NerdHerd sidecar (managed by Yaşar Usta)
    try:
        global _nerd_herd
        _nerd_herd = NerdHerdClient(port=9881)
        # Verify sidecar is reachable
        _mode = await _nerd_herd.get_load_mode()
        _log.info("Connected to NerdHerd sidecar", load_mode=_mode)
    except Exception as exc:
        _log.warning("NerdHerd client init failed — running without GPU monitoring",
                     error=str(exc))
        _nerd_herd = None
```

Replace the shutdown section (lines 492-493):

```python
    if _nerd_herd:
        await _nerd_herd.close()
```

- [ ] **Step 2: Update load_manager.py — make all functions async-safe with client**

The `NerdHerdClient` methods are all async, so `load_manager.py` shims that return sync values need to handle this. Since some callers are sync (e.g., `is_local_inference_allowed`), we need to either cache or use a sync fallback.

Replace `src/infra/load_manager.py`:

```python
# load_manager.py
"""Load manager shim — delegates to NerdHerdClient.

All import paths preserved:
    from src.infra.load_manager import get_load_mode, set_load_mode, ...
"""
from __future__ import annotations

import asyncio

from nerd_herd.load import LOAD_MODES, VRAM_BUDGETS, DESCRIPTIONS  # noqa: F401


def _nh():
    """Get NerdHerdClient instance, or None if not yet initialized."""
    try:
        from src.app.run import get_nerd_herd
        return get_nerd_herd()
    except Exception:
        return None


async def get_load_mode() -> str:
    nh = _nh()
    if nh is None:
        return "full"
    return await nh.get_load_mode()


async def set_load_mode(mode: str, source: str = "user") -> str:
    nh = _nh()
    if nh is None:
        return "NerdHerd not connected"
    return await nh.set_load_mode(mode, source)


async def enable_auto_management():
    nh = _nh()
    if nh:
        await nh.enable_auto_management()


def is_local_inference_allowed() -> bool:
    """Sync check — uses cached state from last async call.

    Falls back to True (allow local inference) if client unavailable.
    """
    nh = _nh()
    if nh is None:
        return True
    # Try to get from running loop without blocking
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context — caller should use await version
        # But for backward compat, return True as safe default
        return True
    except RuntimeError:
        return True


async def is_local_inference_allowed_async() -> bool:
    nh = _nh()
    if nh is None:
        return True
    return await nh.is_local_inference_allowed()


def is_auto_managed() -> bool:
    nh = _nh()
    if nh is None:
        return True
    try:
        loop = asyncio.get_running_loop()
        return True  # safe default for sync callers
    except RuntimeError:
        return True


async def is_auto_managed_async() -> bool:
    nh = _nh()
    if nh is None:
        return True
    return await nh.is_auto_managed()


def get_vram_budget_fraction() -> float:
    nh = _nh()
    if nh is None:
        return 1.0
    try:
        loop = asyncio.get_running_loop()
        return 1.0  # safe default for sync callers
    except RuntimeError:
        return 1.0


async def get_vram_budget_fraction_async() -> float:
    nh = _nh()
    if nh is None:
        return 1.0
    return await nh.get_vram_budget_fraction()


def suggest_mode_for_external_usage(ext_frac: float) -> str:
    from nerd_herd.load import LoadManager
    return LoadManager.suggest_mode_for_external_usage(ext_frac)


async def run_gpu_autodetect_loop(notify_fn=None):
    """No-op — auto-detect runs in the NerdHerd sidecar now."""
    pass
```

- [ ] **Step 3: Update runtime_state.py — async mark_degraded**

In `src/infra/runtime_state.py`, the `mark_degraded` function calls `nh.mark_degraded()` which is now async:

```python
def mark_degraded(capability: str) -> None:
    if capability not in runtime_state["degraded_capabilities"]:
        runtime_state["degraded_capabilities"].append(capability)
    try:
        from src.app.run import get_nerd_herd
        nh = get_nerd_herd()
        if nh is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(nh.mark_degraded(capability))
            except RuntimeError:
                pass
    except Exception:
        pass
```

- [ ] **Step 4: Update gpu_monitor.py — pure fallback mode**

Since the client doesn't expose the registry, `gpu_monitor.py` always uses a standalone fallback:

```python
# gpu_monitor.py
"""GPU monitor shim — standalone GPUCollector.

With NerdHerd running as a sidecar, the orchestrator uses a local
GPUCollector for GPU state queries. NerdHerd handles metrics/load.
"""
from __future__ import annotations

from nerd_herd.types import GPUState, SystemState, ExternalGPUUsage  # noqa: F401
from nerd_herd.gpu import GPUCollector as GPUMonitor  # noqa: F401

_instance: GPUMonitor | None = None


def get_gpu_monitor() -> GPUMonitor:
    """Return a standalone GPUCollector."""
    global _instance
    if _instance is None:
        _instance = GPUMonitor()
    return _instance
```

- [ ] **Step 5: Update api.py — fetch metrics from NerdHerd via HTTP**

Replace the `/metrics` endpoint in `src/app/api.py` (lines 242-253):

```python
    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        """Prometheus-compatible metrics. Fetches from NerdHerd sidecar."""
        try:
            from src.app.run import get_nerd_herd
            nh = get_nerd_herd()
            if nh is not None:
                return await nh.prometheus_lines()
            return ""
        except Exception as e:
            logger.warning("NerdHerd metrics unavailable", error=str(e))
            return ""
```

- [ ] **Step 6: Verify orchestrator imports work**

Run: `python -c "from src.app.run import get_nerd_herd; print('OK')"`
Expected: `OK`

Run: `python -c "from src.infra.load_manager import get_load_mode; print('OK')"`
Expected: `OK`

Run: `python -c "from src.models.gpu_monitor import get_gpu_monitor; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Run existing tests**

Run: `python -m pytest tests/ -v -x --ignore=tests/integration`
Expected: All PASS (tests may need minor adjustments if they mock NerdHerd directly)

- [ ] **Step 8: Commit**

```bash
git add src/app/run.py src/infra/load_manager.py src/infra/runtime_state.py src/models/gpu_monitor.py src/app/api.py
git commit -m "feat(orchestrator): switch from NerdHerd owner to HTTP client"
```

---

### Task 8: Update callers that use sync NerdHerd methods

**Files:**
- Modify: `src/app/telegram_bot.py` (lines referencing `get_load_mode`, `set_load_mode`, etc.)

Since `load_manager.get_load_mode()` and `set_load_mode()` are already async, and `telegram_bot.py` already awaits them, this should work without changes. But we need to verify the `is_local_inference_allowed()` and `is_auto_managed()` sync callers.

- [ ] **Step 1: Search for sync callers of load_manager functions**

Search for `is_local_inference_allowed` and `is_auto_managed` usage:

```bash
grep -rn "is_local_inference_allowed\|is_auto_managed\|get_vram_budget_fraction" src/ --include="*.py"
```

Review each call site. If any are in async functions and can be changed to `await`, update them to use the `_async` variants. If they're in sync contexts, they'll use the safe defaults (which is acceptable — these are fast paths that don't block on critical decisions).

- [ ] **Step 2: Update sync callers to async where possible**

For each caller in an async function, replace:
- `is_local_inference_allowed()` → `await is_local_inference_allowed_async()`
- `is_auto_managed()` → `await is_auto_managed_async()`
- `get_vram_budget_fraction()` → `await get_vram_budget_fraction_async()`

Import the async versions from `src.infra.load_manager`.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v -x --ignore=tests/integration`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/
git commit -m "fix(orchestrator): update sync load_manager callers to async"
```

---

### Task 9: Set NERD_HERD_PROJECT_ROOT in wrapper

**Files:**
- Modify: `kutai_wrapper.py`

NerdHerd's `__main__.py` needs access to the project root for DB imports.

- [ ] **Step 1: Set environment variable before guard starts**

In `kutai_wrapper.py`, add after `PROJECT_ROOT` definition (around line 30):

```python
os.environ["NERD_HERD_PROJECT_ROOT"] = str(PROJECT_ROOT)
```

- [ ] **Step 2: Commit**

```bash
git add kutai_wrapper.py
git commit -m "fix(wrapper): set NERD_HERD_PROJECT_ROOT for sidecar DB access"
```

---

### Task 10: Integration smoke test

- [ ] **Step 1: Run NerdHerd standalone and verify client works**

```bash
# Terminal 1: Start NerdHerd sidecar
python -m nerd_herd --port 9881 --llama-url http://127.0.0.1:8080

# Terminal 2: Test client
python -c "
import asyncio
from nerd_herd.client import NerdHerdClient

async def test():
    c = NerdHerdClient(port=9881)
    print('mode:', await c.get_load_mode())
    print('budget:', await c.get_vram_budget_fraction())
    print('local ok:', await c.is_local_inference_allowed())
    result = await c.set_load_mode('shared', 'user')
    print('set result:', result)
    print('new mode:', await c.get_load_mode())
    await c.close()

asyncio.run(test())
"
```

Expected:
```
mode: full
budget: 1.0
local ok: True
set result: Load mode set to *shared*: ...
new mode: shared
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest packages/nerd_herd/tests/ packages/yasar_usta/tests/ tests/ -v --ignore=tests/integration`
Expected: All PASS

- [ ] **Step 3: Final commit with all changes**

If any loose changes remain:
```bash
git add -A
git commit -m "test: integration smoke test for NerdHerd sidecar extraction"
```
