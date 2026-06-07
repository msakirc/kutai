# Image Generation — Plan 2 (v2): Local `clair_obscur` + GPU handover

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill in the local half of the image-generation lane on top of Plan 1 v3's cloud spine. Adds a local image-server wrapper (`clair_obscur`) parallel to dallama, wires it through paintress as the `local_server` provider, registers a local SDXL entry in hoca's image catalog, replaces Plan 1's eviction-cost stub with the real formula reading nerd_herd, performs the GPU handover (`get_local_manager().shutdown()` → poll free VRAM → `clair_obscur.start()` → record-swap) **inside `husam._run_image`, wrapped in `heartbeat.keepalive()` so the 30-60s+ handover never trips the 300s no-progress watchdog**, and adds beckman warm-batch awareness that drives clair_obscur's idle backstop (not a direct hard-stop — the backstop **is the stop** under normal lane switches).

**Architecture:** `clair_obscur` is the image-world `dallama` — a thin async process wrapper around ComfyUI (default) or AUTOMATIC1111 (env flag), holding a PID-lock at `image_server.lock`, reconciling its own orphan on boot using `psutil.Process(pid).children(recursive=True)` against the **PID written to its own lock** (NEVER process-name kills, NEVER llama-server, per CLAUDE.md). The local handover is mechanical follow-through inside Plan 1 v3's full-telemetry envelope **in `husam._run_image`** (there is no `dispatcher.dispatch()`; image execution lives in husam): hoca picks `clair_obscur/sdxl-turbo`, husam unloads dallama → polls free VRAM → starts clair_obscur → records ONE swap, all inside `heartbeat.keepalive()` so the watchdog stays satisfied through the cold-start window. After each image task, beckman calls `clair_obscur.record_release_hint()` on lane switch; the backstop times the actual `stop()` after `idle_release_seconds` so a back-to-back image batch reuses the warm server without a restart.

**Tech Stack:** Python 3.10, async/await, httpx (HTTP to ComfyUI/A1111), aiosqlite, psutil (transitive via nerd_herd; declared explicitly in clair_obscur's pyproject). Package layout mirrors `packages/dallama/` src-layout.

**Scope boundary (in this plan):** `clair_obscur` package · paintress `local_server` provider · hoca local SDXL catalog entry · real eviction-cost in `image_select.py` (reads nerd_herd via `refresh_snapshot()`) · VRAM-fit eligibility gate · nerd_herd `image_server_resident` + `image_server_vram_mb` fields + `record_image_server_state` · husam `_run_image` local-image handover wrapped in `keepalive()` · beckman warm-batch hook that drives `record_release_hint()` (NOT direct stop) · e2e host-path test with mocked ComfyUI/A1111.

**NOT in this plan (Plan 3):** i2p prototype `swap_placeholder_images` mechanical · prompt-writing coulson task + templates · asset serving wired into the web-preview host · ComfyUI/A1111 actually installed on the dev box · live GPU process in CI.

**Dependency: Plan 1 v3 MUST be merged first.** Plan 2 extends `image_select.py`, `paintress/__init__.py`, **`husam.worker._run_image`** (NOT a dispatcher method — `dispatcher.dispatch()`/`_dispatch_image` do not exist), `image_providers.py`, and beckman's admission shape — all of which Plan 1 v3 introduces. Recon confirmed `_select_for_admission` and `husam._run_image` don't exist on `main` today; Plan 1 v3 must land before Plan 2 starts.

**Inviolable rules (CLAUDE.md):**
- `clair_obscur.start()` / `.stop()` / boot-orphan-reconcile MUST target **only** the image-server backend's PID via the PID written to its own `image_server.lock`. Verified by `psutil.Process(pid).cmdline()` matching the expected backend launcher (ComfyUI's `main.py` / A1111's `webui.py` / `launch.py`). MUST NEVER call `taskkill /F /IM llama-server.exe`, `psutil.process_iter(["name"])` style sweeps, or anything that could hit a co-tenant.
- Plan 2 MUST NOT modify Yaşar Usta or the wrapper.

---

## Audit findings this rewrite addresses

Prior Plan 2 had: (1) **no `heartbeat.keepalive()` around the long unload+poll+start sequence** — the 300s watchdog would kill mid-handover and orphan ~7GB of VRAM, (2) a **dead idle-release backstop** — `record_release_hint()` had zero production callers because beckman called `clair_obscur.stop()` directly, (3) a **hand-rolled `_peek_next_admittable` SQL** that didn't mirror `next_task()`'s real eligibility (lane cap, pool pressure, dependency check), (4) **`nerd_herd.snapshot()` called as a module-level function that doesn't exist** (recon: callers use `refresh_snapshot()` or the singleton), (5) **psutil not declared** in clair_obscur's pyproject. v2 fixes each at the structural level.

Recon confirmed (verbatim file:line):
- `nerd_herd.snapshot()` is NOT a module-level function. `NerdHerd.snapshot()` is the instance method at `packages/nerd_herd/src/nerd_herd/nerd_herd.py:140`; module-level access is `nerd_herd.refresh_snapshot()` (used by `general_beckman/__init__.py:330`).
- `record_swap(model_name: str = "")` is positional-optional at `nerd_herd.py:114`.
- `keepalive(interval: float = 30.0)` at `src/core/heartbeat.py:75-111` is reentrant + contextvar-safe — safe to wrap an outer span containing both `get_local_manager().shutdown()` and `clair_obscur.start()`.
- `next_task()` admission at `general_beckman/__init__.py:262-459` has NO extracted peek helper; `_queue.pick_ready_top_k(k, lane)` returns top-K candidates, and `_claim_task(id)` does the claim.
- Safe psutil PID-kill template at `packages/nerd_herd/src/nerd_herd/platform.py:1-21`: `psutil.Process(pid).children(recursive=True)` with try/except.
- i2p step convention for mechanicals (confirmed in JSON): `"agent": "mechanical"`, `"executor": "mechanical"`, verb name in `payload.action`. (Affects Plan 3, not Plan 2 — but worth noting since Plan 2's e2e test forces a local pick.)

---

## File structure

**New package:**
- `packages/clair_obscur/pyproject.toml` — declares `httpx` AND `psutil` as dependencies.
- `packages/clair_obscur/src/clair_obscur/__init__.py` — module-level API.
- `packages/clair_obscur/src/clair_obscur/config.py` — `ClairObscurConfig` dataclass.
- `packages/clair_obscur/src/clair_obscur/server.py` — `ImageServer` class with lifecycle + PID-lock + orphan-reconcile + idle backstop.
- `packages/clair_obscur/tests/test_config.py`, `test_server_lifecycle.py`, `test_orphan_reconcile.py`, `test_idle_backstop.py`, `test_nerd_herd_wiring.py`.

**New file in paintress:**
- `packages/paintress/src/paintress/providers/local_server.py`
- `packages/paintress/tests/test_local_server.py`

**Modified (single-anchor append/replace each):**
- `packages/fatih_hoca/src/fatih_hoca/image_providers.py` — APPEND one `ImageModelInfo` for `clair_obscur/sdxl-turbo`.
- `packages/fatih_hoca/src/fatih_hoca/image_select.py` — replace `_eviction_cost` body (real formula via `refresh_snapshot()`); extend `_provider_available` (clair_obscur arm); add VRAM-fit eligibility gate.
- `packages/nerd_herd/src/nerd_herd/types.py` — add `image_server_resident: bool = False` + `image_server_vram_mb: int = 0` to `SystemSnapshot`.
- `packages/nerd_herd/src/nerd_herd/__init__.py` — add module-level `record_image_server_state(resident, vram_mb)`.
- `packages/nerd_herd/src/nerd_herd/nerd_herd.py` — add `push_image_server_state` method; init two new attrs; extend `snapshot()` builder.
- `packages/husam/src/husam/worker.py` — extend `_run_image` with `if pick.model.is_local:` handover branch **inside the existing `heartbeat.keepalive()`**.
- `packages/paintress/src/paintress/__init__.py` — APPEND `"clair_obscur": LocalServerProvider()` to existing `_PROVIDERS`.
- `packages/general_beckman/src/general_beckman/__init__.py` — add post-completion hook that calls `clair_obscur.record_release_hint()` (NOT direct stop) when image lane is finishing; stamp `preselected_pick_provider` at admission.
- root `conftest.py` `_PACKAGE_SRCS` — append `clair_obscur`.

**Test infra:**
- `tests/integration/test_image_local_e2e.py` — host-path e2e against a mock backend.

---

## Task 1: `clair_obscur` config + scaffolding (psutil declared)

**Files:**
- Create: `packages/clair_obscur/pyproject.toml`, `packages/clair_obscur/src/clair_obscur/__init__.py`, `.../config.py`
- Test: `packages/clair_obscur/tests/test_config.py`
- Modify: root `conftest.py` (`_PACKAGE_SRCS`)

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_config.py
import pytest
from clair_obscur.config import ClairObscurConfig, load_config


def test_default_backend_is_comfyui(monkeypatch):
    monkeypatch.delenv("CLAIR_OBSCUR_BACKEND", raising=False)
    monkeypatch.delenv("CLAIR_OBSCUR_URL", raising=False)
    cfg = load_config()
    assert cfg.backend == "comfyui"
    assert cfg.port == 8188
    assert cfg.base_url == "http://127.0.0.1:8188"


def test_env_selects_a1111(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "7860")
    monkeypatch.delenv("CLAIR_OBSCUR_URL", raising=False)
    cfg = load_config()
    assert cfg.backend == "a1111"
    assert cfg.port == 7860
    assert cfg.base_url == "http://127.0.0.1:7860"


def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "midjourney")
    with pytest.raises(ValueError):
        load_config()


def test_explicit_url_overrides_host_port(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_URL", "http://192.168.1.7:9000")
    cfg = load_config()
    assert cfg.base_url == "http://192.168.1.7:9000"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: clair_obscur`.

- [ ] **Step 3: Create the package skeleton + config**

`packages/clair_obscur/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "clair_obscur"
version = "0.1.0"
description = "Local image-server wrapper (image-side DaLLaMa). ComfyUI/A1111."
requires-python = ">=3.10"
dependencies = ["httpx>=0.27", "psutil>=5.9"]

[tool.setuptools.packages.find]
where = ["src"]
```

(psutil is declared even though it's transitively available via nerd_herd — clair_obscur's PID-lock + orphan-reconcile cannot run without it, so it's a hard requirement.)

`packages/clair_obscur/src/clair_obscur/config.py`:
```python
"""ClairObscur config — backend (comfyui|a1111), URL/port, model, weights dir.

Loaded from env so absent backend → clair_obscur.available() returns False
and hoca's image_select filters the local entry out (no crash)."""
from __future__ import annotations

import os
from dataclasses import dataclass


_VALID_BACKENDS = ("comfyui", "a1111")


@dataclass(frozen=True)
class ClairObscurConfig:
    backend: str            # "comfyui" | "a1111"
    host: str               # "127.0.0.1"
    port: int               # 8188 (comfyui) / 7860 (a1111) defaults
    base_url: str           # env CLAIR_OBSCUR_URL overrides host:port
    model: str              # SDXL / SD1.5 model filename or repo id
    weights_dir: str        # absolute path to backend's models directory
    exe_path: str           # absolute path to launcher
    idle_release_seconds: int = 60   # backstop after record_release_hint


def load_config() -> ClairObscurConfig:
    backend = os.getenv("CLAIR_OBSCUR_BACKEND", "comfyui").strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"CLAIR_OBSCUR_BACKEND={backend!r} not in {_VALID_BACKENDS}"
        )
    default_port = 8188 if backend == "comfyui" else 7860
    host = os.getenv("CLAIR_OBSCUR_HOST", "127.0.0.1")
    port = int(os.getenv("CLAIR_OBSCUR_PORT", str(default_port)))
    base_url = os.getenv("CLAIR_OBSCUR_URL", "").strip() or f"http://{host}:{port}"
    model = os.getenv("CLAIR_OBSCUR_MODEL", "sdxl-turbo")
    weights_dir = os.getenv("CLAIR_OBSCUR_WEIGHTS_DIR", "")
    exe_path = os.getenv("CLAIR_OBSCUR_EXE", "")
    idle = int(os.getenv("CLAIR_OBSCUR_IDLE_RELEASE_SECONDS", "60"))
    return ClairObscurConfig(
        backend=backend, host=host, port=port, base_url=base_url,
        model=model, weights_dir=weights_dir, exe_path=exe_path,
        idle_release_seconds=idle,
    )
```

`packages/clair_obscur/src/clair_obscur/__init__.py`:
```python
"""clair_obscur — local image-server wrapper. Parallel to dallama (LLM-side).

Lifecycle: start() / stop() / status() / available() / base_url() /
record_release_hint(). Holds a PID-lock at logs/image_server.lock and
reconciles its own orphan (ComfyUI / A1111 process — NEVER llama-server)."""
from __future__ import annotations

from .config import ClairObscurConfig, load_config
from .server import ImageServer

__all__ = [
    "ClairObscurConfig", "load_config", "ImageServer",
    "start", "stop", "status", "available", "base_url",
    "record_release_hint", "get_singleton",
]

_singleton: ImageServer | None = None


def get_singleton() -> ImageServer:
    global _singleton
    if _singleton is None:
        _singleton = ImageServer(load_config())
    return _singleton


async def start() -> str:
    return await get_singleton().start()


async def stop() -> None:
    await get_singleton().stop()


def status() -> dict:
    return get_singleton().status()


def available() -> bool:
    return get_singleton().available()


def base_url() -> str:
    return get_singleton().base_url()


def record_release_hint() -> None:
    """Beckman tells clair_obscur it MAY release (lane switch). The idle
    backstop in ImageServer.start() then times the actual stop() after
    config.idle_release_seconds. Direct .stop() is for forced/emergency
    shutdown only — normal lane switches go through this function."""
    get_singleton().record_release_hint()
```

- [ ] **Step 4: Register the package**

In root `conftest.py`, append to `_PACKAGE_SRCS` (same place `safety_guard` was added in commit `ae004547`):
```python
    _ROOT / "packages" / "clair_obscur" / "src",
```

Run: `.venv/Scripts/python -m pip install -e packages/clair_obscur`
Expected: `Successfully installed clair_obscur-0.1.0 psutil-...`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_config.py -q`
Expected: PASS (4 passed). (`server.py` doesn't exist yet — `__init__.py` will fail to import. If pytest's collector imports `__init__.py` and crashes, jump to Task 2 first, then return; otherwise the `from .server import ImageServer` line raises ImportError and the config test catches it fine because `clair_obscur.config` is reachable on its own.)

- [ ] **Step 6: Commit**

```bash
git add packages/clair_obscur/pyproject.toml packages/clair_obscur/src/clair_obscur/__init__.py packages/clair_obscur/src/clair_obscur/config.py packages/clair_obscur/tests/test_config.py conftest.py
git commit -m "feat(image): clair_obscur scaffold + config (psutil declared)"
```

---

## Task 2: `ImageServer` lifecycle — start/stop/health, **backstop wired to record_release_hint**

**Files:**
- Create: `packages/clair_obscur/src/clair_obscur/server.py`
- Test: `packages/clair_obscur/tests/test_server_lifecycle.py`

Key v2 change: the idle backstop watcher fires when `_release_hint_at` is non-None AND `(time.time() - _release_hint_at) >= idle_release_seconds`. Beckman's warm-batch hook (Task 11) calls `record_release_hint()` on lane switch — which sets `_release_hint_at` — and the watcher then stops the server after the configured idle window. If another image task arrives mid-window, the dispatcher's idempotent `start()` resets `_release_hint_at = None` (because the server is already up — start is a no-op), and the watcher waits for the next hint. This wires the backstop. **The direct `await self.stop()` path is for forced/emergency shutdown only** (wrapper shutdown, OOM emergency).

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_server_lifecycle.py
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path, idle=60):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(exe), idle_release_seconds=idle,
    )


def test_status_when_not_started(tmp_path):
    s = ImageServer(_cfg(tmp_path))
    st = s.status()
    assert st["resident"] is False and st["pid"] is None


@pytest.mark.asyncio
async def test_start_polls_health_until_ready(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path))
    calls = {"n": 0}

    async def _fake_launch(): s._pid = 12345
    async def _fake_health():
        calls["n"] += 1
        return calls["n"] >= 3
    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=4500: None)
    s._health_poll_interval = 0.01

    url = await s.start()
    assert url == "http://127.0.0.1:8188"
    assert calls["n"] >= 3
    assert s.status()["resident"] is True


@pytest.mark.asyncio
async def test_start_is_idempotent_and_clears_release_hint(monkeypatch, tmp_path):
    """Repeat start() when already resident clears any pending release hint —
    so an in-flight idle backstop window resets when a new image arrives."""
    s = ImageServer(_cfg(tmp_path))
    s._pid = 555
    s._resident = True
    s._release_hint_at = 100.0  # pretend a hint was recorded

    url = await s.start()
    assert url == "http://127.0.0.1:8188"
    assert s._release_hint_at is None, "start() must clear pending release hint"


@pytest.mark.asyncio
async def test_start_times_out_when_health_never_up(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path))
    async def _fake_launch(): s._pid = 99
    async def _fake_health(): return False
    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=0: None)
    monkeypatch.setattr(s, "_kill_own_pid", lambda pid: None)
    s._health_timeout_seconds = 0.3
    s._health_poll_interval = 0.05

    with pytest.raises(TimeoutError):
        await s.start()


@pytest.mark.asyncio
async def test_stop_releases_and_notifies(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path))
    s._pid = 12345; s._resident = True
    killed = {"pid": None}
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("pid", pid))
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=0: None)
    monkeypatch.setattr(s, "_release_lock", lambda: None)

    await s.stop()
    assert killed["pid"] == 12345
    st = s.status()
    assert st["resident"] is False and st["pid"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_server_lifecycle.py -q`
Expected: FAIL — `ModuleNotFoundError: server`.

- [ ] **Step 3: Implement `ImageServer`**

`packages/clair_obscur/src/clair_obscur/server.py`:
```python
"""ImageServer — async lifecycle wrapper around ComfyUI/A1111.

PID-locked at logs/image_server.lock. Boot orphan-reconcile validates the
stale PID's cmdline (via psutil) matches the configured backend launcher
BEFORE killing — never touches llama-server or any unrelated tenant.

Backstop discipline (v2): the watcher fires when record_release_hint() has
been called AND idle_release_seconds have elapsed since the hint. Direct
stop() is for forced/emergency shutdown only — normal lane switches drive
release through record_release_hint() so a back-to-back image batch can
reuse the warm server without restart."""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Optional

import httpx

from .config import ClairObscurConfig

_LOCK_PATH = os.path.join("logs", "image_server.lock")
_IS_WINDOWS = sys.platform == "win32"


class ImageServer:
    def __init__(self, config: ClairObscurConfig):
        self._config = config
        self._pid: Optional[int] = None
        self._resident: bool = False
        self._release_hint_at: Optional[float] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._health_timeout_seconds: float = 120.0
        self._health_poll_interval: float = 1.0

    # ───────── Public API ─────────
    def available(self) -> bool:
        return bool(self._config.exe_path) and os.path.exists(self._config.exe_path)

    def base_url(self) -> str:
        return self._config.base_url

    def status(self) -> dict:
        return {
            "resident": self._resident, "pid": self._pid,
            "backend": self._config.backend, "base_url": self._config.base_url,
            "release_hint_at": self._release_hint_at,
        }

    async def start(self) -> str:
        if self._resident and self._pid is not None:
            # Idempotent + clears any pending release hint so the backstop
            # window resets for the new image task.
            self._release_hint_at = None
            return self._config.base_url
        self._reconcile_orphan()
        await self._launch_process()
        self._acquire_lock()
        deadline = time.time() + self._health_timeout_seconds
        while time.time() < deadline:
            if await self._health_probe():
                self._resident = True
                self._release_hint_at = None
                self._notify_nerd_herd_resident(
                    vram_mb=self._estimated_resident_vram_mb()
                )
                self._arm_idle_backstop()
                return self._config.base_url
            await asyncio.sleep(self._health_poll_interval)
        # Health never came up → tear down + surface TimeoutError.
        if self._pid is not None:
            try: self._kill_own_pid(self._pid)
            finally: self._pid = None
        self._release_lock()
        raise TimeoutError(
            f"clair_obscur {self._config.backend} did not become healthy "
            f"within {self._health_timeout_seconds}s"
        )

    async def stop(self) -> None:
        """Forced/emergency stop. Normal lane switches go through
        record_release_hint() so the backstop handles release after idle."""
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
        pid = self._pid
        self._pid = None
        self._resident = False
        self._release_hint_at = None
        if pid is not None:
            self._kill_own_pid(pid)
        self._release_lock()
        self._notify_nerd_herd_resident(vram_mb=0)

    def record_release_hint(self) -> None:
        """Beckman tells us we may release. The watcher fires the actual
        stop() after idle_release_seconds — gives a back-to-back image task
        a window to reuse the warm server without restart."""
        self._release_hint_at = time.time()

    # ───────── Internals (test-mockable) ─────────
    async def _launch_process(self) -> None:
        """Spawn the backend. ComfyUI: `python main.py --listen <host> --port
        <port>`. A1111: `webui-user.bat` / `launch.py --api --listen --port
        <port>`. PID captured for our lock."""
        import subprocess
        cmd = self._build_launch_cmd()
        creationflags = 0
        if _IS_WINDOWS:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(  # noqa: S603 — caller-supplied exe
            cmd, creationflags=creationflags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._pid = proc.pid

    def _build_launch_cmd(self) -> list[str]:
        cfg = self._config
        if cfg.backend == "comfyui":
            return [cfg.exe_path, "-u", "main.py",
                    "--listen", cfg.host, "--port", str(cfg.port)]
        # a1111
        return [cfg.exe_path, "--api", "--listen",
                "--port", str(cfg.port), "--nowebui"]

    async def _health_probe(self) -> bool:
        url = self._health_url()
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                resp = await c.get(url)
                return 200 <= resp.status_code < 500
        except Exception:
            return False

    def _health_url(self) -> str:
        if self._config.backend == "comfyui":
            return f"{self._config.base_url}/system_stats"
        return f"{self._config.base_url}/sdapi/v1/sd-models"

    def _acquire_lock(self) -> None:
        if self._pid is None:
            return
        try:
            os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
            with open(_LOCK_PATH, "w", encoding="utf-8") as f:
                f.write(f"{self._pid}\n{self._config.backend}\n")
        except Exception:
            pass

    def _release_lock(self) -> None:
        try:
            if os.path.exists(_LOCK_PATH):
                os.remove(_LOCK_PATH)
        except Exception:
            pass

    def _reconcile_orphan(self) -> None:
        """Boot orphan-reconcile. Kills ONLY the PID written into our lock if
        AND ONLY IF its cmdline matches our configured backend launcher.
        Mirrors packages/nerd_herd/src/nerd_herd/platform.py:1-21 safety
        discipline. Never touches a PID we didn't write."""
        if not os.path.exists(_LOCK_PATH):
            return
        try:
            with open(_LOCK_PATH, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            stale_pid = int(lines[0]) if lines else 0
            stale_backend = lines[1] if len(lines) > 1 else ""
        except Exception:
            try: os.remove(_LOCK_PATH)
            except Exception: pass
            return

        if stale_pid <= 0 or stale_backend != self._config.backend:
            try: os.remove(_LOCK_PATH)
            except Exception: pass
            return

        if self._is_own_backend_pid(stale_pid):
            self._kill_own_pid(stale_pid)
        try: os.remove(_LOCK_PATH)
        except Exception: pass

    def _is_own_backend_pid(self, pid: int) -> bool:
        """True iff pid is alive AND cmdline matches our backend launcher."""
        if pid <= 0:
            return False
        try:
            import psutil
            if not psutil.pid_exists(pid):
                return False
            p = psutil.Process(pid)
            cmdline = " ".join(p.cmdline()).lower()
        except Exception:
            return False
        if self._config.backend == "comfyui":
            return "main.py" in cmdline or "comfyui" in cmdline
        return "webui" in cmdline or "launch.py" in cmdline

    def _kill_own_pid(self, pid: int) -> None:
        """Kill ONLY the given PID. NEVER taskkill-by-name."""
        try:
            import psutil
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                proc.terminate()
                try: proc.wait(timeout=5)
                except Exception: proc.kill()
        except Exception:
            pass

    def _estimated_resident_vram_mb(self) -> int:
        """SDXL-Turbo fp16 + activations @ 1024×1024 ≈ 4.5 GB."""
        return 4500

    def _notify_nerd_herd_resident(self, vram_mb: int) -> None:
        try:
            import nerd_herd
            nerd_herd.record_image_server_state(
                resident=(vram_mb > 0), vram_mb=vram_mb,
            )
        except Exception:
            pass

    def _arm_idle_backstop(self) -> None:
        """Schedule the safety/normal release. Watcher fires the actual
        stop() when (now - release_hint_at) >= idle_release_seconds AND
        resident is still True. Cancelled on stop() / restart."""
        if self._idle_task is not None and not self._idle_task.done():
            return
        async def _watcher():
            cfg = self._config
            tick = min(5.0, max(1.0, cfg.idle_release_seconds / 4))
            while self._resident:
                await asyncio.sleep(tick)
                if not self._resident:
                    return
                hint = self._release_hint_at
                if hint is not None and (time.time() - hint) >= cfg.idle_release_seconds:
                    await self.stop()
                    return
        try:
            loop = asyncio.get_running_loop()
            self._idle_task = loop.create_task(_watcher())
        except RuntimeError:
            self._idle_task = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_server_lifecycle.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/clair_obscur/src/clair_obscur/server.py packages/clair_obscur/tests/test_server_lifecycle.py
git commit -m "feat(image): clair_obscur ImageServer (start/stop/health/idempotent-start)"
```

---

## Task 3: PID-lock + boot orphan-reconcile safety contract

**Files:**
- Modify (if needed): `packages/clair_obscur/src/clair_obscur/server.py`
- Test: `packages/clair_obscur/tests/test_orphan_reconcile.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_orphan_reconcile.py
import os
from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer, _LOCK_PATH


def _cfg(tmp_path, backend="comfyui"):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend=backend, host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(exe), idle_release_seconds=60,
    )


def _write_lock(pid: int, backend: str):
    os.makedirs(os.path.dirname(_LOCK_PATH) or ".", exist_ok=True)
    with open(_LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(f"{pid}\n{backend}\n")


def _clear_lock():
    try: os.remove(_LOCK_PATH)
    except FileNotFoundError: pass


def test_orphan_killed_when_lock_points_at_our_backend(monkeypatch, tmp_path):
    _clear_lock(); _write_lock(12345, "comfyui")
    killed = {"pid": None}
    s = ImageServer(_cfg(tmp_path))
    monkeypatch.setattr(s, "_is_own_backend_pid", lambda pid: pid == 12345)
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("pid", pid))
    s._reconcile_orphan()
    assert killed["pid"] == 12345
    assert not os.path.exists(_LOCK_PATH)


def test_orphan_skipped_when_pid_is_not_our_backend(monkeypatch, tmp_path):
    """SAFETY: if the PID in image_server.lock no longer belongs to a
    ComfyUI/A1111 process (e.g. it now belongs to llama-server or any
    other tenant — PID-reuse case), the sweep MUST NOT kill it."""
    _clear_lock(); _write_lock(67890, "comfyui")
    killed = {"called": False}
    s = ImageServer(_cfg(tmp_path))
    monkeypatch.setattr(s, "_is_own_backend_pid", lambda pid: False)
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("called", True))
    s._reconcile_orphan()
    assert killed["called"] is False
    assert not os.path.exists(_LOCK_PATH)


def test_orphan_skipped_when_backend_in_lock_does_not_match(monkeypatch, tmp_path):
    """Lock says 'a1111' but config says 'comfyui' — refuse to touch."""
    _clear_lock(); _write_lock(11111, "a1111")
    killed = {"called": False}
    s = ImageServer(_cfg(tmp_path, backend="comfyui"))
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("called", True))
    s._reconcile_orphan()
    assert killed["called"] is False


def test_no_lock_no_action(tmp_path):
    _clear_lock()
    s = ImageServer(_cfg(tmp_path))
    s._reconcile_orphan()  # must not raise


def test_corrupt_lock_clears_quietly(tmp_path):
    """Lock with garbage content (e.g. half-written from a previous crash)
    must clear the file without raising or killing anything."""
    os.makedirs(os.path.dirname(_LOCK_PATH) or ".", exist_ok=True)
    with open(_LOCK_PATH, "w", encoding="utf-8") as f:
        f.write("not-a-pid\n")
    s = ImageServer(_cfg(tmp_path))
    s._reconcile_orphan()
    assert not os.path.exists(_LOCK_PATH)
```

- [ ] **Step 2: Run + verify the contract holds (Task 2 wrote `_reconcile_orphan` correctly; this task is a safety-contract sweep)**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_orphan_reconcile.py -q`
Expected: PASS (5 passed). If any test fails, the bug is in `_reconcile_orphan` from Task 2 — fix the guard there, not the test.

- [ ] **Step 3: Commit**

```bash
git add packages/clair_obscur/tests/test_orphan_reconcile.py
git commit -m "test(image): clair_obscur orphan-reconcile safety contract"
```

---

## Task 4: Idle-release backstop — fires on hint + idle window

**Files:**
- Modify (if needed): `packages/clair_obscur/src/clair_obscur/server.py:_arm_idle_backstop`
- Test: `packages/clair_obscur/tests/test_idle_backstop.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_idle_backstop.py
import asyncio
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path, idle=0.2):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(exe), idle_release_seconds=idle,
    )


@pytest.mark.asyncio
async def test_release_hint_then_idle_triggers_stop(monkeypatch, tmp_path):
    """After record_release_hint(), the watcher fires stop() once the idle
    window elapses. This is the normal lane-switch release path."""
    s = ImageServer(_cfg(tmp_path, idle=0.2))
    s._resident = True
    s._pid = 555
    stops = {"n": 0}
    async def _fake_stop():
        stops["n"] += 1
        s._resident = False; s._pid = None
    monkeypatch.setattr(s, "stop", _fake_stop)

    s._arm_idle_backstop()
    s.record_release_hint()
    await asyncio.sleep(0.5)
    assert stops["n"] == 1, "backstop must call stop() after hint + idle"


@pytest.mark.asyncio
async def test_no_release_hint_means_no_stop(monkeypatch, tmp_path):
    """Without record_release_hint(), the watcher must NOT stop the server.
    This is the warm-batch case: beckman never hinted, so we hold."""
    s = ImageServer(_cfg(tmp_path, idle=0.2))
    s._resident = True
    s._pid = 555
    stops = {"n": 0}
    async def _fake_stop():
        stops["n"] += 1; s._resident = False
    monkeypatch.setattr(s, "stop", _fake_stop)

    s._arm_idle_backstop()
    await asyncio.sleep(0.5)
    assert stops["n"] == 0
    s._resident = False  # let watcher exit


@pytest.mark.asyncio
async def test_hint_cleared_by_idempotent_start_extends_window(
    monkeypatch, tmp_path,
):
    """If a new image task arrives mid-window, the dispatcher's idempotent
    start() clears the hint (set in Task 2). The watcher must then NOT
    stop the server, allowing the warm batch to continue."""
    s = ImageServer(_cfg(tmp_path, idle=0.2))
    s._resident = True
    s._pid = 555
    stops = {"n": 0}
    async def _fake_stop():
        stops["n"] += 1; s._resident = False
    monkeypatch.setattr(s, "stop", _fake_stop)

    s._arm_idle_backstop()
    s.record_release_hint()
    await asyncio.sleep(0.05)
    # Mid-window, dispatcher calls start() for the next image — idempotent
    # branch clears the hint:
    await s.start()  # idempotent branch (resident=True already)
    await asyncio.sleep(0.5)
    assert stops["n"] == 0
```

- [ ] **Step 2: Run + verify**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_idle_backstop.py -q`
Expected: PASS (3 passed). If `test_release_hint_then_idle_triggers_stop` fails, the watcher's guard is wrong — fix the `hint is not None and (now - hint) >= idle_release_seconds` condition.

- [ ] **Step 3: Commit**

```bash
git add packages/clair_obscur/tests/test_idle_backstop.py
git commit -m "test(image): clair_obscur idle backstop (hint + window → stop)"
```

---

## Task 5: nerd_herd `image_server_resident` + `image_server_vram_mb`

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py` (`SystemSnapshot`)
- Modify: `packages/nerd_herd/src/nerd_herd/nerd_herd.py:19-59` (`__init__`) + `:140-166` (`snapshot()`) + new `push_image_server_state` method
- Modify: `packages/nerd_herd/src/nerd_herd/__init__.py` — add module-level `record_image_server_state`
- Test: `packages/nerd_herd/tests/test_image_server_state.py`

Anchored on recon: `__init__` initializes 11 attributes ending with `self._server = MetricsServer(...)`. `snapshot()` builder takes 6 keyword args. Module-level access uses `refresh_snapshot()` — there is NO `nerd_herd.snapshot()` function.

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_image_server_state.py
import nerd_herd


def test_default_snapshot_has_image_server_fields():
    nh = nerd_herd.NerdHerd()
    snap = nh.snapshot()
    assert hasattr(snap, "image_server_resident")
    assert hasattr(snap, "image_server_vram_mb")
    assert snap.image_server_resident is False
    assert snap.image_server_vram_mb == 0


def test_push_image_server_state_flips_snapshot():
    nh = nerd_herd.NerdHerd()
    nh.push_image_server_state(resident=True, vram_mb=4500)
    snap = nh.snapshot()
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb == 4500
    nh.push_image_server_state(resident=False, vram_mb=0)
    snap = nh.snapshot()
    assert snap.image_server_resident is False


def test_module_level_record_image_server_state():
    """Singleton path. record_image_server_state writes through to the
    singleton's push_image_server_state, observable via refresh_snapshot()."""
    nerd_herd.record_image_server_state(resident=True, vram_mb=4500)
    snap = nerd_herd.refresh_snapshot()
    # SystemSnapshot now has the two new fields regardless of state.
    assert hasattr(snap, "image_server_resident")
    assert hasattr(snap, "image_server_vram_mb")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/test_image_server_state.py -q`
Expected: FAIL — `AttributeError: image_server_resident`.

- [ ] **Step 3: Extend `SystemSnapshot`**

In `packages/nerd_herd/src/nerd_herd/types.py`, in the `SystemSnapshot` dataclass, append two fields (after `recent_swap_count`):
```python
    # Image-server residency (clair_obscur). Read by fatih_hoca's
    # image_select._eviction_cost. Written via NerdHerd.push_image_server_state()
    # / module-level record_image_server_state(), driven by clair_obscur on
    # start()/stop() (Plan 2).
    image_server_resident: bool = False
    image_server_vram_mb: int = 0
```

- [ ] **Step 4: Init two new attrs + push method + extend `snapshot()`**

In `packages/nerd_herd/src/nerd_herd/nerd_herd.py`:

After `self._server = MetricsServer(...)` (line ~57), append:
```python
        # Image-server residency (clair_obscur). Default 0/False until
        # clair_obscur.start()/.stop() pushes state.
        self._image_server_resident: bool = False
        self._image_server_vram_mb: int = 0
```

After `record_swap` (line ~114), append:
```python
    def push_image_server_state(self, *, resident: bool, vram_mb: int) -> None:
        """Replace image-server residency state (pushed by clair_obscur on
        start/stop). Read by fatih_hoca.image_select._eviction_cost."""
        self._image_server_resident = bool(resident)
        self._image_server_vram_mb = int(vram_mb or 0)
```

In `snapshot()` (line ~159-166), extend the `SystemSnapshot(...)` call:
```python
        return SystemSnapshot(
            vram_available_mb=self.get_vram_budget_mb() if gpu.available else 0,
            local=local,
            cloud=dict(self._cloud_state),
            queue_profile=self._queue_profile,
            in_flight_calls=list(self._in_flight_calls),
            recent_swap_count=self._swap_budget.recent_count(),
            image_server_resident=self._image_server_resident,
            image_server_vram_mb=self._image_server_vram_mb,
        )
```

- [ ] **Step 5: Module-level `record_image_server_state`**

In `packages/nerd_herd/src/nerd_herd/__init__.py`, after `record_swap` (around line 73), add:
```python
def record_image_server_state(*, resident: bool, vram_mb: int) -> None:
    """Record clair_obscur (local image-server) residency. Called by
    clair_obscur on start/stop. Read by fatih_hoca.image_select via the
    snapshot."""
    _get_singleton().push_image_server_state(resident=resident, vram_mb=vram_mb)
```

Add `"record_image_server_state"` to `__all__`.

- [ ] **Step 6: Run + regression**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/test_image_server_state.py packages/nerd_herd/tests/ -q -x`
Expected: PASS (3 new + no regressions).

- [ ] **Step 7: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/src/nerd_herd/nerd_herd.py packages/nerd_herd/src/nerd_herd/__init__.py packages/nerd_herd/tests/test_image_server_state.py
git commit -m "feat(image): nerd_herd image_server_resident + vram_mb"
```

---

## Task 6: clair_obscur ↔ nerd_herd wiring on start/stop

**Files:**
- Test: `packages/clair_obscur/tests/test_nerd_herd_wiring.py`

Task 2 already calls `nerd_herd.record_image_server_state(...)` inside `_notify_nerd_herd_resident`. Task 5 implemented the recipient. This task asserts the round-trip.

- [ ] **Step 1: Write the test**

```python
# packages/clair_obscur/tests/test_nerd_herd_wiring.py
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path):
    exe = tmp_path / "fake_exe"; exe.write_text("x")
    return ClairObscurConfig(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(exe), idle_release_seconds=60,
    )


@pytest.mark.asyncio
async def test_start_pushes_resident_true(monkeypatch, tmp_path):
    import nerd_herd
    nerd_herd.record_image_server_state(resident=False, vram_mb=0)

    s = ImageServer(_cfg(tmp_path))
    async def _fake_launch(): s._pid = 111
    async def _fake_health(): return True
    monkeypatch.setattr(s, "_launch_process", _fake_launch)
    monkeypatch.setattr(s, "_health_probe", _fake_health)
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)

    await s.start()
    snap = nerd_herd.refresh_snapshot()
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb > 0


@pytest.mark.asyncio
async def test_stop_pushes_resident_false(monkeypatch, tmp_path):
    import nerd_herd
    nerd_herd.record_image_server_state(resident=True, vram_mb=4500)

    s = ImageServer(_cfg(tmp_path))
    s._pid = 222; s._resident = True
    monkeypatch.setattr(s, "_kill_own_pid", lambda pid: None)
    monkeypatch.setattr(s, "_release_lock", lambda: None)

    await s.stop()
    snap = nerd_herd.refresh_snapshot()
    assert snap.image_server_resident is False
    assert snap.image_server_vram_mb == 0
```

- [ ] **Step 2: Run + commit**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_nerd_herd_wiring.py -q`
Expected: PASS (2 passed).

```bash
git add packages/clair_obscur/tests/test_nerd_herd_wiring.py
git commit -m "test(image): clair_obscur ↔ nerd_herd start/stop wiring"
```

---

## Task 7: paintress `local_server` provider (ComfyUI + A1111)

**Files:**
- Create: `packages/paintress/src/paintress/providers/local_server.py`
- Modify: `packages/paintress/src/paintress/__init__.py` — APPEND `"clair_obscur": LocalServerProvider()` to `_PROVIDERS`
- Test: `packages/paintress/tests/test_local_server.py`

(Identical to Plan 2 v1 Task 7 — the provider doesn't depend on the audit findings. Body kept verbatim for completeness; husam's local handover in `_run_image` (Task 10) is where the v2 fix lives.)

- [ ] **Step 1: Write the failing test**

```python
# packages/paintress/tests/test_local_server.py
import base64
import io

import pytest
from PIL import Image

from paintress.providers.local_server import LocalServerProvider
from paintress.types import ImageSpec


def _png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (50, 100, 150)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_a1111_post_returns_bytes(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "7860")

    class _Resp:
        status_code = 200
        def json(self): return {"images": [_png_b64()], "info": "{\"seed\": 99}"}
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw):
            assert "/sdapi/v1/txt2img" in url
            return _Resp()

    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)

    prov = LocalServerProvider()
    data, meta = await prov.generate(
        ImageSpec(prompt="a cat", out_dir="/tmp", seed=99, width=512, height=512),
        base_url="http://127.0.0.1:7860",
    )
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 99


@pytest.mark.asyncio
async def test_comfyui_prompt_then_history_poll(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "comfyui")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "8188")

    state = {"posted": False, "history_calls": 0, "viewed": False}

    class _PromptResp:
        status_code = 200
        def json(self): return {"prompt_id": "abc-123"}
        def raise_for_status(self): pass
    class _HistEmpty:
        status_code = 200
        def json(self): return {}
        def raise_for_status(self): pass
    class _HistDone:
        status_code = 200
        def json(self):
            return {"abc-123": {"outputs": {"9": {"images": [{
                "filename": "out.png", "subfolder": "", "type": "output",
            }]}}}}
        def raise_for_status(self): pass
    class _Bytes:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nFAKE"
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw):
            state["posted"] = True
            return _PromptResp()
        async def get(self, url, **kw):
            if "/history/" in url:
                state["history_calls"] += 1
                return _HistEmpty() if state["history_calls"] < 2 else _HistDone()
            if "/view" in url:
                state["viewed"] = True; return _Bytes()
            raise AssertionError(url)

    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)
    import paintress.providers.local_server as ls
    monkeypatch.setattr(ls, "_PROMPT_POLL_INTERVAL", 0.01)

    data, meta = await LocalServerProvider().generate(
        ImageSpec(prompt="a dog", out_dir="/tmp", seed=7, width=512, height=512),
        base_url="http://127.0.0.1:8188",
    )
    assert state["posted"] and state["viewed"]
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 7


def test_available_reflects_exe_present(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(tmp_path / "no_such_exe"))
    assert LocalServerProvider().available() is False
    exe = tmp_path / "exe"; exe.write_text("x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(exe))
    assert LocalServerProvider().available() is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_local_server.py -q`
Expected: FAIL — `ModuleNotFoundError: local_server`.

- [ ] **Step 3: Implement the provider**

`packages/paintress/src/paintress/providers/local_server.py`:
```python
"""LocalServerProvider — paintress provider that calls clair_obscur's local
backend (ComfyUI default, A1111 via env). Backend chosen via clair_obscur
config so paintress doesn't duplicate config logic."""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Tuple

import httpx

from ..types import ImageSpec

_PROMPT_POLL_INTERVAL = 1.0
_PROMPT_TIMEOUT = 180.0


class LocalServerProvider:
    name = "clair_obscur"

    def available(self) -> bool:
        """Mirrors clair_obscur.available(). False when no backend exe set."""
        try:
            from clair_obscur import available as _co_available
            return bool(_co_available())
        except Exception:
            exe = os.getenv("CLAIR_OBSCUR_EXE", "")
            return bool(exe) and os.path.exists(exe)

    async def generate(
        self, spec: ImageSpec, *, base_url: str | None = None
    ) -> Tuple[bytes, dict]:
        backend = (os.getenv("CLAIR_OBSCUR_BACKEND", "comfyui") or "comfyui").lower()
        if not base_url:
            try:
                from clair_obscur import base_url as _co_base_url
                base_url = _co_base_url()
            except Exception:
                base_url = "http://127.0.0.1:8188"
        if backend == "a1111":
            return await self._a1111(spec, base_url)
        return await self._comfyui(spec, base_url)

    async def _a1111(self, spec: ImageSpec, base_url: str) -> Tuple[bytes, dict]:
        url = f"{base_url.rstrip('/')}/sdapi/v1/txt2img"
        payload = {
            "prompt": spec.prompt or "",
            "negative_prompt": spec.negative_prompt or "",
            "width": int(spec.width), "height": int(spec.height),
            "steps": int(spec.steps) if spec.steps else 20,
            "seed": -1 if spec.seed is None else int(spec.seed),
        }
        async with httpx.AsyncClient(timeout=_PROMPT_TIMEOUT) as c:
            resp = await c.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        images = body.get("images") or []
        if not images:
            raise RuntimeError("a1111_no_image")
        data = base64.b64decode(images[0])
        info = body.get("info") or "{}"
        try:
            info_d = json.loads(info)
            seed_used = info_d.get("seed", spec.seed)
        except Exception:
            seed_used = spec.seed
        return data, {"seed_used": seed_used}

    async def _comfyui(self, spec: ImageSpec, base_url: str) -> Tuple[bytes, dict]:
        base = base_url.rstrip("/")
        workflow = self._build_comfyui_workflow(spec)
        async with httpx.AsyncClient(timeout=_PROMPT_TIMEOUT) as c:
            resp = await c.post(f"{base}/prompt", json={"prompt": workflow})
            resp.raise_for_status()
            prompt_id = (resp.json() or {}).get("prompt_id")
            if not prompt_id:
                raise RuntimeError("comfyui_no_prompt_id")
            deadline = asyncio.get_event_loop().time() + _PROMPT_TIMEOUT
            image_meta = None
            while asyncio.get_event_loop().time() < deadline:
                h = await c.get(f"{base}/history/{prompt_id}")
                h.raise_for_status()
                hist = h.json() or {}
                entry = hist.get(prompt_id)
                if entry and entry.get("outputs"):
                    for _id, out in entry["outputs"].items():
                        imgs = out.get("images") or []
                        if imgs:
                            image_meta = imgs[0]
                            break
                if image_meta is not None:
                    break
                await asyncio.sleep(_PROMPT_POLL_INTERVAL)
            if image_meta is None:
                raise RuntimeError("comfyui_timeout")
            params = {
                "filename": image_meta["filename"],
                "subfolder": image_meta.get("subfolder", ""),
                "type": image_meta.get("type", "output"),
            }
            v = await c.get(f"{base}/view", params=params)
            v.raise_for_status()
            return v.content, {"seed_used": spec.seed}

    def _build_comfyui_workflow(self, spec: ImageSpec) -> dict:
        seed = int(spec.seed) if spec.seed is not None else 0
        steps = int(spec.steps) if spec.steps else 20
        return {
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": steps, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0],
                "negative": ["7", 0], "latent_image": ["5", 0],
            }},
            "4": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": os.getenv("CLAIR_OBSCUR_MODEL",
                                                    "sdxl-turbo")}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {
                "width": int(spec.width), "height": int(spec.height),
                "batch_size": 1,
            }},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": spec.prompt or "", "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": spec.negative_prompt or "", "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "kutai", "images": ["8", 0]}},
        }
```

- [ ] **Step 4: Register in `_PROVIDERS`**

In `packages/paintress/src/paintress/__init__.py`, after the HuggingFace import (Plan 1 v2 wired both cloud entries), add:
```python
from .providers.local_server import LocalServerProvider
_PROVIDERS = {
    "pollinations": PollinationsProvider(),
    "huggingface": HuggingFaceProvider(),
    "clair_obscur": LocalServerProvider(),
}
```

- [ ] **Step 5: Run + commit**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/ -q`
Expected: PASS (Plan 1 v2's 10 + Plan 2's 3).

```bash
git add packages/paintress/src/paintress/providers/local_server.py packages/paintress/src/paintress/__init__.py packages/paintress/tests/test_local_server.py
git commit -m "feat(image): paintress local_server provider (clair_obscur)"
```

---

## Task 8: Add local SDXL entry to fatih_hoca image catalog

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/image_providers.py` — APPEND one entry
- Test: `packages/fatih_hoca/tests/test_image_providers_local.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_image_providers_local.py
from fatih_hoca.image_providers import image_catalog
from fatih_hoca.registry import ImageModelInfo


def test_catalog_has_clair_obscur_entry():
    cat = {m.name: m for m in image_catalog()}
    assert "clair_obscur/sdxl-turbo" in cat
    co = cat["clair_obscur/sdxl-turbo"]
    assert isinstance(co, ImageModelInfo)
    assert co.provider == "clair_obscur"
    assert co.is_local is True
    assert 3000 <= co.vram_mb <= 6000  # fits 8GB after llama unload
    assert co.cost_per_image == 0.0


def test_cloud_entries_still_present():
    """Plan 2 must NOT remove or reorder Plan 1 v2's cloud entries."""
    names = {m.name for m in image_catalog()}
    assert "pollinations/flux" in names
    assert "huggingface/flux-schnell" in names
```

- [ ] **Step 2: Run + implement**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_providers_local.py -q`
Expected: FAIL.

In `image_providers.py`, APPEND inside `image_catalog()`'s returned list:
```python
        ImageModelInfo(
            name="clair_obscur/sdxl-turbo", provider="clair_obscur",
            location="local", endpoint="",  # set at dispatch from base_url()
            quality_rank=7.5, cost_per_image=0.0, vram_mb=4500,
            supports_seed=True, tier="local",
        ),
```

Run again: PASS.

```bash
git add packages/fatih_hoca/src/fatih_hoca/image_providers.py packages/fatih_hoca/tests/test_image_providers_local.py
git commit -m "feat(image): catalog adds clair_obscur/sdxl-turbo"
```

---

## Task 9: Real eviction-cost + VRAM-fit gate — reads `refresh_snapshot()`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/image_select.py`
- Test: `packages/fatih_hoca/tests/test_image_select_eviction.py`

Recon: there is **no module-level `nerd_herd.snapshot()`**. Plan 2 v2 reads `refresh_snapshot()` (the same function beckman uses). The `_snapshot()` helper in `image_select.py` is the test-mockable seam.

Eviction formula (calibrated against quality_rank spread 6.0/7.5/8.0):
- `image_server_resident` → 0 (warm batch)
- `llm_in_flight > 0` → HUGE = 100.0 (kills any chance of local)
- `llm_loaded or llm_queue > 0` → HIGH = 50.0 (cloud wins)
- idle → LOW = 2.0 (cloud still wins first call; resident=True flips for batches)

Plus `_WARM_BATCH_BONUS = 1.0` on resident local so 7.5 + 1.0 = 8.5 > HF 8.0, batching falls out.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_image_select_eviction.py
import pytest

from fatih_hoca.image_select import select_image
from fatih_hoca.types import Pick


def _snap(*, llm_in_flight=0, llm_loaded=False, llm_queue=0,
          image_resident=False, vram_mb=6000):
    class _Local:
        model_name = "qwen2.5" if llm_loaded else None
        requests_processing = llm_in_flight
    class _QP:
        total_ready_count = llm_queue
    class _S:
        local = _Local()
        queue_profile = _QP()
        in_flight_calls = []
        image_server_resident = image_resident
        image_server_vram_mb = 4500 if image_resident else 0
        vram_available_mb = vram_mb
    return _S()


def test_huge_when_llm_in_flight(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(llm_in_flight=1))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.is_local is False


def test_high_when_llm_loaded(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(llm_loaded=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False


def test_low_when_idle_first_call_still_cloud(monkeypatch):
    """8.0 (HF) > 7.5 − 2.0 = 5.5 (local LOW) → HF wins cold start."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot", lambda: _snap())
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"


def test_resident_with_warm_bonus_picks_local(monkeypatch):
    """Image-server warm → local score = 7.5 + 1.0 = 8.5 > HF 8.0."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.provider == "clair_obscur"


def test_vram_too_low_filters_local(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(vram_mb=2000))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False


def test_local_unavailable_filters_local(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(vram_mb=8000))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.delenv("CLAIR_OBSCUR_EXE", raising=False)
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_select_eviction.py -q`
Expected: FAIL.

- [ ] **Step 3: Replace `_eviction_cost`, extend `_provider_available`, add VRAM-fit gate, use `refresh_snapshot()`**

In `packages/fatih_hoca/src/fatih_hoca/image_select.py`:

Replace `_eviction_cost(m)` (Plan 1 v2's stub returning 0.0) and add the snapshot helper + warm bonus + provider arm:
```python
import os  # if not already imported


_EVICTION_HUGE = 100.0
_EVICTION_HIGH = 50.0
_EVICTION_LOW = 2.0
_WARM_BATCH_BONUS = 1.0


def _snapshot():
    """Read nerd_herd state. Module-level `refresh_snapshot()` exists
    (recon-verified); there is NO `nerd_herd.snapshot()` symbol."""
    try:
        import nerd_herd
        return nerd_herd.refresh_snapshot()
    except Exception:
        from nerd_herd.types import SystemSnapshot
        return SystemSnapshot()


def _eviction_cost(m, snap=None) -> float:
    if not getattr(m, "is_local", False):
        return 0.0
    s = snap if snap is not None else _snapshot()
    if getattr(s, "image_server_resident", False):
        return 0.0
    in_flight = len(getattr(s, "in_flight_calls", []) or [])
    if in_flight == 0:
        local = getattr(s, "local", None)
        in_flight = int(getattr(local, "requests_processing", 0) or 0)
    if in_flight > 0:
        return _EVICTION_HUGE
    llm_loaded = bool(getattr(getattr(s, "local", None), "model_name", None))
    qp = getattr(s, "queue_profile", None)
    llm_queue = int(getattr(qp, "total_ready_count", 0) or 0) if qp else 0
    if llm_loaded or llm_queue > 0:
        return _EVICTION_HIGH
    return _EVICTION_LOW


def _warm_batch_bonus(m, snap) -> float:
    if not getattr(m, "is_local", False):
        return 0.0
    return _WARM_BATCH_BONUS if getattr(snap, "image_server_resident", False) else 0.0
```

Extend `_provider_available` with the clair_obscur arm:
```python
def _provider_available(m, hf_available):
    if m.provider == "huggingface":
        return os.getenv("HF_TOKEN") is not None if hf_available is None else hf_available
    if m.provider == "pollinations":
        return True
    if m.provider == "clair_obscur":
        try:
            import clair_obscur
            return bool(clair_obscur.available())
        except Exception:
            exe = os.getenv("CLAIR_OBSCUR_EXE", "")
            return bool(exe) and os.path.exists(exe)
    return False
```

Replace `select_image`'s candidate loop with the VRAM-fit gate + new scoring:
```python
def select_image(*, quality_tier="fast", failures=None,
                 hf_available=None, remaining_budget_usd=None):
    failed = set(failures or [])
    snap = _snapshot()
    candidates = []
    for m in image_catalog():
        if m.name in failed:
            continue
        if not _provider_available(m, hf_available):
            continue
        # VRAM-fit eligibility (mirrors selector.py:459 needs_vision gate).
        # Local: refuse if free VRAM (after a hypothetical llama unload — we
        # add a conservative 4GB local-recoverable allowance) can't fit.
        if m.is_local and m.vram_mb > 0:
            free_mb = int(getattr(snap, "vram_available_mb", 0) or 0)
            llm_loaded_mb = 4000 if getattr(
                getattr(snap, "local", None), "model_name", None
            ) else 0
            if (free_mb + llm_loaded_mb) < m.vram_mb:
                continue
        if remaining_budget_usd is not None and m.cost_per_image > remaining_budget_usd:
            continue
        score = m.quality_rank - _eviction_cost(m, snap) + _warm_batch_bonus(m, snap)
        candidates.append((score, m))

    if not candidates:
        return SelectionFailure(reason="availability",
                                detail="no eligible image provider")
    candidates.sort(key=lambda t: t[0], reverse=True)
    top_summary = "; ".join(f"{m.name}:{s:.1f}" for s, m in candidates[:5])
    best_score, best = candidates[0]
    return Pick(model=best, min_time_seconds=0.0, score=best_score,
                top_summary=top_summary)
```

- [ ] **Step 4: Run + regression**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_select_eviction.py packages/fatih_hoca/tests/test_image_select.py packages/fatih_hoca/tests/ -q -x`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/image_select.py packages/fatih_hoca/tests/test_image_select_eviction.py
git commit -m "feat(image): real eviction-cost + VRAM-fit (reads refresh_snapshot)"
```

---

## Task 10: Husam local handover **wrapped in `heartbeat.keepalive()`**

**Files:**
- Modify: `packages/husam/src/husam/worker.py` — extend `_run_image` with `if pick.model.is_local:` handover branch INSIDE the `keepalive()` context
- Test: `packages/husam/tests/test_husam_image_local.py`

The fix: `get_local_manager().shutdown()` + VRAM poll + clair_obscur cold start can take 30-60s+. The orchestrator's 300s no-progress watchdog reads heartbeats. Plan 1 v3's `husam._run_image` wraps `paintress.generate` in `keepalive()` (the existing block at `worker.py:113-114`). Plan 2 expands the wrap to cover unload + poll + start so the watchdog stays satisfied through the cold-start window. `shutdown()` is internally guarded (`if self._started: await self._dallama.stop()`) so it's a safe no-op when no local LLM is loaded; DaLLaMa lazy-reloads on the next LLM task's `ensure_model`. Recon confirmed `keepalive()` is reentrant + contextvar-safe.

- [ ] **Step 1: Write the failing test**

```python
# packages/husam/tests/test_husam_image_local.py
import pytest

from husam import run as husam_run


@pytest.mark.asyncio
async def test_local_image_handover_ordering(monkeypatch, tmp_path):
    """Order must be: unload → poll → clair_obscur.start → record_swap →
    paintress.generate. All inside the existing keepalive() span (worker.py)
    so the watchdog sees bumps through the cold-start window."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick

    order = []

    class _Mgr:
        async def shutdown(self):
            order.append("unload")
    # _run_image imports get_local_manager from src.models.local_model_manager.
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())

    poll_calls = {"n": 0}
    def _snap():
        poll_calls["n"] += 1
        # First snapshot: VRAM still low; second: high enough.
        class _S:
            vram_available_mb = 1000 if poll_calls["n"] == 1 else 7000
        return _S()
    monkeypatch.setattr("nerd_herd.refresh_snapshot", _snap)

    swaps = []
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": swaps.append(name))

    async def _co_start():
        order.append("clair_obscur.start")
        return "http://127.0.0.1:8188"
    monkeypatch.setattr("clair_obscur.start", _co_start)

    async def _fake_generate(pick, spec):
        order.append("paintress.generate")
        from paintress import ImageResult
        out = tmp_path / "out.png"
        out.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
        return ImageResult(path=str(out), provider="clair_obscur",
                           model="clair_obscur/sdxl-turbo", cost=0.0, seed_used=7)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    model = ImageModelInfo(
        name="clair_obscur/sdxl-turbo", provider="clair_obscur",
        location="local", endpoint="", quality_rank=7.5,
        cost_per_image=0.0, vram_mb=4500, supports_seed=True,
    )
    pick = Pick(model=model, min_time_seconds=0.0, score=8.5, top_summary="t")
    task = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a fox", "out_dir": str(tmp_path),
            "width": 512, "height": 512, "quality_tier": "fast",
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    res = await husam_run(task)
    assert res["path"].endswith(".png")
    # unload → start → swap come before generate (handover precedes the call).
    assert order == ["unload", "clair_obscur.start", "paintress.generate"]
    assert swaps == ["clair_obscur/sdxl-turbo"]
    assert poll_calls["n"] >= 2  # polled until VRAM fit


@pytest.mark.asyncio
async def test_keepalive_wraps_long_handover(monkeypatch, tmp_path):
    """The handover (shutdown + poll + start) must run INSIDE the keepalive()
    span so heartbeat bumps remain reachable. We assert the context does not
    raise and the local branch executes."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick

    bumps = {"n": 0}
    monkeypatch.setattr("src.core.heartbeat.bump",
                        lambda task_id=None: bumps.__setitem__("n", bumps["n"] + 1))

    class _Mgr:
        async def shutdown(self):
            # Simulate a slow unload (>30s would normally trip watchdog;
            # we only assert the keepalive span stays reachable).
            import asyncio
            await asyncio.sleep(0)
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())
    class _S:
        vram_available_mb = 7000
    monkeypatch.setattr("nerd_herd.refresh_snapshot", lambda: _S())
    monkeypatch.setattr("nerd_herd.record_swap", lambda name="": None)
    async def _co_start(): return "http://127.0.0.1:8188"
    monkeypatch.setattr("clair_obscur.start", _co_start)

    async def _gen(pick, spec):
        from paintress import ImageResult
        p = tmp_path / "x.png"; p.write_bytes(b"\x89PNG")
        return ImageResult(path=str(p), provider="clair_obscur",
                           model="clair_obscur/sdxl-turbo", cost=0.0)
    monkeypatch.setattr("paintress.generate", _gen)

    model = ImageModelInfo(name="clair_obscur/sdxl-turbo", provider="clair_obscur",
                           location="local", endpoint="", vram_mb=4500)
    pick = Pick(model=model, min_time_seconds=0.0)
    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}
    res = await husam_run(task)
    assert res["path"].endswith(".png")
    # bump may not fire in this fast-mock path, but the keepalive context
    # must not raise — that's the contract this test enforces.


@pytest.mark.asyncio
async def test_cloud_image_path_skips_handover(monkeypatch, tmp_path):
    """Sanity: cloud pick must NOT touch shutdown, clair_obscur, or record_swap."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick

    touched = {"unload": False, "start": False, "swap": False}

    class _Mgr:
        async def shutdown(self): touched["unload"] = True
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())
    async def _co_start():
        touched["start"] = True; return ""
    monkeypatch.setattr("clair_obscur.start", _co_start)
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": touched.__setitem__("swap", True))

    async def _gen(pick, spec):
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "x.png"),
                           provider="pollinations", model="pollinations/flux",
                           cost=0.0)
    monkeypatch.setattr("paintress.generate", _gen)

    model = ImageModelInfo(name="pollinations/flux", provider="pollinations",
                           location="cloud", endpoint="https://x/", vram_mb=0)
    pick = Pick(model=model, min_time_seconds=0.0)
    task = {"context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                       "out_dir": str(tmp_path)}},
            "kind": "image", "preselected_pick": pick}
    await husam_run(task)
    assert touched == {"unload": False, "start": False, "swap": False}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/husam/tests/test_husam_image_local.py -q`
Expected: FAIL — handover branch absent (cloud test already passes; the local-ordering test fails because the unload/poll/start/swap sequence doesn't exist yet).

- [ ] **Step 3: Extend `_run_image` — handover INSIDE the existing `keepalive()`**

In `packages/husam/src/husam/worker.py`, in `_run_image`, the existing block at lines 113-114 is:
```python
            async with _hb.keepalive():
                result = await paintress.generate(pick, s)
```

REPLACE that block so the local handover runs INSIDE the same `keepalive()` span, BEFORE `paintress.generate`. `_run_image` already imports `time as _time` and `ModelCallFailed` (`from src.core.router import ModelCallFailed`); add `import asyncio` lazily inside the branch if not already present:
```python
            async with _hb.keepalive():
                if getattr(pick.model, "is_local", False):
                    import asyncio
                    # 1. Free VRAM by unloading any current local LLM. shutdown()
                    #    is internally guarded (no-op if nothing loaded); DaLLaMa
                    #    lazy-reloads on the next LLM task's ensure_model.
                    try:
                        from src.models.local_model_manager import get_local_manager
                        await get_local_manager().shutdown()
                    except Exception as _e:
                        from src.infra.logging_config import get_logger
                        get_logger("husam.image").warning(
                            "local_image: dallama shutdown failed: %s", _e)
                    # 2. Poll free VRAM until the image model fits (or ~30s).
                    try:
                        import nerd_herd
                        deadline = _time.time() + 30.0
                        needed = int(getattr(pick.model, "vram_mb", 0) or 0)
                        while _time.time() < deadline:
                            snap = nerd_herd.refresh_snapshot()
                            if int(getattr(snap, "vram_available_mb", 0) or 0) >= needed:
                                break
                            await asyncio.sleep(0.5)
                    except Exception as _e:
                        from src.infra.logging_config import get_logger
                        get_logger("husam.image").warning(
                            "local_image: vram poll failed: %s", _e)
                    # 3. Start clair_obscur (idempotent; returns base_url).
                    try:
                        import clair_obscur
                        co_base = await clair_obscur.start()
                        try:
                            pick.model.endpoint = co_base
                        except Exception:
                            pass  # paintress local_server falls back to clair_obscur.base_url()
                    except Exception as _e:
                        raise ModelCallFailed(
                            call_id=getattr(pick.model, "name", "image"),
                            last_error=f"clair_obscur_start_failed:{_e}",
                            error_category="availability",
                        )
                    # 4. Record exactly one swap (charge against hoca's budget).
                    try:
                        import nerd_herd
                        nerd_herd.record_swap(getattr(pick.model, "name", ""))
                    except Exception:
                        pass
                result = await paintress.generate(pick, s)
```

Notes:
- There is NO `manager.unload()` and NO `manager.current_model` — `get_local_manager().shutdown()` is the real API and is a safe no-op when nothing is loaded, so no `current_model` guard is needed.
- `ModelCallFailed` is already imported at the top of `_run_image`; do not re-import it.
- The handover is gated on `if getattr(pick.model, "is_local", False):` so a cloud pick falls straight through to `paintress.generate`.

- [ ] **Step 4: Run + regression**

Run: `.venv/Scripts/python -m pytest packages/husam/tests/test_husam_image_local.py packages/husam/tests/ -q -x`
Expected: PASS (Plan 1 v3's cloud-path tests + Plan 2's local tests).

- [ ] **Step 5: Commit**

```bash
git add packages/husam/src/husam/worker.py packages/husam/tests/test_husam_image_local.py
git commit -m "feat(image): husam local handover inside keepalive() envelope"
```

---

## Task 11: Beckman warm-batch hook — calls `record_release_hint()` (not direct stop)

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Test: `packages/general_beckman/tests/test_image_warm_batch.py`

v2 change: the hook calls `clair_obscur.record_release_hint()` on lane switch instead of `await clair_obscur.stop()` directly. The backstop in clair_obscur (Task 4) then times the actual stop after `idle_release_seconds`. This wires the backstop and gives a back-to-back image batch a window to reuse the warm server (husam `_run_image`'s idempotent `clair_obscur.start()` clears the hint if a new image task lands within the window).

For the **warm-batch case** (next admittable task is another local image), the hook does nothing — clair_obscur stays warm because no hint was recorded.

For the **emergency stop** case (wrapper shutdown, OOM), callers can still invoke `clair_obscur.stop()` directly — beckman's hook just doesn't use that path under normal lane switches.

Also (v2 carry-over): admission stamps `task["preselected_pick_provider"]` so the hook can read it without re-loading the pick.

`_peek_next_admittable` v2: rather than hand-rolled SQL that diverges from `next_task()`, use beckman's own `_queue.pick_ready_top_k(k=1)` (recon-confirmed it returns candidates without claiming).

- [ ] **Step 1: Write the failing test**

```python
# packages/general_beckman/tests/test_image_warm_batch.py
import pytest


@pytest.mark.asyncio
async def test_local_image_followed_by_image_no_hint(monkeypatch):
    """Two local image tasks back-to-back → NO release hint → clair_obscur
    stays warm. The next image dispatch will idempotently clear any pending
    hint (in husam `_run_image`, Task 10)."""
    import general_beckman as gb

    hints = {"n": 0}
    stops = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)

    async def _peek():
        return {"id": 2, "kind": "image", "agent_type": "image",
                "context": '{"image_call": {"raw_dispatch": true}}'}
    monkeypatch.setattr(gb, "_peek_next_admittable", _peek, raising=False)

    await gb._post_completion_image_lane({
        "id": 1, "kind": "image", "agent_type": "image",
        "context": '{"image_call": {"raw_dispatch": true}}',
        "preselected_pick_provider": "clair_obscur",
    }, {"status": "completed"})
    assert hints["n"] == 0, "warm-batch must NOT hint release"
    assert stops["n"] == 0, "warm-batch must NOT direct-stop"


@pytest.mark.asyncio
async def test_local_image_followed_by_llm_hints_release(monkeypatch):
    """Lane switch → record_release_hint (NOT direct stop). Backstop times
    the actual stop after idle_release_seconds."""
    import general_beckman as gb
    hints = {"n": 0}
    stops = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)

    async def _peek():
        return {"id": 3, "kind": "llm", "agent_type": "coder",
                "context": "{}"}
    monkeypatch.setattr(gb, "_peek_next_admittable", _peek, raising=False)

    await gb._post_completion_image_lane({
        "id": 1, "kind": "image", "agent_type": "image",
        "context": '{"image_call": {"raw_dispatch": true}}',
        "preselected_pick_provider": "clair_obscur",
    }, {"status": "completed"})
    assert hints["n"] == 1
    assert stops["n"] == 0, "lane switch hints; backstop fires the stop"


@pytest.mark.asyncio
async def test_cloud_image_never_touches_clair_obscur(monkeypatch):
    """A cloud-image task must not call record_release_hint or stop."""
    import general_beckman as gb
    hints = {"n": 0}; stops = {"n": 0}
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)

    await gb._post_completion_image_lane({
        "id": 4, "kind": "image", "agent_type": "image",
        "context": "{}", "preselected_pick_provider": "pollinations",
    }, {"status": "completed"})
    assert hints["n"] == 0 and stops["n"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_image_warm_batch.py -q`
Expected: FAIL.

- [ ] **Step 3: Add the helpers + hook**

In `packages/general_beckman/src/general_beckman/__init__.py`:

Add a peek helper that mirrors `next_task`'s candidate fetch (recon: `_queue.pick_ready_top_k(k=1, lane=None)`):
```python
async def _peek_next_admittable() -> dict | None:
    """Read-only peek at the highest-priority ready candidate. Mirrors
    next_task()'s candidate fetch — uses the same _queue.pick_ready_top_k
    so the prediction matches what beckman will admit next."""
    try:
        candidates = await _queue.pick_ready_top_k(k=1, lane=None)
        if not candidates:
            return None
        return candidates[0]
    except Exception:
        return None
```

Add the lane-switch hook:
```python
async def _post_completion_image_lane(task: dict, result: dict) -> None:
    """After an image task finishes, decide whether to keep clair_obscur warm
    or hint release. Only acts when the just-finished task was a LOCAL image
    (preselected_pick_provider == 'clair_obscur'). Cloud-image tasks left
    clair_obscur untouched — nothing to release.

    v2: on lane switch, calls clair_obscur.record_release_hint() — NOT a
    direct stop. The backstop in clair_obscur.server.ImageServer fires the
    actual stop() after idle_release_seconds. This wires the backstop and
    gives a back-to-back image batch a window to reuse the warm server."""
    if (task.get("kind") or "").lower() != "image":
        return
    if task.get("preselected_pick_provider") != "clair_obscur":
        return

    nxt = await _peek_next_admittable()
    is_image_next = (
        nxt is not None
        and ((nxt.get("kind") or "").lower() == "image"
             or "image_call" in (nxt.get("context") or ""))
    )
    if is_image_next:
        # Warm-batch: NO hint. Idempotent start() in the next dispatch
        # also clears any stale hint.
        return

    # Lane switch — hint release; backstop times the actual stop.
    try:
        import clair_obscur
        clair_obscur.record_release_hint()
    except Exception as _e:
        from src.infra.logging_config import get_logger
        get_logger("beckman.image_lane").warning(
            "clair_obscur.record_release_hint failed", error=str(_e),
        )
```

At admission (Plan 1 v2's `_handle_admission_pick` call site, after `task["preselected_pick"] = pick`), stamp the provider:
```python
        task["preselected_pick"] = pick
        try:
            task["preselected_pick_provider"] = getattr(pick.model, "provider", "")
        except Exception:
            pass
```

Wire the hook into `on_task_finished`. Find the existing post-completion sequence (after `await apply_actions(task, actions)`) and add:
```python
    # Image-lane warm-batch decision (Plan 2).
    try:
        await _post_completion_image_lane(task, result or {})
    except Exception as _e:
        log.debug("image_lane hook failed", task_id=task_id, error=str(_e))
```

- [ ] **Step 4: Run + regression**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_image_warm_batch.py packages/general_beckman/tests/ -q -x`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py packages/general_beckman/tests/test_image_warm_batch.py
git commit -m "feat(image): beckman warm-batch hook drives clair_obscur backstop"
```

---

## Task 12: End-to-end local lane host-path test (mock backend)

**Files:**
- Test: `tests/integration/test_image_local_e2e.py`

Drives the full local lane in one test via the Plan 1 v3 manual-pump pattern — `next_task()` → `await husam.run(task)` → `on_task_finished(tid, result)`: a mock A1111 HTTP server (or ComfyUI — Plan 2 uses A1111 because the mock is simpler) → forced local pick via snapshot override → `husam._run_image` unloads the local LLM via `get_local_manager().shutdown()` (mock) → clair_obscur starts (mock via monkeypatch) → paintress.local_server hits mock → PNG written → record_swap fires once → telemetry round-trip rows land. The handover is driven inside `husam._run_image` (Task 10), not the dispatcher.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_image_local_e2e.py
import asyncio
import base64
import io
import json
import os

import pytest
from PIL import Image


def _png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (180, 90, 70)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_local_image_lane_e2e(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "7860")
    fake_exe = tmp_path / "fake_exe"; fake_exe.write_text("x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(fake_exe))

    # Snapshot: idle, plenty of VRAM, no LLM, no image-server-yet. Then
    # post-dispatch the snapshot should reflect resident=True via clair_obscur.
    class _Snap:
        vram_available_mb = 8000
        in_flight_calls = []
        class local: model_name = None; requests_processing = 0
        class queue_profile: total_ready_count = 0
        image_server_resident = False
        image_server_vram_mb = 0
    monkeypatch.setattr("fatih_hoca.image_select._snapshot", lambda: _Snap())
    monkeypatch.setattr("nerd_herd.refresh_snapshot", lambda: _Snap())

    class _Mgr:
        async def shutdown(self):
            pass  # no-op; handover proceeds to start()
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())
    async def _co_start(): return "http://127.0.0.1:7860"
    monkeypatch.setattr("clair_obscur.start", _co_start)
    swaps = []
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": swaps.append(name))

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"images": [_png_b64()], "info": "{\"seed\": 33}"}
    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw):
            assert "/sdapi/v1/txt2img" in url
            return _Resp()
    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)

    import fatih_hoca
    import husam

    # Force the local pick (snapshot override above makes clair_obscur eligible).
    pick = fatih_hoca.select(needs_image=True, quality_tier="fast")
    assert pick.model.provider == "clair_obscur"
    task = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a fox in snow",
            "out_dir": str(tmp_path), "width": 512, "height": 512, "seed": 33,
            "filename_hint": "fox", "quality_tier": "fast",
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    # Manual pump (Plan 1 v3 pattern): husam.run drives the handover inside
    # husam._run_image (Task 10). on_task_finished closes the telemetry round-trip.
    res = await husam.run(task)
    assert os.path.isfile(res["path"])
    assert os.path.getsize(res["path"]) > 0
    assert res["provider"] == "clair_obscur"
    assert res["seed_used"] == 33
    assert swaps == ["clair_obscur/sdxl-turbo"]  # exactly one swap recorded
```

> If the test wants the full beckman round-trip rather than a bare `husam.run`, use the canonical manual pump: seed the task via `add_task`/`enqueue` (no `await_inline`) → `tid = await next_task()` → `result = await husam.run(task)` → `await on_task_finished(tid, result)`, mirroring `tests/test_beckman_next_task.py`. The handover is still driven inside `husam._run_image`; beckman only pumps.

- [ ] **Step 2: Run + smoke**

Run: `.venv/Scripts/python -m pytest tests/integration/test_image_local_e2e.py -q`
Expected: PASS.

Full Plan 2 suite green-check (split per Plan 1 v2 §13 conftest-collision rule):
```
.venv/Scripts/python -m pytest packages/clair_obscur/tests packages/paintress/tests/test_local_server.py packages/fatih_hoca/tests/test_image_providers_local.py packages/fatih_hoca/tests/test_image_select_eviction.py packages/nerd_herd/tests/test_image_server_state.py packages/general_beckman/tests/test_image_warm_batch.py packages/husam/tests/test_husam_image_local.py -q
.venv/Scripts/python -m pytest tests/integration/test_image_local_e2e.py -q
```
Expected: all green. Plan 1 v3's `tests/integration/test_image_e2e.py` also still green.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_image_local_e2e.py
git commit -m "test(image): end-to-end local lane host-path (mock A1111)"
```

---

## Plan 2 v2 done-when

- `clair_obscur` package builds, installs editable (with psutil declared in pyproject), holds a PID-lock at `logs/image_server.lock`, reconciles orphans by **cmdline match against the configured backend** (NEVER process-name sweeps).
- paintress `local_server` provider speaks ComfyUI (`/prompt` + history poll + `/view`) and A1111 (`/sdapi/v1/txt2img`).
- hoca image catalog has `clair_obscur/sdxl-turbo` at quality_rank 7.5, vram_mb 4500; cloud entries untouched.
- `image_select._eviction_cost` reads `nerd_herd.refresh_snapshot()` (NOT a non-existent `nerd_herd.snapshot()`) and implements the four-arm formula with `_WARM_BATCH_BONUS=1.0`; VRAM-fit gate refuses local when free + recoverable < `model.vram_mb`.
- Husam's `_run_image` handovers GPU on `is_local` picks (`get_local_manager().shutdown()` → poll free VRAM → `clair_obscur.start()` → `record_swap`) **inside the existing `heartbeat.keepalive()`** so the 300s watchdog stays satisfied through the cold-start window. Plan 1 v3's telemetry envelope (begin/end_call, pick_log, tokens, cost) still surrounds.
- Beckman's `on_task_finished` peeks the queue via `_queue.pick_ready_top_k(k=1)`: holds clair_obscur warm when next is another image, calls `clair_obscur.record_release_hint()` on lane switch; clair_obscur's backstop fires the actual stop after `idle_release_seconds`. **No dead code** — the backstop is the normal-path stop trigger.
- E2E host-path test drives the full local lane against mocks; PNG written; one swap recorded; no real GPU.
- All new tests green; no regressions.

## Follow-on (Plan 3)
- i2p prototype `swap_placeholder_images` mechanical + prompt-writing coulson task + asset serving — Plan 3 (file-disjoint from Plan 2).
