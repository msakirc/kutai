# Image Generation — Plan 2: Local `clair_obscur` + GPU handover

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill in the local half of the image-generation lane. After Plan 1 merged the cloud spine (paintress + renoir + image scorer + dispatcher image branch + `/image`), Plan 2 adds a local image-server wrapper (`clair_obscur`), wires it through paintress as the `local_server` provider, registers a local SDXL entry in the hoca image catalog, replaces Plan 1's stub eviction-cost with the real formula reading nerd_herd, performs the dispatcher GPU handover (`dallama.unload()` → poll free VRAM → `clair_obscur.start()` → `paintress.generate()` → record-swap), and adds beckman warm-batch awareness so a sequence of consecutive local image tasks doesn't ping-pong llama-server.

**Architecture:** `clair_obscur` is the image-world `dallama` — a thin async process wrapper around ComfyUI (default) or AUTOMATIC1111 (env flag), holding a PID-lock at `image_server.lock`, reconciling its own orphan on boot (NEVER touches llama-server, per CLAUDE.md), reporting VRAM residency to `nerd_herd` so hoca's eviction-cost has live state to read. The local handover is mechanical follow-through: hoca picks `clair_obscur/sdxl-...`, the dispatcher unloads dallama, polls free VRAM, starts clair_obscur, calls paintress (which now has a `local_server` provider), records ONE swap. Beckman keeps clair_obscur warm across a batch of consecutive image picks, releases on lane switch; a clair_obscur idle-timeout is the safety backstop.

**Tech Stack:** Python 3.10, async/await, httpx (HTTP to ComfyUI/A1111), aiosqlite (existing), pytest (with `aiohttp` test fixture for the mock ComfyUI server). Package layout mirrors `packages/dallama/` src-layout.

**Scope boundary (in this plan):** `clair_obscur` package · paintress `local_server` provider · hoca local SDXL catalog entry · real eviction-cost in `image_select.py` (reads nerd_herd) · VRAM-fit eligibility gate · nerd_herd `image_server_resident` + `image_server_vram_mb` fields + `record_image_server_state` · dispatcher local-image handover (unload → poll → start → record-swap) · beckman post-completion warm-batch hook + clair_obscur idle backstop · e2e host-path test with a mock ComfyUI HTTP server.
**NOT in this plan (Plan 3):** i2p prototype `swap_placeholder_images` mechanical · prompt-writing coulson task + templates · asset serving wired into the web-preview host · ComfyUI/A1111 actually installed on the dev box · live GPU process in CI.

**Inviolable rules (CLAUDE.md):**
- `clair_obscur.start()` / `.stop()` / boot-orphan-reconcile MUST target **only** the image-server backend's PID (ComfyUI's `python.exe`/`main.py` or A1111's `webui.py` / `launch.py`, via the PID written to its own `image_server.lock`). It MUST NEVER call `taskkill /F /IM llama-server.exe` or anything that could hit the wrapper.
- Plan 2 MUST NOT modify Yaşar Usta or the wrapper. The dispatcher path only touches dallama (LLM-side) and clair_obscur (image-side).

---

## File structure

**New package:**
- `packages/clair_obscur/pyproject.toml`
- `packages/clair_obscur/src/clair_obscur/__init__.py` — public module-level API (`start()`, `stop()`, `status()`, `available()`, `base_url()`, `record_release_hint()`).
- `packages/clair_obscur/src/clair_obscur/config.py` — `ClairObscurConfig` dataclass (backend, port, model, weights dir, exe path), env-driven factory.
- `packages/clair_obscur/src/clair_obscur/server.py` — `ImageServer` class with lifecycle, PID-lock, orphan-reconcile, health-poll, idle backstop.
- `packages/clair_obscur/tests/test_config.py`, `test_server_lifecycle.py`, `test_orphan_reconcile.py`, `test_idle_backstop.py`

**New file in paintress:**
- `packages/paintress/src/paintress/providers/local_server.py`
- `packages/paintress/tests/test_local_server.py`

**Modified (single-anchor append/replace each — Plan 2 owns these specific edits):**
- `packages/fatih_hoca/src/fatih_hoca/image_providers.py` — APPEND one `ImageModelInfo` for `clair_obscur/sdxl-turbo`.
- `packages/fatih_hoca/src/fatih_hoca/image_select.py` — replace `_eviction_cost` body (real formula); extend `_provider_available` (add clair_obscur arm); add VRAM-fit eligibility gate to the loop.
- `packages/nerd_herd/src/nerd_herd/types.py:242-256` — add `image_server_resident: bool = False` + `image_server_vram_mb: int = 0` to `SystemSnapshot`.
- `packages/nerd_herd/src/nerd_herd/__init__.py` — add module-level `record_image_server_state(resident, vram_mb)` mirroring `record_swap`'s shape.
- `packages/nerd_herd/src/nerd_herd/nerd_herd.py` — add `push_image_server_state`/internal field + include both new fields in the `snapshot()` builder.
- `src/core/llm_dispatcher.py` — extend `_dispatch_image` with `if pick.model.is_local:` branch (unload → poll → start → record-swap) + error-path cleanup.
- `packages/paintress/src/paintress/__init__.py` — APPEND one entry to the existing `_PROVIDERS` dict: `"clair_obscur": LocalServerProvider()`.
- `packages/general_beckman/src/general_beckman/__init__.py` — add post-completion warm-batch hook inside `on_task_finished` (peek queue → decide keep-warm vs release).
- root `conftest.py` `_PACKAGE_SRCS` — append `clair_obscur` (same place `safety_guard` was added in `ae004547`).

**Test infra:**
- `tests/integration/test_image_local_e2e.py` — host-path e2e against a mock ComfyUI HTTP server (no real GPU).

---

## Task 1: `clair_obscur` config + scaffolding

**Files:**
- Create: `packages/clair_obscur/pyproject.toml`, `packages/clair_obscur/src/clair_obscur/__init__.py`, `.../config.py`
- Test: `packages/clair_obscur/tests/test_config.py`
- Modify: root `conftest.py` (`_PACKAGE_SRCS`)

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_config.py
import os
import pytest
from clair_obscur.config import ClairObscurConfig, load_config


def test_default_backend_is_comfyui(monkeypatch):
    monkeypatch.delenv("CLAIR_OBSCUR_BACKEND", raising=False)
    cfg = load_config()
    assert cfg.backend == "comfyui"
    assert cfg.port == 8188
    assert cfg.base_url == "http://127.0.0.1:8188"


def test_env_selects_a1111(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "7860")
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

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'clair_obscur'`.

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
dependencies = ["httpx>=0.27"]

[tool.setuptools.packages.find]
where = ["src"]
```

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
    backend: str           # "comfyui" | "a1111"
    host: str              # "127.0.0.1"
    port: int              # 8188 (comfyui) / 7860 (a1111) defaults
    base_url: str          # full URL; env CLAIR_OBSCUR_URL overrides host:port
    model: str             # SDXL / SD1.5 model filename or repo id
    weights_dir: str       # absolute path to the backend's models directory
    exe_path: str          # absolute path to the launcher (python -m comfyui / webui.bat)
    idle_release_seconds: int = 60   # safety backstop if beckman never signals release


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

Lifecycle: start() / stop() / status() / available() / base_url(). Holds a
PID-lock at logs/image_server.lock and reconciles its own orphan (ComfyUI /
A1111 process — NEVER llama-server, per CLAUDE.md)."""
from __future__ import annotations

from .config import ClairObscurConfig, load_config
from .server import ImageServer

__all__ = [
    "ClairObscurConfig", "load_config", "ImageServer",
    "start", "stop", "status", "available", "base_url",
    "record_release_hint", "get_singleton",
]

# Module-level singleton so dispatcher and beckman share one ImageServer
# without threading a reference through call sites.
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
    """Beckman tells clair_obscur it MAY release (lane switch)."""
    get_singleton().record_release_hint()
```

(Server class lands in Task 2; this module compiles against `ImageServer` defined there.)

- [ ] **Step 4: Register the package**

In root `conftest.py`, append to `_PACKAGE_SRCS`:
```python
    _ROOT / "packages" / "clair_obscur" / "src",
```
And add `"clair_obscur"` to the eviction set on lines 48-53 (next to `safety_guard`).

Run: `.venv/Scripts/python -m pip install -e packages/clair_obscur`
Expected: `Successfully installed clair_obscur-0.1.0`.

The config test imports `clair_obscur.config` directly, not the package init, so the missing `ImageServer` in Task 1 doesn't break it. (If pytest collection imports `__init__` eagerly, Task 2 makes it whole; if you hit `ImportError: ImageServer` first, jump Task 2 then return.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_config.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add packages/clair_obscur/pyproject.toml packages/clair_obscur/src/clair_obscur/__init__.py packages/clair_obscur/src/clair_obscur/config.py packages/clair_obscur/tests/test_config.py conftest.py
git commit -m "feat(image): clair_obscur package scaffold + config"
```

---

## Task 2: `ImageServer` lifecycle (start / stop / health-poll / status)

**Files:**
- Create: `packages/clair_obscur/src/clair_obscur/server.py`
- Test: `packages/clair_obscur/tests/test_server_lifecycle.py`

Mirrors the dallama `ServerProcess` shape (`packages/dallama/src/dallama/server.py`): launch the backend as a subprocess, poll a health endpoint until ready, expose `is_alive()` and `base_url`. The dallama platform layer's `kill_orphans` template (`packages/dallama/src/dallama/platform.py:214`) is copied for **clair_obscur's own backend exe only** — never llama-server.

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_server_lifecycle.py
import asyncio
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path, **overrides):
    base = dict(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(tmp_path / "fake_exe.bat"),
        idle_release_seconds=60,
    )
    base.update(overrides)
    return ClairObscurConfig(**base)


@pytest.mark.asyncio
async def test_status_when_not_started(tmp_path):
    s = ImageServer(_cfg(tmp_path))
    st = s.status()
    assert st["resident"] is False
    assert st["pid"] is None


@pytest.mark.asyncio
async def test_start_polls_health_until_ready(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path))

    # 1) Skip the real subprocess spawn — pretend we launched it.
    async def _fake_launch():
        s._pid = 12345  # type: ignore[attr-defined]
    monkeypatch.setattr(s, "_launch_process", _fake_launch)

    # 2) Health probe says "down" twice, then "up".
    calls = {"n": 0}
    async def _fake_health():
        calls["n"] += 1
        return calls["n"] >= 3
    monkeypatch.setattr(s, "_health_probe", _fake_health)

    # 3) Skip lock + orphan reconcile (covered in Task 3).
    monkeypatch.setattr(s, "_acquire_lock", lambda: None)
    monkeypatch.setattr(s, "_reconcile_orphan", lambda: None)
    monkeypatch.setattr(s, "_notify_nerd_herd_resident", lambda vram_mb=4500: None)

    url = await s.start()
    assert url == "http://127.0.0.1:8188"
    assert calls["n"] >= 3
    assert s.status()["resident"] is True


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
    monkeypatch.setattr(s, "_health_timeout_seconds", 0.5)

    with pytest.raises(TimeoutError):
        await s.start()


@pytest.mark.asyncio
async def test_stop_releases_and_notifies(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path))
    s._pid = 12345  # type: ignore[attr-defined]
    s._resident = True  # type: ignore[attr-defined]
    killed = {"pid": None}
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("pid", pid))
    notified = {"resident": None}
    monkeypatch.setattr(s, "_notify_nerd_herd_resident",
                        lambda vram_mb=0: notified.__setitem__("resident", False))

    await s.stop()
    assert killed["pid"] == 12345
    assert s.status()["resident"] is False
    assert s.status()["pid"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_server_lifecycle.py -q`
Expected: FAIL — `ModuleNotFoundError: ...server`.

- [ ] **Step 3: Implement `ImageServer`**

`packages/clair_obscur/src/clair_obscur/server.py`:
```python
"""ImageServer — async lifecycle wrapper around ComfyUI/A1111.

Mirrors dallama's ServerProcess shape: launch subprocess → poll health →
expose base_url + status. PID-locked at logs/image_server.lock; on boot
reconciles a stale lock pointing at a still-alive backend process from a
prior crash.

CRITICAL (CLAUDE.md): _kill_own_pid + _reconcile_orphan target ONLY the
image-server backend's PID (read from image_server.lock) or its exe name
(matching expected backend, e.g. comfyui's `python.exe -m main` or a1111's
`webui.py`). They MUST NEVER call taskkill on llama-server.exe or anything
else KutAI didn't spawn.
"""
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
        """Backend installed (exe path resolves) + config sane."""
        if not self._config.exe_path:
            return False
        return os.path.exists(self._config.exe_path)

    def base_url(self) -> str:
        return self._config.base_url

    def status(self) -> dict:
        return {
            "resident": self._resident,
            "pid": self._pid,
            "backend": self._config.backend,
            "base_url": self._config.base_url,
        }

    async def start(self) -> str:
        if self._resident and self._pid is not None:
            return self._config.base_url   # already up; idempotent
        self._reconcile_orphan()
        await self._launch_process()
        self._acquire_lock()
        deadline = time.time() + self._health_timeout_seconds
        while time.time() < deadline:
            if await self._health_probe():
                self._resident = True
                self._notify_nerd_herd_resident(
                    vram_mb=self._estimated_resident_vram_mb()
                )
                self._arm_idle_backstop()
                return self._config.base_url
            await asyncio.sleep(self._health_poll_interval)
        # Health never came up → tear down and surface a TimeoutError.
        if self._pid is not None:
            try:
                self._kill_own_pid(self._pid)
            finally:
                self._pid = None
        self._release_lock()
        raise TimeoutError(
            f"clair_obscur {self._config.backend} did not become healthy "
            f"within {self._health_timeout_seconds}s"
        )

    async def stop(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
        pid = self._pid
        self._pid = None
        self._resident = False
        if pid is not None:
            self._kill_own_pid(pid)
        self._release_lock()
        self._notify_nerd_herd_resident(vram_mb=0)

    def record_release_hint(self) -> None:
        """Beckman tells us a release is acceptable. Reset idle timer."""
        self._release_hint_at = time.time()

    # ───────── Internals (test-mockable) ─────────
    async def _launch_process(self) -> None:
        """Spawn the backend. Implementation note: ComfyUI's launcher is
        `python main.py --listen 127.0.0.1 --port {port}`; A1111 is
        `webui-user.bat` / `launch.py --api --listen --port {port}`.
        On Windows we attach to a Job Object via the same template
        dallama uses (packages/dallama/src/dallama/platform.py)."""
        cmd = self._build_launch_cmd()
        # subprocess.Popen — fire-and-forget; PID captured for our lock.
        import subprocess
        creationflags = 0
        if _IS_WINDOWS:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(  # noqa: S603 — caller-supplied exe path
            cmd, creationflags=creationflags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._pid = proc.pid

    def _build_launch_cmd(self) -> list[str]:
        cfg = self._config
        if cfg.backend == "comfyui":
            return [
                cfg.exe_path, "-u", "main.py",
                "--listen", cfg.host, "--port", str(cfg.port),
            ]
        # a1111
        return [
            cfg.exe_path,
            "--api", "--listen",
            "--port", str(cfg.port),
            "--nowebui",
        ]

    async def _health_probe(self) -> bool:
        url = self._health_url()
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                resp = await c.get(url)
                return 200 <= resp.status_code < 500   # 401 still means "up"
        except Exception:
            return False

    def _health_url(self) -> str:
        if self._config.backend == "comfyui":
            # ComfyUI exposes /system_stats once the server is up.
            return f"{self._config.base_url}/system_stats"
        # A1111: /sdapi/v1/sd-models is a cheap auth-free liveness probe.
        return f"{self._config.base_url}/sdapi/v1/sd-models"

    def _acquire_lock(self) -> None:
        if self._pid is None:
            return
        try:
            os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
            with open(_LOCK_PATH, "w", encoding="utf-8") as f:
                f.write(f"{self._pid}\n{self._config.backend}\n")
        except Exception:
            # Lock is advisory; failure to write must not crash boot.
            pass

    def _release_lock(self) -> None:
        try:
            if os.path.exists(_LOCK_PATH):
                os.remove(_LOCK_PATH)
        except Exception:
            pass

    def _reconcile_orphan(self) -> None:
        """If image_server.lock from a prior crash points at a live image-
        server backend process, kill it. Verifies the PID's exe name
        matches the expected backend so we NEVER hit a co-tenant
        (especially llama-server)."""
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
        """True iff `pid` is alive AND its exe name matches our expected
        backend launcher. This is the safety net: it prevents the orphan
        sweep from ever touching llama-server or any other resident
        process whose PID happened to land in our stale lock."""
        if pid <= 0:
            return False
        try:
            import psutil  # nerd_herd already depends on psutil indirectly
            if not psutil.pid_exists(pid):
                return False
            p = psutil.Process(pid)
            exe = (p.name() or "").lower()
            cmdline = " ".join(p.cmdline()).lower()
        except Exception:
            return False

        if self._config.backend == "comfyui":
            return "main.py" in cmdline or "comfyui" in cmdline
        # a1111
        return "webui" in cmdline or "launch.py" in cmdline

    def _kill_own_pid(self, pid: int) -> None:
        """Kill ONLY the given PID. NOT a broad taskkill /IM — we always
        kill by PID we wrote into image_server.lock ourselves (or
        verified via _is_own_backend_pid). This keeps the blast radius
        to one process and rules out llama-server collateral."""
        try:
            import psutil
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except Exception:
            # Best-effort; nothing else to clean up.
            pass

    def _estimated_resident_vram_mb(self) -> int:
        """How much VRAM clair_obscur is expected to consume once resident.
        Hand-set from the catalog entry for now — the actual measurement
        will come from nerd_herd's GPU collector once an LLM-style live
        signal exists. SDXL-Turbo fp16 weights are ~4.5 GB; activations
        push to ~5 GB at 1024×1024."""
        return 4500

    def _notify_nerd_herd_resident(self, vram_mb: int) -> None:
        """Push residency state to nerd_herd so hoca's eviction-cost reads
        it. Best-effort; nerd_herd may not be wired in unit tests."""
        try:
            import nerd_herd
            nerd_herd.record_image_server_state(
                resident=(vram_mb > 0), vram_mb=vram_mb,
            )
        except Exception:
            pass

    def _arm_idle_backstop(self) -> None:
        """Schedule a safety idle-stop. Beckman drives release via
        record_release_hint(); the backstop ensures we don't hold the
        GPU forever if beckman's hook is dead."""
        if self._idle_task is not None and not self._idle_task.done():
            return
        async def _watcher():
            cfg = self._config
            while self._resident:
                await asyncio.sleep(min(15.0, cfg.idle_release_seconds))
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
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/clair_obscur/src/clair_obscur/server.py packages/clair_obscur/tests/test_server_lifecycle.py
git commit -m "feat(image): clair_obscur ImageServer lifecycle (start/stop/health)"
```

---

## Task 3: PID-lock + boot orphan-reconcile

**Files:**
- Modify: `packages/clair_obscur/src/clair_obscur/server.py` (already-stubbed `_reconcile_orphan` + `_kill_own_pid`)
- Test: `packages/clair_obscur/tests/test_orphan_reconcile.py`

Task 2 wrote the orphan-reconcile internals. Task 3 verifies the safety contract: the sweep MUST not kill a PID whose exe is anything other than the configured backend, and it MUST never reach for llama-server.

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_orphan_reconcile.py
import os
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer, _LOCK_PATH


def _cfg(tmp_path, backend="comfyui"):
    return ClairObscurConfig(
        backend=backend, host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(tmp_path / "exe"), idle_release_seconds=60,
    )


def _write_lock(pid: int, backend: str):
    os.makedirs(os.path.dirname(_LOCK_PATH) or ".", exist_ok=True)
    with open(_LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(f"{pid}\n{backend}\n")


def _clear_lock():
    try: os.remove(_LOCK_PATH)
    except FileNotFoundError: pass


def test_orphan_killed_when_lock_points_at_live_comfyui(monkeypatch, tmp_path):
    _clear_lock()
    _write_lock(12345, "comfyui")
    killed = {"pid": None}
    s = ImageServer(_cfg(tmp_path))
    monkeypatch.setattr(s, "_is_own_backend_pid", lambda pid: pid == 12345)
    monkeypatch.setattr(s, "_kill_own_pid",
                        lambda pid: killed.__setitem__("pid", pid))
    s._reconcile_orphan()
    assert killed["pid"] == 12345
    assert not os.path.exists(_LOCK_PATH)


def test_orphan_skipped_when_pid_is_not_own_backend(monkeypatch, tmp_path):
    """SAFETY: if the PID in image_server.lock no longer belongs to a
    ComfyUI/A1111 process — say, it now belongs to llama-server or any
    other tenant — the sweep MUST NOT kill it."""
    _clear_lock()
    _write_lock(67890, "comfyui")
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
    _clear_lock()
    _write_lock(11111, "a1111")
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
```

- [ ] **Step 2: Run to verify it fails or passes incidentally**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_orphan_reconcile.py -q`
Expected: likely already PASS — Task 2's `_reconcile_orphan` shape matches the tests. If any case fails (e.g. lock path setup), fix `_reconcile_orphan` per the failure message rather than the tests (the contract is the safety property, not the impl).

- [ ] **Step 3: If any test fails, harden `_reconcile_orphan`**

The contract Task 2 must already satisfy:
- Lock backend mismatch → no kill, lock deleted (stale).
- `_is_own_backend_pid(pid)` False → no kill, lock deleted.
- No lock file → no-op, no exceptions.

Re-read `server.py:_reconcile_orphan` and patch any divergence. (Most likely: missing lock-mismatch guard or a stray broad `_kill_own_pid` call.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_orphan_reconcile.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/clair_obscur/src/clair_obscur/server.py packages/clair_obscur/tests/test_orphan_reconcile.py
git commit -m "test(image): clair_obscur orphan-reconcile safety contract"
```

---

## Task 4: Idle-release safety backstop

**Files:**
- Modify (only if needed): `packages/clair_obscur/src/clair_obscur/server.py:_arm_idle_backstop`
- Test: `packages/clair_obscur/tests/test_idle_backstop.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_idle_backstop.py
import asyncio
import time
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path, idle_seconds=0.2):
    return ClairObscurConfig(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(tmp_path / "exe"), idle_release_seconds=idle_seconds,
    )


@pytest.mark.asyncio
async def test_release_hint_then_idle_triggers_stop(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path, idle_seconds=0.2))
    s._resident = True
    s._pid = 555
    stopped = {"n": 0}
    async def _fake_stop():
        stopped["n"] += 1
        s._resident = False
        s._pid = None
    monkeypatch.setattr(s, "stop", _fake_stop)

    # Tune the watcher tick to be fast for test.
    monkeypatch.setattr("clair_obscur.server.asyncio.sleep",
                        lambda t: asyncio.sleep(0.05))

    s._arm_idle_backstop()
    s.record_release_hint()
    # Wait longer than idle_release_seconds (0.2s).
    await asyncio.sleep(0.5)

    assert stopped["n"] == 1


@pytest.mark.asyncio
async def test_no_release_hint_no_stop(monkeypatch, tmp_path):
    s = ImageServer(_cfg(tmp_path, idle_seconds=0.2))
    s._resident = True
    s._pid = 555
    stopped = {"n": 0}
    async def _fake_stop():
        stopped["n"] += 1
        s._resident = False
    monkeypatch.setattr(s, "stop", _fake_stop)
    monkeypatch.setattr("clair_obscur.server.asyncio.sleep",
                        lambda t: asyncio.sleep(0.05))

    s._arm_idle_backstop()
    # No record_release_hint() called — must NOT stop.
    await asyncio.sleep(0.5)

    assert stopped["n"] == 0
    s._resident = False  # let watcher loop exit
```

- [ ] **Step 2: Run test to verify it fails (or surface a fix needed)**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_idle_backstop.py -q`
Expected: likely PASS already; if test_no_release_hint_no_stop fails because the watcher checks `hint is None` incorrectly, that's the bug — fix the guard.

- [ ] **Step 3: Fix `_arm_idle_backstop` if needed**

Confirm the watcher condition is:
```python
if hint is not None and (time.time() - hint) >= cfg.idle_release_seconds:
    await self.stop()
```
Not `if hint is None or ...`. (The hint exists to mean "beckman gave permission"; absent hint → keep warm indefinitely until beckman calls stop directly.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_idle_backstop.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/clair_obscur/src/clair_obscur/server.py packages/clair_obscur/tests/test_idle_backstop.py
git commit -m "feat(image): clair_obscur idle-release safety backstop"
```

---

## Task 5: nerd_herd `image_server_resident` + `image_server_vram_mb` fields

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py:242-256` (SystemSnapshot)
- Modify: `packages/nerd_herd/src/nerd_herd/nerd_herd.py:140-166` (`snapshot()` builder + `push_image_server_state`)
- Modify: `packages/nerd_herd/src/nerd_herd/__init__.py` (module-level `record_image_server_state`)
- Test: `packages/nerd_herd/tests/test_image_server_state.py`

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


def test_record_image_server_state_flips_snapshot():
    nh = nerd_herd.NerdHerd()
    nh.push_image_server_state(resident=True, vram_mb=4500)
    snap = nh.snapshot()
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb == 4500
    nh.push_image_server_state(resident=False, vram_mb=0)
    snap = nh.snapshot()
    assert snap.image_server_resident is False
    assert snap.image_server_vram_mb == 0


def test_module_level_record_image_server_state(monkeypatch):
    # The singleton wired by record_image_server_state should be visible
    # via the module-level snapshot() call too.
    nerd_herd.record_image_server_state(resident=True, vram_mb=4500)
    # snapshot() may default to an empty SystemSnapshot when no client is
    # wired — accept either, but if a NerdHerdClient IS wired the new state
    # must be visible.
    snap = nerd_herd.snapshot()
    assert hasattr(snap, "image_server_resident")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/test_image_server_state.py -q`
Expected: FAIL — `AttributeError: 'SystemSnapshot' has no attribute 'image_server_resident'`.

- [ ] **Step 3: Extend `SystemSnapshot`**

In `packages/nerd_herd/src/nerd_herd/types.py`, in the `SystemSnapshot` dataclass (line ~242-256), after the `recent_swap_count` field add:
```python
    # Image-server residency (clair_obscur). Read by fatih_hoca's
    # image_select._eviction_cost: when resident, eviction is 0 (already
    # warm; consecutive image picks pay no swap penalty). Written via
    # NerdHerd.push_image_server_state() / module-level
    # record_image_server_state(), driven by clair_obscur start/stop.
    image_server_resident: bool = False
    image_server_vram_mb: int = 0
```

- [ ] **Step 4: Add `push_image_server_state` on NerdHerd**

In `packages/nerd_herd/src/nerd_herd/nerd_herd.py`, near `push_in_flight` (line ~132), add:
```python
    def push_image_server_state(self, *, resident: bool, vram_mb: int) -> None:
        """Replace image-server residency state (pushed by clair_obscur on
        start/stop). Read by hoca's image_select eviction-cost formula."""
        self._image_server_resident = bool(resident)
        self._image_server_vram_mb = int(vram_mb or 0)
```

In `__init__` (around line ~80, near other state fields), initialize:
```python
        self._image_server_resident: bool = False
        self._image_server_vram_mb: int = 0
```
(Match the exact init style of `_local_state`/`_cloud_state`. If they use a `_state_lock` or similar pattern, follow it.)

In `snapshot()` (line ~159-166), extend the `SystemSnapshot(...)` constructor call to include the two new fields:
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

- [ ] **Step 5: Add module-level `record_image_server_state`**

In `packages/nerd_herd/src/nerd_herd/__init__.py`, after `record_swap` (line ~73), add:
```python
def record_image_server_state(*, resident: bool, vram_mb: int) -> None:
    """Record clair_obscur (local image-server) residency. Called by
    clair_obscur on start/stop. Read by hoca's image_select eviction-cost
    formula via SystemSnapshot.image_server_resident."""
    _get_singleton().push_image_server_state(resident=resident, vram_mb=vram_mb)
```

And add `"record_image_server_state"` to the `__all__` list.

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/test_image_server_state.py -q`
Expected: PASS (3 passed).

- [ ] **Step 7: Regression**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/ -q -x`
Expected: no new failures vs baseline.

- [ ] **Step 8: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/src/nerd_herd/nerd_herd.py packages/nerd_herd/src/nerd_herd/__init__.py packages/nerd_herd/tests/test_image_server_state.py
git commit -m "feat(image): nerd_herd image_server_resident + vram_mb"
```

---

## Task 6: clair_obscur ↔ nerd_herd wiring on start/stop

**Files:**
- (already touched in Task 2) `packages/clair_obscur/src/clair_obscur/server.py:_notify_nerd_herd_resident`
- Test: `packages/clair_obscur/tests/test_nerd_herd_wiring.py`

Task 2 stubbed `_notify_nerd_herd_resident`; Task 5 added the recipient. This task asserts the round-trip.

- [ ] **Step 1: Write the failing test**

```python
# packages/clair_obscur/tests/test_nerd_herd_wiring.py
import pytest

from clair_obscur.config import ClairObscurConfig
from clair_obscur.server import ImageServer


def _cfg(tmp_path):
    return ClairObscurConfig(
        backend="comfyui", host="127.0.0.1", port=8188,
        base_url="http://127.0.0.1:8188",
        model="sdxl-turbo", weights_dir=str(tmp_path),
        exe_path=str(tmp_path / "exe"), idle_release_seconds=60,
    )


@pytest.mark.asyncio
async def test_start_pushes_resident_true_to_nerd_herd(monkeypatch, tmp_path):
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
    # NerdHerd singleton's snapshot should now report residency.
    nh = nerd_herd._get_singleton()
    snap = nh.snapshot()
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb > 0


@pytest.mark.asyncio
async def test_stop_pushes_resident_false(monkeypatch, tmp_path):
    import nerd_herd
    nerd_herd.record_image_server_state(resident=True, vram_mb=4500)

    s = ImageServer(_cfg(tmp_path))
    s._pid = 222
    s._resident = True
    monkeypatch.setattr(s, "_kill_own_pid", lambda pid: None)
    await s.stop()

    nh = nerd_herd._get_singleton()
    snap = nh.snapshot()
    assert snap.image_server_resident is False
    assert snap.image_server_vram_mb == 0
```

- [ ] **Step 2: Run to verify it fails or already passes**

Run: `.venv/Scripts/python -m pytest packages/clair_obscur/tests/test_nerd_herd_wiring.py -q`
Expected: most likely PASS already (Task 2 calls `nerd_herd.record_image_server_state`, Task 5 implemented it). If the test fails because of import-order in the singleton (e.g. `_notify_nerd_herd_resident` swallows the exception silently), surface the wiring by ensuring `nerd_herd.record_image_server_state` is invoked in `start()` AFTER `_resident=True` and in `stop()` AFTER pid clear.

- [ ] **Step 3: Verify + commit**

```bash
git add packages/clair_obscur/tests/test_nerd_herd_wiring.py
git commit -m "test(image): clair_obscur ↔ nerd_herd start/stop wiring"
```

---

## Task 7: paintress `local_server` provider (ComfyUI + A1111 backends)

**Files:**
- Create: `packages/paintress/src/paintress/providers/local_server.py`
- Modify: `packages/paintress/src/paintress/__init__.py` — APPEND `"clair_obscur": LocalServerProvider()` to `_PROVIDERS`
- Test: `packages/paintress/tests/test_local_server.py`

The provider reads `clair_obscur.load_config()` to discover which backend protocol to use (ComfyUI `/prompt` + history poll, vs A1111 `/sdapi/v1/txt2img`), so paintress doesn't re-declare config.

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
    spec = ImageSpec(prompt="a cat", out_dir="/tmp", seed=99, width=512, height=512)
    data, meta = await prov.generate(spec, base_url="http://127.0.0.1:7860")
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

    class _HistoryEmpty:
        status_code = 200
        def json(self): return {}
        def raise_for_status(self): pass

    class _HistoryDone:
        status_code = 200
        def json(self):
            return {"abc-123": {
                "outputs": {"9": {"images": [{
                    "filename": "out.png", "subfolder": "", "type": "output",
                }]}}}}
        def raise_for_status(self): pass

    class _ImageBytesResp:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nFAKE_BYTES"
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
                return _HistoryEmpty() if state["history_calls"] < 2 else _HistoryDone()
            if "/view" in url:
                state["viewed"] = True
                return _ImageBytesResp()
            raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("paintress.providers.local_server.httpx.AsyncClient", _Client)
    monkeypatch.setattr("paintress.providers.local_server.asyncio.sleep",
                        lambda t: __import__("asyncio").sleep(0))

    prov = LocalServerProvider()
    spec = ImageSpec(prompt="a dog", out_dir="/tmp", seed=7, width=512, height=512)
    data, meta = await prov.generate(spec, base_url="http://127.0.0.1:8188")
    assert state["posted"] and state["viewed"]
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 7


def test_available_reflects_clair_obscur(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(tmp_path / "no_such_exe"))
    assert LocalServerProvider().available() is False
    exe = tmp_path / "exe"; exe.write_text("x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(exe))
    assert LocalServerProvider().available() is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_local_server.py -q`
Expected: FAIL — `ModuleNotFoundError: ...local_server`.

- [ ] **Step 3: Implement the provider**

`packages/paintress/src/paintress/providers/local_server.py`:
```python
"""LocalServerProvider — paintress provider that calls clair_obscur's
local backend (ComfyUI default, A1111 via env). Reads the backend from
clair_obscur.load_config() so paintress doesn't duplicate config logic."""
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
        """Mirrors clair_obscur.available(). Returns False when no backend
        exe is configured — hoca then filters the local entry out."""
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
            # Fallback to clair_obscur's configured URL (dispatcher normally
            # passes it through pick.model.endpoint, but tests / direct calls
            # may omit it).
            try:
                from clair_obscur import base_url as _co_base_url
                base_url = _co_base_url()
            except Exception:
                base_url = "http://127.0.0.1:8188"
        if backend == "a1111":
            return await self._a1111(spec, base_url)
        return await self._comfyui(spec, base_url)

    # ───────── A1111 ─────────
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
        # `info` is a JSON string with the actual seed used.
        info = body.get("info") or "{}"
        try:
            info_d = json.loads(info)
            seed_used = info_d.get("seed", spec.seed)
        except Exception:
            seed_used = spec.seed
        return data, {"seed_used": seed_used}

    # ───────── ComfyUI ─────────
    async def _comfyui(self, spec: ImageSpec, base_url: str) -> Tuple[bytes, dict]:
        base = base_url.rstrip("/")
        workflow = self._build_comfyui_workflow(spec)
        async with httpx.AsyncClient(timeout=_PROMPT_TIMEOUT) as c:
            resp = await c.post(f"{base}/prompt", json={"prompt": workflow})
            resp.raise_for_status()
            prompt_id = (resp.json() or {}).get("prompt_id")
            if not prompt_id:
                raise RuntimeError("comfyui_no_prompt_id")
            # Poll /history/{id} until the output node has an image.
            deadline = asyncio.get_event_loop().time() + _PROMPT_TIMEOUT
            image_meta = None
            while asyncio.get_event_loop().time() < deadline:
                h = await c.get(f"{base}/history/{prompt_id}")
                h.raise_for_status()
                hist = h.json() or {}
                entry = hist.get(prompt_id)
                if entry and entry.get("outputs"):
                    for _node_id, out in entry["outputs"].items():
                        imgs = out.get("images") or []
                        if imgs:
                            image_meta = imgs[0]
                            break
                if image_meta is not None:
                    break
                await asyncio.sleep(_PROMPT_POLL_INTERVAL)
            if image_meta is None:
                raise RuntimeError("comfyui_timeout")
            # Fetch the actual bytes via /view.
            params = {
                "filename": image_meta["filename"],
                "subfolder": image_meta.get("subfolder", ""),
                "type": image_meta.get("type", "output"),
            }
            v = await c.get(f"{base}/view", params=params)
            v.raise_for_status()
            return v.content, {"seed_used": spec.seed}

    def _build_comfyui_workflow(self, spec: ImageSpec) -> dict:
        """Minimal SDXL/SD1.5 workflow. Real ComfyUI deployments use the UI
        to author + export a workflow JSON; this is the smallest hand-built
        equivalent that works against the default ComfyUI install."""
        seed = int(spec.seed) if spec.seed is not None else 0
        steps = int(spec.steps) if spec.steps else 20
        return {
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": steps, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal",
                "denoise": 1.0,
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
            "6": {"class_type": "CLIPTextEncode", "inputs": {
                "text": spec.prompt or "", "clip": ["4", 1],
            }},
            "7": {"class_type": "CLIPTextEncode", "inputs": {
                "text": spec.negative_prompt or "", "clip": ["4", 1],
            }},
            "8": {"class_type": "VAEDecode", "inputs": {
                "samples": ["3", 0], "vae": ["4", 2],
            }},
            "9": {"class_type": "SaveImage", "inputs": {
                "filename_prefix": "kutai", "images": ["8", 0],
            }},
        }
```

- [ ] **Step 4: Register the provider in paintress**

In `packages/paintress/src/paintress/__init__.py`, after `from .providers.huggingface import HuggingFaceProvider` (Plan 1 line ~), add:
```python
from .providers.local_server import LocalServerProvider
```
And in the `_PROVIDERS` dict, APPEND one entry (Plan 1 created the dict with `pollinations` + `huggingface`):
```python
_PROVIDERS = {
    "pollinations": PollinationsProvider(),
    "huggingface": HuggingFaceProvider(),
    "clair_obscur": LocalServerProvider(),
}
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest packages/paintress/tests/test_local_server.py packages/paintress/tests/test_dispatch.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add packages/paintress/src/paintress/providers/local_server.py packages/paintress/src/paintress/__init__.py packages/paintress/tests/test_local_server.py
git commit -m "feat(image): paintress local_server provider (clair_obscur)"
```

---

## Task 8: Add local SDXL entry to fatih_hoca image catalog

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/image_providers.py` (APPEND one entry)
- Test: `packages/fatih_hoca/tests/test_image_providers_local.py`

The local entry uses `name="clair_obscur/sdxl-turbo"`. SDXL-Turbo fp16 weights are ~4.5 GB on disk; loaded resident with activations at 1024×1024 lands ~5 GB. We set `vram_mb=4500` as a defensible footprint estimate — under the 6 GB budget that remains on an 8 GB GPU after llama-server unloads, after desktop+browser overhead. `endpoint=""` (set at dispatch time from `clair_obscur.base_url()`).

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
    assert co.cost_per_image == 0.0
    assert co.vram_mb >= 3000  # SDXL footprint floor
    assert co.vram_mb <= 6000  # below the 8GB - LLM-unload headroom


def test_cloud_entries_still_present():
    """Plan 2 must NOT remove or reorder Plan 1's cloud entries."""
    names = {m.name for m in image_catalog()}
    assert "pollinations/flux" in names
    assert "huggingface/flux-schnell" in names
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_providers_local.py -q`
Expected: FAIL — `clair_obscur/sdxl-turbo` not in catalog.

- [ ] **Step 3: Append the local entry**

In `packages/fatih_hoca/src/fatih_hoca/image_providers.py`, in `image_catalog()`'s returned list, APPEND a new `ImageModelInfo` AFTER the existing two cloud entries (do not remove or reorder):
```python
        ImageModelInfo(
            name="clair_obscur/sdxl-turbo", provider="clair_obscur",
            location="local",
            endpoint="",          # set at dispatch time from clair_obscur.base_url()
            quality_rank=7.5,     # below HF flux-schnell (8.0); above pollinations (6.0)
            cost_per_image=0.0,
            vram_mb=4500,         # SDXL-Turbo fp16 + activations @ 1024×1024
            supports_seed=True,
            tier="local",
        ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_providers_local.py packages/fatih_hoca/tests/test_image_providers.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/image_providers.py packages/fatih_hoca/tests/test_image_providers_local.py
git commit -m "feat(image): hoca image catalog adds clair_obscur/sdxl-turbo"
```

---

## Task 9: Real eviction-cost + VRAM-fit gate in image_select

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/image_select.py` — replace `_eviction_cost` body, extend `_provider_available` (clair_obscur arm), add VRAM-fit eligibility gate inside the loop
- Test: `packages/fatih_hoca/tests/test_image_select_eviction.py`

Per spec §4 the formula is:
```
if image_server_resident:          eviction = 0
elif llm_in_flight > 0:            eviction = HUGE   = 100.0
elif llm_loaded or llm_queue > 0:  eviction = HIGH   = 50.0
else (GPU idle):                   eviction = LOW    = 2.0
```
With quality ranks ~6.0 (pollinations), ~8.0 (HF), ~7.5 (clair_obscur), the math:
- HUGE=100 → cloud always wins under load (7.5 − 100 = −92.5; HF stays at 8.0). ✓
- HIGH=50 → cloud still wins when LLM loaded (7.5 − 50 = −42.5; HF stays at 8.0). ✓
- LOW=2 → idle: 7.5 − 2 = 5.5 < 8.0; HF still beats local. Local wins only over the budget-exhausted / unavailable case OR when local is mid-batch (resident=True, eviction=0, score 7.5 ≤ 8.0 — HF still wins). The fix: **resident → eviction=0** AND we add a small batch bonus so that once warm, the local batch keeps running rather than swapping back to cloud per image:
- Once resident, local's effective score is 7.5 + WARM_BATCH_BONUS (=1.0) = 8.5 > 8.0 → batching falls out naturally.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_image_select_eviction.py
import pytest

from fatih_hoca.image_select import select_image
from fatih_hoca.types import Pick


def _snapshot(*, llm_in_flight=0, llm_loaded=False, llm_queue=0,
              image_resident=False, vram_available_mb=6000):
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
        vram_available_mb = vram_available_mb
    return _S()


def test_huge_eviction_when_llm_in_flight(monkeypatch):
    """LLM mid-call → image task MUST go cloud (eviction=100 dominates)."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snapshot(llm_in_flight=1))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.provider in ("huggingface", "pollinations")
    assert pick.model.is_local is False


def test_high_eviction_when_llm_loaded_but_idle(monkeypatch):
    """LLM loaded, no in-flight → still cloud (HIGH=50 dominates)."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snapshot(llm_loaded=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False


def test_low_eviction_when_idle_still_favors_cloud_first_call(monkeypatch):
    """Cold GPU + HF available → HF still wins (8.0 > 7.5 − 2.0)."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snapshot())
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"


def test_resident_image_server_wins_batched(monkeypatch):
    """Image-server already warm → local wins (eviction=0 + warm bonus)."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snapshot(image_resident=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.provider == "clair_obscur"


def test_local_filtered_when_vram_too_low(monkeypatch):
    """vram_available_mb < model.vram_mb → local is ineligible, cloud wins."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snapshot(image_resident=False, vram_available_mb=2000))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False


def test_local_filtered_when_clair_obscur_unavailable(monkeypatch):
    """No exe configured → clair_obscur ineligible, cloud-only."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snapshot(image_resident=False, vram_available_mb=8000))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.delenv("CLAIR_OBSCUR_EXE", raising=False)
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_select_eviction.py -q`
Expected: FAIL — Plan 1's `_eviction_cost` returns 0.0 unconditionally, so the local entry would win whenever it's eligible; the LLM-in-flight test fails first.

- [ ] **Step 3: Replace `_eviction_cost`, extend `_provider_available`, add VRAM-fit gate**

In `packages/fatih_hoca/src/fatih_hoca/image_select.py`:

Replace the `_eviction_cost(m)` function with the snapshot-reading version:
```python
# Eviction-cost magnitudes — calibrated against quality_rank spread of 6.0
# (pollinations) → 8.0 (HF) → 7.5 (clair_obscur/sdxl). HUGE/HIGH must
# dominate the ~2-point quality spread; LOW must not (so cold-start
# always tries cloud first when both are eligible).
_EVICTION_HUGE = 100.0
_EVICTION_HIGH = 50.0
_EVICTION_LOW = 2.0
_WARM_BATCH_BONUS = 1.0   # local jumps cloud once image-server is resident


def _snapshot():
    """Thin wrapper so tests can monkeypatch a SystemSnapshot-like object."""
    try:
        import nerd_herd
        return nerd_herd.snapshot()
    except Exception:
        from nerd_herd.types import SystemSnapshot
        return SystemSnapshot()


def _eviction_cost(m: ImageModelInfo, snap=None) -> float:
    if not m.is_local:
        return 0.0
    s = snap if snap is not None else _snapshot()
    if getattr(s, "image_server_resident", False):
        return 0.0
    in_flight = len(getattr(s, "in_flight_calls", []) or [])
    # Backward-compat: legacy snapshot shape used local.requests_processing
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


def _warm_batch_bonus(m: ImageModelInfo, snap) -> float:
    if not m.is_local:
        return 0.0
    return _WARM_BATCH_BONUS if getattr(snap, "image_server_resident", False) else 0.0
```

Extend `_provider_available` with the clair_obscur arm:
```python
def _provider_available(m: ImageModelInfo, hf_available: bool | None) -> bool:
    if m.provider == "huggingface":
        return os.getenv("HF_TOKEN") is not None if hf_available is None else hf_available
    if m.provider == "pollinations":
        return True
    if m.provider == "clair_obscur":
        # clair_obscur.available() reads CLAIR_OBSCUR_EXE; absent exe → False.
        # We import lazily so the package can be missing on test envs that
        # don't install clair_obscur.
        try:
            import clair_obscur
            return bool(clair_obscur.available())
        except Exception:
            exe = os.getenv("CLAIR_OBSCUR_EXE", "")
            return bool(exe) and os.path.exists(exe)
    return False
```

Add the VRAM-fit gate inside `select_image`'s candidate loop (mirroring `selector.py:459` `excluded` shape — refuse before scoring):
```python
def select_image(*, ...):
    failed = set(failures or [])
    snap = _snapshot()
    candidates: list[tuple[float, ImageModelInfo]] = []
    for m in image_catalog():
        if m.name in failed:
            continue
        if not _provider_available(m, hf_available):
            continue
        # VRAM-fit eligibility for local entries (mirrors selector's
        # needs_vision gate at packages/fatih_hoca/src/fatih_hoca/selector.py:459).
        if m.is_local and m.vram_mb > 0:
            free_mb = int(getattr(snap, "vram_available_mb", 0) or 0)
            if free_mb > 0 and free_mb < m.vram_mb:
                # llama-server holding all VRAM doesn't disqualify — the
                # dispatcher unloads it on dispatch. We only refuse when
                # the GPU genuinely cannot fit the model after the
                # known-budget add-back of any LLM that will be unloaded.
                # Conservative fix: also consider the LOCAL LLM's footprint
                # as recoverable.
                local = getattr(snap, "local", None)
                local_recoverable = 0
                if local and getattr(local, "model_name", None):
                    # Conservative recoverable estimate: assume the local
                    # LLM holds 4 GB (fits the LLMs in current models.yaml
                    # registry — Qwen2.5-7B ~5 GB, Phi-3-mini ~3 GB; this
                    # is the average and avoids hard-coding registry data
                    # into the selector).
                    local_recoverable = 4000
                if (free_mb + local_recoverable) < m.vram_mb:
                    continue
        if remaining_budget_usd is not None and m.cost_per_image > remaining_budget_usd:
            continue
        score = m.quality_rank
        score -= _eviction_cost(m, snap)
        score += _warm_batch_bonus(m, snap)
        candidates.append((score, m))

    if not candidates:
        return SelectionFailure(reason="availability",
                                detail="no eligible image provider")
    candidates.sort(key=lambda t: t[0], reverse=True)
    best = candidates[0][1]
    return Pick(model=best, min_time_seconds=0.0, score=candidates[0][0],
                top_summary="; ".join(f"{m.name}:{s:.1f}" for s, m in candidates[:5]))
```

Add `from .image_providers import image_catalog` (already present), and import `os` if not already.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_image_select_eviction.py packages/fatih_hoca/tests/test_image_select.py -q`
Expected: PASS (10 passed — Plan 1's 4 + Plan 2's 6).

- [ ] **Step 5: Regression**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/ -q -x`
Expected: no new failures vs baseline.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/image_select.py packages/fatih_hoca/tests/test_image_select_eviction.py
git commit -m "feat(image): real eviction-cost + VRAM-fit gate in image_select"
```

---

## Task 10: Dispatcher local-image handover (unload → poll → start → record-swap)

**Files:**
- Modify: `src/core/llm_dispatcher.py` — extend `_dispatch_image` with `if pick.model.is_local:` branch
- Test: `tests/core/test_dispatcher_image_local.py`

Plan 1's `_dispatch_image` calls `paintress.generate(pick, spec)` unconditionally. Plan 2 wraps the call with the GPU handover when `pick.model.is_local`:
1. `local_model_manager.get_local_manager().unload()` (the `dallama.stop()` equivalent — confirm import; this codebase uses `src.models.local_model_manager` from `llm_dispatcher.py:851,975` — read those lines once to choose the canonical entry point).
2. Poll `nerd_herd.snapshot().vram_available_mb` until `>= pick.model.vram_mb` (or timeout).
3. `await clair_obscur.start()` — returns `base_url`.
4. Inject `base_url` into the spec (so paintress's `local_server` provider uses it). Set `pick.model.endpoint = base_url` (or pass via `image_call`).
5. `nerd_herd.record_swap(pick.model.name)` — counts as ONE swap against hoca's swap budget (per spec §6).
6. `await paintress.generate(pick, spec)`.
7. On error after start: do NOT stop clair_obscur (beckman's warm-batch decides). On error before start (unload/poll): no state to clean.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_dispatcher_image_local.py
import pytest


@pytest.mark.asyncio
async def test_local_image_unloads_llm_then_starts_clair_obscur(monkeypatch, tmp_path):
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick
    from src.core.llm_dispatcher import get_dispatcher

    order = []

    class _Mgr:
        current_model = "qwen2.5-7b"
        async def unload(self):
            order.append("unload")
            _Mgr.current_model = None
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())

    class _Snap:
        vram_available_mb = 7000
        in_flight_calls = []
        class local: model_name = None; requests_processing = 0
        class queue_profile: total_ready_count = 0
        image_server_resident = False
        image_server_vram_mb = 0
    monkeypatch.setattr("nerd_herd.snapshot", lambda: _Snap())

    swap_calls = []
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": swap_calls.append(name))

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
                           model="clair_obscur/sdxl-turbo",
                           cost=0.0, seed_used=7)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    model = ImageModelInfo(
        name="clair_obscur/sdxl-turbo", provider="clair_obscur",
        location="local", endpoint="", quality_rank=7.5,
        cost_per_image=0.0, vram_mb=4500, supports_seed=True,
    )
    pick = Pick(model=model, min_time_seconds=0.0)
    spec = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a fox", "out_dir": str(tmp_path),
            "width": 512, "height": 512,
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    res = await get_dispatcher().dispatch(spec)
    assert res["path"].endswith(".png")
    assert order == ["unload", "clair_obscur.start", "paintress.generate"]
    assert swap_calls == ["clair_obscur/sdxl-turbo"]


@pytest.mark.asyncio
async def test_cloud_image_path_unchanged(monkeypatch, tmp_path):
    """Sanity: a cloud pick MUST NOT touch dallama or clair_obscur."""
    from fatih_hoca.registry import ImageModelInfo
    from fatih_hoca.types import Pick
    from src.core.llm_dispatcher import get_dispatcher

    touched = {"unload": False, "co_start": False, "swap": False}

    class _Mgr:
        async def unload(self): touched["unload"] = True
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())
    async def _co_start():
        touched["co_start"] = True; return ""
    monkeypatch.setattr("clair_obscur.start", _co_start)
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": touched.__setitem__("swap", True))

    async def _fake_generate(pick, spec):
        from paintress import ImageResult
        return ImageResult(path=str(tmp_path / "out.png"),
                           provider="pollinations",
                           model="pollinations/flux", cost=0.0)
    monkeypatch.setattr("paintress.generate", _fake_generate)

    model = ImageModelInfo(name="pollinations/flux", provider="pollinations",
                           location="cloud", endpoint="https://x/", vram_mb=0)
    pick = Pick(model=model, min_time_seconds=0.0)
    spec = {
        "context": {"image_call": {"raw_dispatch": True, "prompt": "x",
                                   "out_dir": str(tmp_path)}},
        "kind": "image",
        "preselected_pick": pick,
    }
    await get_dispatcher().dispatch(spec)
    assert touched == {"unload": False, "co_start": False, "swap": False}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_dispatcher_image_local.py -q`
Expected: FAIL — `local` branch not present in `_dispatch_image`; order list will be `["paintress.generate"]` only.

- [ ] **Step 3: Extend `_dispatch_image`**

In `src/core/llm_dispatcher.py`, in the `_dispatch_image` method (Plan 1 created), at the top of the method after extracting the `pick`, BEFORE the `paintress.generate` call, add:

```python
        # Local handover: unload llama, wait for VRAM, start clair_obscur,
        # record-swap (counts as 1 against hoca's swap budget). Cloud picks
        # skip this entirely.
        if getattr(pick.model, "is_local", False):
            # 1. Free VRAM.
            try:
                from src.models.local_model_manager import get_local_manager
                manager = get_local_manager()
                if getattr(manager, "current_model", None):
                    await manager.unload()
            except Exception as _e:
                # Non-fatal: if unload fails, we still attempt to fit.
                logger.warning("local_image: dallama unload failed: %s", _e)
            # 2. Poll vram_available_mb until model fits (or timeout).
            try:
                import nerd_herd
                deadline = time.time() + 30.0
                needed = int(getattr(pick.model, "vram_mb", 0) or 0)
                while time.time() < deadline:
                    snap = nerd_herd.snapshot()
                    if int(getattr(snap, "vram_available_mb", 0) or 0) >= needed:
                        break
                    await asyncio.sleep(0.5)
            except Exception as _e:
                logger.warning("local_image: vram poll failed: %s", _e)
            # 3. Start clair_obscur (returns base_url; idempotent if already up).
            try:
                import clair_obscur
                co_base = await clair_obscur.start()
                # Inject base_url into the pick so paintress's local_server
                # provider sees it via getattr(model, "endpoint").
                try:
                    pick.model.endpoint = co_base
                except Exception:
                    # ImageModelInfo is a dataclass — `frozen=False` default
                    # allows attribute assignment. If a future change freezes
                    # it, fall through; paintress also reads from
                    # clair_obscur.base_url() as a fallback.
                    pass
            except Exception as _e:
                from src.core.router import ModelCallFailed
                raise ModelCallFailed(
                    call_id=getattr(pick.model, "name", "image"),
                    last_error=f"clair_obscur_start_failed:{_e}",
                    error_category="availability",
                )
            # 4. Record-swap: one swap charge per local image dispatch
            #    (spec §6 — batching is rewarded because resident=True
            #    flips eviction-cost to 0).
            try:
                import nerd_herd
                nerd_herd.record_swap(getattr(pick.model, "name", ""))
            except Exception:
                pass
```

(Confirm the exact unload method name on `LocalModelManager` via `grep -n "async def unload\|def unload" src/models/local_model_manager.py` and adjust the `await manager.unload()` call if it's named differently. If `local_model_manager.py` is itself a shim re-export to `packages/dallama/`, follow the chain.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_dispatcher_image_local.py tests/core/test_dispatcher_image.py -q`
Expected: PASS (3 passed — Plan 1's cloud test + Plan 2's two local tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/llm_dispatcher.py tests/core/test_dispatcher_image_local.py
git commit -m "feat(image): dispatcher local-image handover (unload → start → record-swap)"
```

---

## Task 11: Beckman post-completion warm-batch hook

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` — extend `on_task_finished` with a post-completion image-lane release decision
- Test: `packages/general_beckman/tests/test_image_warm_batch.py`

After an image task completes, peek the queue: if the next admittable task is ALSO a local image AND it out-prioritizes any waiting LLM task, hold clair_obscur warm (call `clair_obscur.record_release_hint()` is NOT triggered — meaning idle backstop stays unarmed). If the next task is an LLM (or queue empty), call `await clair_obscur.stop()` so dallama can lazy-reload on the next LLM task.

Anchor: `on_task_finished` body, after `route_result → rewrite → apply_actions` (around line 913 in the package) but before the progress-ping block.

- [ ] **Step 1: Write the failing test**

```python
# packages/general_beckman/tests/test_image_warm_batch.py
import pytest


@pytest.mark.asyncio
async def test_local_image_followed_by_image_keeps_warm(monkeypatch):
    """Two local image tasks back-to-back → clair_obscur stays up."""
    import general_beckman as gb

    stops = {"n": 0}
    hints = {"n": 0}
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)
    monkeypatch.setattr("clair_obscur.record_release_hint",
                        lambda: hints.__setitem__("n", hints["n"] + 1))

    # Pretend the next admittable task is another local image with higher
    # priority than any LLM task in the queue.
    async def _peek():
        return {"kind": "image", "agent_type": "image", "priority": 5,
                "context": '{"image_call": {"raw_dispatch": true, "prompt": "x"}}'}
    monkeypatch.setattr(gb, "_peek_next_admittable", _peek, raising=False)

    await gb._post_completion_image_lane({
        "id": 1, "kind": "image", "agent_type": "image",
        "context": '{"image_call": {"raw_dispatch": true}}',
        "preselected_pick_provider": "clair_obscur",
    }, {"status": "completed"})
    assert stops["n"] == 0  # held warm
    assert hints["n"] == 0  # no hint = idle backstop stays unarmed


@pytest.mark.asyncio
async def test_local_image_followed_by_llm_releases(monkeypatch):
    """Image done, next task is an LLM → release clair_obscur so dallama can reload."""
    import general_beckman as gb
    stops = {"n": 0}
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)
    monkeypatch.setattr("clair_obscur.record_release_hint", lambda: None)

    async def _peek():
        return {"kind": "llm", "agent_type": "coder", "priority": 5,
                "context": "{}"}
    monkeypatch.setattr(gb, "_peek_next_admittable", _peek, raising=False)

    await gb._post_completion_image_lane({
        "id": 2, "kind": "image", "agent_type": "image",
        "context": '{"image_call": {"raw_dispatch": true}}',
        "preselected_pick_provider": "clair_obscur",
    }, {"status": "completed"})
    assert stops["n"] == 1


@pytest.mark.asyncio
async def test_cloud_image_never_touches_clair_obscur(monkeypatch):
    """A cloud-image task must NOT call clair_obscur.stop() (it was never started)."""
    import general_beckman as gb
    stops = {"n": 0}
    async def _stop(): stops["n"] += 1
    monkeypatch.setattr("clair_obscur.stop", _stop)
    await gb._post_completion_image_lane({
        "id": 3, "kind": "image", "agent_type": "image",
        "context": "{}",
        "preselected_pick_provider": "pollinations",
    }, {"status": "completed"})
    assert stops["n"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_image_warm_batch.py -q`
Expected: FAIL — `_post_completion_image_lane` / `_peek_next_admittable` don't exist.

- [ ] **Step 3: Add the helpers + hook**

In `packages/general_beckman/src/general_beckman/__init__.py`, near `on_task_finished` (line ~670), add the two helpers and the hook call:

```python
async def _peek_next_admittable() -> dict | None:
    """Peek the highest-priority admittable task in the queue without
    claiming it. Returns the task dict or None.

    Cheap read: replicates next_task's selection ordering but stops after
    finding one row. No claim, no side-effects."""
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        """
        SELECT id, kind, agent_type, priority, context
          FROM tasks
         WHERE status IN ('ready', 'pending')
         ORDER BY priority DESC, created_at ASC
         LIMIT 1
        """
    )
    row = await cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0], "kind": row[1], "agent_type": row[2],
        "priority": row[3], "context": row[4],
    }


async def _post_completion_image_lane(task: dict, result: dict) -> None:
    """After an image task finishes, decide whether to keep clair_obscur
    warm (consecutive local image batch) or release it (lane switch).

    Only acts when the just-finished task was a LOCAL image (clair_obscur
    provider). Cloud-image tasks left clair_obscur untouched, so there is
    nothing to release."""
    if (task.get("kind") or "").lower() != "image":
        return
    provider = task.get("preselected_pick_provider") or ""
    # The provider name is the writer's responsibility — beckman stamps it
    # when it attaches preselected_pick at admission. If absent (legacy
    # rows), bail without touching clair_obscur.
    if provider != "clair_obscur":
        return

    nxt = await _peek_next_admittable()
    is_image_next = (
        nxt is not None
        and ((nxt.get("kind") or "").lower() == "image"
             or "image_call" in (nxt.get("context") or ""))
    )
    if is_image_next:
        # Keep warm. Do NOT call record_release_hint (idle backstop stays
        # unarmed; the next image task will re-arm on its own start path).
        return

    # Lane switch — release so dallama can lazy-reload on the next LLM call.
    try:
        import clair_obscur
        await clair_obscur.stop()
    except Exception as _e:
        from src.infra.logging_config import get_logger
        get_logger("beckman.image_lane").warning(
            "clair_obscur.stop failed", error=str(_e),
        )
```

Also: when beckman admits a task and stamps `preselected_pick`, it must record the provider name so the post-completion hook can read it without re-loading the pick. In the admission section (around `task["preselected_pick"] = pick` near line 608, confirm via `grep -n "preselected_pick" packages/general_beckman/src/general_beckman/__init__.py`):
```python
        task["preselected_pick"] = pick
        # Stamp provider name for the post-completion warm-batch hook
        # (image lane) — avoids re-loading the pick at task-finished time.
        try:
            task["preselected_pick_provider"] = getattr(pick.model, "provider", "")
        except Exception:
            pass
```

Wire the hook into `on_task_finished`. After `await apply_actions(task, actions)` (around line 913), and before the progress-ping `try:` block:
```python
    # Image-lane warm-batch decision (Plan 2).
    try:
        await _post_completion_image_lane(task, result or {})
    except Exception as _e:
        log.debug("image_lane hook failed", task_id=task_id, error=str(_e))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/test_image_warm_batch.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Regression**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests/ -q -x`
Expected: no new failures vs baseline.

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py packages/general_beckman/tests/test_image_warm_batch.py
git commit -m "feat(image): beckman warm-batch hook (clair_obscur keep/release)"
```

---

## Task 12: End-to-end local lane host-path test (mock ComfyUI)

**Files:**
- Test: `tests/integration/test_image_local_e2e.py`

Proves the full local lane in one test: a mock ComfyUI HTTP server (aiohttp) → beckman admits local pick (forced via test setup that disables HF + flips snapshot to LOW eviction → local wins) → dispatcher unloads llama (mock) → clair_obscur starts (mock subprocess) → paintress.local_server hits mock ComfyUI → PNG written → assets logged.

Recurring lesson (Z9-era): unit-green ≠ wired. The host-path test is the integration gate.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_image_local_e2e.py
import asyncio
import base64
import io
import os

import pytest
from PIL import Image


def _png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (180, 90, 70)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_local_image_lane_e2e(monkeypatch, tmp_path):
    # Force HF off so HF is filtered and a1111 mock-server path activates.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "7860")
    fake_exe = tmp_path / "fake_exe"; fake_exe.write_text("x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", str(fake_exe))

    # 1) Force image_select to pick clair_obscur by flipping the snapshot
    #    to LOW eviction + abundant VRAM (so local is eligible and HF is
    #    absent).
    class _Snap:
        vram_available_mb = 8000
        in_flight_calls = []
        class local: model_name = None; requests_processing = 0
        class queue_profile: total_ready_count = 0
        image_server_resident = False
        image_server_vram_mb = 0
    monkeypatch.setattr("fatih_hoca.image_select._snapshot", lambda: _Snap())
    monkeypatch.setattr("nerd_herd.snapshot", lambda: _Snap())

    # 2) Mock the dallama unload + clair_obscur start.
    class _Mgr:
        current_model = "qwen2.5-7b"
        async def unload(self):
            _Mgr.current_model = None
    monkeypatch.setattr("src.models.local_model_manager.get_local_manager",
                        lambda: _Mgr())
    async def _co_start(): return "http://127.0.0.1:7860"
    monkeypatch.setattr("clair_obscur.start", _co_start)
    swaps = []
    monkeypatch.setattr("nerd_herd.record_swap",
                        lambda name="": swaps.append(name))

    # 3) Mock the a1111 HTTP response (paintress local_server uses httpx).
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

    # 4) Drive the pipeline.
    import fatih_hoca
    from src.core.llm_dispatcher import get_dispatcher

    pick = fatih_hoca.select(needs_image=True, quality_tier="fast")
    assert pick.model.provider == "clair_obscur"
    spec = {
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a fox in snow",
            "out_dir": str(tmp_path),
            "width": 512, "height": 512, "seed": 33,
            "filename_hint": "fox",
        }},
        "kind": "image",
        "preselected_pick": pick,
    }
    res = await get_dispatcher().dispatch(spec)

    # 5) Assert the full chain executed.
    assert os.path.isfile(res["path"])
    assert os.path.getsize(res["path"]) > 0
    assert res["provider"] == "clair_obscur"
    assert res["seed_used"] == 33
    assert swaps == ["clair_obscur/sdxl-turbo"]
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python -m pytest tests/integration/test_image_local_e2e.py -q`
Expected: PASS (1 passed).

- [ ] **Step 3: Full new-suite green-check**

Run the Plan 2 tests together (split across the two conftest roots per the Plan 1 §13 note):
```
.venv/Scripts/python -m pytest packages/clair_obscur/tests packages/paintress/tests/test_local_server.py packages/fatih_hoca/tests/test_image_providers_local.py packages/fatih_hoca/tests/test_image_select_eviction.py packages/nerd_herd/tests/test_image_server_state.py packages/general_beckman/tests/test_image_warm_batch.py -q
.venv/Scripts/python -m pytest tests/core/test_dispatcher_image_local.py tests/integration/test_image_local_e2e.py -q
```
Expected: all green.

- [ ] **Step 4: Regression — Plan 1's e2e still green**

Run: `.venv/Scripts/python -m pytest tests/integration/test_image_e2e.py -q`
Expected: PASS — Plan 2's changes do not regress the cloud-only path.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_image_local_e2e.py
git commit -m "test(image): end-to-end local lane host-path test (mock ComfyUI)"
```

---

## Plan 2 done-when

- `clair_obscur` package builds, installs editable, holds a PID-lock at `logs/image_server.lock`, reconciles its own orphan on boot, and NEVER touches llama-server's PID under any tested path.
- paintress `local_server` provider speaks both ComfyUI (`/prompt` + history poll + `/view`) and A1111 (`/sdapi/v1/txt2img`).
- fatih_hoca's image catalog has `clair_obscur/sdxl-turbo` at quality_rank 7.5, vram_mb 4500; cloud entries untouched.
- `image_select._eviction_cost` reads `nerd_herd.snapshot()` and implements the four-arm formula (resident=0 / in_flight=100 / loaded-or-queued=50 / idle=2); VRAM-fit gate refuses local when free VRAM cannot accommodate the model even after dallama unload.
- Dispatcher's `_dispatch_image` handovers GPU on `is_local` picks: unload → poll free VRAM → start clair_obscur → record-swap (1 charge against hoca's swap budget) → paintress.generate; cloud path unchanged.
- Beckman's `on_task_finished` peeks the queue after each image task: holds clair_obscur warm when the next task is another local image, releases (calls `await clair_obscur.stop()`) on lane switch; clair_obscur's idle backstop is the safety net.
- E2E host-path test drives the full local lane against a mock ComfyUI/A1111 HTTP server; PNG written; one swap recorded; no real GPU.
- All new tests green; no regressions in `packages/fatih_hoca/tests/`, `packages/general_beckman/tests/`, `packages/nerd_herd/tests/`, `tests/core/`, `tests/integration/`.

## Follow-on plan (Plan 3, write after Plan 2 executes)

- **Plan 3 — i2p prototype integration:**
  - Prompt-writing coulson task (beckman-admitted full-lifecycle, reads design tokens + screen plan + brand voice, emits enriched diffusion prompts per placeholder).
  - `swap_placeholder_images` mr_roboto mechanical: scan prototype HTML for placeholder `<img>` markers, enqueue prompt-writing → per-placeholder image tasks, write to `mission_{id}/assets/`, rewrite `src`.
  - Wire `mission_{id}/assets/` into the web-preview host (`project_web_preview_hosting_20260522`).
  - i2p workflow JSON: prototype-phase step using the new mechanical, with `done_when` that does NOT hard-require real images (degradation: keep placeholders if generation unavailable).
