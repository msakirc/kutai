# DaLLaMa Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract llama-server process management into `packages/dallama/` as a standalone async Python package with zero KutAI coupling.

**Architecture:** Six internal modules (config, server, swap, watchdog, metrics, platform) composed by a thin `DaLLaMa` main class. Three public methods (`infer`, `keep_alive`, `status`). Host communicates via `on_ready` callback and `get_vram_free_mb` injection. KutAI's `local_model_manager.py` becomes a ~30-line shim, `gpu_scheduler.py` gets deleted.

**Tech Stack:** Python 3.10+, asyncio, httpx, subprocess. No pynvml/psutil/litellm/pydantic.

**Spec:** `docs/superpowers/specs/2026-04-12-dallama-design.md`

**Reference code:** `src/models/local_model_manager.py` (the source being extracted — read it for any implementation detail not fully specified here)

---

### Task 1: Package Skeleton + Config Dataclasses

**Files:**
- Create: `packages/dallama/pyproject.toml`
- Create: `packages/dallama/src/dallama/__init__.py`
- Create: `packages/dallama/src/dallama/config.py`
- Create: `packages/dallama/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/dallama/tests/test_config.py
"""Tests for DaLLaMa config dataclasses."""
import pytest
from dallama.config import (
    DaLLaMaConfig,
    ServerConfig,
    ServerStatus,
    InferenceSession,
    DaLLaMaLoadError,
)


def test_dallama_config_defaults():
    cfg = DaLLaMaConfig()
    assert cfg.llama_server_path == "llama-server"
    assert cfg.port == 8080
    assert cfg.host == "127.0.0.1"
    assert cfg.idle_timeout_seconds == 60.0
    assert cfg.circuit_breaker_threshold == 2
    assert cfg.circuit_breaker_cooldown_seconds == 300.0
    assert cfg.inference_drain_timeout_seconds == 30.0
    assert cfg.health_check_interval_seconds == 30.0
    assert cfg.health_fail_threshold == 3
    assert cfg.min_free_vram_mb == 4096
    assert cfg.on_ready is None
    assert cfg.get_vram_free_mb is None


def test_dallama_config_custom():
    cfg = DaLLaMaConfig(
        llama_server_path="/usr/bin/llama-server",
        port=9090,
        idle_timeout_seconds=120,
        on_ready=lambda m, r: None,
        get_vram_free_mb=lambda: 8000,
    )
    assert cfg.port == 9090
    assert cfg.idle_timeout_seconds == 120
    assert cfg.on_ready is not None
    assert cfg.get_vram_free_mb() == 8000


def test_server_config_minimal():
    sc = ServerConfig(
        model_path="/models/test.gguf",
        model_name="test-model",
        context_length=4096,
    )
    assert sc.thinking is False
    assert sc.vision_projector == ""
    assert sc.extra_flags == []


def test_server_config_full():
    sc = ServerConfig(
        model_path="/models/qwen.gguf",
        model_name="qwen3-30b",
        context_length=16384,
        thinking=True,
        vision_projector="/models/mmproj.gguf",
        extra_flags=["--no-jinja", "--chat-template", "chatml"],
    )
    assert sc.thinking is True
    assert sc.vision_projector == "/models/mmproj.gguf"
    assert len(sc.extra_flags) == 3


def test_server_status_no_model():
    st = ServerStatus(
        model_name=None,
        healthy=False,
        busy=False,
        measured_tps=0.0,
        context_length=0,
    )
    assert st.model_name is None
    assert st.healthy is False


def test_server_status_loaded():
    st = ServerStatus(
        model_name="qwen3-30b",
        healthy=True,
        busy=True,
        measured_tps=12.5,
        context_length=16384,
    )
    assert st.busy is True
    assert st.measured_tps == 12.5


def test_inference_session():
    s = InferenceSession(url="http://127.0.0.1:8080", model_name="test")
    assert s.url == "http://127.0.0.1:8080"
    assert s.model_name == "test"


def test_dallama_load_error():
    err = DaLLaMaLoadError("qwen3-30b")
    assert "qwen3-30b" in str(err)
    assert isinstance(err, RuntimeError)
```

- [ ] **Step 2: Create pyproject.toml**

```toml
# packages/dallama/pyproject.toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "dallama"
version = "0.1.0"
description = "Python async llama-server process manager"
requires-python = ">=3.10"
dependencies = ["httpx>=0.27.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Create __init__.py with public API re-exports**

```python
# packages/dallama/src/dallama/__init__.py
"""DaLLaMa — Python async llama-server process manager."""

from .config import (
    DaLLaMaConfig,
    DaLLaMaLoadError,
    InferenceSession,
    ServerConfig,
    ServerStatus,
)

__all__ = [
    "DaLLaMaConfig",
    "DaLLaMaLoadError",
    "InferenceSession",
    "ServerConfig",
    "ServerStatus",
]
```

- [ ] **Step 4: Implement config.py**

```python
# packages/dallama/src/dallama/config.py
"""Configuration dataclasses for DaLLaMa."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DaLLaMaConfig:
    """Engine settings — configured once at startup."""

    llama_server_path: str = "llama-server"
    port: int = 8080
    host: str = "127.0.0.1"
    idle_timeout_seconds: float = 60.0
    circuit_breaker_threshold: int = 2
    circuit_breaker_cooldown_seconds: float = 300.0
    inference_drain_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 30.0
    health_fail_threshold: int = 3
    min_free_vram_mb: int = 4096
    on_ready: Callable[[str | None, str], None] | None = None
    get_vram_free_mb: Callable[[], int] | None = None


@dataclass
class ServerConfig:
    """Job description — what model to load and how."""

    model_path: str
    model_name: str
    context_length: int
    thinking: bool = False
    vision_projector: str = ""
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class ServerStatus:
    """What the dispatcher needs for routing decisions."""

    model_name: str | None
    healthy: bool
    busy: bool
    measured_tps: float
    context_length: int


@dataclass
class InferenceSession:
    """Context holder yielded by DaLLaMa.infer()."""

    url: str
    model_name: str


class DaLLaMaLoadError(RuntimeError):
    """Raised when DaLLaMa cannot load the requested model."""

    def __init__(self, model_name: str, reason: str = ""):
        msg = f"Failed to load model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.model_name = model_name
        self.reason = reason
```

- [ ] **Step 5: Run tests**

Run: `pytest packages/dallama/tests/test_config.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/dallama/
git commit -m "feat(dallama): package skeleton + config dataclasses"
```

---

### Task 2: Platform Helper

**Files:**
- Create: `packages/dallama/src/dallama/platform.py`
- Create: `packages/dallama/tests/test_platform.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/dallama/tests/test_platform.py
"""Tests for PlatformHelper — OS-specific process management."""
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from dallama.platform import PlatformHelper


@pytest.fixture
def helper():
    return PlatformHelper()


def test_create_process_returns_popen(helper, tmp_path):
    """create_process launches a subprocess and returns a Popen."""
    stderr_path = str(tmp_path / "stderr.log")
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    proc = helper.create_process(cmd, stderr_path)
    assert isinstance(proc, subprocess.Popen)
    assert proc.poll() is None  # still running
    proc.kill()
    proc.wait()


def test_create_process_writes_stderr(helper, tmp_path):
    """stderr output lands in the specified file."""
    stderr_path = str(tmp_path / "stderr.log")
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('hello\\n')"]
    proc = helper.create_process(cmd, stderr_path)
    proc.wait(timeout=5)
    with open(stderr_path) as f:
        assert "hello" in f.read()


@pytest.mark.asyncio
async def test_graceful_stop_terminates(helper, tmp_path):
    """graceful_stop terminates a running process."""
    stderr_path = str(tmp_path / "stderr.log")
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    proc = helper.create_process(cmd, stderr_path)
    assert proc.poll() is None
    await helper.graceful_stop(proc, timeout=5)
    assert proc.poll() is not None


@pytest.mark.asyncio
async def test_graceful_stop_force_kills_on_timeout(helper, tmp_path):
    """If terminate doesn't work within timeout, force kill."""
    stderr_path = str(tmp_path / "stderr.log")
    # Use a process that ignores SIGTERM (traps it on Unix)
    code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    cmd = [sys.executable, "-c", code]
    proc = helper.create_process(cmd, stderr_path)
    await helper.graceful_stop(proc, timeout=2)
    assert proc.poll() is not None


def test_kill_orphans_no_crash(helper):
    """kill_orphans should not raise even if no orphans exist."""
    # Just verify it doesn't raise — no orphans to find
    helper.kill_orphans("nonexistent-process-name-xyz")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/dallama/tests/test_platform.py -v`
Expected: FAIL — `dallama.platform` not found.

- [ ] **Step 3: Implement platform.py**

```python
# packages/dallama/src/dallama/platform.py
"""OS-specific process management.

Windows: Job Objects, CREATE_NO_WINDOW, CTRL_BREAK_EVENT.
Linux/Mac: SIGTERM → SIGKILL.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import subprocess

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"


class PlatformHelper:
    """Abstracts OS-specific subprocess management."""

    def __init__(self):
        self._job_object = self._create_job_object() if _IS_WINDOWS else None

    def create_process(
        self,
        cmd: list[str],
        stderr_path: str,
    ) -> subprocess.Popen:
        """Launch a subprocess with OS-appropriate flags.

        stderr is written to ``stderr_path`` for crash diagnostics.
        On Windows, the process is hidden (no console window) and
        assigned to a Job Object so it dies with the parent.
        """
        creation_flags = 0
        if _IS_WINDOWS:
            creation_flags = subprocess.CREATE_NO_WINDOW

        stderr_file = open(stderr_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            creationflags=creation_flags,
        )
        # Attach file handle to proc so we can close it later
        proc._dallama_stderr = stderr_file  # type: ignore[attr-defined]

        if self._job_object is not None:
            self._assign_to_job(proc)

        return proc

    async def graceful_stop(
        self,
        process: subprocess.Popen,
        timeout: float = 10.0,
    ) -> None:
        """Gracefully stop a process: terminate → wait → force kill."""
        if process.poll() is not None:
            self._close_stderr(process)
            return

        try:
            if _IS_WINDOWS:
                # CTRL_BREAK_EVENT is more reliable on Windows than terminate()
                import os
                import signal
                try:
                    os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                except OSError:
                    process.terminate()
            else:
                process.terminate()

            # Poll for exit without blocking the event loop
            for _ in range(int(timeout / 0.5)):
                if process.poll() is not None:
                    break
                await asyncio.sleep(0.5)
            else:
                logger.warning(
                    "Process %d didn't stop gracefully, killing...", process.pid
                )
                process.kill()
                loop = asyncio.get_running_loop()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, process.wait), timeout=5
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Process %d did not exit within 5s after kill", process.pid
                    )
        except Exception as e:
            logger.warning("Error stopping process: %s", e)
            try:
                process.kill()
            except Exception:
                pass

        self._close_stderr(process)

    def kill_orphans(self, executable_name: str = "llama-server") -> None:
        """Kill leftover processes from prior crashes."""
        try:
            if _IS_WINDOWS:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", f"{executable_name}.exe"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.warning("Killed orphaned %s: %s",
                                   executable_name, result.stdout.strip())
            else:
                result = subprocess.run(
                    ["pkill", "-f", executable_name],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.warning("Killed orphaned %s process(es)", executable_name)
        except Exception as e:
            logger.debug("Orphan cleanup skipped: %s", e)

    @staticmethod
    def _close_stderr(process: subprocess.Popen) -> None:
        f = getattr(process, "_dallama_stderr", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            process._dallama_stderr = None  # type: ignore[attr-defined]

    # ── Windows Job Object ────────────────────────────────────

    @staticmethod
    def _create_job_object():
        """Create a Windows Job Object with KILL_ON_JOB_CLOSE.

        When the parent process exits (even on crash), Windows closes
        all handles — including this job — which auto-kills every
        process assigned to it.  Returns None on failure.
        """
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return None

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
            JobObjectExtendedLimitInformation = 9

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            ok = kernel32.SetInformationJobObject(
                handle, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info),
            )
            if not ok:
                kernel32.CloseHandle(handle)
                return None

            logger.info("Windows Job Object created (KILL_ON_JOB_CLOSE)")
            return handle
        except Exception as e:
            logger.debug("Job Object setup failed: %s", e)
            return None

    def _assign_to_job(self, process: subprocess.Popen) -> None:
        """Assign a child process to the Job Object."""
        if self._job_object is None:
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_ALL_ACCESS = 0x1F0FFF
            h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, process.pid)
            if h_process:
                ok = kernel32.AssignProcessToJobObject(self._job_object, h_process)
                kernel32.CloseHandle(h_process)
                if ok:
                    logger.info("PID %d assigned to Job Object", process.pid)
                else:
                    logger.warning("Failed to assign PID %d to Job Object", process.pid)
        except Exception as e:
            logger.debug("Job assignment failed: %s", e)
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/dallama/tests/test_platform.py -v`
Expected: All 5 tests PASS (the SIGTERM-ignore test may behave differently on Windows — it should still pass because `kill()` is the fallback).

- [ ] **Step 5: Commit**

```bash
git add packages/dallama/src/dallama/platform.py packages/dallama/tests/test_platform.py
git commit -m "feat(dallama): platform helper — OS-specific process management"
```

---

### Task 3: Metrics Parser

**Files:**
- Create: `packages/dallama/src/dallama/metrics.py`
- Create: `packages/dallama/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/dallama/tests/test_metrics.py
"""Tests for MetricsParser — Prometheus /metrics parsing."""
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from dallama.metrics import MetricsParser, MetricsSnapshot

# Sample Prometheus output from llama-server (underscore format)
SAMPLE_METRICS_UNDERSCORE = """
# HELP llamacpp_tokens_predicted_total Total predicted tokens
# TYPE llamacpp_tokens_predicted_total counter
llamacpp_tokens_predicted_total 1234
llamacpp_prompt_tokens_total 5678
llamacpp_tokens_predicted_seconds_total 98.5
llamacpp_prompt_seconds_total 12.3
llamacpp_tokens_predicted_seconds 12.5
llamacpp_prompt_tokens_seconds 461.8
llamacpp_requests_processing 1
llamacpp_requests_pending 0
llamacpp_kv_cache_usage_ratio 0.42
""".strip()

# Same metrics but using colon format (older llama.cpp versions)
SAMPLE_METRICS_COLON = SAMPLE_METRICS_UNDERSCORE.replace(
    "llamacpp_", "llamacpp:"
)


@pytest.fixture
def parser():
    return MetricsParser()


@pytest.mark.asyncio
async def test_parse_underscore_format(parser):
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_METRICS_UNDERSCORE

    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance

        snap = await parser.fetch("http://127.0.0.1:8080")

    assert snap.generation_tokens_per_second == 12.5
    assert snap.prompt_tokens_per_second == 461.8
    assert snap.kv_cache_usage_percent == 42.0
    assert snap.requests_processing == 1
    assert snap.requests_pending == 0
    assert snap.generation_tokens_total == 1234
    assert snap.prompt_tokens_total == 5678


@pytest.mark.asyncio
async def test_parse_colon_format(parser):
    """Older llama.cpp uses colons instead of underscores."""
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_METRICS_COLON

    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance

        snap = await parser.fetch("http://127.0.0.1:8080")

    assert snap.generation_tokens_per_second == 12.5


@pytest.mark.asyncio
async def test_fetch_failure_returns_empty(parser):
    """Connection failures return zero-valued snapshot."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = AsyncMock()
        instance.get.side_effect = httpx.ConnectError("refused")
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance

        snap = await parser.fetch("http://127.0.0.1:8080")

    assert snap.generation_tokens_per_second == 0.0
    assert snap.generation_tokens_total == 0


@pytest.mark.asyncio
async def test_fetch_non_200_returns_empty(parser):
    mock_resp = AsyncMock()
    mock_resp.status_code = 503

    with patch("httpx.AsyncClient") as mock_client_cls:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = instance

        snap = await parser.fetch("http://127.0.0.1:8080")

    assert snap.generation_tokens_per_second == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/dallama/tests/test_metrics.py -v`
Expected: FAIL — `dallama.metrics` not found.

- [ ] **Step 3: Implement metrics.py**

```python
# packages/dallama/src/dallama/metrics.py
"""Parse llama-server's Prometheus-format /metrics endpoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Parsed metrics from llama-server."""

    generation_tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    kv_cache_usage_percent: float = 0.0
    requests_processing: int = 0
    requests_pending: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0


# Map normalized metric names to (snapshot_field, converter).
_METRIC_MAP: dict[str, tuple[str, type]] = {
    "llamacpp_tokens_predicted_seconds": ("generation_tokens_per_second", float),
    "llamacpp_prompt_tokens_seconds": ("prompt_tokens_per_second", float),
    "llamacpp_kv_cache_usage_ratio": ("kv_cache_usage_percent", lambda v: round(v * 100, 1)),
    "llamacpp_requests_processing": ("requests_processing", int),
    "llamacpp_requests_pending": ("requests_pending", int),
    "llamacpp_prompt_tokens_total": ("prompt_tokens_total", int),
    "llamacpp_tokens_predicted_total": ("generation_tokens_total", int),
}


class MetricsParser:
    """Fetches and parses llama-server /metrics."""

    async def fetch(self, api_base: str) -> MetricsSnapshot:
        """GET /metrics and parse Prometheus lines.

        Returns a zero-valued snapshot on any failure.
        """
        snap = MetricsSnapshot()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{api_base}/metrics", timeout=3.0)
                if resp.status_code != 200:
                    return snap

                for line in resp.text.splitlines():
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    # Normalize: llamacpp:foo → llamacpp_foo
                    metric = parts[0].split("{")[0].replace(":", "_")
                    try:
                        val = float(parts[-1])
                    except ValueError:
                        continue

                    mapping = _METRIC_MAP.get(metric)
                    if mapping is not None:
                        field_name, converter = mapping
                        setattr(snap, field_name, converter(val))

        except Exception as e:
            logger.debug("Failed to fetch /metrics: %s", e)

        return snap
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/dallama/tests/test_metrics.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/dallama/src/dallama/metrics.py packages/dallama/tests/test_metrics.py
git commit -m "feat(dallama): metrics parser — /metrics Prometheus parsing"
```

---

### Task 4: Server Process

**Files:**
- Create: `packages/dallama/src/dallama/server.py`
- Create: `packages/dallama/tests/test_server.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/dallama/tests/test_server.py
"""Tests for ServerProcess — cmd building, start/stop, health."""
import os
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dallama.config import DaLLaMaConfig, ServerConfig
from dallama.platform import PlatformHelper
from dallama.server import ServerProcess


@pytest.fixture
def dallama_cfg():
    return DaLLaMaConfig(
        llama_server_path="/usr/bin/llama-server",
        port=8080,
        host="127.0.0.1",
    )


@pytest.fixture
def platform_helper():
    return PlatformHelper()


@pytest.fixture
def server(dallama_cfg, platform_helper):
    return ServerProcess(dallama_cfg, platform_helper)


def test_build_cmd_minimal(server):
    cfg = ServerConfig(
        model_path="/models/test.gguf",
        model_name="test",
        context_length=4096,
    )
    cmd = server.build_cmd(cfg)
    assert "/usr/bin/llama-server" == cmd[0]
    assert "--model" in cmd
    assert "/models/test.gguf" in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "--ctx-size" in cmd
    assert "4096" in cmd
    assert "--metrics" in cmd
    assert "--jinja" in cmd
    # No reasoning flags when thinking=False and no --no-jinja
    assert "--reasoning" not in cmd


def test_build_cmd_thinking_on(server):
    cfg = ServerConfig(
        model_path="/m/test.gguf",
        model_name="test",
        context_length=8192,
        thinking=True,
    )
    cmd = server.build_cmd(cfg)
    idx = cmd.index("--reasoning")
    assert cmd[idx + 1] == "on"


def test_build_cmd_thinking_off_explicit(server):
    """When thinking=False, add --reasoning off --reasoning-budget 0."""
    cfg = ServerConfig(
        model_path="/m/test.gguf",
        model_name="test",
        context_length=8192,
        thinking=False,
    )
    cmd = server.build_cmd(cfg)
    # Should not have reasoning flags at all — only thinking models need them
    # and we don't know if it's a thinking model without registry info.
    # Thinking flags are only added when thinking=True.
    assert "--reasoning" not in cmd


def test_build_cmd_no_jinja_skips_reasoning(server):
    """--no-jinja models skip reasoning flags entirely."""
    cfg = ServerConfig(
        model_path="/m/test.gguf",
        model_name="test",
        context_length=4096,
        thinking=True,
        extra_flags=["--no-jinja"],
    )
    cmd = server.build_cmd(cfg)
    assert "--reasoning" not in cmd
    assert "--jinja" not in cmd
    assert "--no-jinja" in cmd


def test_build_cmd_vision(server):
    cfg = ServerConfig(
        model_path="/m/test.gguf",
        model_name="test",
        context_length=4096,
        vision_projector="/m/mmproj.gguf",
    )
    cmd = server.build_cmd(cfg)
    assert "--mmproj" in cmd
    assert "/m/mmproj.gguf" in cmd


def test_build_cmd_extra_flags(server):
    cfg = ServerConfig(
        model_path="/m/test.gguf",
        model_name="test",
        context_length=4096,
        extra_flags=["--chat-template", "chatml", "--override-kv", "key=val"],
    )
    cmd = server.build_cmd(cfg)
    assert "--chat-template" in cmd
    assert "chatml" in cmd
    assert "--override-kv" in cmd


def test_is_alive_no_process(server):
    assert server.is_alive() is False


@pytest.mark.asyncio
async def test_health_check_no_process(server):
    assert await server.health_check() is False


@pytest.mark.asyncio
async def test_stop_no_process(server):
    """stop() on a server with no process should not raise."""
    await server.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/dallama/tests/test_server.py -v`
Expected: FAIL — `dallama.server` not found.

- [ ] **Step 3: Implement server.py**

```python
# packages/dallama/src/dallama/server.py
"""ServerProcess — llama-server subprocess lifecycle.

Builds the command line from ServerConfig, starts/stops the process,
polls /health for readiness.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time

import httpx

from .config import DaLLaMaConfig, ServerConfig
from .platform import PlatformHelper

logger = logging.getLogger(__name__)


class ServerProcess:
    """Manages a single llama-server subprocess."""

    def __init__(self, config: DaLLaMaConfig, platform: PlatformHelper):
        self._config = config
        self._platform = platform
        self.process: subprocess.Popen | None = None
        self._stderr_path: str = ""

    @property
    def api_base(self) -> str:
        return f"http://{self._config.host}:{self._config.port}"

    def build_cmd(self, config: ServerConfig) -> list[str]:
        """Build the llama-server command line from config."""
        cmd = [
            self._config.llama_server_path,
            "--model", config.model_path,
            "--alias", "local-model",
            "--port", str(self._config.port),
            "--host", self._config.host,
            "--ctx-size", str(config.context_length),
            "--flash-attn", "auto",
            "--metrics",
            "--batch-size", "2048",
            "--ubatch-size", "512",
        ]

        # Thread count: auto-detect or let llama-server decide
        try:
            logical = os.cpu_count() or 4
            physical = max(2, logical // 2)
            threads = max(2, physical - 2)
            cmd.extend(["--threads", str(threads)])
        except Exception:
            pass  # let llama-server default

        # Jinja templating (required for function calling)
        has_no_jinja = "--no-jinja" in config.extra_flags
        if not has_no_jinja:
            cmd.append("--jinja")

        # Thinking/reasoning mode (skip for --no-jinja models)
        if config.thinking and not has_no_jinja:
            cmd.extend(["--reasoning", "on"])

        # Vision projector
        if config.vision_projector:
            cmd.extend(["--mmproj", config.vision_projector])

        # Extra per-model flags
        if config.extra_flags:
            cmd.extend(config.extra_flags)

        return cmd

    async def start(self, config: ServerConfig) -> bool:
        """Launch llama-server and wait for it to become healthy.

        Returns True if the server is healthy, False on failure.
        """
        cmd = self.build_cmd(config)
        logger.info("Starting llama-server: %s", " ".join(cmd))

        # Determine stderr log path
        self._stderr_path = os.path.join(
            os.environ.get("DALLAMA_LOG_DIR", "."),
            "llama-server.stderr.log",
        )

        try:
            self.process = self._platform.create_process(cmd, self._stderr_path)
        except FileNotFoundError:
            logger.error(
                "llama-server not found at '%s'. "
                "Set llama_server_path in DaLLaMaConfig.",
                self._config.llama_server_path,
            )
            return False
        except Exception as e:
            logger.error("Failed to start llama-server: %s", e)
            return False

        # Adaptive timeout: estimate from file size if available
        max_wait = self._estimate_load_timeout(config)
        healthy = await self._wait_for_healthy(timeout=max_wait)

        if not healthy:
            logger.error(
                "llama-server failed to become healthy within %.0fs", max_wait
            )
            await self.stop()
            return False

        return True

    async def stop(self) -> None:
        """Stop the llama-server process."""
        if self.process is None:
            return
        logger.info("Stopping llama-server (PID %d)...", self.process.pid)
        await self._platform.graceful_stop(self.process)
        self.process = None

    async def health_check(self) -> bool:
        """Quick health check — is the server responding?"""
        return (await self._health_check_status()) == 200

    async def _health_check_status(self) -> int:
        """Returns HTTP status code, or 0 on connection failure."""
        if self.process is None or self.process.poll() is not None:
            return 0
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base}/health", timeout=3.0,
                )
                return resp.status_code
        except Exception:
            return 0

    def is_alive(self) -> bool:
        """Is the llama-server process still running?"""
        return self.process is not None and self.process.poll() is None

    def read_stderr_tail(self, lines: int = 10) -> str:
        """Read the last N lines of stderr for crash diagnostics."""
        if not self._stderr_path:
            return ""
        try:
            with open(self._stderr_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:]).strip()
        except Exception:
            return ""

    async def _wait_for_healthy(self, timeout: float = 60) -> bool:
        """Poll /health endpoint until server is ready."""
        start = time.time()
        check_interval = 1.0

        while (time.time() - start) < timeout:
            # Check if process died
            if self.process and self.process.poll() is not None:
                stderr_tail = self.read_stderr_tail()
                logger.error(
                    "llama-server exited with code %d%s",
                    self.process.returncode,
                    f"\nStderr:\n{stderr_tail}" if stderr_tail else "",
                )
                return False

            status = await self._health_check_status()
            if status == 200:
                return True

            await asyncio.sleep(check_interval)
            if check_interval < 3.0:
                check_interval += 0.5

        return False

    @staticmethod
    def _estimate_load_timeout(config: ServerConfig) -> float:
        """Estimate how long the server needs to load a model."""
        try:
            file_size_mb = os.path.getsize(config.model_path) / (1024 * 1024)
        except Exception:
            file_size_mb = 5000  # conservative fallback

        # Weight loading: ~500 MB/s from disk to VRAM
        weight_time = file_size_mb / 500
        # KV cache pre-allocation scales with context
        ctx_factor = config.context_length / 8192
        kv_time = ctx_factor * 15
        estimated = weight_time + kv_time
        # 2x safety margin, clamped to [45, 300]
        return min(300, max(45, estimated * 2.0))
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/dallama/tests/test_server.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/dallama/src/dallama/server.py packages/dallama/tests/test_server.py
git commit -m "feat(dallama): server process — cmd building, start/stop, health"
```

---

### Task 5: Swap Manager

**Files:**
- Create: `packages/dallama/src/dallama/swap.py`
- Create: `packages/dallama/tests/test_swap.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/dallama/tests/test_swap.py
"""Tests for SwapManager — drain, circuit breaker, swap orchestration."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from dallama.config import DaLLaMaConfig, ServerConfig
from dallama.swap import SwapManager


@pytest.fixture
def cfg():
    return DaLLaMaConfig(
        circuit_breaker_threshold=2,
        circuit_breaker_cooldown_seconds=1.0,  # short for tests
        inference_drain_timeout_seconds=2.0,
    )


@pytest.fixture
def swap(cfg):
    return SwapManager(cfg)


@pytest.fixture
def server():
    s = AsyncMock()
    s.is_alive.return_value = True
    s.health_check = AsyncMock(return_value=True)
    s.stop = AsyncMock()
    s.start = AsyncMock(return_value=True)
    s.process = MagicMock()
    return s


@pytest.fixture
def server_config():
    return ServerConfig(
        model_path="/m/test.gguf",
        model_name="test-model",
        context_length=4096,
    )


# ── Inference tracking ───────────────────────────────────────

def test_inflight_initially_false(swap):
    assert swap.has_inflight is False


def test_mark_inference_start_end(swap):
    gen = swap.mark_inference_start()
    assert swap.has_inflight is True
    swap.mark_inference_end(gen)
    assert swap.has_inflight is False


def test_mark_inference_end_wrong_generation(swap):
    """Orphaned inference end from old generation is ignored."""
    gen = swap.mark_inference_start()
    swap.force_reset_inflight()  # simulates generation bump
    swap.mark_inference_end(gen)  # old generation — should be ignored
    assert swap.has_inflight is False  # counter was already reset


def test_multiple_inflight(swap):
    g1 = swap.mark_inference_start()
    g2 = swap.mark_inference_start()
    assert swap.has_inflight is True
    swap.mark_inference_end(g1)
    assert swap.has_inflight is True  # g2 still active
    swap.mark_inference_end(g2)
    assert swap.has_inflight is False


# ── Circuit breaker ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_circuit_breaker_blocks_after_threshold(swap, server, server_config):
    """After 2 consecutive failures, circuit breaker blocks the model."""
    server.start = AsyncMock(return_value=False)

    result1 = await swap.swap(server, server_config)
    assert result1 is False

    result2 = await swap.swap(server, server_config)
    assert result2 is False

    # Third attempt should be blocked by circuit breaker (not even try)
    result3 = await swap.swap(server, server_config)
    assert result3 is False
    # start should only be called twice (third was blocked)
    assert server.start.call_count == 2


@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_success(swap, server, server_config):
    server.start = AsyncMock(return_value=False)
    await swap.swap(server, server_config)  # fail 1

    server.start = AsyncMock(return_value=True)
    result = await swap.swap(server, server_config)  # success
    assert result is True
    # Circuit breaker should be reset — next failure starts fresh
    assert swap._fail_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_cooldown_expires(swap, server, server_config):
    """After cooldown expires, model can be loaded again."""
    server.start = AsyncMock(return_value=False)
    await swap.swap(server, server_config)  # fail 1
    await swap.swap(server, server_config)  # fail 2 → breaker trips

    # Wait for cooldown (set to 1s in fixture)
    await asyncio.sleep(1.1)

    server.start = AsyncMock(return_value=True)
    result = await swap.swap(server, server_config)
    assert result is True


# ── Drain ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_swap_drains_inflight(swap, server, server_config):
    """Swap waits for in-flight inferences to finish."""
    gen = swap.mark_inference_start()

    # Schedule inference end after 0.5s
    async def finish_inference():
        await asyncio.sleep(0.5)
        swap.mark_inference_end(gen)

    asyncio.create_task(finish_inference())

    result = await swap.swap(server, server_config)
    assert result is True
    assert swap.has_inflight is False


@pytest.mark.asyncio
async def test_swap_force_drains_on_timeout(swap, server, server_config):
    """If drain times out, force-reset and proceed."""
    swap._config = DaLLaMaConfig(inference_drain_timeout_seconds=0.5)
    swap.mark_inference_start()  # never ended

    result = await swap.swap(server, server_config)
    assert result is True  # should proceed after force-drain
    assert swap.has_inflight is False  # force-reset cleared it


# ── VRAM check ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_swap_refuses_insufficient_vram(server, server_config):
    cfg = DaLLaMaConfig(
        min_free_vram_mb=4096,
        get_vram_free_mb=lambda: 2000,  # below threshold
    )
    swap = SwapManager(cfg)
    result = await swap.swap(server, server_config)
    assert result is False
    server.start.assert_not_called()


@pytest.mark.asyncio
async def test_swap_proceeds_without_vram_callback(swap, server, server_config):
    """If get_vram_free_mb is None, skip VRAM check."""
    result = await swap.swap(server, server_config)
    assert result is True


# ── on_ready callback ────────────────────────────────────────

@pytest.mark.asyncio
async def test_swap_calls_on_ready(server, server_config):
    calls = []
    cfg = DaLLaMaConfig(on_ready=lambda m, r: calls.append((m, r)))
    swap = SwapManager(cfg)

    await swap.swap(server, server_config)
    assert len(calls) == 1
    assert calls[0] == ("test-model", "model_loaded")


@pytest.mark.asyncio
async def test_swap_calls_on_ready_failure(server, server_config):
    calls = []
    cfg = DaLLaMaConfig(on_ready=lambda m, r: calls.append((m, r)))
    swap = SwapManager(cfg)
    server.start = AsyncMock(return_value=False)

    await swap.swap(server, server_config)
    assert calls[0] == (None, "load_failed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/dallama/tests/test_swap.py -v`
Expected: FAIL — `dallama.swap` not found.

- [ ] **Step 3: Implement swap.py**

```python
# packages/dallama/src/dallama/swap.py
"""SwapManager — model transition orchestration.

Handles: asyncio lock, inference drain, circuit breaker, VRAM check,
stop → start sequencing.  Designed so the swap strategy (process restart
vs future hot-swap) can be changed without touching callers.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .config import DaLLaMaConfig, ServerConfig

logger = logging.getLogger(__name__)


class SwapManager:
    """Orchestrates model swaps with safety guarantees."""

    def __init__(self, config: DaLLaMaConfig):
        self._config = config
        self._lock = asyncio.Lock()

        # Inference tracking
        self._inflight_count: int = 0
        self._inflight_idle = asyncio.Event()
        self._inflight_idle.set()
        self._generation: int = 0

        # Circuit breaker
        self._fail_count: int = 0
        self._fail_model: str | None = None
        self._cooldown_until: float = 0.0

        # Swap-in-progress flag for watchdog coordination
        self.swap_in_progress: bool = False

    # ── Inference tracking ────────────────────────────────────

    def mark_inference_start(self) -> int:
        """Mark an inference as in-flight. Returns generation ID."""
        self._inflight_count += 1
        self._inflight_idle.clear()
        return self._generation

    def mark_inference_end(self, generation: int) -> None:
        """Mark an inference as complete. Ignores stale generations."""
        if generation != self._generation:
            return
        self._inflight_count = max(0, self._inflight_count - 1)
        if self._inflight_count == 0:
            self._inflight_idle.set()

    @property
    def has_inflight(self) -> bool:
        return self._inflight_count > 0

    def force_reset_inflight(self) -> None:
        """Bump generation and reset counter. Used during force-drain."""
        self._generation += 1
        self._inflight_count = 0
        self._inflight_idle.set()

    # ── Circuit breaker ───────────────────────────────────────

    def _is_blocked(self, model_name: str) -> bool:
        if self._cooldown_until <= 0:
            return False
        if time.time() >= self._cooldown_until:
            self._fail_count = 0
            self._cooldown_until = 0.0
            return False
        return self._fail_model == model_name

    def _record_failure(self, model_name: str) -> None:
        if self._fail_model != model_name:
            self._fail_model = model_name
            self._fail_count = 0
        self._fail_count += 1
        if self._fail_count >= self._config.circuit_breaker_threshold:
            self._cooldown_until = (
                time.time() + self._config.circuit_breaker_cooldown_seconds
            )
            logger.error(
                "Circuit breaker: %s failed %d times — refusing for %.0fs",
                model_name, self._fail_count,
                self._config.circuit_breaker_cooldown_seconds,
            )

    def _record_success(self) -> None:
        self._fail_count = 0
        self._fail_model = None
        self._cooldown_until = 0.0

    # ── Swap ──────────────────────────────────────────────────

    async def swap(
        self,
        server,  # ServerProcess — not typed to avoid circular import
        config: ServerConfig,
    ) -> bool:
        """Execute a model swap. Returns True on success.

        Flow:
        1. Circuit breaker check
        2. Drain in-flight inferences
        3. Stop current server
        4. VRAM check
        5. Start new server
        6. Record result + callback
        """
        async with self._lock:
            return await self._do_swap(server, config)

    async def _do_swap(self, server, config: ServerConfig) -> bool:
        # 1. Circuit breaker
        if self._is_blocked(config.model_name):
            remaining = self._cooldown_until - time.time()
            logger.warning(
                "Circuit breaker active for %s (%.0fs remaining)",
                config.model_name, remaining,
            )
            self._notify(None, "circuit_breaker_active")
            return False

        self.swap_in_progress = True
        try:
            # 2. Drain in-flight inferences
            if self._inflight_count > 0:
                logger.info(
                    "Draining %d in-flight inference(s) before swap to %s...",
                    self._inflight_count, config.model_name,
                )
                try:
                    await asyncio.wait_for(
                        self._inflight_idle.wait(),
                        timeout=self._config.inference_drain_timeout_seconds,
                    )
                    logger.info("Inference drained, proceeding with swap")
                except asyncio.TimeoutError:
                    logger.warning(
                        "Drain timed out (%d still active). Force-draining.",
                        self._inflight_count,
                    )
                    self.force_reset_inflight()

            # 3. Stop current server
            if server.is_alive():
                await server.stop()
                # CUDA VRAM release lags behind process exit
                await asyncio.sleep(2)

            # 4. VRAM check
            get_vram = self._config.get_vram_free_mb
            if get_vram is not None:
                free_mb = get_vram()
                if free_mb < self._config.min_free_vram_mb:
                    logger.error(
                        "Insufficient VRAM: %dMB free, need %dMB",
                        free_mb, self._config.min_free_vram_mb,
                    )
                    self._notify(None, "insufficient_vram")
                    return False

            # 5. Start new server
            success = await server.start(config)

            # 6. Record result
            if success:
                self._record_success()
                logger.info("Model %s loaded successfully", config.model_name)
                self._notify(config.model_name, "model_loaded")
            else:
                self._record_failure(config.model_name)
                self._notify(None, "load_failed")

            return success
        finally:
            self.swap_in_progress = False

    def _notify(self, model_name: str | None, reason: str) -> None:
        cb = self._config.on_ready
        if cb is not None:
            try:
                cb(model_name, reason)
            except Exception as e:
                logger.warning("on_ready callback error: %s", e)
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/dallama/tests/test_swap.py -v`
Expected: All 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/dallama/src/dallama/swap.py packages/dallama/tests/test_swap.py
git commit -m "feat(dallama): swap manager — drain, circuit breaker, VRAM check"
```

---

### Task 6: Watchdog + Idle Unloader

**Files:**
- Create: `packages/dallama/src/dallama/watchdog.py`
- Create: `packages/dallama/tests/test_watchdog.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/dallama/tests/test_watchdog.py
"""Tests for HealthWatchdog and IdleUnloader."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from dallama.config import DaLLaMaConfig, ServerConfig
from dallama.watchdog import HealthWatchdog, IdleUnloader


# ── HealthWatchdog ────────────────────────────────────────────

@pytest.fixture
def watchdog_cfg():
    return DaLLaMaConfig(
        health_check_interval_seconds=0.1,  # fast for tests
        health_fail_threshold=2,
    )


@pytest.fixture
def mock_server():
    s = AsyncMock()
    s.is_alive.return_value = True
    s.health_check = AsyncMock(return_value=True)
    s._health_check_status = AsyncMock(return_value=200)
    return s


@pytest.fixture
def mock_swap():
    s = MagicMock()
    s.swap_in_progress = False
    s.swap = AsyncMock(return_value=True)
    return s


@pytest.mark.asyncio
async def test_watchdog_detects_crash(watchdog_cfg, mock_server, mock_swap):
    """Process exit triggers restart via swap."""
    current_config = ServerConfig(
        model_path="/m/test.gguf", model_name="test", context_length=4096,
    )
    wd = HealthWatchdog(watchdog_cfg, mock_server, mock_swap)

    # Simulate: server alive initially, then dies
    call_count = 0
    def is_alive_side_effect():
        nonlocal call_count
        call_count += 1
        return call_count <= 2  # alive for first 2 checks, then dead

    mock_server.is_alive.side_effect = is_alive_side_effect
    mock_server.process = MagicMock()
    mock_server.process.poll.side_effect = [None, None, 1]  # exits on 3rd check
    mock_server.process.returncode = 1

    task = asyncio.create_task(wd.run(lambda: current_config))
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    mock_swap.swap.assert_called()


@pytest.mark.asyncio
async def test_watchdog_detects_hang(watchdog_cfg, mock_server, mock_swap):
    """Consecutive health failures trigger restart."""
    current_config = ServerConfig(
        model_path="/m/test.gguf", model_name="test", context_length=4096,
    )
    wd = HealthWatchdog(watchdog_cfg, mock_server, mock_swap)

    mock_server.is_alive.return_value = True
    mock_server._health_check_status = AsyncMock(return_value=0)

    task = asyncio.create_task(wd.run(lambda: current_config))
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert mock_server.stop.called or mock_swap.swap.called


@pytest.mark.asyncio
async def test_watchdog_skips_during_swap(watchdog_cfg, mock_server, mock_swap):
    """No health checks during an active swap."""
    current_config = ServerConfig(
        model_path="/m/test.gguf", model_name="test", context_length=4096,
    )
    mock_swap.swap_in_progress = True
    mock_server._health_check_status = AsyncMock(return_value=0)

    wd = HealthWatchdog(watchdog_cfg, mock_server, mock_swap)
    task = asyncio.create_task(wd.run(lambda: current_config))
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Should NOT have tried to restart during swap
    mock_swap.swap.assert_not_called()


# ── IdleUnloader ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_idle_unloader_unloads_after_timeout():
    cfg = DaLLaMaConfig(idle_timeout_seconds=0.3)
    server = AsyncMock()
    server.is_alive.return_value = True
    swap = MagicMock()
    swap.has_inflight = False

    calls = []
    cfg.on_ready = lambda m, r: calls.append((m, r))

    unloader = IdleUnloader(cfg, server, swap)
    unloader.reset_timer()  # start the clock

    task = asyncio.create_task(unloader.run())
    await asyncio.sleep(0.6)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    server.stop.assert_called()
    assert any(r == "idle_unload" for _, r in calls)


@pytest.mark.asyncio
async def test_idle_unloader_reset_prevents_unload():
    cfg = DaLLaMaConfig(idle_timeout_seconds=0.5)
    server = AsyncMock()
    server.is_alive.return_value = True
    swap = MagicMock()
    swap.has_inflight = False

    unloader = IdleUnloader(cfg, server, swap)
    unloader.reset_timer()

    task = asyncio.create_task(unloader.run())
    # Keep resetting before timeout hits
    await asyncio.sleep(0.3)
    unloader.reset_timer()
    await asyncio.sleep(0.3)
    unloader.reset_timer()
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    server.stop.assert_not_called()


@pytest.mark.asyncio
async def test_idle_unloader_skips_when_inflight():
    cfg = DaLLaMaConfig(idle_timeout_seconds=0.2)
    server = AsyncMock()
    server.is_alive.return_value = True
    swap = MagicMock()
    swap.has_inflight = True  # active inference

    unloader = IdleUnloader(cfg, server, swap)
    unloader.reset_timer()

    task = asyncio.create_task(unloader.run())
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    server.stop.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/dallama/tests/test_watchdog.py -v`
Expected: FAIL — `dallama.watchdog` not found.

- [ ] **Step 3: Implement watchdog.py**

```python
# packages/dallama/src/dallama/watchdog.py
"""Background tasks: health watchdog + idle unloader."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from .config import DaLLaMaConfig, ServerConfig

logger = logging.getLogger(__name__)


class HealthWatchdog:
    """Detects crashed or hung llama-server and auto-restarts.

    Two failure modes:
      1. Process exit (crash) — detected via is_alive()
      2. Unresponsive (hang) — N consecutive /health failures
    """

    def __init__(self, config: DaLLaMaConfig, server, swap):
        self._config = config
        self._server = server
        self._swap = swap

    async def run(self, get_current_config: Callable[[], ServerConfig | None]) -> None:
        """Run forever. Cancel the task to stop."""
        consecutive_failures = 0
        threshold = self._config.health_fail_threshold

        while True:
            await asyncio.sleep(self._config.health_check_interval_seconds)

            config = get_current_config()
            if config is None:
                consecutive_failures = 0
                continue

            # Skip during active swap or idle unload
            if self._swap.swap_in_progress:
                consecutive_failures = 0
                continue

            # Crash detection
            if not self._server.is_alive():
                logger.error(
                    "llama-server crashed! Restarting %s...", config.model_name
                )
                consecutive_failures = 0
                await self._swap.swap(self._server, config)
                continue

            # Hang detection
            status = await self._server._health_check_status()
            if status == 200 or status == 503:
                # 200 = healthy, 503 = busy loading (not a hang)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logger.warning(
                    "llama-server /health failed (%d/%d)",
                    consecutive_failures, threshold,
                )
                if consecutive_failures >= threshold:
                    logger.error(
                        "llama-server hung (%d consecutive failures). "
                        "Restarting %s...",
                        threshold, config.model_name,
                    )
                    consecutive_failures = 0
                    await self._server.stop()
                    await self._swap.swap(self._server, config)


class IdleUnloader:
    """Unloads the model after a period of inactivity."""

    def __init__(self, config: DaLLaMaConfig, server, swap):
        self._config = config
        self._server = server
        self._swap = swap
        self._last_activity: float = 0.0

    def reset_timer(self) -> None:
        """Reset the idle timer (called on infer exit and keep_alive)."""
        self._last_activity = time.time()

    @property
    def idle_seconds(self) -> float:
        if self._last_activity == 0:
            return 0.0
        return time.time() - self._last_activity

    async def run(self) -> None:
        """Run forever. Cancel the task to stop."""
        while True:
            await asyncio.sleep(30)

            if not self._server.is_alive():
                continue

            if self._swap.swap_in_progress:
                continue

            if self._last_activity == 0:
                continue

            if self.idle_seconds <= self._config.idle_timeout_seconds:
                continue

            if self._swap.has_inflight:
                continue

            logger.info(
                "Model idle for %.0fs (>%.0fs), unloading",
                self.idle_seconds, self._config.idle_timeout_seconds,
            )
            await self._server.stop()

            cb = self._config.on_ready
            if cb is not None:
                try:
                    cb(None, "idle_unload")
                except Exception as e:
                    logger.warning("on_ready callback error: %s", e)
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/dallama/tests/test_watchdog.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/dallama/src/dallama/watchdog.py packages/dallama/tests/test_watchdog.py
git commit -m "feat(dallama): watchdog + idle unloader — background health tasks"
```

---

### Task 7: DaLLaMa Main Class

**Files:**
- Modify: `packages/dallama/src/dallama/__init__.py`
- Create: `packages/dallama/src/dallama/dallama.py`
- Create: `packages/dallama/tests/test_dallama.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/dallama/tests/test_dallama.py
"""Integration tests for the DaLLaMa main class."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dallama import DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus, DaLLaMaLoadError


@pytest.fixture
def cfg():
    return DaLLaMaConfig(
        llama_server_path="/usr/bin/llama-server",
        port=8080,
        idle_timeout_seconds=60,
    )


@pytest.fixture
def dallama(cfg):
    return DaLLaMa(cfg)


# ── Status ───────────────────────────────────────────────────

def test_initial_status(dallama):
    st = dallama.status
    assert st.model_name is None
    assert st.healthy is False
    assert st.busy is False
    assert st.measured_tps == 0.0
    assert st.context_length == 0


# ── Infer ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_infer_loads_model(dallama):
    """First infer() call triggers a swap (load)."""
    config = ServerConfig(
        model_path="/m/test.gguf",
        model_name="test",
        context_length=4096,
    )

    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True):
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config) as session:
                    assert session.url == "http://127.0.0.1:8080"
                    assert session.model_name == "test"

    assert dallama.status.model_name == "test"


@pytest.mark.asyncio
async def test_infer_same_model_no_swap(dallama):
    """Second infer() with same model skips swap."""
    config = ServerConfig(
        model_path="/m/test.gguf",
        model_name="test",
        context_length=4096,
    )

    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True) as mock_swap:
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config):
                    pass
                async with dallama.infer(config):
                    pass

    assert mock_swap.call_count == 1  # only first call triggered swap


@pytest.mark.asyncio
async def test_infer_different_model_triggers_swap(dallama):
    """Switching models triggers a new swap."""
    config1 = ServerConfig(
        model_path="/m/a.gguf", model_name="model-a", context_length=4096,
    )
    config2 = ServerConfig(
        model_path="/m/b.gguf", model_name="model-b", context_length=8192,
    )

    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True) as mock_swap:
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config1):
                    pass
                async with dallama.infer(config2):
                    pass

    assert mock_swap.call_count == 2


@pytest.mark.asyncio
async def test_infer_thinking_change_triggers_swap(dallama):
    """Changing thinking mode triggers swap even for same model."""
    config_no_think = ServerConfig(
        model_path="/m/a.gguf", model_name="model-a", context_length=4096,
        thinking=False,
    )
    config_think = ServerConfig(
        model_path="/m/a.gguf", model_name="model-a", context_length=4096,
        thinking=True,
    )

    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True) as mock_swap:
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config_no_think):
                    pass
                async with dallama.infer(config_think):
                    pass

    assert mock_swap.call_count == 2


@pytest.mark.asyncio
async def test_infer_raises_on_load_failure(dallama):
    """Failed swap raises DaLLaMaLoadError."""
    config = ServerConfig(
        model_path="/m/test.gguf", model_name="test", context_length=4096,
    )

    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=False):
        with pytest.raises(DaLLaMaLoadError, match="test"):
            async with dallama.infer(config):
                pass


@pytest.mark.asyncio
async def test_infer_tracks_inflight(dallama):
    """During infer, status.busy is True."""
    config = ServerConfig(
        model_path="/m/test.gguf", model_name="test", context_length=4096,
    )

    with patch.object(dallama._swap, "swap", new_callable=AsyncMock, return_value=True):
        with patch.object(dallama._server, "is_alive", return_value=True):
            with patch.object(dallama._server, "health_check", new_callable=AsyncMock, return_value=True):
                async with dallama.infer(config):
                    assert dallama.status.busy is True
                assert dallama.status.busy is False


# ── Keep alive ───────────────────────────────────────────────

def test_keep_alive(dallama):
    """keep_alive should not raise."""
    dallama.keep_alive()


# ── Start / Stop ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_stop(dallama):
    """start() and stop() should not raise."""
    with patch.object(dallama._platform, "kill_orphans"):
        await dallama.start()
    with patch.object(dallama._server, "stop", new_callable=AsyncMock):
        await dallama.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/dallama/tests/test_dallama.py -v`
Expected: FAIL — `DaLLaMa` class not found.

- [ ] **Step 3: Implement dallama.py**

```python
# packages/dallama/src/dallama/dallama.py
"""DaLLaMa — main class composing all modules."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from .config import (
    DaLLaMaConfig,
    DaLLaMaLoadError,
    InferenceSession,
    ServerConfig,
    ServerStatus,
)
from .metrics import MetricsParser
from .platform import PlatformHelper
from .server import ServerProcess
from .swap import SwapManager
from .watchdog import HealthWatchdog, IdleUnloader

logger = logging.getLogger(__name__)


class DaLLaMa:
    """Python async llama-server process manager.

    Three public methods: infer(), keep_alive(), status.
    """

    def __init__(self, config: DaLLaMaConfig):
        self._config = config
        self._platform = PlatformHelper()
        self._server = ServerProcess(config, self._platform)
        self._swap = SwapManager(config)
        self._metrics = MetricsParser()
        self._idle_unloader = IdleUnloader(config, self._server, self._swap)
        self._watchdog = HealthWatchdog(config, self._server, self._swap)

        self._current_config: ServerConfig | None = None
        self._last_tps: float = 0.0

        self._watchdog_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start background tasks (watchdog, idle unloader)."""
        self._platform.kill_orphans()
        self._watchdog_task = asyncio.create_task(
            self._watchdog.run(lambda: self._current_config)
        )
        self._idle_task = asyncio.create_task(self._idle_unloader.run())
        logger.info("DaLLaMa started (port %d)", self._config.port)

    async def stop(self) -> None:
        """Stop server and background tasks."""
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
            self._idle_task = None

        await self._server.stop()
        self._current_config = None
        logger.info("DaLLaMa stopped")

    @asynccontextmanager
    async def infer(self, config: ServerConfig) -> AsyncIterator[InferenceSession]:
        """Ensure model is loaded, yield session, track lifecycle.

        If a different model (or different thinking/vision mode) is
        requested, DaLLaMa swaps automatically.  Raises DaLLaMaLoadError
        if the model cannot be loaded.
        """
        needs_swap = (
            self._current_config is None
            or config.model_name != self._current_config.model_name
            or config.thinking != self._current_config.thinking
            or config.vision_projector != self._current_config.vision_projector
        )

        if needs_swap:
            success = await self._swap.swap(self._server, config)
            if not success:
                raise DaLLaMaLoadError(config.model_name)
            self._current_config = config

        gen = self._swap.mark_inference_start()
        try:
            yield InferenceSession(
                url=f"http://{self._config.host}:{self._config.port}",
                model_name=config.model_name,
            )
        finally:
            self._swap.mark_inference_end(gen)
            self._idle_unloader.reset_timer()
            # Update measured tps in the background
            asyncio.ensure_future(self._refresh_tps())

    def keep_alive(self) -> None:
        """Reset idle timer without starting inference."""
        self._idle_unloader.reset_timer()

    @property
    def status(self) -> ServerStatus:
        """Current state for routing decisions."""
        return ServerStatus(
            model_name=(
                self._current_config.model_name
                if self._current_config else None
            ),
            healthy=self._server.is_alive(),
            busy=self._swap.has_inflight,
            measured_tps=self._last_tps,
            context_length=(
                self._current_config.context_length
                if self._current_config else 0
            ),
        )

    async def _refresh_tps(self) -> None:
        """Fetch /metrics and update measured_tps."""
        try:
            snap = await self._metrics.fetch(self._server.api_base)
            if snap.generation_tokens_per_second > 0:
                self._last_tps = snap.generation_tokens_per_second
        except Exception:
            pass
```

- [ ] **Step 4: Update __init__.py to export DaLLaMa**

```python
# packages/dallama/src/dallama/__init__.py
"""DaLLaMa — Python async llama-server process manager."""

from .config import (
    DaLLaMaConfig,
    DaLLaMaLoadError,
    InferenceSession,
    ServerConfig,
    ServerStatus,
)
from .dallama import DaLLaMa

__all__ = [
    "DaLLaMa",
    "DaLLaMaConfig",
    "DaLLaMaLoadError",
    "InferenceSession",
    "ServerConfig",
    "ServerStatus",
]
```

- [ ] **Step 5: Run all DaLLaMa tests**

Run: `pytest packages/dallama/tests/ -v`
Expected: All tests PASS across all test files.

- [ ] **Step 6: Commit**

```bash
git add packages/dallama/src/dallama/dallama.py packages/dallama/src/dallama/__init__.py packages/dallama/tests/test_dallama.py
git commit -m "feat(dallama): main class — infer, keep_alive, status"
```

---

### Task 8: Editable Install + Full Test Run

**Files:**
- Modify: `requirements.txt` (add editable install line)

- [ ] **Step 1: Add editable install to requirements.txt**

Add this line after the existing `-e ./packages/yasar_usta` line:

```
-e ./packages/dallama
```

- [ ] **Step 2: Install the package**

Run: `pip install -e ./packages/dallama`
Expected: Successfully installed dallama-0.1.0.

- [ ] **Step 3: Run full DaLLaMa test suite**

Run: `pytest packages/dallama/tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Verify imports work from outside the package**

Run: `python -c "from dallama import DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus, DaLLaMaLoadError; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "build(dallama): add editable install to requirements.txt"
```

---

### Task 9: KutAI Shim — Replace local_model_manager.py

**Files:**
- Modify: `src/models/local_model_manager.py` (replace 1,193 lines with ~80-line shim)

**Important context:** This task replaces the old `local_model_manager.py` with a thin shim that delegates to DaLLaMa. It preserves the existing function signatures (`get_local_manager()`, `get_runtime_state()`) so that all 8 consumer files continue working without changes. The shim translates between KutAI's `ModelInfo` and DaLLaMa's `ServerConfig`.

Read the current `src/models/local_model_manager.py` for the full public API that consumers depend on. Key exports used by consumers:

- `get_local_manager() -> LocalModelManager` — used by 8 files
- `get_runtime_state() -> ModelRuntimeState | None` — used by router.py and llm_dispatcher.py
- `LocalModelManager.ensure_model(model_name, reason, enable_thinking, enable_vision)` — called by dispatcher
- `LocalModelManager.acquire_inference_slot(priority, task_id, ...)` — called by router
- `LocalModelManager.release_inference_slot()` — called by router
- `LocalModelManager.mark_inference_start()` / `mark_inference_end()` — called by router
- `LocalModelManager.get_status()` — called by telegram_bot, api.py
- `LocalModelManager.get_metrics()` — called by orchestrator
- `LocalModelManager.runtime_state` — used by router, dispatcher
- `LocalModelManager.current_model` — used by many
- `LocalModelManager.is_loaded` — used by several
- `LocalModelManager.swap_started_at` — used by dispatcher
- `LocalModelManager.idle_seconds` — used by orchestrator
- `LocalModelManager.run_idle_unloader()` — started by orchestrator
- `LocalModelManager.run_health_watchdog()` — started by orchestrator
- `ModelSwapRequest` — not used externally
- `ModelRuntimeState` — used by router, dispatcher

The shim must provide **backward-compatible wrappers** for all of the above. The dispatcher integration changes (checking `dallama.status.busy` instead of `acquire_inference_slot`) are a separate future task — this shim keeps everything working as-is first.

- [ ] **Step 1: Write the shim test**

```python
# tests/test_dallama_shim.py
"""Verify the DaLLaMa shim preserves backward compatibility."""
from unittest.mock import patch, AsyncMock, MagicMock
import pytest


def test_shim_imports():
    """All existing import paths still work."""
    from src.models.local_model_manager import get_local_manager
    from src.models.local_model_manager import get_runtime_state
    from src.models.local_model_manager import ModelRuntimeState
    from src.models.local_model_manager import LocalModelManager


def test_get_local_manager_returns_instance():
    from src.models.local_model_manager import get_local_manager
    mgr = get_local_manager()
    assert mgr is not None
    assert hasattr(mgr, "ensure_model")
    assert hasattr(mgr, "get_status")
    assert hasattr(mgr, "current_model")
    assert hasattr(mgr, "is_loaded")
```

- [ ] **Step 2: Run test to verify current code passes**

Run: `pytest tests/test_dallama_shim.py -v`
Expected: PASS (current code already exports these).

- [ ] **Step 3: Replace local_model_manager.py with shim**

Replace the entire file content with a backward-compatible shim. The shim wraps DaLLaMa and translates between the old API and the new one. Read the current file carefully before writing the shim to ensure all consumer-facing attributes and methods are preserved.

The shim should:
1. Create a `DaLLaMa` instance with config from environment variables
2. Wrap it in a `LocalModelManager` class that provides the old API signatures
3. Translate `ModelInfo` → `ServerConfig` in `ensure_model()`
4. Delegate `get_status()`, `get_metrics()`, health watchdog, idle unloader to DaLLaMa
5. Keep `acquire_inference_slot` / `release_inference_slot` as passthroughs to DaLLaMa's `infer()` tracking (for now — full removal is a future dispatcher refactor)
6. Keep `ModelRuntimeState` dataclass and `get_runtime_state()` for backward compat
7. Keep `swap_started_at` property that reads from `dallama._swap.swap_in_progress`
8. Keep the atexit handler for orphan cleanup

This is the most complex task — read the existing consumers (listed above) to ensure nothing breaks. The shim should be ~80-120 lines.

- [ ] **Step 4: Run shim test**

Run: `pytest tests/test_dallama_shim.py -v`
Expected: PASS.

- [ ] **Step 5: Run existing KutAI tests that touch local_model_manager**

Run: `pytest tests/ -v -k "model" --timeout=30`
Expected: Existing tests PASS.

- [ ] **Step 6: Verify imports from all 8 consumer files**

Run:
```bash
python -c "from src.models.local_model_manager import get_local_manager; print('OK')"
python -c "from src.models.local_model_manager import get_runtime_state; print('OK')"
python -c "from src.core.llm_dispatcher import LLMDispatcher; print('OK')"
python -c "from src.core.router import call_model; print('OK')"
```
Expected: All print `OK`.

- [ ] **Step 7: Commit**

```bash
git add src/models/local_model_manager.py tests/test_dallama_shim.py
git commit -m "refactor(models): replace local_model_manager with DaLLaMa shim"
```

---

### Task 10: Delete gpu_scheduler.py + Clean Up References

**Files:**
- Delete: `src/models/gpu_scheduler.py`
- Modify: `src/core/orchestrator.py` (remove gpu_scheduler import)
- Modify: `src/infra/load_manager.py` (remove gpu_scheduler reference)

**Context:** `gpu_scheduler.py` is no longer needed — the dispatcher should not send work if DaLLaMa is busy (checking `dallama.status.busy`). The current consumers:

- `src/core/orchestrator.py:1023` — imports `get_gpu_scheduler` for status display
- `src/core/router.py:1268` — calls `acquire_inference_slot` (via local_model_manager wrapper)
- `src/infra/load_manager.py:9` — mentions gpu_scheduler in a comment

The `acquire_inference_slot` / `release_inference_slot` calls in router.py are preserved in the shim for now (they map to DaLLaMa's inference tracking). The direct `get_gpu_scheduler()` import in orchestrator.py needs to be removed or replaced with DaLLaMa status.

- [ ] **Step 1: Find all gpu_scheduler references**

Run: `grep -rn "gpu_scheduler\|GPURequest\|get_gpu_scheduler" src/ --include="*.py" | grep -v __pycache__`
Note exact files and line numbers.

- [ ] **Step 2: Update orchestrator.py**

Find the `get_gpu_scheduler` import (around line 1023-1025) and replace the scheduler status with DaLLaMa status. Read the surrounding code to understand what status fields are needed.

- [ ] **Step 3: Update load_manager.py**

Remove the comment referencing gpu_scheduler (line 9). Read the file first to understand the context.

- [ ] **Step 4: Delete gpu_scheduler.py**

```bash
git rm src/models/gpu_scheduler.py
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/ -v --timeout=30`
Expected: All PASS. If any test imports gpu_scheduler directly, update or remove it.

- [ ] **Step 6: Verify no broken imports**

Run: `python -c "from src.core.orchestrator import Orchestrator; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(models): remove gpu_scheduler — DaLLaMa handles inference tracking"
```

---

### Task 11: Full Regression Test

**Files:** None — testing only.

- [ ] **Step 1: Run all DaLLaMa package tests**

Run: `pytest packages/dallama/tests/ -v`
Expected: All PASS.

- [ ] **Step 2: Run all KutAI tests**

Run: `pytest tests/ -v --timeout=60`
Expected: All PASS.

- [ ] **Step 3: Verify import chain for all 8 consumer files**

Run:
```bash
python -c "from src.app.api import app; print('api OK')"
python -c "from src.app.telegram_bot import TelegramInterface; print('telegram OK')"
python -c "from src.core.llm_dispatcher import LLMDispatcher; print('dispatcher OK')"
python -c "from src.core.orchestrator import Orchestrator; print('orchestrator OK')"
python -c "from src.core.router import call_model; print('router OK')"
python -c "from src.infra.load_manager import LoadManager; print('load_manager OK')"
python -c "from src.tools.vision import run_vision_task; print('vision OK')"
```
Expected: All print OK (some may need adjusted import paths based on actual exports — check each file).

- [ ] **Step 4: Verify DaLLaMa package can be imported standalone**

Run: `python -c "from dallama import DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus, DaLLaMaLoadError; d = DaLLaMa(DaLLaMaConfig()); print(d.status); print('standalone OK')"`
Expected: Prints a ServerStatus with None model and `standalone OK`.
