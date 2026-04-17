# Yaşar Usta — Process Guard Extraction

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a generic Telegram-controlled process manager from `kutai_wrapper.py` into a standalone package called `yasar-usta` under `packages/yasar_usta/`. KutAI's `kutai_wrapper.py` becomes a thin consumer that configures and runs the package with KutAI-specific settings.

**Architecture:** The package provides `ProcessGuard` — a class that manages any subprocess via Telegram. It owns: lock management, subprocess lifecycle, heartbeat watchdog, escalating backoff, Telegram polling (non-destructive offset), Claude Code remote trigger, log viewer subprocess management, status panel, and built-in commands (start/stop/restart/status/logs/remote). All user-facing strings are configurable via a `messages` dict for i18n. KutAI-specific behavior (orphan llama-server killing, custom exit code meanings) is injected via hooks/config.

**Tech Stack:** Python 3.10+, aiohttp (for Telegram API), setuptools, pytest

---

## File Structure

```
packages/yasar_usta/
├── pyproject.toml
└── src/yasar_usta/
    ├── __init__.py              # Public API: ProcessGuard, GuardConfig, Messages
    ├── config.py                # GuardConfig dataclass, Messages dataclass with defaults
    ├── lock.py                  # Cross-platform single-instance lock (msvcrt/fcntl)
    ├── subprocess_mgr.py        # Start/stop/wait, output piping, heartbeat watchdog
    ├── backoff.py               # Escalating backoff with stability reset
    ├── telegram.py              # Telegram API (send, edit, poll loop, callback handling)
    ├── commands.py              # Built-in command handlers (start/stop/restart/status/logs)
    ├── remote.py                # Claude Code remote-control trigger
    ├── sidecar.py               # Sidecar subprocess management (log viewer etc.)
    ├── status.py                # Status panel builder (process list, heartbeat health)
    └── guard.py                 # ProcessGuard main class — ties everything together

Modified KutAI files:
  kutai_wrapper.py              # Becomes thin consumer: import ProcessGuard, configure, run
```

**Why this split:**
- `lock.py` is pure OS-level code, no async, no Telegram — testable in isolation
- `backoff.py` is pure math — testable in isolation
- `telegram.py` handles all HTTP to Telegram API — single point for mocking in tests
- `commands.py` contains the poll loop command dispatch — depends on telegram.py
- `subprocess_mgr.py` handles the child process lifecycle — depends on backoff.py
- `remote.py` is optional (feature-detected) — isolated so it doesn't pollute core
- `sidecar.py` manages auxiliary processes (log viewer) — generic pattern
- `status.py` builds the status panel — depends on subprocess state + sidecar state
- `guard.py` is the orchestrator that wires everything together
- `config.py` centralizes all configuration and i18n strings

---

### Task 1: Create package skeleton with config and lock modules

**Files:**
- Create: `packages/yasar_usta/pyproject.toml`
- Create: `packages/yasar_usta/src/yasar_usta/__init__.py`
- Create: `packages/yasar_usta/src/yasar_usta/config.py`
- Create: `packages/yasar_usta/src/yasar_usta/lock.py`
- Create: `packages/yasar_usta/.gitignore`
- Test: `packages/yasar_usta/tests/test_lock.py`
- Test: `packages/yasar_usta/tests/test_config.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p packages/yasar_usta/src/yasar_usta
mkdir -p packages/yasar_usta/tests
```

- [ ] **Step 2: Write pyproject.toml**

Write `packages/yasar_usta/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "yasar-usta"
version = "0.1.0"
description = "Telegram-controlled process manager with heartbeat watchdog and auto-restart"
requires-python = ">=3.10"
dependencies = ["aiohttp>=3.9.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write .gitignore**

Write `packages/yasar_usta/.gitignore`:

```
*.egg-info/
__pycache__/
```

- [ ] **Step 4: Write config.py — all configuration and i18n**

Write `packages/yasar_usta/src/yasar_usta/config.py`:

```python
"""Configuration and i18n for yasar-usta."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Messages:
    """All user-facing strings. Override for i18n."""

    # Notifications
    announce: str = "🔧 *{name} Process Manager*\n\nStarting {app_name}..."
    started: str = "✅ *{app_name} Started*"
    stopped: str = "⏹ *{app_name} Stopped*\nSend /start to restart."
    crash: str = (
        "🔴 *{app_name} Crashed*\n"
        "Exit code: `{exit_code}`\n"
        "Crash #{crash_count}\n"
        "Restarting in {backoff}s\n\n"
        "```\n{stderr}\n```"
    )
    hung: str = "🔴 {app_name} not responding — restarting in {delay}s"
    restarting: str = "♻️ *{app_name} restarting...*"
    self_restarting: str = "🔄 *{name} restarting...*"

    # Down-state prompt
    down_prompt: str = "⚠️ {app_name} is down. Press the button to start."
    down_with_reason: str = "{reason}\n⚠️ {app_name} is down. Press the button to start."
    down_reply: str = "⏸ {app_name} is currently stopped."
    starting: str = "🚀 Starting {app_name}..."

    # Keyboard
    btn_start: str = "▶️ Start"
    btn_status: str = "🔧 Status"
    btn_system: str = "⚙️ System"
    btn_refresh: str = "🔄 Refresh"
    btn_restart_guard: str = "♻️ Restart {name}"
    btn_restart_sidecar: str = "📊 Restart {sidecar_name}"

    # Status panel
    status_title: str = "🔧 *{name}*\n"
    status_guard: str = "🔵 {name}: running ({uptime})"
    status_app_healthy: str = "💚 {app_name}: healthy (heartbeat {age}s ago)"
    status_app_unresponsive: str = "🔴 {app_name}: UNRESPONSIVE ({age}s silent)"
    status_app_down: str = "💀 {app_name}: not running"
    status_app_no_heartbeat: str = "⚪ {app_name}: no heartbeat file"
    status_crashes: str = "\nCrashes: {count}"
    status_updated: str = "\n\n_Last update: {time}_"

    # Logs
    no_log_file: str = "📋 No log file found."
    no_log_entries: str = "📋 No log entries found."
    log_error: str = "❌ Error reading logs: {error}"

    # Remote
    remote_starting: str = "🖥️ Starting Claude Code session..."
    remote_started: str = "🖥️ *Claude Code Remote Control*\n\n🔗 [Connect]({url})\n\nPID: `{pid}`"
    remote_started_no_url: str = "🖥️ *Claude Code Remote Control started*\nPID: `{pid}`"
    remote_not_found: str = "❌ `claude` command not found. Is Claude Code installed?"
    remote_failed: str = "❌ Failed to start Claude Code: `{error}`"

    # Errors
    process_list_error: str = "⚠️ Could not get process list: {error}"
    wrapper_error: str = "⚠️ *Guard Error*\n`{error}`\n\nGuard is still alive. Send /start to retry."


@dataclass
class SidecarConfig:
    """Configuration for a sidecar subprocess (e.g., log viewer)."""

    name: str = "sidecar"
    command: list[str] = field(default_factory=list)
    health_url: str | None = None
    health_timeout: float = 3.0
    pid_file: str | None = None
    detached: bool = True
    auto_start: bool = True
    auto_restart: bool = True


@dataclass
class GuardConfig:
    """All configuration for ProcessGuard."""

    # What to manage
    name: str = "Yaşar Usta"
    app_name: str = "App"
    command: list[str] = field(default_factory=list)
    cwd: str | None = None

    # Telegram
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # Backoff
    backoff_steps: list[int] = field(default_factory=lambda: [5, 15, 60, 300])
    backoff_reset_after: int = 600

    # Heartbeat
    heartbeat_file: str | None = None
    heartbeat_stale_seconds: int = 120
    heartbeat_healthy_seconds: int = 90

    # Exit codes
    restart_exit_code: int = 42

    # Directories
    log_dir: str = "logs"
    log_file: str | None = None

    # Process management
    auto_restart: bool = True
    stop_timeout: int = 30

    # Claude Code remote
    claude_enabled: bool = True
    claude_cmd: str | None = None
    claude_name: str | None = None
    claude_signal_file: str | None = None

    # Sidecar (log viewer etc.)
    sidecar: SidecarConfig | None = None

    # Hooks
    on_exit: None = None  # callable(exit_code: int) -> None, called after process exits

    # i18n
    messages: Messages = field(default_factory=Messages)

    # Extra commands
    extra_commands: dict = field(default_factory=dict)

    # Extra process names to check in status panel
    extra_processes: list[dict] = field(default_factory=list)
```

- [ ] **Step 5: Write lock.py — cross-platform single-instance lock**

Write `packages/yasar_usta/src/yasar_usta/lock.py`:

```python
"""Cross-platform single-instance lock with stale recovery."""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

_lock_handle = None
_lock_file: Path | None = None
_sentinel_file: Path | None = None
_PID_WIDTH = 10


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
        except Exception:
            pass
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False


def acquire_lock(lock_dir: str | Path, name: str = "guard") -> None:
    """Acquire an exclusive lock. Exits if another instance is running.

    Creates two files in lock_dir:
      {name}.lock  — PID (plain text, never locked, always readable)
      {name}.lk    — sentinel (OS-locked, content irrelevant)

    Args:
        lock_dir: Directory for lock files.
        name: Base name for lock files.

    Raises:
        SystemExit: If another instance holds the lock.
    """
    global _lock_handle, _lock_file, _sentinel_file

    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    _lock_file = lock_dir / f"{name}.lock"
    _sentinel_file = lock_dir / f"{name}.lk"

    try:
        import msvcrt
    except ImportError:
        msvcrt = None

    if msvcrt is not None:
        _acquire_lock_msvcrt(msvcrt)
    else:
        _acquire_lock_unix()

    atexit.register(release_lock)


def _acquire_lock_msvcrt(msvcrt) -> None:
    """Windows lock using a separate sentinel file."""
    global _lock_handle

    def _try_lock(fh):
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (OSError, IOError):
            return False

    def _write_pid():
        with open(_lock_file, "w") as f:
            f.write(str(os.getpid()).zfill(_PID_WIDTH))

    def _read_pid():
        try:
            raw = _lock_file.read_text().strip()
            return int(raw) if raw else None
        except (ValueError, OSError):
            return None

    if not _sentinel_file.exists():
        _sentinel_file.write_text("L")

    _lock_handle = open(_sentinel_file, "r+")

    if _try_lock(_lock_handle):
        _write_pid()
        return

    existing_pid = _read_pid()

    if existing_pid is not None and is_pid_alive(existing_pid):
        _lock_handle.close()
        _lock_handle = None
        print(f"ERROR: Already running (PID {existing_pid}).")
        sys.exit(1)

    print(f"Stale lock detected (PID {existing_pid or '?'} is dead). Cleaning up.")
    _lock_handle.close()
    _lock_handle = None
    try:
        _sentinel_file.unlink(missing_ok=True)
    except Exception:
        pass

    _sentinel_file.write_text("L")
    _lock_handle = open(_sentinel_file, "r+")
    if _try_lock(_lock_handle):
        _write_pid()
        return

    _lock_handle.close()
    _lock_handle = None
    print("ERROR: Could not acquire lock even after stale-lock cleanup.")
    sys.exit(1)


def _acquire_lock_unix() -> None:
    """Unix lock using fcntl, with PID-based fallback."""
    global _lock_handle
    try:
        import fcntl
        _lock_handle = open(_lock_file, "w")
        fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_handle.write(str(os.getpid()).zfill(_PID_WIDTH))
        _lock_handle.flush()
    except (OSError, IOError):
        print("ERROR: Already running.")
        sys.exit(1)
    except ImportError:
        if _lock_file.exists():
            try:
                old_pid = int(_lock_file.read_text().strip())
                if is_pid_alive(old_pid):
                    print(f"ERROR: Already running (PID {old_pid}).")
                    sys.exit(1)
                else:
                    print(f"Stale lock (PID {old_pid} is dead). Cleaning up.")
            except Exception:
                pass
        _lock_file.write_text(str(os.getpid()).zfill(_PID_WIDTH))


def release_lock() -> None:
    """Release lock and remove lock files."""
    global _lock_handle
    if _lock_handle:
        try:
            import msvcrt
            _lock_handle.seek(0)
            msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLOCK, _PID_WIDTH)
        except Exception:
            pass
        try:
            _lock_handle.close()
        except Exception:
            pass
        _lock_handle = None
    if _lock_file:
        try:
            _lock_file.unlink(missing_ok=True)
        except Exception:
            pass
```

- [ ] **Step 6: Write test_config.py**

Write `packages/yasar_usta/tests/test_config.py`:

```python
"""Tests for yasar_usta.config."""

from yasar_usta.config import GuardConfig, Messages, SidecarConfig


class TestMessages:
    def test_default_messages_have_no_empty_strings(self):
        msgs = Messages()
        for field_name in vars(msgs):
            val = getattr(msgs, field_name)
            assert isinstance(val, str) and len(val) > 0, f"Empty message: {field_name}"

    def test_messages_support_format_placeholders(self):
        msgs = Messages()
        result = msgs.started.format(app_name="MyApp")
        assert "MyApp" in result

    def test_custom_messages(self):
        msgs = Messages(started="✅ *{app_name} Başladı*")
        assert "Başladı" in msgs.started


class TestGuardConfig:
    def test_defaults(self):
        cfg = GuardConfig()
        assert cfg.backoff_steps == [5, 15, 60, 300]
        assert cfg.restart_exit_code == 42
        assert cfg.heartbeat_stale_seconds == 120
        assert cfg.auto_restart is True

    def test_custom_config(self):
        cfg = GuardConfig(
            name="Yaşar Usta",
            app_name="Kutay",
            command=["python", "run.py"],
            restart_exit_code=42,
            backoff_steps=[1, 5, 30],
        )
        assert cfg.app_name == "Kutay"
        assert cfg.backoff_steps == [1, 5, 30]

    def test_independent_list_defaults(self):
        a = GuardConfig()
        b = GuardConfig()
        a.backoff_steps.append(999)
        assert 999 not in b.backoff_steps


class TestSidecarConfig:
    def test_defaults(self):
        sc = SidecarConfig()
        assert sc.detached is True
        assert sc.auto_start is True
        assert sc.command == []

    def test_custom(self):
        sc = SidecarConfig(
            name="yazbunu",
            command=["python", "-m", "yazbunu.server"],
            health_url="http://127.0.0.1:9880/",
            pid_file="logs/yazbunu.pid",
        )
        assert sc.name == "yazbunu"
        assert sc.health_url == "http://127.0.0.1:9880/"
```

- [ ] **Step 7: Write test_lock.py**

Write `packages/yasar_usta/tests/test_lock.py`:

```python
"""Tests for yasar_usta.lock."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from yasar_usta.lock import is_pid_alive, acquire_lock, release_lock


class TestIsPidAlive:
    def test_current_process_is_alive(self):
        assert is_pid_alive(os.getpid()) is True

    def test_impossible_pid_is_dead(self):
        # PID 0 is kernel on unix, nonexistent on Windows
        # Use a very high PID unlikely to exist
        assert is_pid_alive(4_000_000) is False


class TestAcquireLock:
    def test_acquire_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            acquire_lock(tmp, name="test_guard")
            lock_file = Path(tmp) / "test_guard.lock"
            assert lock_file.exists()
            pid = int(lock_file.read_text().strip())
            assert pid == os.getpid()
            release_lock()

    def test_stale_lock_recovery(self):
        """If lock file has a dead PID, acquire should succeed."""
        with tempfile.TemporaryDirectory() as tmp:
            # Write a stale PID
            lock_file = Path(tmp) / "test_stale.lock"
            lock_file.write_text("0000000002")  # PID 2 is unlikely to be alive
            sentinel = Path(tmp) / "test_stale.lk"
            sentinel.write_text("L")
            # Should succeed (stale recovery)
            acquire_lock(tmp, name="test_stale")
            pid = int(lock_file.read_text().strip())
            assert pid == os.getpid()
            release_lock()
```

- [ ] **Step 8: Write initial __init__.py**

Write `packages/yasar_usta/src/yasar_usta/__init__.py`:

```python
"""Yaşar Usta — Telegram-controlled process manager.

Manages any subprocess with:
- Heartbeat-based hung detection
- Escalating backoff with stability reset
- Telegram as control plane when process is down
- Non-destructive offset polling (no message loss)
- Claude Code remote trigger
- Sidecar process management (log viewer etc.)
- Configurable i18n for all user-facing strings
"""

from .config import GuardConfig, Messages, SidecarConfig

__all__ = [
    "GuardConfig",
    "Messages",
    "SidecarConfig",
]
```

- [ ] **Step 9: Install and run tests**

```bash
pip install -e ./packages/yasar_usta
pytest packages/yasar_usta/tests/ -v
```

Expected: All tests pass.

- [ ] **Step 10: Commit**

```bash
git add packages/yasar_usta/
git commit -m "feat(yasar-usta): package skeleton with config and lock modules"
```

---

### Task 2: Backoff and subprocess manager

**Files:**
- Create: `packages/yasar_usta/src/yasar_usta/backoff.py`
- Create: `packages/yasar_usta/src/yasar_usta/subprocess_mgr.py`
- Test: `packages/yasar_usta/tests/test_backoff.py`
- Test: `packages/yasar_usta/tests/test_subprocess_mgr.py`

- [ ] **Step 1: Write test_backoff.py**

Write `packages/yasar_usta/tests/test_backoff.py`:

```python
"""Tests for yasar_usta.backoff."""

from yasar_usta.backoff import BackoffTracker


class TestBackoffTracker:
    def test_initial_backoff(self):
        bt = BackoffTracker(steps=[5, 15, 60, 300])
        assert bt.get_delay() == 5

    def test_escalation(self):
        bt = BackoffTracker(steps=[5, 15, 60, 300])
        bt.record_crash()
        assert bt.get_delay() == 15
        bt.record_crash()
        assert bt.get_delay() == 60

    def test_clamps_at_max(self):
        bt = BackoffTracker(steps=[5, 15])
        bt.record_crash()
        bt.record_crash()
        bt.record_crash()
        assert bt.get_delay() == 15

    def test_reset_after_stability(self):
        bt = BackoffTracker(steps=[5, 15], reset_after=10)
        bt.record_crash()
        bt.record_crash()
        assert bt.get_delay() == 15
        # Simulate stable run
        bt.mark_started()
        import time
        bt._start_time = time.time() - 11  # pretend 11s ago
        bt.maybe_reset()
        assert bt.get_delay() == 5
        assert bt.crash_count == 0

    def test_total_crashes_not_reset(self):
        bt = BackoffTracker(steps=[5, 15], reset_after=10)
        bt.record_crash()
        bt.record_crash()
        assert bt.total_crashes == 2
        bt.mark_started()
        import time
        bt._start_time = time.time() - 11
        bt.maybe_reset()
        assert bt.total_crashes == 2  # total preserved
        assert bt.crash_count == 0  # window reset
```

- [ ] **Step 2: Write backoff.py**

Write `packages/yasar_usta/src/yasar_usta/backoff.py`:

```python
"""Escalating backoff with stability reset."""

from __future__ import annotations

import time


class BackoffTracker:
    """Tracks crash count and computes escalating backoff delays.

    Args:
        steps: List of delay seconds, indexed by crash count.
        reset_after: Reset crash count if process runs longer than this (seconds).
    """

    def __init__(self, steps: list[int] | None = None, reset_after: int = 600):
        self.steps = steps or [5, 15, 60, 300]
        self.reset_after = reset_after
        self.crash_count: int = 0
        self.total_crashes: int = 0
        self._start_time: float | None = None
        self.last_crash_time: float = 0

    def get_delay(self) -> int:
        """Return the current backoff delay in seconds."""
        idx = min(self.crash_count, len(self.steps) - 1)
        return self.steps[idx]

    def record_crash(self) -> None:
        """Record a crash event."""
        self.crash_count += 1
        self.total_crashes += 1
        self.last_crash_time = time.time()

    def mark_started(self) -> None:
        """Mark that the managed process has started."""
        self._start_time = time.time()

    def maybe_reset(self) -> None:
        """Reset crash count if process has been stable long enough."""
        if self._start_time and (time.time() - self._start_time) > self.reset_after:
            self.crash_count = 0
```

- [ ] **Step 3: Write test_subprocess_mgr.py**

Write `packages/yasar_usta/tests/test_subprocess_mgr.py`:

```python
"""Tests for yasar_usta.subprocess_mgr."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from yasar_usta.subprocess_mgr import SubprocessManager


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSubprocessManager:
    def test_start_and_wait(self):
        """Start a trivial process, wait for clean exit."""
        python = sys.executable
        mgr = SubprocessManager(
            command=[python, "-c", "import sys; sys.exit(0)"],
            log_dir=tempfile.mkdtemp(),
        )
        run_async(mgr.start())
        assert mgr.running is True
        code = run_async(mgr.wait_for_exit())
        assert code == 0
        assert mgr.running is False

    def test_crash_exit_code(self):
        """Process that exits with code 1."""
        python = sys.executable
        mgr = SubprocessManager(
            command=[python, "-c", "import sys; sys.exit(1)"],
            log_dir=tempfile.mkdtemp(),
        )
        run_async(mgr.start())
        code = run_async(mgr.wait_for_exit())
        assert code == 1

    def test_stop_graceful(self):
        """Stop a long-running process gracefully."""
        python = sys.executable
        mgr = SubprocessManager(
            command=[python, "-c", "import time; time.sleep(60)"],
            log_dir=tempfile.mkdtemp(),
            stop_timeout=5,
        )
        run_async(mgr.start())
        assert mgr.running is True
        run_async(mgr.stop())
        assert mgr.running is False

    def test_stderr_capture(self):
        """Stderr lines are captured in the tail buffer."""
        python = sys.executable
        mgr = SubprocessManager(
            command=[python, "-c", "import sys; sys.stderr.write('error line\\n'); sys.exit(0)"],
            log_dir=tempfile.mkdtemp(),
        )
        run_async(mgr.start())
        run_async(mgr.wait_for_exit())
        # Give pipe reader a moment to finish
        run_async(asyncio.sleep(0.2))
        assert any("error line" in line for line in mgr.stderr_tail)

    def test_heartbeat_detection(self):
        """Heartbeat file check works."""
        with tempfile.TemporaryDirectory() as tmp:
            hb_file = Path(tmp) / "heartbeat"
            import time
            hb_file.write_text(str(time.time()))

            mgr = SubprocessManager(
                command=["echo", "noop"],
                log_dir=tmp,
                heartbeat_file=str(hb_file),
                heartbeat_stale_seconds=120,
            )
            assert mgr.is_heartbeat_stale() is False

            # Write old timestamp
            hb_file.write_text(str(time.time() - 200))
            assert mgr.is_heartbeat_stale() is True

    def test_no_heartbeat_file_not_stale(self):
        """Missing heartbeat file = not stale (still starting up)."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SubprocessManager(
                command=["echo", "noop"],
                log_dir=tmp,
                heartbeat_file=str(Path(tmp) / "nonexistent"),
            )
            assert mgr.is_heartbeat_stale() is False
```

- [ ] **Step 4: Write subprocess_mgr.py**

Write `packages/yasar_usta/src/yasar_usta/subprocess_mgr.py`:

```python
"""Subprocess lifecycle management with heartbeat watchdog."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger("yasar_usta.subprocess")


class SubprocessManager:
    """Manages a single subprocess with output piping and heartbeat monitoring.

    Args:
        command: Command to run as a subprocess.
        log_dir: Directory for log files.
        cwd: Working directory for the subprocess (defaults to current dir).
        stop_timeout: Seconds to wait for graceful shutdown before killing.
        heartbeat_file: Path to heartbeat timestamp file (optional).
        heartbeat_stale_seconds: Kill process if heartbeat older than this.
    """

    def __init__(
        self,
        command: list[str],
        log_dir: str,
        cwd: str | None = None,
        stop_timeout: int = 30,
        heartbeat_file: str | None = None,
        heartbeat_stale_seconds: int = 120,
    ):
        self.command = command
        self.log_dir = Path(log_dir)
        self.cwd = cwd
        self.stop_timeout = stop_timeout
        self.heartbeat_file = heartbeat_file
        self.heartbeat_stale_seconds = heartbeat_stale_seconds

        self.process: asyncio.subprocess.Process | None = None
        self.running: bool = False
        self.start_time: float | None = None
        self.last_exit_code: int | None = None
        self.stderr_tail: deque[str] = deque(maxlen=50)
        self._stop_requested: bool = False

    async def start(self) -> None:
        """Launch the subprocess."""
        if self.process and self.process.returncode is None:
            return

        self.process = None
        self.stderr_tail.clear()
        self._stop_requested = False

        self.log_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {}
        if sys.platform == "win32":
            import subprocess as _sp
            kwargs["creationflags"] = (
                _sp.CREATE_NEW_PROCESS_GROUP | _sp.CREATE_NO_WINDOW
            )

        logger.info("Starting subprocess: %s", " ".join(self.command))
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
                cwd=self.cwd,
                **kwargs,
            )
        except Exception as e:
            logger.error("Failed to spawn subprocess: %s", e)
            self.process = None
            self.running = False
            return

        self.running = True
        self.start_time = time.time()

        # Reset heartbeat so hung detector doesn't see stale timestamp
        if self.heartbeat_file:
            try:
                with open(self.heartbeat_file, "w") as f:
                    f.write(str(time.time()))
            except Exception:
                pass

        asyncio.create_task(self._pipe_output(self.process.stdout, "stdout"))
        asyncio.create_task(self._pipe_output(self.process.stderr, "stderr"))

    async def stop(self, timeout: int | None = None) -> None:
        """Send SIGINT and wait for graceful shutdown."""
        if not self.process or self.process.returncode is not None:
            return
        self._stop_requested = True
        timeout = timeout or self.stop_timeout
        logger.info("Sending shutdown signal...")
        try:
            self.process.send_signal(signal.SIGINT)
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Graceful shutdown timed out, killing...")
            self.process.kill()
            await self.process.wait()
        self.running = False

    async def wait_for_exit(self) -> int:
        """Wait for the subprocess to exit. Returns exit code (-1 if hung/killed)."""
        if not self.process:
            self.running = False
            return -1
        if self.process.returncode is not None:
            code = self.process.returncode
            self.process = None
            self.running = False
            self.last_exit_code = code
            return code

        while True:
            try:
                code = await asyncio.wait_for(self.process.wait(), timeout=30)
                self.process = None
                self.running = False
                self.last_exit_code = code
                return code
            except asyncio.TimeoutError:
                if self.is_heartbeat_stale():
                    logger.error(
                        "Heartbeat stale >%ds — killing hung process",
                        self.heartbeat_stale_seconds,
                    )
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                    self.process = None
                    self.running = False
                    self.last_exit_code = -1
                    return -1

    def is_heartbeat_stale(self) -> bool:
        """Check if the heartbeat file is older than the stale threshold."""
        if not self.heartbeat_file:
            return False
        try:
            with open(self.heartbeat_file, "r") as f:
                last_beat = float(f.read().strip())
            return (time.time() - last_beat) > self.heartbeat_stale_seconds
        except (FileNotFoundError, ValueError):
            return False

    def is_heartbeat_healthy(self, healthy_seconds: int = 90) -> bool:
        """Check if heartbeat is fresh (within healthy_seconds)."""
        if not self.heartbeat_file:
            return False
        try:
            ts = float(Path(self.heartbeat_file).read_text().strip())
            return (time.time() - ts) < healthy_seconds
        except (FileNotFoundError, ValueError, OSError):
            return False

    def heartbeat_age(self) -> float | None:
        """Return seconds since last heartbeat, or None if no heartbeat."""
        if not self.heartbeat_file:
            return None
        try:
            ts = float(Path(self.heartbeat_file).read_text().strip())
            return time.time() - ts
        except (FileNotFoundError, ValueError, OSError):
            return None

    async def _pipe_output(self, stream, name: str) -> None:
        """Read subprocess output line by line, print to console and save."""
        log_file = self.log_dir / "guard.log"
        while True:
            try:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
            except ValueError:
                try:
                    await stream.readuntil(b"\n")
                except Exception:
                    try:
                        stream._buffer.clear()
                    except Exception:
                        pass
                continue
            except Exception:
                break

            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode())
            except Exception:
                pass

            if name == "stderr":
                self.stderr_tail.append(line)
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{name}] {line}\n")
            except Exception:
                pass
```

- [ ] **Step 5: Run tests**

```bash
pytest packages/yasar_usta/tests/test_backoff.py packages/yasar_usta/tests/test_subprocess_mgr.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/
git commit -m "feat(yasar-usta): add backoff tracker and subprocess manager"
```

---

### Task 3: Telegram API, sidecar, remote, and status modules

**Files:**
- Create: `packages/yasar_usta/src/yasar_usta/telegram.py`
- Create: `packages/yasar_usta/src/yasar_usta/sidecar.py`
- Create: `packages/yasar_usta/src/yasar_usta/remote.py`
- Create: `packages/yasar_usta/src/yasar_usta/status.py`
- Test: `packages/yasar_usta/tests/test_telegram.py`
- Test: `packages/yasar_usta/tests/test_sidecar.py`
- Test: `packages/yasar_usta/tests/test_status.py`

This is the largest task. It creates the Telegram API layer, sidecar manager, Claude remote trigger, and status panel builder.

- [ ] **Step 1: Write telegram.py**

Write `packages/yasar_usta/src/yasar_usta/telegram.py`:

```python
"""Telegram Bot API helpers — send, edit, poll."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("yasar_usta.telegram")


class TelegramAPI:
    """Minimal Telegram Bot API client using aiohttp.

    Args:
        token: Bot token.
        chat_id: Admin chat ID.
    """

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{token}"

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(
        self,
        text: str,
        reply_markup: dict | None = None,
        parse_mode: str = "Markdown",
    ) -> dict | None:
        """Send a message to the admin chat."""
        if not self.enabled:
            return None
        import aiohttp
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/sendMessage",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    return await resp.json()
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return None

    async def edit(
        self,
        message_id: int,
        text: str,
        reply_markup: dict | None = None,
        parse_mode: str = "Markdown",
    ) -> dict | None:
        """Edit an existing message."""
        if not self.enabled:
            return None
        import aiohttp
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/editMessageText",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    return await resp.json()
        except Exception as e:
            logger.warning("Telegram edit failed: %s", e)
            return None

    async def answer_callback(self, callback_query_id: str) -> None:
        """Answer a callback query (removes loading spinner)."""
        if not self.enabled:
            return
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{self._base_url}/answerCallbackQuery",
                    json={"callback_query_id": callback_query_id},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception:
            pass

    async def get_updates(self, offset: int = 0, timeout: int = 5) -> list[dict]:
        """Long-poll for updates."""
        if not self.enabled:
            return []
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/getUpdates",
                    params={"offset": offset, "timeout": timeout},
                    timeout=aiohttp.ClientTimeout(total=timeout + 10),
                ) as resp:
                    data = await resp.json()
            return data.get("result", [])
        except Exception as e:
            logger.warning("Telegram poll error: %s", e)
            return []

    async def flush_updates(self) -> None:
        """Confirm all pending updates (used before self-restart)."""
        if not self.enabled:
            return
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(
                    f"{self._base_url}/getUpdates",
                    params={"offset": -1, "timeout": 0},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception:
            pass
```

- [ ] **Step 2: Write sidecar.py**

Write `packages/yasar_usta/src/yasar_usta/sidecar.py`:

```python
"""Sidecar subprocess management (log viewer, etc.)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from .lock import is_pid_alive

logger = logging.getLogger("yasar_usta.sidecar")


class SidecarManager:
    """Manages a detached sidecar subprocess with PID file tracking.

    Args:
        name: Display name.
        command: Command to run.
        pid_file: Path to PID file.
        health_url: HTTP URL to check for liveness (optional).
        health_timeout: Timeout for health check (seconds).
        log_file: Path to redirect sidecar stdout/stderr.
        cwd: Working directory.
        detached: Run as detached process (survives parent death).
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        pid_file: str | None = None,
        health_url: str | None = None,
        health_timeout: float = 3.0,
        log_file: str | None = None,
        cwd: str | None = None,
        detached: bool = True,
    ):
        self.name = name
        self.command = command
        self.pid_file = Path(pid_file) if pid_file else None
        self.health_url = health_url
        self.health_timeout = health_timeout
        self.log_file = Path(log_file) if log_file else None
        self.cwd = cwd
        self.detached = detached

    def pid_alive(self) -> int | None:
        """Return PID if the sidecar process is alive, else None."""
        if not self.pid_file:
            return None
        try:
            pid = int(self.pid_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None
        if is_pid_alive(pid):
            return pid
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    async def http_alive(self) -> bool:
        """Check if sidecar is responding on its health URL."""
        if not self.health_url:
            return False
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    self.health_url,
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout),
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    async def is_alive(self) -> bool:
        """Check if sidecar is alive (PID or HTTP)."""
        if self.pid_alive():
            return True
        return await self.http_alive()

    async def start(self) -> None:
        """Start the sidecar subprocess."""
        if await self.is_alive():
            logger.info("%s already running", self.name)
            return

        if not self.command:
            logger.warning("No command configured for sidecar %s", self.name)
            return

        logger.info("Starting %s", self.name)
        try:
            import subprocess as _sp
            kwargs = {}
            out_fh = None
            if self.detached and sys.platform == "win32":
                kwargs["creationflags"] = (
                    _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
                )
                kwargs["close_fds"] = True
            if self.log_file:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                out_fh = open(self.log_file, "a", encoding="utf-8")
                kwargs["stdout"] = out_fh
                kwargs["stderr"] = out_fh

            proc = _sp.Popen(
                self.command,
                cwd=self.cwd,
                **kwargs,
            )
            if self.pid_file:
                self.pid_file.parent.mkdir(parents=True, exist_ok=True)
                self.pid_file.write_text(str(proc.pid))
            logger.info("%s started (PID %d)", self.name, proc.pid)
        except Exception as e:
            logger.error("Failed to start %s: %s", self.name, e)

    async def stop(self) -> None:
        """Kill the sidecar by PID."""
        pid = self.pid_alive()
        if not pid:
            return
        logger.info("Stopping %s (PID %d)", self.name, pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            logger.error("%s stop error: %s", self.name, e)
        if self.pid_file:
            try:
                self.pid_file.unlink(missing_ok=True)
            except Exception:
                pass

    async def ensure(self) -> None:
        """Restart sidecar if it's dead."""
        if await self.is_alive():
            return
        logger.warning("%s not running, restarting", self.name)
        await self.start()
```

- [ ] **Step 3: Write remote.py**

Write `packages/yasar_usta/src/yasar_usta/remote.py`:

```python
"""Claude Code remote-control trigger."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger("yasar_usta.remote")


def find_claude_cmd(custom_path: str | None = None) -> str | None:
    """Find the claude CLI command. Returns None if not installed."""
    if custom_path and Path(custom_path).exists():
        return custom_path
    # Check common locations
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        candidate = Path(appdata) / "npm" / "claude.cmd"
        if candidate.exists():
            return str(candidate)
    # Check PATH
    import shutil
    which = shutil.which("claude")
    if which:
        return which
    return None


async def start_claude_remote(
    claude_cmd: str,
    name: str = "App",
    cwd: str | None = None,
) -> tuple[asyncio.subprocess.Process | None, str | None]:
    """Start a Claude Code remote-control session.

    Returns:
        (process, session_url) — process may be None on failure,
        session_url may be None if URL couldn't be extracted.
    """
    logger.info("Starting Claude Code remote-control server")
    try:
        proc = await asyncio.create_subprocess_exec(
            claude_cmd, "remote-control",
            "--name", name,
            "--permission-mode", "bypassPermissions",
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("claude command not found at %s", claude_cmd)
        return None, None
    except Exception as e:
        logger.error("Failed to start Claude remote-control: %s", e)
        return None, None

    # Read stdout to capture session URL
    session_url = None
    try:
        for _ in range(20):
            line_bytes = await asyncio.wait_for(
                proc.stdout.readline(), timeout=10,
            )
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            logger.info("[claude-rc] %s", line)
            if "claude.ai" in line or "http" in line.lower():
                url_match = re.search(r"https?://\S+", line)
                if url_match:
                    session_url = url_match.group(0)
                    break
    except asyncio.TimeoutError:
        pass

    return proc, session_url
```

- [ ] **Step 4: Write status.py**

Write `packages/yasar_usta/src/yasar_usta/status.py`:

```python
"""Status panel builder."""

from __future__ import annotations

import logging
import subprocess as _sp
import sys
import time

logger = logging.getLogger("yasar_usta.status")


def build_status_text(
    *,
    name: str,
    app_name: str,
    guard_start_time: float,
    app_running: bool,
    heartbeat_age: float | None,
    heartbeat_healthy_seconds: int,
    total_crashes: int,
    sidecar_name: str | None = None,
    sidecar_alive: bool = False,
    sidecar_pid: int | None = None,
    sidecar_health_url: str | None = None,
    sidecar_http_alive: bool = False,
    extra_processes: list[dict] | None = None,
    messages=None,
) -> str:
    """Build the status panel text.

    Args:
        extra_processes: List of dicts with keys: name, exe (process name to check),
            label (display label).
    """
    uptime_w = int(time.time() - guard_start_time)
    uptime_str = f"{uptime_w // 3600}h {(uptime_w % 3600) // 60}m"

    lines = [f"🔧 *{name}*\n"]
    lines.append(f"🔵 {name}: running ({uptime_str})")

    # App health
    if not app_running:
        lines.append(f"💀 {app_name}: not running")
    elif heartbeat_age is not None and heartbeat_age < heartbeat_healthy_seconds:
        lines.append(f"💚 {app_name}: healthy (heartbeat {int(heartbeat_age)}s ago)")
    elif heartbeat_age is not None:
        lines.append(f"🔴 {app_name}: UNRESPONSIVE ({int(heartbeat_age)}s silent)")
    else:
        lines.append(f"⚪ {app_name}: no heartbeat file")

    # Extra processes (e.g., llama-server)
    for proc_info in (extra_processes or []):
        exe = proc_info.get("exe", "")
        label = proc_info.get("label", exe)
        found = _check_process_running(exe)
        if found:
            lines.append(f"🟡 {label}: running")
        else:
            lines.append(f"⚫ {label}: not running")

    # Sidecar
    if sidecar_name:
        if sidecar_http_alive:
            pid_str = f", PID {sidecar_pid}" if sidecar_pid else ""
            lines.append(f"📊 {sidecar_name}: running ({sidecar_health_url}{pid_str})")
        elif sidecar_pid:
            lines.append(f"🟠 {sidecar_name}: process alive but not responding (PID {sidecar_pid})")
        elif sidecar_alive:
            lines.append(f"🟢 {sidecar_name}: running")
        else:
            lines.append(f"⚫ {sidecar_name}: not running")

    lines.append(f"\nCrashes: {total_crashes}")
    ts = time.strftime("%H:%M:%S")
    lines.append(f"\n_Last update: {ts}_")
    return "\n".join(lines)


def _check_process_running(exe_name: str) -> bool:
    """Check if a process with given executable name is running (Windows only for now)."""
    if not exe_name:
        return False
    if sys.platform == "win32":
        try:
            result = _sp.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return exe_name.lower() in result.stdout.lower()
        except Exception:
            return False
    else:
        try:
            result = _sp.run(
                ["pgrep", "-f", exe_name],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
```

- [ ] **Step 5: Write test_telegram.py**

Write `packages/yasar_usta/tests/test_telegram.py`:

```python
"""Tests for yasar_usta.telegram."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from yasar_usta.telegram import TelegramAPI


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestTelegramAPI:
    def test_disabled_when_no_token(self):
        api = TelegramAPI(token="", chat_id="123")
        assert api.enabled is False

    def test_disabled_when_no_chat_id(self):
        api = TelegramAPI(token="tok", chat_id="")
        assert api.enabled is False

    def test_enabled_with_both(self):
        api = TelegramAPI(token="tok", chat_id="123")
        assert api.enabled is True

    def test_send_returns_none_when_disabled(self):
        api = TelegramAPI(token="", chat_id="")
        result = run_async(api.send("test"))
        assert result is None

    def test_get_updates_returns_empty_when_disabled(self):
        api = TelegramAPI(token="", chat_id="")
        result = run_async(api.get_updates())
        assert result == []


class TestSidecar:
    """Basic import test for sidecar module."""
    def test_import(self):
        from yasar_usta.sidecar import SidecarManager
        mgr = SidecarManager(name="test", command=["echo", "hi"])
        assert mgr.name == "test"


class TestStatus:
    def test_build_status_text(self):
        from yasar_usta.status import build_status_text
        import time
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time() - 3600,
            app_running=True,
            heartbeat_age=5.0,
            heartbeat_healthy_seconds=90,
            total_crashes=2,
        )
        assert "Guard" in text
        assert "MyApp" in text
        assert "healthy" in text
        assert "Crashes: 2" in text

    def test_build_status_app_down(self):
        from yasar_usta.status import build_status_text
        import time
        text = build_status_text(
            name="Guard",
            app_name="MyApp",
            guard_start_time=time.time(),
            app_running=False,
            heartbeat_age=None,
            heartbeat_healthy_seconds=90,
            total_crashes=0,
        )
        assert "not running" in text
```

- [ ] **Step 6: Write test_sidecar.py**

Write `packages/yasar_usta/tests/test_sidecar.py`:

```python
"""Tests for yasar_usta.sidecar."""

import os
import sys
import tempfile
import asyncio
from pathlib import Path

from yasar_usta.sidecar import SidecarManager


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSidecarManager:
    def test_pid_alive_no_file(self):
        mgr = SidecarManager(name="test", command=["echo"])
        assert mgr.pid_alive() is None

    def test_pid_alive_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "test.pid"
            pid_file.write_text("4000000")  # unlikely PID
            mgr = SidecarManager(
                name="test",
                command=["echo"],
                pid_file=str(pid_file),
            )
            assert mgr.pid_alive() is None
            assert not pid_file.exists()  # cleaned up

    def test_pid_alive_current_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "test.pid"
            pid_file.write_text(str(os.getpid()))
            mgr = SidecarManager(
                name="test",
                command=["echo"],
                pid_file=str(pid_file),
            )
            assert mgr.pid_alive() == os.getpid()

    def test_start_with_no_command(self):
        mgr = SidecarManager(name="test", command=[])
        run_async(mgr.start())
        # Should not crash, just log warning
```

- [ ] **Step 7: Run all tests**

```bash
pytest packages/yasar_usta/tests/ -v
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add packages/yasar_usta/
git commit -m "feat(yasar-usta): add telegram, sidecar, remote, and status modules"
```

---

### Task 4: ProcessGuard main class and command handling

**Files:**
- Create: `packages/yasar_usta/src/yasar_usta/commands.py`
- Create: `packages/yasar_usta/src/yasar_usta/guard.py`
- Modify: `packages/yasar_usta/src/yasar_usta/__init__.py`
- Test: `packages/yasar_usta/tests/test_guard.py`

This is the core task — the `ProcessGuard` class that ties everything together with the Telegram poll loop and command dispatch.

- [ ] **Step 1: Write commands.py — keyboard builders and log formatter**

Write `packages/yasar_usta/src/yasar_usta/commands.py`:

```python
"""Keyboard builders and log formatting helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Messages

logger = logging.getLogger("yasar_usta.commands")


def build_start_keyboard(messages: Messages) -> dict:
    """Build the reply keyboard shown when the app is down."""
    return {
        "keyboard": [
            [{"text": messages.btn_start}, {"text": messages.btn_status}],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }


def build_status_inline_keyboard(messages: Messages, name: str, sidecar_name: str | None = None) -> dict:
    """Build inline keyboard for the status panel."""
    buttons = [
        [{"text": messages.btn_refresh, "callback_data": "guard_refresh"}],
        [{"text": messages.btn_restart_guard.format(name=name), "callback_data": "restart_guard"}],
    ]
    if sidecar_name:
        buttons[1].append({
            "text": messages.btn_restart_sidecar.format(sidecar_name=sidecar_name),
            "callback_data": "restart_sidecar",
        })
    return {"inline_keyboard": buttons}


def format_log_entries(log_path: str | Path, n: int = 20) -> str | None:
    """Read and format the last N lines of a JSONL log file.

    Returns formatted text or None if no entries found.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 100_000))
            chunk = f.read()
            lines = chunk.strip().split("\n")

        last_n = lines[-n:]
        formatted = []
        for line in last_n:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = entry.get("ts", "?")
                if "T" in ts:
                    ts = ts.split("T")[1][:8]
                elif " " in ts:
                    ts = ts.split(" ")[1][:8]
                level = entry.get("level", "?")[:4]
                comp = entry.get("src", "?").split(".")[-1]
                msg = entry.get("msg", "")[:120]
                icon = {
                    "ERRO": "🔴", "CRIT": "🔴", "WARN": "🟡",
                    "INFO": "⚪", "DEBU": "⚫",
                }.get(level, "⚪")
                formatted.append(f"{icon} `{ts}` *{comp}*: {msg}")
            except (ValueError, KeyError):
                formatted.append(f"⚫ {line[:120]}")

        if not formatted:
            return None

        msg = "\n".join(formatted)
        if len(msg) > 4000:
            msg = msg[-4000:]
            msg = "...(truncated)\n" + msg[msg.index("\n") + 1:]
        return msg
    except Exception:
        return None
```

- [ ] **Step 2: Write guard.py — ProcessGuard main class**

Write `packages/yasar_usta/src/yasar_usta/guard.py`:

```python
"""ProcessGuard — the main orchestrator class."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from .backoff import BackoffTracker
from .commands import build_start_keyboard, build_status_inline_keyboard, format_log_entries
from .config import GuardConfig, SidecarConfig
from .lock import acquire_lock, release_lock
from .remote import find_claude_cmd, start_claude_remote
from .sidecar import SidecarManager
from .status import build_status_text
from .subprocess_mgr import SubprocessManager
from .telegram import TelegramAPI

logger = logging.getLogger("yasar_usta")


class ProcessGuard:
    """Telegram-controlled process manager.

    Args:
        config: GuardConfig with all settings.
    """

    def __init__(self, config: GuardConfig):
        self.cfg = config
        self.msgs = config.messages

        self.telegram = TelegramAPI(config.telegram_token, config.telegram_chat_id)
        self.subprocess = SubprocessManager(
            command=config.command,
            log_dir=config.log_dir,
            cwd=config.cwd,
            stop_timeout=config.stop_timeout,
            heartbeat_file=config.heartbeat_file,
            heartbeat_stale_seconds=config.heartbeat_stale_seconds,
        )
        self.backoff = BackoffTracker(
            steps=config.backoff_steps,
            reset_after=config.backoff_reset_after,
        )

        # Sidecar
        self.sidecar: SidecarManager | None = None
        if config.sidecar and config.sidecar.command:
            sc = config.sidecar
            self.sidecar = SidecarManager(
                name=sc.name,
                command=sc.command,
                pid_file=sc.pid_file,
                health_url=sc.health_url,
                health_timeout=sc.health_timeout,
                log_file=str(Path(config.log_dir) / f"{sc.name}.log"),
                cwd=config.cwd,
                detached=sc.detached,
            )

        # Claude remote
        self._claude_cmd = find_claude_cmd(config.claude_cmd) if config.claude_enabled else None
        self._claude_process: asyncio.subprocess.Process | None = None

        # State
        self._telegram_poller: asyncio.Task | None = None
        self._signal_watcher: asyncio.Task | None = None
        self._shutdown = False
        self._guard_start_time = time.time()

    # ── Telegram helpers ──────────────────────────────────────────────

    def _kb(self) -> dict:
        return build_start_keyboard(self.msgs)

    async def _send(self, text: str, reply_markup: dict | None = None) -> None:
        await self.telegram.send(text, reply_markup=reply_markup)

    async def _send_start_prompt(self, reason: str = "") -> None:
        if reason:
            msg = self.msgs.down_with_reason.format(
                reason=reason, app_name=self.cfg.app_name)
        else:
            msg = self.msgs.down_prompt.format(app_name=self.cfg.app_name)
        await self._send(msg, reply_markup=self._kb())

    async def _notify_crash(self, exit_code: int) -> None:
        stderr = "\n".join(self.subprocess.stderr_tail) or "(no output)"
        if len(stderr) > 1500:
            stderr = stderr[-1500:]
        msg = self.msgs.crash.format(
            app_name=self.cfg.app_name,
            exit_code=exit_code,
            crash_count=self.backoff.total_crashes,
            backoff=self.backoff.get_delay(),
            stderr=stderr,
        )
        await self._send(msg, reply_markup=self._kb())

    async def _notify_stopped(self) -> None:
        await self._send(
            self.msgs.stopped.format(app_name=self.cfg.app_name),
            reply_markup=self._kb(),
        )

    async def _notify_started(self) -> None:
        await self._send(self.msgs.started.format(app_name=self.cfg.app_name))

    # ── Status panel ──────────────────────────────────────────────────

    async def _send_status(self, edit_message_id: int | None = None) -> None:
        sidecar_name = self.sidecar.name if self.sidecar else None
        sidecar_alive = await self.sidecar.is_alive() if self.sidecar else False
        sidecar_pid = self.sidecar.pid_alive() if self.sidecar else None
        sidecar_http = await self.sidecar.http_alive() if self.sidecar else False

        text = build_status_text(
            name=self.cfg.name,
            app_name=self.cfg.app_name,
            guard_start_time=self._guard_start_time,
            app_running=self.subprocess.running,
            heartbeat_age=self.subprocess.heartbeat_age(),
            heartbeat_healthy_seconds=self.cfg.heartbeat_healthy_seconds,
            total_crashes=self.backoff.total_crashes,
            sidecar_name=sidecar_name,
            sidecar_alive=sidecar_alive,
            sidecar_pid=sidecar_pid,
            sidecar_health_url=self.sidecar.health_url if self.sidecar else None,
            sidecar_http_alive=sidecar_http,
            extra_processes=self.cfg.extra_processes,
        )
        inline_kb = build_status_inline_keyboard(
            self.msgs, self.cfg.name, sidecar_name)

        if edit_message_id:
            await self.telegram.edit(edit_message_id, text, reply_markup=inline_kb)
        else:
            await self._send(
                self.msgs.status_title.format(name=self.cfg.name) + "panel:",
                reply_markup=self._kb(),
            )
            await self.telegram.send(text, reply_markup=inline_kb)

    # ── Logs ──────────────────────────────────────────────────────────

    async def _send_logs(self, text: str) -> None:
        parts = text.strip().split()
        n = 20
        if len(parts) > 1:
            try:
                n = min(int(parts[1]), 50)
            except ValueError:
                pass

        log_path = self.cfg.log_file or str(Path(self.cfg.log_dir) / "orchestrator.jsonl")
        formatted = format_log_entries(log_path, n)
        if formatted is None:
            await self._send(self.msgs.no_log_file)
            return

        if self.sidecar and self.sidecar.health_url:
            pid = self.sidecar.pid_alive()
            if pid:
                formatted += f"\n\n📊 Full viewer: {self.sidecar.health_url}"
        await self._send(formatted)

    # ── Claude remote ─────────────────────────────────────────────────

    async def _handle_remote(self) -> None:
        if not self._claude_cmd:
            await self._send(self.msgs.remote_not_found)
            return
        await self._send(self.msgs.remote_starting)
        proc, url = await start_claude_remote(
            self._claude_cmd,
            name=self.cfg.claude_name or self.cfg.app_name,
            cwd=self.cfg.cwd,
        )
        self._claude_process = proc
        if proc is None:
            await self._send(self.msgs.remote_failed.format(error="process failed to start"))
        elif url:
            await self._send(self.msgs.remote_started.format(url=url, pid=proc.pid))
        else:
            await self._send(self.msgs.remote_started_no_url.format(pid=proc.pid))

    # ── Signal file watcher ───────────────────────────────────────────

    async def _start_signal_watcher(self) -> None:
        if self._signal_watcher or not self.cfg.claude_signal_file:
            return
        self._signal_watcher = asyncio.create_task(self._signal_watch_loop())

    async def _stop_signal_watcher(self) -> None:
        if self._signal_watcher:
            self._signal_watcher.cancel()
            try:
                await self._signal_watcher
            except asyncio.CancelledError:
                pass
            self._signal_watcher = None

    async def _signal_watch_loop(self) -> None:
        signal_file = Path(self.cfg.claude_signal_file)
        while True:
            try:
                await asyncio.sleep(3)
                if signal_file.exists():
                    signal_file.unlink()
                    await self._handle_remote()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Signal watcher error: %s", e)
                await asyncio.sleep(10)

    # ── Telegram poller ───────────────────────────────────────────────

    async def _start_telegram_poller(self) -> None:
        if self._telegram_poller or not self.telegram.enabled:
            return
        self._telegram_poller = asyncio.create_task(self._telegram_poll_loop())

    async def _stop_telegram_poller(self) -> None:
        if self._telegram_poller:
            self._telegram_poller.cancel()
            try:
                await self._telegram_poller
            except asyncio.CancelledError:
                pass
            self._telegram_poller = None

    async def _telegram_poll_loop(self) -> None:
        offset = 0
        last_down_reply: float = 0
        DOWN_REPLY_COOLDOWN = 30
        logger.info("Telegram poller started")

        while True:
            try:
                updates = await self.telegram.get_updates(offset=offset)
                if not updates:
                    continue

                max_uid = 0
                for update in updates:
                    uid = update["update_id"]
                    if uid > max_uid:
                        max_uid = uid

                    # Callback queries
                    cb = update.get("callback_query")
                    if cb:
                        cb_chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                        if cb_chat == str(self.cfg.telegram_chat_id):
                            await self.telegram.answer_callback(cb["id"])
                            cb_data = cb.get("data", "")
                            cb_msg_id = cb.get("message", {}).get("message_id")
                            if cb_data == "restart_guard":
                                offset = max_uid + 1
                                await self._restart_self()
                                return
                            elif cb_data == "guard_refresh":
                                await self._send_status(edit_message_id=cb_msg_id)
                            elif cb_data == "restart_sidecar" and self.sidecar:
                                await self.sidecar.stop()
                                await self.sidecar.start()
                                await self._send_status(edit_message_id=cb_msg_id)
                        continue

                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    if chat_id != str(self.cfg.telegram_chat_id):
                        continue

                    # Built-in commands
                    if text == self.msgs.btn_start or text.startswith("/start"):
                        await self._send(self.msgs.starting.format(app_name=self.cfg.app_name))
                        await self._start_app_from_poller()
                        return

                    elif text == self.msgs.btn_status or text.startswith("/status"):
                        await self._send_status()

                    elif text.startswith("/restart_guard"):
                        await self._restart_self()
                        return

                    elif text.startswith("/logs"):
                        await self._send_logs(text)

                    elif text.startswith("/remote"):
                        await self._handle_remote()

                    elif text == self.msgs.btn_system:
                        if self.subprocess.process and self.subprocess.process.returncode is None:
                            logger.info("System tap — killing hung app")
                            try:
                                self.subprocess.process.kill()
                                await self.subprocess.process.wait()
                            except Exception:
                                pass
                            self.subprocess.process = None
                            self.subprocess.running = False
                        await self._send_start_prompt(
                            f"🔴 {self.cfg.app_name} not responding.")

                    # Extra commands
                    elif text in self.cfg.extra_commands:
                        handler = self.cfg.extra_commands[text]
                        result = handler(self)
                        if asyncio.iscoroutine(result):
                            await result

                    elif text:
                        now = time.time()
                        if now - last_down_reply > DOWN_REPLY_COOLDOWN:
                            last_down_reply = now
                            await self._send_start_prompt(
                                self.msgs.down_reply.format(app_name=self.cfg.app_name))

                offset = max_uid + 1

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Telegram poll error: %s", e)
                await asyncio.sleep(5)

    async def _start_app_from_poller(self) -> None:
        """Start the managed app from the Telegram poller context."""
        self._telegram_poller = None
        await asyncio.sleep(2)
        await self.subprocess.start()

    async def _restart_self(self) -> None:
        """Restart the guard process itself."""
        logger.info("Self-restart requested")
        await self._send(self.msgs.self_restarting.format(name=self.cfg.name))

        if self.subprocess.running:
            await self.subprocess.stop()
        await self._stop_telegram_poller()
        await self.telegram.flush_updates()

        import subprocess as _sp
        python = self.subprocess.command[0] if self.subprocess.command else sys.executable
        script = str(Path(sys.argv[0]).resolve())
        argv = [a for a in sys.argv[1:] if a != script]
        logger.info("Spawning new guard: %s %s", python, script)

        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS | _sp.CREATE_NO_WINDOW
            )
        _sp.Popen(
            [sys.executable, script] + argv,
            close_fds=True,
            cwd=self.cfg.cwd or str(Path(script).parent),
            **kwargs,
        )

        release_lock()
        logger.info("Old guard exiting for restart")
        os._exit(0)

    # ── Main loop ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main guard loop."""
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        acquire_lock(self.cfg.log_dir, name="guard")
        logger.info("%s started (auto_restart=%s)", self.cfg.name, self.cfg.auto_restart)

        # Clean up stale signal files
        if self.cfg.claude_signal_file:
            sf = Path(self.cfg.claude_signal_file)
            if sf.exists():
                sf.unlink()

        # Start sidecar
        if self.sidecar and self.sidecar.command:
            await self.sidecar.start()

        # Announce
        await self._send(
            self.msgs.announce.format(name=self.cfg.name, app_name=self.cfg.app_name),
            reply_markup=self._kb(),
        )

        # Start app
        await self.subprocess.start()
        if self.subprocess.running:
            self.backoff.mark_started()
            await self._notify_started()
            await self._start_signal_watcher()
        else:
            logger.info("Initial start failed — entering Telegram poll mode")
            await self._start_telegram_poller()

        while not self._shutdown:
            try:
                exit_code = await self.subprocess.wait_for_exit()

                if self.sidecar:
                    await self.sidecar.ensure()
                await self._stop_signal_watcher()
                if self.cfg.on_exit:
                    self.cfg.on_exit(exit_code)
                self.backoff.maybe_reset()

                if exit_code == -1:
                    if (self.backoff.last_crash_time
                            and (time.time() - self.backoff.last_crash_time) < 10):
                        logger.info("No process — entering Telegram poll mode")
                        await self._start_telegram_poller()
                        while not self._shutdown and not self.subprocess.running:
                            await asyncio.sleep(1)
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()
                        continue

                    self.backoff.record_crash()
                    logger.error("App hung — restarting after 5s")
                    await self._send(self.msgs.hung.format(
                        app_name=self.cfg.app_name, delay=5))
                    await self._start_telegram_poller()
                    for _ in range(5):
                        if self._shutdown or self.subprocess.running:
                            break
                        await asyncio.sleep(1)
                    if not self.subprocess.running and not self._shutdown:
                        await self.subprocess.start()
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()
                    continue

                logger.info("App exited with code %d", exit_code)

                if exit_code == self.cfg.restart_exit_code:
                    await self._send(self.msgs.restarting.format(app_name=self.cfg.app_name))
                    await asyncio.sleep(1)
                    await self.subprocess.start()
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                elif exit_code == 0:
                    logger.info("App stopped cleanly")
                    await self._notify_stopped()
                    await self._start_telegram_poller()
                    while not self._shutdown and not self.subprocess.running:
                        await asyncio.sleep(1)
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                else:
                    self.backoff.record_crash()
                    backoff_delay = self.backoff.get_delay()
                    logger.error(
                        "App crashed (exit %d), crash #%d, backoff %ds",
                        exit_code, self.backoff.total_crashes, backoff_delay,
                    )
                    await self._notify_crash(exit_code)

                    if not self.cfg.auto_restart:
                        await self._start_telegram_poller()
                        while not self._shutdown and not self.subprocess.running:
                            await asyncio.sleep(1)
                        continue

                    await self._start_telegram_poller()
                    for _ in range(backoff_delay):
                        if self._shutdown or self.subprocess.running:
                            break
                        await asyncio.sleep(1)

                    if not self.subprocess.running and not self._shutdown:
                        await self.subprocess.start()
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as exc:
                logger.critical("UNHANDLED ERROR: %s", exc, exc_info=True)
                try:
                    await self._send(self.msgs.wrapper_error.format(error=repr(exc)))
                except Exception:
                    pass
                await self._start_telegram_poller()
                while not self._shutdown and not self.subprocess.running:
                    await asyncio.sleep(5)

        # Shutdown
        if self.subprocess.running:
            await self.subprocess.stop()
        await self._stop_signal_watcher()
        await self._stop_telegram_poller()
        logger.info("Guard exiting.")

    def request_shutdown(self) -> None:
        """Request the guard to shut down."""
        self._shutdown = True
```

- [ ] **Step 3: Update __init__.py**

Replace `packages/yasar_usta/src/yasar_usta/__init__.py` with:

```python
"""Yaşar Usta — Telegram-controlled process manager.

Manages any subprocess with:
- Heartbeat-based hung detection
- Escalating backoff with stability reset
- Telegram as control plane when process is down
- Non-destructive offset polling (no message loss)
- Claude Code remote trigger
- Sidecar process management (log viewer etc.)
- Configurable i18n for all user-facing strings
"""

from .config import GuardConfig, Messages, SidecarConfig
from .guard import ProcessGuard

__all__ = [
    "ProcessGuard",
    "GuardConfig",
    "Messages",
    "SidecarConfig",
]
```

- [ ] **Step 4: Write test_guard.py**

Write `packages/yasar_usta/tests/test_guard.py`:

```python
"""Tests for yasar_usta.guard — ProcessGuard integration."""

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from yasar_usta import ProcessGuard, GuardConfig, Messages


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestProcessGuardConstruction:
    def test_creates_with_minimal_config(self):
        cfg = GuardConfig(
            command=[sys.executable, "-c", "pass"],
            log_dir=tempfile.mkdtemp(),
        )
        guard = ProcessGuard(cfg)
        assert guard.cfg.app_name == "App"
        assert guard.telegram.enabled is False
        assert guard.sidecar is None

    def test_creates_with_full_config(self):
        cfg = GuardConfig(
            name="Yaşar Usta",
            app_name="Kutay",
            command=["python", "run.py"],
            telegram_token="tok",
            telegram_chat_id="123",
            sidecar=__import__("yasar_usta.config", fromlist=["SidecarConfig"]).SidecarConfig(
                name="yazbunu",
                command=["python", "-m", "yazbunu.server"],
                health_url="http://127.0.0.1:9880/",
                pid_file="/tmp/yazbunu.pid",
            ),
        )
        guard = ProcessGuard(cfg)
        assert guard.telegram.enabled is True
        assert guard.sidecar is not None
        assert guard.sidecar.name == "yazbunu"

    def test_custom_messages(self):
        cfg = GuardConfig(
            command=["echo"],
            log_dir=tempfile.mkdtemp(),
            messages=Messages(
                started="✅ *{app_name} Başladı*",
                btn_start="▶️ Başlat",
            ),
        )
        guard = ProcessGuard(cfg)
        assert "Başladı" in guard.msgs.started
        assert guard.msgs.btn_start == "▶️ Başlat"


class TestProcessGuardNotifications:
    def test_notify_started(self):
        cfg = GuardConfig(
            command=["echo"],
            log_dir=tempfile.mkdtemp(),
            app_name="TestApp",
        )
        guard = ProcessGuard(cfg)
        guard.telegram.send = AsyncMock()
        run_async(guard._notify_started())
        guard.telegram.send.assert_called_once()
        call_text = guard.telegram.send.call_args[0][0]
        assert "TestApp" in call_text
        assert "Started" in call_text

    def test_notify_crash(self):
        cfg = GuardConfig(
            command=["echo"],
            log_dir=tempfile.mkdtemp(),
            app_name="TestApp",
        )
        guard = ProcessGuard(cfg)
        guard.telegram.send = AsyncMock()
        guard.subprocess.stderr_tail.append("some error")
        guard.backoff.record_crash()
        run_async(guard._notify_crash(1))
        call_text = guard.telegram.send.call_args[0][0]
        assert "Crashed" in call_text
        assert "some error" in call_text
```

- [ ] **Step 5: Run full test suite**

```bash
pip install -e ./packages/yasar_usta
pytest packages/yasar_usta/tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/
git commit -m "feat(yasar-usta): add ProcessGuard main class with command handling"
```

---

### Task 5: Wire into KutAI — replace kutai_wrapper.py

**Files:**
- Modify: `kutai_wrapper.py` (rewrite as thin consumer)
- Modify: `requirements.txt` (add editable install)

- [ ] **Step 1: Add editable install to requirements.txt**

Add after the vecihi entry:

```
-e ./packages/yasar_usta
```

- [ ] **Step 2: Install the package**

```bash
pip install -e ./packages/yasar_usta
```

- [ ] **Step 3: Rewrite kutai_wrapper.py as a thin consumer**

Replace the entire contents of `kutai_wrapper.py` with:

```python
#!/usr/bin/env python3
"""
Yaşar Usta — KutAI'nin süreç yöneticisi (process manager).

Thin consumer of the yasar-usta package, configured for KutAI.
"""
import asyncio
import os
import signal
import subprocess as _sp
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Venv guard ──
_EXPECTED_VENV = Path(__file__).parent / ".venv"
_in_venv = hasattr(sys, "real_prefix") or (
    hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
)
if not _in_venv and _EXPECTED_VENV.exists():
    print(f"ERROR: Running with system Python ({sys.executable})")
    print(f"Use: .venv\\Scripts\\python.exe kutai_wrapper.py")
    sys.exit(1)

from yasar_usta import ProcessGuard, GuardConfig, Messages, SidecarConfig

PROJECT_ROOT = Path(__file__).resolve().parent


def _find_python() -> str:
    venv = PROJECT_ROOT / ".venv"
    if sys.platform == "win32":
        p = venv / "Scripts" / "python.exe"
    else:
        p = venv / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def _kill_orphan_processes(exit_code: int) -> None:
    """Kill orphaned llama-server after orchestrator exits (KutAI-specific)."""
    if exit_code == 42:
        return  # Clean restart — don't kill llama-server

    targets = [
        ("llama-server.exe", "llama-server"),
        ("ollama.exe", "Ollama"),
        ("ollama_llama_server.exe", "Ollama runner"),
    ]
    for exe, label in targets:
        try:
            check = _sp.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            if exe.lower() not in check.stdout.lower():
                continue
            result = _sp.run(
                ["taskkill", "/F", "/IM", exe],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"[Yasar Usta] Killed orphaned {label}: {result.stdout.strip()}")
        except Exception as e:
            print(f"[Yasar Usta] {label} cleanup error: {e}")


venv_python = _find_python()

config = GuardConfig(
    name="Yaşar Usta",
    app_name="Kutay",
    command=[venv_python, str(PROJECT_ROOT / "src" / "app" / "run.py")],
    cwd=str(PROJECT_ROOT),

    telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    telegram_chat_id=os.getenv("TELEGRAM_ADMIN_CHAT_ID", ""),

    backoff_steps=[5, 15, 60, 300],
    backoff_reset_after=600,

    heartbeat_file=str(PROJECT_ROOT / "logs" / "orchestrator.heartbeat"),
    heartbeat_stale_seconds=120,
    heartbeat_healthy_seconds=90,

    restart_exit_code=42,
    log_dir=str(PROJECT_ROOT / "logs"),
    log_file=str(PROJECT_ROOT / "logs" / "orchestrator.jsonl"),
    stop_timeout=30,
    auto_restart="--no-auto-restart" not in sys.argv,

    claude_enabled=True,
    claude_name="Kutay",
    claude_signal_file=str(PROJECT_ROOT / "logs" / "claude_remote.signal"),

    sidecar=SidecarConfig(
        name="yazbunu",
        command=[venv_python, "-m", "yazbunu.server",
                 "--log-dir", "./logs", "--port", "9880", "--host", "0.0.0.0"],
        health_url="http://127.0.0.1:9880/",
        pid_file=str(PROJECT_ROOT / "logs" / "yazbunu.pid"),
        detached=True,
        auto_start=True,
    ),

    on_exit=_kill_orphan_processes,

    extra_processes=[
        {"exe": "llama-server.exe", "label": "llama-server"},
    ],

    messages=Messages(
        announce="🔧 *Bennn... Yaşar Usta!*\n\nKutay'ı başlatıyorum...",
        started="✅ *Kutay Started*",
        stopped="⏹ *Kutay Stopped*\nSend /kutai\\_start to restart.",
        crash=(
            "🔴 *Kutay Crashed*\n"
            "Exit code: `{exit_code}`\n"
            "Crash #{crash_count}\n"
            "Restarting in {backoff}s\n\n"
            "```\n{stderr}\n```"
        ),
        hung="🔴 Kutay dondu — Yaşar Usta {delay}sn içinde yeniden başlatıyor",
        restarting="♻️ *Kutay yeniden başlatılıyor...*",
        self_restarting="🔄 *Yaşar Usta yeniden başlatılıyor...*",
        down_prompt="⚠️ Kutay durdu. Başlatmak için butona bas.",
        down_reply="⏸ Kutay şu an kapalı.",
        starting="🚀 Kutay başlatılıyor...",
        btn_start="▶️ Başlat",
        btn_status="🔧 Yaşar Usta",
        btn_system="⚙️ Sistem",
        btn_refresh="🔄 Yenile",
        btn_restart_guard="♻️ Usta'yı Yeniden Başlat",
        btn_restart_sidecar="📊 Yazbunu Yeniden Başlat",
        remote_starting="🖥️ Claude Code oturumu başlatılıyor...",
        remote_not_found="❌ `claude` command not found. Claude Code kurulu mu?",
    ),
)


async def main():
    guard = ProcessGuard(config)

    def _sig(sig, frame):
        print(f"\n[Yasar Usta] Signal {sig} received, shutting down...")
        guard.request_shutdown()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)

    if sys.platform == "win32":
        try:
            import ctypes

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)
            def _console_handler(event):
                if event in (0, 2):
                    guard.request_shutdown()
                    return True
                return False

            ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)
            guard._console_handler = _console_handler
        except Exception:
            pass

    await guard.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Yasar Usta] KeyboardInterrupt — exiting")
    except Exception as exc:
        print(f"[Yasar Usta] FATAL: {exc!r}")
        raise
```

- [ ] **Step 4: Verify it imports correctly**

```bash
python -c "from yasar_usta import ProcessGuard, GuardConfig, Messages, SidecarConfig; print('yasar_usta OK')"
```

Expected: `yasar_usta OK`

- [ ] **Step 5: Verify kutai_wrapper.py loads without errors**

```bash
python -c "import kutai_wrapper; print('wrapper OK')" 2>&1 || echo "Expected — lock prevents double start"
```

Note: This will likely fail because the lock system runs at import time in the original. After the rewrite, the lock is acquired inside `guard.run()`, so this should print `wrapper OK`.

- [ ] **Step 6: Run yasar_usta tests**

```bash
pytest packages/yasar_usta/tests/ -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add kutai_wrapper.py requirements.txt
git commit -m "refactor: rewrite kutai_wrapper.py as thin consumer of yasar-usta package"
```

---

### Task 6: Verify end-to-end

**Files:** None — verification only.

- [ ] **Step 1: Verify yasar_usta works independently**

```bash
python -c "
import sys
sys.path = [p for p in sys.path if 'kutay' not in p.lower() or 'packages' in p.lower()]
from yasar_usta import ProcessGuard, GuardConfig, Messages, SidecarConfig
print('yasar_usta standalone OK')
"
```

Expected: `yasar_usta standalone OK`

- [ ] **Step 2: Run all package tests**

```bash
pytest packages/yasar_usta/tests/ -v --tb=short
```

Expected: All tests pass.

- [ ] **Step 3: Verify kutai_wrapper.py syntax and imports**

```bash
python -c "
import ast
with open('kutai_wrapper.py') as f:
    ast.parse(f.read())
print('syntax OK')
"
python -c "from yasar_usta import ProcessGuard, GuardConfig; print('import OK')"
```

Expected: Both print OK.

- [ ] **Step 4: Verify KutAI can still start** (manual — don't actually start)

```bash
python -c "
from kutai_wrapper import config
print(f'name={config.name}')
print(f'app={config.app_name}')
print(f'command={config.command}')
print(f'heartbeat={config.heartbeat_file}')
print(f'sidecar={config.sidecar.name if config.sidecar else None}')
print('config OK')
"
```

Expected: Prints KutAI-specific config values.
