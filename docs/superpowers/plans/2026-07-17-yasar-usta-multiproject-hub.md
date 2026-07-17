# Yaşar Usta Multi-Project Hub (Sub-Project 1: Hub Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve Yaşar Usta from a single-process manager into a hub that supervises N local process targets across multiple projects from one Telegram surface, with KutAI migrated in as the first registry entry — behavior-preserving.

**Architecture:** Split today's monolithic `ProcessGuard` into `Hub` (owns the single Telegram poller, the single `hub` lock, N supervisors, dashboard rendering, coordinated shutdown, hub self-restart, process-global signal handlers) and `TargetSupervisor` (supervises exactly one target: `run()` state machine, backoff, sidecars, signal-watcher, heartbeat, intent flags; sends via an injected one-way `notify`). A declarative `registry.yaml` + optional per-project Python hook module drives everything. `kutai_wrapper.py` is rewritten to load the registry and run the `Hub`.

**Tech Stack:** Python 3.10, asyncio, aiohttp, pytest (`pytest-asyncio`), PyYAML, dataclasses. Package: `packages/yasar_usta/src/yasar_usta/` (src-layout, editable-installed).

**Spec:** `docs/superpowers/specs/2026-07-17-yasar-usta-multiproject-hub-design.md`

**Conventions:**
- Run tests with a timeout: `python -m pytest packages/yasar_usta/tests/... -v` (project rule: never run pytest without a timeout; targeted runs only). On Windows use the venv python.
- Commit at each task (project rule: commit at each milestone). Conventional commits + `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Do NOT push — this work is restart-gated; push happens only after live-verify at the end.
- Naming already in codebase: `TelegramAPI.send/edit/delete/answer_callback/get_updates/flush_updates`, `SubprocessManager`, `SidecarManager`, `BackoffTracker`, `build_status_text`, `build_start_keyboard`, `build_status_inline_keyboard`, `format_log_entries`, `acquire_lock(lock_dir, name=)`, `release_lock()`.

---

## Task 0: Add PyYAML dependency + verify baseline

**Files:**
- Modify: `packages/yasar_usta/pyproject.toml` (add `pyyaml` to dependencies)

- [ ] **Step 1: Confirm the current suite is green (baseline)**

Run: `python -m pytest packages/yasar_usta/tests/ -q`
Expected: all pass (10 test modules). Record the count — every later task must keep these green (or update them in lockstep when a responsibility moves, noted per-task).

- [ ] **Step 2: Add PyYAML to the package deps**

In `packages/yasar_usta/pyproject.toml`, add `"pyyaml>=6.0"` to the `[project].dependencies` list (create the key if the package currently has none). Then confirm import works:

Run: `python -c "import yaml; print(yaml.__version__)"`
Expected: prints a version (already present transitively in this repo; the explicit dep documents it).

- [ ] **Step 3: Commit**

```bash
git add packages/yasar_usta/pyproject.toml
git commit -m "chore(yasar_usta): declare pyyaml dep for registry loader"
```

---

## Task 1: Characterization tests for `ProcessGuard.run()` state machine

**Why (spec finding #9):** `run()` (`guard.py:579-791`) has zero tests. "Behavior-preserving" is unverifiable until we lock current behavior. These tests target today's `ProcessGuard` and are the regression net for the split.

**Files:**
- Create: `packages/yasar_usta/tests/test_guard_run_characterization.py`

- [ ] **Step 1: Write failing characterization tests**

```python
"""Characterization tests: lock ProcessGuard.run() branch behavior BEFORE the
Hub/TargetSupervisor split. These assert observable effects (notifications sent,
app (re)started, intent flags consumed) for each exit-code branch."""
import asyncio
import types
import pytest

from yasar_usta.config import GuardConfig
from yasar_usta.guard import ProcessGuard


def _guard(tmp_path, **over):
    cfg = GuardConfig(
        name="T", app_name="App",
        command=["python", "-c", "pass"],
        log_dir=str(tmp_path / "logs"),
        telegram_token="", telegram_chat_id="",  # telegram disabled → no network
        backoff_steps=[1], auto_restart=True,
        **over,
    )
    return ProcessGuard(cfg)


class _FakeSub:
    """Scriptable stand-in for SubprocessManager."""
    def __init__(self, exit_codes):
        self._codes = list(exit_codes)
        self.running = False
        self.started = 0
        self.stopped = 0
        self.stderr_tail = []
        self.command = ["python"]
        self.process = None
    async def start(self):
        self.started += 1
        self.running = bool(self._codes)
    async def stop(self, timeout=None):
        self.stopped += 1
        self.running = False
    async def wait_for_exit(self):
        if not self._codes:
            self.running = False
            return 0
        code = self._codes.pop(0)
        self.running = False
        return code
    def heartbeat_age(self):
        return 0.0


@pytest.mark.asyncio
async def test_run_restart_exit_code_restarts_app(tmp_path):
    g = _guard(tmp_path)
    # exit 42 (restart), then a clean 0 that we use to end the loop
    g.subprocess = _FakeSub([42, 0])
    sent = []
    g._send = lambda text, reply_markup=None: sent.append(text) or asyncio.sleep(0)
    g._notify_started = lambda: asyncio.sleep(0)
    g._start_signal_watcher = lambda: asyncio.sleep(0)
    # end the loop after two exits
    orig_wait = g.subprocess.wait_for_exit
    async def wait():
        code = await orig_wait()
        if not g.subprocess._codes:
            g._shutdown = True
        return code
    g.subprocess.wait_for_exit = wait
    await g.run()
    assert g.subprocess.started >= 2  # initial + restart


@pytest.mark.asyncio
async def test_run_restart_flag_consumed(tmp_path):
    g = _guard(tmp_path)
    g.subprocess = _FakeSub([0])
    g._restart_requested = True
    g._notify_started = lambda: asyncio.sleep(0)
    g._start_signal_watcher = lambda: asyncio.sleep(0)
    g._send = lambda *a, **k: asyncio.sleep(0)
    async def wait():
        g._shutdown = True
        g.subprocess.running = False
        return 0
    g.subprocess.wait_for_exit = wait
    await g.run()
    assert g._restart_requested is False  # flag consumed by the loop
```

- [ ] **Step 2: Run to verify they pass against CURRENT code**

Run: `python -m pytest packages/yasar_usta/tests/test_guard_run_characterization.py -v`
Expected: PASS. (These characterize existing behavior — they should pass now. If a test does not pass, the test is wrong about current behavior; fix the test, not the code.)

- [ ] **Step 3: Commit**

```bash
git add packages/yasar_usta/tests/test_guard_run_characterization.py
git commit -m "test(yasar_usta): characterize ProcessGuard.run() branches before split"
```

---

## Task 2: Characterization tests for the poller callback dispatch

**Files:**
- Create: `packages/yasar_usta/tests/test_guard_poll_characterization.py`

- [ ] **Step 1: Write tests that assert each callback drives the expected action**

```python
"""Characterize the callback dispatch of _telegram_poll_loop BEFORE the split.
We drive one batch of updates through the loop, then cancel it, asserting the
side effects (restart flag set, shutdown signal written, subprocess.stop called)."""
import asyncio
import pytest

from yasar_usta.config import GuardConfig
from yasar_usta.guard import ProcessGuard


def _guard(tmp_path):
    cfg = GuardConfig(
        name="T", app_name="App", command=["python"],
        log_dir=str(tmp_path / "logs"),
        telegram_token="x", telegram_chat_id="42",
    )
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return ProcessGuard(cfg)


class _Sub:
    def __init__(self):
        self.running = True
        self.stopped = 0
        self.process = None
    async def stop(self, timeout=None):
        self.stopped += 1
        self.running = False


async def _run_one_batch(g, updates):
    """Feed exactly one getUpdates batch, then make the loop exit."""
    calls = {"n": 0}
    async def get_updates(offset=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return updates
        raise asyncio.CancelledError()
    g.telegram.get_updates = get_updates
    g.telegram.answer_callback = lambda *a, **k: asyncio.sleep(0)
    g.telegram.delete = lambda *a, **k: asyncio.sleep(0)
    g.telegram.send = lambda *a, **k: asyncio.sleep(0)
    g._send = lambda *a, **k: asyncio.sleep(0)
    try:
        await g._telegram_poll_loop(0)
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_confirm_restart_sets_flag_and_stops(tmp_path):
    g = _guard(tmp_path)
    g.subprocess = _Sub()
    upd = [{"update_id": 1, "callback_query": {
        "id": "c", "data": "confirm_restart",
        "message": {"message_id": 9, "chat": {"id": 42}}}}]
    await _run_one_batch(g, upd)
    assert g._restart_requested is True
    assert g.subprocess.stopped == 1
    assert (tmp_path / "logs" / "shutdown.signal").read_text() == "restart"


@pytest.mark.asyncio
async def test_confirm_stop_sets_flag_and_stops(tmp_path):
    g = _guard(tmp_path)
    g.subprocess = _Sub()
    upd = [{"update_id": 1, "callback_query": {
        "id": "c", "data": "confirm_stop",
        "message": {"message_id": 9, "chat": {"id": 42}}}}]
    await _run_one_batch(g, upd)
    assert g._stop_requested is True
    assert g.subprocess.stopped == 1
    assert (tmp_path / "logs" / "shutdown.signal").read_text() == "stop"
```

- [ ] **Step 2: Run to verify PASS against current code**

Run: `python -m pytest packages/yasar_usta/tests/test_guard_poll_characterization.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add packages/yasar_usta/tests/test_guard_poll_characterization.py
git commit -m "test(yasar_usta): characterize poller callback dispatch before split"
```

---

## Task 3: Config layer — `HubConfig`, `ProjectConfig`, `load_registry`

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/config.py` (add `env` to `GuardConfig`; add `HubConfig`, `ProjectConfig`)
- Create: `packages/yasar_usta/src/yasar_usta/registry.py` (`load_registry`, `${}` resolution)
- Create: `packages/yasar_usta/tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from yasar_usta.registry import load_registry


def _write(tmp_path, body):
    p = tmp_path / "registry.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_registry_parses_hub_and_projects(tmp_path):
    reg = _write(tmp_path, """
hub:
  telegram_token_env: MY_TOKEN
  telegram_chat_id_env: MY_CHAT
  log_dir: "${project_root}/logs"
projects:
  demo:
    name: Demo
    targets:
      - id: web
        command: ["python", "-m", "http.server"]
        cwd: "${project_root}"
        log_dir: "${project_root}/logs"
        heartbeat_file: "${project_root}/logs/web.heartbeat"
""")
    import os
    os.environ["MY_TOKEN"] = "tok"
    os.environ["MY_CHAT"] = "99"
    hub, projects = load_registry(reg, project_root=str(tmp_path))
    assert hub.telegram_token == "tok"
    assert hub.telegram_chat_id == "99"
    assert len(projects) == 1
    p = projects[0]
    assert p.id == "demo"
    assert p.name == "Demo"
    assert len(p.targets) == 1
    t = p.targets[0]
    assert t.command == ["python", "-m", "http.server"]
    assert t.cwd == str(tmp_path)
    assert t.log_dir == str(tmp_path / "logs")


def test_load_registry_fails_fast_on_missing_targets(tmp_path):
    reg = _write(tmp_path, """
hub: {telegram_token_env: T, telegram_chat_id_env: C, log_dir: "l"}
projects:
  bad: {name: Bad}
""")
    with pytest.raises(ValueError, match="targets"):
        load_registry(reg, project_root=str(tmp_path))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: yasar_usta.registry`.

- [ ] **Step 3: Add config dataclasses**

In `packages/yasar_usta/src/yasar_usta/config.py`, add `env` to `GuardConfig` (after the `cwd` field, around line 96):

```python
    env: dict = field(default_factory=dict)  # merged onto os.environ at spawn
```

Then append two dataclasses at the end of the file:

```python
@dataclass
class HubConfig:
    """Hub-level config: the single shared Telegram bot + hub lock/log dir."""

    name: str = "Yaşar Usta"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    log_dir: str = "logs"  # where the hub lock + hub meta log live
    messages: Messages = field(default_factory=Messages)


@dataclass
class ProjectConfig:
    """One project = a display name + N targets + an optional hook module."""

    id: str
    name: str
    targets: list[GuardConfig] = field(default_factory=list)
    hook_module: str | None = None
```

- [ ] **Step 4: Write the registry loader**

Create `packages/yasar_usta/src/yasar_usta/registry.py`:

```python
"""Declarative registry loader: registry.yaml -> (HubConfig, [ProjectConfig])."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .config import GuardConfig, HubConfig, ProjectConfig, SidecarConfig


def _resolve(value, tokens: dict):
    """Substitute ${token} placeholders in strings (recursively in lists/dicts)."""
    if isinstance(value, str):
        for k, v in tokens.items():
            value = value.replace("${" + k + "}", v)
        return value
    if isinstance(value, list):
        return [_resolve(x, tokens) for x in value]
    if isinstance(value, dict):
        return {k: _resolve(v, tokens) for k, v in value.items()}
    return value


def _build_target(raw: dict, tokens: dict) -> GuardConfig:
    raw = _resolve(raw, tokens)
    if "id" not in raw or "command" not in raw:
        raise ValueError(f"target missing id/command: {raw!r}")
    sidecars = []
    for sc in raw.get("sidecars", []):
        sidecars.append(SidecarConfig(**sc))
    return GuardConfig(
        name=raw["id"],
        app_name=raw.get("app_name", raw["id"]),
        command=raw["command"],
        cwd=raw.get("cwd"),
        env=raw.get("env", {}),
        heartbeat_file=raw.get("heartbeat_file"),
        heartbeat_stale_seconds=raw.get("heartbeat_stale_seconds", 120),
        heartbeat_healthy_seconds=raw.get("heartbeat_healthy_seconds", 90),
        restart_exit_code=raw.get("restart_exit_code", 42),
        log_dir=raw.get("log_dir", "logs"),
        log_file=raw.get("log_file"),
        stop_timeout=raw.get("stop_timeout", 30),
        auto_restart=raw.get("auto_restart", True),
        backoff_steps=raw.get("backoff_steps", [5, 15, 60, 300]),
        claude_enabled=raw.get("claude_enabled", True),
        claude_cmd=raw.get("claude_cmd"),
        claude_name=raw.get("claude_name"),
        claude_signal_file=raw.get("claude_signal_file"),
        sidecars=sidecars,
        extra_processes=raw.get("extra_processes", []),
    )


def load_registry(path, project_root: str) -> tuple[HubConfig, list[ProjectConfig]]:
    """Parse registry.yaml. Fail fast on structural error (no partial start)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "projects" not in data:
        raise ValueError("registry.yaml must have a top-level 'projects' mapping")

    tokens = {"project_root": project_root}

    raw_hub = data.get("hub", {})
    hub = HubConfig(
        name=raw_hub.get("name", "Yaşar Usta"),
        telegram_token=os.getenv(raw_hub.get("telegram_token_env", ""), ""),
        telegram_chat_id=os.getenv(raw_hub.get("telegram_chat_id_env", ""), ""),
        log_dir=_resolve(raw_hub.get("log_dir", "logs"), tokens),
    )

    projects: list[ProjectConfig] = []
    for pid, raw_proj in data["projects"].items():
        if "targets" not in raw_proj or not raw_proj["targets"]:
            raise ValueError(f"project {pid!r} has no targets")
        targets = [_build_target(t, tokens) for t in raw_proj["targets"]]
        projects.append(ProjectConfig(
            id=pid,
            name=raw_proj.get("name", pid),
            targets=targets,
            hook_module=raw_proj.get("hook_module"),
        ))
    return hub, projects
```

- [ ] **Step 5: Run tests to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_registry.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/config.py packages/yasar_usta/src/yasar_usta/registry.py packages/yasar_usta/tests/test_registry.py
git commit -m "feat(yasar_usta): registry loader + HubConfig/ProjectConfig + per-target env"
```

---

## Task 4: Thread per-target `env` through `SubprocessManager`

**Why (spec finding R2):** `SubprocessManager.start()` never passes `env=`. Add it with an explicit `{**os.environ, **env}` merge (bare `env` would drop PATH/venv).

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/subprocess_mgr.py:51-59` (constructor) and `:122-129` (spawn)
- Modify: `packages/yasar_usta/tests/test_subprocess_mgr.py` (add env test)

- [ ] **Step 1: Write the failing test**

Append to `packages/yasar_usta/tests/test_subprocess_mgr.py`:

```python
import os
import sys
import pytest
from yasar_usta.subprocess_mgr import SubprocessManager


@pytest.mark.asyncio
async def test_env_merged_onto_os_environ(tmp_path):
    marker = tmp_path / "out.txt"
    code = "import os,pathlib; pathlib.Path(os.environ['OUT']).write_text(os.environ['MYVAR'])"
    mgr = SubprocessManager(
        command=[sys.executable, "-c", code],
        log_dir=str(tmp_path / "logs"),
        env={"MYVAR": "hello", "OUT": str(marker)},
    )
    await mgr.start()
    await mgr.wait_for_exit()
    assert marker.read_text() == "hello"
    # os.environ was NOT mutated
    assert "MYVAR" not in os.environ
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_subprocess_mgr.py -k env_merged -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'env'`.

- [ ] **Step 3: Add `env` param + merge at spawn**

In `subprocess_mgr.py`, add `env: dict | None = None,` to `__init__` params (after `heartbeat_stale_seconds`) and store `self.env = env or {}`.

Then in `start()`, replace the spawn block (`:120-129`) to pass a merged env:

```python
        logger.info("Starting subprocess: %s", " ".join(self.command))
        _child_env = None
        if self.env:
            import os as _os
            _child_env = {**_os.environ, **self.env}
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
                cwd=self.cwd,
                env=_child_env,
                **kwargs,
            )
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_subprocess_mgr.py -v`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/subprocess_mgr.py packages/yasar_usta/tests/test_subprocess_mgr.py
git commit -m "feat(yasar_usta): per-target env merged onto os.environ at spawn"
```

---

## Task 5: Thread per-target `env` through `SidecarManager`

**Why (spec finding R2b):** the nerd_herd sidecar reads `NERD_HERD_PROJECT_ROOT`/`LLAMA_SERVER_PORT` from `os.environ`; once project-root leaves the process env, sidecars need per-project env too.

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/sidecar.py:31-49` (ctor) and `:99-118` (Popen)
- Modify: `packages/yasar_usta/src/yasar_usta/config.py` (`SidecarConfig` gains `env`)
- Modify: `packages/yasar_usta/tests/test_sidecar.py`

- [ ] **Step 1: Write the failing test**

Append to `packages/yasar_usta/tests/test_sidecar.py`:

```python
import os, sys
import pytest
from yasar_usta.sidecar import SidecarManager


@pytest.mark.asyncio
async def test_sidecar_env_merged(tmp_path):
    marker = tmp_path / "sc.txt"
    code = "import os,pathlib; pathlib.Path(os.environ['OUT']).write_text(os.environ['SCVAR'])"
    sc = SidecarManager(
        name="t", command=[sys.executable, "-c", code],
        pid_file=str(tmp_path / "t.pid"), detached=False,
        env={"SCVAR": "sc", "OUT": str(marker)},
    )
    await sc.start()
    import asyncio; await asyncio.sleep(1.0)
    assert marker.read_text() == "sc"
    assert "SCVAR" not in os.environ
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_sidecar.py -k sidecar_env -v`
Expected: FAIL — unexpected keyword `env`.

- [ ] **Step 3: Add `env` to `SidecarConfig`**

In `config.py`, add to `SidecarConfig` (after `auto_restart`):

```python
    env: dict = field(default_factory=dict)
```

- [ ] **Step 4: Add `env` to `SidecarManager`**

In `sidecar.py`, add `env: dict | None = None,` to `__init__` and store `self.env = env or {}`. In `start()`, before `proc = _sp.Popen(...)`, build the merged env and pass it:

```python
            if self.env:
                kwargs["env"] = {**os.environ, **self.env}

            proc = _sp.Popen(
                self.command,
                cwd=self.cwd,
                **kwargs,
            )
```

(`os` is already imported at the top of `sidecar.py`.)

- [ ] **Step 5: Run to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_sidecar.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/sidecar.py packages/yasar_usta/src/yasar_usta/config.py packages/yasar_usta/tests/test_sidecar.py
git commit -m "feat(yasar_usta): per-project env for sidecars"
```

---

## Task 6: Introduce `TargetSupervisor` — intent API + injected notify + status()

**Approach:** Create `TargetSupervisor` as a refactor of `ProcessGuard` that supervises ONE target and owns NO poller/lock/signal-handlers. It is constructed with a `GuardConfig` and an injected `notify` coroutine. It exposes `request_start/restart/stop`, `status()`, `run()`, `stop_all()`.

**Files:**
- Create: `packages/yasar_usta/src/yasar_usta/supervisor.py`
- Create: `packages/yasar_usta/tests/test_supervisor.py`

- [ ] **Step 1: Write the failing test**

```python
import asyncio
import pytest
from yasar_usta.config import GuardConfig
from yasar_usta.supervisor import TargetSupervisor


def _sup(tmp_path):
    cfg = GuardConfig(name="web", app_name="web", command=["python"],
                      log_dir=str(tmp_path / "logs"), backoff_steps=[1])
    sent = []
    async def notify(text, reply_markup=None):
        sent.append(text)
    sup = TargetSupervisor("demo", cfg, notify=notify)
    return sup, sent


def test_request_restart_sets_flag_and_signal(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.request_restart()
    assert sup._restart_requested is True
    assert (tmp_path / "logs" / "shutdown.signal").read_text() == "restart"


def test_request_stop_sets_flag_and_signal(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.request_stop()
    assert sup._stop_requested is True
    assert (tmp_path / "logs" / "shutdown.signal").read_text() == "stop"


def test_status_snapshot_shape(tmp_path):
    sup, _ = _sup(tmp_path)
    s = sup.status()
    assert s["project_id"] == "demo"
    assert s["running"] is False
    assert "heartbeat_age" in s and "total_crashes" in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_supervisor.py -v`
Expected: FAIL — `ModuleNotFoundError: yasar_usta.supervisor`.

- [ ] **Step 3: Write `TargetSupervisor`**

Create `packages/yasar_usta/src/yasar_usta/supervisor.py`. This is `ProcessGuard` reduced to one target. Copy the following methods **verbatim from `guard.py`**, changing only `self._send(...)` → `await self.notify(...)` and dropping telegram/lock/poller ownership:

- from `guard.py`: `_extract_traceback` (module fn), `_write_shutdown_signal`, `_notify_crash`, `_notify_stopped`, `_notify_started`, `_send_start_prompt`, `_handle_remote`, `_start_signal_watcher`, `_stop_signal_watcher`, `_signal_watch_loop`, `_start_app`, and the entire `run()` **loop body** (everything from `while not self._shutdown:` at `:654` through the shutdown block at `:787`) — but NOT the lock/announce/poller/flush setup at `:581-653` (that moves to Hub).

Write the class as:

```python
"""TargetSupervisor — supervises exactly one process target. No telegram
poller, no lock, no signal handlers (all owned by Hub)."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

from .backoff import BackoffTracker
from .config import GuardConfig
from .remote import find_claude_cmd, list_sessions, start_claude_remote
from .sidecar import SidecarManager
from .subprocess_mgr import SubprocessManager

logger = logging.getLogger("yasar_usta.supervisor")

# NOTE: copy _extract_traceback verbatim from guard.py here (module-level fn).


class TargetSupervisor:
    def __init__(self, project_id: str, config: GuardConfig,
                 notify: Callable[..., Awaitable[None]]):
        self.project_id = project_id
        self.cfg = config
        self.msgs = config.messages
        self.notify = notify  # injected one-way sender (Hub owns TelegramAPI)

        self.subprocess = SubprocessManager(
            command=config.command, log_dir=config.log_dir, cwd=config.cwd,
            stop_timeout=config.stop_timeout, heartbeat_file=config.heartbeat_file,
            heartbeat_stale_seconds=config.heartbeat_stale_seconds,
            env=config.env,
        )
        self.backoff = BackoffTracker(steps=config.backoff_steps,
                                      reset_after=config.backoff_reset_after)
        self.sidecars: dict[str, SidecarManager] = {}
        for sc in config.sidecars:
            if sc.command:
                self.sidecars[sc.name] = SidecarManager(
                    name=sc.name, command=sc.command, pid_file=sc.pid_file,
                    health_url=sc.health_url, health_timeout=sc.health_timeout,
                    log_file=str(Path(config.log_dir) / f"{sc.name}.log"),
                    cwd=config.cwd, detached=sc.detached, env=config.env,
                )
        self._claude_cmd = find_claude_cmd(config.claude_cmd) if config.claude_enabled else None
        self._claude_session_dir = str(Path(config.log_dir) / "claude_sessions")

        self._signal_watcher: asyncio.Task | None = None
        self._shutdown = False
        self._restart_requested = False
        self._stop_requested = False

    # ── Intent API (called by Hub's poller; never touches subprocess directly)
    def request_restart(self) -> None:
        self._restart_requested = True
        self._write_shutdown_signal("restart")

    def request_stop(self) -> None:
        self._stop_requested = True
        self._write_shutdown_signal("stop")

    async def do_restart_now(self) -> None:
        """Stop the subprocess so run() picks up the restart flag."""
        if self.subprocess.running:
            await self.subprocess.stop()

    async def do_stop_now(self) -> None:
        if self.subprocess.running:
            await self.subprocess.stop()

    def request_shutdown(self) -> None:
        self._shutdown = True

    @property
    def is_running(self) -> bool:
        return self.subprocess.running

    def status(self) -> dict:
        return {
            "project_id": self.project_id,
            "name": self.cfg.name,
            "app_name": self.cfg.app_name,
            "running": self.subprocess.running,
            "heartbeat_age": self.subprocess.heartbeat_age(),
            "heartbeat_healthy_seconds": self.cfg.heartbeat_healthy_seconds,
            "total_crashes": self.backoff.total_crashes,
            "sidecars": self.sidecars,
            "extra_processes": self.cfg.extra_processes,
        }

    # ── _write_shutdown_signal, _notify_*, _send_start_prompt, _handle_remote,
    #    _start_signal_watcher, _stop_signal_watcher, _signal_watch_loop,
    #    _start_app: copied from guard.py, with self._send(...) -> await self.notify(...)

    async def run(self) -> None:
        """Per-target supervision loop. Body copied from guard.py:654-787
        (the `while not self._shutdown:` loop + shutdown block), with
        self._send(...) -> await self.notify(...). Before the loop, start the
        app once (mirrors guard.py:645-652):"""
        await self._start_app()
        if self.subprocess.running:
            self.backoff.mark_started()
            await self._notify_started()
            await self._start_signal_watcher()
        # ... loop body verbatim from guard.py:654-787 ...
```

**Critical during the copy (spec finding #8):** the `run()` loop calls `_start_signal_watcher()` at 7 points and `_stop_signal_watcher()` at 2 — preserve every one. Grep after writing: `grep -n "_signal_watcher" supervisor.py` must show 9 call sites plus the 3 method defs.

- [ ] **Step 4: Run supervisor tests to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_supervisor.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify old ProcessGuard characterization tests still pass (ProcessGuard untouched this task)**

Run: `python -m pytest packages/yasar_usta/tests/test_guard_run_characterization.py packages/yasar_usta/tests/test_guard_poll_characterization.py -v`
Expected: PASS (ProcessGuard unchanged — supervisor.py is additive).

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/supervisor.py packages/yasar_usta/tests/test_supervisor.py
git commit -m "feat(yasar_usta): TargetSupervisor (one-target supervision, intent API, status)"
```

---

## Task 7: Multi-project dashboard rendering (`status.py` + `commands.py`)

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/status.py` (add `build_project_section`, `build_dashboard_text`)
- Modify: `packages/yasar_usta/src/yasar_usta/commands.py` (add `build_dashboard_keyboard`)
- Create: `packages/yasar_usta/tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
from yasar_usta.status import build_dashboard_text
from yasar_usta.commands import build_dashboard_keyboard


def test_dashboard_lists_all_projects():
    projects = [
        {"project_id": "kutai", "name": "Kutay", "app_name": "Kutay",
         "running": True, "heartbeat_age": 3.0, "heartbeat_healthy_seconds": 90,
         "total_crashes": 0, "extra_processes": []},
        {"project_id": "foo", "name": "Foo", "app_name": "Foo",
         "running": False, "heartbeat_age": None, "heartbeat_healthy_seconds": 90,
         "total_crashes": 2, "extra_processes": []},
    ]
    text = build_dashboard_text("Yaşar Usta", projects, guard_start_time=0.0)
    assert "Kutay" in text and "Foo" in text
    assert "healthy" in text  # kutai
    assert "not running" in text  # foo


def test_dashboard_keyboard_has_per_project_callbacks():
    kb = build_dashboard_keyboard([
        {"project_id": "kutai", "name": "Kutay", "running": True},
        {"project_id": "foo", "name": "Foo", "running": False},
    ])
    flat = str(kb)
    assert "restart:kutai" in flat
    assert "start:foo" in flat
    assert "restart_hub" in flat
    assert "dashboard_refresh" in flat
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_dashboard.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_dashboard_text'`.

- [ ] **Step 3: Add dashboard text builder**

In `status.py`, add (reuse the existing per-line logic):

```python
def build_project_section(st: dict) -> str:
    """One project's health block for the multi-project dashboard."""
    name = st["name"]
    app_name = st["app_name"]
    lines = [f"*{name}*"]
    if not st["running"]:
        lines.append(f"💀 {app_name}: not running")
    else:
        age = st["heartbeat_age"]
        healthy = st["heartbeat_healthy_seconds"]
        if age is not None and age < healthy:
            lines.append(f"💚 {app_name}: healthy (heartbeat {int(age)}s ago)")
        elif age is not None:
            lines.append(f"🔴 {app_name}: UNRESPONSIVE ({int(age)}s silent)")
        else:
            lines.append(f"⚪ {app_name}: no heartbeat file")
    for proc_info in (st.get("extra_processes") or []):
        exe = proc_info.get("exe", "")
        label = proc_info.get("label", exe)
        lines.append(f"{'🟡' if _check_process_running(exe) else '⚫'} {label}")
    if st.get("total_crashes"):
        lines.append(f"  crashes: {st['total_crashes']}")
    return "\n".join(lines)


def build_dashboard_text(hub_name: str, projects: list[dict],
                         guard_start_time: float) -> str:
    up = int(time.time() - guard_start_time)
    header = f"🔧 *{hub_name}* — {up // 3600}h {(up % 3600) // 60}m up\n"
    sections = [build_project_section(p) for p in projects]
    ts = time.strftime("%H:%M:%S")
    return header + "\n\n".join(sections) + f"\n\n_Last update: {ts}_"
```

- [ ] **Step 4: Add dashboard inline keyboard**

In `commands.py`, add:

```python
def build_dashboard_keyboard(projects: list[dict]) -> dict:
    """Inline keyboard: per-project control rows + hub controls."""
    rows = []
    for p in projects:
        pid = p["project_id"]
        label = p.get("name", pid)
        if p.get("running"):
            rows.append([
                {"text": f"♻️ {label}", "callback_data": f"restart:{pid}"},
                {"text": f"⏹ {label}", "callback_data": f"stop:{pid}"},
                {"text": "📋", "callback_data": f"logs:{pid}"},
            ])
        else:
            rows.append([
                {"text": f"▶️ {label}", "callback_data": f"start:{pid}"},
                {"text": "📋", "callback_data": f"logs:{pid}"},
            ])
    rows.append([
        {"text": "🔄 Refresh", "callback_data": "dashboard_refresh"},
        {"text": "♻️ Restart Hub", "callback_data": "restart_hub"},
    ])
    return {"inline_keyboard": rows}
```

- [ ] **Step 5: Run to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_dashboard.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/status.py packages/yasar_usta/src/yasar_usta/commands.py packages/yasar_usta/tests/test_dashboard.py
git commit -m "feat(yasar_usta): multi-project dashboard text + inline keyboard"
```

---

## Task 8: `Hub` — poller, lock, per-target routing, shutdown, self-restart

**Files:**
- Create: `packages/yasar_usta/src/yasar_usta/hub.py`
- Create: `packages/yasar_usta/tests/test_hub.py`

- [ ] **Step 1: Write failing tests (multi-supervisor + routing)**

```python
import asyncio
import pytest
from yasar_usta.config import GuardConfig, HubConfig, ProjectConfig
from yasar_usta.hub import Hub


def _project(pid, tmp_path):
    cfg = GuardConfig(name=pid, app_name=pid, command=["python"],
                      log_dir=str(tmp_path / pid / "logs"), backoff_steps=[1])
    return ProjectConfig(id=pid, name=pid.title(), targets=[cfg])


def _hub(tmp_path, pids):
    hub_cfg = HubConfig(name="Hub", telegram_token="", telegram_chat_id="",
                        log_dir=str(tmp_path / "hublogs"))
    projects = [_project(p, tmp_path) for p in pids]
    return Hub(hub_cfg, projects)


def test_hub_builds_one_supervisor_per_target(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    assert set(hub.supervisors.keys()) == {"kutai", "foo"}


@pytest.mark.asyncio
async def test_route_restart_targets_only_named_project(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    called = {"kutai": 0, "foo": 0}
    for pid, sup in hub.supervisors.items():
        sup.request_restart = lambda p=pid: called.__setitem__(p, called[p] + 1)
        sup.do_restart_now = lambda: asyncio.sleep(0)
    await hub._route_callback("restart:foo", cb_msg_id=None)
    assert called == {"kutai": 0, "foo": 1}


def test_bare_verb_rejected_when_multiple_projects(tmp_path):
    hub = _hub(tmp_path, ["kutai", "foo"])
    assert hub._resolve_bare_target() is None  # ambiguous → None


def test_bare_verb_resolves_when_single_project(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    assert hub._resolve_bare_target().project_id == "kutai"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_hub.py -v`
Expected: FAIL — `ModuleNotFoundError: yasar_usta.hub`.

- [ ] **Step 3: Write `Hub`**

Create `packages/yasar_usta/src/yasar_usta/hub.py`:

```python
"""Hub — owns the shared Telegram poller, the single lock, N supervisors,
the dashboard, coordinated shutdown, and hub self-restart."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from .commands import build_dashboard_keyboard, build_start_keyboard, format_log_entries
from .config import HubConfig, ProjectConfig
from .hooks import load_hook, run_pre_boot
from .lock import acquire_lock, release_lock
from .status import build_dashboard_text
from .supervisor import TargetSupervisor
from .telegram import TelegramAPI

logger = logging.getLogger("yasar_usta.hub")


class Hub:
    def __init__(self, hub_cfg: HubConfig, projects: list[ProjectConfig]):
        self.cfg = hub_cfg
        self.msgs = hub_cfg.messages
        self.projects = projects
        self.telegram = TelegramAPI(hub_cfg.telegram_token, hub_cfg.telegram_chat_id)
        self._guard_start_time = time.time()
        self._shutdown = False
        self._restart_hub = False
        self._telegram_poller: asyncio.Task | None = None

        # One supervisor per target, keyed by a unique routing id. For a
        # single-target project the routing id is the project id; multi-target
        # projects get `${project_id}:${target_id}`.
        self.supervisors: dict[str, TargetSupervisor] = {}
        for proj in projects:
            for tgt in proj.targets:
                rid = proj.id if len(proj.targets) == 1 else f"{proj.id}:{tgt.name}"
                self.supervisors[rid] = TargetSupervisor(rid, tgt, notify=self._notify)

    async def _notify(self, text: str, reply_markup: dict | None = None) -> None:
        await self.telegram.send(text, reply_markup=reply_markup)

    def _kb(self) -> dict:
        return build_start_keyboard(self.msgs, app_name=self.cfg.name)

    def _resolve_bare_target(self) -> TargetSupervisor | None:
        """A bare per-target verb resolves only when there is exactly one
        supervisor (no ambiguity). Otherwise None → reject with a hint."""
        if len(self.supervisors) == 1:
            return next(iter(self.supervisors.values()))
        return None

    # ── Dashboard ────────────────────────────────────────────────────────
    async def _send_dashboard(self, edit_message_id: int | None = None) -> None:
        states = [s.status() for s in self.supervisors.values()]
        text = build_dashboard_text(self.cfg.name, states, self._guard_start_time)
        kb = build_dashboard_keyboard(states)
        try:
            if edit_message_id:
                result = await self.telegram.edit(edit_message_id, text, reply_markup=kb)
                if result and not result.get("ok"):
                    await self.telegram.edit(edit_message_id, text, reply_markup=kb, parse_mode=None)
            else:
                result = await self.telegram.send(text, reply_markup=kb)
                if result and not result.get("ok"):
                    await self.telegram.send(text, reply_markup=kb, parse_mode=None)
        except Exception as e:
            logger.error("dashboard failed: %s", e)

    # ── Callback routing ─────────────────────────────────────────────────
    async def _route_callback(self, cb_data: str, cb_msg_id) -> None:
        if cb_data == "dashboard_refresh":
            await self._send_dashboard(edit_message_id=cb_msg_id)
            return
        if cb_data == "restart_hub":
            await self._notify("♻️ *Hub yeniden başlatılıyor...*")
            await self._do_restart_hub()
            return
        if ":" not in cb_data:
            return
        verb, pid = cb_data.split(":", 1)
        sup = self.supervisors.get(pid)
        if not sup:
            return
        if verb in ("restart", "confirm_restart"):
            sup.request_restart()
            await sup.do_restart_now()
        elif verb in ("stop", "confirm_stop"):
            sup.request_stop()
            await sup.do_stop_now()
        elif verb == "start":
            await sup._start_app()
        elif verb == "logs":
            await self._send_logs_for(sup)

    async def _send_logs_for(self, sup: TargetSupervisor) -> None:
        log_path = sup.cfg.log_file or str(Path(sup.cfg.log_dir) / "orchestrator.jsonl")
        formatted = format_log_entries(log_path, 20)
        await self._notify(formatted or "📋 No log entries.")

    # ── Hub self-restart (spec finding #1) ───────────────────────────────
    async def _do_restart_hub(self) -> None:
        self._shutdown = True
        for sup in self.supervisors.values():
            sup.request_shutdown()
            if sup.is_running:
                await sup.subprocess.stop()
        await self._stop_poller()
        await self.telegram.flush_updates()
        import subprocess as _sp
        script = str(Path(sys.argv[0]).resolve())
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS | _sp.CREATE_NO_WINDOW)
        _sp.Popen([sys.executable, script] + sys.argv[1:], close_fds=True,
                  cwd=str(Path(script).parent), **kwargs)
        release_lock()
        os._exit(0)

    async def _stop_poller(self) -> None:
        if self._telegram_poller:
            self._telegram_poller.cancel()
            try:
                await self._telegram_poller
            except asyncio.CancelledError:
                pass
            self._telegram_poller = None
        await self.telegram.close()

    def request_shutdown(self) -> None:
        self._shutdown = True
        for sup in self.supervisors.values():
            sup.request_shutdown()

    # ── Poll loop ────────────────────────────────────────────────────────
    async def _poll_loop(self, initial_offset: int = 0) -> None:
        # Full implementation added in Task 9.
        ...

    # ── Run ──────────────────────────────────────────────────────────────
    async def run(self) -> None:
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
        acquire_lock(self.cfg.log_dir, name="hub")
        logger.info("Hub started with %d supervisors", len(self.supervisors))

        # pre_boot hooks (per project, once, after lock — spec finding #4)
        for proj in self.projects:
            hook = load_hook(proj.hook_module)
            run_pre_boot(hook, proj)

        offset = await self.telegram.flush_updates()
        await self._notify(
            f"🔧 *{self.cfg.name}* — {len(self.supervisors)} target(s) starting...",
            reply_markup=self._kb())
        if self.telegram.enabled:
            self._telegram_poller = asyncio.create_task(self._poll_loop(offset))

        sup_tasks = [asyncio.create_task(s.run()) for s in self.supervisors.values()]
        try:
            await asyncio.gather(*sup_tasks)
        except asyncio.CancelledError:
            pass
        await self._stop_poller()
        logger.info("Hub exiting.")
```

Note: `test_hub.py` Step 1 references `hooks` (Task 10) — write a temporary no-op `load_hook`/`run_pre_boot` in `hub.py` imports OR implement Task 10 first. **Recommended: do Task 10 before Task 8's Step 4 run.** (The subagent executing this plan should reorder 8↔10 as noted; the two are interdependent only through the `hooks` import.)

- [ ] **Step 4: Run hub tests to verify PASS** (after Task 10 exists)

Run: `python -m pytest packages/yasar_usta/tests/test_hub.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/hub.py packages/yasar_usta/tests/test_hub.py
git commit -m "feat(yasar_usta): Hub — supervisors, lock, routing, dashboard, self-restart"
```

---

## Task 9: Hub poll loop — full command + callback dispatch

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/hub.py` (`_poll_loop`)
- Modify: `packages/yasar_usta/tests/test_hub.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_poll_status_command_sends_dashboard(tmp_path):
    hub = _hub(tmp_path, ["kutai"])
    hub.cfg.telegram_chat_id = "42"
    hub.telegram.token = "x"; hub.telegram.chat_id = "42"
    dash = {"n": 0}
    async def _send_dash(edit_message_id=None):
        dash["n"] += 1
    hub._send_dashboard = _send_dash
    calls = {"n": 0}
    async def get_updates(offset=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"update_id": 1, "message": {"text": "/status", "chat": {"id": 42}}}]
        raise asyncio.CancelledError()
    hub.telegram.get_updates = get_updates
    try:
        await hub._poll_loop(0)
    except asyncio.CancelledError:
        pass
    assert dash["n"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_hub.py -k poll_status -v`
Expected: FAIL — `_poll_loop` is a stub (`...`), dashboard not sent.

- [ ] **Step 3: Implement `_poll_loop`**

Replace the `_poll_loop` stub in `hub.py` with the routing loop (adapted from `guard.py:333-534`):

```python
    async def _poll_loop(self, initial_offset: int = 0) -> None:
        offset = initial_offset
        fail = 0
        logger.info("Hub poller started")
        while True:
            try:
                updates = await self.telegram.get_updates(offset=offset)
                fail = 0
                if not updates:
                    continue
                offset = max(u["update_id"] for u in updates) + 1
                for update in updates:
                    cb = update.get("callback_query")
                    if cb:
                        chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                        if chat == str(self.cfg.telegram_chat_id):
                            await self.telegram.answer_callback(cb["id"])
                            await self._route_callback(
                                cb.get("data", ""),
                                cb.get("message", {}).get("message_id"))
                        continue
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if chat_id != str(self.cfg.telegram_chat_id):
                        continue
                    await self._route_text(text)
            except asyncio.CancelledError:
                return
            except Exception as e:
                fail += 1
                if fail <= 3 or fail % 60 == 0:
                    logger.error("Hub poll error (#%d): %s", fail, e)
                await asyncio.sleep(5)

    async def _route_text(self, text: str) -> None:
        # Hub-global commands
        if text.startswith("/status") or text == self.msgs.btn_status:
            await self._send_dashboard()
            return
        if text.startswith("/restart_hub") or text.startswith("/restart_usta") \
                or text.startswith("/restart_guard"):
            await self._notify("♻️ *Hub yeniden başlatılıyor...*")
            await self._do_restart_hub()
            return
        # Per-target verbs: allowed bare only when exactly one supervisor
        for verb in ("start", "restart", "stop", "logs"):
            if text.startswith("/" + verb):
                sup = self._resolve_bare_target()
                if sup is None:
                    await self._notify(
                        "⚠️ Multiple projects — use the dashboard buttons "
                        "(/status) to pick one.")
                    return
                await self._route_callback(f"{verb}:{sup.project_id}", None)
                return
        # Unknown text while everything is addressable via dashboard
        if text.startswith("/"):
            await self._send_dashboard()
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_hub.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/hub.py packages/yasar_usta/tests/test_hub.py
git commit -m "feat(yasar_usta): Hub poll loop — Hub-global text + per-target callback routing"
```

---

## Task 10: Hook system — `pre_boot`/`on_exit` loader + KutAI hooks module

**Note:** Implement before Task 8 Step 4 (Hub imports `hooks`).

**Files:**
- Create: `packages/yasar_usta/src/yasar_usta/hooks.py`
- Create: `packages/yasar_usta/src/yasar_usta/projects/__init__.py` (empty)
- Create: `packages/yasar_usta/src/yasar_usta/projects/kutai/__init__.py` (empty)
- Create: `packages/yasar_usta/src/yasar_usta/projects/kutai/hooks.py`
- Create: `packages/yasar_usta/tests/test_hooks.py`

- [ ] **Step 1: Write the failing test**

```python
from yasar_usta.hooks import load_hook, run_pre_boot
from yasar_usta.config import GuardConfig, ProjectConfig


def test_load_hook_returns_none_for_missing():
    assert load_hook(None) is None
    assert load_hook("yasar_usta.projects.nope.hooks") is None


def test_pre_boot_invoked(tmp_path):
    # a fake module path won't exist; use a stub object with pre_boot
    calls = []
    class _Hook:
        @staticmethod
        def pre_boot(project):
            calls.append(project.id)
    proj = ProjectConfig(id="demo", name="Demo",
                         targets=[GuardConfig(name="t", command=["python"])])
    run_pre_boot(_Hook, proj)
    assert calls == ["demo"]


def test_kutai_hook_importable():
    hook = load_hook("yasar_usta.projects.kutai.hooks")
    assert hook is not None
    assert hasattr(hook, "pre_boot")
    assert hasattr(hook, "on_exit")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_hooks.py -v`
Expected: FAIL — `ModuleNotFoundError: yasar_usta.hooks`.

- [ ] **Step 3: Write the hook loader**

Create `packages/yasar_usta/src/yasar_usta/hooks.py`:

```python
"""Optional per-project lifecycle hooks. A hook module may define:
  pre_boot(project)   -> runs once, after the hub lock, before supervisors start
  on_exit(exit_code)  -> passed to each target's GuardConfig.on_exit
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("yasar_usta.hooks")


def load_hook(module_path: str | None):
    if not module_path:
        return None
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        logger.error("Hook module %s failed to import: %s", module_path, e)
        return None


def run_pre_boot(hook, project) -> None:
    if hook is None or not hasattr(hook, "pre_boot"):
        return
    try:
        hook.pre_boot(project)
    except Exception as e:
        # Surfaced, not swallowed (spec: KutAI pre_boot failure must be visible)
        logger.error("pre_boot for %s failed: %s", project.id, e)
        raise
```

- [ ] **Step 4: Write the KutAI hooks module**

Create `packages/yasar_usta/src/yasar_usta/projects/kutai/hooks.py` — move the three functions from `kutai_wrapper.py` verbatim (`_kill_stale_orchestrators`, `_reconcile_stray_llama`, `_kill_orphan_processes`), wrapped:

```python
"""KutAI-specific lifecycle hooks (moved verbatim from kutai_wrapper.py)."""

from __future__ import annotations

import os
import subprocess as _sp


def _kill_orphan_processes(exit_code: int) -> None:
    # ... verbatim body from kutai_wrapper.py:43-68 ...
    ...


def _kill_stale_orchestrators() -> None:
    # ... verbatim body from kutai_wrapper.py:71-97 ...
    ...


def _reconcile_stray_llama() -> None:
    # ... verbatim body from kutai_wrapper.py:100-126 ...
    ...


def pre_boot(project) -> None:
    """Runs once before KutAI's supervisor starts (was module-import cleanup)."""
    _kill_stale_orchestrators()
    _reconcile_stray_llama()


def on_exit(exit_code: int) -> None:
    _kill_orphan_processes(exit_code)
```

Also add empty `__init__.py` files at `projects/` and `projects/kutai/`.

- [ ] **Step 5: Run to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_hooks.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/hooks.py packages/yasar_usta/src/yasar_usta/projects/
git add packages/yasar_usta/tests/test_hooks.py
git commit -m "feat(yasar_usta): pre_boot/on_exit hook loader + KutAI hooks module"
```

---

## Task 11: Wire `on_exit` hook into supervisors + export `Hub`

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/hub.py` (attach `on_exit` from hook to each target's `GuardConfig`)
- Modify: `packages/yasar_usta/src/yasar_usta/__init__.py` (export `Hub`, `HubConfig`, `ProjectConfig`, `load_registry`, `TargetSupervisor`)

- [ ] **Step 1: Attach on_exit in Hub.__init__**

In `hub.py` `__init__`, before building supervisors, load each project's hook and set `on_exit`:

```python
        for proj in projects:
            hook = load_hook(proj.hook_module)
            for tgt in proj.targets:
                if hook is not None and hasattr(hook, "on_exit"):
                    tgt.on_exit = hook.on_exit
                rid = proj.id if len(proj.targets) == 1 else f"{proj.id}:{tgt.name}"
                self.supervisors[rid] = TargetSupervisor(rid, tgt, notify=self._notify)
```

(Remove the earlier plain supervisor-building loop from Task 8 Step 3 so this replaces it. The `run()` method's separate `load_hook`/`run_pre_boot` loop stays for `pre_boot`.)

- [ ] **Step 2: Export the new public API**

In `__init__.py`, add imports + `__all__` entries:

```python
from .config import GuardConfig, HubConfig, Messages, ProjectConfig, SidecarConfig
from .guard import ProcessGuard
from .hub import Hub
from .registry import load_registry
from .supervisor import TargetSupervisor
```

Add `"Hub"`, `"HubConfig"`, `"ProjectConfig"`, `"TargetSupervisor"`, `"load_registry"` to `__all__`.

- [ ] **Step 3: Verify imports + existing suite**

Run: `python -c "from yasar_usta import Hub, HubConfig, ProjectConfig, load_registry, TargetSupervisor; print('ok')"`
Expected: `ok`.

Run: `python -m pytest packages/yasar_usta/tests/ -q`
Expected: all pass (old ProcessGuard tests + all new).

- [ ] **Step 4: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/hub.py packages/yasar_usta/src/yasar_usta/__init__.py
git commit -m "feat(yasar_usta): wire on_exit hook into supervisors + export Hub API"
```

---

## Task 12: `registry.yaml` for KutAI + migration guard test

**Files:**
- Create: `registry.yaml` (repo root)
- Create: `packages/yasar_usta/tests/test_migration_kutai.py`

- [ ] **Step 1: Write the migration guard test (config-equivalence — spec finding #9 test 3)**

```python
"""Prove the KutAI registry block reproduces the hardcoded kutai_wrapper.py
GuardConfig 1:1 (config-equivalence only; runtime behavior is gated by the
characterization suite)."""
import os
from pathlib import Path
from yasar_usta.registry import load_registry

ROOT = Path(__file__).resolve().parents[3]  # repo root


def test_kutai_registry_matches_legacy_config():
    os.environ.setdefault("YASAR_USTA_BOT_TOKEN", "tok")
    os.environ.setdefault("TELEGRAM_ADMIN_CHAT_ID", "42")
    hub, projects = load_registry(ROOT / "registry.yaml", project_root=str(ROOT))
    kutai = next(p for p in projects if p.id == "kutai")
    t = kutai.targets[0]
    assert t.command[-1].endswith("run.py")
    assert t.cwd == str(ROOT)
    assert t.restart_exit_code == 42
    assert t.heartbeat_file == str(ROOT / "logs" / "orchestrator.heartbeat")
    assert t.log_file == str(ROOT / "logs" / "orchestrator.jsonl")
    assert t.env["NERD_HERD_PROJECT_ROOT"] == str(ROOT)
    names = {sc.name for sc in t.sidecars}
    assert names == {"yazbunu", "nerd_herd"}
    assert kutai.hook_module == "yasar_usta.projects.kutai.hooks"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_migration_kutai.py -v`
Expected: FAIL — `registry.yaml` does not exist.

- [ ] **Step 3: Write `registry.yaml`**

Create `registry.yaml` at repo root. Mirror `kutai_wrapper.py:135-192` exactly. Note `${project_root}` resolves to the repo root; the `LLAMA_SERVER_PORT`/`DB_PATH` remain read from the process env by the sidecars themselves at runtime (KutAI keeps the single `.env` — spec finding R2c), so they are passed through target `env` where needed.

```yaml
hub:
  name: "Yaşar Usta"
  telegram_token_env: YASAR_USTA_BOT_TOKEN
  telegram_chat_id_env: TELEGRAM_ADMIN_CHAT_ID
  log_dir: "${project_root}/logs"

projects:
  kutai:
    name: Kutay
    hook_module: yasar_usta.projects.kutai.hooks
    targets:
      - id: orchestrator
        app_name: Kutay
        command: ["${project_root}/.venv/Scripts/python.exe", "${project_root}/src/app/run.py"]
        cwd: "${project_root}"
        env:
          NERD_HERD_PROJECT_ROOT: "${project_root}"
        heartbeat_file: "${project_root}/logs/orchestrator.heartbeat"
        heartbeat_stale_seconds: 120
        heartbeat_healthy_seconds: 90
        restart_exit_code: 42
        log_dir: "${project_root}/logs"
        log_file: "${project_root}/logs/orchestrator.jsonl"
        stop_timeout: 30
        claude_name: Kutay
        claude_signal_file: "${project_root}/logs/claude_remote.signal"
        extra_processes:
          - {exe: "llama-server.exe", label: "llama-server"}
        sidecars:
          - name: yazbunu
            command: ["${project_root}/.venv/Scripts/python.exe", "-m", "yazbunu.server", "--log-dir", "./logs", "--port", "9880", "--host", "0.0.0.0"]
            health_url: "http://127.0.0.1:9880/health"
            pid_file: "${project_root}/logs/yazbunu.pid"
            detached: true
          - name: nerd_herd
            command: ["${project_root}/.venv/Scripts/python.exe", "-m", "nerd_herd", "--port", "9881", "--pid-file", "${project_root}/logs/nerd_herd.pid"]
            health_url: "http://127.0.0.1:9881/health"
            pid_file: "${project_root}/logs/nerd_herd.pid"
            detached: true
```

Note: the `claude_cmd`, `--db-path`, and `--no-auto-restart` argv handling from the legacy wrapper move to the entry point (Task 13), which computes them and injects into the loaded config before constructing the Hub (registry stays declarative; process-specific runtime values are applied in code).

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_migration_kutai.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add registry.yaml packages/yasar_usta/tests/test_migration_kutai.py
git commit -m "feat(yasar_usta): KutAI registry.yaml + migration config-equivalence guard"
```

---

## Task 13: Rewrite `kutai_wrapper.py` as the hub entry point

**Files:**
- Modify: `kutai_wrapper.py` (rewrite: load registry → apply runtime values → run Hub; keep signal handlers + venv guard Hub-global)

- [ ] **Step 1: Rewrite the entry point**

Replace the body of `kutai_wrapper.py` after the venv guard (`:28` onward) with:

```python
from pathlib import Path
import asyncio, os, signal, sys

from yasar_usta import Hub, load_registry

PROJECT_ROOT = Path(__file__).resolve().parent


def _apply_runtime_values(hub_cfg, projects) -> None:
    """Inject process-specific runtime values the declarative registry can't hold."""
    appdata = os.environ.get("APPDATA", "")
    claude_cmd = str(Path(appdata) / "npm" / "claude.cmd") if appdata else None
    auto_restart = "--no-auto-restart" not in sys.argv
    db_path = os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "kutai.db"))
    for proj in projects:
        for tgt in proj.targets:
            tgt.auto_restart = auto_restart
            if claude_cmd:
                tgt.claude_cmd = claude_cmd
            # nerd_herd sidecar needs --db-path appended (kept out of YAML)
            for sc in tgt.sidecars:
                if sc.name == "nerd_herd" and "--db-path" not in sc.command:
                    sc.command += ["--db-path", db_path]


async def main():
    hub_cfg, projects = load_registry(PROJECT_ROOT / "registry.yaml",
                                      project_root=str(PROJECT_ROOT))
    _apply_runtime_values(hub_cfg, projects)
    hub = Hub(hub_cfg, projects)

    def _sig(sig, frame):
        print(f"\n[Yasar Usta] Signal {sig} received, shutting down...")
        hub.request_shutdown()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)

    if sys.platform == "win32":
        try:
            import ctypes

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)
            def _console_handler(event):
                if event in (0, 2):
                    hub.request_shutdown()
                    return True
                return False

            ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)
            hub._console_handler = _console_handler  # GC anchor
        except Exception:
            pass

    await hub.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Yasar Usta] KeyboardInterrupt — exiting")
    except Exception as exc:
        print(f"[Yasar Usta] FATAL: {exc!r}")
        raise
```

Delete the now-relocated `_kill_orphan_processes`, `_kill_stale_orchestrators`, `_reconcile_stray_llama`, `_find_python`, the module-level cleanup calls, the giant `GuardConfig(...)` literal, and the `os.environ["NERD_HERD_PROJECT_ROOT"] = ...` mutation (now per-target env). Keep `load_dotenv()` and the venv guard.

- [ ] **Step 2: Smoke-test the entry point loads (no bot token → poller disabled, supervisors would try to start; use --help-style dry check)**

Run: `python -c "import kutai_wrapper; from yasar_usta import load_registry; from pathlib import Path; h,p=load_registry(Path('registry.yaml'), project_root='.'); print(len(p), 'projects')"`
Expected: prints `1 projects` (import side-effect free; `load_dotenv` + venv guard run at import — acceptable).

- [ ] **Step 3: Run the FULL yasar_usta suite**

Run: `python -m pytest packages/yasar_usta/tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add kutai_wrapper.py
git commit -m "feat(yasar_usta): rewrite entry point as multi-project hub (KutAI first target)"
```

---

## Task 14: `remote.py` temp-log collision fix (spec finding #7)

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/remote.py:119`
- Modify: `packages/yasar_usta/src/yasar_usta/supervisor.py` (pass a `project_id` into `start_claude_remote`)
- Modify: `packages/yasar_usta/tests/test_remote.py`

- [ ] **Step 1: Write the failing test**

Append to `packages/yasar_usta/tests/test_remote.py`:

```python
import inspect
from yasar_usta import remote


def test_start_claude_remote_accepts_session_label():
    sig = inspect.signature(remote.start_claude_remote)
    assert "session_label" in sig.parameters
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/yasar_usta/tests/test_remote.py -k session_label -v`
Expected: FAIL — no such parameter.

- [ ] **Step 3: Add a uniqueness label**

In `remote.py`, add `session_label: str | None = None,` to `start_claude_remote`'s signature. Replace line 119:

```python
            import uuid as _uuid
            label = session_label or "s"
            log_path = sdir / f"_starting_{label}_{_uuid.uuid4().hex[:8]}.log"
```

In `supervisor.py` `_handle_remote`, pass `session_label=self.project_id` into the `start_claude_remote(...)` call.

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest packages/yasar_usta/tests/test_remote.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/remote.py packages/yasar_usta/src/yasar_usta/supervisor.py packages/yasar_usta/tests/test_remote.py
git commit -m "fix(yasar_usta): unique claude temp-log name per target (avoid hub-PID collision)"
```

---

## Task 15: Integration test + full suite + live-verify checklist

**Files:**
- Create: `packages/yasar_usta/tests/test_hub_integration.py`

- [ ] **Step 1: Write an integration test — two fake targets, isolation**

```python
"""Two short-lived fake targets under one Hub: one crash-loops, the other
runs clean; assert isolation (one crash does not stop the other) and that
coordinated shutdown stops both."""
import asyncio
import sys
import pytest
from yasar_usta.config import GuardConfig, HubConfig, ProjectConfig
from yasar_usta.hub import Hub


def _proj(pid, tmp_path, code):
    cfg = GuardConfig(
        name=pid, app_name=pid,
        command=[sys.executable, "-c", code],
        log_dir=str(tmp_path / pid / "logs"),
        backoff_steps=[1], auto_restart=False,
        telegram_token="", telegram_chat_id="",
    )
    return ProjectConfig(id=pid, name=pid, targets=[cfg])


@pytest.mark.asyncio
async def test_two_targets_run_independently(tmp_path):
    hub_cfg = HubConfig(telegram_token="", telegram_chat_id="",
                        log_dir=str(tmp_path / "hub"))
    projects = [
        _proj("good", tmp_path, "import time; time.sleep(0.3)"),
        _proj("bad", tmp_path, "import sys; sys.exit(3)"),
    ]
    hub = Hub(hub_cfg, projects)
    # Stop the hub shortly after start so run() returns.
    async def _killer():
        await asyncio.sleep(1.5)
        hub.request_shutdown()
        for s in hub.supervisors.values():
            if s.is_running:
                await s.subprocess.stop()
    asyncio.create_task(_killer())
    await asyncio.wait_for(hub.run(), timeout=8)
    # both supervisors were constructed and reachable
    assert set(hub.supervisors) == {"good", "bad"}
```

- [ ] **Step 2: Run the integration test**

Run: `python -m pytest packages/yasar_usta/tests/test_hub_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Run the ENTIRE yasar_usta suite**

Run: `python -m pytest packages/yasar_usta/tests/ -q`
Expected: all pass (old + new). If any old ProcessGuard test now fails because a responsibility moved, update it in lockstep and note it in the commit.

- [ ] **Step 4: Commit**

```bash
git add packages/yasar_usta/tests/test_hub_integration.py
git commit -m "test(yasar_usta): hub integration — two-target isolation + coordinated shutdown"
```

- [ ] **Step 5: Live-verify checklist (restart-gated — do BEFORE any push)**

Manual, on the real machine (this is the gate before pushing):
1. Full wrapper relaunch (not `/restart` — the entry point changed): stop the running wrapper, start `python kutai_wrapper.py`.
2. Telegram: confirm the announce + reply keyboard arrive; `/status` shows the KutAI dashboard block (healthy heartbeat).
3. Press ♻️ restart button for KutAI → orchestrator restarts, heartbeat recovers, no spurious crash card.
4. Press ⏹ stop → orchestrator stops; press ▶️ start → it comes back.
5. Confirm sidecars (yazbunu :9880, nerd_herd :9881) are alive and `NERD_HERD_PROJECT_ROOT` reached nerd_herd (its health endpoint responds).
6. `/restart_hub` → hub respawns, re-reads registry, KutAI comes back up.
7. Confirm `logs/hub.lock` / `logs/hub.lk` exist (not `guard.lock`), and only one hub instance.
8. Watch `logs/` for one full crash→backoff→restart cycle if convenient (kill the orchestrator PID, NOT llama-server).

- [ ] **Step 6: Push (only after all live-verify steps pass)**

```bash
git push
```

---

## Self-Review

**Spec coverage:**
- Hub/TargetSupervisor split → Tasks 6, 8, 9, 11. ✅
- YAML registry + `${}` resolution → Task 3, 12. ✅
- Per-project hook module (`pre_boot`/`on_exit`) → Task 10, 11. ✅
- Inline multi-project dashboard + per-target callbacks → Task 7, 9. ✅
- KutAI migrated 1:1 + migration guard → Task 12, 13. ✅
- Finding #1 (self-restart → Hub) → Task 8 `_do_restart_hub` + `os._exit` only in Hub. ✅
- Finding #2 (`request_*()` API, poller never touches subprocess) → Task 6 + Task 9 routing. ✅
- Finding #3 (KutAI stays on `logs/`) → registry uses `${project_root}/logs`; app-side unchanged. ✅
- Finding #4 (`pre_boot` after lock; per-target env; Hub-global signal handlers) → Task 10 (pre_boot), Task 4/5 (env), Task 13 (signal handlers). ✅
- Finding #5 (stateful callback ids) → Task 7/9 (`verb:pid`). ✅
- Finding #6 (lock hoist to one hub lock) → Task 8 `acquire_lock(..., name="hub")`; supervisors never lock. ✅
- Finding #7 (claude temp-log uuid) → Task 14. ✅
- Finding #8 (9 watcher call sites) → Task 6 Step 3 grep gate. ✅
- Finding #9 (characterization-first) → Tasks 1, 2. ✅
- Finding #10 (preserve `_send_start_prompt`) → copied verbatim in Task 6. ✅
- R1 (Hub owns TelegramAPI + status render; supervisor gets `notify`) → Task 6/8. ✅
- R2 (env through SubprocessManager + SidecarManager, merge) → Task 4, 5. ✅
- R3 (token/chat → HubConfig, dropped from migration assert) → Task 3, 12. ✅
- R4 (text Hub-global; per-target via buttons; bare verb single-project-only) → Task 9. ✅

**Known deviation from spec (noted, consistent with R4 intent):** a bare per-target text verb (`/restart`) IS honored when exactly one project exists (no ambiguity → no guessing). With >1 project it is rejected with a hint. This preserves KutAI's current single-target UX 1:1 while staying safe at scale.

**Ordering note:** Task 8 (Hub) imports `hooks` from Task 10 — implement Task 10 before running Task 8 Step 4. Both are committed independently.

**Placeholder scan:** the `...` markers in Task 6 (`TargetSupervisor` method bodies) and Task 10 (KutAI hook bodies) are explicit "copy verbatim from `guard.py`/`kutai_wrapper.py` at the cited line ranges" instructions, not vague TODOs — the source is exact and in-repo. All test code and all new logic (registry, env merge, Hub routing, dashboard, hooks loader, entry point) is complete.

**Type consistency:** `TargetSupervisor(project_id, config, notify=)`, `Hub(hub_cfg, projects)`, `load_registry(path, project_root=) -> (HubConfig, [ProjectConfig])`, `status()` dict keys used identically in Task 6/7/8. `request_restart/request_stop/request_shutdown/do_restart_now/do_stop_now/is_running/status` consistent across Tasks 6, 8, 9.
