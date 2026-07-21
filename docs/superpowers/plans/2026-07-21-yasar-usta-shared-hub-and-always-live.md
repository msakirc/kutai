# Yaşar Usta Shared Hub ⨝ Always-Live — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate Yaşar Usta out of the KutAI monorepo into a standalone, multi-project sibling repo with its own venv, decouple all KutAI-specific code, move runtime state out of Dropbox, and preserve the always-live/never-duplicate guarantees — so more projects can connect with zero shared-venv pollution.

**Architecture:** One shared hub daemon (own repo `Workspaces/yasar_usta/`, own `.venv`, singleton kernel mutex). Consumer projects contribute a declarative registry block + a `yasar_hooks.py` that runs as a **subprocess in that project's own venv**. Code lives in Dropbox; runtime state lives in `%LOCALAPPDATA%\YasarUsta\`. Delivered in 4 phases, each independently live-verified.

**Tech Stack:** Python 3.10, `asyncio`, `aiohttp`, `pyyaml`, `python-dotenv`, `psutil`, Windows Task Scheduler, editable pip installs, pytest.

**Design spec:** `docs/superpowers/specs/2026-07-21-yasar-usta-shared-hub-and-always-live-design.md`
**Handoff merged:** `docs/handoff/2026-07-21-yasar-usta-always-live-handoff.md`

**Conventions in this plan:**
- Every task states its **Repo/CWD**: `HUB` = `C:\Users\sakir\Dropbox\Workspaces\yasar_usta` · `KUTAY` = `C:\Users\sakir\Dropbox\Workspaces\kutay`.
- `HUB_PY` = `C:\Users\sakir\Dropbox\Workspaces\yasar_usta\.venv\Scripts\python.exe`
- `KUTAY_PY` = `C:\Users\sakir\Dropbox\Workspaces\kutay\.venv\Scripts\python.exe`
- Run pytest **targeted, foreground, with a timeout**; kill only your own hung processes; **never** taskkill llama-server. Do not launch the live hub during test tasks.
- The live production hub keeps running the OLD code from `packages/yasar_usta` until Phase 1's final live-verify; do not disturb it mid-phase.

---

## File Structure

### New repo `HUB` (`Workspaces/yasar_usta/`)
- `src/yasar_usta/` — generic package, moved via subtree. `projects/` subpackage **deleted**.
- `src/yasar_usta/__main__.py` — **NEW** generic entry (was `kutai_wrapper.main`): `load_dotenv`, argparse `--registry`/`--no-auto-restart`, signal handling, `Hub.run()`.
- `src/yasar_usta/hooks.py` — **rewritten**: subprocess dispatch (spawn `<project_venv_python> <hook_path> <phase> --context <json>`), replaces in-process `importlib`.
- `src/yasar_usta/registry.py` — **extended**: `${env:VAR}` tokens, per-project `root`/`state_dir` tokens, `venv_python` + `hook` (path) + `messages` per project.
- `src/yasar_usta/config.py` — **extended**: `ProjectConfig` gains `venv_python`, `hook_path`; `GuardConfig`/`ProjectConfig` gain `state_dir` (Phase 2).
- `src/yasar_usta/watchdog.py` — **modified**: `find_hub_pids` matcher off `kutai_wrapper.py`; Phase 3 M4b grace/stopped/kill-verify.
- `src/yasar_usta/hub.py` — **modified**: `_do_restart_hub` `-m` form; boot-asserts after mutex; subprocess pre_boot; Phase 2 state_dir.
- `src/yasar_usta/guard.py` — **moved as-is** (legacy, still exported); retire later.
- `registry.yaml` — **NEW hub-side** (moved from `KUTAY`), extended per project.
- `start.bat` — **NEW** hub launcher.
- `.env` — **NEW** hub creds (`YASAR_USTA_BOT_TOKEN`, `TELEGRAM_ADMIN_CHAT_ID`). Not committed.
- `pyproject.toml` — deps `aiohttp`, `pyyaml`, `python-dotenv`; `console_scripts` `yasar-usta`.
- `scripts/install_yasar_autostart.ps1` — **moved + rewritten** for generic entry + LOCALAPPDATA.
- `tests/` — generic tests moved; new subprocess-hook, matcher, loader-token, boot-assert, M4b tests.

### `KUTAY` (`Workspaces/kutay/`)
- `packages/yasar_usta/` — **deleted** after subtree move.
- `yasar_hooks.py` — **NEW** at repo root (was `projects/kutai/hooks.py`), CLI `pre_boot`/`on_exit`, imports `dallama`/`psutil` in kutay venv.
- `kutai_wrapper.py` — **deleted**.
- `start_kutai.bat` — **modified** to launch the hub.
- `requirements.txt` — `-e ./packages/yasar_usta` → `-e ../yasar_usta`.
- `src/app/run.py`, `src/core/orchestrator.py` — Phase 2: heartbeat path from `YASAR_USTA_STATE_DIR`.
- `tests/integration/test_restart_shutdown.py`, `tests/test_wrapper_logs.py` — rewrite against new entry.
- `tests/yasar/test_kutai_hooks.py`, `tests/yasar/test_migration_kutai.py` — moved from the package.
- `CLAUDE.md`, `.claude/settings.local.json` — doc/perms refresh.

---

# PHASE 1 — Relocate + decouple + re-point entry (state stays in project `logs/`)

**Phase goal:** Hub runs from its own repo + own venv, manages KutAI exactly as before, with all KutAI code on the kutay side. **No state-location change.** Live-verify = clean regression.

---

### Task 1.1: Create the sibling repo via subtree split (preserve history)

**Repo/CWD:** `KUTAY`
**Files:** creates `Workspaces/yasar_usta/` (new git repo)

- [ ] **Step 1: Split the package subtree into a branch**

Run (in `KUTAY`):
```bash
git subtree split --prefix=packages/yasar_usta -b yasar-usta-export
```
Expected: prints a new commit sha; branch `yasar-usta-export` created. If subtree misbehaves on Windows, fallback: skip to Step 2b.

- [ ] **Step 2: Create the new repo from that branch**

Run:
```bash
mkdir /c/Users/sakir/Dropbox/Workspaces/yasar_usta
cd /c/Users/sakir/Dropbox/Workspaces/yasar_usta
git init
git pull /c/Users/sakir/Dropbox/Workspaces/kutay yasar-usta-export
```
Expected: `src/yasar_usta/…`, `tests/…`, `pyproject.toml` present with history.

- [ ] **Step 2b (FALLBACK only if subtree failed): plain copy**

```bash
mkdir -p /c/Users/sakir/Dropbox/Workspaces/yasar_usta
cp -r /c/Users/sakir/Dropbox/Workspaces/kutay/packages/yasar_usta/* /c/Users/sakir/Dropbox/Workspaces/yasar_usta/
cd /c/Users/sakir/Dropbox/Workspaces/yasar_usta && git init && git add -A && git commit -m "chore: import yasar_usta from kutay monorepo"
```

- [ ] **Step 3: Verify the tree landed**

Run: `ls /c/Users/sakir/Dropbox/Workspaces/yasar_usta/src/yasar_usta/`
Expected: `hub.py config.py registry.py hooks.py watchdog.py singleton.py heartbeat.py guard.py … projects/` present.

- [ ] **Step 4: Carry the installer (it lives OUTSIDE the package — subtree won't take it)**

`install_yasar_autostart.ps1` is at `KUTAY/scripts/`, **not** in `packages/yasar_usta/`, so neither the subtree split nor the fallback copy carries it. Bring it over:
```bash
mkdir -p /c/Users/sakir/Dropbox/Workspaces/yasar_usta/scripts
cp /c/Users/sakir/Dropbox/Workspaces/kutay/scripts/install_yasar_autostart.ps1 /c/Users/sakir/Dropbox/Workspaces/yasar_usta/scripts/
```
Expected: `HUB/scripts/install_yasar_autostart.ps1` exists (it gets rewritten in Task 3.4; deleted from KUTAY in Task 1.14).

- [ ] **Step 5: Clean the export branch**

Run (in `KUTAY`): `git branch -D yasar-usta-export`
Expected: branch deleted. (Do NOT delete `packages/yasar_usta` yet — that happens in Task 1.14 after verification.)

---

### Task 1.2: Add python-dotenv dep + console script to the hub pyproject

**Repo/CWD:** `HUB`
**Files:** Modify: `pyproject.toml`

- [ ] **Step 1: Edit pyproject**

Replace the `[project]` deps and add scripts:
```toml
[project]
name = "yasar-usta"
version = "0.2.0"
description = "Multi-project Telegram-controlled process hub with singleton mutex, heartbeat watchdog, and auto-restart"
requires-python = ">=3.10"
dependencies = ["aiohttp>=3.9.0", "pyyaml>=6.0", "python-dotenv>=1.0.0", "psutil>=5.9"]

[project.optional-dependencies]
test = ["pytest>=8.0", "pytest-timeout>=2.1"]

[project.scripts]
yasar-usta = "yasar_usta.__main__:main_cli"

[tool.setuptools.packages.find]
where = ["src"]
```
(`main_cli` is defined in Task 1.9.) **`psutil` is mandatory** — `watchdog.py` imports it (`find_hub_pids`, `is_pid_alive`); without it the standalone-venv watchdog dies with `ModuleNotFoundError` and always-live is silently defeated.

- [ ] **Step 2: Commit**

```bash
cd /c/Users/sakir/Dropbox/Workspaces/yasar_usta
git add pyproject.toml && git commit -m "build: add python-dotenv dep + yasar-usta console script"
```

---

### Task 1.3: Create the hub's own venv + editable install

**Repo/CWD:** `HUB`
**Files:** creates `Workspaces/yasar_usta/.venv`

- [ ] **Step 1: Create venv (system Python 3.10, same as kutay)**

Run:
```bash
cd /c/Users/sakir/Dropbox/Workspaces/yasar_usta
py -3.10 -m venv .venv
```
Expected: `.venv/Scripts/python.exe` exists. If the `py` launcher is absent, fall back to the explicit interpreter: `C:/Python310/python.exe -m venv .venv` (the box's system Python is Python310 per kutay CLAUDE.md).

- [ ] **Step 2: Editable install (with the test extra — pytest is NOT otherwise present in this fresh venv)**

Run: `.venv/Scripts/python.exe -m pip install -e ".[test]"`
Expected: installs `yasar-usta 0.2.0` + `aiohttp`, `pyyaml`, `python-dotenv`, `psutil`, `pytest`, `pytest-timeout`. **Every test task below runs `-m pytest` in this venv — without this, Task 1.4 Step 2 errors with `No module named pytest`.**

- [ ] **Step 3: Verify import**

Run: `.venv/Scripts/python.exe -c "import yasar_usta; from yasar_usta import Hub, load_registry, HeartbeatWriter; print('hub import OK')"`
Expected: `hub import OK`.

- [ ] **Step 4: Add .venv to .gitignore + commit**

Append `.venv/` and `.env` to `HUB/.gitignore` (create if missing), then:
```bash
git add .gitignore && git commit -m "chore: ignore .venv and .env"
```

---

### Task 1.4: Registry loader — `${env:VAR}` + per-project `root`/`state_dir`

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/registry.py` · Test: `tests/test_registry_tokens.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry_tokens.py`:
```python
import os
from pathlib import Path
from yasar_usta.registry import load_registry


def _write(tmp_path, text):
    p = tmp_path / "registry.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_env_token_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_APPDATA", r"C:\Users\x\AppData\Roaming")
    reg = _write(tmp_path, """
hub:
  name: T
projects:
  p1:
    root: C:/proj/p1
    targets:
      - id: orch
        command: ["${env:MY_APPDATA}/tool.exe", "${project_root}/run.py"]
""")
    hub, projects = load_registry(reg, project_root="C:/UNUSED")
    cmd = projects[0].targets[0].command
    assert cmd[0].endswith("tool.exe") and "AppData" in cmd[0]
    assert cmd[1].endswith("run.py") and "p1" in cmd[1].replace("\\", "/")


def test_per_project_root_overrides_global(tmp_path):
    reg = _write(tmp_path, """
projects:
  p1:
    root: C:/proj/one
    targets:
      - {id: a, command: ["${project_root}/a.py"]}
  p2:
    root: C:/proj/two
    targets:
      - {id: b, command: ["${project_root}/b.py"]}
""")
    _, projects = load_registry(reg, project_root="C:/GLOBAL")
    p1 = next(p for p in projects if p.id == "p1")
    p2 = next(p for p in projects if p.id == "p2")
    assert "one" in p1.targets[0].command[0].replace("\\", "/")
    assert "two" in p2.targets[0].command[0].replace("\\", "/")


def test_env_token_missing_raises(tmp_path):
    reg = _write(tmp_path, """
projects:
  p1:
    root: C:/x
    targets:
      - {id: a, command: ["${env:DEFINITELY_UNSET_VAR_XYZ}/a"]}
""")
    import pytest
    with pytest.raises(ValueError):
        load_registry(reg, project_root="C:/x")
```

- [ ] **Step 2: Run test — verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_tokens.py -v`
Expected: FAIL (env token not substituted / per-project root ignored).

- [ ] **Step 3: Implement the token changes**

In `registry.py`, extend `_resolve` to handle `${env:VAR}` (fail loud on unset) and thread a per-project token dict. Replace the string branch of `_resolve`:
```python
import re

_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve(value, tokens: dict):
    if isinstance(value, str):
        def _env(m):
            name = m.group(1)
            val = os.environ.get(name)
            if val is None:
                raise ValueError(f"registry references unset env var: {name}")
            return val
        value = _ENV_RE.sub(_env, value)
        for k, v in tokens.items():
            value = value.replace("${" + k + "}", v)
        return value
    if isinstance(value, list):
        return [_resolve(x, tokens) for x in value]
    if isinstance(value, dict):
        return {k: _resolve(v, tokens) for k, v in value.items()}
    return value
```
In `load_registry`, build per-project tokens (each project's `root` overrides the global `project_root`):
```python
    for pid, raw_proj in data["projects"].items():
        proj_root = raw_proj.get("root", project_root)
        tokens = {"project_root": proj_root}
        if "targets" not in raw_proj or not raw_proj["targets"]:
            raise ValueError(f"project {pid!r} has no targets")
        targets = [_build_target(t, tokens) for t in raw_proj["targets"]]
        projects.append(ProjectConfig(
            id=pid,
            name=raw_proj.get("name", pid),
            targets=targets,
            hook_module=raw_proj.get("hook_module"),
        ))
```
(Remove the old single-`tokens` construction above the loop; keep the `hub` block using a `{"project_root": project_root}` token dict for `log_dir`.)

- [ ] **Step 4: Run test — verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_tokens.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the moved registry suite for regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry.py -v`
Expected: all pass (existing behavior preserved).

- [ ] **Step 6: Commit**

```bash
git add src/yasar_usta/registry.py tests/test_registry_tokens.py
git commit -m "feat(registry): \${env:} tokens + per-project root override"
```

---

### Task 1.5: Config — `ProjectConfig` gains `venv_python` + `hook_path`

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/config.py`, `src/yasar_usta/registry.py` · Test: `tests/test_registry_hookfields.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry_hookfields.py`:
```python
from yasar_usta.registry import load_registry


def test_project_venv_python_and_hook_path(tmp_path):
    reg = tmp_path / "r.yaml"
    reg.write_text("""
projects:
  kutai:
    root: C:/kutay
    venv_python: ${project_root}/.venv/Scripts/python.exe
    hook: ${project_root}/yasar_hooks.py
    targets:
      - {id: orch, command: ["${project_root}/.venv/Scripts/python.exe", "run.py"]}
""", encoding="utf-8")
    _, projects = load_registry(reg, project_root="C:/kutay")
    p = projects[0]
    assert p.venv_python.replace("\\", "/").endswith("kutay/.venv/Scripts/python.exe")
    assert p.hook_path.replace("\\", "/").endswith("kutay/yasar_hooks.py")
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_hookfields.py -v`
Expected: FAIL (`ProjectConfig` has no `venv_python`).

- [ ] **Step 3: Add fields**

In `config.py`, add to `ProjectConfig` (keep existing fields):
```python
    venv_python: str | None = None
    hook_path: str | None = None
    state_dir: str | None = None   # Phase 2; harmless now
```
In `registry.py` `load_registry`, resolve + pass them (normalize as paths):
```python
        projects.append(ProjectConfig(
            id=pid,
            name=raw_proj.get("name", pid),
            targets=targets,
            hook_module=raw_proj.get("hook_module"),
            venv_python=_norm(_resolve(raw_proj.get("venv_python"), tokens)),
            hook_path=_norm(_resolve(raw_proj.get("hook"), tokens)),
        ))
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_hookfields.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/config.py src/yasar_usta/registry.py tests/test_registry_hookfields.py
git commit -m "feat(config): ProjectConfig.venv_python + hook_path"
```

---

### Task 1.6: Messages block parsed from the registry

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/registry.py`, `src/yasar_usta/config.py` · Test: `tests/test_registry_messages.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry_messages.py`:
```python
from yasar_usta.registry import load_registry


def test_messages_block_maps_to_project(tmp_path):
    reg = tmp_path / "r.yaml"
    reg.write_text("""
projects:
  kutai:
    root: C:/kutay
    messages:
      announce: "Ben Yaşar Usta"
      btn_status: "Durum"
    targets:
      - {id: orch, command: ["run.py"]}
""", encoding="utf-8")
    _, projects = load_registry(reg, project_root="C:/kutay")
    m = projects[0].messages
    assert m is not None
    assert m.announce == "Ben Yaşar Usta"
    assert m.btn_status == "Durum"
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_messages.py -v`
Expected: FAIL (`ProjectConfig.messages` absent).

- [ ] **Step 3: Implement**

In `config.py`, add to `ProjectConfig`:
```python
    messages: "Messages | None" = None
```
In `registry.py`, import `Messages` (already imported alongside configs) and in `load_registry`, before appending:
```python
        raw_msgs = raw_proj.get("messages")
        msgs = None
        if raw_msgs:
            valid = {f.name for f in dataclasses.fields(Messages)}
            msgs = Messages(**{k: v for k, v in raw_msgs.items() if k in valid})
```
Add `messages=msgs` to the `ProjectConfig(...)` call. Import `Messages` in the `from .config import ...` line.

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_messages.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/config.py src/yasar_usta/registry.py tests/test_registry_messages.py
git commit -m "feat(registry): parse per-project messages block into Messages"
```

---

### Task 1.7: Subprocess hook dispatch (shared side)

**Repo/CWD:** `HUB`
**Files:** Rewrite: `src/yasar_usta/hooks.py` · Test: `tests/test_hooks_subprocess.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hooks_subprocess.py`:
```python
import json
from yasar_usta.hooks import build_hook_command, run_hook_subprocess
from yasar_usta.config import ProjectConfig, GuardConfig


def _project():
    return ProjectConfig(
        id="kutai", name="Kutay",
        venv_python="C:/kutay/.venv/Scripts/python.exe",
        hook_path="C:/kutay/yasar_hooks.py",
        targets=[GuardConfig(name="orch", command=["C:/kutay/.venv/Scripts/python.exe", "C:/kutay/src/app/run.py"])],
    )


def test_build_hook_command_is_argv_list_with_context():
    cmd = build_hook_command(_project(), "pre_boot", extra={"exit_code": None})
    assert isinstance(cmd, list)
    assert cmd[0].endswith("python.exe")
    assert cmd[1].endswith("yasar_hooks.py")
    assert cmd[2] == "pre_boot"
    assert cmd[3] == "--context"
    ctx = json.loads(cmd[4])
    # script paths from the target command's .py args, for M6 stale-kill
    assert any(p.endswith("run.py") for p in ctx["script_paths"])


def test_run_hook_subprocess_invokes_and_returns_rc(monkeypatch):
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        class R: returncode = 0
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    rc = run_hook_subprocess(_project(), "on_exit", extra={"exit_code": 7})
    assert rc == 0
    assert seen["cmd"][2] == "on_exit"
    assert json.loads(seen["cmd"][4])["exit_code"] == 7


def test_no_hook_path_is_noop():
    p = ProjectConfig(id="x", name="x", targets=[GuardConfig(name="o", command=["run.py"])])
    assert run_hook_subprocess(p, "pre_boot", extra={}) is None
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hooks_subprocess.py -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Rewrite hooks.py**

Replace `src/yasar_usta/hooks.py` with:
```python
"""Per-project lifecycle hooks, dispatched as a SUBPROCESS in the project's own
venv so the hub never imports project packages.

Contract: the project ships a ``yasar_hooks.py`` runnable by its venv python:
    <project_venv_python> yasar_hooks.py <phase> --context <json>
where <phase> is 'pre_boot' or 'on_exit'. The JSON context carries what the
old in-process hook read from the project object.
"""
from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger("yasar_usta.hooks")


def _script_paths(project) -> list:
    paths = []
    for tgt in getattr(project, "targets", []) or []:
        for arg in getattr(tgt, "command", []) or []:
            if isinstance(arg, str) and arg.lower().endswith(".py"):
                paths.append(arg)
    return paths


def build_hook_command(project, phase: str, extra: dict) -> list | None:
    """Argv list (never a shell string — Windows backslash JSON) or None if the
    project declares no hook."""
    venv_python = getattr(project, "venv_python", None)
    hook_path = getattr(project, "hook_path", None)
    if not (venv_python and hook_path):
        return None
    context = {"project_id": project.id, "script_paths": _script_paths(project)}
    context.update(extra or {})
    return [venv_python, hook_path, phase, "--context", json.dumps(context)]


def run_hook_subprocess(project, phase: str, extra: dict) -> int | None:
    """Spawn the project's hook. Returns rc, or None if no hook declared.
    pre_boot surfaces failure (raises); on_exit is fail-soft."""
    cmd = build_hook_command(project, phase, extra)
    if cmd is None:
        return None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        logger.error("hook %s for %s failed to spawn: %s", phase, project.id, e)
        if phase == "pre_boot":
            raise
        return None
    if result.stdout:
        logger.info("[hook %s/%s] %s", project.id, phase, result.stdout.strip())
    if result.returncode != 0:
        logger.error("[hook %s/%s] rc=%s stderr=%s", project.id, phase,
                     result.returncode, (result.stderr or "").strip())
        if phase == "pre_boot":
            raise RuntimeError(f"pre_boot hook for {project.id} failed rc={result.returncode}")
    return result.returncode


def run_pre_boot(project) -> None:
    """Back-compat name used by hub.run(). Subprocess dispatch."""
    run_hook_subprocess(project, "pre_boot", extra={})
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hooks_subprocess.py -v`
Expected: 3 passed.

- [ ] **Step 5: Remediate the OLD hooks test (it imports the deleted API)**

The moved `tests/test_hooks.py` imports `load_hook` and calls `run_pre_boot(hook, proj)` (2-arg) + loads `yasar_usta.projects.kutai.hooks` — all deleted now. It will poison the whole `pytest` run (Task 1.15) with import/arity errors. **Delete it** (its behavior is replaced by `test_hooks_subprocess.py`):
```bash
rm tests/test_hooks.py
```
Confirm nothing else imports `load_hook`: `grep -rn "load_hook\|projects.kutai" tests/ src/` → expect no hits.

- [ ] **Step 6: Commit**

```bash
git add src/yasar_usta/hooks.py tests/test_hooks_subprocess.py
git rm tests/test_hooks.py
git commit -m "feat(hooks): subprocess dispatch in project venv (replaces in-process import + old test)"
```

---

### Task 1.8: Wire hub to subprocess hooks (pre_boot AND on_exit) + delete `projects/` subpackage

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/hub.py` · Delete: `src/yasar_usta/projects/` · Test: `tests/test_hub_hook_wiring.py`

> **CRITICAL — do not lose `on_exit`.** The real `__init__` loop (`hub.py:49-60`) does THREE things at once: (a) `load_hook`, (b) `tgt.on_exit = hook.on_exit`, (c) builds `self.supervisors[rid] = TargetSupervisor(...)`. `on_exit` fires at `supervisor.py:331-332` (`if self.cfg.on_exit: self.cfg.on_exit(exit_code)`) and is what runs KutAI's llama-server/Ollama orphan-kill on a non-clean exit. If you delete the loop you MUST keep (c) and re-wire (b) to the subprocess hook — otherwise GPU processes leak on every crash. Only (a) `load_hook` goes away.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hub_hook_wiring.py`:
```python
from yasar_usta.hub import Hub
from yasar_usta.config import HubConfig, ProjectConfig, GuardConfig


def _hub_with_project():
    proj = ProjectConfig(id="kutai", name="K", venv_python="py", hook_path="h.py",
                         targets=[GuardConfig(name="o", command=["run.py"])])
    return Hub(HubConfig(name="T"), [proj]), proj


def test_pre_boot_dispatches_subprocess(monkeypatch):
    called = []
    monkeypatch.setattr("yasar_usta.hub.run_pre_boot", lambda project: called.append(project.id))
    hub, _ = _hub_with_project()
    from yasar_usta.hub import run_pre_boot
    for p in hub.projects:
        run_pre_boot(p)
    assert called == ["kutai"]


def test_on_exit_is_wired_to_subprocess(monkeypatch):
    seen = {}
    monkeypatch.setattr("yasar_usta.hub.run_hook_subprocess",
                        lambda project, phase, extra: seen.update(id=project.id, phase=phase, extra=extra))
    hub, proj = _hub_with_project()
    tgt = proj.targets[0]
    assert tgt.on_exit is not None            # must be wired, not None
    tgt.on_exit(42)                           # the supervisor calls this on child exit
    assert seen == {"id": "kutai", "phase": "on_exit", "extra": {"exit_code": 42}}


def test_supervisors_still_built():
    hub, _ = _hub_with_project()
    assert "kutai" in hub.supervisors


def test_hub_init_has_no_load_hook():
    import yasar_usta.hub as hubmod
    assert "load_hook" not in dir(hubmod)
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hub_hook_wiring.py -v`
Expected: FAIL (`run_hook_subprocess` not imported in hub; `on_exit` still set from `hook.on_exit`).

- [ ] **Step 3: Edit hub.py**

- Import at `hub.py:17`: change `from .hooks import load_hook, run_pre_boot` → `from .hooks import run_pre_boot, run_hook_subprocess`.
- Rewrite the `__init__` loop (`:48-60`) — drop `load_hook`/`self._hooks`, keep supervisor building, re-wire `on_exit` to the subprocess dispatch:
```python
        for proj in projects:
            for tgt in proj.targets:
                tgt.on_exit = self._make_on_exit(proj)
                single = len(proj.targets) == 1
                rid = proj.id if single else f"{proj.id}:{tgt.name}"
                display = proj.name if single else f"{proj.name} · {tgt.app_name}"
                self.supervisors[rid] = TargetSupervisor(
                    rid, tgt, notify=self._notify, reply_keyboard=self._reply_kb,
                    display_name=display)
```
- Add the factory (closure binds `proj` by default-arg so it isn't late-bound):
```python
    @staticmethod
    def _make_on_exit(proj):
        def _on_exit(exit_code, _proj=proj):
            run_hook_subprocess(_proj, "on_exit", {"exit_code": exit_code})
        return _on_exit
```
  (`run_hook_subprocess` is blocking `subprocess.run`; `supervisor.py:331` already calls `on_exit` synchronously — matches the old blocking `_kill_orphan_processes` contract, no event-loop regression.)
- Delete the `self._hooks: dict = {}` initializer line above the loop.
- At the pre_boot call site (`~:388`) change `run_pre_boot(self._hooks.get(proj.id), proj)` → `run_pre_boot(proj)`.

- [ ] **Step 4: Delete the projects subpackage**

Run: `rm -rf /c/Users/sakir/Dropbox/Workspaces/yasar_usta/src/yasar_usta/projects`

- [ ] **Step 5: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hub_hook_wiring.py tests/test_hub.py -v`
Expected: pass. If `test_hub.py` monkeypatches `load_hook` or asserts on `self._hooks`, update those stragglers (grep `_hooks`/`load_hook` in `tests/test_hub.py`).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(hub): subprocess pre_boot + on_exit; keep supervisors; drop projects/kutai"
```

---

### Task 1.9: Generic entry `__main__.py` (load_dotenv, argparse, signals) + boot-asserts

**Repo/CWD:** `HUB`
**Files:** Create: `src/yasar_usta/__main__.py` · Modify: `src/yasar_usta/hub.py` (boot-asserts in `run()`) · Test: `tests/test_entry_and_bootasserts.py`

- [ ] **Step 1: Write the failing test (boot-asserts)**

Create `tests/test_entry_and_bootasserts.py`:
```python
import pytest
from yasar_usta.hub import assert_consumer_imports, assert_hub_credentials
from yasar_usta.config import HubConfig, ProjectConfig, GuardConfig


def _proj(venv):
    return ProjectConfig(id="kutai", name="K", venv_python=venv, hook_path="h.py",
                         targets=[GuardConfig(name="o", command=["run.py"])])


def test_consumer_import_assert_fails_loud(monkeypatch):
    def fake_run(cmd, **kw):
        class R: returncode = 1; stderr = "ModuleNotFoundError: yasar_usta"
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(SystemExit) as e:
        assert_consumer_imports([_proj("C:/kutay/.venv/Scripts/python.exe")])
    assert "pip install -e ../yasar_usta" in str(e.value)


def test_consumer_import_assert_passes(monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: type("R", (), {"returncode": 0, "stderr": ""})())
    assert_consumer_imports([_proj("py")])  # no raise


def test_credentials_assert_fails_on_empty_token():
    with pytest.raises(SystemExit):
        assert_hub_credentials(HubConfig(name="T", telegram_token=""))


def test_credentials_assert_passes():
    assert_hub_credentials(HubConfig(name="T", telegram_token="abc"))
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_entry_and_bootasserts.py -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Add the asserts to hub.py**

Add module-level functions in `hub.py`:
```python
import subprocess as _subprocess


def assert_consumer_imports(projects) -> None:
    """Each project's venv must resolve the heartbeat client symbols before we
    manage its child. Fail loud (SystemExit) rather than a late ImportError."""
    for p in projects:
        vp = getattr(p, "venv_python", None)
        if not vp:
            continue
        try:
            r = _subprocess.run(
                [vp, "-c", "from yasar_usta import HeartbeatWriter, write_heartbeat"],
                capture_output=True, text=True, timeout=30)
        except Exception as e:
            raise SystemExit(f"[Yasar Usta] cannot probe {p.id} venv {vp}: {e}")
        if r.returncode != 0:
            raise SystemExit(
                f"[Yasar Usta] project {p.id}: 'yasar_usta' not importable in {vp}. "
                f"Run: pip install -e ../yasar_usta in that venv. ({(r.stderr or '').strip()})")


def assert_hub_credentials(cfg) -> None:
    if not getattr(cfg, "telegram_token", ""):
        raise SystemExit(
            "[Yasar Usta] YASAR_USTA_BOT_TOKEN is empty — refusing to boot a "
            "credential-less hub (alerts would be silently dark). Check the hub .env.")
```
Then in `Hub.run()`, immediately **after** `self._acquire_singleton()` (line ~382) and before `acquire_lock`, insert:
```python
        assert_hub_credentials(self.cfg)
        assert_consumer_imports(self.projects)
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_entry_and_bootasserts.py -v`
Expected: 4 passed.

- [ ] **Step 5: Create `__main__.py`**

Create `src/yasar_usta/__main__.py` (port from the old `kutai_wrapper.py`, generic):
```python
"""Generic hub entry point: python -m yasar_usta --registry <path>."""
import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import Hub, load_registry


def _parse(argv):
    ap = argparse.ArgumentParser(prog="yasar-usta")
    ap.add_argument("--registry", required=True, help="path to registry.yaml")
    ap.add_argument("--no-auto-restart", action="store_true")
    return ap.parse_args(argv)


async def _amain(args) -> None:
    reg_path = Path(args.registry).resolve()
    hub_cfg, projects = load_registry(reg_path, project_root=str(reg_path.parent))
    if args.no_auto_restart:
        for proj in projects:
            for tgt in proj.targets:
                tgt.auto_restart = False
    # Per-project Turkish/localized messages from the registry override hub + targets.
    for proj in projects:
        if getattr(proj, "messages", None) is not None:
            hub_cfg.messages = proj.messages
            for tgt in proj.targets:
                tgt.messages = proj.messages
    hub = Hub(hub_cfg, projects)

    def _sig(sig, frame):
        print(f"\n[Yasar Usta] Signal {sig} — shutting down")
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


def main(argv=None) -> None:
    load_dotenv()  # hub-owned .env next to CWD / registry
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("[Yasar Usta] KeyboardInterrupt — exiting")


def main_cli() -> None:  # console_scripts entry
    main()


if __name__ == "__main__":
    main()
```
Note: `load_dotenv()` searches from CWD upward; `start.bat` (Task 1.11) sets CWD to the hub repo so its `.env` loads. For robustness the registry parent is also a candidate — acceptable since start.bat cd's to the hub dir.

- [ ] **Step 6: Commit**

```bash
git add src/yasar_usta/__main__.py src/yasar_usta/hub.py tests/test_entry_and_bootasserts.py
git commit -m "feat(entry): generic __main__ with load_dotenv + post-mutex boot asserts"
```

---

### Task 1.10: `_do_restart_hub` self-fork uses `-m yasar_usta`

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/hub.py` (`_do_restart_hub`, ~:240-260) · Test: `tests/test_hub_restart_cmd.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hub_restart_cmd.py`:
```python
from yasar_usta.hub import build_restart_command


def test_restart_command_uses_dash_m(monkeypatch):
    monkeypatch.setattr("sys.executable", "C:/hub/.venv/Scripts/python.exe")
    cmd = build_restart_command(["--registry", "C:/hub/registry.yaml"])
    assert cmd[0].endswith("python.exe")
    assert cmd[1:3] == ["-m", "yasar_usta"]
    assert "--registry" in cmd
    # must NOT re-run a bare __main__.py path
    assert not any(a.endswith("__main__.py") for a in cmd)
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hub_restart_cmd.py -v`
Expected: FAIL (`build_restart_command` not defined).

- [ ] **Step 3: Implement + wire**

Add to `hub.py`:
```python
def build_restart_command(registry_args: list) -> list:
    return [sys.executable, "-m", "yasar_usta"] + list(registry_args)
```
The real block (`hub.py:248` + `:259-260`) is:
```python
        script = str(Path(sys.argv[0]).resolve())
        ...
        _sp.Popen([sys.executable, script] + sys.argv[1:], close_fds=True,
                  cwd=str(Path(script).parent), **kwargs)
        os._exit(0)
```
Change ONLY the command and the `cwd` — **keep `**kwargs`** (it carries the detach/no-window creationflags) and keep `os._exit(0)` and the `release_singleton()`/`release_lock()` above it:
```python
        # Restart via `-m yasar_usta` (package imports resolve); pin CWD to the
        # hub repo root so __main__.load_dotenv() finds the hub .env on restart.
        _hub_root = str(Path(__file__).resolve().parents[2])  # src/yasar_usta/hub.py -> repo root
        _sp.Popen(build_restart_command(sys.argv[1:]), close_fds=True,
                  cwd=_hub_root, **kwargs)
```
Delete the now-unused `script = ...` line only if nothing else in the function uses it (grep within the function first). **Do NOT drop `**kwargs`** — dropping it removes DETACHED_PROCESS/CREATE_NO_WINDOW and the restarted hub gets a stray console / wrong process group.

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hub_restart_cmd.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/hub.py tests/test_hub_restart_cmd.py
git commit -m "fix(hub): self-restart via -m yasar_usta (package imports resolve)"
```

---

### Task 1.11: Watchdog matcher off `kutai_wrapper.py`

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/watchdog.py` (`find_hub_pids`, `:46-58`) · Test: `tests/test_watchdog_matcher.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_watchdog_matcher.py`:
```python
from yasar_usta.watchdog import cmdline_is_hub


def test_matches_dash_m_invocation():
    assert cmdline_is_hub(["C:/hub/.venv/Scripts/python.exe", "-m", "yasar_usta",
                           "--registry", "C:/hub/registry.yaml"])


def test_matches_real_interpreter_child():
    assert cmdline_is_hub(["C:/Python310/python.exe", "-m", "yasar_usta"])


def test_does_not_match_pip_editable_line():
    assert not cmdline_is_hub(["pip", "install", "-e", "../yasar_usta"])


def test_does_not_match_loose_substring():
    assert not cmdline_is_hub(["python", "tools/format_yasar_usta_docs.py"])


def test_does_not_match_watchdog_itself():
    assert not cmdline_is_hub(["python", "-m", "yasar_usta.watchdog", "--alive", "x"])
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_watchdog_matcher.py -v`
Expected: FAIL (`cmdline_is_hub` undefined).

- [ ] **Step 3: Implement**

In `watchdog.py`, add and use a token-pair matcher:
```python
def cmdline_is_hub(argv: list) -> bool:
    """True iff argv is a `-m yasar_usta` hub launch (adjacent tokens), NOT the
    watchdog submodule and NOT a loose substring."""
    for i in range(len(argv) - 1):
        if argv[i] == "-m" and argv[i + 1] == "yasar_usta":
            return True
    return False
```
Rewrite `find_hub_pids` to use it (and fix the docstring):
```python
def find_hub_pids() -> list:
    """Live hub processes (python -m yasar_usta). psutil; skips inaccessible."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (p.info.get("name") or "").lower():
                continue
            if cmdline_is_hub(p.info.get("cmdline") or []):
                out.append(p.info["pid"])
        except Exception:
            continue
    return out
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_watchdog_matcher.py tests/test_watchdog.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/watchdog.py tests/test_watchdog_matcher.py
git commit -m "fix(watchdog): match -m yasar_usta token pair, not kutai_wrapper.py string"
```

---

### Task 1.12: KutAI-side `yasar_hooks.py` (CLI, runs in kutay venv, keeps M6)

**Repo/CWD:** `KUTAY`
**Files:** Create: `yasar_hooks.py` · Test: `tests/yasar/test_yasar_hooks_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/yasar/test_yasar_hooks_cli.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "yasar_hooks.py"


def test_on_exit_clean_restart_is_noop_rc0():
    ctx = json.dumps({"project_id": "kutai", "script_paths": [], "exit_code": 42})
    r = subprocess.run([sys.executable, str(HOOK), "on_exit", "--context", ctx],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0


def test_pre_boot_parses_context_and_runs(monkeypatch, tmp_path):
    # No stale orchestrators, no LLAMA_SERVER_PORT → clean rc0
    ctx = json.dumps({"project_id": "kutai", "script_paths": ["C:/kutay/src/app/run.py"]})
    env = {k: v for k, v in __import__("os").environ.items() if k != "LLAMA_SERVER_PORT"}
    r = subprocess.run([sys.executable, str(HOOK), "pre_boot", "--context", ctx],
                       capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode == 0
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/yasar/test_yasar_hooks_cli.py -v`
Expected: FAIL (`yasar_hooks.py` missing).

- [ ] **Step 3: Create `yasar_hooks.py`**

Port `projects/kutai/hooks.py` logic into a root-level CLI. Preserve M6 (`_kill_stale_orchestrators` absolute-path psutil match), `_reconcile_stray_llama`, `_kill_orphan_processes`:
```python
#!/usr/bin/env python3
"""KutAI lifecycle hooks, invoked by the Yaşar Usta hub as a subprocess in
kutay's OWN venv: python yasar_hooks.py <pre_boot|on_exit> --context <json>.
Free to import dallama/psutil (kutay venv)."""
import argparse
import json
import os
import subprocess as _sp
import sys


def _norm(s: str) -> str:
    return s.replace("\\", "/").lower()


def _iter_python_processes():
    import psutil
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (p.info.get("name") or "").lower():
                continue
            yield (p.info["pid"], " ".join(p.info.get("cmdline") or []))
        except Exception:
            continue


def _kill_pid(pid) -> None:
    try:
        import psutil
        psutil.Process(pid).kill()
    except Exception as e:
        print(f"[yasar_hooks] kill {pid} failed: {e}")


def _kill_stale_orchestrators(script_paths) -> None:
    paths = {_norm(p) for p in script_paths if p.lower().endswith(".py")}
    if not paths:
        return
    my_pid = os.getpid()
    for pid, cmdline in _iter_python_processes():
        if pid == my_pid:
            continue
        if any(sp in _norm(cmdline) for sp in paths):
            print(f"[yasar_hooks] killing stale orchestrator PID {pid}")
            _kill_pid(pid)


def _reconcile_stray_llama() -> None:
    raw = os.environ.get("LLAMA_SERVER_PORT")
    if raw is None:
        print("[yasar_hooks] LLAMA_SERVER_PORT unset — skipping stray-llama reconcile")
        return
    try:
        port = int(raw)
    except ValueError:
        print(f"[yasar_hooks] LLAMA_SERVER_PORT={raw!r} invalid — skipping")
        return
    try:
        from dallama.platform import PlatformHelper
        n = PlatformHelper().kill_stray_servers(port)
        if n:
            print(f"[yasar_hooks] reconciled {n} stray llama-server(s) not on {port}")
    except Exception as e:
        print(f"[yasar_hooks] stray-llama reconcile error: {e}")


def _kill_orphan_processes(exit_code: int) -> None:
    if exit_code == 42:
        return
    for exe, label in (("llama-server.exe", "llama-server"),
                       ("ollama.exe", "Ollama"),
                       ("ollama_llama_server.exe", "Ollama runner")):
        try:
            check = _sp.run(["tasklist", "/FI", f"IMAGENAME eq {exe}", "/NH"],
                            capture_output=True, text=True, timeout=5)
            if exe.lower() not in check.stdout.lower():
                continue
            r = _sp.run(["taskkill", "/F", "/IM", exe], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                print(f"[yasar_hooks] killed orphaned {label}")
        except Exception as e:
            print(f"[yasar_hooks] {label} cleanup error: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pre_boot", "on_exit"])
    ap.add_argument("--context", default="{}")
    args = ap.parse_args(argv)
    ctx = json.loads(args.context)
    if args.phase == "pre_boot":
        _kill_stale_orchestrators(ctx.get("script_paths", []))
        _reconcile_stray_llama()
    elif args.phase == "on_exit":
        _kill_orphan_processes(int(ctx.get("exit_code") or 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/yasar/test_yasar_hooks_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit (KUTAY)**

```bash
cd /c/Users/sakir/Dropbox/Workspaces/kutay
git add yasar_hooks.py tests/yasar/test_yasar_hooks_cli.py
git commit -m "feat: KutAI yasar_hooks.py CLI (subprocess hook, keeps M6 orphan-kill)"
```

---

### Task 1.13: Hub-side `registry.yaml` (extended) + `start.bat` + `.env`

**Repo/CWD:** `HUB`
**Files:** Create: `registry.yaml`, `start.bat`, `.env`

- [ ] **Step 1: Write `registry.yaml`** (kutai block, Phase-1 state still in project logs)

Create `HUB/registry.yaml`:
```yaml
hub:
  name: "Yaşar Usta"
  telegram_token_env: YASAR_USTA_BOT_TOKEN
  telegram_chat_id_env: TELEGRAM_ADMIN_CHAT_ID
  log_dir: "${env:LOCALAPPDATA}/YasarUsta/hub"   # hub state out of Dropbox from day 1

projects:
  kutai:
    name: Kutay
    root: "C:/Users/sakir/Dropbox/Workspaces/kutay"
    venv_python: "${project_root}/.venv/Scripts/python.exe"
    hook: "${project_root}/yasar_hooks.py"
    messages:
      announce: "🔧 *Bennn... Yaşar Usta!*\n\nKutay'ı başlatıyorum..."
      started: "✅ *Kutay Started*"
      stopped: "⏹ *Kutay Stopped*\nSend /start to restart."
      hung: "🔴 Kutay dondu — Yaşar Usta {delay}sn içinde yeniden başlatıyor"
      restarting: "♻️ *Kutay yeniden başlatılıyor...*"
      self_restarting: "🔄 *Yaşar Usta yeniden başlatılıyor...*"
      down_prompt: "⚠️ Kutay durdu. Başlatmak için butona bas."
      down_reply: "⏸ Kutay şu an kapalı."
      starting: "🚀 Kutay başlatılıyor..."
      btn_status: "🔧 Durum"
      btn_logs: "📋 Loglar"
      btn_remote: "🖥️ Claude Code"
      remote_starting: "🖥️ Claude Code oturumu başlatılıyor..."
      remote_not_found: "❌ `claude` command not found. Claude Code kurulu mu?"
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
        claude_cmd: "${env:APPDATA}/npm/claude.cmd"
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
            command: ["${project_root}/.venv/Scripts/python.exe", "-m", "nerd_herd", "--port", "9881", "--pid-file", "${project_root}/logs/nerd_herd.pid", "--db-path", "${project_root}/data/kutai.db"]
            health_url: "http://127.0.0.1:9881/health"
            pid_file: "${project_root}/logs/nerd_herd.pid"
            detached: true
```
Note: `db_path` is now a **path token** (`${project_root}/data/kutai.db`), not the hub's `${env:DB_PATH}` — the hub must not depend on a project env var (spec §4.6). If kutay's `.env` overrides DB_PATH elsewhere, keep that override inside run.py; the sidecar uses the canonical file path.

- [ ] **Step 2: Write `start.bat`**

Create `HUB/start.bat`:
```bat
@echo off
cd /d "C:\Users\sakir\Dropbox\Workspaces\yasar_usta"
.venv\Scripts\python.exe -m yasar_usta --registry "C:\Users\sakir\Dropbox\Workspaces\yasar_usta\registry.yaml"
```

- [ ] **Step 3: Create `.env`** (copy the two hub tokens from kutay `.env`)

Create `HUB/.env` with the real values from `KUTAY/.env`:
```
YASAR_USTA_BOT_TOKEN=<copy from kutay .env>
TELEGRAM_ADMIN_CHAT_ID=<copy from kutay .env>
```
(This file is gitignored; fill actual values from `KUTAY/.env`.)

- [ ] **Step 4: Static validation (no live launch)**

Run:
```bash
.venv/Scripts/python.exe -c "from yasar_usta import load_registry; h,p=load_registry('registry.yaml', project_root='C:/Users/sakir/Dropbox/Workspaces/yasar_usta'); print('projects:', [x.id for x in p]); print('hook:', p[0].hook_path); print('venv:', p[0].venv_python)"
```
Expected: `projects: ['kutai']`, hook path ends `kutay/yasar_hooks.py`, venv ends `kutay/.venv/Scripts/python.exe`. (LOCALAPPDATA must be set in env — it always is on Windows.)

- [ ] **Step 5: Commit (registry + start.bat only; .env is ignored)**

```bash
git add registry.yaml start.bat && git commit -m "feat: hub-side registry.yaml (kutai block) + start.bat launcher"
```

---

### Task 1.14: Repoint KUTAY consumers; delete wrapper + package; move tests

**Repo/CWD:** `KUTAY`
**Files:** Modify: `requirements.txt`, `start_kutai.bat`, `CLAUDE.md`, `.claude/settings.local.json`; Delete: `kutai_wrapper.py`, `packages/yasar_usta/`; Move: 2 test files

- [ ] **Step 1: Repoint editable install in requirements.txt**

Edit `requirements.txt` line 102: `-e ./packages/yasar_usta` → `-e ../yasar_usta`. Then reinstall into kutay venv:
```bash
cd /c/Users/sakir/Dropbox/Workspaces/kutay
.venv/Scripts/python.exe -m pip install -e ../yasar_usta
.venv/Scripts/python.exe -c "from yasar_usta import HeartbeatWriter, write_heartbeat; print('kutay client import OK')"
```
Expected: `kutay client import OK`.

- [ ] **Step 2: Repoint `start_kutai.bat`**

Replace its body with a call to the hub launcher:
```bat
@echo off
cd /d "C:\Users\sakir\Dropbox\Workspaces\yasar_usta"
.venv\Scripts\python.exe -m yasar_usta --registry "C:\Users\sakir\Dropbox\Workspaces\yasar_usta\registry.yaml"
```

- [ ] **Step 3: Retire the two kutai-specific PACKAGE tests (they assert the deleted in-process API)**

`test_kutai_hooks.py:11` imports `yasar_usta.projects.kutai.hooks`; `test_migration_kutai.py:26` asserts `hook_module == "yasar_usta.projects.kutai.hooks"`. Both test the OLD in-process contract that no longer exists (hooks are now a subprocess CLI, `hook:` is a path not a module). They are **replaced** by `tests/yasar/test_yasar_hooks_cli.py` (Task 1.12), which tests the real new behavior. Delete them (they go away with the package in Step 5) — do **not** try to repoint them, the API changed shape:
```bash
# nothing to move — Task 1.12 already created the replacement CLI test.
# these die with packages/yasar_usta in Step 5; note the replacement in the commit.
grep -rn "projects.kutai\|projects/kutai" tests/ src/ packages/ 2>/dev/null   # expect: none outside the doomed package dir
```

- [ ] **Step 4: Rewrite wrapper-referencing tests**

- `tests/integration/test_restart_shutdown.py:96-109`: it asserts `kutai_wrapper.py` contains exit-code 42. Repoint to assert `../yasar_usta/src/yasar_usta/heartbeat.py` defines `EXIT_RESTART = 42` (the canonical constant), or delete if redundant.
- `tests/test_wrapper_logs.py:8,14`: opens `kutai_wrapper.py`. Repoint to the new entry or delete if it only tested wrapper log lines that no longer exist.

- [ ] **Step 5: Delete the wrapper + the in-repo package + the KUTAY installer copy (now carried to HUB)**

```bash
rm kutai_wrapper.py
rm -rf packages/yasar_usta
rm scripts/install_yasar_autostart.ps1   # relocated to HUB/scripts in Task 1.1 Step 4
```

- [ ] **Step 6: Refresh docs + perms**

- `CLAUDE.md`: update the entry-point line(s) (`kutai_wrapper.py` → `python -m yasar_usta` from the sibling repo; note the relocation + own venv). Update the Key Files table row.
- `.claude/settings.local.json`: refresh the stale `kutai_wrapper.py` permission entries (harmless but confusing) — replace with the new launch command form. (Permission list only; no behavior.)

- [ ] **Step 7: Verify kutay still imports its stack**

Run: `.venv/Scripts/python.exe -c "import src.app.run" 2>&1 | tail -5` (or a lighter import that exercises `from yasar_usta import HeartbeatWriter`). Expected: no ImportError on `yasar_usta`.

- [ ] **Step 8: Commit (KUTAY)**

```bash
git add -A
git commit -m "refactor: relocate Yaşar Usta to sibling repo; delete wrapper+package; repoint launchers/tests/docs"
```

---

### Task 1.15: PHASE 1 LIVE-VERIFY (regression) + full suites

**Repo/CWD:** both

- [ ] **Step 1: Full hub test suite**

Run (HUB): `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass (the moved suite + new tests). Fix stragglers (imports of deleted `projects`, old `load_hook`).

- [ ] **Step 2: KUTAY targeted suites**

Run (KUTAY): `.venv/Scripts/python.exe -m pytest tests/yasar/ -q --timeout=60`
Expected: pass.

- [ ] **Step 3: USER live-verify (hand to Sakir — do not launch the hub yourself while the prod hub runs)**

Checklist for the user:
1. Stop the current (old-code) hub via Telegram.
2. Launch new hub: double-click `HUB/start.bat` (or `KUTAY/start_kutai.bat`).
3. Confirm: singleton probe `Global\YasarUstaHub` → `err==183`; a 2nd `start.bat` exits immediately.
4. Telegram dashboard shows Kutay; orchestrator + yazbunu + nerd_herd sidecars come up; Turkish messages render.
5. `/restart` works (self-fork relaunch via `-m yasar_usta`).
6. Kill the hub process (not llama-server); confirm it can be relaunched and finds no dupes.

- [ ] **Step 4: Commit any fixups, then push**

```bash
# in each repo
git add -A && git commit -m "test: phase-1 regression green" || true
```
(Push decision: `main` per user. Push after live-verify passes.)

---

# PHASE 2 — State-dir / M5 + split-brain heartbeat fix

**Phase goal:** All Yaşar-owned runtime state under `%LOCALAPPDATA%\YasarUsta\`; the orchestrator writes its heartbeat to a hub-supplied absolute path (no CWD-relative split-brain). Dedicated live-verify: hub does **not** false-kill a healthy orchestrator.

---

### Task 2.1: `${state_dir}` token + per-project state dir

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/registry.py` · Test: `tests/test_registry_statedir.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry_statedir.py`:
```python
from yasar_usta.registry import load_registry


def test_state_dir_token_resolves_per_project(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\x\AppData\Local")
    reg = tmp_path / "r.yaml"
    reg.write_text("""
projects:
  kutai:
    root: C:/kutay
    targets:
      - id: orch
        command: ["run.py"]
        heartbeat_file: "${state_dir}/orchestrator.heartbeat"
""", encoding="utf-8")
    _, projects = load_registry(reg, project_root="C:/kutay")
    hbf = projects[0].targets[0].heartbeat_file.replace("\\", "/")
    assert hbf.endswith("YasarUsta/kutai/orchestrator.heartbeat")
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_statedir.py -v`
Expected: FAIL (`${state_dir}` not substituted).

- [ ] **Step 3: Implement**

In `load_registry`, derive each project's `state_dir` and add it to that project's tokens:
```python
        proj_root = raw_proj.get("root", project_root)
        localappdata = os.environ.get("LOCALAPPDATA", "")
        default_state = f"{localappdata}/YasarUsta/{pid}" if localappdata else f"{proj_root}/logs"
        state_dir = raw_proj.get("state_dir", default_state)
        tokens = {"project_root": proj_root, "state_dir": state_dir}
```
Also set `state_dir=_norm(state_dir)` on the `ProjectConfig(...)`.

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_statedir.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/registry.py tests/test_registry_statedir.py
git commit -m "feat(registry): \${state_dir} per-project token (LOCALAPPDATA)"
```

---

### Task 2.2: Hub passes `YASAR_USTA_STATE_DIR` to each managed child

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/supervisor.py` (child spawn env) · Test: `tests/test_child_env_statedir.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_child_env_statedir.py`:
```python
from yasar_usta.supervisor import build_child_env
from yasar_usta.config import GuardConfig


def test_child_env_includes_state_dir():
    tgt = GuardConfig(name="orch", command=["run.py"], env={"FOO": "bar"})
    env = build_child_env(tgt, state_dir="C:/state/kutai", base_env={"PATH": "x"})
    assert env["YASAR_USTA_STATE_DIR"] == "C:/state/kutai"
    assert env["FOO"] == "bar"
    assert env["PATH"] == "x"
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_child_env_statedir.py -v`
Expected: FAIL (`build_child_env` undefined or lacks state_dir).

- [ ] **Step 3: Implement**

In `supervisor.py`, locate where the child env is built for `Popen` (currently merges `os.environ` + `tgt.env`). Extract/extend to a `build_child_env(tgt, state_dir, base_env=None)` helper that injects `YASAR_USTA_STATE_DIR=state_dir`, and pass the project's `state_dir` down from the hub when constructing the supervisor. If `TargetSupervisor` doesn't yet know its `state_dir`, thread it from `ProjectConfig.state_dir` at supervisor construction (hub `__init__`).

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_child_env_statedir.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/supervisor.py src/yasar_usta/hub.py tests/test_child_env_statedir.py
git commit -m "feat(supervisor): pass YASAR_USTA_STATE_DIR to managed children"
```

---

### Task 2.3: Orchestrator reads heartbeat path from env (kill split-brain)

**Repo/CWD:** `KUTAY`
**Files:** Modify: `src/app/run.py:379-381`, `src/core/orchestrator.py:487-488` · Test: `tests/yasar/test_heartbeat_path.py`

- [ ] **Step 1: Write the failing test**

Create `tests/yasar/test_heartbeat_path.py`:
```python
import importlib


def test_heartbeat_paths_from_state_dir(monkeypatch):
    monkeypatch.setenv("YASAR_USTA_STATE_DIR", r"C:\state\kutai")
    from src.app.hb_paths import heartbeat_paths  # new tiny helper
    paths = heartbeat_paths()
    assert paths[0].replace("\\", "/").endswith("state/kutai/orchestrator.heartbeat")


def test_heartbeat_paths_fallback_when_env_absent(monkeypatch):
    monkeypatch.delenv("YASAR_USTA_STATE_DIR", raising=False)
    from src.app.hb_paths import heartbeat_paths
    paths = heartbeat_paths()
    assert paths[0].replace("\\", "/").endswith("logs/orchestrator.heartbeat")
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/yasar/test_heartbeat_path.py -v`
Expected: FAIL (`src/app/hb_paths.py` missing).

- [ ] **Step 3: Create the helper + use it**

Create `src/app/hb_paths.py`:
```python
"""Single source of truth for where the orchestrator writes its heartbeat.
Under the Yaşar Usta hub, YASAR_USTA_STATE_DIR is authoritative so hub (reader)
and orchestrator (writer) never disagree. Falls back to the legacy relative
path for a non-hub launch."""
import os


def heartbeat_paths() -> tuple:
    sd = os.environ.get("YASAR_USTA_STATE_DIR")
    if sd:
        return (os.path.join(sd, "orchestrator.heartbeat"),
                os.path.join(sd, "heartbeat"))
    return ("logs/orchestrator.heartbeat", "logs/heartbeat")
```
In `run.py` replace the hardcoded `_hb_paths = ("logs/orchestrator.heartbeat", "logs/heartbeat")` with:
```python
    from src.app.hb_paths import heartbeat_paths
    _hb_paths = heartbeat_paths()
    write_heartbeat(*_hb_paths)
    _hb_writer = HeartbeatWriter(*_hb_paths, interval=15.0)
```
In `orchestrator.py:487-488` replace the hardcoded `"logs/orchestrator.heartbeat"` argument with the first element of `heartbeat_paths()`:
```python
        from src.app.hb_paths import heartbeat_paths
        _hbp = heartbeat_paths()
        await HeartbeatWriter(
            _hbp[0],
            ...  # keep the rest of the existing args (state_path/state_provider/interval)
        ).run()
```
(Preserve the existing `state_path`/`state_provider`/`interval` args on that `HeartbeatWriter`.)

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/yasar/test_heartbeat_path.py -v`
Expected: 2 passed.

- [ ] **Step 5: Registry — switch kutai heartbeat_file + state paths to `${state_dir}`**

In `HUB/registry.yaml` kutai target, change `heartbeat_file` to `"${state_dir}/orchestrator.heartbeat"`; move pid files/`claude_signal_file` under `${state_dir}` too (keep `log_file` orchestrator.jsonl in `${project_root}/logs` — it's KutAI's app log). Add a hub-reader/child-writer parity check:
```bash
.venv/Scripts/python.exe -c "import os; os.environ.setdefault('LOCALAPPDATA', r'C:/Users/sakir/AppData/Local'); from yasar_usta import load_registry; _,p=load_registry('registry.yaml', project_root='C:/Users/sakir/Dropbox/Workspaces/kutay'); print(p[0].targets[0].heartbeat_file)"
```
Expected: prints `.../YasarUsta/kutai/orchestrator.heartbeat`.

- [ ] **Step 6: Commit (both repos)**

```bash
cd /c/Users/sakir/Dropbox/Workspaces/kutay && git add src/app/hb_paths.py src/app/run.py src/core/orchestrator.py tests/yasar/test_heartbeat_path.py && git commit -m "fix: orchestrator heartbeat path from YASAR_USTA_STATE_DIR (kill split-brain)"
cd /c/Users/sakir/Dropbox/Workspaces/yasar_usta && git add registry.yaml && git commit -m "feat: kutai state files under \${state_dir} (LOCALAPPDATA)"
```

---

### Task 2.4: `log_dir` relative-default is fail-loud (close CWD trap)

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/hub.py` (`run()` start) · Test: `tests/test_logdir_absolute.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_logdir_absolute.py`:
```python
import pytest
from yasar_usta.hub import assert_state_dir_absolute
from yasar_usta.config import HubConfig


def test_relative_log_dir_rejected():
    with pytest.raises(SystemExit):
        assert_state_dir_absolute(HubConfig(name="T", log_dir="logs"))


def test_absolute_log_dir_ok():
    assert_state_dir_absolute(HubConfig(name="T", log_dir=r"C:\Users\x\AppData\Local\YasarUsta\hub"))
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_logdir_absolute.py -v`
Expected: FAIL (undefined).

- [ ] **Step 3: Implement + wire**

Add to `hub.py`:
```python
import os as _os


def assert_state_dir_absolute(cfg) -> None:
    if not _os.path.isabs(cfg.log_dir):
        raise SystemExit(
            f"[Yasar Usta] hub log_dir is relative ({cfg.log_dir!r}) — refusing to "
            "scatter state in CWD. Set an absolute ${state_dir}/LOCALAPPDATA path in registry.yaml.")
```
Call it at the very top of `run()` (before mkdir).

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_logdir_absolute.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/hub.py tests/test_logdir_absolute.py
git commit -m "fix(hub): fail loud on relative log_dir (CWD state trap)"
```

---

### Task 2.5: PHASE 2 LIVE-VERIFY (no false-kill) — USER

- [ ] **Step 1: Full suites**

Run (HUB): `.venv/Scripts/python.exe -m pytest -q` → all pass.
Run (KUTAY): `.venv/Scripts/python.exe -m pytest tests/yasar/ -q --timeout=60` → pass.

- [ ] **Step 2: USER live-verify checklist**

1. Restart the hub via `start.bat`.
2. Confirm `%LOCALAPPDATA%\YasarUsta\hub\hub.alive` updates; `…\kutai\orchestrator.heartbeat` updates; **nothing new churns in Dropbox `kutay/logs`** except `orchestrator.jsonl`.
3. Leave running 5+ min: the orchestrator is NOT false-killed (hub reads the same heartbeat the child writes).
4. Confirm the singleton probe + dashboard still healthy.

- [ ] **Step 3: Push** (main, after verify).

---

# PHASE 3 — Watchdog M4b must-fixes + installer rewrite

**Phase goal:** the hung-hub watchdog is safe to activate (no crash-loop, respects a deliberate stop, verifies the kill), and the auto-start installer targets the generic entry + LOCALAPPDATA. **Running the installer stays user-gated (Phase 4).**

---

### Task 3.1: M4b#1 — grace-after-kill marker

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/watchdog.py` (`run_once`) · Test: `tests/test_watchdog_grace.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_watchdog_grace.py`:
```python
from yasar_usta import watchdog as w


def test_no_kill_within_grace_after_prior_kill(tmp_path):
    alive = tmp_path / "hub.alive"
    alive.write_text("0")  # ancient → stale
    marker = tmp_path / ".watchdog_killed"
    marker.write_text(str(1000.0))  # killed at t=1000
    killed = w.run_once(str(alive), now=1100.0, threshold=360,
                        find_pids=lambda: [111], kill=lambda pid: None,
                        marker_path=str(marker), grace=360)
    assert killed == []  # 100s < 360s grace → skip


def test_kill_after_grace_expires(tmp_path):
    alive = tmp_path / "hub.alive"; alive.write_text("0")
    marker = tmp_path / ".watchdog_killed"; marker.write_text(str(1000.0))
    got = []
    killed = w.run_once(str(alive), now=1500.0, threshold=360,
                        find_pids=lambda: [111], kill=lambda pid: got.append(pid),
                        marker_path=str(marker), grace=360)
    assert killed == [111] and got == [111]
    # marker refreshed to the new kill time
    assert float((tmp_path / ".watchdog_killed").read_text()) == 1500.0
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_watchdog_grace.py -v`
Expected: FAIL (`run_once` has no `marker_path`/`grace`).

- [ ] **Step 3: Implement**

Extend `run_once` signature + logic:
```python
def _read_ts(path):
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None


def run_once(alive_path, now: float, *, threshold: float = DEFAULT_STALE_SECONDS,
             find_pids=find_hub_pids, kill=kill_pid,
             marker_path=None, grace: float = 3 * DEFAULT_INTERVAL_SECONDS) -> list:
    ts = read_alive_ts(alive_path)
    if not is_stale(ts, now, threshold):
        return []
    if marker_path:
        kts = _read_ts(marker_path)
        if kts is not None and (now - kts) < grace:
            print(f"[Yasar Watchdog] within grace ({now - kts:.0f}s<{grace}s) — skip")
            return []
    to_kill = list(find_pids())
    for pid in to_kill:
        print(f"[Yasar Watchdog] hub hung — killing PID {pid}")
        kill(pid)
    if to_kill and marker_path:
        try:
            Path(marker_path).write_text(str(now))
        except Exception:
            pass
    return to_kill
```
(Default `grace = 3 ticks` per spec §4.8 headroom.)

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_watchdog_grace.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/watchdog.py tests/test_watchdog_grace.py
git commit -m "fix(watchdog): grace-after-kill marker (no crash-loop on slow boot) — M4b#1"
```

---

### Task 3.2: M4b#2 — `hub.stopped` gate

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/watchdog.py` · Test: `tests/test_watchdog_stopped.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_watchdog_stopped.py`:
```python
from yasar_usta import watchdog as w


def test_deliberate_stop_suppresses_kill(tmp_path):
    alive = tmp_path / "hub.alive"; alive.write_text("0")  # stale
    stopped = tmp_path / "hub.stopped"; stopped.write_text("1")
    killed = w.run_once(str(alive), now=10_000.0, threshold=360,
                        find_pids=lambda: [222], kill=lambda p: None,
                        stopped_path=str(stopped))
    assert killed == []


def test_no_stopped_file_allows_kill(tmp_path):
    alive = tmp_path / "hub.alive"; alive.write_text("0")
    killed = w.run_once(str(alive), now=10_000.0, threshold=360,
                        find_pids=lambda: [222], kill=lambda p: None,
                        stopped_path=str(tmp_path / "hub.stopped"))
    assert killed == [222]
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_watchdog_stopped.py -v`
Expected: FAIL (`stopped_path` unknown).

- [ ] **Step 3: Implement**

Add `stopped_path=None` param to `run_once`; right after the staleness check:
```python
    if stopped_path and Path(stopped_path).exists():
        print("[Yasar Watchdog] hub.stopped present — deliberate stop, no kill")
        return []
```
(Also: the hub must `write hub.stopped` on any deliberate hub-down and remove it on boot — add a note/TODO in `hub.run()` start to `unlink(hub.stopped)` if present, and in the shutdown-hub command to create it. Wire the unlink in `run()` now; the create-on-deliberate-stop lands when a `/shutdown-hub` command exists — reference handoff §4-2.)

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_watchdog_stopped.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/yasar_usta/watchdog.py src/yasar_usta/hub.py tests/test_watchdog_stopped.py
git commit -m "fix(watchdog): hub.stopped gate (respect deliberate stop) — M4b#2"
```

---

### Task 3.3: M4b#3 — verify the kill actually killed the mutex-holder

**Repo/CWD:** `HUB`
**Files:** Modify: `src/yasar_usta/watchdog.py` · Test: `tests/test_watchdog_killverify.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_watchdog_killverify.py`:
```python
from yasar_usta import watchdog as w


def test_alerts_when_kill_fails(tmp_path):
    alive = tmp_path / "hub.alive"; alive.write_text("0")
    alerts = []
    def fake_is_alive(pid):  # pid still alive after kill → failure
        return True
    killed = w.run_once(str(alive), now=10_000.0, threshold=360,
                        find_pids=lambda: [333], kill=lambda p: None,
                        is_alive=fake_is_alive, alert=lambda m: alerts.append(m))
    assert any("still alive" in a.lower() or "failed" in a.lower() for a in alerts)


def test_no_alert_when_kill_succeeds(tmp_path):
    alive = tmp_path / "hub.alive"; alive.write_text("0")
    alerts = []
    killed = w.run_once(str(alive), now=10_000.0, threshold=360,
                        find_pids=lambda: [333], kill=lambda p: None,
                        is_alive=lambda pid: False, alert=lambda m: alerts.append(m))
    assert alerts == []
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_watchdog_killverify.py -v`
Expected: FAIL (`is_alive`/`alert` unknown).

- [ ] **Step 3: Implement**

Add `is_alive=None, alert=None` params. Provide a default `is_pid_alive`:
```python
def is_pid_alive(pid) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        return False
```
After the kill loop, if `is_alive` provided, re-check each killed pid; collect survivors; if any survive, call `alert(...)` (default: print) with a clear "hub PID(s) survived kill — possible zero-effective-hub" message. This is the silent-zero-hub guard.

- [ ] **Step 4: Run — verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_watchdog_killverify.py -v`
Expected: 2 passed.

- [ ] **Step 5: Wire main() to use real defaults + alert**

In `watchdog.main`, pass `marker_path`/`stopped_path` next to the alive path (same `\hub\` dir), `is_alive=is_pid_alive`, and an `alert` that writes to a log the installer can surface. Add `--marker`/`--stopped` args defaulting to siblings of `--alive`.

- [ ] **Step 6: Commit**

```bash
git add src/yasar_usta/watchdog.py tests/test_watchdog_killverify.py
git commit -m "fix(watchdog): verify kill-death, alert on survivor — M4b#3"
```

---

### Task 3.4: Rewrite `install_yasar_autostart.ps1` for the generic entry + LOCALAPPDATA

**Repo/CWD:** `HUB`
**Files:** Modify: `scripts/install_yasar_autostart.ps1`

- [ ] **Step 1: Edit the installer**

- `$root` → the hub repo dir; `$python` → hub `.venv\Scripts\python.exe`.
- Main action: `-Execute $python -Argument '-m yasar_usta --registry "<hub>\registry.yaml"'` `-WorkingDirectory $root`.
- `$alivePath` → `"$env:LOCALAPPDATA\YasarUsta\hub\hub.alive"`.
- Watchdog action: `-Argument "-m yasar_usta.watchdog --alive `"$alivePath`" --marker `"$env:LOCALAPPDATA\YasarUsta\hub\.watchdog_killed`" --stopped `"$env:LOCALAPPDATA\YasarUsta\hub\hub.stopped`""`.
- Keep the elevated at-logon trigger + 3-min watchdog repetition.

- [ ] **Step 2: Syntax check (do NOT run/register)**

Run: `powershell -NoProfile -Command "Get-Command -Syntax { . './scripts/install_yasar_autostart.ps1' } " 2>&1 | head` — or parse-check:
`powershell -NoProfile -Command "[System.Management.Automation.Language.Parser]::ParseFile('scripts/install_yasar_autostart.ps1',[ref]$null,[ref]$null) | Out-Null; 'parse OK'"`
Expected: `parse OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/install_yasar_autostart.ps1
git commit -m "chore(installer): target -m yasar_usta + LOCALAPPDATA alive/marker/stopped paths"
```

---

### Task 3.5: PHASE 3 verify

- [ ] **Step 1: Full hub suite**

Run (HUB): `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 2: USER live-verify (watchdog dry-run, still not registered)**

1. With the hub running, manually run one watchdog tick: `HUB/.venv/Scripts/python.exe -m yasar_usta.watchdog --alive "%LOCALAPPDATA%\YasarUsta\hub\hub.alive"` — expect it to find the healthy hub and kill nothing (fresh alive).
2. Simulate hang: stop the hub's alive writer (or hand-edit `hub.alive` to an old ts) with the process still up → the tick should find + kill it, write `.watchdog_killed`; a second immediate tick must skip (grace).

- [ ] **Step 3: Push** (main).

---

# PHASE 4 — Activation (USER-gated) + cleanup

**Not code. Hand to Sakir.**

- [ ] Run `scripts/install_yasar_autostart.ps1` **elevated** (registers at-logon main task + 3-min watchdog task).
- [ ] Configure auto-logon (`netplwiz`) for reboot-without-login (security tradeoff: stored password — user's call).
- [ ] Remove any legacy `start_kutai.vbs`/startup-folder launcher that could double-launch.
- [ ] Reboot test: PC restart → hub auto-starts, singleton holds, watchdog task present (`Get-ScheduledTask YasarUsta*`).
- [ ] Optional cleanup: `os._exit(0)` → `os._exit(42)` in `_do_restart_hub` so Task Scheduler becomes the sole relauncher (the Popen bridge also works; defer if unsure).
- [ ] Optional: retire legacy `guard.py` + its 2 characterization tests once nothing imports `ProcessGuard`.

---

## Self-Review (author checklist — completed)

- **Spec coverage:** §4.1 state axes → T1.13/T2.1/T2.3; §4.2 layout + guard.py → T1.1/Phase4; §4.3 boot-order asserts → T1.9; §4.4 subprocess hooks + Windows argv → T1.7/T1.12; §4.5 split-brain → T2.3; §4.6 declarative registry + **secrets** → T1.4/T1.6/T1.9/T1.13; §4.7 entry + `-m` self-fork + **watchdog matcher** → T1.9/T1.10/T1.11; §4.8 singleton (carried) + M4b#1-3 + grace headroom → T3.1/3.2/3.3; §5 onboarding → registry pattern in T1.13; §6 phasing → phase headers; §7 testing → per-task tests + T1.15/T2.5/T3.5; §8 risks → mitigations in T1.9/T2.3/T2.4/T1.14.
- **Placeholder scan:** no TBD/TODO left except the intentional `hub.stopped`-create deferral (T3.2 note, tied to a future `/shutdown-hub` command — flagged, not silent).
- **Type consistency:** `heartbeat_paths()` used identically in run.py + orchestrator.py + tests; `cmdline_is_hub`/`build_restart_command`/`build_hook_command`/`run_hook_subprocess`/`assert_consumer_imports`/`assert_hub_credentials`/`build_child_env`/`assert_state_dir_absolute` each defined once and referenced consistently; `run_once` param names (`marker_path`,`grace`,`stopped_path`,`is_alive`,`alert`) stable across T3.1-3.3.

## Plan-review (2nd adversarial pass) — blockers folded
Verified against real code; the following were caught pre-execution and fixed in-plan:
- **B1 on_exit orphaned** → T1.8 rewritten: keep supervisor-building, re-wire `tgt.on_exit` to `run_hook_subprocess(proj,"on_exit",…)` (else llama-server orphan-kill dies; `supervisor.py:331` seam preserved).
- **B2 psutil missing** → added to T1.2 deps (watchdog imports it).
- **B3 installer outside package** → T1.1 Step 4 copies `scripts/install_yasar_autostart.ps1` into HUB; T1.14 deletes the KUTAY copy.
- **B4 pre-existing tests break** → T1.7 deletes `test_hooks.py`; T1.14 retires `test_kutai_hooks.py`/`test_migration_kutai.py` (replaced by T1.12 CLI test).
- **pytest not in hub venv** → T1.3 installs `.[test]`.
- **self-restart dropped cwd/flags** → T1.10 pins `cwd=<hub repo root>` + keeps `**kwargs` (so `.env` loads on `/restart`).
- Confirmed sound by review: `Hub(hub_cfg, projects)`+`self.projects` (hub.py:32); registry per-project token refactor; `_norm(None)` safe; Phase-1 hub-state-in-LOCALAPPDATA vs child-heartbeat-in-logs is coherent (no early split-brain); `claude_cmd`/`db_path` declarative migration loses nothing; `status.py guard_script` correctly left alone.

## Known assumptions to validate during execution
- `supervisor.py` child-env construction site (T2.2) — reviewer confirmed it merges `os.environ` + `tgt.env`; confirm the exact function name before extracting `build_child_env`.
- `orchestrator.py:487-488` `HeartbeatWriter(...)` keeps `state_path`/`state_provider`/`interval` args — preserve them when swapping the path arg.
- Real `.env` token values copied into `HUB/.env` (T1.13 step 3) — the hub will refuse to boot without them (by design, T1.9).
- `test_hub.py` may reference `self._hooks`/`load_hook` — grep + fix stragglers in T1.8 Step 5.
