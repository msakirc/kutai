# Plan: Phase 2a — Extract Mechanical Dispatcher into `packages/salako/`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-04-18
**Base branch:** `main` (Phase 1 landed at commit 95404d6)
**Handoff doc:** `docs/superpowers/plans/2026-04-17-phase2-handoff.md`
**Independence:** Does NOT depend on Plan A. Routing is inserted in `process_task` before the LLM dispatch path; never touches the `_handle_*` chain.

## Goal

Promote the in-tree `src/core/mechanical/` module to a standalone Turkish-nickname package `packages/salako/`, a sibling executor to the LLM Dispatcher. Re-enable `auto_commit` as an explicit workflow step in i2p. Leave `src/core/mechanical/` as a re-export shim (mirroring `src/core/router.py` → `fatih_hoca`) so existing tests and imports keep working.

## Naming convention

All executors/packages use Turkish nicknames. The new package is **`salako`** — the mechanical worker who just does the grunt work without asking.

## Target architecture

```
Orchestrator.process_task
  ├─ _prepare(task)            # claim → classify → enrich → gate → (snapshot via salako)
  ├─ if task.executor == "mechanical":
  │     action = await salako.run(task)   # NEW routing, before LLM path
  │     return
  └─ <existing LLM dispatch path unchanged>
```

Salako public API:
- `salako.run(task: dict) -> Action` — routes `task["payload"]["action"]` ∈ `{"workspace_snapshot", "git_commit", ...}`.
- `salako.snapshot_workspace(...)` — ported verbatim.
- `salako.auto_commit(task, result)` — ported verbatim.

## File structure

**Created:**
- `packages/salako/pyproject.toml`
- `packages/salako/README.md`
- `packages/salako/src/salako/__init__.py` — public API
- `packages/salako/src/salako/workspace_snapshot.py`
- `packages/salako/src/salako/git_commit.py`
- `packages/salako/src/salako/actions.py` — `Action` result type + dispatcher
- `packages/salako/tests/__init__.py`
- `packages/salako/tests/test_workspace_snapshot.py`
- `packages/salako/tests/test_git_commit.py`
- `packages/salako/tests/test_run.py`
- `packages/salako/tests/test_init.py`

**Modified:**
- `src/core/mechanical/__init__.py` — becomes shim
- `src/core/mechanical/workspace_snapshot.py` — 1-line re-export
- `src/core/mechanical/git_commit.py` — 1-line re-export
- `src/core/orchestrator.py` — mechanical-executor routing in `process_task`
- `src/workflows/i2p/i2p_v3.json` — explicit `auto_commit` step after coder steps
- Root install script — add `-e packages/salako` editable install

**Invariants to preserve:**
- `pytest tests/core/mechanical/test_workspace_snapshot.py` still passes unchanged through the shim.
- Full `timeout 120 pytest tests/` stays green.
- No behavioral change visible from Telegram.

---

## Task 1: Scaffold `packages/salako/`

**Goal:** Empty package installable in editable mode; mirrors `fatih_hoca` layout.

**Files:** `pyproject.toml`, `README.md`, `src/salako/__init__.py`, `tests/__init__.py`, `tests/test_init.py`.

**pyproject skeleton:**
```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "salako"
version = "0.1.0"
description = "Mechanical dispatcher — non-LLM task executors (workspace snapshot, git commit)"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
```

**Verification:** `pip install -e packages/salako` → `python -c "import salako"` → `pytest packages/salako/tests/test_init.py`

**Acceptance:** Import works; editable install succeeds.

---

## Task 2: Move `workspace_snapshot` into salako

**Files:**
- Create `packages/salako/src/salako/workspace_snapshot.py` (logger → `"salako.workspace_snapshot"`).
- Create `packages/salako/tests/test_workspace_snapshot.py` (patch paths updated).
- Replace `src/core/mechanical/workspace_snapshot.py` with shim: `from salako.workspace_snapshot import snapshot_workspace  # noqa: F401`
- Re-export in `salako/__init__.py`.

**Verification:** Both test files green; `from src.core.mechanical.workspace_snapshot import snapshot_workspace` resolves.

---

## Task 3: Move `git_commit` into salako and add direct tests

**Files:**
- Create `packages/salako/src/salako/git_commit.py` (logger → `"salako.git_commit"`).
- Replace `src/core/mechanical/git_commit.py` with shim.
- Re-export in `salako/__init__.py`.
- Create `packages/salako/tests/test_git_commit.py`:
  - `test_auto_commit_runs_git_commit_on_success`
  - `test_auto_commit_skips_on_nothing_to_commit`
  - `test_auto_commit_swallows_exceptions`
  - `test_auto_commit_uses_mission_workspace_path_when_mission_id_present`

**Acceptance:** First real coverage on `auto_commit`.

---

## Task 4: Public API — `salako.run(task) -> Action`

**Files:**
- `packages/salako/src/salako/actions.py`:
  ```python
  from dataclasses import dataclass, field
  from typing import Any

  @dataclass
  class Action:
      status: str                  # "completed" | "failed" | "skipped"
      result: dict[str, Any] = field(default_factory=dict)
      error: str | None = None
  ```
- Extend `packages/salako/src/salako/__init__.py`:
  ```python
  from .actions import Action
  from .workspace_snapshot import snapshot_workspace
  from .git_commit import auto_commit

  async def run(task: dict) -> Action:
      payload = task.get("payload") or {}
      action = payload.get("action")
      if action == "workspace_snapshot":
          snap = await snapshot_workspace(
              mission_id=task["mission_id"],
              task_id=task["id"],
              workspace_path=payload["workspace_path"],
              repo_path=payload.get("repo_path"),
          )
          if snap is None:
              return Action(status="failed", error="snapshot failed")
          return Action(status="completed", result=snap)
      if action == "git_commit":
          await auto_commit(task, payload.get("result") or {})
          return Action(status="completed")
      return Action(status="failed", error=f"unknown mechanical action: {action!r}")
  ```
- `packages/salako/tests/test_run.py` — unknown action, both happy paths, negative snapshot path.

**Acceptance:** Public `run()` routes 2 actions + one negative path, unit-tested.

---

## Task 5: Orchestrator routing — send mechanical tasks to `salako.run()`

**Files:** `src/core/orchestrator.py`

- Add `import salako` near other imports.
- In `process_task`, after `_prepare` but BEFORE existing LLM dispatch block:
  ```python
  if task.get("executor") == "mechanical":
      mech_action = await salako.run(task)
      if mech_action.status == "completed":
          await update_task(task_id, status="completed",
                            result=json.dumps(mech_action.result))
      else:
          await update_task(task_id, status="failed",
                            error=mech_action.error or "mechanical action failed")
      return
  ```

**New test:** `tests/core/test_orchestrator_mechanical_routing.py`
- `test_mechanical_executor_routes_to_salako` — mocks `salako.run`, asserts `update_task(status="completed")` AND LLM path NOT invoked.
- `test_mechanical_executor_failure_marks_task_failed`.
- `test_no_executor_tag_still_takes_llm_path` (regression guard).

**Acceptance:** Mechanical tasks never hit model selection. LLM-tagged / untagged tasks unchanged.

---

## Task 6: Re-wire `auto_commit` as an explicit i2p workflow step

**Files:**
- `src/workflows/i2p/i2p_v3.json`: for curated 2–3 primary coder-family steps, append:
  ```json
  {
    "step_id": "<coder_step_id>.git_commit",
    "agent_type": "mechanical",
    "executor": "mechanical",
    "payload": {"action": "git_commit"},
    "depends_on": ["<coder_step_id>"],
    "difficulty": 0,
    "tools_hint": []
  }
  ```
- `src/workflows/engine/dispatch.py` (grep for step→task materialization): step with `agent_type == "mechanical"` sets `task["executor"] = "mechanical"` on creation.

**Tests:**
- `tests/workflows/test_i2p_mechanical_commit_step.py` — load JSON, assert mechanical step exists with correct `depends_on`.
- `tests/workflows/test_mechanical_step_materializes_with_executor_tag.py` — materialize through engine, assert `executor == "mechanical"`.

**Acceptance:** `auto_commit` no longer dormant; runs only on explicit mechanical steps.

---

## Task 7: Migrate `_prepare` workspace snapshot through salako (Option B)

**Goal:** Today, `_prepare` calls `snapshot_workspace()` directly. Replace with inline `salako.run` call so both entry points funnel through one seam.

**Files:**
- `src/core/orchestrator.py` `_prepare`: replace direct call with inline `salako.run(synthetic_mechanical_task_dict)`. Behavior (None-on-failure semantics) preserved since `Action.status == "failed"` maps to "snapshot skipped, continue".
- Add `tests/core/test_prepare_snapshot_uses_salako.py`.

**Fallback:** If benchmarks show overhead, revert to Option A (direct call through shim).

**Acceptance:** Both snapshot and git-commit flow exclusively through `salako.run()`. `src/core/mechanical/` is purely shim.

---

## Task 8: Install wiring + docs

**Files:**
- Root install script / `requirements.txt` (grep for how `fatih_hoca` is installed): add `-e packages/salako`.
- Update `CLAUDE.md` package-boundaries section to list `salako` next to `fatih_hoca`.
- Update `docs/architecture-modularization.md`.
- `packages/salako/README.md`: purpose, public API, example task payload, test command.

**Verification:** Fresh install sanity + `grep -rn "src.core.mechanical" src/` shows only shim references.

---

## Global verification (plan acceptance)

1. `pip install -e packages/salako` succeeds clean.
2. `pytest packages/salako/tests/` — all green.
3. `pytest tests/core/mechanical/test_workspace_snapshot.py` — passes unchanged via shim.
4. `timeout 180 pytest tests/` — no new regressions.
5. Grep confirms `src/core/mechanical/*.py` contain only re-exports.
6. Orchestrator diff: mechanical tasks routed to `salako.run` before LLM path; inline snapshot call replaced.
7. i2p_v3.json contains at least one `agent_type: mechanical` step with `payload.action: git_commit`.
8. Telegram smoke: coder task through i2p produces a git commit.

## Risks / open questions

- **Python path shape.** `src/` layout imports (`from src.infra.db import ...`) inside `salako.workspace_snapshot` are load-bearing. Keep them — they work because repo root is on `sys.path`. If ever used outside this repo, inject as deps (leave as TODO in README).
- **Workflow engine dispatch path.** Task 6 assumes a single materialization point. Grep first; if >3 call sites, descope to "force `executor='mechanical'` at runtime based on `agent_type`" inside orchestrator.
- **`_prepare` semantic change in Task 7.** Option B adds one Python function call overhead. Acceptable; fall back to Option A if benchmarks say otherwise.
- **Auto-commit scope.** Re-enabling via explicit step means ad-hoc `/task` submissions with a coder agent no longer auto-commit. Intended; call out in CLAUDE.md.
