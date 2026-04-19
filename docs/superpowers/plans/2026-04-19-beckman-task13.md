# Beckman Task 13 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Phase 2b Task 13 — consolidate orchestrator's `_handle_*` lifecycle handlers into Beckman's internal rule set, rewrite `Orchestrator.run_loop` as a ≤300-line dispatch pump against `beckman.next_task()` / `beckman.on_task_finished()`, and extract the workflow engine into its own package invoked via a thin `salako.workflow_advance` executor.

**Architecture:** Follow the approved spec at `docs/superpowers/specs/2026-04-19-beckman-simplification-design.md`. Beckman's public API reduces to three methods (`next_task`, `on_task_finished`, `enqueue`); all cron firing, queue hygiene, and result-driven task creation happens inside those entry points. Lanes are deleted; swap/affinity concerns move to Hoca at per-call scope; `result_guards.py` dissolves with per-guard re-homing.

**Tech Stack:** Python 3.10, asyncio, aiosqlite (WAL), pytest, litellm. Windows 11 dev environment. Bash shell (Unix syntax). Uses `rtk` as a token-optimizing command prefix for git/pytest/etc. (binary at `C:/Users/sakir/ai/util/rtk.exe`).

**Migration style:** User chose single-branch high-tolerance (ship it, smoke-test the spec's success criteria, `git revert` the merge commit if something serious breaks). Still: every commit should pass `timeout 120 rtk pytest tests/` cleanly so `git bisect` stays useful after merge.

**Baseline:** 248 pre-existing test failures at parent commit `03a1def`. No new failures allowed in touched modules (`packages/general_beckman`, `packages/salako`, `src/core/`).

---

## Pre-flight (do ONCE before Task 1)

- [ ] **P.1: Confirm baseline.** From repo root, with PATH including `/c/Users/sakir/ai/util`:

  ```bash
  export PATH="$PATH:/c/Users/sakir/ai/util"
  rtk git status
  rtk git log --oneline -5
  ```

  Expected: clean working tree (any `egg-info` diffs are ignorable — they're the "deferred cleanup" item from the handoff), HEAD at a descendant of `03a1def` (the Phase 2b merge).

- [ ] **P.2: Create the worktree.**

  ```bash
  rtk git worktree add ../kutay-task13 -b feat/general-beckman-task13
  cd ../kutay-task13
  export PATH="$PATH:/c/Users/sakir/ai/util"
  ```

  All subsequent commands run from the worktree. If you're using an isolated Agent via `isolation: "worktree"`, the harness handles this — skip the manual command.

- [ ] **P.3: Capture baseline test failure count.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5 > /tmp/baseline_pytest.txt
  cat /tmp/baseline_pytest.txt
  ```

  Record the failure count (expected ≈ 248). You'll diff against this at the end of each task.

---

## Task 1: Schema extension + internal cron seeder scaffold

Foundation work. Extends `scheduled_tasks` schema to support both cron expressions (existing) and simple interval-based internal cadences. Adds an upsert-by-title helper and a placeholder seeder that will be fleshed out in Task 2. Intentionally non-invasive — nothing reads the new columns yet.

**Files:**
- Modify: `src/infra/db.py` — `CREATE TABLE scheduled_tasks` block (~line 198) + `get_due_scheduled_tasks` (~line 1746)
- Create: `packages/general_beckman/src/general_beckman/cron_seed.py`
- Create: `tests/test_beckman_cron_seed.py`

- [ ] **Step 1: Extend `scheduled_tasks` schema.**

  In `src/infra/db.py`, replace the `CREATE TABLE IF NOT EXISTS scheduled_tasks (…)` block at lines 199–212 with:

  ```python
  await db.execute("""
      CREATE TABLE IF NOT EXISTS scheduled_tasks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL,
          description TEXT,
          cron_expression TEXT,
          interval_seconds INTEGER,
          kind TEXT DEFAULT 'user',
          agent_type TEXT DEFAULT 'executor',
          tier TEXT DEFAULT 'cheap',
          enabled BOOLEAN DEFAULT 1,
          last_run TIMESTAMP,
          next_run TIMESTAMP,
          context JSON DEFAULT '{}'
      )
  """)
  ```

  Immediately after the CREATE TABLE, add idempotent column migrations (SQLite doesn't fail if columns exist when wrapped):

  ```python
  for col_sql in (
      "ALTER TABLE scheduled_tasks ADD COLUMN interval_seconds INTEGER",
      "ALTER TABLE scheduled_tasks ADD COLUMN kind TEXT DEFAULT 'user'",
  ):
      try:
          await db.execute(col_sql)
      except Exception:
          pass  # column already exists
  ```

- [ ] **Step 2: Write the failing test for the cron-seeder scaffold.**

  Create `tests/test_beckman_cron_seed.py`:

  ```python
  import pytest
  from general_beckman.cron_seed import seed_internal_cadences, INTERNAL_CADENCES
  from src.infra.db import get_db, init_db


  @pytest.mark.asyncio
  async def test_seed_internal_cadences_inserts_expected_rows(tmp_path, monkeypatch):
      monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
      # Force a fresh DB connection for the temp path.
      from src.infra import db as db_mod
      db_mod._db = None
      await init_db()
      await seed_internal_cadences()

      conn = await get_db()
      cursor = await conn.execute(
          "SELECT title, interval_seconds, kind FROM scheduled_tasks WHERE kind='internal'"
      )
      rows = [dict(r) for r in await cursor.fetchall()]
      titles = {r["title"] for r in rows}
      expected_titles = {c["title"] for c in INTERNAL_CADENCES}
      assert titles == expected_titles
      for r in rows:
          assert r["interval_seconds"] is not None


  @pytest.mark.asyncio
  async def test_seed_is_idempotent(tmp_path, monkeypatch):
      monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
      from src.infra import db as db_mod
      db_mod._db = None
      await init_db()
      await seed_internal_cadences()
      await seed_internal_cadences()  # second call must not duplicate

      conn = await get_db()
      cursor = await conn.execute(
          "SELECT COUNT(*) FROM scheduled_tasks WHERE kind='internal'"
      )
      (count,) = await cursor.fetchone()
      assert count == len(INTERNAL_CADENCES)
  ```

- [ ] **Step 3: Run the test and verify it fails.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_cron_seed.py -v
  ```

  Expected: ImportError / ModuleNotFoundError for `general_beckman.cron_seed`.

- [ ] **Step 4: Implement the cron-seeder scaffold.**

  Create `packages/general_beckman/src/general_beckman/cron_seed.py`:

  ```python
  """Seed internal cron cadences into the unified scheduled_tasks table.

  Seeded on first call to beckman.next_task() (lazy, guarded by a module-level
  flag) so orchestrator startup doesn't gain an explicit init() dependency.

  Internal rows are identified by kind='internal'. User rows are kind='user'.
  Sweep + benchmark refresh use 'marker' payloads — the cron processor
  recognises these and dispatches internally rather than inserting a new
  task row.
  """
  from __future__ import annotations

  import json

  from src.infra.db import get_db
  from src.infra.logging_config import get_logger

  logger = get_logger("beckman.cron_seed")

  # Single source of truth for internal cadences. Each entry becomes a row in
  # scheduled_tasks with kind='internal'. Marker payloads are dispatched by
  # general_beckman.cron._fire, not inserted as tasks.
  INTERNAL_CADENCES = [
      {
          "title": "beckman_sweep",
          "description": "Beckman internal: queue hygiene sweep",
          "interval_seconds": 300,
          "payload": {"_marker": "sweep"},
      },
      {
          "title": "hoca_benchmark_refresh",
          "description": "Beckman internal: Hoca benchmark cache refresh hook",
          "interval_seconds": 300,
          "payload": {"_marker": "benchmark_refresh"},
      },
      {
          "title": "todo_reminder",
          "description": "Every 2h: remind user of outstanding todos",
          "interval_seconds": 7200,
          "payload": {
              "_executor": "todo_reminder",
          },
      },
      {
          "title": "daily_digest",
          "description": "Daily 12:00 local: send digest",
          "interval_seconds": 86400,
          "payload": {
              "_executor": "daily_digest",
          },
      },
      {
          "title": "api_discovery",
          "description": "Daily API discovery cycle",
          "interval_seconds": 86400,
          "payload": {
              "_executor": "api_discovery",
          },
      },
      {
          "title": "nerd_herd_health_alert",
          "description": "Every 10min: resource health check + Telegram alert",
          "interval_seconds": 600,
          "payload": {"_marker": "nerd_herd_health"},
      },
  ]

  _seeded = False


  async def seed_internal_cadences() -> None:
      """Upsert internal cron rows. Idempotent. Safe to call repeatedly."""
      global _seeded
      if _seeded:
          return
      db = await get_db()
      for entry in INTERNAL_CADENCES:
          cursor = await db.execute(
              "SELECT id FROM scheduled_tasks WHERE title = ? AND kind = 'internal'",
              (entry["title"],),
          )
          row = await cursor.fetchone()
          if row:
              continue
          await db.execute(
              """INSERT INTO scheduled_tasks
                 (title, description, interval_seconds, kind, context, enabled)
                 VALUES (?, ?, ?, 'internal', ?, 1)""",
              (
                  entry["title"],
                  entry["description"],
                  entry["interval_seconds"],
                  json.dumps(entry["payload"]),
              ),
          )
      await db.commit()
      _seeded = True
      logger.info("internal cron cadences seeded",
                  count=len(INTERNAL_CADENCES))
  ```

- [ ] **Step 5: Run the test and verify it passes.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_cron_seed.py -v
  ```

  Expected: 2 passed.

- [ ] **Step 6: Run full suite; verify failure count not worse than baseline.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

  Expected: same or fewer failures than the baseline captured in P.3.

- [ ] **Step 7: Commit.**

  ```bash
  rtk git add src/infra/db.py packages/general_beckman/src/general_beckman/cron_seed.py tests/test_beckman_cron_seed.py
  rtk git commit -m "$(cat <<'EOF'
  feat(beckman): scheduled_tasks schema + internal-cadence seeder

  Add interval_seconds + kind columns to scheduled_tasks so a single
  table holds both user crons (cron_expression) and Beckman's internal
  cadences (interval_seconds). New cron_seed module lazy-upserts the
  canonical internal rows. Nothing reads the new columns yet.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 2: Beckman internal modules (pure logic + unit tests)

Build the pure-logic modules that Task 3–6 will wire up. Nothing is imported from `beckman.__init__` yet. All self-contained; testable without a running orchestrator.

**Files:**
- Create: `packages/general_beckman/src/general_beckman/rewrite.py` (action-rewriting rules — replaces guard suite)
- Create: `packages/general_beckman/src/general_beckman/apply.py` (action → DB row dispatch)
- Create: `packages/general_beckman/src/general_beckman/retry.py` (retry policy + DLQ write)
- Create: `packages/general_beckman/src/general_beckman/sweep.py` (queue hygiene — stuck, cascade, subtask rollup, escalation, workflow timeout)
- Create: `packages/general_beckman/src/general_beckman/cron.py` (scheduled_tasks processor — marker dispatch + task insertion)
- Create: `packages/general_beckman/src/general_beckman/paused_patterns.py` (module state)
- Create: `tests/test_beckman_rewrite.py`
- Create: `tests/test_beckman_retry.py`
- Create: `tests/test_beckman_apply.py`

- [ ] **Step 1: Read source to verify action types.**

  Read `packages/general_beckman/src/general_beckman/result_router.py` (already seen in brainstorm). Action types to reuse without modification: `Complete, SpawnSubtasks, RequestClarification, RequestReview, Exhausted, Failed`. A new `MissionAdvance` action will be added in a later step of this task.

- [ ] **Step 2: Extend `result_router` with `MissionAdvance` action.**

  Modify `packages/general_beckman/src/general_beckman/result_router.py`. Add after `Failed`:

  ```python
  @dataclass(frozen=True)
  class MissionAdvance:
      """Signal that a mission task completed cleanly — spawn workflow_advance."""
      task_id: int
      mission_id: int
      completed_task_id: int
      raw: dict = field(default_factory=dict)


  @dataclass(frozen=True)
  class CompleteWithReusedAnswer:
      """Complete derived from existing clarification_history (no re-ask)."""
      task_id: int
      result: str
      raw: dict = field(default_factory=dict)
  ```

  Extend the `Action` union:

  ```python
  Action = Union[
      Complete, SpawnSubtasks, RequestClarification, RequestReview,
      Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
  ]
  ```

  Do NOT modify `route_result` yet — rewriting rules inject `MissionAdvance` and `CompleteWithReusedAnswer` in `rewrite.py`, not in the router.

- [ ] **Step 3: Write failing tests for the rewrite rules.**

  Create `tests/test_beckman_rewrite.py`:

  ```python
  import pytest
  from general_beckman.rewrite import rewrite_actions
  from general_beckman.result_router import (
      Complete, SpawnSubtasks, RequestClarification, Failed,
      MissionAdvance, CompleteWithReusedAnswer,
  )


  def _task(**kw):
      base = {"id": 1, "mission_id": None, "context": "{}", "agent_type": "coder"}
      base.update(kw)
      return base


  def _ctx(**kw):
      return kw  # parse_context output is a plain dict


  def test_mission_task_complete_emits_mission_advance():
      task = _task(id=10, mission_id=5)
      actions = [Complete(task_id=10, result="done", raw={"status": "completed"})]
      out = rewrite_actions(task, _ctx(), actions)
      assert any(isinstance(a, MissionAdvance) for a in out)


  def test_non_mission_task_complete_unchanged():
      task = _task(id=10, mission_id=None)
      actions = [Complete(task_id=10, result="done", raw={})]
      out = rewrite_actions(task, _ctx(), actions)
      assert out == actions


  def test_workflow_step_blocking_subtask_emission():
      task = _task(id=20, mission_id=1)
      ctx = _ctx(workflow_step=True, mission_id=1)
      actions = [SpawnSubtasks(parent_task_id=20, subtasks=[{"t": "x"}], raw={})]
      out = rewrite_actions(task, ctx, actions)
      assert len(out) == 1
      assert isinstance(out[0], Failed)
      assert "decompose" in out[0].error.lower()


  def test_silent_task_clarify_becomes_failed():
      task = _task(id=30)
      ctx = _ctx(silent=True)
      actions = [RequestClarification(task_id=30, question="?", raw={})]
      out = rewrite_actions(task, ctx, actions)
      assert len(out) == 1
      assert isinstance(out[0], Failed)


  def test_may_need_clarification_false_clarify_becomes_failed():
      task = _task(id=31)
      ctx = _ctx(may_need_clarification=False)
      actions = [RequestClarification(task_id=31, question="?", raw={})]
      out = rewrite_actions(task, ctx, actions)
      assert len(out) == 1
      assert isinstance(out[0], Failed)


  def test_clarification_history_reused():
      task = _task(id=32)
      history = [{"question": "A?", "answer": "B"}]
      ctx = _ctx(clarification_history=history)
      actions = [RequestClarification(task_id=32, question="?", raw={})]
      out = rewrite_actions(task, ctx, actions)
      assert len(out) == 1
      assert isinstance(out[0], CompleteWithReusedAnswer)
      assert "A?" in out[0].result and "B" in out[0].result
  ```

- [ ] **Step 4: Run tests — expect import failure.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_rewrite.py -v
  ```

  Expected: ImportError for `general_beckman.rewrite`.

- [ ] **Step 5: Implement `rewrite.py`.**

  Create `packages/general_beckman/src/general_beckman/rewrite.py`:

  ```python
  """Pure action-rewriting rules. Replaces the old result_guards.py.

  Runs between result_router.route_result() and apply._apply_actions().
  No I/O — pure transformation of the action list given (task, task_ctx).

  Rules (order matters — earlier rules can short-circuit):
    1. Mission-task completion → inject MissionAdvance
    2. Workflow step emitted subtasks → replace with Failed (quality)
    3. Silent task requested clarification → replace with Failed
    4. may_need_clarification=False + clarify request → Failed
    5. Existing clarification_history + clarify request → CompleteWithReusedAnswer
  """
  from __future__ import annotations

  from typing import Iterable

  from general_beckman.result_router import (
      Action, Complete, SpawnSubtasks, RequestClarification,
      Failed, MissionAdvance, CompleteWithReusedAnswer,
  )


  def _is_workflow_step(task_ctx: dict) -> bool:
      return bool(task_ctx.get("workflow_step") or task_ctx.get("is_workflow_step"))


  def _format_history(history: list) -> str:
      parts = []
      for entry in history:
          if isinstance(entry, dict):
              q = entry.get("question", "")
              a = entry.get("answer", "")
          else:
              q, a = "", str(entry)
          if q or a:
              parts.append(f"**Q:** {q}\n**A:** {a}")
      return "\n\n".join(parts)


  def rewrite_actions(
      task: dict, task_ctx: dict, actions: Iterable[Action]
  ) -> list[Action]:
      out: list[Action] = []
      for a in actions:
          out.extend(_rewrite_one(task, task_ctx, a))
      return out


  def _rewrite_one(task: dict, task_ctx: dict, a: Action) -> list[Action]:
      # Rule 1: mission-task clean completion → also emit MissionAdvance
      if isinstance(a, Complete) and task.get("mission_id"):
          return [
              a,
              MissionAdvance(
                  task_id=a.task_id,
                  mission_id=task["mission_id"],
                  completed_task_id=a.task_id,
                  raw=a.raw,
              ),
          ]
      # Rule 2: workflow step tried to decompose
      if isinstance(a, SpawnSubtasks) and _is_workflow_step(task_ctx):
          return [Failed(
              task_id=a.parent_task_id,
              error="Workflow step tried to decompose instead of producing artifact",
              raw=a.raw,
          )]
      # Rules 3–5: clarification rewrites
      if isinstance(a, RequestClarification):
          if task_ctx.get("silent"):
              return [Failed(
                  task_id=a.task_id,
                  error="Insufficient info (silent task, no clarification)",
                  raw=a.raw,
              )]
          if task_ctx.get("may_need_clarification") is False:
              return [Failed(
                  task_id=a.task_id,
                  error="Agent requested clarification on no-clarification step",
                  raw=a.raw,
              )]
          history = task_ctx.get("clarification_history")
          if history:
              body = _format_history(history) or task_ctx.get("user_clarification", "")
              return [CompleteWithReusedAnswer(
                  task_id=a.task_id, result=body, raw=a.raw,
              )]
      return [a]
  ```

- [ ] **Step 6: Run the rewrite tests — expect pass.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_rewrite.py -v
  ```

  Expected: 6 passed.

- [ ] **Step 7: Write failing tests for retry policy.**

  Create `tests/test_beckman_retry.py`:

  ```python
  import pytest
  from general_beckman.retry import (
      RetryDecision, decide_retry, DLQAction,
  )


  def _failure(category="quality", attempts=1, max_attempts=3):
      return {
          "category": category,
          "worker_attempts": attempts,
          "max_worker_attempts": max_attempts,
          "model": "test-model",
      }


  def test_first_failure_retries_immediately():
      decision = decide_retry(_failure(attempts=1))
      assert decision.action == "immediate"


  def test_mid_attempts_retries_with_delay():
      decision = decide_retry(_failure(attempts=2, max_attempts=3))
      assert decision.action == "delayed"
      assert decision.delay_seconds > 0


  def test_exhausted_attempts_become_dlq():
      decision = decide_retry(_failure(attempts=3, max_attempts=3))
      assert isinstance(decision, DLQAction)


  def test_quality_bonus_granted_with_progress():
      # Task exhausted on quality, but progress >= 0.5: grant one bonus attempt.
      decision = decide_retry(
          _failure(category="quality", attempts=3, max_attempts=3),
          progress=0.75,
          bonus_count=0,
      )
      assert decision.action == "immediate"
      assert decision.bonus_used is True


  def test_quality_bonus_caps_at_two():
      decision = decide_retry(
          _failure(category="quality", attempts=3, max_attempts=3),
          progress=0.75,
          bonus_count=2,  # already used 2 bonuses
      )
      assert isinstance(decision, DLQAction)
  ```

- [ ] **Step 8: Run retry tests — expect import failure.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_retry.py -v
  ```

  Expected: ImportError.

- [ ] **Step 9: Implement `retry.py`.**

  Create `packages/general_beckman/src/general_beckman/retry.py`:

  ```python
  """Beckman's retry policy (replaces the inline `_quality_retry_flow`).

  Pure decision function. Callers (apply.py) do the DB work and DLQ writes.

  The quality bonus-attempt heuristic is preserved (flagged in the spec for
  a sideways look during migration, but not removed — it solves real
  DLQ-too-eagerly incidents).
  """
  from __future__ import annotations

  from dataclasses import dataclass
  from typing import Union

  _BACKOFF_SECONDS = [0, 10, 30, 120, 600]
  _MAX_BONUS = 2


  @dataclass(frozen=True)
  class RetryDecision:
      action: str  # "immediate" | "delayed"
      delay_seconds: int = 0
      bonus_used: bool = False


  @dataclass(frozen=True)
  class DLQAction:
      action: str = "dlq"
      category: str = "unknown"
      reason: str = ""


  Decision = Union[RetryDecision, DLQAction]


  def decide_retry(
      failure: dict,
      progress: float | None = None,
      bonus_count: int = 0,
  ) -> Decision:
      """Decide whether to retry a failed task.

      ``failure`` carries category, worker_attempts, max_worker_attempts, model.
      ``progress`` is the executor's self-assessed progress (0.0–1.0). Only
      considered when the category is ``quality`` and the task is otherwise
      exhausted.
      ``bonus_count`` is the number of bonus attempts already granted for
      this task (lives in task_ctx). Capped at ``_MAX_BONUS``.
      """
      attempts = int(failure.get("worker_attempts", 0))
      max_attempts = int(failure.get("max_worker_attempts", 3))
      category = failure.get("category", "unknown")

      if attempts < max_attempts:
          idx = min(attempts, len(_BACKOFF_SECONDS) - 1)
          delay = _BACKOFF_SECONDS[idx]
          return RetryDecision(
              action="immediate" if delay == 0 else "delayed",
              delay_seconds=delay,
          )

      # Exhausted. Consider quality bonus.
      if (
          category == "quality"
          and progress is not None
          and progress >= 0.5
          and bonus_count < _MAX_BONUS
      ):
          return RetryDecision(action="immediate", bonus_used=True)

      return DLQAction(
          category=category,
          reason=failure.get("error", "")[:300] or f"exhausted after {attempts} attempts",
      )
  ```

- [ ] **Step 10: Run retry tests — expect pass.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_retry.py -v
  ```

  Expected: 5 passed.

- [ ] **Step 11: Write failing tests for apply.**

  Create `tests/test_beckman_apply.py` with a fixture-based test that spins up a temp DB and verifies each action type produces the expected side-effect:

  ```python
  import json
  import pytest
  from general_beckman.apply import apply_actions
  from general_beckman.result_router import (
      Complete, SpawnSubtasks, RequestClarification, RequestReview,
      Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
  )
  from src.infra.db import init_db, add_task, get_task, get_db


  @pytest.mark.asyncio
  async def _fresh_db(tmp_path, monkeypatch):
      monkeypatch.setenv("DB_PATH", str(tmp_path / "apply.db"))
      from src.infra import db as db_mod
      db_mod._db = None
      await init_db()


  @pytest.mark.asyncio
  async def test_complete_marks_task_completed(tmp_path, monkeypatch):
      await _fresh_db(tmp_path, monkeypatch)
      tid = await add_task(title="t", description="", agent_type="coder")
      await apply_actions(
          await get_task(tid),
          [Complete(task_id=tid, result="done", raw={})],
      )
      row = await get_task(tid)
      assert row["status"] == "completed"


  @pytest.mark.asyncio
  async def test_subtasks_spawns_child_rows(tmp_path, monkeypatch):
      await _fresh_db(tmp_path, monkeypatch)
      parent = await add_task(title="p", description="", agent_type="planner")
      subs = [{"title": "c1", "description": "", "agent_type": "coder"}]
      await apply_actions(
          await get_task(parent),
          [SpawnSubtasks(parent_task_id=parent, subtasks=subs, raw={})],
      )
      row = await get_task(parent)
      assert row["status"] == "waiting_subtasks"
      conn = await get_db()
      cursor = await conn.execute(
          "SELECT COUNT(*) FROM tasks WHERE parent_task_id = ?", (parent,)
      )
      (n,) = await cursor.fetchone()
      assert n == 1


  @pytest.mark.asyncio
  async def test_clarification_spawns_salako_task(tmp_path, monkeypatch):
      await _fresh_db(tmp_path, monkeypatch)
      tid = await add_task(title="t", description="", agent_type="coder", chat_id=42)
      await apply_actions(
          await get_task(tid),
          [RequestClarification(task_id=tid, question="Why?", chat_id=42, raw={})],
      )
      row = await get_task(tid)
      assert row["status"] == "waiting_human"
      conn = await get_db()
      cursor = await conn.execute(
          """SELECT agent_type, payload FROM tasks WHERE parent_task_id = ?""", (tid,),
      )
      child = await cursor.fetchone()
      assert child is not None
      assert child["agent_type"] == "mechanical"
      pl = json.loads(child["payload"])
      assert pl["action"] == "clarify" and pl["question"] == "Why?"


  @pytest.mark.asyncio
  async def test_mission_advance_spawns_workflow_advance_task(tmp_path, monkeypatch):
      await _fresh_db(tmp_path, monkeypatch)
      parent = await add_task(title="mt", description="", agent_type="coder",
                              mission_id=7)
      await apply_actions(
          await get_task(parent),
          [MissionAdvance(task_id=parent, mission_id=7,
                          completed_task_id=parent, raw={})],
      )
      conn = await get_db()
      cursor = await conn.execute(
          """SELECT agent_type, payload FROM tasks
             WHERE mission_id = 7 AND id != ?""", (parent,),
      )
      child = await cursor.fetchone()
      assert child is not None
      assert child["agent_type"] == "mechanical"
      pl = json.loads(child["payload"])
      assert pl["executor"] == "workflow_advance"
      assert pl["mission_id"] == 7
      assert pl["completed_task_id"] == parent
  ```

- [ ] **Step 12: Run apply tests — expect import failure.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_apply.py -v
  ```

- [ ] **Step 13: Implement `apply.py`.**

  Create `packages/general_beckman/src/general_beckman/apply.py`:

  ```python
  """Apply Beckman actions to the DB. One branch per action type.

  Every function returns None. Side-effects: insert rows, update task status.
  Retry / DLQ decisions come from `general_beckman.retry`. Clarify and notify
  tasks are created as mechanical salako rows — salako executors do the
  actual Telegram I/O at dispatch time.
  """
  from __future__ import annotations

  import json
  from datetime import timedelta
  from typing import Iterable

  from src.infra.db import add_task, get_task, update_task, get_db
  from src.infra.logging_config import get_logger
  from src.infra.times import to_db, utc_now

  from general_beckman.result_router import (
      Action, Complete, SpawnSubtasks, RequestClarification, RequestReview,
      Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
  )
  from general_beckman.retry import decide_retry, DLQAction, RetryDecision

  logger = get_logger("beckman.apply")


  async def apply_actions(task: dict, actions: Iterable[Action]) -> None:
      for a in actions:
          await _apply_one(task, a)


  async def _apply_one(task: dict, a: Action) -> None:
      if isinstance(a, Complete):
          await _apply_complete(task, a)
      elif isinstance(a, CompleteWithReusedAnswer):
          await _apply_complete_reused(task, a)
      elif isinstance(a, SpawnSubtasks):
          await _apply_subtasks(task, a)
      elif isinstance(a, RequestClarification):
          await _apply_clarify(task, a)
      elif isinstance(a, RequestReview):
          await _apply_review(task, a)
      elif isinstance(a, Exhausted):
          await _apply_exhausted(task, a)
      elif isinstance(a, Failed):
          await _apply_failed(task, a)
      elif isinstance(a, MissionAdvance):
          await _apply_mission_advance(task, a)
      else:
          logger.warning("unknown action type", action=type(a).__name__)


  async def _apply_complete(task: dict, a: Complete) -> None:
      await update_task(
          a.task_id, status="completed",
          completed_at=to_db(utc_now()),
          result=a.result,
      )


  async def _apply_complete_reused(task: dict, a: CompleteWithReusedAnswer) -> None:
      await update_task(
          a.task_id, status="completed",
          completed_at=to_db(utc_now()),
          result=a.result,
      )


  async def _apply_subtasks(task: dict, a: SpawnSubtasks) -> None:
      for sub in a.subtasks:
          await add_task(
              title=sub.get("title", ""),
              description=sub.get("description", ""),
              agent_type=sub.get("agent_type", "coder"),
              parent_task_id=a.parent_task_id,
              mission_id=task.get("mission_id"),
              depends_on=sub.get("depends_on", []),
              context=sub.get("context", {}),
              priority=sub.get("priority", task.get("priority", 5)),
          )
      await update_task(a.parent_task_id, status="waiting_subtasks")


  async def _apply_clarify(task: dict, a: RequestClarification) -> None:
      await update_task(a.task_id, status="waiting_human")
      await add_task(
          title=f"Clarify: {task.get('title','')[:40]}",
          description=a.question,
          mission_id=task.get("mission_id"),
          parent_task_id=a.task_id,
          agent_type="mechanical",
          payload={
              "action": "clarify",
              "question": a.question,
              "chat_id": a.chat_id,
          },
          depends_on=[],
      )


  async def _apply_review(task: dict, a: RequestReview) -> None:
      # Dedup: if a review task already exists for this parent, skip.
      conn = await get_db()
      cursor = await conn.execute(
          """SELECT id FROM tasks
             WHERE parent_task_id = ? AND agent_type = 'reviewer'
               AND status IN ('pending', 'processing', 'ungraded')""",
          (a.task_id,),
      )
      if await cursor.fetchone():
          logger.info("review task deduped", parent=a.task_id)
          return
      await add_task(
          title=f"Review: {task.get('title','')[:40]}",
          description=a.summary,
          mission_id=task.get("mission_id"),
          parent_task_id=a.task_id,
          agent_type="reviewer",
          depends_on=[],
      )


  async def _apply_exhausted(task: dict, a: Exhausted) -> None:
      await _retry_or_dlq(task, category="exhausted", error=a.error)


  async def _apply_failed(task: dict, a: Failed) -> None:
      await _retry_or_dlq(task, category=task.get("error_category") or "worker",
                          error=a.error)


  async def _apply_mission_advance(task: dict, a: MissionAdvance) -> None:
      await add_task(
          title=f"Workflow advance: mission #{a.mission_id}",
          description="",
          agent_type="mechanical",
          mission_id=a.mission_id,
          depends_on=[],
          payload={
              "executor": "workflow_advance",
              "mission_id": a.mission_id,
              "completed_task_id": a.completed_task_id,
          },
      )


  async def _retry_or_dlq(task: dict, *, category: str, error: str) -> None:
      """Shared retry/DLQ path for Failed and Exhausted."""
      attempts = int(task.get("worker_attempts") or 0) + 1
      max_attempts = int(task.get("max_worker_attempts") or 3)
      progress = _parse_progress(task)
      ctx = _parse_ctx(task)
      bonus_count = int(ctx.get("_bonus_count", 0))

      decision = decide_retry(
          {
              "category": category,
              "worker_attempts": attempts,
              "max_worker_attempts": max_attempts,
              "model": task.get("model", ""),
              "error": error,
          },
          progress=progress,
          bonus_count=bonus_count,
      )

      if isinstance(decision, DLQAction):
          await _dlq_write(task, error=error, category=category, attempts=attempts)
          return

      if decision.bonus_used:
          ctx["_bonus_count"] = bonus_count + 1
          max_attempts += 1

      next_retry_at = None
      if decision.action == "delayed":
          next_retry_at = to_db(utc_now() + timedelta(seconds=decision.delay_seconds))

      await update_task(
          task["id"],
          status="pending",
          error=error[:500],
          worker_attempts=attempts,
          max_worker_attempts=max_attempts,
          error_category=category,
          next_retry_at=next_retry_at,
          context=json.dumps(ctx),
      )


  async def _dlq_write(task: dict, *, error: str, category: str, attempts: int) -> None:
      from src.infra.dead_letter import quarantine_task
      await update_task(
          task["id"], status="failed",
          error=error[:500],
          failed_in_phase=task.get("failed_in_phase") or "worker",
      )
      try:
          await quarantine_task(
              task_id=task["id"],
              mission_id=task.get("mission_id"),
              error=error[:500],
              error_category=category,
              original_agent=task.get("agent_type", "executor"),
              attempts_snapshot=attempts,
          )
      except Exception as exc:
          logger.warning("DLQ write failed", task_id=task["id"], error=str(exc))
      # Telegram DLQ notification → mechanical salako task (no inline send).
      await add_task(
          title=f"Notify: DLQ task #{task['id']}",
          description="",
          agent_type="mechanical",
          mission_id=task.get("mission_id"),
          payload={
              "executor": "notify_user",
              "message": (
                  f"❌ Task #{task['id']} → DLQ\n"
                  f"**{(task.get('title') or '')[:60]}**\n"
                  f"Reason: {error[:100]}"
              ),
          },
          depends_on=[],
      )


  def _parse_ctx(task: dict) -> dict:
      raw = task.get("context") or "{}"
      if isinstance(raw, dict):
          return dict(raw)
      try:
          parsed = json.loads(raw)
          return parsed if isinstance(parsed, dict) else {}
      except Exception:
          return {}


  def _parse_progress(task: dict) -> float | None:
      ctx = _parse_ctx(task)
      p = ctx.get("_last_progress")
      if isinstance(p, (int, float)):
          return float(p)
      return None
  ```

- [ ] **Step 14: Run apply tests — expect pass.**

  ```bash
  timeout 60 rtk pytest tests/test_beckman_apply.py -v
  ```

  Expected: 4 passed. If `add_task` doesn't accept `payload=` directly, inspect its signature at `src/infra/db.py` and update the tests and apply.py to match the real parameter name (the existing `lifecycle.py:77–89` already uses `payload=` so this should work).

- [ ] **Step 15: Implement `sweep.py`.**

  Port the body of `packages/general_beckman/src/general_beckman/watchdog.py::check_stuck_tasks` (already-read in brainstorm). Create `packages/general_beckman/src/general_beckman/sweep.py`. The port:

  - Copy the whole `check_stuck_tasks` function body verbatim.
  - Replace every `await telegram.send_notification(...)` call with a `salako.notify_user` task insertion using the same `_insert_notify_task` helper pattern as `apply._dlq_write`:

    ```python
    await add_task(
        title=f"Notify: stuck-task sweep",
        description="",
        agent_type="mechanical",
        payload={"executor": "notify_user", "message": <the message string>},
        depends_on=[],
    )
    ```

  - Rename the function to `sweep_queue()` and remove the `telegram=None` parameter entirely.
  - Keep all seven numbered sections (stuck-processing, ungraded safety, dep cascade, subtask rollup, overdue retry gates, waiting_human escalation, workflow timeout).

  No new unit tests — the function is a direct port with notifications redirected, and integration tests in Task 3 exercise it.

- [ ] **Step 16: Implement `cron.py`.**

  Create `packages/general_beckman/src/general_beckman/cron.py`:

  ```python
  """Scheduled-tasks processor.

  Reads due rows from scheduled_tasks, dispatches marker payloads internally
  (sweep, benchmark refresh, nerd_herd health) and inserts concrete task
  rows for non-marker payloads.
  """
  from __future__ import annotations

  import json
  from datetime import timedelta

  from src.infra.db import (
      get_due_scheduled_tasks, update_scheduled_task, add_task,
  )
  from src.infra.logging_config import get_logger
  from src.infra.times import utc_now, to_db

  from general_beckman.cron_seed import seed_internal_cadences
  from general_beckman.sweep import sweep_queue

  logger = get_logger("beckman.cron")


  async def fire_due() -> None:
      """Fire every scheduled_tasks row whose next_run is due.

      Called from beckman.next_task(). Idempotent per row via last_run/next_run
      advancement.
      """
      await seed_internal_cadences()
      rows = await get_due_scheduled_tasks()
      now = utc_now()
      for row in rows:
          try:
              payload = _parse_payload(row.get("context"))
              marker = payload.get("_marker")
              if marker == "sweep":
                  await sweep_queue()
              elif marker == "benchmark_refresh":
                  await _refresh_benchmarks_if_stale()
              elif marker == "nerd_herd_health":
                  await _nerd_herd_health_alert()
              else:
                  await _insert_scheduled_task(row, payload)
              await _advance_schedule(row, now)
          except Exception as e:
              logger.warning("cron fire failed",
                             sched_id=row.get("id"),
                             title=row.get("title"),
                             error=str(e))


  def _parse_payload(raw) -> dict:
      if isinstance(raw, dict):
          return raw
      if not raw:
          return {}
      try:
          out = json.loads(raw)
          return out if isinstance(out, dict) else {}
      except Exception:
          return {}


  async def _insert_scheduled_task(row: dict, payload: dict) -> None:
      executor = payload.get("_executor")
      if executor:
          await add_task(
              title=row.get("title", "scheduled"),
              description=row.get("description", ""),
              agent_type="mechanical",
              payload={"executor": executor, **{k: v for k, v in payload.items() if k != "_executor"}},
              depends_on=[],
          )
      else:
          # user-scheduled row with an agent_type — insert as that agent.
          await add_task(
              title=row.get("title", "scheduled"),
              description=row.get("description", ""),
              agent_type=row.get("agent_type", "executor"),
              context=payload,
              depends_on=[],
          )


  async def _advance_schedule(row: dict, now) -> None:
      interval = row.get("interval_seconds")
      cron_expr = row.get("cron_expression")
      if interval:
          next_run = to_db(now + timedelta(seconds=int(interval)))
      elif cron_expr:
          # Reuse add_scheduled_task's inline parser via croniter if available,
          # else advance by 1h as a conservative fallback.
          try:
              from croniter import croniter
              next_run = to_db(croniter(cron_expr, now).get_next(type(now)))
          except Exception:
              next_run = to_db(now + timedelta(hours=1))
      else:
          next_run = to_db(now + timedelta(hours=1))
      await update_scheduled_task(row["id"], last_run=to_db(now), next_run=next_run)


  async def _refresh_benchmarks_if_stale() -> None:
      try:
          import fatih_hoca
          hoca = getattr(fatih_hoca, "refresh_benchmarks_if_stale", None)
          if hoca is not None:
              await hoca()
      except Exception as e:
          logger.debug("hoca benchmark refresh skipped", error=str(e))


  async def _nerd_herd_health_alert() -> None:
      try:
          import nerd_herd
          summary = getattr(nerd_herd, "health_summary", None)
          if summary is None:
              return
          report = await summary() if callable(summary) else summary
          if not report or not report.get("alerts"):
              return
          await add_task(
              title="Notify: resource health",
              description="",
              agent_type="mechanical",
              payload={
                  "executor": "notify_user",
                  "message": "\n".join(f"• {a}" for a in report["alerts"]),
              },
              depends_on=[],
          )
      except Exception as e:
          logger.debug("nerd_herd health alert skipped", error=str(e))
  ```

- [ ] **Step 17: Implement `paused_patterns.py`.**

  Create `packages/general_beckman/src/general_beckman/paused_patterns.py`:

  ```python
  """DLQ pause-pattern filter (module state).

  Moved from Orchestrator.paused_patterns during Task 13. Telegram /dlq
  commands mutate these via pause() / unpause(); queue.pick reads via is_paused.
  """
  _patterns: set[str] = set()


  def pause(pattern: str) -> None:
      _patterns.add(pattern)


  def unpause(pattern: str) -> None:
      _patterns.discard(pattern)


  def all_paused() -> set[str]:
      return set(_patterns)


  def is_paused(task: dict) -> bool:
      cat = task.get("error_category")
      if not cat:
          return False
      return f"category:{cat}" in _patterns
  ```

- [ ] **Step 18: Run full suite; verify no new failures.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

  Expected: same or fewer failures than baseline.

- [ ] **Step 19: Commit.**

  ```bash
  rtk git add packages/general_beckman/src/general_beckman/{rewrite,apply,retry,sweep,cron,paused_patterns}.py \
              packages/general_beckman/src/general_beckman/result_router.py \
              tests/test_beckman_rewrite.py tests/test_beckman_retry.py tests/test_beckman_apply.py
  rtk git commit -m "$(cat <<'EOF'
  feat(beckman): internal modules for sweep, cron, rewrite, apply, retry

  Pure-logic modules that Task 3-6 will wire up. Nothing consumes them
  yet — public beckman API is unchanged. Includes new MissionAdvance
  and CompleteWithReusedAnswer action types in result_router.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 3: Rewrite `beckman.next_task()` to sweep + fire cron + pick

Switch Beckman's public `next_task()` to the new internal pipeline. Replaces lane-aware `pick_ready_task` with a simpler saturation-bit + priority-boost + paused-pattern-filter path. Orchestrator's `run_loop` still holds cron/scheduled_jobs logic for now — Task 7 deletes it.

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` — rewrite `next_task`
- Modify: `packages/general_beckman/src/general_beckman/queue.py` — add priority boost + paused-pattern filter; lane classification stays for now (Task 4 deletes it)
- Create: `tests/test_beckman_next_task.py`

- [ ] **Step 1: Write integration tests for `next_task()`.**

  Create `tests/test_beckman_next_task.py`:

  ```python
  import pytest
  from general_beckman import next_task, enqueue
  from general_beckman.paused_patterns import pause, unpause
  from src.infra.db import init_db, add_task, get_task, get_db


  @pytest.mark.asyncio
  async def _fresh(tmp_path, monkeypatch):
      monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
      from src.infra import db as db_mod
      db_mod._db = None
      await init_db()


  @pytest.mark.asyncio
  async def test_returns_none_when_queue_empty(tmp_path, monkeypatch):
      await _fresh(tmp_path, monkeypatch)
      assert await next_task() is None


  @pytest.mark.asyncio
  async def test_returns_single_pending_task(tmp_path, monkeypatch):
      await _fresh(tmp_path, monkeypatch)
      tid = await add_task(title="t", description="", agent_type="coder")
      task = await next_task()
      assert task is not None
      assert task["id"] == tid
      # claimed
      row = await get_task(tid)
      assert row["status"] == "processing"


  @pytest.mark.asyncio
  async def test_paused_pattern_excludes_matching_task(tmp_path, monkeypatch):
      await _fresh(tmp_path, monkeypatch)
      blocked = await add_task(title="b", description="", agent_type="coder",
                               error_category="quality")
      ok = await add_task(title="o", description="", agent_type="coder")
      pause("category:quality")
      try:
          task = await next_task()
          assert task is not None and task["id"] == ok
      finally:
          unpause("category:quality")
  ```

- [ ] **Step 2: Run — expect failures (priority/paused logic not in queue.py yet).**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_next_task.py -v
  ```

- [ ] **Step 3: Extend `queue.py` with priority boost + paused-pattern filter.**

  Modify `packages/general_beckman/src/general_beckman/queue.py`:

  ```python
  """Task queue: eligibility + priority boost + paused-pattern filter.

  Lane classification stays for Task 3; Task 4 deletes it.
  """
  from __future__ import annotations

  from src.infra.db import get_ready_tasks, claim_task, update_task
  from src.infra.times import from_db, utc_now

  from general_beckman.paused_patterns import is_paused


  def classify_lane(task: dict) -> str:  # DELETED in Task 4
      if task.get("agent_type") == "mechanical":
          return "mechanical"
      if task.get("agent_type") in {"researcher", "planner", "architect"}:
          return "cloud_llm"
      return "local_llm"


  def _effective_priority(task: dict) -> float:
      base = float(task.get("priority", 5))
      created = task.get("created_at", "")
      if not created:
          return base
      try:
          age_h = (utc_now() - from_db(created)).total_seconds() / 3600
      except Exception:
          return base
      return base + min(age_h * 0.1, 1.0)


  async def pick_ready_task(saturated_lanes: set[str]) -> dict | None:
      rows = await get_ready_tasks(limit=8)
      # Age-boost sort (stable: preserves DB tie-break for equal boosts)
      rows.sort(key=_effective_priority, reverse=True)
      for row in rows:
          if is_paused(row):
              continue
          lane = classify_lane(row)
          if lane in saturated_lanes:
              continue
          claimed = await claim_task(row["id"])
          if claimed:
              return row
      return None


  async def count_pending_cloud_tasks() -> int:  # DELETED in Task 4
      rows = await get_ready_tasks(limit=30)
      return sum(1 for r in rows if classify_lane(r) == "cloud_llm")


  async def unclaim(task: dict) -> None:
      await update_task(task["id"], status="pending")
  ```

- [ ] **Step 4: Rewrite `beckman.__init__.py` public API.**

  Overwrite `packages/general_beckman/src/general_beckman/__init__.py`:

  ```python
  """General Beckman — the task master.

  Public API (everything else is internal):
    - next_task() -> Task | None
    - on_task_finished(task_id, result) -> None
    - enqueue(spec) -> int
  """
  from __future__ import annotations

  from general_beckman.types import Task, AgentResult

  __all__ = ["next_task", "on_task_finished", "enqueue", "Task", "AgentResult"]


  def _capacity_snapshot():
      try:
          import nerd_herd
          nh = getattr(nerd_herd, "_singleton", None)
          if nh is None:
              return None
          return nh.snapshot()
      except Exception:
          return None


  def _saturated_lanes(snap) -> set[str]:
      """Transitional: kept until Task 4 deletes lanes entirely."""
      saturated: set[str] = set()
      if snap is None:
          return saturated
      try:
          if int(getattr(snap, "vram_available_mb", 0)) < 500 and \
             getattr(snap, "local", None) is not None:
              saturated.add("local_llm")
      except Exception:
          pass
      return saturated


  async def next_task():
      """Cycle: sweep (throttled) + fire due crons + pick one.

      Called by orchestrator on its ~3s cycle.
      """
      from general_beckman.cron import fire_due
      from general_beckman.queue import pick_ready_task

      # Cron processor internally seeds and throttles sweep.
      await fire_due()

      snap = _capacity_snapshot()
      saturated = _saturated_lanes(snap)
      return await pick_ready_task(saturated)


  async def on_task_finished(task_id: int, result: dict) -> None:
      # Kept as-is for now — Task 6 rewrites this to use rewrite+apply.
      from general_beckman.lifecycle import on_task_finished as _legacy
      await _legacy(task_id, result)


  async def enqueue(spec: dict) -> int:
      """Single external write path for user-/bot-initiated tasks."""
      from src.infra.db import add_task
      return await add_task(**spec)
  ```

  Note: `tick()` and `set_orchestrator` are removed from `__all__`. The orchestrator still calls `set_orchestrator(self)` during Phase 2b — temporarily keep `set_orchestrator` reachable as a non-public helper (just not exported). Add at the bottom:

  ```python
  from general_beckman.lifecycle import set_orchestrator  # noqa: F401, transitional
  ```

- [ ] **Step 5: Run next_task tests — expect pass.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_next_task.py -v
  ```

- [ ] **Step 6: Adjust the orchestrator's `run_loop` to stop firing scheduled_jobs directly.**

  Beckman's `next_task` now fires crons. Remove the duplicate fires from `src/core/orchestrator.py`:

  - Delete the block around lines 1977–1989 that calls `self.scheduled_jobs.check_scheduled_tasks()` and `self.scheduled_jobs.tick_benchmark_refresh()`.
  - Delete the age-boost block at lines 1994–2008 (moved to `queue._effective_priority`).
  - Delete the paused-pattern filter block at lines 2010–2020 (moved to `paused_patterns.is_paused`).

  Keep the following intact for Task 4:
  - `get_ready_tasks(limit=8)` + `max_concurrent` logic (being replaced in Task 4).
  - Swap-aware deferral (`_should_defer_for_loaded_model`) — Task 4.
  - Model-affinity reordering (`_reorder_by_model_affinity`) — Task 4.
  - Quota-planner forward scan — Task 4.

- [ ] **Step 7: Seed internal cadences so the sweep/benchmark markers exist in production DB.**

  No manual step — `seed_internal_cadences()` is called on every `next_task()` cycle and is idempotent. Double-check by running:

  ```bash
  timeout 30 rtk pytest tests/test_beckman_cron_seed.py tests/test_beckman_next_task.py -v
  ```

- [ ] **Step 8: Full suite.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

  Expected: same or fewer failures than baseline.

- [ ] **Step 9: Commit.**

  ```bash
  rtk git add packages/general_beckman/src/general_beckman/__init__.py \
              packages/general_beckman/src/general_beckman/queue.py \
              src/core/orchestrator.py tests/test_beckman_next_task.py
  rtk git commit -m "$(cat <<'EOF'
  refactor(beckman): next_task drives sweep + cron + pick

  Beckman's next_task now internally fires due crons and sweeps the
  queue (throttled via scheduled_tasks markers). Orchestrator's
  run_loop stops calling scheduled_jobs directly and no longer
  computes age-priority or paused-pattern filter inline.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 4: Delete lanes + swap-aware deferral + affinity; move model concerns to Hoca

Lanes don't exist under the new model. Swap budget and affinity are per-call concerns, not batch ordering — they move into Hoca's `select()`. Orchestrator's run_loop stops doing any model-specific batch logic.

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/queue.py` — delete `classify_lane`, `count_pending_cloud_tasks`, simplify `pick_ready_task` signature
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` — drop `_saturated_lanes`; saturation bit becomes a single boolean
- Modify: `packages/general_beckman/src/general_beckman/lookahead.py` — simpler or delete if count_pending_cloud_tasks was its only signal
- Modify: `src/core/orchestrator.py` — delete `_reorder_by_model_affinity` (lines ~147–232), delete `_should_defer_for_loaded_model`, delete the max_concurrent + local/cloud partition block (lines ~2022–2127), delete the quota planner scan block (lines ~2044–2091)
- Modify: `packages/fatih_hoca/src/fatih_hoca/` — audit `select()` to confirm swap budget + affinity already handle the per-call cases (they do per earlier spec notes); add a test if any gap appears

- [ ] **Step 1: Read Hoca's `select()` to confirm current swap/affinity behavior.**

  ```bash
  rtk grep -n "def select" packages/fatih_hoca/src/fatih_hoca/
  ```

  Confirm that Hoca's swap-budget enforcement and loaded-model preference already apply per-call. If yes, no Hoca change is needed. If a gap exists, note it for a follow-up commit inside this task.

- [ ] **Step 2: Simplify `queue.py`.**

  Replace `packages/general_beckman/src/general_beckman/queue.py` with:

  ```python
  """Task queue: eligibility + priority boost + paused-pattern filter."""
  from __future__ import annotations

  from src.infra.db import get_ready_tasks, claim_task, update_task
  from src.infra.times import from_db, utc_now

  from general_beckman.paused_patterns import is_paused


  def _effective_priority(task: dict) -> float:
      base = float(task.get("priority", 5))
      created = task.get("created_at", "")
      if not created:
          return base
      try:
          age_h = (utc_now() - from_db(created)).total_seconds() / 3600
      except Exception:
          return base
      return base + min(age_h * 0.1, 1.0)


  async def pick_ready_task(system_busy: bool) -> dict | None:
      if system_busy:
          # Allow mechanical tasks even when local GPU is saturated —
          # they never touch the LLM.
          pass
      rows = await get_ready_tasks(limit=8)
      rows.sort(key=_effective_priority, reverse=True)
      for row in rows:
          if is_paused(row):
              continue
          if system_busy and row.get("agent_type") != "mechanical":
              continue
          claimed = await claim_task(row["id"])
          if claimed:
              return row
      return None


  async def unclaim(task: dict) -> None:
      await update_task(task["id"], status="pending")
  ```

- [ ] **Step 3: Simplify `__init__.next_task()` to use a boolean saturation bit.**

  In `packages/general_beckman/src/general_beckman/__init__.py`, replace `_saturated_lanes` with:

  ```python
  def _system_busy(snap) -> bool:
      if snap is None:
          return False
      try:
          if int(getattr(snap, "vram_available_mb", 0)) < 500:
              return True
      except Exception:
          pass
      return False
  ```

  Update `next_task()`:

  ```python
  async def next_task():
      from general_beckman.cron import fire_due
      from general_beckman.queue import pick_ready_task

      await fire_due()
      snap = _capacity_snapshot()
      return await pick_ready_task(_system_busy(snap))
  ```

- [ ] **Step 4: Delete `lookahead.py` if it only existed to read `count_pending_cloud_tasks`.**

  ```bash
  rtk grep -rn "from general_beckman.lookahead\|from general_beckman import.*lookahead" .
  ```

  If only Beckman itself imports it, delete `packages/general_beckman/src/general_beckman/lookahead.py` and remove the `should_hold_back` call site from `__init__.py` (it was removed implicitly when Step 3 replaced `next_task`). If Hoca or orchestrator imports it, keep the file and replace its single cloud-pressure signal with a call to `hoca.cloud_pressure()` or similar — pick whatever makes the resulting plan smallest.

- [ ] **Step 5: Delete orchestrator's batch-level model concerns.**

  In `src/core/orchestrator.py`, delete:

  - The top-level `_reorder_by_model_affinity` function (roughly lines 147–232).
  - The top-level `_should_defer_for_loaded_model` helper (grep-find it; deletion is straightforward).
  - Inside `run_loop` (~lines 2022–2127):
    - The swap-aware deferral block (`loaded_model = …` and the `runnable/deferred` partition).
    - The `_reorder_by_model_affinity` call.
    - The `_compute_max_concurrent` block + local/cloud partition + `batch/deferred` calculation.
    - The quota-planner forward scan block (lines ~2044–2091) — replaced by Hoca's per-call cloud-pressure awareness.
    - Keep only: `candidate_tasks = await get_ready_tasks(limit=8)` → this whole fetch is redundant now (Beckman does it); collapse the loop section so it just calls `beckman.next_task()` inside the cycle.

  After this step the candidate-fetch / batching block in `run_loop` should be simply:

  ```python
  task = await beckman.next_task()
  if task is None:
      await asyncio.sleep(3)
      continue
  # Dispatch path unchanged for now — Task 8 shrinks it further.
  ```

  Keep the shutdown-signal, cycle-count, watchdog-every-10 branches in place; Task 8 cleans them.

- [ ] **Step 6: Full suite.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

  Expected: same or fewer failures than baseline. Watch specifically for any test that imported `classify_lane` or `count_pending_cloud_tasks`; if one pops up, update that test to the new shape.

- [ ] **Step 7: Commit.**

  ```bash
  rtk git add packages/general_beckman/src/general_beckman/__init__.py \
              packages/general_beckman/src/general_beckman/queue.py \
              packages/general_beckman/src/general_beckman/lookahead.py \
              src/core/orchestrator.py
  rtk git commit -m "$(cat <<'EOF'
  refactor: delete lanes, swap-aware batch deferral, affinity reorder

  Beckman's saturation check collapses to a single system-busy bit.
  Swap budget + affinity are per-call Hoca concerns, not batch-level
  orchestrator concerns. Quota-planner forward scan deleted; Hoca
  enforces cloud pressure at select() time.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 5: Workflow engine package + `salako.workflow_advance` executor

Extract the workflow engine into its own package and expose a single `advance()` entry point. Add a thin salako executor that delegates to it. Not wired yet — Task 6 spawns `workflow_advance` tasks via the MissionAdvance action.

**Files:**
- Create: `packages/workflow_engine/` — new package, follows the `dallama/nerd_herd` layout (`pyproject.toml`, `src/workflow_engine/`, `tests/`)
- Create: `packages/workflow_engine/src/workflow_engine/__init__.py` — re-exports `advance`
- Create: `packages/workflow_engine/src/workflow_engine/advance.py` — single entry point
- Create: `packages/salako/src/salako/workflow_advance.py`
- Modify: `packages/salako/src/salako/actions.py` — register the new executor
- Modify: `requirements.txt` — add `packages/workflow_engine` editable install

The existing `src/workflows/engine/` already has the required primitives (`hooks.py`, `pipeline_artifacts.py`, `post_execute_workflow_step`). Promote them by wrapping — NOT by moving — so this task is reversible.

- [ ] **Step 1: Scaffold the new package.**

  ```bash
  mkdir -p packages/workflow_engine/src/workflow_engine packages/workflow_engine/tests
  ```

  Create `packages/workflow_engine/pyproject.toml`:

  ```toml
  [build-system]
  requires = ["setuptools>=61.0"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "workflow_engine"
  version = "0.1.0"
  description = "KutAI workflow engine: recipe advance + artifact capture"
  requires-python = ">=3.10"
  dependencies = []

  [tool.setuptools.packages.find]
  where = ["src"]
  ```

  Create `packages/workflow_engine/src/workflow_engine/__init__.py`:

  ```python
  from workflow_engine.advance import advance, AdvanceResult
  __all__ = ["advance", "AdvanceResult"]
  ```

- [ ] **Step 2: Implement `advance.py` — thin wrapper over existing `src/workflows/engine` primitives.**

  Create `packages/workflow_engine/src/workflow_engine/advance.py`:

  ```python
  """Workflow engine: advance one mission by consuming a completed step's result.

  Delegates to src/workflows/engine primitives until/unless they are migrated
  wholesale into this package. Minimal surface: one function.
  """
  from __future__ import annotations

  from dataclasses import dataclass, field
  from typing import Any


  @dataclass
  class AdvanceResult:
      status: str = "completed"   # 'completed' | 'needs_clarification' | 'failed'
      error: str = ""
      next_subtasks: list[dict] = field(default_factory=list)
      artifacts: dict[str, Any] = field(default_factory=dict)


  async def advance(mission_id: int, completed_task_id: int,
                    previous_result: dict) -> AdvanceResult:
      """Post-step hook + artifact capture + next-phase subtask emission."""
      from src.workflows.engine.hooks import (
          is_workflow_step, post_execute_workflow_step, get_artifact_store,
      )
      from src.workflows.engine.pipeline_artifacts import extract_pipeline_artifacts
      from src.tools.workspace import get_mission_workspace
      from src.infra.db import get_task

      out = AdvanceResult()
      task = await get_task(completed_task_id)
      if task is None:
          out.status = "failed"
          out.error = f"completed_task_id {completed_task_id} not found"
          return out
      task_ctx = _parse_ctx(task)
      if not is_workflow_step(task_ctx):
          # Not a workflow step; nothing to advance. Callers should guard,
          # but we defend here too.
          return out

      # 1. Artifact capture (from guard_pipeline_artifacts).
      try:
          ws = None
          if task.get("mission_id"):
              try:
                  ws = get_mission_workspace(task["mission_id"])
              except Exception:
                  ws = None
          extra = await extract_pipeline_artifacts(task, previous_result, ws)
          if extra:
              store = get_artifact_store()
              for name, content in extra.items():
                  await store.store(mission_id, name, content)
              out.artifacts = dict(extra)
      except Exception:
          pass

      # 2. Post-hook: may flip status.
      try:
          await post_execute_workflow_step(task, previous_result)
      except Exception as e:
          out.status = "failed"
          out.error = str(e)[:300]
          return out

      flipped = previous_result.get("status")
      if flipped == "needs_clarification":
          out.status = "needs_clarification"
          out.error = previous_result.get("question", "")
          return out
      if flipped == "failed":
          out.status = "failed"
          out.error = previous_result.get("error", "Post-hook failed")
          return out

      # 3. Next-phase subtasks (if engine emits them).
      try:
          from src.workflows.engine.recipe import advance_recipe
          next_subs = await advance_recipe(mission_id, completed_task_id,
                                           previous_result)
          out.next_subtasks = list(next_subs or [])
      except ImportError:
          # No recipe-advance primitive yet — no-op. Phase transition logic
          # stays in _handle_complete until migrated.
          pass
      except Exception as e:
          out.status = "failed"
          out.error = f"advance_recipe: {e}"[:300]
      return out


  def _parse_ctx(task: dict) -> dict:
      import json
      raw = task.get("context") or "{}"
      if isinstance(raw, dict):
          return dict(raw)
      try:
          out = json.loads(raw)
          return out if isinstance(out, dict) else {}
      except Exception:
          return {}
  ```

  NOTE: `src.workflows.engine.recipe.advance_recipe` may not exist today. If it doesn't, the `except ImportError` branch is fine — the workflow engine will need a `recipe.py` with the phase-progression logic extracted from `_handle_complete`. **Defer that extraction to Task 6's subagent** — the handoff allows it (spec §"Deferred to migration plan").

- [ ] **Step 3: Register the package as an editable install.**

  Append to `requirements.txt`:

  ```
  -e ./packages/workflow_engine
  ```

  Run:

  ```bash
  pip install -e ./packages/workflow_engine
  ```

- [ ] **Step 4: Implement the salako executor.**

  Create `packages/salako/src/salako/workflow_advance.py`:

  ```python
  """Salako executor: delegate mission advance to workflow_engine."""
  from __future__ import annotations


  async def run(task: dict) -> dict:
      import json
      raw_payload = task.get("payload") or {}
      if isinstance(raw_payload, str):
          try:
              payload = json.loads(raw_payload)
          except Exception:
              payload = {}
      else:
          payload = dict(raw_payload)

      mission_id = payload.get("mission_id")
      completed_task_id = payload.get("completed_task_id")
      previous_result = payload.get("previous_result") or {}

      if mission_id is None or completed_task_id is None:
          return {
              "status": "failed",
              "error": "workflow_advance payload missing mission_id/completed_task_id",
          }

      from workflow_engine import advance

      result = await advance(mission_id, completed_task_id, previous_result)

      if result.status == "needs_clarification":
          return {"status": "needs_clarification", "question": result.error}
      if result.status == "failed":
          return {"status": "failed", "error": result.error}
      if result.next_subtasks:
          return {"status": "needs_subtasks", "subtasks": result.next_subtasks}
      return {"status": "completed", "result": "advance complete"}
  ```

- [ ] **Step 5: Register the executor in salako's dispatcher.**

  Modify `packages/salako/src/salako/actions.py`. Find the existing executor-registry block (search for `clarify` or `notify_user` registration) and add `workflow_advance` following the same pattern. The dispatch shape is already established by `salako/__init__.py::run`; confirm with:

  ```bash
  rtk read packages/salako/src/salako/__init__.py
  rtk read packages/salako/src/salako/actions.py
  ```

  Add a line like `"workflow_advance": workflow_advance.run,` to whichever dict is used.

- [ ] **Step 6: Smoke-test the executor.**

  Create `tests/test_salako_workflow_advance.py`:

  ```python
  import pytest
  from salako.workflow_advance import run


  @pytest.mark.asyncio
  async def test_missing_payload_fails():
      r = await run({"id": 1, "payload": {}})
      assert r["status"] == "failed"


  @pytest.mark.asyncio
  async def test_payload_with_str_json():
      # String payload is valid JSON; shouldn't explode. The advance call
      # will likely fail with no mission in the test DB, but we're testing
      # the payload-parsing path only.
      import json
      r = await run({"id": 1, "payload": json.dumps({})})
      assert r["status"] == "failed"
  ```

  Run:

  ```bash
  timeout 30 rtk pytest tests/test_salako_workflow_advance.py -v
  ```

- [ ] **Step 7: Full suite.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

- [ ] **Step 8: Commit.**

  ```bash
  rtk git add packages/workflow_engine/ packages/salako/src/salako/workflow_advance.py \
              packages/salako/src/salako/actions.py requirements.txt \
              tests/test_salako_workflow_advance.py
  rtk git commit -m "$(cat <<'EOF'
  feat: workflow_engine package + salako workflow_advance executor

  Thin wrapper around src/workflows/engine primitives. Salako executor
  delegates to workflow_engine.advance(). Not wired to task flow yet;
  Task 6 spawns workflow_advance tasks via the MissionAdvance action.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 6: Wire `on_task_finished` to rewrite + apply (delete the circular delegation)

The risk peak. `on_task_finished` stops delegating to `orchestrator._handle_*` and instead routes through `rewrite_actions` → `apply_actions`. Mission-task completions produce a `MissionAdvance` action that spawns a `salako.workflow_advance` task instead of calling `_handle_complete` inline. `_handle_*` stubs remain in place so Task 7 can delete them separately.

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` — `on_task_finished` rewritten
- Modify: `packages/general_beckman/src/general_beckman/lifecycle.py` — replaces the whole file with a thin "fast-path" wrapper used only for the `set_orchestrator` transitional reference
- Create: `tests/test_beckman_on_task_finished.py`

- [ ] **Step 1: Write integration tests for on_task_finished end-to-end.**

  Create `tests/test_beckman_on_task_finished.py`:

  ```python
  import json
  import pytest
  from general_beckman import on_task_finished
  from src.infra.db import init_db, add_task, get_task, get_db


  @pytest.mark.asyncio
  async def _fresh(tmp_path, monkeypatch):
      monkeypatch.setenv("DB_PATH", str(tmp_path / "otf.db"))
      from src.infra import db as db_mod
      db_mod._db = None
      await init_db()


  @pytest.mark.asyncio
  async def test_completed_result_marks_task_completed(tmp_path, monkeypatch):
      await _fresh(tmp_path, monkeypatch)
      tid = await add_task(title="t", description="", agent_type="coder")
      await on_task_finished(tid, {"status": "completed", "result": "ok"})
      row = await get_task(tid)
      assert row["status"] == "completed"


  @pytest.mark.asyncio
  async def test_mission_task_complete_spawns_workflow_advance(tmp_path, monkeypatch):
      await _fresh(tmp_path, monkeypatch)
      tid = await add_task(title="mt", description="", agent_type="coder",
                           mission_id=9)
      await on_task_finished(tid, {"status": "completed", "result": "ok"})
      conn = await get_db()
      cursor = await conn.execute(
          """SELECT agent_type, payload FROM tasks
             WHERE mission_id = 9 AND id != ?""", (tid,),
      )
      child = await cursor.fetchone()
      assert child is not None
      assert child["agent_type"] == "mechanical"
      pl = json.loads(child["payload"])
      assert pl["executor"] == "workflow_advance"


  @pytest.mark.asyncio
  async def test_clarify_spawns_salako_clarify_task(tmp_path, monkeypatch):
      await _fresh(tmp_path, monkeypatch)
      tid = await add_task(title="t", description="", agent_type="coder",
                           chat_id=42)
      await on_task_finished(tid, {"status": "needs_clarification",
                                   "question": "What?"})
      conn = await get_db()
      cursor = await conn.execute(
          """SELECT agent_type, payload FROM tasks
             WHERE parent_task_id = ?""", (tid,),
      )
      child = await cursor.fetchone()
      pl = json.loads(child["payload"])
      assert pl["action"] == "clarify"


  @pytest.mark.asyncio
  async def test_silent_task_clarify_becomes_failure(tmp_path, monkeypatch):
      await _fresh(tmp_path, monkeypatch)
      tid = await add_task(title="t", description="", agent_type="coder",
                           context={"silent": True})
      await on_task_finished(tid, {"status": "needs_clarification",
                                   "question": "?"})
      row = await get_task(tid)
      assert row["status"] == "pending"  # retry, eventually failed
  ```

- [ ] **Step 2: Run — expect existing legacy behavior to fail (mission spawn won't happen).**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_on_task_finished.py -v
  ```

- [ ] **Step 3: Rewrite `on_task_finished` to use rewrite + apply.**

  In `packages/general_beckman/src/general_beckman/__init__.py`, replace `on_task_finished` with:

  ```python
  async def on_task_finished(task_id: int, result: dict) -> None:
      """Mark terminal + create any follow-up tasks the result implies."""
      from general_beckman.result_router import route_result
      from general_beckman.rewrite import rewrite_actions
      from general_beckman.apply import apply_actions
      from general_beckman.task_context import parse_context
      from src.infra.db import get_task
      from src.infra.logging_config import get_logger

      log = get_logger("beckman.on_task_finished")
      task = await get_task(task_id)
      if task is None:
          log.warning("on_task_finished: missing task", task_id=task_id)
          return
      task_ctx = parse_context(task)
      actions = route_result(task, result)
      if actions is None:
          return
      if not isinstance(actions, (list, tuple)):
          actions = [actions]
      actions = rewrite_actions(task, task_ctx, actions)
      await apply_actions(task, actions)
  ```

- [ ] **Step 4: Prune `lifecycle.py` to a thin transitional stub.**

  Overwrite `packages/general_beckman/src/general_beckman/lifecycle.py`:

  ```python
  """Transitional shim kept for set_orchestrator compatibility.

  Task 7 deletes this file entirely once Orchestrator stops calling
  set_orchestrator(self) in its __init__.
  """
  from __future__ import annotations

  from typing import Any

  _ORCH_INSTANCE: Any = None


  def set_orchestrator(instance: Any) -> None:
      global _ORCH_INSTANCE
      _ORCH_INSTANCE = instance


  def get_orchestrator() -> Any:
      if _ORCH_INSTANCE is None:
          raise RuntimeError("orchestrator not registered")
      return _ORCH_INSTANCE
  ```

  The circular `handle_complete(...) → orch._handle_complete(...)` wrappers are gone. The `_handle_*` methods on Orchestrator are now orphan code, removed in Task 7.

- [ ] **Step 5: Run the new tests + full suite.**

  ```bash
  timeout 30 rtk pytest tests/test_beckman_on_task_finished.py -v
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

  Expected: new tests pass; full suite failures ≤ baseline. If a test that used to pass via `_handle_complete`'s mission-progression logic now fails because `workflow_engine.advance_recipe` doesn't exist, extract the phase-progression logic from `Orchestrator._handle_complete` (lines ~1141–1343) into `src/workflows/engine/recipe.py::advance_recipe`, called by `workflow_engine.advance.advance`. Commit that extraction as a follow-up step inside this task before moving on.

- [ ] **Step 6: Commit.**

  ```bash
  rtk git add packages/general_beckman/src/general_beckman/__init__.py \
              packages/general_beckman/src/general_beckman/lifecycle.py \
              tests/test_beckman_on_task_finished.py \
              src/workflows/engine/recipe.py packages/workflow_engine/src/workflow_engine/advance.py
  rtk git commit -m "$(cat <<'EOF'
  refactor(beckman): on_task_finished uses rewrite + apply + MissionAdvance

  Drops the circular get_orchestrator()._handle_* delegation. Mission
  task completions spawn a salako workflow_advance task instead of
  running _handle_complete inline. Recipe-advance logic extracted
  into workflow_engine.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 7: Delete `_handle_*`, `result_guards.py`, `scheduled_jobs.py` (Beckman shim), old watchdog

All the logic these hold was re-homed in Tasks 2–6. Confirm zero external callers, then delete.

**Files:**
- Modify: `src/core/orchestrator.py` — delete `_handle_availability_failure`, `_handle_unexpected_failure`, `_handle_complete`, `_handle_subtasks`, `_handle_clarification`, `_handle_review`, `_handle_exhausted`, `_handle_failed`, plus `_dispatch_action` / `_run_guards_for` helpers added in Phase 2b. Delete `set_orchestrator(self)` call from `__init__`. Delete `self.paused_patterns`.
- Delete: `packages/general_beckman/src/general_beckman/result_guards.py`
- Delete: `packages/general_beckman/src/general_beckman/lifecycle.py` (the thin stub from Task 6; no longer needed once orchestrator stops registering itself)
- Delete: `packages/general_beckman/src/general_beckman/scheduled_jobs.py` (grab-bag replaced by cron.py)
- Delete: `packages/general_beckman/src/general_beckman/watchdog.py` — `check_stuck_tasks` was ported to `sweep.py`, `check_resources` moves to nerd_herd in Task 8
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` — remove `set_orchestrator` re-export

- [ ] **Step 1: Grep for external callers of the handlers.**

  ```bash
  rtk grep -rn "_handle_availability_failure\|_handle_unexpected_failure\|_handle_complete\|_handle_subtasks\|_handle_clarification\|_handle_review\|_handle_exhausted\|_handle_failed" packages/ src/ tests/ | grep -v "orchestrator.py"
  ```

  Expected: no results outside `orchestrator.py`. If any appear, either they're legitimate migration-leftover (fix on the spot) or tests (update to use `on_task_finished` as the public entry).

- [ ] **Step 2: Delete the eight handlers from `src/core/orchestrator.py`.**

  Line ranges from earlier grep (re-verify with `rtk grep -n "async def _handle_" src/core/orchestrator.py` before deleting):

  | Method | Approx lines |
  |---|---|
  | `_handle_availability_failure` | 994–1048 |
  | `_handle_unexpected_failure`   | 1049–1140 |
  | `_handle_complete`             | 1141–1343 |
  | `_handle_subtasks`             | 1344–1537 |
  | `_handle_clarification`        | 1538–1544 |
  | `_handle_review`               | 1545–1566 |
  | `_handle_exhausted`            | 1567–1662 |
  | `_handle_failed`               | 1663–1904 |

  Delete all eight. Also delete `_dispatch_action` and `_run_guards_for` helpers introduced in Phase 2b (grep for them).

- [ ] **Step 3: Remove `set_orchestrator(self)` from `Orchestrator.__init__`.**

  ```bash
  rtk grep -n "set_orchestrator" src/core/orchestrator.py
  ```

  Remove the call. Remove `self.paused_patterns: set[str] = set()` initialization (moved to `general_beckman.paused_patterns`). Update any `/dlq` Telegram commands that mutated `self.paused_patterns` to call `general_beckman.paused_patterns.pause(...)` / `.unpause(...)` directly.

- [ ] **Step 4: Delete the dead beckman modules.**

  ```bash
  rtk git rm packages/general_beckman/src/general_beckman/result_guards.py
  rtk git rm packages/general_beckman/src/general_beckman/lifecycle.py
  rtk git rm packages/general_beckman/src/general_beckman/scheduled_jobs.py
  rtk git rm packages/general_beckman/src/general_beckman/watchdog.py
  ```

- [ ] **Step 5: Remove the `set_orchestrator` re-export from beckman's `__init__`.**

  Delete the `from general_beckman.lifecycle import set_orchestrator` line added in Task 3.

- [ ] **Step 6: Delete tests that exercised `_handle_*` directly.**

  ```bash
  rtk grep -rln "_handle_complete\|_handle_subtasks\|_handle_failed\|_handle_exhausted" tests/
  ```

  For each match: if the test covers a lifecycle behavior already covered by `test_beckman_on_task_finished.py`, delete it. If it covers a behavior not yet tested, move the assertion into `test_beckman_on_task_finished.py`.

- [ ] **Step 7: Full suite.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

  Expected: failure count ≤ baseline. Investigate any increase before committing.

- [ ] **Step 8: Commit.**

  ```bash
  rtk git add -u
  rtk git commit -m "$(cat <<'EOF'
  refactor(orchestrator): delete _handle_* handlers and beckman shims

  All lifecycle logic re-homed to beckman rewrite/apply and
  workflow_engine. Deletes: 8 _handle_* methods, _dispatch_action,
  _run_guards_for, paused_patterns state on Orchestrator;
  result_guards.py, lifecycle.py, scheduled_jobs.py, watchdog.py
  in general_beckman (check_resources moves to nerd_herd in Task 8).

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 8: Shrink `run_loop` to the dispatch pump + move `check_resources` to Nerd Herd

Final main-loop collapse. Orchestrator's `run_loop` becomes a ≤30-line pump + startup + shutdown. Resource health check moves into nerd_herd.

**Files:**
- Modify: `src/core/orchestrator.py` — `run_loop` and `_dispatch` simplified to the spec's shape. Delete top-level helpers no longer used.
- Create: `packages/nerd_herd/src/nerd_herd/health.py`
- Modify: `packages/nerd_herd/src/nerd_herd/__init__.py` — export `health_summary`

- [ ] **Step 1: Port `check_resources` into nerd_herd.**

  Create `packages/nerd_herd/src/nerd_herd/health.py` with the body of the old `check_resources` function (seen in brainstorm, lines 361–524 of the deleted `watchdog.py`). Rename to `async def health_summary() -> dict:` — return a structured dict:

  ```python
  {
      "alerts": [str, ...],         # strings for Telegram alerting
      "issues": [str, ...],         # full issue list for logs / /status
      "providers_degraded": [...],
      "vram_leak_suspected": bool,
      "gpu_throttling": bool,
      "low_ram": bool,
      "credentials_expired": [...],
      "local_model_healthy": bool,
  }
  ```

  Drop all `telegram.send_notification` calls. The `beckman.cron._nerd_herd_health_alert` marker already spawns `salako.notify_user` tasks from `report["alerts"]`.

  Export from `packages/nerd_herd/src/nerd_herd/__init__.py`:

  ```python
  from nerd_herd.health import health_summary  # noqa: F401
  ```

- [ ] **Step 2: Rewrite `Orchestrator.run_loop`.**

  Replace the whole `run_loop` body in `src/core/orchestrator.py` with:

  ```python
  async def run_loop(self):
      self.running = True
      logger.info("🚀 Autonomous orchestrator started")
      try:
          import os
          from src.app.config import WORKSPACE_ROOT
          os.makedirs(WORKSPACE_ROOT, exist_ok=True)
          await ensure_git_repo()
      except Exception as e:
          logger.warning(f"Workspace/git init: {e}")

      from pathlib import Path
      shutdown_signal = Path("logs") / "shutdown.signal"
      import general_beckman

      while self.running and not self.shutdown_event.is_set():
          try:
              if shutdown_signal.exists():
                  intent = shutdown_signal.read_text().strip()
                  shutdown_signal.unlink()
                  logger.info("External shutdown signal: %s", intent)
                  self.requested_exit_code = 42 if intent == "restart" else 0
                  self.shutdown_event.set()
                  break
              if self._shutting_down:
                  logger.info("Shutdown flag set — draining running tasks")
                  break

              task = await general_beckman.next_task()
              if task is not None:
                  asyncio.create_task(self._dispatch(task))
              await asyncio.sleep(3)
          except asyncio.CancelledError:
              raise
          except Exception as e:
              logger.exception("run_loop iteration failed: %s", e)
              await asyncio.sleep(3)

      logger.info("Orchestrator main loop exited")
  ```

- [ ] **Step 3: Simplify `Orchestrator._dispatch(task)`.**

  Replace the existing `process_task` / `_dispatch_action` chain with a single dispatch method:

  ```python
  async def _dispatch(self, task: dict) -> None:
      import general_beckman
      from packages.salako.src import salako  # or however salako is imported today
      try:
          if task.get("agent_type") == "mechanical":
              result = await asyncio.wait_for(
                  salako.run(task), timeout=self._timeout_for(task),
              )
          else:
              result = await asyncio.wait_for(
                  self.llm_dispatcher.request(task),
                  timeout=self._timeout_for(task),
              )
      except asyncio.TimeoutError:
          result = {"status": "failed", "error": "dispatch timeout"}
      except Exception as e:
          logger.exception("dispatch failed for task #%s: %s", task.get("id"), e)
          result = {"status": "failed", "error": str(e)[:300]}
      try:
          await general_beckman.on_task_finished(task["id"], result)
      except Exception as e:
          logger.exception("on_task_finished raised for #%s: %s", task.get("id"), e)
  ```

  `_timeout_for(task)` is an existing helper; keep it.

- [ ] **Step 4: Delete newly-orphan top-level helpers in orchestrator.**

  Grep for helpers no longer called: `_compute_max_concurrent`, `_parse_task_difficulty` (if unused now), any of the old `process_task` sub-phases (`_prepare`, `_record`, etc.). Delete any with zero callers. Keep `_timeout_for`, `ensure_git_repo`, and anything still referenced.

- [ ] **Step 5: Line-count check.**

  ```bash
  wc -l src/core/orchestrator.py
  ```

  Target: ≤ 300 lines. If over, find what's still inline that doesn't belong and extract (e.g. `check_inflight_timeouts` if still there → delete; Task 13 already agreed dispatch-time `asyncio.wait_for` is sufficient).

- [ ] **Step 6: Manual smoke (required by spec).**

  - Start the orchestrator: `python kutai_wrapper.py` (dry-run: let it tick for ≥ 60s, confirm no exception flood in `logs/orchestrator.jsonl`).
  - Via Telegram: `/task echo hello` → task appears, runs, completes, Telegram reply arrives.
  - `/shop coffee beans` → shopping pipeline runs through the mission path (at least one scrape succeeds, or the mission hits DLQ gracefully).
  - Kill orchestrator (`/restart` via Telegram), confirm Yaşar Usta auto-restarts it, confirm stuck-`processing` rows get reset by Beckman's sweep on the first cycle.
  - Let a clarification happen (ask a `/task` that'll need one), reply with `/clarify` or via normal flow, confirm the round-trip completes.

  Record observations in the commit message.

- [ ] **Step 7: Full suite.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

- [ ] **Step 8: Commit.**

  ```bash
  rtk git add src/core/orchestrator.py packages/nerd_herd/src/nerd_herd/health.py \
              packages/nerd_herd/src/nerd_herd/__init__.py
  rtk git commit -m "$(cat <<'EOF'
  refactor(orchestrator): run_loop collapses to dispatch pump

  run_loop is now <30 lines: shutdown signal + beckman.next_task()
  pump + asyncio.create_task dispatch. _dispatch wraps the runner in
  wait_for(timeout=...) and calls beckman.on_task_finished on return.
  check_resources moved to nerd_herd.health_summary(); alerts reach
  Telegram via the nerd_herd_health cron marker + salako notify_user.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 9: Docs update + final cleanup

Update architecture docs to reflect the shipped state; tidy lingering debris.

**Files:**
- Modify: `docs/architecture-modularization.md` — Phase 2b section updated; new Phase 2b-final or "Task 13 completion" note
- Modify: `CLAUDE.md` — adjust the "Mechanical executor" and "Critical Rules" sections if inaccurate after the migration
- Modify: `.gitignore` — add `*.egg-info/` (the handoff's deferred cleanup)

- [ ] **Step 1: Update architecture doc.**

  In `docs/architecture-modularization.md`, locate the "Phase 2b — General Beckman" section. Add a new subsection at the end:

  ```markdown
  ### Phase 2b — Task 13 (shipped 2026-04-19+)

  Orchestrator.run_loop is now ~30 lines: a beckman.next_task pump wrapped in
  asyncio.create_task dispatch, with asyncio.wait_for enforcing dispatch-time
  timeouts. All _handle_* lifecycle methods are removed; their logic is split:

  - task creation & retry / DLQ decisions → `general_beckman.apply` + `retry`
  - action rewriting (workflow-step block, clarification suppression, mission
    advance injection) → `general_beckman.rewrite`
  - queue hygiene (stuck, cascade, subtask rollup, escalations, workflow
    timeout) → `general_beckman.sweep`, fired by the internal "sweep" cron
    marker every ~5min
  - workflow-step post-hook + artifact capture + next-phase emission →
    `packages/workflow_engine/` via a thin `salako.workflow_advance` executor
  - resource health (GPU, KDV, credentials) → `nerd_herd.health_summary`,
    alerts dispatched via a `nerd_herd_health` cron marker that spawns
    `salako.notify_user` tasks

  Beckman's public API is exactly 3 methods: next_task, on_task_finished,
  enqueue. Lanes, swap-aware batch deferral, and model-affinity reordering
  are deleted; swap budget + loaded-model affinity are per-call Hoca concerns
  inside fatih_hoca.select().
  ```

- [ ] **Step 2: Fix CLAUDE.md references.**

  Scan `CLAUDE.md` for:
  - Mentions of `_handle_*` — remove/rephrase.
  - Mentions of `paused_patterns` living on Orchestrator — update to `general_beckman.paused_patterns`.
  - Mentions of `scheduled_jobs` module in general_beckman — remove; replace with `cron.py`.

- [ ] **Step 3: .gitignore cleanup.**

  ```bash
  rtk grep -c "egg-info" .gitignore || echo "not present"
  ```

  If not present, append:

  ```
  *.egg-info/
  ```

  Then:

  ```bash
  rtk git rm -r --cached packages/fatih_hoca/src/fatih_hoca.egg-info/ 2>/dev/null || true
  rtk git rm -r --cached packages/hallederiz_kadir/src/hallederiz_kadir.egg-info/ 2>/dev/null || true
  ```

- [ ] **Step 4: Full suite.**

  ```bash
  timeout 300 rtk pytest tests/ 2>&1 | tail -5
  ```

- [ ] **Step 5: Final line-count check.**

  ```bash
  wc -l src/core/orchestrator.py
  rtk grep -c "^" packages/general_beckman/src/general_beckman/*.py
  ```

  `orchestrator.py` ≤ 300 lines. Beckman internals each ≤ 400 lines (sweep being the largest; if it's huge, leave a follow-up note rather than split).

- [ ] **Step 6: Commit + merge handoff.**

  ```bash
  rtk git add docs/architecture-modularization.md CLAUDE.md .gitignore
  rtk git commit -m "$(cat <<'EOF'
  docs(arch): Phase 2b Task 13 shipped — Beckman simplification complete

  orchestrator.py ~30-line pump; _handle_* handlers removed; workflow
  engine extracted as its own package; result_guards.py dissolved.
  Updates architecture-modularization.md and CLAUDE.md to reflect the
  shipped state. Adds *.egg-info/ to .gitignore.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **Step 7: Back out to main and open for user review.**

  From the worktree:

  ```bash
  rtk git log --oneline origin/main..HEAD
  ```

  Expected: 9 commits (one per task). Do **not** push. User will inspect/merge.

---

## Self-review

Run this checklist before handing back to the user.

1. **Spec coverage.** Every section of `docs/superpowers/specs/2026-04-19-beckman-simplification-design.md` has at least one task addressing it:
   - Public API (3 methods) — Tasks 3, 4, 6
   - `next_task` internals — Task 3
   - `on_task_finished` internals — Task 6
   - Action schema stays + rewrite rules — Task 2
   - Retry policy inside Beckman — Task 2
   - DLQ writes inside Beckman — Task 2
   - Cron table unification + seeder + markers — Tasks 1, 2, 3
   - Workflow engine as package + salako executor — Task 5
   - Orchestrator shape ≤ 300 lines — Task 8
   - Hoca absorbs swap/affinity — Task 4
   - `result_guards.py` deleted with per-guard re-home — Task 7
   - `_handle_*` deleted — Task 7
   - Paused-patterns moved to Beckman — Task 7
   - Nerd Herd gains `health_summary` — Task 8
   - Docs + final cleanup — Task 9

2. **Placeholder scan.** All "TBD" / "TODO" eliminated from the plan body. Where implementations lean on reading existing source (port of `check_stuck_tasks` into `sweep.py`, deletion of `_reorder_by_model_affinity` at known line ranges), the plan gives exact file/line targets + rewrite instructions, not vague prose.

3. **Type consistency.** Action type names (`Complete`, `SpawnSubtasks`, `RequestClarification`, `RequestReview`, `Exhausted`, `Failed`, `MissionAdvance`, `CompleteWithReusedAnswer`) are used consistently across Tasks 2, 3, 6. Function names (`rewrite_actions`, `apply_actions`, `decide_retry`, `sweep_queue`, `fire_due`, `seed_internal_cadences`) are consistent across tasks.

4. **Migration style.** User chose single-branch (C); plan reflects that. Each commit still lands test-green so bisect works after merge.

5. **Behavior risk.** Task 6 is the flagged high-risk commit (on_task_finished cutover). If Task 6 breaks something after merge, `git revert <task-6-sha>` is the surgical revert; if the whole branch misbehaves, `git revert -m 1 <merge-sha>` backs out the lot.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-beckman-task13.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?
