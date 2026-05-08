# Beckman — Singular Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every LLM call and every mechanical job goes through `beckman.enqueue()`. Beckman's pump is the only dispatcher. Dispatcher (`src/core/llm_dispatcher.py`) becomes Beckman's worker — never a public entry. 16 LLM callsites + 2 mechanical loops migrated. Three exclusions documented.

**Architecture:** Single admission contract on `general_beckman.enqueue()` widened with `kind`, `parent_id`, `await_inline`, `on_complete`, `next_task_spec`. Caller patterns split: continuation (default) vs direct (`await_inline=True`, telegram-only, 3 sites). Agent ReAct refactored to checkpointable state so sub-call awaits don't hold open coroutines. Mechanical loops (`monitoring_check`, `vector_maint_*`) join existing cron seeds.

**Out of scope (explicit):** wake-on-enqueue, multi-dispatch-per-tick drain, sub-process re-entry. Pump cadence stays at 3s poll, 1 task / iteration.

**Tech Stack:** Python 3.10, aiosqlite WAL, litellm. Packages: `packages/general_beckman/`, `packages/mr_roboto/`, `src/core/`, `src/agents/`, `src/workflows/`, `src/app/`.

**Reference spec:** `docs/superpowers/specs/2026-05-04-beckman-singular-admission-design.md`

**Reference handoffs:**
- `docs/handoff/2026-05-04-unify-non-beckman-paths.md` (LLM site inventory)
- `docs/superpowers/specs/2026-04-29-pool-pressure-utilization-equilibrium-design.md` (admission pressure)

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `packages/general_beckman/src/general_beckman/continuations.py` | `on_complete` handler registry + dispatch on task terminal |
| `packages/mr_roboto/src/mr_roboto/executors/monitoring_check.py` | URL uptime + GitHub poll, emits `notify_user` sub-tasks |
| `packages/mr_roboto/src/mr_roboto/executors/vector_maint.py` | ChromaDB WAL checkpoint + snapshot wrapped in `run_in_executor` |
| `src/agents/checkpoint.py` | Agent state serialize/resume across enqueued sub-call awaits |

### Modified files

| Path | Change |
|---|---|
| `src/infra/db.py` schema bootstrap | Add `kind` column to `tasks` (TEXT NOT NULL DEFAULT 'main_work'); idempotent ALTER |
| `packages/general_beckman/src/general_beckman/__init__.py:607-613` | `enqueue()` signature widens with `parent_id`, `kind`, `await_inline`, `on_complete`, `next_task_spec`; folds in `reserve_task` |
| `packages/general_beckman/src/general_beckman/__init__.py` (router/result_router) | On task terminal, fire registered `on_complete` handler + chain `next_task_spec` |
| `packages/general_beckman/src/general_beckman/cron_seed.py` | Add `monitoring_check` (300s), `vector_maint_wal` (1800s), `vector_maint_snapshot` (86400s) markers |
| `packages/general_beckman/src/general_beckman/cron.py` | Wire new markers into `fire_due()` |
| `src/core/llm_dispatcher.py:80-413` | `request()` becomes deprecation alias → `beckman.enqueue(await_inline=True)`; new `dispatch(task)` is Beckman-internal worker |
| `src/core/llm_dispatcher.py:240-291` | Delete pool_pressure read + KDV.pre_call (Beckman owns); delete est_tokens shim from commit `5f7f905` |
| `src/core/heartbeat.py` | Delete `current_task_id` ContextVar (replaced by explicit `parent_id`) |
| `src/core/orchestrator.py:256` | Delete `_hb.current_task_id.set(...)` site |
| `src/agents/base.py:2498` | ReAct main → enqueue kind=main_work (already-admitted; this is the worker side) |
| `src/agents/base.py:3782, 3870, 3977` | structured_emit / alt-prompt-retry / self-reflection → enqueue kind=overhead, parent_id=self.task_id |
| `src/agents/base.py` (loop) | Use `agents/checkpoint.py` to suspend/resume on sub-call await |
| `src/core/grading.py:305` | Grader → enqueue kind=overhead, parent_id=task_id arg |
| `src/workflows/engine/hooks.py:46` | Summarizer → enqueue kind=overhead, on_complete="hook.resume_post_execute" |
| `src/tools/vision.py:29` | Vision → enqueue kind=tool_call, parent_id=agent.task_id |
| `src/app/telegram_bot.py:4145, 4570` | Router classifier + casual chat → enqueue kind=classifier/chat, await_inline=True |
| `src/core/task_classifier.py:258` | Pre-task classifier → enqueue kind=classifier, await_inline=True |
| `src/shopping/intelligence/_llm.py:42` | Shopping helper → enqueue kind=overhead, parent_id=parent shopping task |
| `src/workflows/shopping/labels.py:22` | Labeler → enqueue kind=overhead |
| `src/workflows/shopping/pipeline_v2.py:363, 487` | Pipeline stages → enqueue kind=overhead, next_task_spec=stage+1 |
| `src/infra/monitoring.py:134-142` | DELETE `run_monitoring_loop`; logic moves to mr_roboto executor |
| `src/app/run.py:439, 595, 628-655` | DELETE `_vector_maint_loop` and its `create_task`; keep snapshot_refresh (excluded by §3.7 of spec) |
| `packages/mr_roboto/src/mr_roboto/__init__.py` | Register new executors |

### Deleted files

| Path | Reason |
|---|---|
| `src/infra/monitoring.py` `run_monitoring_loop` (function only) | Replaced by mr_roboto `monitoring_check` executor |
| `src/app/run.py` `_vector_maint_loop` (function only) | Replaced by mr_roboto `vector_maint_*` executors |

### Test files

`tests/beckman/test_enqueue_contract.py`, `tests/beckman/test_continuations.py`, `tests/beckman/test_kind_column.py`, `tests/agents/test_checkpoint_resume.py`, `tests/integration/test_overhead_subtask_chain.py`, `tests/integration/test_telegram_await_inline.py`, `tests/integration/test_shopping_pipeline_chain.py`, `tests/integration/test_monitoring_check_executor.py`, `tests/integration/test_vector_maint_executor.py`, `tests/migration/test_dispatcher_alias_compat.py`.

---

## Phase 1 — Schema + Beckman Contract

### Task 1: Add `kind` column to `tasks`

**Files:**
- Modify: `src/infra/db.py` (init_db schema bootstrap)
- Test: `tests/beckman/test_kind_column.py` (new)

- [ ] **Step 1: Write failing test**
```python
# tests/beckman/test_kind_column.py
@pytest.mark.asyncio
async def test_tasks_table_has_kind_column(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db, connect_aux
    await init_db()
    async with connect_aux(str(tmp_path / "test.db"), _label="t") as db:
        cur = await db.execute("PRAGMA table_info(tasks)")
        cols = {row[1] for row in await cur.fetchall()}
    assert "kind" in cols

@pytest.mark.asyncio
async def test_kind_defaults_to_main_work(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db, add_task
    await init_db()
    tid = await add_task(title="t", description="d")
    # ... query kind, assert == 'main_work'
```

- [ ] **Step 2: Run test — must fail with no `kind` column**
- [ ] **Step 3: Implement** — idempotent `ALTER TABLE tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'main_work'` in schema bootstrap, mirroring existing `model_pick_log.provider` pattern.
- [ ] **Step 4: Run test — passes**
- [ ] **Step 5: Commit**: `feat(db): tasks.kind column for sub-task admission`

### Task 2: Widen `general_beckman.enqueue()` signature

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py:607-613`
- Modify: `src/infra/db.py` `add_task` to accept `kind`, `await_inline`, `on_complete`, `next_task_spec` (last three live as task `context` JSON fields, NOT new columns)
- Test: `tests/beckman/test_enqueue_contract.py` (new)

- [ ] **Step 1: Write failing tests**
```python
async def test_enqueue_accepts_kind():
    tid = await beckman.enqueue({"title":"t","description":"d","kind":"overhead"})
    # assert kind=overhead in DB row

async def test_enqueue_propagates_parent_id():
    parent = await beckman.enqueue({"title":"p","description":"d"})
    child = await beckman.enqueue({"title":"c","description":"d"}, parent_id=parent)
    # assert child.parent_task_id == parent

async def test_enqueue_stores_continuation_in_context():
    tid = await beckman.enqueue({...}, on_complete="agent.resume", next_task_spec={"title":"next"})
    # assert context.beckman.on_complete == "agent.resume"
    # assert context.beckman.next_task_spec == {"title":"next"}

async def test_enqueue_await_inline_blocks_until_terminal():
    # mock pump dispatch to fire after 100ms
    # call enqueue(await_inline=True)
    # assert returns TaskResult, time elapsed >= 100ms
```

- [ ] **Step 2: Run tests — must fail**
- [ ] **Step 3: Implement**
  - `enqueue()` signature: `enqueue(spec, *, parent_id=None, await_inline=False, on_complete=None, next_task_spec=None)`
  - Stash `on_complete` + `next_task_spec` into `spec["context"]["beckman"]` JSON envelope; pump's terminal handler reads them
  - When `await_inline=True`: create `asyncio.Future`; register in module-level `_inline_waiters: dict[task_id, Future]`; pump's terminal router resolves it. Block on future, return `TaskResult`. Add safety timeout (default 600s) to avoid permanent hang on bug.
  - `parent_id` flows into existing `add_task(parent_task_id=...)` parameter — already supported.
- [ ] **Step 4: Run tests — passes**
- [ ] **Step 5: Commit**: `feat(beckman): enqueue gains kind/parent/await_inline/on_complete/next_task_spec`

### Task 3: Continuation registry + chain dispatch

**Files:**
- New: `packages/general_beckman/src/general_beckman/continuations.py`
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (terminal router — `on_task_finished` or equivalent)
- Test: `tests/beckman/test_continuations.py` (new)

- [ ] **Step 1: Write failing tests** for handler registration, dispatch on terminal, error isolation (handler crash doesn't crash pump), `next_task_spec` chain with `parent_id` propagation.
- [ ] **Step 2-4: Implement and verify.** Registry = `dict[str, Callable[[int, dict], Awaitable[None]]]`. Boot-time registration. Pump's task-terminal hook reads `context.beckman.on_complete`, dispatches handler in detached `asyncio.create_task` so failure doesn't wedge pump. `next_task_spec` enqueued with `parent_id=this_task_id` post-handler.
- [ ] **Step 5: Commit**: `feat(beckman): on_complete + next_task_spec continuation dispatch`

---

## Phase 2 — Dispatcher Reshape

### Task 4: Dispatcher's public surface becomes deprecation alias

**Files:**
- Modify: `src/core/llm_dispatcher.py:80-413`
- Test: `tests/migration/test_dispatcher_alias_compat.py` (new)

- [ ] **Step 1: Write tests** — current callers using `dispatcher.request(category, ...)` must keep working unchanged. Result shape identical. Latency unchanged within margin.
- [ ] **Step 2: Run — passes today (no-op).** Implement alias.
- [ ] **Step 3: Implement**
  - Rename internal call path → `dispatch(task: BeckmanTask) -> TaskOutcome`
  - `request(...)` becomes a thin shim:
    ```python
    async def request(category, **kwargs):
        spec = _request_kwargs_to_spec(category, **kwargs)
        return await beckman.enqueue(
            spec, parent_id=None, await_inline=True
        )
    ```
  - All existing callers see no behavior change. parent_id stays None during this transition (proper parent_id wiring lands per-callsite in Phase 4).
- [ ] **Step 4: Run all existing dispatcher.request tests — must still pass.**
- [ ] **Step 5: Commit**: `refactor(dispatcher): request() becomes alias over beckman.enqueue(await_inline=True)`

### Task 5: Move pool_pressure + KDV.pre_call into Beckman admission

**Files:**
- Modify: `src/core/llm_dispatcher.py:240-291` (delete pool_pressure read + est_tokens shim)
- Modify: `packages/general_beckman/src/general_beckman/admission.py` (gain pool_pressure + KDV.pre_call calls)
- Modify: `src/core/llm_dispatcher.py:dispatch()` (assume admission already passed)

- [ ] **Step 1: Write tests** — admission gate rejects when pool_pressure says no; admission gate rejects when KDV.pre_call returns block; admission gate registers in_flight slot before invoking dispatcher.
- [ ] **Step 2-4: Implement.** Move ownership; dispatcher's `dispatch()` assumes invariant: admission already passed, in_flight registered, slot reserved. Delete dispatcher's defensive re-checks.
- [ ] **Step 5: Commit**: `refactor(beckman): admission owns pool_pressure + KDV.pre_call + in_flight`

---

## Phase 3 — Agent Checkpointable State

### Task 6: Serializable agent state + resume entrypoint

**Files:**
- New: `src/agents/checkpoint.py`
- Modify: `src/agents/base.py` (ReAct loop refactor)
- Test: `tests/agents/test_checkpoint_resume.py` (new)

- [ ] **Step 1: Write failing test** — agent runs N iterations, suspends on iteration K (sub-call enqueued with on_complete), checkpoint serialized to `tasks.context.agent_checkpoint`. Resume loads state, picks up at iteration K+1 with sub-call result spliced in. Final result identical to non-checkpointed run.
- [ ] **Step 2: Run — fails.**
- [ ] **Step 3: Implement**
  - `AgentCheckpoint` dataclass: `iteration_no, messages, tool_results_log, pending_sub_call_task_id, awaited_kind`
  - Serialize with size cap (~32KB compressed); messages compressed via existing message-history pruning utility
  - `BaseAgent.suspend(sub_call_task_id, kind)` → write checkpoint to current task's context, return `AgentSuspended` sentinel
  - `BaseAgent.resume(checkpoint, sub_result)` → splice `sub_result` into messages, continue loop from `iteration_no + 1`
  - Beckman calls `BaseAgent.resume()` via `on_complete="agent.resume"` handler registered in continuations registry
- [ ] **Step 4: Run — passes.**
- [ ] **Step 5: Commit**: `feat(agents): checkpointable ReAct state for sub-call awaits`

---

## Phase 4 — Migrate LLM Sites (continuation paths first)

### Task 7: Migrate sites 2-4 — agent overhead (structured_emit, alt-prompt-retry, self-reflection)

**Files:**
- Modify: `src/agents/base.py:3782, 3870, 3977`
- Test: `tests/integration/test_overhead_subtask_chain.py` (new)

- [ ] **Step 1: Write failing integration test** — agent runs to point of structured_emit; verify sub-task created with kind=overhead, parent_id=agent's task; agent suspends; sub-task admitted; result splices back; agent finalizes. Single end-to-end mission.
- [ ] **Step 2: Run — fails (alias still in place at site).**
- [ ] **Step 3: Implement** — replace `dispatcher.request(...)` with `beckman.enqueue(spec, parent_id=self.task_id, on_complete="agent.resume")` + `return AgentSuspended(...)` at all 3 sites.
- [ ] **Step 4: Run — passes.** Run regression on agent unit tests; verify no behavior delta on happy path.
- [ ] **Step 5: Commit**: `refactor(agents): structured_emit/alt-prompt/reflection enqueue as overhead sub-tasks`

### Task 8: Migrate site 5 — grader

**Files:**
- Modify: `src/core/grading.py:305`
- Test: extend `tests/integration/test_overhead_subtask_chain.py`

- [ ] Steps as above. Grader becomes overhead sub-task with parent_id=graded task. on_complete writes grade row + advances parent task's grading status.
- [ ] **Commit**: `refactor(grading): grader call enqueues as overhead sub-task`

### Task 9: Migrate site 6 — workflow summarizer hook

**Files:**
- Modify: `src/workflows/engine/hooks.py:46`
- Test: extend integration suite

- [ ] Steps as above. Hook returns `HookSuspended(sub_task_id)`; workflow engine's pump-side step-advance respects suspension. on_complete="hook.resume_post_execute" writes summarized artifact and advances workflow.
- [ ] **Commit**: `refactor(workflow): summarizer hook enqueues as overhead sub-task`

### Task 10: Migrate site 7 — vision tool

**Files:**
- Modify: `src/tools/vision.py:29`
- Test: extend integration suite (or skip-if-no-image-fixtures)

- [ ] Steps as above. Vision call → enqueue kind=tool_call, parent_id=agent.task_id. Agent suspends.
- [ ] **Commit**: `refactor(tools): vision enqueues as tool_call sub-task`

### Task 11: Migrate site 1 — agent ReAct main loop

**Files:**
- Modify: `src/agents/base.py:2498`
- Test: full agent regression

- [ ] **Note**: this is the parent-task call. Agent itself is already an admitted Beckman task. The dispatcher.request inside is what fires the LLM. After Task 4, this site already routes through `beckman.enqueue(await_inline=True)` via the alias. Goal of this task: replace the `await_inline=True` with proper continuation, so agent suspends instead of blocking pump-side.
- [ ] Implement suspend/resume with `agent.resume` handler.
- [ ] Run full agent test suite; verify no regression.
- [ ] **Commit**: `refactor(agents): ReAct main loop suspends on LLM await`

### Task 12: Migrate sites 11-14 — shopping pipeline

**Files:**
- Modify: `src/shopping/intelligence/_llm.py:42`
- Modify: `src/workflows/shopping/labels.py:22`
- Modify: `src/workflows/shopping/pipeline_v2.py:363, 487`
- Test: `tests/integration/test_shopping_pipeline_chain.py` (new)

- [ ] Pipeline stages restructured to next_task_spec chain. Each stage's enqueue includes next_task_spec describing stage+1. Beckman's terminal router enqueues stage+1 with parent_id=stage_n's task. Removes pipeline's in-coroutine sequential await.
- [ ] **Commit**: `refactor(shopping): pipeline stages chain via next_task_spec`

### Task 13: Migrate sites 8, 9, 10 — telegram + classifier (await_inline)

**Files:**
- Modify: `src/app/telegram_bot.py:4145, 4570`
- Modify: `src/core/task_classifier.py:258`
- Test: `tests/integration/test_telegram_await_inline.py` (new)

- [ ] Replace `dispatcher.request(...)` with `beckman.enqueue(spec, kind="chat"/"classifier", await_inline=True)`. Caller blocks on TaskResult, replies to telegram. Test asserts end-to-end ≤ 4s on idle pump.
- [ ] **Commit**: `refactor(telegram): chat + classifier enqueue with await_inline`

---

## Phase 5 — Mechanical Migration

### Task 14: Migrate `monitoring_check`

**Files:**
- New: `packages/mr_roboto/src/mr_roboto/executors/monitoring_check.py`
- Modify: `packages/general_beckman/src/general_beckman/cron_seed.py` (add marker)
- Modify: `packages/general_beckman/src/general_beckman/cron.py` (wire marker)
- Modify: `src/infra/monitoring.py` (delete `run_monitoring_loop`)
- Modify: `src/app/run.py:439` (delete `create_task(run_monitoring_loop())`)
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (register executor)
- Test: `tests/integration/test_monitoring_check_executor.py` (new)

- [ ] Cron seed: `monitoring_check` every 300s (env-overridable via existing `MONITOR_INTERVAL`).
- [ ] Mr. Roboto executor reads MONITOR_URLS + MONITOR_GITHUB_REPOS, performs checks, enqueues per-target `notify_user` mechanical sub-tasks (parent_id = monitoring_check task) when alert detected.
- [ ] Test: simulate URL down + GitHub release; verify `notify_user` sub-tasks enqueued with correct payload; verify no direct `tg.send_notification` call remains.
- [ ] **Commit**: `refactor(monitoring): cron-seeded monitoring_check, alerts via notify_user sub-tasks`

### Task 15: Migrate `vector_maint_*`

**Files:**
- New: `packages/mr_roboto/src/mr_roboto/executors/vector_maint.py`
- Modify: cron_seed.py + cron.py (markers `vector_maint_wal` 1800s, `vector_maint_snapshot` 86400s)
- Modify: `src/app/run.py:628-655` (delete `_vector_maint_loop` + `create_task`)
- Test: `tests/integration/test_vector_maint_executor.py` (new)

- [ ] Mr. Roboto executor wraps ChromaDB ops in `loop.run_in_executor(...)` so pump's event loop is not blocked. Mission 46 incident regression test: simulate slow ChromaDB sync; verify pump continues dispatching during.
- [ ] **Commit**: `refactor(memory): cron-seeded vector_maint via mr_roboto, fixes event-loop wedge`

---

## Phase 6 — Residue Cleanup

### Task 16: Delete dispatcher.request alias

**Files:**
- Modify: `src/core/llm_dispatcher.py` — delete `request()`
- Modify: any remaining importer (sweep with grep)
- Delete: `tests/migration/test_dispatcher_alias_compat.py` (obsolete)

- [ ] Grep `dispatcher\.request|get_dispatcher\(\)\.request` — must be zero hits in production code.
- [ ] Delete alias.
- [ ] **Commit**: `chore(dispatcher): delete request() alias, all callers migrated`

### Task 17: Delete `current_task_id` ContextVar

**Files:**
- Modify: `src/core/heartbeat.py` — delete `current_task_id`
- Modify: `src/core/orchestrator.py:256` — delete `_hb.current_task_id.set(...)`
- Grep + sweep for any other reader

- [ ] Migration replaced piggy-back with explicit `parent_id`. ContextVar must have zero readers.
- [ ] **Commit**: `chore(heartbeat): delete current_task_id contextvar, parent_id is explicit`

### Task 18: Delete est_tokens shim from commit `5f7f905`

**Files:**
- Modify: `src/core/llm_dispatcher.py:268-291` — delete shim that wired est_tokens for non-Beckman paths

- [ ] Beckman admission now owns est_tokens calc. Dispatcher's shim is redundant.
- [ ] **Commit**: `chore(dispatcher): drop est_tokens shim, admission owns it`

### Task 19: Final audit pass

- [ ] Grep `asyncio.create_task` in `src/`, `packages/` — every fire-and-forget background loop must justify itself per spec §3.7 or be migrated.
- [ ] Grep `while True` + `asyncio.sleep` — same.
- [ ] Verify 3 documented exclusions still match: NerdHerd snapshot 2s, heartbeat 15s, DaLLaMa subprocess.
- [ ] Document any newly-discovered residue in handoff for follow-up.
- [ ] **Commit**: `docs: post-migration audit, exclusions confirmed`

---

## Validation

- [ ] Run full pytest with timeout: `timeout 120 pytest tests/`
- [ ] Run targeted Beckman + agent + integration: `timeout 60 pytest tests/beckman tests/agents tests/integration`
- [ ] Run stateful sim: `python packages/fatih_hoca/tests/sim/run_scenarios.py`
- [ ] Run swap-storm check: `python packages/fatih_hoca/tests/sim/run_swap_storm_check.py`
- [ ] Live mission test: enqueue a 3-step coding mission via Telegram, verify all sub-calls (grader, structured_emit) appear in DB as proper Beckman tasks with parent_id, verify no `dispatcher.request` log lines.
- [ ] Telegram chat latency probe: send 10 casual messages, verify p50 ≤ 4s, p95 ≤ 6s.
- [ ] Monitoring check probe: artificially break MONITOR_URL, verify `notify_user` sub-task enqueued within 5min, verify no direct `tg.send_notification` call.
- [ ] Vector maint probe: trigger WAL checkpoint, verify pump continues admitting during 30s+ ChromaDB op.
- [ ] DLQ visibility probe: induce 5x grader failure, verify parent task DLQ entry surfaces grader sub-task chain.

## Rollback

Each phase is independently reversible because:
- Phase 1 schema add is idempotent and column has default
- Phase 2 alias preserves caller API; revert restores direct dispatcher path
- Phase 3 checkpoint module is opt-in; agent base.py revert restores in-coroutine state
- Phase 4 each callsite revertible in isolation
- Phase 5 mechanical loops can be re-added; cron seeds idempotent-removable
- Phase 6 cleanup must wait until 4-5 are stable in production for ≥1 week

If a phase fails in production, revert that phase's commits; system stays runnable from any prior phase's HEAD.

## Pitfalls (from spec §6)

- Agent checkpoint serialization size — compression mandatory
- Continuation handler registry late-binding — boot-check coverage
- Pump 1-task-per-tick under overhead load — measure, may force out-of-scope drain step
- Re-entry deadlock on saturated GPU — admission must discount sleeping parents
- Telegram +3s latency — measure, accept or revisit await_inline strategy
- Workflow hook self-blocking — on_complete only, never await_inline
- Old enqueue() callers' kwargs widening — backward-compatible defaults
- `agent_type="mechanical"` vs `kind="mechanical"` — consolidate as separate cleanup commit if needed

## Open questions tracked

See spec §7. Plan does not block on them. Each can be addressed post-migration as separate small specs:

1. Per-kind retry policy
2. Sticky-model bias plumbing for overhead
3. DLQ filtering by kind
4. await_inline timeout strategy

## Done criteria

- All 16 LLM callsites enqueue, none call dispatcher directly
- 2 mechanical loops migrated to cron + mr_roboto
- Dispatcher `request()` alias deleted
- `current_task_id` ContextVar deleted
- est_tokens shim from `5f7f905` deleted
- 3 exclusions documented + verified
- Live mission round-trip clean
- Telegram latency within budget
- Stateful sim passing
- DLQ surfaces parent→child task chains
