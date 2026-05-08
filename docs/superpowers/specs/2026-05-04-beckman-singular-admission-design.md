# Beckman — Singular Admission

**Date:** 2026-05-04
**Status:** Design — pending plan
**Predecessors:**
- `docs/handoff/2026-05-04-unify-non-beckman-paths.md` (LLM dispatcher inventory)
- `docs/superpowers/specs/2026-04-18-phase2b-general-beckman-design.md` (Beckman as task master)
- `docs/superpowers/specs/2026-04-29-pool-pressure-utilization-equilibrium-design.md` (admission pressure model)

**Touchpoints:** `packages/general_beckman/`, `src/core/llm_dispatcher.py`, `src/core/orchestrator.py`, `src/agents/base.py`, `src/core/grading.py`, `src/workflows/engine/hooks.py`, `src/tools/vision.py`, `src/app/telegram_bot.py`, `src/core/task_classifier.py`, `src/shopping/intelligence/_llm.py`, `src/workflows/shopping/labels.py`, `src/workflows/shopping/pipeline_v2.py`, `src/infra/monitoring.py`, `src/app/run.py`, `packages/mr_roboto/`, `src/infra/db.py`

---

## 1. Principle

**Beckman is the brain.** Every tick, Beckman evaluates its task queue and decides what to dispatch. No path circles outside Beckman. Whatever does = residue of extraction, must be migrated.

Every LLM call and every mechanical job is a task. Caller's job ends at `enqueue`. Beckman's pump owns dispatch timing, capacity gating, model selection, retry policy, DLQ admission. Dispatcher (`src/core/llm_dispatcher.py`) is preserved as Beckman's worker — never as a public entry.

This spec covers two convergent migrations:

1. **LLM admission**: 16 `dispatcher.request()` callsites → `beckman.enqueue()`
2. **Mechanical admission**: 2 background loops not yet routed through Beckman's cron → cron-seeded mechanical tasks

Three justified exclusions are documented so future audits don't re-litigate.

## 2. Out of scope

The following are NOT in this design — accepted as current behavior:

- **Wake-on-enqueue**: pump stays at 3s `asyncio.sleep`-based polling (`src/core/orchestrator.py:372`). Continuations and `await_inline` paths eat up to 3s pump-tick latency. Acceptable.
- **Multi-dispatch per tick**: `next_task()` returns one task per pump iteration. Throughput cap stays at 1 admission / 3s. Capacity-driven drain loop deferred.
- **Beckman re-entry from sub-process / sidecar contexts**: only orchestrator's pump is a Beckman dispatcher. Yaşar Usta and DaLLaMa subprocess have their own watchdogs.

These are independent throughput improvements. This spec changes admission topology, not pump cadence.

## 3. Architecture

### 3.1 Roles after migration

| Component | Role |
|-----------|------|
| **Caller (any code that wants LLM/mechanical work done)** | Builds `TaskSpec`, calls `beckman.enqueue(spec, parent_id=…, await_inline=False)`, returns. Never imports dispatcher. Never calls KDV. Never touches in_flight. |
| **Beckman pump** | Reads `next_task()` per tick. Per task: runs admission (eligibility + pool_pressure + KDV.pre_call + fatih_hoca.select via `next_task()`'s existing path), reserves slot, fires dispatcher. |
| **Dispatcher (`llm_dispatcher.py`)** | Beckman's LLM worker. Receives a Beckman-admitted task. Owns: DaLLaMa load, KDV pre_call hook (provider-side), nerd_herd state push, HaLLederiz Kadir invocation, in-attempt retry loop, mid-attempt model switch via fatih_hoca, return result to Beckman. **Not a public surface.** |
| **Mr. Roboto (`packages/mr_roboto/`)** | Beckman's mechanical worker. Receives `agent_type="mechanical"` admitted tasks. Routes to executor (workspace_snapshot, git_commit, clarify, notify_user, monitoring_check, vector_maint_*). |
| **Cron (`packages/general_beckman/cron.py`)** | Beckman's clock. Fires due markers from `scheduled_tasks` into the queue as concrete mechanical tasks. |

### 3.2 Single enqueue contract

```python
# packages/general_beckman/__init__.py
async def enqueue(
    spec: dict,
    *,
    parent_id: int | None = None,
    await_inline: bool = False,
    on_complete: str | None = None,
    next_task_spec: dict | None = None,
) -> int | TaskResult:
    """Single admission entry. Caller never decides dispatch timing.

    spec: same shape `add_task()` already accepts —
        title, description, agent_type, kind, mission_id, priority,
        depends_on, context, plus new optional fields:
            estimated_input_tokens, estimated_output_tokens,
            difficulty, call_category (for kind="llm"),
            executor, payload (for kind="mechanical")

    parent_id: when this task is a child of another (e.g. agent's
        grader sub-call, workflow hook). Beckman's accounting rolls
        token cost up to parent for pool_pressure overlay; DLQ
        forensics surface parent→child chain. Replaces
        current_task_id ContextVar piggy-back.

    await_inline: when True, caller blocks on a future bound to
        the task row's terminal status. Reserved for high-urgency
        direct-conversation callers (telegram user-facing). Eats
        up to 3s pump-tick latency. Default False.

    on_complete: callback registry key. When task terminates,
        Beckman fires the named handler with (task_id, result).
        Used by workflow engine to advance phases.

    next_task_spec: chain a follow-up task on success. Beckman
        enqueues it with parent_id=this_task. Used by shopping
        pipeline stages.

    Returns:
        int (task_id) when await_inline=False
        TaskResult when await_inline=True
    """
```

`reserve_task` and the existing `enqueue` external-write path collapse into this single function. `add_task` stays as the DB-row helper Beckman calls internally.

### 3.3 Task `kind` column

`tasks` table gains `kind` column (TEXT, NOT NULL DEFAULT 'main_work'):

| `kind` value | Meaning | Source |
|--------------|---------|--------|
| `main_work` | Today's user-/mission-initiated work | unchanged |
| `overhead` | Within-task LLM helpers (grader, structured_emit, summarizer, reflection, alt-prompt retry) | new |
| `chat` | Telegram direct user conversation | new |
| `classifier` | Pre-task message classifier | new |
| `tool_call` | Agent tool dispatch (vision, etc.) | new |
| `mechanical` | Mr. Roboto-routed (existing `agent_type="mechanical"` consolidates here) | new |

`kind` distinguishes accounting & DLQ visibility. All kinds share the same admission lifecycle — pool_pressure check, fatih_hoca.select (LLM kinds only), in_flight registration, worker_attempts, DLQ-on-exhaust.

`parent_id` column already exists as `parent_task_id` (`add_task` signature). Reused — no schema change.

### 3.4 Caller patterns

**Continuation (default — agents, hooks, shopping stages, classifier→task creation, monitoring loop)**

```python
# inside agent ReAct loop
task_id = await beckman.enqueue(
    {
        "title": "grader",
        "description": ...,
        "agent_type": "reviewer",
        "kind": "overhead",
        "estimated_input_tokens": 1200,
        "estimated_output_tokens": 400,
    },
    parent_id=current_task_id,
    on_complete="agent.resume_after_grade",
)
return AgentSuspended(awaiting=task_id)
```

Agent returns. Beckman's pump dispatches grader when capacity allows. On completion, `agent.resume_after_grade(task_id, result)` re-enqueues parent agent's continuation. Agent state must be checkpointable for this pattern — see §4.2.

**Direct (telegram user-facing chat / classifier)**

```python
result = await beckman.enqueue(
    {
        "title": "telegram-chat",
        "description": user_msg,
        "agent_type": "assistant",
        "kind": "chat",
        ...
    },
    parent_id=None,
    await_inline=True,
)
await update.message.reply_text(result.content)
```

`await_inline=True` blocks caller. Caller must already be in a context where blocking is OK (telegram user is waiting on chat).

### 3.5 Dispatcher's preserved role

Per user direction, dispatcher is NOT removed. Becomes Beckman-internal worker, signature changes to:

```python
# src/core/llm_dispatcher.py
async def dispatch(task: BeckmanTask) -> TaskOutcome:
    """Beckman's LLM worker. Called from Beckman pump after admission.

    task: fully-admitted Beckman task — slot reserved, model selected
        by fatih_hoca, KDV.pre_call passed.

    Pre-call ops:
        - DaLLaMa load (if local)
        - nerd_herd state push (call begin)
        - logging open

    Call:
        - HaLLederiz Kadir invocation
        - in-attempt retry loop (max_recursion=5 stays here, internal)
        - mid-attempt model switch via fatih_hoca on failure category

    Post-call:
        - nerd_herd state push (call end)
        - log close
        - return outcome to Beckman (Complete / Failed / Exhausted / etc.)
    """
```

`request()` becomes deprecated alias for one release cycle: internally calls `beckman.enqueue(..., await_inline=True)` for migration safety. Removed at end of migration.

### 3.6 Mechanical migration

Two background loops gain Beckman cron entries:

**`monitoring_check` — replaces `src/infra/monitoring.py:run_monitoring_loop`**

- Cron seed in `cron_seed.py`: every 300s (env-overridable to keep `MONITOR_INTERVAL` semantics)
- Mr. Roboto executor `monitoring_check`: hits MONITOR_URLS + MONITOR_GITHUB_REPOS, returns alert payload
- On alert detection, mr_roboto enqueues per-target `notify_user` mechanical sub-tasks (parent_id = monitoring_check task)
- Direct `tg.send_notification` calls inside monitoring loop deleted

**`vector_maint_wal` + `vector_maint_snapshot` — replaces `src/app/run.py:_vector_maint_loop`**

- Cron seeds: `vector_maint_wal` every 1800s, `vector_maint_snapshot` every 86400s
- Mr. Roboto executors wrap ChromaDB ops in `loop.run_in_executor(...)` so pump's event loop is not blocked (this also fixes mission 46 incident where 120s sync I/O wedged dispatch)
- `_vector_maint_loop` deleted

Both join existing cron-seeded jobs (sweep, price_watch, kdv_persist, cloud_refresh, todo_reminder, nerd_herd_health, hoca_benchmark, btable_rollup) — no new mechanism.

### 3.7 Justified exclusions (stay outside Beckman)

Recorded so future audits don't re-litigate:

| Loop | File:line | Why excluded |
|------|-----------|--------------|
| NerdHerd snapshot refresh (2s) | `src/app/run.py:595` | Admission system's own input cache. Can't gate on itself — Beckman would deadlock waiting on its own snapshot. |
| Orchestrator heartbeat (15s) | `src/core/orchestrator.py:_heartbeat_loop` | Yaşar Usta's hung-orchestrator detector. Chicken-egg: would need orchestrator pump running to dispatch its own heartbeat. |
| DaLLaMa subprocess watchdog | `packages/dallama/` | Owned by llama-server subprocess lifecycle, not orchestrator's process. |

These are infrastructure-internal; not "tasks" by any meaningful definition.

## 4. Migration

### 4.1 LLM callsite table (16 sites)

Pulled from `docs/handoff/2026-05-04-unify-non-beckman-paths.md` + this design's continuation/direct decision per user direction (agents = helpers, idle is fine, only telegram is direct).

| # | Site | Today | After | Direct? | parent_id source |
|---|------|-------|-------|---------|------------------|
| 1 | `agents/base.py:2498` | dispatcher.request (ReAct main) | enqueue kind=main_work | NO | self.task_id |
| 2 | `agents/base.py:3782` | structured_emit | enqueue kind=overhead | NO | self.task_id |
| 3 | `agents/base.py:3870` | alt-prompt retry | enqueue kind=overhead | NO | self.task_id |
| 4 | `agents/base.py:3977` | self-reflection | enqueue kind=overhead | NO | self.task_id |
| 5 | `core/grading.py:305` | grader | enqueue kind=overhead | NO | task_id arg |
| 6 | `workflows/engine/hooks.py:46` | summarizer hook | enqueue kind=overhead | NO | hook context task_id |
| 7 | `tools/vision.py:29` | vision tool | enqueue kind=tool_call | NO | self.task_id (agent caller) |
| 8 | `telegram_bot.py:4145` | router classifier | enqueue kind=classifier | YES | None |
| 9 | `telegram_bot.py:4570` | casual chat | enqueue kind=chat | YES | None |
| 10 | `core/task_classifier.py:258` | pre-task classifier | enqueue kind=classifier | YES | None |
| 11 | `shopping/intelligence/_llm.py:42` | shopping helper | enqueue kind=overhead | NO | parent shopping task_id |
| 12 | `workflows/shopping/labels.py:22` | labeler | enqueue kind=overhead | NO | parent shopping task_id |
| 13 | `workflows/shopping/pipeline_v2.py:363` | pipeline stage | enqueue kind=overhead, next_task_spec=stage+1 | NO | parent shopping task_id |
| 14 | `workflows/shopping/pipeline_v2.py:487` | pipeline stage | enqueue kind=overhead, next_task_spec=stage+1 | NO | parent shopping task_id |
| 15 | `app/run.py` startup banner check | (if exists) | direct dispatcher.dispatch — not migrated, internal | — | — |
| 16 | (placeholder for any site discovered during impl) | | | | |

Direct sites: 8, 9, 10. Three.

### 4.2 Agent continuation — checkpointable state

Today's ReAct agents hold state in coroutine local. After migration, agents must serialize state at suspension and resume from it.

State to checkpoint per ReAct iteration:
- `iteration_no`
- `messages_so_far` (or compressed summary)
- `tool_results_log`
- pending tool call (if grader was awaiting)
- task_id, mission_id

Stored in `tasks.context` JSON column. On `on_complete` callback, Beckman re-enqueues a new `agent.resume` task with `context=checkpoint`. Resumed agent loads state, processes the awaited result, continues loop or terminates.

This is a real refactor of `src/agents/base.py`. Likely 200-400 LOC delta. Spec'd in implementation plan, not here.

### 4.3 Migration order

Ship-able increments. Each step shippable; system stays runnable.

1. **Schema**: add `kind` column to `tasks` (default `main_work`). Idempotent migration in `src/infra/db.py` schema bootstrap. No behavior change.
2. **Beckman enqueue contract**: extend `enqueue()` signature with `kind`, `parent_id`, `await_inline`, `on_complete`, `next_task_spec`. Backward-compatible — existing callers see no change.
3. **Dispatcher rewrite (internal)**: rename `request()` → `dispatch()`. Add `request()` as deprecation alias that calls `beckman.enqueue(await_inline=True)`. All current callers keep working.
4. **Workflow `on_complete` registry**: adds named-handler dispatch for continuation paths. Used by workflow engine + shopping pipeline.
5. **Agent checkpointable state**: refactor `base.py` to serialize/resume around sub-task awaits. Largest piece.
6. **Migrate sites 1-7 (agent + grader + hooks + vision)**: LLM overhead → enqueue kind=overhead with parent_id. Dispatcher request() alias still used internally during transition.
7. **Migrate sites 8-10 (telegram + classifier)**: enqueue kind=chat/classifier with await_inline=True. Direct path.
8. **Migrate sites 11-14 (shopping)**: enqueue kind=overhead with continuation chain. Pipeline stages restructured to next_task_spec chains.
9. **Mechanical migration**: cron seeds for `monitoring_check`, `vector_maint_wal`, `vector_maint_snapshot`. Mr. Roboto executors. Old loops deleted.
10. **Residue cleanup** (§5): delete dispatcher.request() alias, delete current_task_id contextvar, delete est_tokens shim from `5f7f905`, delete dispatcher's pool_pressure/KDV.pre_call calls (now Beckman's job).

## 5. Residue cleanup (final step)

After all callsites migrated:

- `src/core/llm_dispatcher.py:request()` — deleted
- `src/core/heartbeat.py:current_task_id` ContextVar — deleted (parent_id is now explicit)
- `src/core/llm_dispatcher.py:268-291` est_tokens plumbing for non-Beckman paths (commit `5f7f905`) — deleted (admission owns it)
- `src/core/llm_dispatcher.py` pool_pressure read — deleted (Beckman's admission owns it)
- KDV.pre_call call from dispatcher — moved to Beckman admission
- 5 shim modules in `src/core/router.py`, `src/models/model_registry.py`, etc. — re-eval whether anyone still imports them
- `src/infra/monitoring.py:run_monitoring_loop` — deleted
- `src/app/run.py:_vector_maint_loop` — deleted

## 6. Pitfalls

- **Agent checkpoint serialization size**: full message history per task can balloon `tasks.context`. Mitigate with compression / message-history pruning at suspension boundary.
- **Continuation handler registry must be deterministic**: `on_complete="agent.resume_after_grade"` requires Beckman to resolve string → handler at boot. Late-binding bugs surface as silent task drops. Bootcheck registry coverage.
- **Pump's 1-task-per-tick cap**: with overhead tasks now flowing through pump, throughput becomes a bottleneck. 32k+ main_work + 8k+ overhead per week = ~57 tasks/min sustained. At 1 / 3s = 20 / min ceiling. **Spec assumes pump cadence is fine, but this is the first place it might not be.** Multi-dispatch drain stays out of scope here but is implied as next phase.
- **Re-entry deadlock (parent awaits child, GPU saturated)**: parent task holds in_flight slot. Child overhead task can't admit because pool_pressure says "full." Beckman must recognize "this enqueue is from inside an active dispatch, parent is now sleeping" — parent's GPU slot transferable to child. Implementation = check `parent_id in active_dispatches` and discount parent from pool count.
- **Telegram await_inline + 3s pump latency**: telegram chat reply now eats up to 3s admission delay on top of LLM call. Total user-perceived latency rises. Acceptable per scope, but flag for measurement post-ship.
- **Workflow hook self-blocking**: post-execute hooks that fire continuation tasks must not await them in the parent workflow's dispatch context. on_complete pattern, not await_inline.
- **Old `enqueue()` callers**: 4-5 places call `general_beckman.enqueue({...})` today (Telegram bot commands, mission planner, mission completion). Their signature widens; existing callsites stay valid because new params are kwargs with defaults.
- **`agent_type="mechanical"` vs `kind="mechanical"`**: today mr_roboto's branch is `agent_type == "mechanical"`. After migration, `kind="mechanical"` is the source of truth, `agent_type` becomes informational (which executor). Consolidate.

## 7. Open questions

1. **Dispatch timing predictability for `await_inline`**: caller can't know how long Beckman holds a task before admitting. For telegram chat that's fine (worst case 3s). For other call patterns, would `await_inline` need a timeout? Currently no — caller just blocks. Defer until a use case demands.
2. **Per-kind retry policy**: today `worker_attempts` ladder is one-size. Should `kind="overhead"` get a tighter cap (3 vs 5) since it's lower-value? Or `kind="chat"` get faster fail (1 attempt) since user is waiting? Defer to plan.
3. **Sticky-model bias for overhead**: today fatih_hoca's stickiness keeps overhead on the loaded model. With overhead now going through proper admission, stickiness signal must propagate (parent's model = preferred for child overhead). Plumb `parent_model` into TaskSpec.
4. **DLQ noise from mechanical**: monitoring_check + vector_maint will surface in DLQ on failure. New noise floor. Probably want `kind="mechanical"` filtered out of `/dlq` default view.

## 8. Test plan

- New unit: `beckman.enqueue` with each `kind`, parent_id, await_inline, on_complete, next_task_spec.
- New integration: agent ReAct round-trip with grader as overhead sub-task. Verify checkpoint→resume.
- New integration: telegram chat with await_inline. Verify ≤ 4s end-to-end on idle pump.
- New integration: shopping pipeline 3-stage chain via next_task_spec. Verify all stages run, parent_id wired.
- New integration: monitoring_check cron fires; mr_roboto executor enqueues notify_user sub-task on simulated alert.
- Stateful sim: `packages/fatih_hoca/tests/sim/run_scenarios.py` — verify pool_pressure with overhead admissions doesn't starve main_work.
- DLQ visibility test: induce 5x grader failures, verify parent task surfaces with grader sub-task chain in `/dlq`.
- Migration safety: run with old dispatcher.request() alias for one cycle, verify behavior identical.

## 9. Plan deliverable

Implementation plan splits into 10 sequential commits matching §4.3 migration order. Each commit shippable independently. Plan written separately at `docs/superpowers/plans/2026-05-04-beckman-singular-admission.md` after spec acceptance.
