# General Beckman — Simplification & Task 13 Scope

**Date:** 2026-04-19
**Parent plan:** `docs/superpowers/plans/2026-04-19-phase2b-task13-handoff.md`
**Supersedes scope parts of:** `docs/superpowers/specs/2026-04-18-phase2b-general-beckman-design.md`

## Why this spec exists

Phase 2b landed a transitional Beckman with lanes, a grab-bag `tick()`, and lifecycle handlers that still delegate back to `Orchestrator._handle_*` via a registered reference. Task 13 (main-loop rewrite + handler consolidation) was deferred. Before writing the migration plan, we need a clean picture of what Beckman *is* and what it is not — otherwise Task 13 pattern-matches the messy intermediate state.

This spec defines the end-state boundaries. The migration strategy is a separate brainstorm that follows this spec's approval.

## Guiding principle

> **Orchestrator orchestrates. Beckman owns tasks. Hoca owns models. Nerd Herd owns utilization. Workflow engine owns recipes.**

Each subsystem minds its own domain. Imports flow one way. No shared registries, no pub/sub, no handler tables wired across packages.

## Beckman's public API (the whole surface)

Three methods. Everything else is internal.

```
next_task() -> Task | None
on_task_finished(task_id: int, result: AgentResult) -> None
enqueue(spec: TaskSpec) -> Task
```

- `next_task()` — called by orchestrator on its 3s cycle. Does opportunistic sweep + due-cron firing + pick. Returns one dispatchable task, or `None`.
- `on_task_finished(id, result)` — called by orchestrator when a dispatch completes. Marks terminal + applies result-driven task creation (subtasks, retries, clarify/notify, grader, workflow advance — all via internal rules).
- `enqueue(spec)` — user/bot-initiated writes (`/task`, `/shop`). The one external create path.

No `tick()`, no `sweep_queue()`, no `create_retry()` / `create_clarify()` / `create_subtasks()` / `advance_mission()` / `spec_for_*` on the public surface.

### Naming note

`next_task()` keeps its name. The docstring is honest about the hidden work (sweep + cron + pick). Orchestrator readers get the correct mental model from `task = await beckman.next_task()`.

## What lives inside `next_task()`

1. **Opportunistic queue sweep** (throttled — e.g., at most every 5 minutes per orchestrator cycle). Replaces the old `check_stuck_tasks`. Handles:
   - Processing tasks stuck > 5min → infra-reset up to N, then fail
   - Ungraded tasks stuck > 30min → promote to completed (safety net)
   - Pending tasks with all deps failed → cascade fail (unless any dep is in DLQ)
   - `waiting_subtasks` with all children terminal → mark complete/failed
   - Pending tasks with `next_retry_at` > 1h in past → clear the gate
   - `waiting_human` escalation tiers (4h nudge / 24h tier 1 / 48h tier 2 / 72h cancel) — each tier *creates* a `salako.notify_user` task (Telegram sends no longer inline)
   - Workflow-level timeouts → mission paused + `salako.notify_user` task

2. **Cron fire.** Reads due rows from the unified `scheduled_tasks` table (see below). For each due row, inserts a concrete task (the "payload" from the cron row becomes the new task's spec). Advances `next_run`.

3. **Pick one dispatchable row.**
   - Saturation check via `nerd_herd.snapshot()` — single boolean "system busy right now?" bit. No lanes, no partitioning.
   - Age-based priority boost (+0.1/h, cap +1.0) applied in the query or post-filter.
   - Paused-pattern filter (DLQ category skip) — state moves from `Orchestrator.paused_patterns` to Beckman module state.
   - Claims the row (status → processing) and returns it.

4. **Internal throttles** — `_last_sweep`, `_last_cron_scan`. Sweep and cron don't run on every call; the pick does.

## What lives inside `on_task_finished(task_id, result)`

```
1. _load(task_id)
2. actions = result_router.route(task, result)   # existing Action schema, kept
3. actions = _rewrite_actions(task, actions)     # policy rules (see below)
4. await _apply_actions(task, actions)           # action → DB rows
5. await _mark_terminal(task_id)
```

### The Action schema stays

`result_router.py` (and its Action dataclasses — `SubtaskEmission`, `ClarifyRequest`, `NotifyUser`, `RetryRequest`, `MissionAdvance`, etc.) is the structured shape Beckman's internal logic keys off. It moves into Beckman's internals but the schema survives.

### Action-rewriting rules (`_rewrite_actions`)

Pure function. Replaces `result_guards.py`. Policy-only, no I/O.

| Rule | Effect |
|---|---|
| Task is a workflow step and result emits subtasks | Replace `SubtaskEmission` with `RetryRequest(quality)` — workflow steps must produce artifacts, not decompose |
| Task is `silent` and result requests clarification | Drop `ClarifyRequest`; insert a terminal-fail action |
| Task has `may_need_clarification=false` and result requests clarification | Replace `ClarifyRequest` with `RetryRequest(quality)` |
| Task has existing `clarification_history` and result requests clarification | Drop `ClarifyRequest`; replace with `CompleteWithReusedAnswer` (the stored Q/A becomes the result) |

Schema validation for workflow clarifications lives in the workflow engine's post-hook, not here (see workflow section).

### Action application (`_apply_actions`)

Small dispatcher keyed by action type. Each branch inserts DB rows.

| Action | Row(s) inserted |
|---|---|
| `SubtaskEmission` | Child task rows with `parent_task_id` set |
| `RetryRequest(category)` | New pending row with bumped attempt count (retry policy resolves: insert or DLQ-write) |
| `ClarifyRequest` | `salako.clarify` mechanical task |
| `NotifyUser` | `salako.notify_user` mechanical task |
| `MissionAdvance` | Single mechanical task of executor `workflow_advance` (see workflow section) |
| `Grading` | Grader task (existing agent type) |
| `Complete` / `CompleteWithReusedAnswer` | No new row; marks original complete |
| `Fail` | No new row; marks original failed |

### Retry policy

Internal to Beckman. Shared helper replaces `_quality_retry_flow`:

- Builds a `RetryContext` from the task.
- Decides `terminal` / `immediate` / `delayed` based on attempt count, category, and bonus-attempt heuristic.
- **Bonus-attempt heuristic** (quality retries with ≥ 50% assessed progress get +1 attempt, capped at 2) stays — it solves real DLQ-too-eagerly incidents — but gets **flagged for a sideways look** during migration. If the progress assessment (`_assess_timeout_progress`) is fragile or poorly covered, either harden it or drop the heuristic in a follow-up. Not a Task 13 blocker.
- DLQ writes (to `dead_letter_tasks`) happen directly here. DLQ is record state, not new work — no DLQ task is spawned.
- DLQ notification → spawn a `salako.notify_user` task. No inline Telegram send.

## Cron — unified table, interval or cron expression

### Schema touch-up

`scheduled_tasks` gains support for both cron expressions (existing user-facing use) and simple interval-based internal cadences. Minimal schema additions:

| Column | Purpose |
|---|---|
| `interval_seconds` (nullable) | Simple interval mode. Mutually exclusive with `cron_expression`. |
| `kind` (`internal` / `user`) | Soft tag; internal rows are seeded at startup and not user-editable. |
| Existing columns retained | `cron_expression`, `payload`, `next_run`, `last_run`, … |

### Seeded internal cadences

Beckman lazy-seeds (upsert-by-name) internal cron rows on the first call to `next_task()` — guarded by a module-level `_seeded` flag so the seed happens exactly once per process. No separate `init()` method is added to the public API.

| Name | Interval | Payload |
|---|---|---|
| `queue_sweep` | 300s | Marker — Beckman's cron processor recognizes this and triggers the internal sweep directly rather than inserting a task |
| `todo_reminder` | 7200s | Mechanical task: `todo_reminder` executor |
| `daily_digest` | 86400s (fire-time aligned to local 12:00) | Mechanical task: `daily_digest` executor |
| `api_discovery` | 86400s | Mechanical task: `api_discovery` executor |
| `benchmark_refresh` | 300s | Marker — delegates to `hoca.refresh_benchmarks_if_stale()` |

"Marker" rows let internal system work (sweep, benchmark refresh) ride the same cadence table without creating actual task rows. Their payload is a sentinel the cron processor dispatches internally instead of inserting a task.

### Firing

Single loop inside `next_task()`:

```python
for row in due_cron_rows():
    if row.payload.get("_marker") == "sweep":
        await self._sweep_queue_if_due()
    elif row.payload.get("_marker") == "benchmark_refresh":
        await hoca.refresh_benchmarks_if_stale()
    else:
        await self._insert_cron_task(row.payload)
    row.advance()
```

## Workflow engine — its own package

Workflow engine is its own domain (recipes, phases, artifact schemas, skip_when, step metadata). It does **not** live in Beckman, the orchestrator, or salako.

### Home: `packages/workflow_engine/` (new, or promoted from `src/workflows/`)

Exports one primary entry:
```
workflow_engine.advance(mission_id, completed_task_id, result) -> AdvanceResult
```

`AdvanceResult` carries:
- artifact captures (from `guard_pipeline_artifacts`)
- post-hook status flip (from `guard_workflow_step_post_hook` / `guard_ungraded_post_hook` — **collapsed into one** hook since their logic is identical)
- next-phase TaskSpecs

### Invocation: via a salako executor

Salako gains **one** executor: `workflow_advance`. It is a thin delegator — calls `workflow_engine.advance(...)` and shapes the result into the standard envelope Beckman's `result_router` understands (Actions: artifact capture = side-effect, status flip = the advance task's own status, next-phase specs = `SubtaskEmission`).

### Flow

```
worker task #500 (agent=coder, mission=M) completes
  → Orchestrator.dispatch calls beckman.on_task_finished(500, result)
     → Beckman's rule: "mission task completed cleanly" → emit MissionAdvance action
     → _apply_actions inserts one mechanical task: {executor: "workflow_advance",
                                                    payload: {mission_id: M, completed: 500}}
Next orchestrator cycle: beckman.next_task() returns that task.
  → salako.run dispatches to workflow_advance executor
     → calls workflow_engine.advance(M, 500, previous_result)
     → returns envelope with captured artifacts + status + subtask list
  → Orchestrator.dispatch → beckman.on_task_finished(advance_task_id, envelope)
     → Subtasks get inserted, or retry/clarify fires via the normal rule set
```

No workflow-engine import appears in Beckman or salako core. Only the one executor in salako references it.

### Adding a new workflow

- Write a recipe JSON. Done. No code change.
- If the recipe needs a new agent type: add under `src/agents/` (same as today).
- If the recipe needs a new step action in the engine's vocabulary: change the workflow_engine package — a single, correct place.

## Orchestrator — the conductor (~100 lines)

```python
async def run_loop(self):
    self.running = True
    await self._startup()
    while self.running and not self.shutdown_event.is_set():
        if self._shutdown_signal_exists():
            break
        task = await beckman.next_task()
        if task:
            asyncio.create_task(self._dispatch(task))
        await asyncio.sleep(3)
    await self._shutdown()

async def _dispatch(self, task):
    runner = salako.run if task["agent_type"] == "mechanical" else self.llm_dispatcher.request
    try:
        result = await asyncio.wait_for(runner(task), timeout=self._timeout_for(task))
    except asyncio.TimeoutError:
        result = AgentResult.timeout(task)
    except Exception as e:
        result = AgentResult.from_exception(task, e)
    await beckman.on_task_finished(task["id"], result)
    push_metrics(task, result)
```

What orchestrator does:
- Process lifecycle (startup, shutdown signal handling, graceful drain)
- The dispatch pump (call Beckman, route by agent type, wait for result, call Beckman back)
- Metrics push at point of observation

What orchestrator does **not** do:
- Cron bookkeeping, "if due" branches — gone (all in Beckman)
- `_handle_*` methods — gone (logic redistributed to Beckman rules / workflow engine / salako)
- Separate plug-puller — the `asyncio.wait_for` at dispatch time pulls the plug. Orphaned "processing" rows from a crashed orchestrator are handled by Beckman's internal sweep.
- Lane partitioning, affinity reordering, swap-aware deferral — moved to Hoca (per-call scope)

## Hoca — gains swap/affinity concerns

Today the orchestrator does batch-level swap-aware deferral + affinity reordering. Under no-lanes, these concerns move to Hoca at **per-call** scope:

- When `hoca.select(task)` is called, Hoca already knows the loaded model. Swap budget and affinity logic apply here, one task at a time. No batch ordering needed — Beckman emits one task at a time and Hoca decides what model serves it.
- If Hoca's swap budget is exhausted, it returns a decision that the dispatcher respects (e.g., retry with `immediate` delay, let the task re-queue). No loop in Beckman.

This means `_reorder_by_model_affinity` and `_should_defer_for_loaded_model` (currently at the top of `orchestrator.py`) are deleted — their intent is absorbed by Hoca's existing selection logic.

## Nerd Herd — gains `health_summary()`

`check_resources` (the other half of today's watchdog — GPU thermal, VRAM leak detection, circuit-breaker aggregation, credential expiry) is not a watchdog. It's a health report.

Move to `nerd_herd.health_summary()`:
- Returns a structured health report.
- Called by `/status` Telegram command.
- Optionally driven on a slow cadence from Beckman's internal cron (e.g. 10min marker row → `nerd_herd.check_and_alert()`) for serious-issue Telegram alerts. Alerts become `salako.notify_user` tasks seeded by that marker.

No other subsystem pulls resource-health logic.

## The `_handle_*` methods — redistributed, not deleted as code

The eight handlers (~900 lines total) don't disappear — their logic moves:

| Handler | Where its logic lands |
|---|---|
| `_handle_availability_failure` | Hoca — already "model didn't become available"; becomes Hoca's internal retry-with-different-model decision. No dedicated handler needed. |
| `_handle_unexpected_failure` | Beckman retry policy (`RetryRequest(category=unexpected)` → retry or DLQ with notify task) |
| `_handle_complete` | Split: mission-progression logic → `workflow_engine.advance`. "Mission done" notification → `salako.notify_user` task Beckman creates. Parent-task rollup → Beckman sweep. |
| `_handle_subtasks` | Beckman's `SubtaskEmission` rule in `_apply_actions` |
| `_handle_clarification` | Already routed to salako in Phase 2b — no further work |
| `_handle_review` | Beckman's review-task dedup rule in `_apply_actions` |
| `_handle_exhausted` | Beckman retry policy (`RetryRequest(category=exhausted)`) |
| `_handle_failed` | Beckman retry policy + `salako.notify_user` task |

After migration, no `_handle_*` methods exist on the orchestrator class. `lifecycle.py`'s `get_orchestrator()._handle_*` circular delegation is removed.

## Result-guards audit (concrete disposition)

| Guard | Disposition |
|---|---|
| `guard_pipeline_artifacts` | Folded into `workflow_engine.advance` (same call that advances the mission captures the completed step's artifacts) |
| `guard_workflow_step_post_hook` | Folded into `workflow_engine.advance` — the post-hook runs as the first thing the advance routine does, producing status flips in the envelope |
| `guard_ungraded_post_hook` | **Collapsed with** `guard_workflow_step_post_hook`. The ungraded/completed split is a historical artifact — the hook is identical. One hook, one call site. |
| `guard_subtasks_blocked_for_workflow` | Beckman `_rewrite_actions` rule (one line) |
| `guard_clarification_suppression` — silent | Beckman `_rewrite_actions` rule |
| `guard_clarification_suppression` — may_need=false | Beckman `_rewrite_actions` rule |
| `guard_clarification_suppression` — history reuse | Beckman `_rewrite_actions` rule |
| `guard_clarification_suppression` — schema validation | Workflow engine post-hook |
| `_quality_retry_flow` | Beckman retry policy (with bonus-attempt heuristic flagged for a sideways look during migration) |

`packages/general_beckman/src/general_beckman/result_guards.py` is **deleted** after migration. Its code re-homes by domain.

## Paused patterns — state moves

The `Orchestrator.paused_patterns: set[str]` moves to Beckman module state (set via a small internal API exposed for the DLQ Telegram commands). Beckman applies the filter in `next_task()`'s pick stage.

## File / package layout after Task 13

```
packages/general_beckman/src/general_beckman/
├── __init__.py                 # next_task, on_task_finished, enqueue only
├── types.py                    # Task, AgentResult, TaskSpec, Action dataclasses
├── queue.py                    # pick logic, priority boost, paused-pattern filter
├── sweep.py                    # stuck-task hygiene (old check_stuck_tasks)
├── cron.py                     # scheduled_tasks table processor + seeding
├── result_router.py            # Action schema + agent-result → Actions mapping
├── rewrite.py                  # _rewrite_actions (policy rules)
├── apply.py                    # _apply_actions (action → DB)
├── retry.py                    # retry policy + DLQ writes + bonus-attempt
├── task_context.py             # parse_context (unchanged)
└── paused_patterns.py          # tiny state module

DELETED:
- general_beckman/lifecycle.py      # circular-delegation to orchestrator is gone
- general_beckman/result_guards.py  # re-homed per the audit table
- general_beckman/watchdog.py       # sweep → beckman/sweep.py; check_resources → nerd_herd
- general_beckman/scheduled_jobs.py # scheduled_tasks is the registry now

packages/salako/src/salako/
├── … existing executors …
└── workflow_advance.py         # new — thin delegator to workflow_engine.advance

packages/workflow_engine/       # new package (or promoted from src/workflows/engine/)
└── advance.py                  # the one entry point

src/core/
├── orchestrator.py             # ~100 lines: startup, pump, shutdown
├── llm_dispatcher.py           # unchanged in Task 13 scope
└── router.py                   # shim unchanged
```

## Success criteria (lifted from the handoff, sharpened)

- `src/core/orchestrator.py` ≤ 300 lines (target 150–200).
- Beckman's public API is exactly `next_task`, `on_task_finished`, `enqueue`. Nothing else is importable from `general_beckman.__init__`.
- No `get_orchestrator()._handle_*` calls anywhere in `packages/`.
- No workflow engine or hoca imports inside `packages/general_beckman/`.
- `result_guards.py` deleted; every guard's logic re-homed per the audit.
- Full test suite baseline ≤ 248 pre-existing failures; no new failures in touched modules.
- Manual smoke: `/task`, `/shop`, a mission end-to-end, a scheduled mission fires, a DLQ retry, a clarification round-trip.
- Behaviors preserved: age-based priority, paused-pattern filtering, stuck-task recovery, retry bonus attempts, waiting-human escalation.

## Explicitly out of scope

- Behavior-preservation migration strategy — separate brainstorm after this spec is approved.
- KDV cloud rate-limit state persistence (in-memory only, lost on restart).
- Full Telegram module extraction.
- Progress-chatter standardization (ephemeral sends still scattered).
- Workflow engine's own internal refactor (beyond collapsing the two post-hook guards into one).
- Hoca Phase 2c perf_score rebalance (separate track).
- Benchmark-cache staleness strategy beyond the marker-row hook.
- `_assess_timeout_progress` hardening (the bonus-attempt heuristic's dependency) — flagged, not fixed.

## Open questions intentionally deferred to the migration plan

- **Strangler vs big-bang.** Whether to migrate handlers one-at-a-time behind a flag, or cut over in one PR. Informs the plan, not the spec.
- **Workflow engine package extraction timing.** Could happen inside Task 13 or as a follow-up; if follow-up, salako's `workflow_advance` temporarily imports from `src/workflows/engine/`.
- **`_assess_timeout_progress` fate.** Leave in place (migrate as-is) or delete the heuristic (accept more DLQ noise). Decide during migration.
