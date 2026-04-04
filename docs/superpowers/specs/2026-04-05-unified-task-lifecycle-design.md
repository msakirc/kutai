# Unified Task Lifecycle & Quality Gate

## Problem

Long-running workflows (i2p, 200 steps) monopolize MAIN_WORK LLM calls. OVERHEAD calls (grading, classification) never get GPU time because:
1. The main loop blocks inside `process_task()` for the duration of each workflow step
2. Grade drain triggers only fire at the top of the main loop (when it's blocked) or on idle (never happens during workflows)
3. Grading is purely informational — bad grades have zero consequence
4. The grade queue is in-memory and lost on restart

Additionally, the retry/failure system has grown into 5 independent mechanisms with different counters, different escalation paths, and inconsistent terminology.

## Goals

1. **Quality gate**: Agent output must be graded before dependents can proceed
2. **Natural batching**: Grade swaps happen at fan-in points in the dependency graph, not after every step
3. **Unified retry model**: One system for all failure types, restart-resilient
4. **Simplify state machine**: Fewer states, enforced transitions, no undocumented states
5. **Kill in-memory queues**: Everything persisted in DB

## Non-Goals

- Changing how agents execute (ReAct loop, tool execution)
- Changing the i2p workflow definition format
- Adding new LLM providers or models
- Refactoring `telegram_bot.py` (separate effort)

---

## Task States

| State | Meaning | Who acts next |
|---|---|---|
| `pending` | Ready to be picked up (or waiting for `next_retry_at`) | Orchestrator |
| `processing` | Agent running | Agent |
| `ungraded` | Agent done, awaiting quality check | Grader (deferred) |
| `completed` | Done, quality confirmed or waived | Terminal |
| `failed` | Permanently failed, in DLQ | Human |
| `waiting_subtasks` | Decomposed, children running | Children |
| `waiting_human` | Needs clarification or approval | Human |
| `cancelled` | Cancelled by user | Terminal |
| `skipped` | Dependency-skipped | Terminal |

### Removed States

| Old state | Replacement |
|---|---|
| `paused` | `pending` with `next_retry_at` in the future |
| `sleeping` | `pending` with `next_retry_at` in the future |
| `needs_clarification` | `waiting_human` |
| `needs_review` | Was never persisted. Agent return value only. |
| `rejected` | `cancelled` |
| `done` | Was a bug. Use `completed`. |

---

## Schema Changes

### New columns on `tasks`

```sql
attempts INTEGER DEFAULT 0,           -- worker quality failures, NEVER resets
max_attempts INTEGER DEFAULT 6,       -- worker hard cap → DLQ
grade_attempts INTEGER DEFAULT 0,     -- grading quality failures, resets on worker retry
max_grade_attempts INTEGER DEFAULT 3, -- grading hard cap → waive grading
avail_retries INTEGER DEFAULT 0,      -- availability failures (shared, both phases)
max_avail_retries INTEGER DEFAULT 5,  -- availability hard cap → DLQ
next_retry_at TIMESTAMP,              -- NULL = immediately eligible, future = delayed
retry_reason TEXT,                     -- "quality" or "availability" (last failure)
failed_in_phase TEXT,                  -- "worker" or "grading" — which phase hit DLQ
```

### Deprecated columns (keep for backward compat, stop writing)

```sql
retry_count    → replaced by attempts
max_retries    → replaced by max_attempts
sleep_state    → replaced by next_retry_at + attempts
error_category → replaced by retry_reason
```

### Migration

```sql
-- Backfill new columns from old ones
UPDATE tasks SET attempts = COALESCE(retry_count, 0)
  WHERE attempts = 0 AND retry_count > 0;
UPDATE tasks SET max_attempts = COALESCE(max_retries, 3) + 3
  WHERE max_attempts = 6 AND max_retries IS NOT NULL AND max_retries != 3;

-- Convert sleeping tasks to pending with next_retry_at
UPDATE tasks SET status = 'pending',
  next_retry_at = json_extract(sleep_state, '$.next_timer_wake')
  WHERE status = 'sleeping';

-- Convert paused tasks to pending with 10-min delay
UPDATE tasks SET status = 'pending',
  next_retry_at = datetime('now', '+10 minutes')
  WHERE status = 'paused';

-- Rename needs_clarification
UPDATE tasks SET status = 'waiting_human'
  WHERE status = 'needs_clarification';

-- Fix rejected → cancelled
UPDATE tasks SET status = 'cancelled'
  WHERE status = 'rejected';
```

---

## Two Failure Types

Every failure, everywhere in the system, is one of two types:

### Quality

Output is bad or missing. The agent ran (or the grader ran) but the result isn't acceptable.

**Triggers:**
- Agent returns `failed` (execution error, tool error, invalid output)
- Post-hook detects disguised failure (artifact validation)
- Schema validation fails
- Grade VERDICT = FAIL (worker output judged bad)
- Grade parse error (grader output is garbage)

**Retry behavior:**
- Attempts 1-2: immediate retry, same model allowed
- Attempt 3: immediate retry, MUST exclude previously failed models
- Attempt 4+: delayed retry (10 min), exclude failed models + difficulty += 2 per attempt past 3
- At `max_attempts`: terminal → `failed` → DLQ

**Model escalation:**
- `context.excluded_models` accumulates models that produced bad output
- Router reads `excluded_models` and skips them during model selection
- Difficulty bump changes the scoring profile, favoring more capable models
- Applies to both worker and grader model selection (separate lists, see Grading section)

### Availability

Couldn't execute at all. The model, GPU, network, or API was unavailable.

**Triggers:**
- `ModelCallFailed` (all models exhausted — local + cloud)
- Network timeout on cloud call
- GPU scheduler timeout (waited 60s, never got access)
- Rate limit hit
- Grade call timeout/OOM (grading-specific availability)
- All eligible models excluded by quality failures (exclusion list exhausted the pool)

**Retry behavior:**
- **Does NOT increment `attempts` or `grade_attempts`** — availability is not the task's fault
- Increments `avail_retries` (shared counter for backoff progression)
- Signal-aware delay with exponential backoff
- Delays: 1 min → 5 min → 15 min → 1 hour → 2 hours
- At `max_avail_retries` (5): terminal → `failed` → DLQ
- Signal wake (`accelerate_retries`) can pull `next_retry_at` to now

**Phase-preserving:** Availability failures return the task to its current phase, not the beginning:
- Worker phase availability → task stays/returns to `pending` (not re-graded)
- Grading phase availability → task stays/returns to `ungraded` (not re-executed)

**DLQ re-enable:** When a task in DLQ due to availability is retried via `/dlq retry`, it returns to the phase it was in when it failed:
- Worker-phase DLQ → `pending`
- Grading-phase DLQ → `ungraded`

**No model change needed** — the model is fine, it was just unavailable.

**Excluded models → availability escalation:** When quality failures accumulate `excluded_models` that cover ALL available models, the next retry attempt becomes an availability failure (no eligible model). This enters the availability backoff path. New models becoming available (added to registry, or DLQ retry clears exclusions) resolves this.

---

## Retry Timing

```python
def compute_retry_timing(
    failure_type: str,  # "quality" or "availability"
    attempts: int = 0,        # quality failure count (worker or grading)
    max_attempts: int = 6,    # quality hard cap
    avail_retries: int = 0,   # availability failure count
    max_avail_retries: int = 5,  # availability hard cap
) -> RetryDecision:
    """
    Returns: IMMEDIATE, DELAYED(seconds), or TERMINAL

    Quality failures increment attempts/grade_attempts (caller's responsibility).
    Availability failures increment avail_retries (caller's responsibility).
    """
    if failure_type == "quality":
        if attempts >= max_attempts:
            return TERMINAL
        if attempts < 3:
            return IMMEDIATE
        else:
            return DELAYED(600)  # 10 min

    elif failure_type == "availability":
        if avail_retries >= max_avail_retries:
            return TERMINAL
        delays = [60, 300, 900, 3600, 7200]
        idx = min(avail_retries, len(delays) - 1)
        return DELAYED(delays[idx])
```

### Model Exclusion on Quality Failure

```python
def update_exclusions_on_failure(task_context: dict, failed_model: str, attempts: int):
    """Track failed models and compute exclusions for next attempt."""
    failed = task_context.setdefault("failed_models", [])
    if failed_model and failed_model not in failed:
        failed.append(failed_model)

def get_model_constraints(task_context: dict, attempts: int) -> tuple[list[str], int]:
    """Returns (excluded_models, difficulty_bump) for the next attempt."""
    failed = task_context.get("failed_models", [])
    excluded = failed if attempts >= 3 else []
    difficulty_bump = max(0, (attempts - 3) * 2) if attempts >= 4 else 0
    return excluded, difficulty_bump
```

### Signal-Aware Wake

Replaces `wake_sleeping_tasks()`:

```python
async def accelerate_retries(reason: str) -> int:
    """Pull next_retry_at to now for pending tasks waiting on availability.

    Called from: model_swap, gpu_available, rate_limit_reset,
    quota_restored, circuit_breaker_reset.

    Returns number of tasks accelerated.
    """
    db = await get_db()
    cursor = await db.execute(
        """UPDATE tasks
           SET next_retry_at = datetime('now')
           WHERE status = 'pending'
           AND next_retry_at > datetime('now')
           AND retry_reason = 'availability'"""
    )
    await db.commit()
    return cursor.rowcount
```

---

## The `ungraded` State

### Two-Phase Execution Model

A task has two phases executed on the same DB row:

1. **Worker phase**: agent executes, produces output → status becomes `ungraded`
2. **Grading phase**: grader evaluates output → `completed` or back to `pending`

Each phase has its own attempt budget:
- `attempts` / `max_attempts` — worker phase
- `grade_attempts` / `max_grade_attempts` — grading phase

`grade_attempts` resets to 0 when the task returns to `pending` (worker retry), because the next worker run produces new output that needs fresh grading.

### Worker Phase Completion

When an agent returns `status="completed"`:

```python
grade_result = await dispatcher.request_grade(...)

if grade_result is not None:
    # Immediate grading (loaded model != generating, or priority >= 8)
    if grade_result.verdict == "PASS":
        await update_task(task_id, status="completed", quality_score=...)
    else:
        # VERDICT=FAIL → worker quality failure
        update_exclusions_on_failure(context, used_model, attempts)
        decision = compute_retry_timing(attempts + 1, max_attempts, "quality")
        if decision == TERMINAL:
            await update_task(task_id, status="failed", ...)
        else:
            await update_task(task_id, status="pending",
                              attempts=attempts + 1,
                              next_retry_at=decision.timestamp,
                              grade_attempts=0,  # reset for next worker run
                              excluded_models=..., ...)
else:
    # Grade deferred — loaded model IS generating model
    await update_task(task_id, status="ungraded",
                      context={...generating_model: used_model})
```

### Grading Phase (Deferred)

Grading happens when main work drains naturally. The existing `ensure_gpu_utilized` path works:

1. `get_ready_tasks()` returns 0 (dependents blocked on `ungraded` predecessors)
2. `ensure_gpu_utilized([])` → `has_pending_overhead_needs()` → finds `ungraded` tasks
3. Loads fastest general model (different from generating model → unbiased)
4. Model swap fires → grade all `ungraded` tasks in batch

**Grade drain on model swap:**

```python
async def drain_ungraded_tasks(new_model: str):
    """Grade all ungraded tasks that the new model can grade.

    Called from on_model_swap(). The new model can grade any task
    NOT generated by itself.
    """
    db = await get_db()
    ungraded = await db.execute(
        "SELECT * FROM tasks WHERE status = 'ungraded'"
    )
    tasks = [dict(row) for row in await ungraded.fetchall()]

    for task in tasks:
        ctx = json.loads(task.get("context", "{}"))
        generating_model = ctx.get("generating_model")

        if generating_model == new_model:
            continue  # can't self-grade

        try:
            verdict = await grade_task(task, new_model)
            await apply_grade_result(task, verdict)
        except AvailabilityError:
            # Grading availability failure — backoff, stay in grading phase
            avail = task.get("avail_retries", 0) + 1
            decision = compute_retry_timing("availability", avail_retries=avail)
            if decision == TERMINAL:
                await update_task(task["id"],
                                  status="failed", failed_in_phase="grading",
                                  avail_retries=avail)
                await quarantine_task(task["id"], ...)
            else:
                await update_task(task["id"],
                                  avail_retries=avail,
                                  next_retry_at=decision.timestamp,
                                  retry_reason="availability")
                # stays ungraded — will be retried on next drain
        except QualityError:
            # Grader parse error — grader quality failure
            g_attempts = task.get("grade_attempts", 0) + 1
            ctx.setdefault("grade_excluded_models", []).append(new_model)
            if g_attempts >= max_grade_attempts:
                # Waive grading — promote with NULL score
                await update_task(task["id"], status="completed", quality_score=None)
            else:
                await update_task(task["id"],
                                  grade_attempts=g_attempts,
                                  context=json.dumps(ctx))
                # stays ungraded, next drain will use a different model
```

### Dependency Resolution

`get_ready_tasks` counts as "resolved" for dependency checks:
- `completed`
- `skipped`

Does NOT count:
- `ungraded`
- Everything else

```sql
-- Dependency check: all deps must be completed or skipped
SELECT COUNT(*) FROM tasks
WHERE id IN (?, ?, ?) AND status IN ('completed', 'skipped')
```

### Natural Batching (i2p example)

The i2p dependency graph has a repeating diamond pattern: fan-out → parallel work → fan-in synthesis step.

```
Phase 3: Requirements
  3.1 ─┐
  3.2 ─┤
  3.3 ─┼──► 3.11 (fan-in: depends on all)
  3.4 ─┤
  3.5 ─┘

Steps 3.1-3.5 execute in parallel → each finishes as "ungraded"
Step 3.11 depends on all of them → blocked (ungraded != completed)
get_ready_tasks() returns 0
ensure_gpu_utilized → load grader model → swap
on_model_swap → drain_ungraded_tasks → grade all 5 in batch
All pass → 3.1-3.5 become "completed" → 3.11 unblocks
Bad grade on 3.4 → 3.4 goes to "pending" (attempts++, model excluded)
  → 3.11 stays blocked until 3.4 re-runs and grades PASS
```

Swap count for full i2p: ~15-20 grade swaps (one per fan-in gate), not 200.

---

## Structured Grading Prompt

Replace the 1-5 numeric scale with structured binary:

```
Evaluate this task response.

Task: {title}
Description: {description}
Response: {response[:2000]}

Answer each with YES or NO only:
RELEVANT: Does the response address the task?
COMPLETE: Does it contain a concrete deliverable, not just a plan or description?
VERDICT: Should this response be accepted?
```

**Parse priority:**
1. Extract VERDICT → use directly
2. If VERDICT unparseable but RELEVANT and COMPLETE parse → derive (both YES → PASS, either NO → FAIL)
3. If nothing parses → QualityError (grader incapable)

**Why binary:** Small local LLMs handle YES/NO much more reliably than 1-5 numeric scales. No calibration problem — there's no scale to miscalibrate.

**Score mapping for analytics:** PASS = 4.0, FAIL = 2.0 (preserves compatibility with existing quality_score column and skill extraction threshold of 4.0).

---

## Watchdog (Simplified)

The watchdog runs every 10 cycles. With the unified model, it handles fewer cases:

| Check | Condition | Action |
|---|---|---|
| **Stuck processing** | `processing` for > 5 min | Quality failure: `attempts++`, `compute_retry_timing` |
| **Stuck ungraded** | `ungraded` for > 30 min, no grade in progress | Promote to `completed`, `quality_score=NULL` (safety net) |
| **Overdue retry** | `pending`, `next_retry_at` > 1 hour in the past | Clear `next_retry_at` (make immediately eligible) |
| **Failed dep** | `pending` with failed deps | If ALL non-skipped deps are `failed` → cascade to `failed`. If only SOME deps failed → wait (failed dep may be retried via DLQ). Never clear deps. |
| **Waiting subtasks** | All children terminal | `completed` if any child completed, `failed` if all children failed |
| **Stale waiting_human** | Escalation: 4h nudge → 24h → 48h → 72h cancel | Same as current |
| **Hard cap** | `attempts >= max_attempts` anywhere | `failed` → DLQ |

### Removed watchdog handlers

- **Paused handler** (10-min retry_count=0 reset) → no more `paused` state
- **Sleeping timer scan** → no more `sleeping` state, `next_retry_at` handles timing
- **Sleeping restart wake** → on restart, tasks with `next_retry_at` in the past are immediately eligible

### Fixed

- **Failed dep clearing** → now cascades failure when ALL non-skipped deps failed (never clears deps, never cascades on partial failure)
- **Waiting subtasks all-failed** → now `failed` instead of `completed`

---

## `get_ready_tasks` Changes

```sql
SELECT * FROM tasks
WHERE status = 'pending'
AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))
ORDER BY priority DESC, created_at ASC
```

Plus the existing dependency check, where `ungraded` does NOT count as resolved.

---

## Restart Resilience

Everything is in the DB. On restart:

| State | What happens |
|---|---|
| `pending` with `next_retry_at` in past | Immediately eligible, picked up next cycle |
| `pending` with `next_retry_at` in future | Waits until time passes, or signal accelerates |
| `processing` | Watchdog detects stuck > 5 min, retries |
| `ungraded` | Next model swap triggers grading drain. Watchdog safety net at 30 min. |
| `ungraded` with `next_retry_at` in future | Availability backoff for grading — waits, then eligible for next drain |
| `waiting_subtasks` | Watchdog checks children completion |
| `waiting_human` | Escalation timers continue |
| `failed` with `failed_in_phase` | DLQ retry restores to correct phase (`pending` or `ungraded`) |

No in-memory queues to rebuild. No lost state.

---

## Components Killed

| Component | Replacement |
|---|---|
| `GradeQueue` class (llm_dispatcher.py) | DB query: `WHERE status = 'ungraded'` |
| `PendingGrade` dataclass | Task row itself (result + context.generating_model) |
| `SwapBudget` class | Unchanged (still needed for MAIN_WORK swap throttling) |
| `sleep_state` JSON | `next_retry_at` column |
| `_SLEEP_TIER_INTERVALS` | `compute_retry_timing` delay table |
| `wake_sleeping_tasks()` | `accelerate_retries()` |
| `make_sleep_state()` | Removed |
| `get_sleeping_tasks()` | Removed |
| `_schema_retry_count` in context | `attempts` column (shared budget) |

---

## State Machine Enforcement

Update `state_machine.py` to match reality and actually use it:

```python
class TaskState(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    UNGRADED = "ungraded"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_SUBTASKS = "waiting_subtasks"
    WAITING_HUMAN = "waiting_human"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

TRANSITIONS = {
    PENDING:          {PROCESSING, CANCELLED, SKIPPED},
    PROCESSING:       {UNGRADED, COMPLETED, PENDING, FAILED,
                       WAITING_SUBTASKS, WAITING_HUMAN, CANCELLED},
    UNGRADED:         {COMPLETED, PENDING, FAILED},  # grade pass, grade fail (worker retry), availability DLQ
    COMPLETED:        set(),  # terminal
    FAILED:           {PENDING, UNGRADED},  # DLQ retry (worker-phase → pending, grading-phase → ungraded)
    WAITING_SUBTASKS: {COMPLETED, FAILED, CANCELLED},
    WAITING_HUMAN:    {PENDING, CANCELLED},
    CANCELLED:        set(),  # terminal
    SKIPPED:          set(),  # terminal
}
```

All status changes go through `transition_task()`. No more raw `update_task(status=...)` bypassing validation.

---

## DLQ

**Entry** — three paths, all lead to `failed` → `quarantine_task()`:
- Worker quality exhausted: `attempts >= max_attempts`
- Grading quality exhausted: `grade_attempts >= max_grade_attempts` → waive grading (promote to `completed` with `quality_score=NULL`). This does NOT go to DLQ — grading waiver is graceful degradation.
- Availability exhausted: `avail_retries >= max_avail_retries` → `failed` → DLQ. `failed_in_phase` records whether the task was in worker or grading phase.

**Exit** — manual only (`/dlq retry`, `/dlq discard`, inline button).

**Phase-aware re-enable:** `/dlq retry` reads `failed_in_phase` to determine where the task resumes:
- `failed_in_phase = "worker"` → `pending` (re-execute from scratch)
- `failed_in_phase = "grading"` → `ungraded` (worker output preserved, retry grading only)
- `failed_in_phase = NULL` (legacy) → `pending` (backward compat)

Re-enable resets `avail_retries = 0` but preserves `attempts` and `grade_attempts` (quality budget is lifetime).

**Mission health:** auto-pause mission if >= 3 DLQ tasks (unchanged).

---

## Telegram Impact

| Change | Telegram side |
|---|---|
| `needs_clarification` → `waiting_human` | Update status display strings |
| `paused` removed | Remove "paused" from status displays, keep "Retry" button for `failed` |
| `sleeping` removed | Remove "sleeping" from status displays |
| `rejected` → `cancelled` | Update display |
| `ungraded` added | Show as "Grading..." in task status. Optional: show grade result on promotion. |
| Grade VERDICT=FAIL notification | Notify user: "Task #{id} output rejected by grader, retrying with different model" |

---

## Interaction with `ensure_gpu_utilized`

The existing mechanism works with minimal changes:

```python
async def ensure_gpu_utilized(self, upcoming_tasks):
    if manager.current_model:
        return  # already loaded

    if upcoming_tasks:
        # Main work available → load best-fit work model
        best = self._find_best_local_for_batch(upcoming_tasks)
        if best:
            await manager.ensure_model(best, reason="proactive_load")
        return

    # No main work — check overhead needs
    if await self._has_pending_overhead_needs():
        best = self._find_fastest_general_model()
        if best:
            await manager.ensure_model(best, reason="overhead_load")
```

`_has_pending_overhead_needs` updated:

```python
async def _has_pending_overhead_needs(self) -> bool:
    # Ungraded tasks waiting for grading
    db = await get_db()
    cursor = await db.execute(
        "SELECT COUNT(*) FROM tasks WHERE status = 'ungraded'"
    )
    ungraded_count = (await cursor.fetchone())[0]
    if ungraded_count > 0:
        return True

    # Pending todos (suggestion calls)
    try:
        from src.infra.db import get_todos
        todos = await get_todos(status="pending")
        return len(todos) > 0
    except Exception:
        return False
```

---

## Summary of Changes by File

| File | Changes |
|---|---|
| `src/core/state_machine.py` | New states, enforced transitions |
| `src/infra/db.py` | New columns, migration, updated `get_ready_tasks`, `accelerate_retries`, remove sleeping queue functions |
| `src/core/orchestrator.py` | Unified retry calls, simplified watchdog, grade drain on swap, remove sleeping/paused handlers |
| `src/core/llm_dispatcher.py` | Remove `GradeQueue`/`PendingGrade`, update `ensure_gpu_utilized`, add `drain_ungraded_tasks` |
| `src/core/router.py` | Updated grading prompt (structured binary), read `excluded_models` from task context |
| `src/agents/base.py` | Worker completion → `ungraded` status, grade result handling |
| `src/app/telegram_bot.py` | Status display updates, remove sleeping/paused references |
| `src/infra/dead_letter.py` | No changes (entry/exit unchanged) |
