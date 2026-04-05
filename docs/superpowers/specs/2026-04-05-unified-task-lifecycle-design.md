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
next_retry_at TIMESTAMP,              -- NULL = immediately eligible, future = delayed
retry_reason TEXT,                     -- "quality" or "availability" (last failure)
failed_in_phase TEXT,                  -- "worker" or "grading" — which phase hit DLQ
```

**Availability backoff** stored as `context.last_avail_delay` (seconds):

```python
# On availability failure:
last_delay = ctx.get("last_avail_delay", 0)
if last_delay >= 7200:
    → TERMINAL → failed → DLQ

new_delay = max(60, min(last_delay * 2, 7200))  # double each time, clamp 1min–2h
ctx["last_avail_delay"] = new_delay
next_retry_at = now + timedelta(seconds=new_delay)

# Signal wake resets backoff: ctx["last_avail_delay"] = 0
```

Total time before availability DLQ: ~5 hours (1m + 2m + 4m + 8m + 16m + 32m + 64m + 2h).
Signal wakes (`accelerate_retries`) pull `next_retry_at` to now AND reset `last_avail_delay` to 0 (fresh backoff on next failure).

### Deprecated columns (keep for backward compat, stop writing)

```sql
retry_count    → replaced by attempts
max_retries    → replaced by max_attempts
sleep_state    → replaced by next_retry_at
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
- Doubling backoff derived from `next_retry_at`: 1m → 2m → 4m → 8m → ... → 2h cap
- At 2h cap still failing: terminal → `failed` → DLQ (~5 hours total)
- Signal wake (`accelerate_retries`) pulls `next_retry_at` to now, resetting backoff progression

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
    attempts: int = 0,           # quality failure count (worker or grading phase)
    max_attempts: int = 6,       # quality hard cap
    last_avail_delay: int = 0,   # from context, seconds
) -> RetryDecision:
    """
    Returns: IMMEDIATE, DELAYED(seconds), or TERMINAL

    Quality: caller increments attempts/grade_attempts before calling.
    Availability: doubling backoff from last_avail_delay (stored in context).
    """
    if failure_type == "quality":
        if attempts >= max_attempts:
            return TERMINAL
        if attempts < 3:
            return IMMEDIATE
        else:
            return DELAYED(600)  # 10 min

    elif failure_type == "availability":
        if last_avail_delay >= 7200:
            return TERMINAL  # was already at 2h cap, still failing → DLQ
        new_delay = max(60, min(last_avail_delay * 2, 7200))
        return DELAYED(new_delay)
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
    """Pull next_retry_at to now for tasks waiting on availability.

    Called from: model_swap, gpu_available, rate_limit_reset,
    quota_restored, circuit_breaker_reset.

    Resets last_avail_delay in context so backoff starts fresh
    if the next attempt also fails.

    Covers both phases: pending (worker) and ungraded (grading).

    Returns number of tasks accelerated.
    """
    db = await get_db()

    # Find eligible tasks (pending or ungraded, waiting on availability)
    cursor = await db.execute(
        """SELECT id, context FROM tasks
           WHERE status IN ('pending', 'ungraded')
           AND next_retry_at > datetime('now')
           AND retry_reason = 'availability'"""
    )
    rows = [dict(r) for r in await cursor.fetchall()]

    for row in rows:
        ctx = json.loads(row.get("context", "{}"))
        ctx["last_avail_delay"] = 0  # reset backoff
        await db.execute(
            """UPDATE tasks SET next_retry_at = datetime('now'),
               context = ? WHERE id = ?""",
            (json.dumps(ctx), row["id"]),
        )

    if rows:
        await db.commit()
    return len(rows)
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

When an agent returns `status="completed"`, the following happens in order:

1. **Post-hook** (`post_execute_workflow_step`) — structural/schema validation. Instant, no LLM.
   If it fails → quality failure, task retries immediately (never reaches `ungraded`).
2. **Grade attempt** — quality assessment via different model. May be immediate or deferred.
   If immediate and FAIL → quality failure, task retries.
   If deferred → task enters `ungraded`.

This ordering matters: post-hook catches hard structural failures fast (missing artifacts,
broken schema) without wasting a model swap on grading. Only structurally valid output
reaches the grading phase.

`context.worker_completed_at` is set when entering `ungraded` (or `completed` if graded
immediately). This tracks how long grading has been pending — useful for diagnostics and
the watchdog's "stuck ungraded" safety net.

Pseudocode:

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

Grading of `ungraded` tasks is triggered by three existing call sites in the main loop.
No new mechanisms needed — just replace in-memory grade queue operations with DB queries.

**Trigger 1: Model swap** (`on_model_swap`)

When any model swap completes, drain `ungraded` tasks the new model can grade.
This is the primary grading path during workflows — fan-in points cause main work
to drain, `ensure_gpu_utilized` loads a grader model, the swap fires this trigger.

```
Main loop cycle:
  get_ready_tasks() → 0 (dependents blocked on ungraded)
  ensure_gpu_utilized([]) → sees ungraded tasks → loads fastest general model
  → model swap completes → on_model_swap() → drain_ungraded_tasks(new_model)
  → grades complete → ungraded → completed → dependents unblock
  → next cycle: get_ready_tasks() finds newly ready tasks → back to main work
```

**Trigger 2: Main loop idle path** (existing line 3040 in orchestrator)

When no tasks are running AND no tasks are ready, the main loop enters the idle path.
If a model is already loaded and there are `ungraded` tasks it can grade, grade them
directly. No swap needed.

```python
# Replaces drain_grades_if_idle():
if no_ready_tasks and no_running_tasks:
    loaded_model = get_loaded_litellm_name()
    if loaded_model:
        ungraded = await get_ungraded_tasks()
        for task in ungraded:
            if task.generating_model != loaded_model:
                await grade_and_apply(task, loaded_model)
```

This handles the case where main work finishes and the model is still loaded —
grade opportunistically before the idle unloader kicks in (60s window).

**Trigger 3: `ensure_gpu_utilized`** (existing line 2923 in orchestrator)

When no model is loaded AND no main work tasks exist, checks for overhead needs
(ungraded tasks, pending todos). If found, loads the fastest general-purpose model.
The load triggers a model swap → Trigger 1 fires → grades drain.

```
ensure_gpu_utilized([]):
  no model loaded, no main work
  → _has_pending_overhead_needs() → finds ungraded tasks
  → loads fastest general model
  → on_model_swap → drain_ungraded_tasks (Trigger 1)
```

**How these three work together:**

| Situation | Which trigger fires |
|---|---|
| Workflow fan-in: main work blocked on ungraded | Trigger 3 (load grader) → Trigger 1 (swap drain) |
| Main work finished, model still loaded, ungraded tasks exist | Trigger 2 (idle drain, no swap) |
| System fully idle, no model loaded, ungraded tasks exist | Trigger 3 (load grader) → Trigger 1 (swap drain) |
| Model swap for main work (different model loaded) | Trigger 1 (opportunistic drain during swap) |

**`apply_grade_result`** — called when grading succeeds (both immediate and deferred):

```python
async def apply_grade_result(task: dict, verdict: GradeResult):
    """Apply grade outcome to task. Handles both PASS and FAIL."""
    task_id = task["id"]
    ctx = json.loads(task.get("context", "{}"))
    score = 4.0 if verdict.passed else 2.0  # analytics compat

    if verdict.passed:
        await transition_task(task_id, "completed", quality_score=score)

        # Skill extraction — same as current _handle_complete logic
        # Triggered here for deferred grades (immediate grades trigger in base.py)
        iterations = task.get("iterations", 1)
        tools_used = ctx.get("tools_used_names", [])
        if iterations >= 2 and tools_used:
            await extract_and_store_skill(task, verdict.grader_data)

        # Record model quality feedback
        await record_model_call(
            model=ctx.get("generating_model"),
            agent_type=task.get("agent_type"),
            success=True, grade=score,
        )
    else:
        # VERDICT=FAIL — worker quality failure, retry with model escalation
        attempts = task.get("attempts", 0) + 1
        update_exclusions_on_failure(ctx, ctx.get("generating_model"), attempts)
        decision = compute_retry_timing("quality", attempts=attempts)

        if decision == TERMINAL:
            await transition_task(task_id, "failed", failed_in_phase="worker")
            await quarantine_task(task_id, ...)
        else:
            excluded, diff_bump = get_model_constraints(ctx, attempts)
            await transition_task(task_id, "pending",
                attempts=attempts,
                grade_attempts=0,  # reset for next worker run
                next_retry_at=decision.timestamp,
                retry_reason="quality",
                context=json.dumps(ctx))
```

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
            last_delay = ctx.get("last_avail_delay", 0)
            decision = compute_retry_timing("availability",
                          last_avail_delay=last_delay)
            if decision == TERMINAL:
                await update_task(task["id"],
                                  status="failed", failed_in_phase="grading")
                await quarantine_task(task["id"], ...)
            else:
                ctx["last_avail_delay"] = decision.seconds
                await update_task(task["id"],
                                  next_retry_at=decision.timestamp,
                                  retry_reason="availability",
                                  context=json.dumps(ctx))
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

## Model Swap Signal Ordering

When `on_model_swap` fires, two things need to happen:
1. `accelerate_retries("model_swap")` — wake pending tasks with availability backoff
2. `drain_ungraded_tasks(new_model)` — grade ungraded tasks

**Order matters:** accelerate first, then drain. Accelerated tasks become immediately
eligible for the next main loop cycle. Grading happens synchronously in the same
`on_model_swap` call — by the time the main loop runs, both wakes and grades are done.

```python
async def on_model_swap(self, old_model, new_model):
    # 1. Wake availability-delayed tasks (pending and ungraded)
    woken = await accelerate_retries("model_swap")

    # 2. Grade ungraded tasks the new model can handle
    if new_model:
        graded = await drain_ungraded_tasks(new_model)

    # 3. All grading/routing goes through dispatcher (OVERHEAD)
    #    Never call grade_response directly — dispatcher handles
    #    timeout, error propagation, metrics.
```

## Mission Completion Query

`_check_mission_completion` must be aware of `ungraded`:

```python
# Current: status not in (completed, failed, rejected)
# Updated: status not in (completed, failed, cancelled, skipped)
# "ungraded" is NOT in the exclusion list — keeps mission open until all grades resolve
pending = [s for s in statuses if s not in ("completed", "failed", "cancelled", "skipped")]
```

Also must handle the renamed states:
- `rejected` → `cancelled` (already in exclusion list)
- `sleeping` / `paused` → no longer exist (tasks are `pending` with `next_retry_at`)

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

**Entry** — two paths lead to `failed` → `quarantine_task()`:
- Worker quality exhausted: `attempts >= max_attempts`
- Availability exhausted: `last_avail_delay >= 7200` and still failing. `failed_in_phase` records whether the task was in worker or grading phase.

Grading quality exhaustion (`grade_attempts >= max_grade_attempts`) does NOT go to DLQ — it waives grading and promotes to `completed` with `quality_score=NULL` (graceful degradation).

**Exit** — manual only (`/dlq retry`, `/dlq discard`, inline button).

**Phase-aware re-enable:** `/dlq retry` reads `failed_in_phase` to determine where the task resumes:
- `failed_in_phase = "worker"` → `pending` (re-execute from scratch)
- `failed_in_phase = "grading"` → `ungraded` (worker output preserved, retry grading only)
- `failed_in_phase = NULL` (legacy) → `pending` (backward compat)

**What resets on DLQ re-enable:**

| Field | Reset? | Why |
|---|---|---|
| `context.last_avail_delay` | Yes → 0 | Fresh backoff progression |
| `context.excluded_models` | Yes → `[]` | Models may have been updated/reloaded |
| `context.grade_excluded_models` | Yes → `[]` (but `generating_model` stays excluded from grading) | Same reason, but self-grading still prevented |
| `attempts` | No (preserved) | Quality budget is lifetime — human chose to retry, doesn't mean the task got easier |
| `grade_attempts` | No (preserved) | Same |
| `next_retry_at` | Yes → NULL | Immediately eligible |
| `retry_reason` | Yes → NULL | Clean slate |

**Mission health:** auto-pause mission if >= 3 DLQ tasks (unchanged).

---

## Telegram Impact

### Status Display

| Change | Telegram side |
|---|---|
| `needs_clarification` → `waiting_human` | Update status display strings, clarification handlers reference new state |
| `paused` removed | Remove "paused" from status displays and keyboard state handlers |
| `sleeping` removed | Remove "sleeping" from status displays and `/status` output |
| `rejected` → `cancelled` | Update display strings |
| `ungraded` added | Show as "Bekliyor: Derecelendirme..." in task status |
| Grade VERDICT=FAIL | Notify: "Task #{id} output rejected by grader, retrying with different model" |
| Grade PASS (deferred) | Notify for non-silent tasks: "Task #{id} graded and completed" |

### Button/Handler Changes

| Button/Handler | Current | Updated |
|---|---|---|
| "Retry" button on task detail | Shows for `failed`, `paused`, `sleeping` | Shows for `failed` only (other states auto-retry) |
| "Resume" for paused | Exists as handler | Remove — no more `paused` state |
| `/reset paused` admin command | Bulk resets paused tasks | Remove — use `/reset pending` for stuck tasks |
| `/reset failed` admin command | Bulk resets failed tasks | Unchanged |
| `/dlq retry` | Sets `pending`, `retry_count=0` | Phase-aware: reads `failed_in_phase`, resets exclusions |
| Task status in `/tasks` list | Shows sleeping/paused counts | Remove those categories, add `ungraded` count |
| Inline "Tekrar Dene" on DLQ | Sets `pending` | Phase-aware: `pending` or `ungraded` based on `failed_in_phase` |

---

## Interaction with `ensure_gpu_utilized`

The current implementation bails out when any model is loaded (`if manager.current_model: return`).
This creates a **60-second hole** at workflow fan-in points:

```
Problem scenario:
  Steps 3.1-3.5 all generated by model X → finish as "ungraded"
  Model X still loaded → ensure_gpu_utilized returns early
  Idle path: all ungraded tasks generated by X → can't self-grade → nothing happens
  60s idle → unloader unloads X
  THEN ensure_gpu_utilized loads grader model Y → swap → drain
  = 60s wasted at every fan-in point × 15-20 fan-in points = 15-20 min total waste
```

**Fix:** Don't bail out when the loaded model can't serve current needs:

```python
async def ensure_gpu_utilized(self, upcoming_tasks):
    if upcoming_tasks:
        if not manager.current_model:
            # No model loaded, main work waiting → load best-fit
            best = self._find_best_local_for_batch(upcoming_tasks)
            if best:
                await manager.ensure_model(best, reason="proactive_load")
        return  # main work exists, proceed to process it

    # No main work. Check overhead needs.
    if not await self._has_pending_overhead_needs():
        return  # nothing to do

    if manager.current_model:
        # Model loaded but no main work. Can it grade ungraded tasks?
        if await self._loaded_model_can_grade():
            return  # idle path (Trigger 2) will handle it
        # Loaded model can't grade (self-generated) → swap to grader
        best = self._find_best_grader_model()
        if best and best != manager.current_model:
            await manager.ensure_model(best, reason="grade_swap")
    else:
        # No model loaded, overhead needs exist → load grader
        best = self._find_fastest_general_model()
        if best:
            await manager.ensure_model(best, reason="overhead_load")
```

```python
async def _loaded_model_can_grade(self) -> bool:
    """Check if loaded model can grade ANY ungraded task (not self-generated)."""
    loaded = self._get_loaded_litellm_name()
    if not loaded:
        return False
    db = await get_db()
    # Check if any ungraded task was NOT generated by the loaded model
    cursor = await db.execute(
        "SELECT COUNT(*) FROM tasks WHERE status = 'ungraded'"
    )
    total = (await cursor.fetchone())[0]
    if total == 0:
        return False
    # Check generating_model in context — need to parse JSON
    cursor2 = await db.execute(
        "SELECT context FROM tasks WHERE status = 'ungraded'"
    )
    for row in await cursor2.fetchall():
        ctx = json.loads(row["context"] or "{}")
        if ctx.get("generating_model") != loaded:
            return True  # at least one task can be graded
    return False
```

`_has_pending_overhead_needs` updated:

```python
async def _has_pending_overhead_needs(self) -> bool:
    # Ungraded tasks waiting for grading
    db = await get_db()
    cursor = await db.execute(
        "SELECT COUNT(*) FROM tasks WHERE status = 'ungraded'"
    )
    if (await cursor.fetchone())[0] > 0:
        return True

    # Pending todos (suggestion calls)
    try:
        from src.infra.db import get_todos
        todos = await get_todos(status="pending")
        return len(todos) > 0
    except Exception:
        return False
```

**Revised fan-in scenario with fix:**

```
Steps 3.1-3.5 all generated by model X → finish as "ungraded"
get_ready_tasks() → 0 (dependents blocked)
ensure_gpu_utilized([]):
  no main work, overhead needs exist, model X loaded
  → _loaded_model_can_grade() → False (all self-generated)
  → swap to fastest general model Y
  → on_model_swap → Trigger 1 → drain all 5 grades
  → 3.1-3.5 promoted to completed → 3.11 unblocks
Total delay: swap time (~30s) only. No 60s idle waste.
```

---

## Summary of Changes by File

| File | Changes |
|---|---|
| `src/core/state_machine.py` | New states (`ungraded`, `waiting_human`, `skipped`), remove old states, enforced transitions via `transition_task()` |
| `src/infra/db.py` | New columns (attempts, grade_attempts, next_retry_at, retry_reason, failed_in_phase), migration SQL, updated `get_ready_tasks` (next_retry_at filter, ungraded exclusion), `accelerate_retries()` replaces `wake_sleeping_tasks()`, remove sleeping queue functions |
| `src/core/orchestrator.py` | Unified retry via `compute_retry_timing()`, simplified watchdog (remove paused/sleeping handlers, add stuck-ungraded/failed-dep-cascade), `drain_ungraded_tasks()` called from `on_model_swap`, idle path grades with loaded model, updated `_check_mission_completion` for new states, `on_model_swap` ordering (accelerate → drain) |
| `src/core/llm_dispatcher.py` | Remove `GradeQueue`/`PendingGrade`/`drain_grades_if_*`, update `ensure_gpu_utilized` (don't bail when loaded model can't grade, add `_loaded_model_can_grade`), `on_model_swap` calls `drain_ungraded_tasks` |
| `src/core/router.py` | Updated grading prompt (structured binary YES/NO), read `excluded_models` + `grade_excluded_models` from task context, `grade_response` still routed through dispatcher OVERHEAD |
| `src/agents/base.py` | Worker completion → `ungraded` (deferred) or `completed` (immediate grade), store `generating_model` + `worker_completed_at` in context |
| `src/app/telegram_bot.py` | Status display updates, remove sleeping/paused/rejected references, add ungraded display, update DLQ retry to be phase-aware, update button handlers |
| `src/infra/dead_letter.py` | `retry_dlq_task` reads `failed_in_phase` for phase-aware re-enable, resets exclusions + backoff |
| `src/workflows/engine/hooks.py` | Remove `_schema_retry_count` from context, use shared `attempts` counter |
