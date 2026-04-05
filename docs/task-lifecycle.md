# Task Lifecycle & Quality Gate

Every task in KutAI follows a two-phase lifecycle: the agent executes work (worker phase), then a different model evaluates quality (grading phase). Dependents wait until grading confirms the output is acceptable.

## Why It Exists

Long-running workflows (i2p, 200 steps) monopolize the GPU. The agent produces output, but grading never happens because the same model is loaded and can't self-grade. Without a quality gate, bad output silently propagates to dependent tasks, wasting the entire downstream chain.

The system also had 5 separate retry mechanisms (agent retry, schema retry, sleeping queue, backpressure pauses, grade queue) with different counters, different escalation paths, and no coordination. A task could retry infinitely through the paused→pending watchdog loop.

## Task States

9 states, no more:

| State | Meaning | Who acts next |
|---|---|---|
| `pending` | Ready for pickup (or waiting for `next_retry_at`) | Orchestrator |
| `processing` | Agent running | Agent |
| `ungraded` | Agent done, awaiting quality check | Grader |
| `completed` | Done, quality confirmed or waived | Terminal |
| `failed` | Permanently failed, in DLQ | Human |
| `waiting_subtasks` | Decomposed, children running | Children |
| `waiting_human` | Needs clarification or approval | Human |
| `cancelled` | Cancelled by user | Terminal |
| `skipped` | Dependency-skipped | Terminal |

States that were removed: `paused` (replaced by `pending` + `next_retry_at`), `sleeping` (same), `needs_clarification` (renamed to `waiting_human`), `needs_review` (was never persisted), `rejected` (merged into `cancelled`).

All state transitions go through `transition_task()` in `state_machine.py`, which validates the transition is legal before writing to the DB.

## Two-Phase Execution

A task has two phases on the same DB row:

```
pending → processing → ungraded → completed
   ↑          │            │
   │          │            ├─ grade PASS → completed
   │          │            ├─ grade FAIL → pending (retry worker)
   │          │            └─ grading unavailable → backoff, stay ungraded
   │          │
   │          ├─ agent succeeds, different model loaded → grade immediately → completed
   │          └─ agent fails → pending (retry)
   │
   └── retry from grade FAIL or agent failure
```

**Worker phase**: The agent executes the task. When it finishes, the system checks if a different model is loaded. If yes, grade immediately. If no (same model = can't self-grade), defer grading and set status to `ungraded`.

**Grading phase**: A different model evaluates the output using a structured binary prompt (YES/NO for RELEVANT, COMPLETE, VERDICT). The grade result determines: PASS → `completed`, FAIL → back to `pending` for retry.

Each phase has its own attempt budget:
- `attempts` / `max_attempts` (default 6) — worker phase quality failures
- `grade_attempts` / `max_grade_attempts` (default 3) — grading phase quality failures

`grade_attempts` resets to 0 when a task returns to `pending` (new worker output needs fresh grading).

## Two Failure Types

Every failure is one of two types. No exceptions.

### Quality

Output is bad or missing. The agent ran (or the grader ran) but the result isn't acceptable.

Triggers: agent returns failed, post-hook detects disguised failure, schema validation fails, grade VERDICT=FAIL, grade parse error.

Retry behavior:
- Attempts 1-2: immediate retry, same model allowed
- Attempt 3: immediate retry, MUST exclude previously failed models (`context.failed_models`)
- Attempt 4+: delayed retry (10 min), exclude failed models, difficulty += 2 per attempt past 3
- At `max_attempts`: terminal → `failed` → DLQ

### Availability

Couldn't execute at all. Model unavailable, GPU busy, rate limited.

Triggers: `ModelCallFailed` (all models exhausted), GPU scheduler timeout, rate limit hit, all eligible models excluded by quality failures.

Retry behavior:
- Does NOT increment `attempts` — availability is not the task's fault
- Doubling backoff: 1m → 2m → 4m → 8m → ... → 2h cap (`context.last_avail_delay`)
- Signal wakes (`accelerate_retries`) pull `next_retry_at` to now and reset backoff
- At 2h cap still failing: terminal → `failed` → DLQ (~5 hours total)

Phase-preserving: availability failures return to current phase. Worker availability → `pending`. Grading availability → stays `ungraded`.

## Natural Batching at Fan-In Points

The i2p workflow has a diamond pattern: fan-out → parallel work → fan-in synthesis step. The `ungraded` state creates natural grading batches at fan-in points:

```
Phase 3: Requirements
  3.1 ─┐
  3.2 ─┤
  3.3 ─┼──► 3.11 (depends on all)
  3.4 ─┤
  3.5 ─┘

Steps 3.1-3.5 execute in parallel → each finishes as "ungraded"
Step 3.11 depends on all → blocked (ungraded ≠ completed)
get_ready_tasks() → 0 → ensure_gpu_utilized loads grader model
Model swap → drain_ungraded_tasks → grade all 5 in one batch
All pass → 3.11 unblocks → next phase
```

Swap count for full i2p: ~15-20 grade swaps (one per fan-in gate), not 200 (one per step).

## Grading Triggers

Three existing call sites handle grading. No background workers needed.

| Trigger | When | How |
|---|---|---|
| `on_model_swap` | Any model swap completes | Queries `ungraded` tasks, grades those not generated by the new model |
| Main loop idle path | No tasks running, model loaded | Same as above, grades with whatever model is loaded |
| `ensure_gpu_utilized` | No main work, no model loaded | Detects ungraded tasks as overhead need, loads grader model → swap → Trigger 1 fires |

Special case: when the loaded model is the generating model for ALL ungraded tasks (can't self-grade), `ensure_gpu_utilized` detects this and proactively swaps to a different model instead of waiting 60s for the idle unloader.

## Structured Binary Grading

Replaces the old 1-5 numeric scale. Small LLMs handle YES/NO reliably; they can't calibrate a 5-point scale.

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

Parse priority:
1. VERDICT → use directly
2. If VERDICT unparseable but RELEVANT+COMPLETE parse → derive (both YES → PASS)
3. Nothing parses → grader incapable → quality failure of the grading phase

Score mapping for analytics: PASS = 4.0, FAIL = 2.0 (compatible with skill extraction threshold).

## DLQ and Recovery

Entry: `attempts >= max_attempts` (worker quality) or availability backoff exhausted (2h cap × repeated failures). The `failed_in_phase` column records which phase caused the failure.

Exit: manual only — `/dlq retry`, `/dlq discard`, or inline button.

Phase-aware re-enable: `/dlq retry` reads `failed_in_phase`:
- `"worker"` → `pending` (re-execute from scratch)
- `"grading"` → `ungraded` (retry grading only, preserve worker output)

Reset on re-enable: `failed_models`, `excluded_models`, `grade_excluded_models`, `last_avail_delay` all cleared. `attempts` and `grade_attempts` preserved (quality budget is lifetime). `generating_model` preserved (prevents self-grading).

## Watchdog

Runs every 10 orchestrator cycles. Simplified from the old system:

| Check | Condition | Action |
|---|---|---|
| Stuck processing | `processing` > 5 min | Quality failure, retry |
| Stuck ungraded | `ungraded` > 30 min | Safety net: promote to `completed` with `quality_score=NULL` |
| Overdue retry | `pending`, `next_retry_at` > 1h in past | Clear `next_retry_at`, make immediately eligible |
| Failed deps | All non-skipped deps `failed` | Cascade failure (never clears deps) |
| Waiting subtasks | All children terminal | `completed` if any completed, `failed` if all failed |
| Stale waiting_human | 4h → 24h → 48h → 72h | Escalation: nudge → remind → urgent → cancel |

Removed: paused handler (infinite retry_count=0 loop), sleeping timer scan, sleeping restart wake.

## Design Decisions

### Why `ungraded` blocks dependents

A task that grades FAIL gets retried. If dependents already started, their inputs are wrong — cascading waste. Blocking at the `ungraded` gate ensures only quality-confirmed output flows downstream. The cost is one model swap per fan-in point (~30s each).

### Why binary grading, not 1-5

Small local models can't reliably distinguish 2/5 from 3/5. A single point of miscalibration could reset good work or pass garbage. YES/NO is a binary signal that even 3B models handle reliably. The RELEVANT/COMPLETE sub-questions give redundancy — if VERDICT parses wrong, we can derive from the sub-questions.

### Why availability doesn't increment attempts

Availability failures aren't the task's fault. Burning the quality retry budget because the GPU was busy would punish the task for infrastructure issues. Availability has its own escalation (doubling backoff → DLQ after ~5h).

### Why the grade queue was killed

The in-memory `GradeQueue` was lost on restart. With `ungraded` as a DB state, all grading state is persistent. On restart, `ungraded` tasks are immediately visible and will be graded on the next model swap.

### Why `paused` and `sleeping` were merged into `pending` + `next_retry_at`

Both meant "try again later." `sleeping` had proper escalation (tiers → DLQ). `paused` had none (infinite 10-min loop with retry_count reset). Unifying them into one mechanism with one escalation path eliminates the infinite loop.

## Files

| File | What it does |
|---|---|
| `src/core/state_machine.py` | 9-state enum, transition validation, `transition_task()` |
| `src/core/retry.py` | `compute_retry_timing()`, `RetryDecision`, model exclusion helpers |
| `src/core/grading.py` | `grade_task()`, `apply_grade_result()`, `drain_ungraded_tasks()`, structured prompt parsing |
| `src/core/orchestrator.py` | Main loop (idle grading), watchdog, `_check_mission_completion`, availability backoff on `ModelCallFailed` |
| `src/core/llm_dispatcher.py` | `on_model_swap` (accelerate + drain), `ensure_gpu_utilized` (grade swap when self-generated), `_loaded_model_can_grade` |
| `src/agents/base.py` | Worker completion → `ungraded` or immediate grade, retry-based model exclusion in `_build_model_requirements` |
| `src/infra/db.py` | Schema migration (7 new columns), `get_ready_tasks` (next_retry_at filter), `accelerate_retries` |
| `src/infra/dead_letter.py` | Phase-aware `retry_dlq_task` |

## DB Schema (task columns)

New columns added by the unified lifecycle:

```sql
attempts INTEGER DEFAULT 0,           -- worker quality failures, never resets
max_attempts INTEGER DEFAULT 6,       -- worker hard cap → DLQ
grade_attempts INTEGER DEFAULT 0,     -- grading quality failures, resets on worker retry
max_grade_attempts INTEGER DEFAULT 3, -- grading hard cap → waive grading
next_retry_at TIMESTAMP,              -- NULL = immediately eligible, future = delayed
retry_reason TEXT,                     -- "quality" or "availability"
failed_in_phase TEXT,                  -- "worker" or "grading" — which phase hit DLQ
```

Deprecated (kept for backward compat, not written to): `retry_count`, `max_retries`, `sleep_state`.

## Context Fields

These fields in the task's `context` JSON drive the retry and grading logic:

| Field | Set by | Used by |
|---|---|---|
| `generating_model` | base.py (on entering `ungraded`) | drain_ungraded_tasks (self-grade prevention) |
| `worker_completed_at` | base.py (on entering `ungraded`) | Watchdog stuck-ungraded check (future) |
| `tools_used_names` | base.py | apply_grade_result (skill extraction) |
| `failed_models` | retry.py (on quality failure) | base.py (model exclusion after 3 attempts) |
| `grade_excluded_models` | drain_ungraded_tasks (on grader parse fail) | drain_ungraded_tasks (skip failed graders) |
| `last_avail_delay` | orchestrator (on availability failure) | compute_retry_timing (doubling backoff) |

## What Agents Need to Know

**If you're modifying the agent completion flow:**
- After an agent returns `status="completed"`, the system either grades immediately (if a different model is loaded) or defers to `ungraded`. Don't bypass this — it's the quality gate.
- `apply_grade_result` handles everything: state transitions, skill extraction, model feedback, Telegram notifications. Never duplicate this logic.

**If you're modifying retry/failure handling:**
- All failures are `quality` or `availability`. There is no third type. Use `compute_retry_timing` from `retry.py`.
- Availability failures DON'T increment `attempts`. They use `context.last_avail_delay` for doubling backoff.
- Quality failures DO increment `attempts` and add the failing model to `context.failed_models`.

**If you're modifying the grading prompt:**
- Keep it structured binary (YES/NO). Don't switch to numeric scales — small models can't handle them.
- The parse fallback (derive from RELEVANT+COMPLETE) is intentional redundancy. Don't remove it.

**If you're modifying the watchdog:**
- The stuck-ungraded safety net (30 min → promote with NULL score) prevents deadlocks. Don't remove it.
- Failed dependency handling cascades only when ALL non-skipped deps failed. Partial failure = wait.

## Known Limitations

1. **`transition_task()` not enforced everywhere yet**: Many raw `update_task(status=...)` calls in the orchestrator bypass validation. These should be migrated to use `transition_task()` to catch invalid transitions at runtime. The failure/retry handlers now use unified `attempts` counters, but still call `update_task` directly rather than `transition_task`.

2. **Stale comments**: Some files still reference "sleeping queue" in comments. The function calls are correct — just the comments are outdated.

## Fixed (2026-04-05)

- `completed_at` timestamps now use `strftime("%Y-%m-%d %H:%M:%S")` everywhere (was `isoformat()` in 3 call sites)
- All failure handlers (disguised failure, agent-failed, general exception) use unified `attempts`/`max_attempts` counters (was `retry_count`/`max_retries`, creating a dual-budget bug allowing 9 retries instead of 6)
- Workflow backpressure infinite loop eliminated — unified retry has terminal condition via `compute_retry_timing`
- Watchdog stuck-ungraded check uses `worker_completed_at` from context (was `started_at`, causing premature promotion for long-running agents)
- Immediate grading injects actual agent result into task dict (was passing stale DB snapshot with empty result, causing every immediate grade to auto-pass)
- Trivial/empty task output fails grading instead of auto-passing
- `_loaded_model_can_grade` checks `grade_excluded_models` (was ignoring exclusions, causing stuck-ungraded tasks)
- `failed_in_phase` set on all failure paths for correct DLQ recovery
