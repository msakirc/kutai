# KutAI Orchestrator X-Ray: Model Routing, Concurrency & Resource Management

> Architecture reference document. Living document — update as system evolves.

---

## Current Architecture (As-Is)

### Request Flow: Task Creation to Inference

1. **Task arrives** via Telegram or scheduled cron → inserted into SQLite `tasks` table (`status='pending'`)
2. **Main loop** (`orchestrator.py`) runs every ~3s:
   - `get_ready_tasks(limit=8)` — fetches pending tasks, filters by dependency completion, ordered by `priority DESC, created_at ASC`
   - Partitions: only `assistant` agent_type is "cloud-safe"; at most 1 "local" task runs concurrently; rest deferred
3. **`process_task()`**: Claims task → classifies (LLM call) → dispatches to agent
4. **Agent `execute()`**: Builds `ModelRequirements` → enters ReAct loop → each iteration calls `call_model()`
5. **`call_model()`**: Runs `select_model()` (scoring pipeline) → iterates top 5 candidates → for local: acquires GPU slot + ensures model loaded → for cloud: checks rate limits → executes via litellm
6. **Grading**: After agent completes, `grade_response()` defers to GradeQueue or grades immediately

### LLM Dispatch Categories

All LLM calls route through `LLMDispatcher` (`src/core/llm_dispatcher.py`):

| Category | Can swap? | GPU queue | Cloud fallback | Backpressure |
|----------|-----------|-----------|----------------|--------------|
| MAIN_WORK | YES | Full priority queue | Yes, unless local_only | Yes |
| OVERHEAD | NEVER | 20s wait, then cloud | Yes, error on total failure | No (propagates) |

### Call Sites

| File | Purpose | Category |
|------|---------|----------|
| `base.py` ReAct loop | Agent iteration | MAIN_WORK |
| `base.py` single-shot | Agent execution | MAIN_WORK |
| `base.py` self-reflection | Quality check | OVERHEAD (failure: skip, continue) |
| `task_classifier.py` | Task classification | OVERHEAD (failure: task stays pending) |
| `router.py` grade_response | Grading | OVERHEAD (via GradeQueue if deferred) |
| `shopping/_llm.py` | Shopping intelligence | OVERHEAD |
| `orchestrator.py` subtask classification | Subtask routing | OVERHEAD |
| `telegram_bot.py` message classification | User message routing | OVERHEAD (failure: user sees error) |
| `workflows/engine/hooks.py` | Artifact validation | OVERHEAD |
| `tools/vision.py` | Image analysis | MAIN_WORK |

### Scoring Pipeline (select_model)

**3-layer scoring:**
- **Layer 1: Eligibility (pass/fail)** — context, capabilities, function calling, vision, budget
- **Layer 2: Capability Gate** — reject if below `effective_min_score` (difficulty-based curve)
- **Layer 3: Ranking** — 5 dimensions × difficulty-weighted profiles + 3 post-composite multipliers

### GPU Management

- Single-slot priority queue (`GPUScheduler`)
- Model swap: drain in-flight (30s) → stop server → start new → wait healthy (up to 180s)
- Inference generation tracking prevents counter corruption on force-swap
- Atomic `swap_started_at` timestamp for stale-read detection
- Stale candidate detection after GPU wait (model may have changed)
- Idle unload after 1 minute
- Circuit breaker: 2 consecutive load failures → 5 minute cooldown
- Watchdog: crash detection (process exit) + hang detection (3× health failure, 503=not hung)

### Vision Support

- On-demand `--mmproj` loading (not loaded by default — saves ~876MB VRAM)
- `_vision_enabled` tracked alongside `_thinking_enabled`
- Vision toggle = full swap (respects budget, circuit breaker, inference drain)
- Vision task batching: +0.8 priority boost when mmproj loaded
- Classifier guard: `needs_vision=True` only for `visual_reviewer` agent

---

## Implemented Solutions (S1-S10)

### S1: Centralized LLM Dispatcher ✅
All call sites route through `dispatcher.request()`. MAIN_WORK can swap, OVERHEAD cannot. No direct `call_model()` outside the dispatcher.

### S2: Deferred Grading Queue ✅
GradeQueue drains on model swap, idle time, or when full (>20). Priority=1 for GPU. Batch of 3 per drain cycle. Already event-driven — no changes needed for new architecture.

### S3: Cloud Quota Management ✅
QueueProfile analyzes upcoming tasks (vision, tools, thinking, difficulty). Graduated availability scoring (smooth curve, no cliff-edge). Daily exhaustion is a hard gate.

### S4: Proactive GPU Loading ✅
`ensure_gpu_utilized()` loads best-fit model when GPU idle + tasks pending. Fires from main loop and on local model load failure (replacement load).

### S5: Model-Aware Task Ordering ✅
`_reorder_by_model_affinity()` boosts tasks matching loaded model by up to +0.9 priority. Never overrides 2+ priority gap. Vision batching when mmproj loaded.

### S6: Runtime State Tracking ✅
`ModelRuntimeState`: thinking_enabled, context_length, gpu_layers, measured_tps. Scorer uses runtime state for loaded model.

### S7: Provider Load Balancing ✅
Congested primary model nudges selection toward underutilized siblings.

### S8: Scoring Reorganization ✅
3 layers (eligibility → capability gate → ranking). 3 post-composite multipliers only.

### S9: Adaptive Timeouts ✅
TPS-based LLM timeout. GPU acquire capped to LLM timeout. Difficulty heuristic fallback.

### S10: Swap Budget ✅
Max 3 swaps per 5 minutes. Recorded synchronously (not in fire-and-forget path). Exemptions: local_only, priority>=9.

---

## P12: Blind Retry Architecture (NEW PROBLEM)

### Current State

The retry system is timer-based with no awareness of what changed:

```
call_model fails
  → backpressure queue: retry in 5s, 10s, 20s, 40s, 60s (blind timer)
  → 5 retries × 5 candidates × 2-3 attempts = up to 75 LLM attempts
  → all fail → task marked failed
  → orchestrator retry: 3 attempts
  → all fail → task paused
  → watchdog: resume every 10 min (blind timer), retry_count=0
  → repeats INDEFINITELY
```

**Problems:**
1. Nothing changes in 5 seconds. Retries burn GPU/cloud for zero benefit.
2. Rate limit resets happen at a known timestamp — we just don't schedule a callback.
3. GPU becomes free at a known moment (release() fires) — we don't signal sleeping tasks.
4. Circuit breaker expires at a known timestamp — we don't signal.
5. Classifier failures eat the task retry budget even though the task was never attempted.
6. No distinction between transient (rate limit) and structural (auth error) failures.
7. Paused→pending watchdog resets retry_count — infinite cycle with no escalation.

### Failure Mode Inventory

Every failure in `call_model()` and what resolves it:

| Failure | Error signature | Resolves when | Signal exists? |
|---------|----------------|---------------|----------------|
| GPU busy | `GPU queue timeout` | Current inference finishes | **NO** — release() only grants next GPU waiter |
| Rate limited (RPM) | `429`, `rate limit` | RPM window resets (~60s) | **NO** — `_header_rpm_reset_at` stored but no callback |
| Rate limited (TPM) | `tokens per minute` | TPM window resets (~60s) | **NO** — `_header_tpm_reset_at` stored but no callback |
| Daily exhausted | `Daily limit exhausted` | UTC midnight (`rpd_reset_at`) | **NO** — timestamp stored, no callback |
| Circuit breaker | `Failed to load local model` | Cooldown expires (300s) | **NO** — `_restart_cooldown_until` stored, no callback |
| No models matched | `No models available!` | Any capacity change | **NO** — composite of all signals |
| Auth/billing | `api key`, `unauthorized` | Manual fix (user updates .env) | **NEVER** — external action only |
| Timeout | `Timeout on {model}` | Transient or structural | Watchdog restarts crashed server |
| Connection error | `Connection refused/reset` | Server restarts | **YES** — watchdog → `on_model_swap` |
| Model swap during wait | `Model swapped during GPU wait` | Automatic (next candidate) | N/A — handled inline |
| Server 503 loading | `ServiceUnavailable` | Model finishes loading | Watchdog treats as not-hung |

### OVERHEAD Failure Behavior

| Call site | Failure behavior | Problem? |
|-----------|-----------------|----------|
| Classifier | RuntimeError → task stays `pending` → re-fetched in 3s | **YES — silent 3s retry storm** |
| Self-reflection | Caught → agent continues without | Clean |
| Grading | GradeQueue defers → drains on signal | Clean (already event-based) |
| Shopping LLM | Propagates → agent fails → retry chain | Goes through sleeping queue |
| Workflow hooks | Propagates → step fails → retry chain | Goes through sleeping queue |
| Telegram classify | User sees error | Acceptable (user retries manually) |

---

## S11: Signal-Based Sleeping Queue (NEW SOLUTION)

### Overview

Replace the blind-timer backpressure queue with an event-driven sleeping queue. Tasks that fail don't retry on a timer — they sleep until a signal indicates something changed, with an escalating safety-net timer as fallback.

### Task Status Lifecycle

```
pending → processing → completed
                    → sleeping (failed, waiting for capacity signal)
                    → failed → DLQ (after exhausting all tiers)

sleeping → pending (woken by signal or timer)
        → pending (manual /retry)
        → DLQ (exhausted all timer tiers)
```

New status `sleeping` is distinct from:
- `pending` — ready to run, orchestrator picks up immediately
- `paused` — stopped by user or workflow timeout, needs manual action
- `sleeping` — failed but will auto-retry on signal or timer

### Sleeping Queue State (per task, persisted in DB)

```sql
ALTER TABLE tasks ADD COLUMN sleep_state TEXT;
-- JSON: {
--   "timer_tier": 0,           -- 0=10m, 1=30m, 2=1h, 3=2h (cap)
--   "signal_failures": 0,      -- signal-wakes that failed (max 3 then signals ignored)
--   "last_error_category": "", -- gpu_busy | rate_limited | no_model | auth | ...
--   "sleeping_since": "",      -- timestamp
--   "next_timer_wake": ""      -- timestamp of next safety-net timer
-- }
```

### Timer Tiers

| Tier | Interval | Cumulative | After failure |
|------|----------|------------|---------------|
| 0 | 10 min | 10 min | Advance to tier 1 |
| 1 | 30 min | 40 min | Advance to tier 2 |
| 2 | 1 hour | 1h 40min | Advance to tier 3 |
| 3 (cap) | 2 hours | 3h 40min | → DLQ |

Timer fires: set task to `pending`. If it fails again, back to `sleeping` at next tier.

### Signal System

**4 new signals + 2 existing:**

| Signal | Source | When | New? |
|--------|--------|------|------|
| `gpu_available` | `gpu_scheduler.release()` | GPU slot freed with no waiters | **NEW** |
| `rate_limit_reset` | `rate_limiter` | Scheduled at `_header_rpm_reset_at` | **NEW** |
| `circuit_breaker_reset` | `local_model_manager` | Scheduled at `_restart_cooldown_until` | **NEW** |
| `daily_reset` | `rate_limiter` | Scheduled at `rpd_reset_at` | **NEW** |
| `model_swap` | `on_model_swap()` | After successful swap | Existing |
| `quota_restored` | `on_quota_restored()` | Rate limit headers show capacity | Existing |

**All signals do the same thing:** wake ALL sleeping tasks (set status to `pending`).

No reason-based filtering — a task classified as `gpu_busy` might also benefit from a `rate_limit_reset` (it never tried cloud because GPU timed out first). Any capacity change is worth re-evaluation.

### Signal-Wake Failure Limit

To prevent signal storms (rapid swaps waking the same broken tasks repeatedly):

```
signal_failures < 3  → signal wakes the task
signal_failures >= 3 → signals ignored, timer only

Signal wake → task fails → signal_failures += 1 (timer_tier unchanged)
Timer wake → task fails → timer_tier += 1 (signal_failures unchanged)
```

3 signal-wakes + 4 timer tiers = 7 total attempts before DLQ. Total worst case: ~3.5 hours.

### Classifier Failure Handling

Classifier failures (OVERHEAD) should NOT count against the task's retry budget. The task was never dispatched to an agent — classifying it is a prerequisite, not an attempt.

```
Classifier fails for a pending task:
  → task goes to sleeping with last_error_category = "classifier_unavailable"
  → wakes on signal (any capacity change)
  → does NOT increment retry_count
  → does NOT increment timer_tier (first attempt hasn't happened yet)
```

### Integration Points

**Orchestrator main loop:**
```
get_ready_tasks()           ← fetches 'pending' only, NOT 'sleeping'
process_task()              ← on failure, set status='sleeping' + sleep_state
                            ← classifier failure: sleeping without retry_count++
check_sleeping_timers()     ← NEW: scan sleeping tasks, wake if timer expired
```

**Signal emitters (fire `wake_sleeping_tasks()`):**
```
gpu_scheduler.release()         → if no waiters: wake_sleeping_tasks()
rate_limiter.update_from_snap() → schedule wake at reset timestamp
local_model_manager             → schedule wake at circuit_breaker_cooldown
on_model_swap()                 → wake_sleeping_tasks() (existing signal)
on_quota_restored()             → wake_sleeping_tasks() (existing signal)
```

**`wake_sleeping_tasks()` implementation:**
```python
async def wake_sleeping_tasks():
    """Wake all sleeping tasks that haven't exhausted signal-wake limit."""
    db = await get_db()
    # Parse sleep_state JSON, check signal_failures < 3
    # Set status='pending' for eligible tasks
    # Increment signal_failures in sleep_state
```

**Grade queue:** Unchanged. Already event-driven. Drains on model swap and idle. Does not interact with sleeping queue (disjoint populations: grade queue = completed tasks pending quality score, sleeping queue = failed tasks pending retry).

**Backpressure queue:** REMOVED. Replaced entirely by sleeping queue. `call_model()` no longer enqueues to backpressure — it raises RuntimeError which propagates to `process_task()` which sets the task to sleeping.

### DLQ Integration

When a sleeping task's timer reaches tier 3 (2h cap) and the timer-wake retry also fails:

```
Task fails after tier 3 timer wake
  → quarantine_task() (existing DLQ function)
  → status = 'failed'
  → DLQ entry created with error_category
  → Telegram notification via todo reminder piggyback
```

**DLQ + Todo reminder integration:**
The todo reminder cron (`0 6,8,10,12,14,16,18 * * *`) already pings every 2 hours. Add a DLQ section to the reminder message:

```
📝 *Yapılacaklar*
  1. Buy groceries
  2. Call dentist

⚠️ *Dikkat Gerektiren Görevler* (DLQ)
  ❌ Task #2280 — component_specs (3.5h, auth_failure)
     [Retry] [Skip] [Details]
```

Same notification, same cadence. No extra pings. The `[Retry]` button calls `retry_dlq_task()` (existing) which sets task to `pending` and resets everything.

### Manual Backdoor

**`/retry <task_id>`** works on any status:
- `sleeping` → set to `pending`, reset timer_tier + signal_failures + retry_count
- `failed` (in DLQ) → `retry_dlq_task()`, resolve DLQ entry, set to `pending`
- `paused` → set to `pending`, reset retry_count

This is your "I pushed a fix, retry this one" command.

### KutAI Restart Behavior

Sleeping tasks persist in DB (status=`sleeping`, sleep_state JSON).

On restart:
```
Orchestrator starts
  → first watchdog cycle scans 'sleeping' tasks
  → sets ALL to 'pending' (restart = state change)
  → resets timer_tier to 0 (fresh start — new code, clean VRAM)
  → signal_failures preserved (structural problems persist across restarts)
  → orchestrator picks them up via normal get_ready_tasks()
```

No flood — they enter the pending pool in one watchdog cycle, compete via normal priority ordering.

### call_model() Change

```python
# BEFORE (backpressure queue):
async def call_model(...):
    for candidate in candidates[:5]:
        ...try each...
    # All failed → enqueue to backpressure
    await bp_queue.enqueue(call_id, priority, last_error, _retry_call)

# AFTER (immediate failure):
async def call_model(...):
    for candidate in candidates[:5]:
        ...try each...
    # All failed → raise immediately, let process_task handle sleeping
    raise ModelCallFailed(
        call_id=call_id,
        last_error=last_error,
        error_category=_classify_error(last_error),
    )
```

`process_task()` catches `ModelCallFailed` and sets the task to `sleeping` with appropriate `sleep_state`.

### Summary of Removed Components

| Component | Replaced by |
|-----------|-------------|
| `BackpressureQueue` | Sleeping queue (DB-persistent) |
| `backpressure.py` | Signal emitters + `wake_sleeping_tasks()` |
| Timer-based retry (5s/10s/20s/40s/60s) | Signal-based wake + escalating safety timer |
| Watchdog paused→pending reset | Sleeping queue timer tiers |
| `MAX_RETRY_ATTEMPTS = 5` | 3 signal-wakes + 4 timer tiers |

---

## Implementation Order

| Phase | Change | Effort | Dependencies |
|-------|--------|--------|-------------|
| 1 | Add `sleeping` status + `sleep_state` column | Small | None |
| 2 | New `ModelCallFailed` exception + process_task handler | Small | Phase 1 |
| 3 | Remove backpressure queue from call_model | Medium | Phase 2 |
| 4 | Signal emitters (gpu_available, rate_limit_reset, etc.) | Medium | Phase 1 |
| 5 | `wake_sleeping_tasks()` + sleeping timer scanner | Medium | Phase 1, 4 |
| 6 | Classifier failure → sleeping without retry_count++ | Small | Phase 2 |
| 7 | DLQ integration (tier 3 → quarantine) | Small | Phase 2 |
| 8 | Todo reminder DLQ piggyback | Small | Phase 7 |
| 9 | `/retry` command for sleeping/failed/paused tasks | Small | Phase 1 |
| 10 | Remove `backpressure.py` and all references | Small | Phase 3 |

---

## Key Metrics to Track

- **Swaps per hour**: Target < 6
- **Overhead LLM calls per task**: Target < 1.5
- **GPU idle time with pending tasks**: Target 0%
- **Cloud quota utilization**: Target even spread across providers
- **Grade queue depth**: Monitor, alert if > 30
- **Sleeping queue depth**: Monitor (normal: 0-3, concerning: >10)
- **Signal-wake success rate**: % of signal wakes that lead to task completion
- **Timer-wake vs signal-wake ratio**: Most wakes should be signal-driven
- **DLQ entries per day**: Target 0, alert if > 3
- **Task completion latency p50/p95**: Track improvement over time
