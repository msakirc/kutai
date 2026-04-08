# Retry Pipeline & Agent Iteration X-Ray

The retry pipeline controls what happens when agent tasks fail, exhaust iterations, or encounter infrastructure problems. It spans four layers: the agent loop (iterations), the orchestrator (task attempts), the grading system (quality gate), and the model scheduler (swap-aware retry). This document covers all four and how they interact.

## Why It Exists

Agent tasks fail for different reasons that need different responses. A quality failure (bad output) needs a better model. An infrastructure failure (process crash) needs a simple restart. An iteration exhaustion (ran out of turns) needs more budget or fewer guard rejections. Treating all these the same — incrementing one counter and hoping for the best — wastes retry budget and lets garbage propagate as "completed" results.

The retry pipeline gives each failure type its own budget, its own escalation path, and its own retry strategy. It also makes the agent loop itself more efficient: guards don't waste iterations, parallel tools execute in one turn, and exhaustion is classified by root cause so the retry strategy matches the actual problem.

## The Four Layers

```
┌─ TASK ATTEMPT (orchestrator) ─────────────────────────────────┐
│  worker_attempts: 0-6 (quality failures)                      │
│  infra_resets: 0-3 (infrastructure crashes)                   │
│  Independent budgets, independent DLQ paths                   │
│                                                                │
│  ┌─ AGENT ITERATION (base.py loop) ────────────────────────┐  │
│  │  max_iterations: 3-12 (per agent, dynamic boost)        │  │
│  │  Sub-iteration corrections (guards don't burn iters)    │  │
│  │  Parallel tool execution (multi_tool_call)              │  │
│  │                                                          │  │
│  │  ┌─ SUB-ITERATION (inner correction loop) ──────────┐  │  │
│  │  │  MAX_SUB_CORRECTIONS = 3 per iteration            │  │  │
│  │  │  Format corrections, guard rejections             │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  ┌─ LLM API RETRY (within single call) ─────────────┐  │  │
│  │  │  Router-level, invisible to agent                 │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## RetryContext — The Single Source of Truth

All retry state is owned by `RetryContext` in `src/core/retry.py`. No code outside RetryContext directly increments attempt counters or manipulates retry fields.

```python
RetryContext.from_task(task)         # reconstruct from DB row + context JSON
retry_ctx.record_failure("quality")  # single entry point for all failures
retry_ctx.to_db_fields()             # serialize for UPDATE tasks SET ...
retry_ctx.to_context_patch()         # serialize for task.context JSON merge
retry_ctx.to_checkpoint()            # serialize for agent checkpoint
```

**Task-level fields** (DB columns): `worker_attempts`, `infra_resets`, `max_worker_attempts`, `grade_attempts`, `max_grade_attempts`, `next_retry_at`, `retry_reason`, `failed_in_phase`, `exhaustion_reason`.

**Model tracking** (task.context JSON): `failed_models`, `grade_excluded_models`.

**Iteration-level** (agent checkpoint): `iteration`, `format_corrections`, `consecutive_tool_failures`, `model_escalated`, `guard_burns`, `useful_iterations`.

Backwards compatibility: `from_task()` reads both old (`attempts`, `max_attempts`) and new (`worker_attempts`, `max_worker_attempts`) column names. Old DBs work without migration.

## Three Failure Types

Every failure is exactly one of three types:

### Quality

Output is bad or missing. The agent ran but the result isn't acceptable.

**Triggers**: grade FAIL, disguised failure (post-hook detects bad output), agent returns `failed`, exhaustion with `tool_failures` reason, execution timeout.

**Counter**: `worker_attempts` (0 → max_worker_attempts, default 6).

**Retry behavior**:
- Attempts 1-2: immediate retry, same model allowed
- Attempt 3: immediate, exclude previously failed models (`failed_models`)
- Attempt 4+: delayed 10 min, exclude models, difficulty += 2 per attempt past 3
- At max: terminal → `failed` → DLQ with `failed_in_phase="worker"`

### Infrastructure

Task didn't finish — process crash, hung agent, model load failure during execution.

**Triggers**: watchdog detects stuck-in-processing (>5 min), module resumption finds interrupted tasks.

**Counter**: `infra_resets` (0 → 3, independent from quality budget).

**Retry behavior**:
- Always immediate retry
- NEVER excludes models (the model wasn't the problem)
- NEVER bumps difficulty
- At 3 resets: terminal → `failed` → DLQ with `failed_in_phase="infrastructure"`

### Availability

Couldn't execute at all — all models busy, rate limited, GPU queue timeout.

**Counter**: uses `worker_attempts` but with separate backoff via `last_avail_delay`.

**Retry behavior**: doubling backoff (60s → 120s → ... → 2h cap). Signal-wakes (`accelerate_retries`) reset backoff on model swap. Terminal after 2h cap.

## Agent Iteration Loop

The agent loop in `base.py` has three structural layers:

### Outer loop (iterations)

```python
for iteration in range(start_iteration, effective_max_iterations):
```

`effective_max_iterations` = `self.max_iterations` (per-agent, 3-8) × `iteration_budget_boost` (1.0 default, 1.5 on budget-exhaustion retry), capped at 12.

Each iteration is ONE round of: LLM call → parse → action. Actions are: `tool_call`, `multi_tool_call`, `final_answer`, `clarify`, `decompose`, `ask_agent`.

### Inner correction loop (sub-iterations)

```python
while sub_corrections <= MAX_SUB_CORRECTIONS:  # 3 per iteration
    response = LLM call
    parsed = parse response
    
    if format correction needed:
        sub_corrections += 1; continue
    
    if guard fires (hallucination, search-required, blocked clarify):
        sub_corrections += 1; continue
    
    if validation rejects final_answer:
        sub_corrections += 1; continue
    
    break  # proceed to action handling
```

Guards that used to burn full iterations now re-prompt within the same iteration. An agent with 5 iterations that hits 2 guard rejections still has all 5 iterations for tool work.

**Category A guards** (sub-iteration, don't burn iterations):
- Hallucination guard (final_answer without tool use, iteration < 2)
- Search-required guard (final_answer without data-fetching, search_depth set)
- Blocked clarification (clarify when may_need_clarification=false)
- Format corrections (JSON parse failures)
- Custom validation (_validate_response)
- Task-type validation (validate_task_output)

**Category B guards** (burn iterations, by design):
- Arg validation errors (LLM needs new tool/args)
- Unknown action (LLM needs to learn schema)

### Parallel tool execution

When the LLM returns multiple tool calls (native function calling or JSON `multi_tool_call` action), they execute in one iteration:

```
tool_calls received
    ├── read-only (CACHEABLE_READ_TOOLS): asyncio.gather
    ├── then: side-effect (everything else): sequential
    └── all results in one user message → 1 iteration consumed
```

Unknown tools default to side-effect (safe). Parallel failures don't block — surviving results are still injected. `consecutive_tool_failures` increments by number of failures in the batch.

## Exhaustion Handling

When iterations run out, the agent returns `status: "exhausted"` (not `"completed"`). The exhaustion block classifies WHY:

| Reason | Condition | Retry strategy |
|--------|-----------|---------------|
| `"budget"` | useful_iterations >= 50% of max | 1st time: retry with 1.5x iterations (cap 12). 2nd+: standard quality retry |
| `"guards"` | guard_burns >= 50% of max | Retry with `suppress_guards=True` (skips Category A guards) |
| `"tool_failures"` | consecutive_tool_failures >= 3 | Standard quality retry (exclude model, bump difficulty) |

The orchestrator handles `"exhausted"` BEFORE `"failed"` in the status routing. It's not a DB status — the orchestrator immediately transitions to `"pending"` (retry) or `"failed"` (terminal).

**Dynamic iteration budget**: When a task retries after budget exhaustion, `task.context["iteration_budget_boost"] = 1.5` is set. The agent reads this at loop start and adjusts `effective_max_iterations`. A coder (normally 8) gets 12. An analyst (normally 5) gets 7.

## Mid-Task Model Escalation

When an agent hits `TOOL_FAILURE_ESCALATION_THRESHOLD` (3) consecutive tool failures AND iteration >= 3, it escalates model requirements:

1. `reqs.escalate()` — difficulty += 2, prefer_quality = True
2. Next LLM call may select a better model (possibly triggering a swap)
3. Message history is **trimmed** for the better model:
   - Keep: system prompt, task description, successful tool results, last error
   - Strip: old model's reasoning, guard corrections, format retries
   - Inject: "A previous attempt encountered difficulties. You have a fresh start."
4. `model_escalated = True` prevents re-escalation

The trim ensures the better model doesn't inherit a polluted conversation full of the weaker model's failed attempts.

## Swap-Aware Retry Scheduling

Four mechanisms prevent wasted retries against excluded models:

### Task pickup filtering

After `get_ready_tasks()`, the orchestrator partitions tasks:
```python
runnable = [t for t in ready if not _should_defer_for_loaded_model(t, loaded_model)]
deferred = [t for t in ready if _should_defer_for_loaded_model(t, loaded_model)]
ready = runnable or deferred  # fallback: run deferred if nothing else
```

A task with `worker_attempts >= 3` and `loaded_model` in `failed_models` is deferred — it would just fail immediately on model selection.

### Affinity reordering

`_reorder_by_model_affinity()` gives up to +0.9 priority boost for tasks matching the loaded model. Tasks that would reject the loaded model get `fit = 0.0` — they don't jump ahead of tasks that CAN use the loaded model.

### Proactive loading exclusions

`_find_best_local_for_batch()` counts how many tasks each local model can serve. Retry tasks with `worker_attempts >= 3` and the model in `failed_models` don't count toward that model's score. This prevents loading a model that the majority of pending retries will reject.

### Quality retry acceleration on swap

`accelerate_retries()` wakes both availability AND quality retries when a new model loads. Quality-retried tasks waiting out their 10-minute delay get woken early — the model change gives them a fresh shot.

## Terminology

| Term | Meaning | Counter |
|------|---------|---------|
| `worker_attempts` | Times the agent executed and failed quality | DB column, 0-6 |
| `infra_resets` | Times the watchdog reset a stuck task | DB column, 0-3 |
| `grade_attempts` | Times the grader failed to parse | DB column, 0-3 |
| `format_corrections` | JSON parse retries within agent | Checkpoint, per execution |
| `model_escalated` | Mid-task quality bump fired | Checkpoint, boolean |
| `guard_burns` | Sub-iteration guard corrections | Checkpoint, per execution |
| `useful_iterations` | Iterations that did real tool work | Checkpoint, per execution |
| `exhaustion_reason` | Why iterations ran out | DB column |
| `attempts_snapshot` | worker_attempts frozen at DLQ entry | Dead letter table |

## Files

| File | What it does |
|------|-------------|
| `src/core/retry.py` | `RetryContext`, `RetryDecision`, `compute_retry_timing()`, model exclusion helpers |
| `src/agents/base.py` | Agent loop (outer iterations, inner corrections, parallel tools, exhaustion), `_check_sub_iteration_guards`, `_trim_for_escalation`, `_partition_tool_calls` |
| `src/core/orchestrator.py` | Task pickup, affinity reordering, watchdog (infra_resets), exhaustion handler, all failure paths via RetryContext |
| `src/core/grading.py` | Grade FAIL → RetryContext.record_failure("quality"), drain_ungraded_tasks |
| `src/core/llm_dispatcher.py` | `_find_best_local_for_batch` (exclusion-aware), `accelerate_retries` trigger on swap |
| `src/infra/db.py` | Schema (worker_attempts, infra_resets, exhaustion_reason), `get_ready_tasks`, `accelerate_retries` |
| `src/infra/dead_letter.py` | `quarantine_task(attempts_snapshot=)`, phase-aware `retry_dlq_task` |

## DB Schema

```sql
-- Task retry columns
worker_attempts INTEGER DEFAULT 0,       -- quality failures (was: attempts)
max_worker_attempts INTEGER DEFAULT 6,   -- quality cap → DLQ (was: max_attempts)
infra_resets INTEGER DEFAULT 0,          -- infrastructure crashes, independent budget
grade_attempts INTEGER DEFAULT 0,        -- grading parse failures
max_grade_attempts INTEGER DEFAULT 3,    -- grading cap → waive grading
next_retry_at TIMESTAMP,                 -- NULL = immediate, future = delayed
retry_reason TEXT,                        -- "quality" | "availability" | "timeout" | "infrastructure"
failed_in_phase TEXT,                     -- "worker" | "grading" | "infrastructure"
exhaustion_reason TEXT,                   -- "budget" | "guards" | "tool_failures"

-- Dead letter queue
attempts_snapshot INTEGER DEFAULT 0,     -- frozen worker_attempts at quarantine (was: retry_count)
```

Deprecated (kept for migration compat, never written): `retry_count`, `max_retries`, `attempts`, `max_attempts`.

## Design Decisions

### Why separate infra from quality budgets

A task stuck in processing (agent crashed, model failed to load) is an infrastructure problem. Burning quality retry budget for it means a task that hits 3 infrastructure resets + 3 quality failures reaches DLQ after only 3 real quality tries. Separate budgets: 6 quality + 3 infra = 9 total chances, all deserved.

### Why guards don't burn iterations

An analyst with 5 iterations that hits hallucination guard + search-required guard has 3 iterations left for work. With sub-iteration corrections, it keeps all 5. The LLM cost of re-prompting within an iteration is less than the cost of exhaustion + retry.

### Why parallel tool calls

The LLM naturally returns multiple tool calls when it wants to read several files or search multiple sources. Executing them in one iteration instead of burning 3 iterations on 3 sequential reads is a 3x efficiency gain. Only read-only tools run in parallel — side-effect tools stay sequential to preserve order.

### Why exhaustion is not completion

A task that ran out of iterations may have scraped together a partial result from the last assistant message. But it never actually produced a `final_answer` action — it was forced to stop. Treating this as "completed" lets garbage propagate to dependent tasks. The `"exhausted"` status enters the retry pipeline with a reason-aware strategy: if it exhausted because of real work (budget), give it more iterations; if guards ate the budget, suppress them; if tools kept failing, try a different model.

### Why trim on escalation

When a better model takes over mid-task, it inherits a conversation full of the weaker model's failed reasoning. The better model wastes remaining iterations trying to understand what went wrong instead of solving the task fresh. Trimming keeps signal (tool results = real data) and strips noise (failed reasoning = wrong mental model).

### Why swap-aware scheduling

Without it: task with `exclude_models=["ModelA"]` gets picked → `_reorder_by_model_affinity` boosts it (matches loaded ModelA) → agent runs → router rejects ModelA → falls to cloud or triggers swap → wasted budget. With it: task stays deferred until a suitable model is loaded, or runs on cloud if it's the only work available.
