# Agent Iteration & Retry Pipeline Overhaul

**Date**: 2026-04-07
**Status**: Design approved, pending implementation plan
**Risk**: HIGH — touches orchestrator, agent loop, model routing, retry pipeline, DB schema
**Approach**: Surgical, sequenced changes — each independently testable and rollback-safe

## Problem Statement

The agent execution and retry system has accumulated terminology confusion and architectural gaps that cause:

1. **Iteration starvation**: 1 tool per iteration + guards burning iterations = agents exhaust budget before completing real work
2. **Silent garbage completion**: Exhausted agents return `status: "completed"` with scraped partial output
3. **Shared attempt budget**: Infrastructure resets (watchdog) eat the same counter as quality failures
4. **Blind retry scheduling**: Retry tasks get picked/boosted for models they'll reject
5. **Polluted escalation context**: Better models inherit weaker model's failed reasoning
6. **Overloaded terminology**: "retry", "attempt", "escalation" each mean 2-3 different things

## Scope

### Files Modified

| File | Changes | Risk |
|------|---------|------|
| `src/agents/base.py` | Parallel tools, sub-iteration guards, exhaustion status, escalation trim, RetryContext integration | HIGH — main agent loop |
| `src/core/orchestrator.py` | Exhaustion handler, swap-aware pickup, infra vs quality separation, RetryContext | HIGH — task lifecycle |
| `src/core/retry.py` | RetryContext class, record_failure(), separate budgets | MEDIUM — new abstraction |
| `src/core/grading.py` | RetryContext integration, column renames | MEDIUM |
| `src/infra/db.py` | Schema migration, column renames, accelerate_retries expansion | MEDIUM |
| `src/infra/dead_letter.py` | Column rename, RetryContext integration | LOW |
| `src/core/state_machine.py` | No change needed — "exhausted" is an agent return value, not a DB status. Orchestrator handles it before any state transition. | NONE |
| `src/core/llm_dispatcher.py` | Exclusion-aware proactive loading | LOW |
| `src/app/config.py` | Rename ESCALATION_THRESHOLD | LOW |

### Files NOT Modified

- `src/core/router.py` — model selection/scoring untouched
- `src/models/local_model_manager.py` — swap execution untouched
- `src/models/capabilities.py` — task profiles untouched
- `src/app/telegram_bot.py` — UI untouched
- Agent subclasses (`src/agents/*.py`) — only base.py changes, subclasses inherit

## Design

### 1. RetryContext — Unified State Object

**File**: `src/core/retry.py` (extend existing file)

A single dataclass that owns ALL retry/iteration/escalation state. Replaces scattered field reads across orchestrator, grading, dead_letter.

```python
@dataclass
class RetryContext:
    # ── Task-level (persisted as DB columns) ──
    worker_attempts: int = 0          # renamed from: attempts
    infra_resets: int = 0             # NEW — watchdog/stuck resets
    max_worker_attempts: int = 6      # renamed from: max_attempts
    grade_attempts: int = 0
    max_grade_attempts: int = 3
    next_retry_at: str | None = None
    retry_reason: str | None = None   # "quality" | "availability" | "timeout" | "infrastructure"
    failed_in_phase: str | None = None

    # ── Model tracking (persisted in task.context JSON) ──
    failed_models: list[str] = field(default_factory=list)
    grade_excluded_models: list[str] = field(default_factory=list)

    # ── Iteration-level (persisted in checkpoint) ──
    iteration: int = 0
    max_iterations: int = 8
    format_corrections: int = 0       # renamed from: format_retries
    consecutive_tool_failures: int = 0
    model_escalated: bool = False     # renamed from: escalated
    guard_burns: int = 0              # NEW — iterations wasted by guards
    useful_iterations: int = 0        # NEW — iterations with real tool work

    # ── Exhaustion tracking (NEW) ──
    exhaustion_reason: str | None = None  # "budget" | "guards" | "tool_failures"
```

**Key methods**:

- `RetryContext.from_task(task: dict) -> RetryContext` — reconstruct from task record + context JSON
- `retry_ctx.to_db_fields() -> dict` — serialize task-level fields for `UPDATE tasks SET ...`
- `retry_ctx.to_checkpoint() -> dict` — serialize iteration-level fields for checkpoint
- `retry_ctx.to_context_patch() -> dict` — serialize model-tracking fields for context JSON merge
- `retry_ctx.record_failure(failure_type, model) -> RetryDecision` — single entry point for all failures; increments the correct counter, updates exclusions, returns retry/terminal decision
- `retry_ctx.record_guard_burn(guard_name) -> None` — track guard waste
- `retry_ctx.record_useful_iteration() -> None` — track real work

**Computed properties**:

- `effective_difficulty_bump: int` — `max(0, (worker_attempts - 3) * 2)` if `worker_attempts >= 4`
- `excluded_models: list[str]` — `list(failed_models)` if `worker_attempts >= 3` else `[]`
- `total_attempts: int` — `worker_attempts + infra_resets` (for display/logging only)

**Replaces**: The ~8 identical `attempts = (task.get("attempts") or 0) + 1` blocks in orchestrator.py, the manual context manipulation in grading.py, and the raw field reads in retry.py functions.

### 2. Parallel Tool Execution

**File**: `src/agents/base.py`

When the LLM returns multiple actions (native FC `tool_calls` or JSON `multi_tool_call`), execute them in one iteration.

**Two entry paths**:

A) **Native function calling** (Qwen3, Llama3.x, Phi4, DeepSeek, Mistral, all cloud):
   - `_parse_function_call_response()` currently takes only `tool_calls[0]`. Change to return all as `{"action": "multi_tool_call", "tools": [...]}` when `len(tool_calls) > 1`.

B) **JSON text models** (Gemma2, CodeLlama, etc.):
   - Add `multi_tool_call` to the advertised JSON action schema in system prompt.
   - `_parse_agent_response()` / `_normalize_action()` recognizes the new action type.
   - Weaker models may not use it — graceful degradation to single tool_call.

**Execution partitioning**:

```
tool_calls received
    ├── classify each tool:
    │     CACHEABLE_READ_TOOLS → read-only group
    │     SIDE_EFFECT_TOOLS → side-effect group
    │     unknown/unclassified → side-effect group (safe default)
    ├── read-only group: asyncio.gather(return_exceptions=True)
    ├── then: side-effect group: sequential, in order
    └── all results combined into one user message
```

**Message format**: Multiple `## Tool Result (tool_name → key_arg):` blocks in a single user message, followed by iteration counter.

**Failure handling**: `asyncio.gather(return_exceptions=True)` — if 2 of 3 tools fail, the 1 success result is still injected. `consecutive_tool_failures` increments by number of failures.

**Iteration accounting**: 1 multi-tool execution = 1 iteration consumed. `retry_ctx.useful_iterations` increments by number of successful tool calls (for metrics).

**Checkpoint**: No format change — checkpoints save messages and iteration counter. Multi-tool results are just longer user messages.

**Guard interaction**: Guards check `parsed.action`. A `multi_tool_call` action bypasses pre-answer guards (hallucination, search-required) since the LLM chose tools, not `final_answer`.

### 3. Guards as Sub-Iteration Corrections

**File**: `src/agents/base.py`

Guards that reject premature `final_answer` become sub-iteration corrections — they re-prompt within the same iteration number instead of consuming a new one.

**Category A — Sub-iteration guards** (re-prompt, don't consume iteration):
- Hallucination guard (final_answer without tool use)
- Search-required guard (final_answer without data-fetching)
- Blocked clarification (suppressed clarify attempt)
- Custom validation (`_validate_response`)
- Task-type validation (`validate_task_output`)
- Format corrections (already sub-iteration in spirit, fix the loop)

**Category B — Iteration-consuming guards** (keep as-is):
- Arg validation errors (LLM needs to see error and pick different tool/args)
- Unknown action (LLM needs to learn the schema)

**Implementation**: Inner correction loop around LLM call + parse + guard check:

```python
for iteration in range(start_iteration, effective_max_iterations):
    sub_corrections = 0
    MAX_SUB_CORRECTIONS = 3

    while sub_corrections <= MAX_SUB_CORRECTIONS:
        response = await dispatcher.request(...)
        parsed = parse(response)

        correction = self._check_sub_iteration_guards(parsed, iteration, ...)
        if correction:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": correction.message})
            sub_corrections += 1
            retry_ctx.record_guard_burn(correction.guard_name)
            continue  # inner loop — same iteration number

        break  # no guard fired — proceed to action handling

    # ... tool execution / final_answer handling as before ...
```

Cap: 3 sub-corrections per iteration. After that, accept whatever the LLM produced (fallback to existing behavior). Note: each sub-correction makes a new LLM call (the correction message needs a fresh response), so sub-corrections have inference cost — but far less than a wasted full iteration that also includes tool execution overhead.

### 4. Exhaustion Handling — "exhausted" Status

**Files**: `src/agents/base.py`, `src/core/orchestrator.py`

When iterations run out, the agent returns `status: "exhausted"` instead of `"completed"`.

**Agent return** (base.py, replaces current exhaustion block):

```python
return {
    "status": "exhausted",
    "result": last_assistant or "",
    "exhaustion_reason": retry_ctx.exhaustion_reason,
    "guard_burns": retry_ctx.guard_burns,
    "useful_iterations": retry_ctx.useful_iterations,
    "model": used_model,
    "cost": total_cost,
    "iterations": effective_max_iterations,
}
```

**Exhaustion reason classification** (automatic in RetryContext):
- `"budget"` — useful_iterations >= 70% of max. Real work, just ran out of room.
- `"guards"` — guard_burns >= 50% of max. Guards ate the budget.
- `"tool_failures"` — consecutive_tool_failures >= TOOL_FAILURE_ESCALATION_THRESHOLD at exit.

**Orchestrator handler** — reason-aware retry strategy:

| Reason | Retry strategy | Model change? | Iteration boost? |
|--------|---------------|---------------|-----------------|
| `"budget"` (1st time) | Immediate retry | No — model was fine | Yes: +50% (capped at 12) |
| `"budget"` (2nd+ time) | Standard quality retry | Yes — exclude + bump difficulty | No |
| `"guards"` | Immediate retry | No | No — suppress Category A guards (hallucination, search-required, blocked clarify, validations) on retry via `task.context["suppress_guards"] = True` |
| `"tool_failures"` | Standard quality retry | Yes — exclude + bump difficulty | No |

**Dynamic iteration budget**: `task.context["iteration_budget_boost"]` read at loop start:

```python
boost = task_ctx.get("iteration_budget_boost", 1.0)
effective_max = min(int(self.max_iterations * boost), 12)  # hard cap
```

**Grading interaction**: Exhausted results with partial output > 200 chars MAY still be sent to grading. If grader says PASS, task completes despite exhaustion. If FAIL, retry pipeline with the reason-aware strategy above.

**"exhausted" is NOT a DB task status** — it's an agent return value. The orchestrator immediately transitions to "pending" (retry) or "failed" (terminal). No state machine change needed.

### 5. Separate Infrastructure vs Quality Attempts

**Files**: `src/core/orchestrator.py`, `src/core/retry.py`, `src/infra/db.py`

Split the unified `attempts` counter into `worker_attempts` (quality) and `infra_resets` (infrastructure) with independent budgets.

**Infrastructure events** (increment `infra_resets`):
- Watchdog: task stuck in "processing" > 5 min
- Process crash: no result, no timeout
- Model load failure during execution

**Quality events** (increment `worker_attempts`) — unchanged:
- Grade FAIL
- Disguised failure
- Agent returns failed/exhausted with tool_failures
- Execution timeout

**Independent budgets and behaviors**:

| Aspect | Quality | Infrastructure |
|--------|---------|---------------|
| Counter | `worker_attempts` | `infra_resets` |
| Max | 6 (configurable) | 3 |
| Timing (early) | Immediate | Always immediate |
| Timing (late) | 10-min delay at attempts >= 3 | Always immediate |
| Model exclusion | At worker_attempts >= 3 | Never |
| Difficulty bump | At worker_attempts >= 4 | Never |
| DLQ reason | `failed_in_phase="worker"` | `failed_in_phase="infrastructure"` |

**DB migration**: `ALTER TABLE tasks ADD COLUMN infra_resets INTEGER DEFAULT 0`. Rename `attempts` to `worker_attempts`, `max_attempts` to `max_worker_attempts`.

### 6. Swap-Aware Retry Scheduling

**Files**: `src/core/orchestrator.py`, `src/core/llm_dispatcher.py`, `src/infra/db.py`

Four fixes at identified gaps:

**Fix A — Exclusion-aware task pickup** (orchestrator.py):

After `get_ready_tasks()`, partition into runnable vs deferred based on loaded model and task exclusions:

```python
loaded_model = self.local_manager.current_litellm_name
runnable, deferred = [], []
for task in ready:
    ctx = parse_context(task)
    excluded = ctx.get("failed_models", [])
    if task.get("worker_attempts", 0) >= 3 and loaded_model in excluded:
        deferred.append(task)
    else:
        runnable.append(task)
tasks_to_process = runnable or deferred
```

No extra DB query — just in-memory filtering of already-fetched tasks.

**Fix B — Affinity reordering respects exclusions** (orchestrator.py):

In `_reorder_by_model_affinity()`, set affinity_score = 0.0 for tasks where loaded model is in their exclude list (worker_attempts >= 3).

**Fix C — Proactive loading considers exclusions** (llm_dispatcher.py):

In `_find_best_local_for_batch()`, skip models excluded by retry tasks when counting "servable" tasks.

**Fix D — Quality retries accelerated on swap** (db.py):

Expand `accelerate_retries()` WHERE clause from `retry_reason = 'availability'` to `retry_reason IN ('availability', 'quality')`. When a new model loads, quality-retried tasks waiting out their 10-min delay get woken early.

### 7. Mid-Iteration Escalation Context Reset

**File**: `src/agents/base.py`

On model escalation (TOOL_FAILURE_ESCALATION_THRESHOLD reached), trim message history to keep signal, strip noise.

**Keep**:
- System prompt (first message)
- Original task description (first user message)
- Successful tool results (user messages containing `## Tool Result` without error prefix)
- Most recent tool error (context for why escalation happened)

**Strip**:
- Old model's assistant messages (failed reasoning)
- Failed tool attempts where same tool was retried
- Format correction exchanges
- Guard rejection exchanges

**Inject** escalation context message:
> "A previous attempt encountered difficulties. Tool results above contain valid data. You have a fresh start with better capabilities. Iterations remaining: N."

**Trigger**: Only on `model_escalated` transition. Saved to checkpoint for crash recovery.

### 8. Terminology Cleanup

**Renames across codebase**:

| Current | New | Files affected |
|---------|-----|---------------|
| `attempts` (DB) | `worker_attempts` | db.py, orchestrator.py, grading.py, retry.py, dead_letter.py |
| `max_attempts` (DB) | `max_worker_attempts` | same |
| `retry_count` (dead_letter) | `attempts_snapshot` | dead_letter.py |
| `format_retries` (base.py) | `format_corrections` | base.py |
| `MAX_FORMAT_RETRIES` (base.py) | `MAX_FORMAT_CORRECTIONS` | base.py |
| `escalated` (base.py) | `model_escalated` | base.py |
| `ESCALATION_THRESHOLD` (base.py) | `TOOL_FAILURE_ESCALATION_THRESHOLD` | base.py, config.py |
| `retry_count` (DB, deprecated) | DROP column | db.py migration |
| `max_retries` (DB, deprecated) | DROP column | db.py migration |

**Log message patterns**:

| Current | New |
|---------|-----|
| `retry {format_retries}/{MAX}` | `format-correction {n}/{MAX}` |
| `retry {attempts}/{max}` | `worker-retry {n}/{max}` |
| `resetting (attempt {n}/{max})` | `infra-reset {n}/3` |
| `Escalating: '{old}' -> '{new}'` | `model-escalation: '{old}' -> '{new}'` |

## DB Migration Plan

Single migration function in `db.py:_migrate_schema()`:

```sql
-- Phase 1: Add new columns
ALTER TABLE tasks ADD COLUMN infra_resets INTEGER DEFAULT 0;
ALTER TABLE tasks ADD COLUMN exhaustion_reason TEXT;

-- Phase 2: Rename columns (SQLite 3.25+)
ALTER TABLE tasks RENAME COLUMN attempts TO worker_attempts;
ALTER TABLE tasks RENAME COLUMN max_attempts TO max_worker_attempts;

-- Phase 3: Dead letter rename
ALTER TABLE dead_letter_tasks RENAME COLUMN retry_count TO attempts_snapshot;

-- Phase 4: Drop deprecated columns (SQLite doesn't support DROP COLUMN before 3.35)
-- For safety: leave retry_count and max_retries as dead columns rather than
-- risking data loss on older SQLite. They are not read anywhere after RetryContext.
```

## Implementation Sequencing

The plan must enforce this ordering so each step is independently testable:

1. **RetryContext class + terminology renames** — pure refactor, no behavior change
2. **DB migration** — add columns, rename columns
3. **Separate infra vs quality attempts** — behavior change, testable in isolation
4. **Guards as sub-iterations** — agent loop change, testable with existing tests
5. **Parallel tool execution** — agent loop change, independent from retry pipeline
6. **Exhaustion handling** — depends on RetryContext + guards (steps 1, 4)
7. **Swap-aware scheduling** — depends on column renames (step 2)
8. **Escalation context reset** — depends on terminology rename (step 1)

Each step produces a working system. No step requires a later step to function. Any step can be reverted independently.

## Testing Strategy

Each section needs:
- **Unit tests**: RetryContext serialization round-trip, compute_retry_timing with new counters, guard sub-iteration loop bounds
- **Integration tests**: task lifecycle from creation through exhaustion and retry, parallel tool execution with mixed read/write tools
- **Regression tests**: existing orchestrator tests must pass after each step (no behavior change for non-retry paths)
- **Manual verification**: run KutAI, trigger a multi-tool task, observe iteration usage in logs

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| RetryContext serialization breaks existing tasks | `from_task()` handles missing fields with defaults; old tasks work as-is |
| Parallel tool execution causes race conditions | Read-only tools only in parallel; side-effect tools always sequential |
| Guard sub-iteration loop infinite-loops | Hard cap: MAX_SUB_CORRECTIONS = 3 per iteration |
| Dynamic iteration boost causes runaway cost | Hard cap: 12 iterations max regardless of boost |
| Column rename breaks running tasks | Migration is idempotent; RetryContext reads both old and new column names during transition |
| Swap-aware filtering starves deferred tasks | Fallback: if no runnable tasks, process deferred anyway |
