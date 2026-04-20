# Post-Hook Task Extraction — Design Spec

**Date:** 2026-04-21
**Status:** Approved design, awaiting implementation plan
**Owner:** Beckman (`packages/general_beckman/`) + agents (`src/agents/`)

---

## 1. Motivation

Two LLM-driven post-hooks currently live outside Beckman's queue:

- **Grading** (`src/core/grading.py::drain_ungraded_tasks`). Reaches Beckman via an opportunistic hook on `LLMDispatcher.on_model_swap` — fires only when a local model actually swaps. If main_work keeps reusing the same model (intentional stickiness), `ungraded` tasks accumulate indefinitely.
- **Per-task LLM artifact summary** (`src/workflows/engine/hooks.py::queue_llm_summary` → `drain_pending_summaries`). No drain caller survives Task 13's orchestrator collapse; every >3KB artifact enqueues to a dead table.

Both are OVERHEAD work that should flow through Beckman's queue as first-class tasks, selected by Fatih Hoca like any other LLM call, dispatched through the pipe. Keeping them outside the queue created:

- **Layer violation.** Dispatcher reaches into grading state; it's supposed to be `ask → select → load → call → retry`.
- **Agent layer violation.** Agent self-transitions to `ungraded` status (`src/agents/base.py:2086`), encoding a lifecycle state it shouldn't own.
- **Silent breakage.** Orphaned drains die unnoticed. i2p mission 34 saw `unknown_status:ungraded` because the agent returned a status Beckman's router didn't recognise.

The fix: treat grading and summary as regular tasks Beckman schedules, with the source task's completion gated on their verdicts.

## 2. Scope

**In scope:**
- Grading as first-class task (blocking verdict, can reject source for retry).
- Per-task LLM summary as first-class task (blocking enrichment, structural fallback on failure).
- Sequential execution: grade first, summary only on grade pass.
- Beckman owns all scheduling; dispatcher becomes a pure pipe.

**Out of scope (deferred):**
- Cloud pool-pressure shared primitive (separate design).
- Pool-aware priority scheduling for grading (no cloud preference, no swap-budget hinting — relies on Fatih Hoca's existing overhead-stickiness).
- Phase summary extraction — currently mechanical (structural concatenation, not LLM), already cheap and working; leave inline.
- A general "post-hook-task" abstraction for future kinds. YAGNI until a third kind appears.

## 3. Architecture

### 3.1 Layers

| Layer | Responsibility |
|---|---|
| Agent | Produce output. Return `{status: "completed", result, cost, iterations, generating_model}`. Never transition task status. |
| Beckman rewrite | Decide which post-hooks a completed task needs (`determine_posthooks` policy). Emit `Complete + RequestPostHook(...)` actions. |
| Beckman apply | Enqueue post-hook task rows. Write `_pending_posthooks` in source context. Handle verdict actions. |
| Beckman on_task_finished | Route source verdicts → post-hook verdicts when post-hook task finishes. |
| Dispatcher | Pure pipe. No grading/summary hooks. |
| `local_model_manager` | Emit swap events to Beckman, not dispatcher. |
| Agents `GraderAgent` / `ArtifactSummarizerAgent` | Thin wrappers over existing `grade_task()` / `_llm_summarize()`; registered in `AGENT_REGISTRY`. |

### 3.2 Post-hook kinds

- `"grade"` — blocking verdict. Can reject (source → retry). Single per source task.
- `"summary:<artifact_name>"` — blocking enrichment. One per output artifact > 3KB. 3 failures → structural fallback, that artifact proceeds. Spawned only after grade passes.

Identifiers are stored in the source's `_pending_posthooks` list. Source reaches `completed` only when the list is empty. Example for a task producing two large artifacts:

```
_pending_posthooks = ["grade"]
  → grade passes
_pending_posthooks = ["summary:search_results", "summary:reviews_compiled"]
  → both summaries land in any order
_pending_posthooks = []
  → source → completed
```

### 3.3 Data flow — happy path

```
Agent returns completed
  ↓
Beckman.on_task_finished(source_id, result)
  ↓
post_execute_workflow_step(source, result)      # artifact storage, schema validation, etc. (unchanged)
  ↓
route_result → [Complete(source_id, ...)]
  ↓
rewrite_actions:
  needed = determine_posthooks(source, ctx, result)   # e.g. ["grade"] (or [] for mechanical/shopping)
  if needed: emit [Complete, RequestPostHook(source_id, kind=first_kind)]
  ↓
apply_actions:
  Complete: update_task(source_id, status="ungraded", _pending_posthooks=needed, result=...)
  RequestPostHook(grade): add_task(agent_type="grader", mission_id=source.mission_id,
                                    context={source_task_id, generating_model, excluded_models: [gen_model]})
```

Post-hook task is picked by `next_task()` like any other task. Orchestrator dispatches via `get_agent("grader").execute(task)`. GraderAgent reads source from DB, calls `grade_task(source_task)`, returns verdict dict.

```
Beckman.on_task_finished(grade_task_id, verdict_result)
  ↓
route_result (normal) → [Complete(grade_task_id, ...)]
  ↓
rewrite: detect grade verdict → emit [Complete(grade_task_id), PostHookVerdict(source_id, kind="grade", passed=True/False, raw=verdict)]
  ↓
apply:
  Complete: marks the grade task itself as completed (regular).
  PostHookVerdict(grade, passed=True):
    remove "grade" from source._pending_posthooks
    for each source output artifact with stored value > 3KB:
      append "summary:<artifact_name>" to _pending_posthooks
      enqueue ArtifactSummarizerAgent task (one per artifact)
    if _pending_posthooks empty (no large artifacts):
      update_task(source_id, status="completed")
  PostHookVerdict(grade, passed=False):
    update_task(source_id, status="pending", error=<verdict>, worker_attempts+=1)
    exclude_models += [generating_model]
    # no summary to cancel (grade-first sequential)
  PostHookVerdict(summary:<name>, passed=True, raw=<summary_text>):
    store "<name>_summary" artifact
    remove "summary:<name>" from source._pending_posthooks
    if empty: update_task(source_id, status="completed")
  PostHookVerdict(summary:<name>, passed=False):
    fall back to structural summary (already stored in post_execute hook)
    remove "summary:<name>" from source._pending_posthooks
    if empty: update_task(source_id, status="completed")
```

### 3.4 Sequence diagram (grade pass → summary)

```
Main Task T — agent returns completed
  T.status: in_progress → ungraded
  T._pending_posthooks = ["grade"]
  Grade Task G enqueued (agent_type=grader)

Grade Task G runs (Fatih Hoca picks OVERHEAD model, excludes T.generating_model)
  G returns {status: completed, verdict: pass}
  → PostHookVerdict(T, "grade", True)
    T._pending_posthooks = [] (grade removed)
    If T.output > 3KB:
      Summary Task S enqueued (agent_type=artifact_summarizer)
      T._pending_posthooks = ["summary"]
    Else:
      T.status: ungraded → completed (dependents unblock)

Summary Task S runs
  S returns {status: completed, summary_text: "..."}
  → PostHookVerdict(T, "summary", True, raw=summary_text)
    Store "<artifact>_summary" artifact
    T._pending_posthooks = []
    T.status: ungraded → completed (dependents unblock)
```

### 3.5 Sequence diagram (grade fail → retry)

```
Main Task T — agent returns completed
  T.status → ungraded, _pending_posthooks = ["grade"]

Grade Task G runs
  G returns {status: completed, verdict: fail}
  → PostHookVerdict(T, "grade", False, raw=verdict_reason)
    T.status: ungraded → pending
    T.worker_attempts += 1
    T.exclude_models += [T.generating_model]
    T.error = verdict_reason
    T._pending_posthooks = []
    # Summary was never spawned — nothing to cancel

Beckman picks T again → agent runs with excluded generating_model → produces new output
  (new output → new grade → potentially new summary, independent cycle)
```

### 3.6 Sequence diagram (grade pass → summary fails 3x)

```
Grade Task G: pass (as 3.4)
Summary Task S: fails (LLM error / degenerate output)
  → S retried by standard Beckman _retry_or_dlq (not a special path)
  → After 3 attempts, S goes to DLQ
  → DLQ write triggers PostHookVerdict(T, "summary", False) from the DLQ path,
    OR Beckman.apply recognises summary-DLQ and synthesises the verdict.
  → Structural summary (already stored by post_execute_workflow_step) stands.
  → T._pending_posthooks = [], T.status: ungraded → completed.
```

### 3.7 New actions & data types

Added to `packages/general_beckman/src/general_beckman/result_router.py`:

```python
@dataclass(frozen=True)
class RequestPostHook:
    """Spawn a post-hook task (grader or artifact_summarizer) for a source."""
    source_task_id: int
    kind: str            # "grade" | "summary"
    source_ctx: dict     # needed by apply for task construction

@dataclass(frozen=True)
class PostHookVerdict:
    """Apply the result of a completed post-hook task back to the source."""
    source_task_id: int
    kind: str            # "grade" | "summary"
    passed: bool
    raw: dict            # verdict / summary text / error details
```

`Action` union extended.

### 3.8 `determine_posthooks` policy (Beckman)

```python
def determine_posthooks(task: dict, task_ctx: dict, result: dict) -> list[str]:
    agent_type = task.get("agent_type", "")
    if agent_type in {"mechanical", "shopping_pipeline", "grader", "artifact_summarizer"}:
        return []   # mechanical + post-hook tasks themselves never get post-hooks

    # Workflow-step tasks that opt in via context flag or default-on for LLM tasks
    needs = []
    if task_ctx.get("requires_grading", True):  # default true for non-mechanical
        needs.append("grade")
    # Summary decision deferred: only scheduled AFTER grade pass, and only
    # if output > 3KB. Handled in PostHookVerdict(grade, passed=True).
    return needs
```

Summary spawn condition moved into the grade-pass verdict handler — keeps the policy function straightforward (`determine_posthooks` only decides what to spawn *immediately*).

### 3.9 Rewrite rule 1 extension (skip MissionAdvance)

`packages/general_beckman/src/general_beckman/rewrite.py`:

```python
# Existing guard (workflow_advance) extended:
payload_action = (task_ctx.get("payload") or {}).get("action")
agent_type = task.get("agent_type", "")
is_bookkeeping = (
    payload_action == "workflow_advance"
    or agent_type in {"grader", "artifact_summarizer"}
)
if isinstance(a, Complete) and task.get("mission_id") and not is_bookkeeping:
    return [a, MissionAdvance(...)]
```

### 3.10 Dispatcher cleanup

`src/core/llm_dispatcher.py`:
- Delete `async def on_model_swap(...)`.
- No other changes.

`src/models/local_model_manager.py:355`:
- Replace `get_dispatcher().on_model_swap(old_litellm, new_litellm)` with `general_beckman.on_model_swap(old_litellm, new_litellm)`.

`packages/general_beckman/src/general_beckman/__init__.py`:
- New `async def on_model_swap(old_model, new_model)` that calls `accelerate_retries("model_swap")`. No grading drain (grade is now queue-based).

### 3.11 New agents

`src/agents/grader.py`:

```python
class GraderAgent(BaseAgent):
    name = "grader"
    # No tools, no ReAct loop. Direct wrap.

    async def execute(self, task: dict) -> dict:
        from src.core.grading import grade_task, apply_grade_result
        ctx = parse_context(task)
        source_task_id = ctx["source_task_id"]
        source = await get_task(source_task_id)
        if source is None:
            return {"status": "failed", "error": f"source task {source_task_id} missing"}
        verdict = await grade_task(source)          # existing function
        # Do NOT apply_grade_result here — Beckman applies via PostHookVerdict.
        return {
            "status": "completed",
            "result": json.dumps(verdict),           # raw verdict dict for route layer
            "model": verdict.get("grader_model", "unknown"),
            "cost": verdict.get("cost", 0.0),
            "iterations": 1,
            "posthook_verdict": {                    # signal for rewrite rule
                "kind": "grade",
                "source_task_id": source_task_id,
                "passed": verdict.get("passed", False),
                "raw": verdict,
            },
        }
```

`src/agents/artifact_summarizer.py`:

```python
class ArtifactSummarizerAgent(BaseAgent):
    name = "artifact_summarizer"

    async def execute(self, task: dict) -> dict:
        from src.workflows.engine.hooks import _llm_summarize
        ctx = parse_context(task)
        source_task_id = ctx["source_task_id"]
        artifact_name = ctx["artifact_name"]
        text = ctx["text"]
        summary = await _llm_summarize(text, artifact_name)   # existing function
        passed = bool(summary) and len(summary) >= 50
        return {
            "status": "completed",
            "result": summary or "",
            "model": "artifact_summarizer",
            "cost": 0.0,
            "iterations": 1,
            "posthook_verdict": {
                "kind": "summary",
                "source_task_id": source_task_id,
                "passed": passed,
                "raw": {"summary": summary or "", "artifact_name": artifact_name},
            },
        }
```

Both registered in `src/agents/__init__.py::AGENT_REGISTRY`.

### 3.12 Route_result detects post-hook verdicts

Post-hook tasks return `result["posthook_verdict"]` alongside the standard fields. Rewrite layer transforms `Complete(posthook_task) + posthook_verdict` into `Complete(posthook_task) + PostHookVerdict(source, kind, passed, raw)`. The post-hook task itself completes normally (bookkeeping row marked completed).

## 4. Failure cases

| Case | Behaviour |
|---|---|
| Grade verdict = pass, no summary needed (output ≤ 3KB) | Source → completed. Dependents unblock. |
| Grade verdict = pass, summary needed | Summary task enqueued. Source stays `ungraded`. |
| Grade verdict = reject | Source → pending, worker_attempts++, excluded_models += generating_model. No summary spawned (sequential). |
| Grade task LLM error | Standard `_retry_or_dlq` retries the grade task with different model (via Fatih Hoca excluded-model logic). 3 failures → grade task DLQ → PostHookVerdict(grade, False, raw={"error": "grade_dlq"}) → source → failed permanently with `error="grade DLQ: no grader succeeded"`. Surfaces in `/dlq` for human review. |
| Grade parse failure | Existing `grade_task()` retries parses internally; post-hook task succeeds/fails as a unit. Same DLQ path. |
| Summary task LLM error | Standard retry. 3 failures → DLQ → PostHookVerdict(summary, False). Structural summary (already stored by post_execute_workflow_step) stands. Source → completed. |
| Summary returns degenerate output | `_llm_summarize()` flags degeneracy, `posthook_verdict.passed = False`. Same outcome as LLM error (structural fallback). |
| Source DLQ'd by sweep while grade in-flight | Grade verdict apply sees source status ≠ `ungraded`, logs & drops. Grade task itself completes normally. No cascade. |
| Shutdown with pending post-hook task | Both rows in DB. Restart picks up normally — standard queue recovery. |
| Source picked by `next_task()` before grade verdict applies | Impossible: `pick_ready_task` only selects `pending`; source is `ungraded`. |
| Post-hook task has `mission_id` → rewrite rule 1 fires MissionAdvance | Skip list (§3.9) excludes `agent_type ∈ {grader, artifact_summarizer}`. |
| Grade task's own progress ping spams user | `_send_step_progress` skips tasks with `agent_type ∈ {grader, artifact_summarizer}` (extend mechanical-skip list). Post-hook tasks are infrastructure, not user-visible. |

## 5. Migration

One-shot boot sweep in Beckman init (called from `next_task()` first cycle, behind a module-level flag like `cron_seed._seeded`):

```python
async def _migrate_stale_ungraded():
    # Rows from pre-refactor world: status='ungraded', no _pending_posthooks in context.
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, context FROM tasks WHERE status = 'ungraded'"
    )
    for row in await cursor.fetchall():
        ctx = json.loads(row["context"] or "{}")
        if ctx.get("_pending_posthooks"):
            continue   # already migrated
        ctx["_pending_posthooks"] = ["grade"]
        await update_task(row["id"], context=json.dumps(ctx))
        # Spawn the grader post-hook task for this source.
        await add_task(
            title=f"Grade task #{row['id']}",
            agent_type="grader",
            mission_id=row.get("mission_id"),
            context=json.dumps({
                "source_task_id": row["id"],
                "generating_model": ctx.get("generating_model", ""),
            }),
        )
```

Drop table `pending_llm_summaries` via a one-shot DDL in the same boot path:

```sql
DROP TABLE IF EXISTS pending_llm_summaries;
```

No back-compat shim for deleted drain functions. Clean break — after migration, they don't exist.

## 6. Deletions

- `src/core/grading.py::drain_ungraded_tasks` — queue replaces batch drain.
- `src/workflows/engine/hooks.py::drain_pending_summaries` — queue replaces batch drain.
- `src/workflows/engine/hooks.py::queue_llm_summary` call site at hooks.py:1125 (call is unreachable after llm_summary is queue-based; `RequestPostHook("summary")` is emitted from Beckman instead).
- `src/agents/base.py:2076-2097` — agent no longer self-transitions to `ungraded`; returns plain `completed` dict.
- `LLMDispatcher.on_model_swap` — method removed.
- Keep `grade_task()`, `apply_grade_result()`, `_llm_summarize()`: these are the units GraderAgent/ArtifactSummarizerAgent wrap.

## 7. Testing strategy

New tests:

- `tests/test_beckman_posthooks.py`
  - `test_complete_with_mission_emits_grade_request`
  - `test_grade_verdict_pass_completes_source_when_small_output`
  - `test_grade_verdict_pass_spawns_summary_when_large_output`
  - `test_grade_verdict_fail_retries_source_excludes_generating_model`
  - `test_summary_verdict_pass_stores_artifact_and_completes_source`
  - `test_summary_verdict_fail_falls_back_to_structural`
  - `test_grade_dlq_fails_source_permanently`
  - `test_mechanical_task_has_no_posthooks`
  - `test_shopping_pipeline_task_has_no_posthooks`
  - `test_posthook_task_does_not_emit_mission_advance`

- `tests/test_grader_agent.py`
  - `test_grader_reads_source_and_returns_verdict`
  - `test_grader_missing_source_returns_failed`

- `tests/test_artifact_summarizer_agent.py`
  - `test_summarizer_returns_posthook_verdict_shape`
  - `test_summarizer_degenerate_output_marked_failed`

- `tests/test_migration_ungraded_to_posthooks.py`
  - `test_stale_ungraded_row_gets_pending_posthooks_and_grader_task`
  - `test_already_migrated_row_unchanged`

Update existing tests:
- `tests/test_beckman_apply.py` — new actions covered.
- `tests/test_beckman_rewrite.py` — skip-list extension.
- Remove `tests/test_beckman_on_task_finished.py` ungraded short-circuit test (behaviour replaced by post-hook spawn).
- Whatever tests referenced `drain_ungraded_tasks` / `drain_pending_summaries` — delete or rewrite.

Full targeted suite (33+ tests today) must stay green; new tests push to ~55.

## 8. Open questions & deferred items

- **Summary-on-retry.** If a source retries after grade reject, does it re-spawn a summary on the new attempt's grade pass? Yes — each run's output is independent; each grade-pass cycle evaluates output size and may spawn its own summary. No memoisation.
- **Cloud preference for graders.** Fatih Hoca's existing overhead-stickiness handles model choice. No explicit cloud-prefer hint; revisit if we see swap-storm evidence under load.
- **`requires_grading` default.** Spec defaults to `True` for non-mechanical agents. If workflow JSON wants to disable grading for some steps, it can set `requires_grading: false` in step context. Mechanical tasks and post-hook tasks themselves never need grading (enforced by the allow-list in `determine_posthooks`).
- **Parallelism.** Different missions' post-hook tasks run in parallel as the queue allows. Within a single mission task's lifecycle, grade → summary is sequential (user-stated requirement).

## 9. Non-goals explicitly called out

- No shared pool-pressure primitive (separate future work).
- No changes to phase summary (structural, not LLM, not orphaned).
- No general post-hook abstraction beyond `grade` and `summary` kinds.
- No UI changes beyond the existing `/queue` / `/dlq` views inheriting post-hook tasks as normal rows.
- No priority adjustments for post-hook tasks yet — they compete on the same queue with normal priority; revisit if user-visible missions stall behind them.
