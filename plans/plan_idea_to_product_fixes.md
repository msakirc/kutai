# Idea-to-Product Engine -- Issues & Fixes

## Current Architecture

The idea-to-product workflow is a 17-phase pipeline (Phase -1 through Phase 15) defined in
`src/workflows/idea_to_product/idea_to_product_v2.json` with **328 steps**, **1 template**
(feature_implementation_template with 31 sub-steps), and **7 conditional groups**.

### Key Components

| Component | File | Role |
|---|---|---|
| **Workflow JSON** | `src/workflows/idea_to_product/idea_to_product_v2.json` | 328-step definition across 17 phases |
| **Loader** | `src/workflows/engine/loader.py` | Parses JSON into `WorkflowDefinition`, validates DAG |
| **Expander** | `src/workflows/engine/expander.py` | Converts steps to task dicts, maps agents, derives priority |
| **Runner** | `src/workflows/engine/runner.py` | Creates mission, expands steps, inserts tasks into DB |
| **Hooks** | `src/workflows/engine/hooks.py` | Pre/post execution: artifact injection, storage, phase completion |
| **Artifacts** | `src/workflows/engine/artifacts.py` | Blackboard-backed artifact store with tiered context budgets |
| **Quality Gates** | `src/workflows/engine/quality_gates.py` | Phase 9/10/13 gates (test pass rate, coverage, security, approval) |
| **Conditions** | `src/workflows/engine/conditions.py` | DSL evaluator for conditional groups |
| **Status** | `src/workflows/engine/status.py` | Phase progress tracking and Telegram-friendly formatting |
| **Policies** | `src/workflows/engine/policies.py` | Review cycle tracking, onboarding approval rules |
| **Pipeline Bridge** | `src/workflows/engine/pipeline_bridge.py` | Delegates Phase 8 feat.N steps to CodingPipeline |
| **Dispatch** | `src/workflows/engine/dispatch.py` | Regex-based intent detection for workflow triggers |
| **Orchestrator** | `src/core/orchestrator.py` | Main loop: get_ready_tasks, process_task, watchdog |
| **Telegram Bot** | `src/app/telegram_bot.py` | Entry point: /mission, classifier, /wfstatus, /resume |
| **DB** | `src/infra/db.py` | missions, tasks, workflow_checkpoints, blackboards tables |

### Execution Flow

1. **Trigger**: User sends message via Telegram. Classifier detects `workflow: "idea_to_product"`, or user explicitly uses `/mission --workflow`.
2. **Start**: `WorkflowRunner.start()` loads JSON, creates a mission, filters steps by context (skip Phase -1 for greenfield), expands steps to task dicts, resolves `depends_on` step IDs to DB task IDs, inserts ~300 tasks into DB.
3. **Execution**: Orchestrator main loop calls `get_ready_tasks()` which fetches ALL pending tasks, checks each one's `depends_on` array against completed/skipped task statuses, returns ready ones.
4. **Pre-hook**: `pre_execute_workflow_step()` should inject artifact context into task description.
5. **Agent call**: Router selects model, agent processes task, returns result.
6. **Post-hook**: `post_execute_workflow_step()` should store output artifacts, check conditional groups, trigger template expansion, track review cycles, check phase completion.
7. **Dependency cascade**: Completed tasks unblock downstream dependents on next `get_ready_tasks()` cycle.

### Phases

| Phase | Name | Steps | Notes |
|---|---|---|---|
| -1 | Existing Project Onboarding | 7 | Skipped for greenfield |
| 0 | Idea Capture & Clarification | 8 | Root: step 0.1 |
| 1 | Market & Competitive Research | 19 | Has 7 conditional groups |
| 2 | Product Strategy & Definition | 17 | |
| 3 | Requirements Engineering | 17 | |
| 4 | Architecture & Technical Design | 33 | |
| 5 | UX/UI Design Specification | 22 | |
| 6 | Project Planning & Sprint Setup | 10 | |
| 7 | Development Environment Setup | 47 | Largest static phase |
| 8 | Core Implementation | 3 | Template expands per feature |
| 9 | Comprehensive Testing | 25 | Quality gate: 95% pass, 60% coverage |
| 10 | Security Hardening | 18 | Quality gate: clean scan |
| 11 | Documentation | 18 | |
| 12 | Legal & Compliance | 9 | |
| 13 | Pre-Launch Preparation | 33 | Quality gate: all tests + human approval |
| 14 | Launch | 13 | |
| 15 | Post-Launch Operations | 29 | 20 are recurring type |

---

## What Works

1. **Workflow JSON structure is well-designed.** 328 steps with clear phases, dependency chains, input/output artifacts, agent assignments, done_when criteria, and conditional groups. The v2 format is comprehensive.

2. **Loader + DAG validation.** `load_workflow()` parses correctly; `validate_dependencies()` checks for unknown refs, cycles, and orphan steps. Critical errors block workflow start.

3. **Runner creates missions and tasks.** `WorkflowRunner.start()` correctly creates a mission, stores initial artifacts, filters steps, expands to tasks, resolves step-ID dependencies to DB task IDs, and inserts all tasks.

4. **Dependency resolution in get_ready_tasks().** The function fetches ALL pending tasks, then checks each one's `depends_on` against completed/skipped statuses. Skipped deps are handled (auto-skip if all deps skipped, ready if at least one completed). Failed deps are logged but not auto-cleared here (watchdog does that).

5. **Watchdog recovery.** Resets stuck "processing" tasks, clears deps on tasks blocked by failed dependencies, completes parent tasks when all subtasks done, escalates stale clarifications (4h nudge, 24h reminder, 48h urgent, 72h cancel), checks workflow timeout.

6. **Telegram UX for workflows.** `/mission --workflow`, `/wfstatus <id>` with progress bars, `/resume <id>`, workflow cancel button, auto-detection of product ideas via classifier.

7. **Checkpoint/resume infrastructure exists.** `workflow_checkpoints` table, `WorkflowRunner.resume()` that resets failed tasks and inserts missing steps. Phase completion detection in post-hook.

8. **Quality gate framework.** Gates defined for phases 9, 10, 13. Parsers for test pass rate, coverage, security scan, human approval. Gate results stored as artifacts.

9. **Conditional group evaluation.** DSL supports `length(field) >= N`, `any(item.field == 'value')`, `field != 'value'`, boolean comparisons, `platforms_include()`, and OR expressions.

10. **Template expansion.** `_trigger_template_expansion()` in hooks responds to `implementation_backlog` artifact to expand `feature_implementation_template` per feature in Phase 8.

11. **Context strategy / tiered budgets.** Artifacts can be tagged as primary (8K), reference (3K), or full_only_if_needed (1.5K) to manage prompt size.

---

## Critical Issues (why workflows stall)

### CRITICAL-1: mission_id never reaches workflow hooks (artifact system is dead)

**Severity: Showstopper -- breaks the entire artifact pipeline**

In `src/workflows/engine/hooks.py`, both `pre_execute_workflow_step()` (line 112) and `post_execute_workflow_step()` (line 165) read `mission_id` from the task's **context JSON**:

```python
mission_id = ctx.get("mission_id")   # hooks.py:112, 165
```

But `src/workflows/engine/expander.py` never puts `mission_id` inside the context dict. It sets `mission_id` as a **top-level task field** (line 120: `"mission_id": mission_id`), which becomes a DB column, not part of the JSON `context` column.

The orchestrator's `_inject_chain_context()` also does NOT inject `mission_id` into the context JSON -- it reads it from `task.get("mission_id")` (the DB column) but never writes it back into `task_context`.

**Consequence:** `mission_id` is always `None` in the hooks. This means:
- Artifacts are **never retrieved** for input injection (pre-hook)
- Artifacts are **never stored** from output (post-hook)
- Phase completion is **never detected**
- Quality gates **never fire**
- Conditional groups are **never evaluated**
- Template expansion is **never triggered**
- Phase summaries are **never generated**

Every workflow step runs blind -- no context from prior steps, no artifact chain.

**Fix:** In `hooks.py`, fall back to `task.get("mission_id")`:
```python
mission_id = ctx.get("mission_id") or task.get("mission_id")
```
Or better: have the expander include `mission_id` in the context dict.

### CRITICAL-2: step_id vs workflow_step_id naming mismatch

**Severity: High -- breaks post-hook artifact storage and conditional exclusion**

In `hooks.py:167`, the post-hook reads:
```python
step_id = ctx.get("step_id", "")
```

But the expander sets `"workflow_step_id"` (expander.py:96), not `"step_id"`. So `step_id` is always `""`.

This breaks:
- Review cycle tracking (always records empty step_id)
- Conditional group exclusion (hooks.py:399 queries `field="step_id"`)

**Fix:** Change to `ctx.get("workflow_step_id", "")`.

### CRITICAL-3: 298+ blocked tasks -- dependency chain failure cascade

**Root cause:** When Phase 0 tasks complete, their artifacts are never stored (due to CRITICAL-1), so Phase 1 tasks that depend on Phase 0 outputs run without input context. If a Phase 1 task fails or needs clarification, every downstream task in Phases 2-15 remains blocked.

The `depends_on` field stores DB task IDs (integers). A task with `depends_on: [5, 6, 7]` only becomes ready when tasks #5, #6, #7 are all completed or skipped. If task #5 gets stuck in `needs_clarification` or `failed`, the entire chain stalls.

The watchdog does clear `depends_on` for tasks with **failed** deps (orchestrator.py:281-311), but:
- It only clears deps for tasks where a dep has status `'failed'` -- not `needs_clarification`, not `cancelled`, not `paused`
- Clearing ALL deps means the task runs without its prerequisites, likely producing garbage
- There's no mechanism to retry the failed dependency itself

**Scenario for 298 blocked tasks:** A mission with 300+ tasks where one early task (e.g., step 1.3 "direct competitor list") fails or gets stuck in `needs_clarification`. All 298 downstream tasks remain in `pending` with unresolved deps. The watchdog's failed-dep clearing is too aggressive (removes ALL deps, not just the failed one) and too narrow (only handles `failed`, not other stuck states).

### CRITICAL-4: Concurrent task processing breaks on shared batch

When multiple tasks run concurrently (`asyncio.wait` with `FIRST_COMPLETED`), if one task completes quickly, the main loop proceeds to the next cycle while other tasks from the same batch are still running. The `return_when=asyncio.FIRST_COMPLETED` at line 2118 means:
- Only the first completed task gets its result checked
- Remaining futures are not awaited or collected
- Their exceptions are logged but execution continues
- No mechanism to track that these tasks are still running

This can lead to the same task being picked up again by `get_ready_tasks()` in the next cycle (since `claim_task` should prevent duplicate processing, this is mitigated but wasteful).

---

## Task Dependency Problems

### DEP-1: Linear dependency chain = single point of failure

The v2 workflow has 323 out of 328 steps with `depends_on`. Only 5 root steps exist (-1.1, 0.1, 8.sprint_ritual, 8.arch_check, 9.1). This means one stuck task can block the entire remaining pipeline.

Phase 9 step 9.1 is a root (no deps) which is unusual -- it can start immediately even if phases 0-8 haven't completed. This may be intentional (test infrastructure setup) but could also be a bug.

### DEP-2: Watchdog dep-clearing is too blunt

When the watchdog finds a task blocked by failed deps (orchestrator.py:308-311), it clears ALL dependencies:
```python
await db.execute(
    "UPDATE tasks SET depends_on = '[]' WHERE id = ?",
    (task["id"],),
)
```

This means the task runs without any prerequisite outputs. For a workflow where steps build on each other's artifacts, this produces meaningless results.

**Better approach:** Only remove the specific failed dep, or retry the failed dep, or mark the blocked task as `failed` with a clear reason.

### DEP-3: No dep status for needs_clarification or cancelled

`get_ready_tasks()` only counts `completed` and `skipped` as "resolved". A dependency stuck in `needs_clarification` or `cancelled` is neither resolved nor failed -- it's in limbo. The task stays blocked forever unless the watchdog intervenes.

The watchdog only handles `failed` deps, not `cancelled` or `needs_clarification`. A cancelled dependency should probably auto-skip or fail the dependent task.

### DEP-4: get_ready_tasks fetches ALL pending tasks every cycle

With 300+ pending tasks, every orchestrator cycle queries all of them and checks each one's dependencies. This is O(n * d) where n = pending tasks and d = average deps per task. For mission #5 with 298 blocked tasks, this means 298 * ~3 = ~900 DB queries per cycle (every 15 seconds).

---

## Missing Features

### MISSING-1: No artifact flow between steps (consequence of CRITICAL-1)

The entire artifact system (input_artifacts, output_artifacts, context_strategy, phase summaries) is architecturally complete but functionally dead because mission_id never reaches the hooks. Every step runs with only its static instruction text -- no context from prior steps' outputs.

### MISSING-2: No mechanism to unblock a stuck workflow

When a workflow stalls (which is inevitable given CRITICAL-1/3), there's no user-facing command to:
- Skip a specific step
- Retry a failed step
- Force-unblock a specific task
- Jump to a specific phase
- Provide manual artifact values

The `/resume` command only resets failed/needs_clarification tasks to pending and inserts missing steps. It doesn't address structural blocking.

### MISSING-3: Conditional groups never fire

Conditional groups depend on post-hook evaluation (hooks.py:360-412), which depends on artifacts being stored, which depends on mission_id being available. Since CRITICAL-1 prevents this, conditional groups never resolve. The 7 conditional groups in v2 (e.g., competitor_deep_dive, mobile_platform_check) are dead code.

### MISSING-4: Template expansion never triggers

The feature_implementation_template (31 steps per feature) should be expanded when the `implementation_backlog` artifact is produced by step 6.7. But since artifacts are never stored, the trigger never fires. Phase 8 only has 3 static steps (sprint_ritual, arch_check, and one more) -- the bulk of implementation depends on template expansion.

### MISSING-5: Quality gates never evaluate

Quality gates for phases 9, 10, 13 require artifacts (test_results, coverage_report, security_scan_results, human_approval). Since artifacts are never stored, gates always fail with "artifact not found". But gates only fire on phase completion, and phase completion is never detected (CRITICAL-1).

### MISSING-6: No progress persistence across restarts

While `workflow_checkpoints` table exists and `_check_phase_completion()` calls `upsert_workflow_checkpoint()`, this never fires (CRITICAL-1). After a restart, `WorkflowRunner.resume()` tries to use the checkpoint but finds nothing, so it can only reset failed tasks.

### MISSING-7: Phase summaries never generated

`_generate_phase_summary()` in hooks.py is well-implemented but never called because phase completion is never detected. Later phases never receive context about what earlier phases produced.

### MISSING-8: No cost tracking or budget limits for workflows

The preview method estimates cost ($0.001-$0.05/step) but there's no runtime cost tracking or budget enforcement. A 328-step workflow with template expansion could easily run 400+ LLM calls.

### MISSING-9: Human-in-the-loop feedback during execution is limited

While `needs_clarification` exists and has escalation tiers, there's no mechanism for:
- Proactive user review at phase boundaries
- User approval before starting expensive phases (4, 7, 8)
- Showing intermediate results for user feedback
- User-triggered course corrections mid-workflow

### MISSING-10: No partial workflow execution

Cannot start from a specific phase, skip phases, or run a subset. It's all-or-nothing from Phase 0 to Phase 15.

---

## Recommendations

### Priority 1: Fix the showstoppers (get artifact flow working)

**Fix CRITICAL-1** -- One-line fix in hooks.py (both pre and post functions):
```python
mission_id = ctx.get("mission_id") or task.get("mission_id")
```

Also fix in `_check_phase_completion` and `_check_conditional_triggers` which both derive mission_id from ctx.

**Fix CRITICAL-2** -- Change `ctx.get("step_id", "")` to `ctx.get("workflow_step_id", "")` in hooks.py:167. Also fix conditional group exclusion at hooks.py:399.

These two fixes alone would unblock the entire artifact pipeline: input injection, output storage, phase completion, quality gates, conditional groups, template expansion, and phase summaries.

### Priority 2: Fix dependency cascade failures

1. **Add `cancelled` as a resolved status** in `get_ready_tasks()` -- treat it like `skipped`.
2. **Handle `needs_clarification` deps** -- if a dep has been in `needs_clarification` for >72h (matching the escalation timer), auto-cancel it and treat as resolved.
3. **Smarter watchdog dep clearing** -- instead of clearing ALL deps, only remove the failed dep ID. Or better: mark the blocked task as `failed` with `error="dependency_failed"`.
4. **Add per-task skip command** -- `/wfskip <task_id>` to manually resolve a blocking task.

### Priority 3: Reduce task volume

328 steps is excessive for an AI agent to execute reliably. Consider:
1. **Phase gating with human confirmation** -- don't create all 328 tasks upfront. Create Phase 0 tasks, wait for completion + user approval, then create Phase 1 tasks, etc.
2. **Lazy task creation** -- only create tasks for the next 1-2 phases. This dramatically reduces the blocked-task problem.
3. **Phase bundling** -- combine closely related steps within a phase into a single task with multi-part instructions.

### Priority 4: User control commands

Add Telegram commands:
- `/wfskip <task_id>` -- skip a blocking task
- `/wfretry <task_id>` -- retry a failed task
- `/wfjump <mission_id> <phase>` -- skip ahead to a phase (mark all prior tasks as skipped)
- `/wfpause <mission_id>` -- pause all pending tasks in a mission
- `/wfartifact <mission_id> <name> <value>` -- manually set an artifact

### Priority 5: Incremental workflow execution

Instead of creating 300+ tasks at start:
1. Create only Phase 0 tasks
2. On Phase 0 completion, evaluate conditional groups, then create Phase 1 tasks
3. At Phase 6 completion, expand templates and create Phase 7-8 tasks
4. Continue phase-by-phase

This eliminates the 298-blocked-tasks problem entirely, reduces DB load, and allows user intervention between phases.

### Priority 6: Cost and progress visibility

1. Track per-task cost (model cost) and accumulate per-mission
2. Set budget limits per mission with alerts at 50%, 75%, 100%
3. Show estimated vs actual cost in `/wfstatus`
4. Send phase completion notifications via Telegram with summary

### Summary of Fix Effort

| Fix | Effort | Impact |
|---|---|---|
| CRITICAL-1 (mission_id in hooks) | 5 min | Unblocks entire artifact system |
| CRITICAL-2 (step_id naming) | 2 min | Fixes artifact storage + conditional groups |
| DEP-2 (smarter watchdog) | 30 min | Prevents garbage execution after dep failure |
| DEP-3 (cancelled dep handling) | 15 min | Unblocks tasks after cancelled deps |
| MISSING-2 (skip/retry commands) | 2 hours | User can unblock stuck workflows |
| Priority 5 (incremental execution) | 1 day | Eliminates mass-blocking, enables phase gating |
