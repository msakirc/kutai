# Orchestrator Refactor — Phase 2 Handoff

**Date:** 2026-04-17
**Branch:** `refactor/orchestrator-phase1` (11 commits ahead of main, not merged)
**Phase 1 plan:** `docs/superpowers/plans/2026-04-17-orchestrator-phase1-in-tree.md`

---

## Where we are

Phase 1 landed the in-tree seams. Orchestrator is untangled but not yet decomposed into packages. Two items were intentionally scoped down; they now block further `process_task` shrinkage and drive Phase 2 design.

### What's in place on `refactor/orchestrator-phase1`

| Module | Purpose | Status |
|---|---|---|
| `src/core/decisions.py` | `Allow/Block/Cancel/GateDecision`, `Dispatch/NotifyUser` | live |
| `src/core/task_context.py` | `parse_context` / `set_context` | live, 4 call sites |
| `src/core/task_gates.py` | async `run_gates() → GateDecision`, injected `approval_fn` | live |
| `src/core/result_router.py` | pure `route_result(task, agent_result) → list[Action]` | **module live, NOT wired into `process_task`** |
| `src/core/watchdog.py` | `check_stuck_tasks` + `check_resources` | live; orchestrator delegates |
| `src/core/mechanical/workspace_snapshot.py` | pre-task workspace snapshot | live |
| `src/core/mechanical/git_commit.py` | ported from `_auto_commit` | **dormant** (call site disconnected) |
| `src/app/scheduled_jobs.py` | `ScheduledJobs` (todos, API discovery, digest, price watches) | live |
| `src/core/orchestrator.py:_prepare` | claim → classify → enrich → gate → snapshot | live |

### Line counts

- `orchestrator.py`: 3,865 → 2,800 (−1,065 / −28%)
- `process_task`: 1,143 → 945 (−198 via `_prepare`)
- `watchdog`: 519 → ~10-line delegator
- Tests: all 32 new-module tests green; no new regressions in the existing suite

### Deferred (the two hard parts)

**D1 — `result_router` wire-up.** The pure function and its 8 tests landed, but `process_task` still contains the old `if status == "completed" / "needs_subtasks" / "needs_clarification" / "needs_review" / "exhausted" / "failed"` branch chain. Three blockers found during Task 5 attempt:
1. Each branch has **30–80 lines of guard code** before delegating (workflow post-hooks, quality retries, bonus attempts, schema validation, DLQ cascade). `route_result → _handle_*` can't replace this without losing behavior.
2. **Handler signature mismatch.** Existing `_handle_complete(task, result: dict)` takes the raw dict; plan called `_handle_complete(task, action.result, action.iterations, action.metadata)`.
3. **Missing handlers.** `exhausted` (90 lines inline) and `failed` (55 lines inline) have no `_handle_*` method at all.

**D2 — `process_task` split.** Plan targeted ~25 lines; actual is 945 because `_dispatch` and `_record` extractions depend on D1. Only `_prepare` was extracted.

---

## What's next — three plans to write

Write each as a `docs/superpowers/plans/...md` file following the plan template used for Phase 1.

### Plan A — Finish `process_task` in-tree (prerequisite for Phase 2b)

**Goal:** Unblock D1 + D2 so `process_task` can shrink toward the original ~25-line target.

Scope:
1. Extract `_handle_exhausted` and `_handle_failed` from inline code (verbatim move; mirrors existing `_handle_complete` / `_handle_subtasks` / `_handle_clarification` / `_handle_review` style).
2. Normalize handler signatures. Two paths to pick between:
   - **Option 1 (small diff):** add `route_result` Action types that carry the full `result` dict so `_handle_complete(task, action.raw)` stays close to current shape.
   - **Option 2 (more invasive):** break each handler into `prepare_handler_args → dispatch → post_hook` so the guard code lives *between* the router and the handler.
3. Port the per-branch guard code (workflow post-hooks, quality retries, bonus attempts, schema validation, DLQ cascade) into typed helper functions (`src/core/result_guards.py`?) that the `_record` stage calls BEFORE dispatching to handlers.
4. Wire `route_result` into `process_task`. Extract `_dispatch` (agent/pipeline invocation + timeout). Extract `_record` (call `route_result`, run guards, call handlers).

Acceptance: `process_task` < 50 lines, all existing behavioral tests pass, no new regressions.

### Plan B — Phase 2a: Mechanical Dispatcher package

**Goal:** Promote `src/core/mechanical/` to `packages/mechanical_dispatcher/` and wire it as a sibling executor to LLM Dispatcher.

Scope:
1. Standard package layout (`pyproject.toml`, src layout, editable install).
2. Public API: `run(task) → Action` where `task.executor == "mechanical"` routes here instead of the LLM path.
3. Re-enable `git_commit.auto_commit` as an explicit workflow step (i2p recipe change). Currently dormant.
4. Add workspace_snapshot as a mechanical executor action (currently called directly from `_prepare`).
5. Orchestrator routing: `if task.get("executor") == "mechanical": await mechanical.run(task)` before LLM path.

Preconditions: Plan A can ship first or in parallel — this package doesn't depend on `_record` shape.

### Plan C — Phase 2b: Task Master (`packages/gorev_ustasi/`)

**Goal:** The brain. Own the task queue, scheduling, gates, context, retry, missions, workflow recipes. Orchestrator becomes a pure router (~200 lines end-state).

Scope:
1. Move `task_context`, `task_gates`, `result_router`, `watchdog.check_stuck_tasks`, `scheduled_jobs` into the package.
2. Replace `approval_fn` injection with `RequestApproval` decision emission (fully invert Telegram coupling).
3. Task master emits `Dispatch(task, executor, payload) | NotifyUser(...)`; orchestrator routes `Dispatch` by executor.
4. `_handle_*` methods become Decision emissions — handlers move into task master; orchestrator just executes the emitted decision.

Preconditions: Plan A must ship first (needs clean `_record` boundary).

---

## Recommended sequence

1. **Telegram smoke test** the current branch (Step 3 of Task 10 — `/task`, `/shop`, workflow, clarification, human-gate). ← you're here
2. Merge `refactor/orchestrator-phase1` to main after smoke passes.
3. Write Plan A, execute. Land in-tree, still no packages.
4. Write Plan B + C in parallel (Plan B doesn't depend on Plan A).
5. Execute B and C independently — each is its own branch / plan.
