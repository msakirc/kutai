# Phase 2b — General Beckman (Task Master Package)

**Date:** 2026-04-18
**Status:** Design approved, ready for plan
**Package:** `packages/general_beckman/`
**Follows:** Plan A (in-tree untangle, merged), Plan B (salako package, merged)

Named after General Beckman from *Chuck* — the NSA commander who hands out missions and approves action. This package owns the task queue and answers the question: **"what should we do next, and how many of it?"**

## 1. Why This Exists

After Plan A + B, `src/core/orchestrator.py` is still ~2,569 lines. The remaining mass is task-queue logic: `_handle_*` methods, gates, scheduled jobs, watchdog, result-routing orchestration. These are not orchestration concerns — they are *decisions about the task queue*. This package extracts them.

Target: `orchestrator.py` drops to ~200 lines of pure glue (LLM dispatch + mechanical dispatch + lifecycle).

## 2. What Beckman Owns

- **Task queue**: eligible-task selection, priority, dependency gates, DLQ, scheduled tasks.
- **Parallelism decision**: given capacity, *how many* tasks to release concurrently per lane (local LLM, cloud LLM, mechanical).
- **Queue look-ahead**: project upcoming demand against quota/budget, hold back when appropriate. (Lost during an earlier `quota_planner` extraction — reinstated here.)
- **Lifecycle transitions**: the current `_handle_complete`, `_handle_subtasks`, `_handle_clarification`, `_handle_review`, `_handle_exhausted`, `_handle_failed`, `_handle_unexpected_failure`, `_handle_availability_failure` logic moves in. Agent results still flow through `result_router` / `result_guards`; those modules also move into beckman.
- **Watchdog**: `check_stuck_tasks` moves in — it's a queue maintenance concern.
- **Scheduled jobs**: `src/app/scheduled_jobs.py` moves in — it's a queue producer.

## 3. What Beckman Does *Not* Own

- **Model selection** — stays in `fatih_hoca`. Runs *after* beckman per dispatched task.
- **LLM call execution** — stays in `hallederiz_kadir` via `llm_dispatcher`.
- **Mechanical execution** — stays in `salako`.
- **System state aggregation** — stays in `nerd_herd`. Beckman consumes snapshots.
- **Cloud rate tracking** — stays in `kdv` (registers to `nerd_herd`).
- **Telegram I/O** — stays in `telegram_bot.py` for inbound parsing. Outbound messages route through salako's mechanical executors (new; see §6).

## 4. Core Interface

```python
# packages/general_beckman/src/general_beckman/__init__.py

async def next_task() -> Task | None:
    """Return one task ready to dispatch, or None if nothing should be released right now.

    Consults nerd_herd.snapshot() for capacity. Applies parallelism policy,
    look-ahead, priority, and eligibility filters. Caller loops to saturation.
    """

async def on_task_finished(task_id: int, result: AgentResult) -> None:
    """Run lifecycle handling for a completed task (Complete / SpawnSubtasks /
    RequestClarification / Exhausted / Failed / etc.). Drains result_router
    internally. Emits new tasks into the queue via DB writes."""

async def tick() -> None:
    """Periodic maintenance: watchdog check, scheduled-job dispatch, queue
    reconciliation. Called by orchestrator every 3s."""
```

**That's the full surface.** Three async functions. No Decision types, no event bus, no callbacks.

## 5. Orchestrator's New Shape

```python
# sketch — final will be ~200 lines
async def main_loop():
    while running:
        # Drain beckman to saturation
        while (task := await beckman.next_task()):
            asyncio.create_task(_dispatch(task))
        # Tick (watchdog + scheduled + reconciliation)
        await beckman.tick()
        await asyncio.sleep(3)

async def _dispatch(task):
    if task["agent_type"] == "mechanical":
        result = await salako.run(task)
    else:
        result = await llm_dispatcher.request(task)
    await beckman.on_task_finished(task["id"], result)
    # Completion triggers immediate re-drain on next loop iteration
```

No `_handle_*` methods. No `run_gates`. No `scheduled_jobs` loop. No watchdog thread. All inside beckman.

Note: `asyncio.create_task` is the *mechanism* for non-blocking dispatch. Actual parallelism is gated by beckman returning `None` once its policy cap is reached — see §14 for the initial cap-at-1-per-lane policy.

## 6. Salako Gains Two Mechanical Executors

Existing: `workspace_snapshot`, `git_commit`.

New:
- **`clarify`**: formats a clarification prompt from task payload, sends via `self.telegram.send_message`, stores pending state so user reply routes back as `user_clarification` in task_context. Replaces orchestrator's current `_handle_clarification` Telegram send.
- **`notify_user`**: sends a plain status message via Telegram. Used for meaningful notifications (mission complete, DLQ alerts, rejections). Ephemeral progress chatter does **not** go through this — it calls `telegram.send_message` directly from wherever it originates.

Beckman emits these as regular tasks (just with `agent_type="mechanical"` and `executor="clarify"` / `"notify_user"`). Orchestrator routes all mechanical tasks to salako identically.

## 7. Capacity Contract

**Source of truth**: `nerd_herd.snapshot()` returns `SystemSnapshot` containing:
- GPU/VRAM state (local capacity signal)
- `local: LocalModelState` (pushed by dallama on swap)
- `cloud: dict[provider, CloudProviderState]` (pushed by kdv on each API response)

**Beckman reads this on every `next_task()` call.** The 2s cache inside `GPUCollector` keeps the cost negligible even at high call frequency.

**Beckman does not receive events.** No subscription API on nerd herd. Pull-only.

**Completion triggers** drive immediate re-draining: when `llm_dispatcher.request()` or `salako.run()` returns, the main loop continues to the next iteration immediately, which pulls from beckman until `None`. No explicit "slot freed" signal needed — capacity changes are reflected in the next snapshot.

**Tick cadence: 3s**, matching slightly above the 2s nerd-herd cache. Trade-off: faster ticks (1–2s) give more reactive scheduled-job dispatch; slower ticks are fine because completion triggers handle the hot path.

## 8. Deletions

Plan C deletes dead code:

- **`src/security/risk_assessor.py`** — runtime risk gate that never triggered in practice. Unrelated to i2p workflows' "risk_assessment" step (that's a planner artifact, different thing).
- **`approval_fn` plumbing** — `human_gate` path in `run_gates`, Telegram `request_approval` method, and the `human_gate` context field (confirm via grep; may be referenced in workflow JSON).
- **`src/core/task_gates.py`** — entirely. Its only remaining logic was the approval + risk gates. If anything survives, it moves into beckman as a private helper.
- **`tests/test_human_gates.py`** and **`tests/test_resilience_approvals.py`** — cover dead paths.

## 9. Package Layout

```
packages/general_beckman/
├── pyproject.toml
├── README.md
├── src/general_beckman/
│   ├── __init__.py              # Public API: next_task, on_task_finished, tick
│   ├── queue.py                 # Task selection + eligibility filters
│   ├── parallelism.py           # "How many to release" logic (consumes snapshot)
│   ├── lookahead.py             # Queue look-ahead against quota/budget (reinstated)
│   ├── lifecycle.py             # on_task_finished + handlers (ex-_handle_*)
│   ├── result_router.py         # Moved from src/core/
│   ├── result_guards.py         # Moved from src/core/
│   ├── task_context.py          # Moved from src/core/
│   ├── watchdog.py              # Moved from src/core/, invoked from tick()
│   ├── scheduled_jobs.py        # Moved from src/app/, invoked from tick()
│   └── types.py                 # Task, AgentResult, lane enums
└── tests/
    ├── test_queue.py
    ├── test_parallelism.py
    ├── test_lifecycle.py
    ├── test_lookahead.py
    └── ... (moved test files for watchdog/scheduled_jobs/result_router/result_guards)
```

## 10. Backward Compatibility Shims

Following the `src/core/router.py → fatih_hoca` pattern, preserve shims so existing tests and imports keep working:

- `src/core/task_context.py` → re-exports from `general_beckman.task_context`
- `src/core/result_router.py` → re-exports
- `src/core/result_guards.py` → re-exports
- `src/core/watchdog.py` → re-exports
- `src/app/scheduled_jobs.py` → re-exports
- `src/core/task_gates.py` — **removed** (module deleted, not shimmed, since content is dead)

Existing test suites must pass unchanged after the move.

## 11. Split of Responsibilities With Fatih Hoca

Same snapshot contract, different questions:

| Question | Answered by | When |
|---|---|---|
| "Should we release more tasks right now, and which ones?" | **beckman** | Every `next_task()` call |
| "For this released task, which model fits best?" | **fatih_hoca** | Inside `llm_dispatcher.request()`, after dispatch |
| "How close are we to cloud quota exhaustion?" | **beckman look-ahead** (reads quota from snapshot) | Before releasing cloud-heavy tasks |

Both read `nerd_herd.snapshot()`. Neither depends on the other.

## 12. Open Follow-Ups (Out of Scope)

Flag but don't fix in Plan C:

- **kdv state persistence**: cloud rate-limit state is in-memory only; lost on restart. Risk of quota overrun while relearning. File as separate fix.
- **Telegram module extraction**: outbound goes through salako executors in Plan C, but inbound reply routing stays in `telegram_bot.py`. Full split is a future project.
- **Progress chatter standardization**: ephemeral notifications (scraping progress, iteration counters) still call `self.telegram.send_message` directly from many sites. No unified API. Worth tidying later, not now.

## 13. Success Criteria

- `orchestrator.py` ≤ ~250 lines (target 200, hard cap 300).
- `packages/general_beckman/` has its own test suite; all tests pass in isolation.
- Full existing test suite passes with the new shims (baseline: 253 pre-existing failures per session notes; no *new* failures touching orchestrator / beckman / result_* / watchdog / scheduled_jobs).
- Manual smoke: start KutAI, create a task via Telegram, confirm dispatch happens, confirm clarification round-trip still works, confirm scheduled jobs fire.
- `src/security/risk_assessor.py` and `src/core/task_gates.py` no longer exist.
- No `approval_fn` parameter anywhere in the codebase.

## 14. Key Risks

- **Hidden couplings in orchestrator**: `process_task` has been reshaped twice now; a third pass may surface edge cases. Mitigation: each extraction task commits separately with verification, per Plan A/B pattern.
- **Parallelism regression**: today's effective concurrency is 1 (serial `process_task`). Enabling true concurrent `asyncio.create_task` dispatch could surface races in shared DB writes, workspace operations, or LLM server state. Mitigation: keep initial parallelism cap at 1 per lane behind a config flag; unlock in a follow-up after soak-testing.
- **Watchdog + scheduled-jobs timing**: those currently run on their own loops. Collapsing them into a shared `tick()` may shift timing enough to expose bugs. Mitigation: preserve their internal cadence (they track their own last-run timestamps) — `tick()` is just the invocation point.
- **Look-ahead reinstatement**: the original quota_planner look-ahead was lost; reimplementing without tests is risk. Mitigation: write tests first for the new `lookahead.py` against synthetic snapshots.
