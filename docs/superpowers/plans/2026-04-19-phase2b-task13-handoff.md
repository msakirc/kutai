# Phase 2b — Task 13 Handoff (main-loop rewrite + `_handle_*` consolidation)

**Date created:** 2026-04-19
**Parent session merge commit:** `03a1def` (Merge branch 'feat/general-beckman')
**Status:** Handoff for fresh session — 13 of 14 tasks of Phase 2b shipped; Task 13 deferred for careful runtime verification.

---

## Why this exists

Phase 2b (general_beckman task master package) landed 13 of 14 tasks on main. The headline goal — **orchestrator.py ≤ 300 lines** — was NOT met. Orchestrator is still **2566 lines** (was 2569; change is within noise).

The executing agent deliberately stopped short of Task 13 (the main-loop rewrite + `_handle_*` consolidation). Its reasoning was sound:

> `Orchestrator.run_loop` is ~370 lines of tightly coupled scheduling logic (age-based priority boost, swap-aware deferral, model affinity reordering, paused-pattern filtering, quota-planner forward scan, task dedup, model manager hooks). Beckman's `next_task()` as implemented covers eligibility + lane saturation + quota look-ahead, but not these orchestrator-only scheduling concerns. Replacing `run_loop` wholesale without runtime verification risks silent regressions (starvation, swap thrashing, quota overruns).
>
> The `_handle_*` methods have hundreds of lines each (complete: 193, subtasks: 202, failed: 241, exhausted: 95, etc.), heavily self-coupled to `self.telegram`, `self.llm_dispatcher`, `self._resolve_model_for_task`. The plan's "copy verbatim, drop self" instruction doesn't account for those couplings.

The fresh session's job is to plan and execute Task 13 **with proper behavior-preservation guarantees**, not by pattern-matching a copy-paste.

---

## What's already in place (don't redo)

- **Package scaffold**: `packages/general_beckman/` with `__init__.py` exposing `next_task`, `on_task_finished`, `tick`, `set_orchestrator`. Fully tested (25 tests green).
- **Public API shape**:
  - `next_task() -> Task | None` — eligibility + lane classification + quota look-ahead. Caller loops to saturation.
  - `on_task_finished(task_id, result)` — drains `result_router` → invokes `lifecycle.*` handlers. Currently delegates back to `get_orchestrator()._handle_*` via a registered orchestrator reference (transitional).
  - `tick()` — runs watchdog + scheduled_jobs on preserved internal cadences. Orchestrator calls every 3s.
- **Modules moved with shims** (re-exports preserve backward compat): `task_context`, `result_router`, `result_guards`, `watchdog`, `scheduled_jobs`.
- **Dead code deleted**: `src/security/risk_assessor.py`, `src/core/task_gates.py`, `tests/test_human_gates.py`, `tests/test_resilience_approvals.py`, `tests/test_risk_assessor_async.py`, `tests/test_task_gates.py`. `cmd_autonomy` also removed from `telegram_bot.py`.
- **Salako executors**: `clarify` and `notify_user` added (`packages/salako/src/salako/clarify.py`, `notify_user.py`). Telegram singleton (`get_telegram`/`set_telegram`) introduced in `telegram_bot.py`.
- **Lifecycle handler for clarification** already routes through salako (emits a clarify task), bypassing the orchestrator. The other `_handle_*` handlers still live on the orchestrator class and are called via `get_orchestrator()._handle_*` from `lifecycle.py` — this is the transitional circular delegation to unwind.
- **Tests**: beckman 25 green, salako 16 green. Full suite baseline went from **253 → 248** failures (dead-test cleanup). Any new failures in orchestrator/beckman/result_*/watchdog/scheduled_jobs/lifecycle must be investigated.

---

## What Task 13 needs to do

Two connected pieces of work:

### 13a. Consolidate `_handle_*` methods into `packages/general_beckman/src/general_beckman/lifecycle.py`

The 8 handlers currently on `Orchestrator`:

| Method | orchestrator.py line | Approx size | Coupling |
|---|---|---|---|
| `_handle_availability_failure` | ~1003 | 55 lines | `self.telegram`, `self.llm_dispatcher` |
| `_handle_unexpected_failure` | ~1058 | 92 lines | `self.telegram`, DB |
| `_handle_complete` | ~1150 | 193 lines | `self.telegram`, `self.llm_dispatcher`, workflow engine |
| `_handle_subtasks` | ~1344 | 202 lines | `self.llm_dispatcher`, `self._resolve_model_for_task`, DB |
| `_handle_clarification` | ~1547 | 7 lines | Already migrated to salako clarify |
| `_handle_review` | ~1554 | 22 lines | DB |
| `_handle_exhausted` | ~1576 | 95 lines | `self.telegram`, DB |
| `_handle_failed` | ~1672 | 241 lines | `self.telegram`, `self.llm_dispatcher`, DB |

Line numbers are from pre-Phase-2b counts — confirm via `grep -n "async def _handle_" src/core/orchestrator.py` before quoting them in your plan.

**The real work is not "move the code" — it's "move the code without breaking behavior."** Handlers reach into `self.llm_dispatcher`, `self.telegram`, `self._resolve_model_for_task`, and DB helpers. They can't be pure functions without addressing those dependencies.

### 13b. Rewrite `Orchestrator.run_loop` to use `beckman.next_task()` / `beckman.tick()`

Currently `run_loop` does ~370 lines of scheduling logic not yet in beckman:

- Age-based priority boost (starvation avoidance)
- Swap-aware deferral (hold tasks requiring a swap when budget is low)
- Model affinity reordering (prefer loaded model when possible)
- Paused-pattern filtering
- Quota-planner forward scan (already moved — verify)
- Task dedup
- Model manager hooks

Some of this is beckman's job (quota, swap budget). Some is orchestrator's (model affinity is specific to llm_dispatcher's pre-call phase, arguably). Decide during brainstorming.

---

## Start by brainstorming: how do we preserve behavior?

This is the first-class question. A new session should **not** jump to code. Before writing any plan, use `superpowers:brainstorming` on:

> **"How do we migrate `_handle_*` and `run_loop` into beckman without regressing scheduling behavior, given that we have no test harness covering the scheduling logic and no safe way to run production workloads in a test?"**

Candidate approaches to put on the table (not exhaustive):

- **Behavioral test harness first**: write integration tests that exercise today's `run_loop` + `_handle_*` via synthetic tasks before migrating. Then verify the tests still pass after. Cost: significant test-writing up front.
- **One handler at a time**: migrate `_handle_review` (smallest, DB-only) first with its own test. Prove the pattern. Then `_handle_availability_failure`, ... working up to `_handle_failed` last. Each lands as its own commit. Rollback per-handler if something breaks in runtime.
- **Strangler pattern**: keep the orchestrator's `_handle_*` methods alive; have `lifecycle.on_task_finished` call them via the registered ref (already the case). Remove them only once behavior parity is confirmed with manual runtime verification across multiple mission types.
- **Rewrite `run_loop` separately from handlers**: these are two independent migrations; don't couple them. `run_loop`'s scheduling concerns touch different code paths than lifecycle handling.
- **Runtime parity logging**: add temporary dual-run logging (run both old and new code paths, compare outputs) for N days before cutting over. Heavy but provides confidence.

Your brainstorm should also surface:

- **What scheduling behavior is load-bearing?** `/task`, `/shop`, mission phases, DLQ retries, human interrupts, scheduled missions. The fresh session needs to enumerate these and decide what each one relies on from `run_loop`.
- **What's the rollback plan if Task 13 regresses something in production?** git revert works, but only if caught quickly.
- **Should any of `run_loop`'s scheduling logic stay in orchestrator?** The agent hinted model affinity reordering might legitimately belong with llm_dispatcher's pre-call phase, not beckman. Worth re-litigating.

Don't just adopt one of the above. Brainstorm, pick one, defend it in the spec.

---

## Files to read first (in order)

1. **This handoff** (you're reading it).
2. `docs/superpowers/specs/2026-04-18-phase2b-general-beckman-design.md` — original design. Task 13 goal is in §5 and §13.
3. `docs/superpowers/plans/2026-04-19-phase2b-general-beckman.md` — the executed plan. Task 13 is the one that didn't land; read it to understand what was expected.
4. `docs/architecture-modularization.md` — the agent added a "Phase 2b" section documenting the transitional state. Read for current ground truth.
5. `src/core/orchestrator.py` — the target. Read `run_loop` and each `_handle_*` carefully. Map couplings.
6. `packages/general_beckman/src/general_beckman/lifecycle.py` — see how `on_task_finished` currently delegates. This is the circular indirection to unwind.
7. `packages/general_beckman/src/general_beckman/queue.py` + `lookahead.py` — what beckman already knows how to do, so the new plan doesn't duplicate it.

---

## Hard constraints for the fresh session

- **Pre-flight checks first**: `git status`, `git log --oneline origin/main..main`. If the uncommitted shopping-scraper / remote.py / egg-info work is still there, work around it (use a worktree). If new commits landed since `03a1def`, that's expected (parallel sessions keep shipping) — rebase mental model accordingly.
- **Always run pytest with `timeout` prefix** (CLAUDE.md hard rule).
- **Do not push.** Parent session or user merges. Use a worktree with `isolation: "worktree"` and branch `feat/general-beckman-task13`.
- **Dispatch subagents for plans** (feedback memory: "always use subagents for plans, don't ask").
- **Baseline**: 248 pre-existing failures after the Phase 2b merge. Any new failures in orchestrator/beckman/result_*/watchdog/scheduled_jobs/lifecycle must be investigated before the merge.
- **Preserve src/core shims.** Existing test suites must pass unchanged.
- **Salako `clarify` / `notify_user` are already shipped** — do not reimplement. `_handle_clarification` already routes through salako. Don't redo that migration.

---

## Deferred cleanups (tackle if convenient, otherwise leave)

- `packages/fatih_hoca/src/fatih_hoca.egg-info/` and `packages/hallederiz_kadir/src/hallederiz_kadir.egg-info/` are tracked in git but should be gitignored (build artifacts; they thrash with `pip install -e` runs). Add `*.egg-info/` to `.gitignore` and `git rm -r --cached` them. Separate commit.
- The Telegram `get_telegram()` / `set_telegram()` module-level singleton introduced in Task 1 is fine for now but should eventually become part of a proper Telegram module extraction (spec §12 out-of-scope).

---

## Out of scope (confirmed by spec §12)

- kdv state persistence (cloud rate-limit state is in-memory only, lost on restart)
- Full Telegram module extraction (inbound reply routing still in `telegram_bot.py`)
- Progress-chatter standardization (ephemeral notifications still use direct `self.telegram.send_message` at call sites)

Don't add these to the Task 13 plan. They're separate projects.

---

## Success criteria for the Task 13 follow-up session

- `src/core/orchestrator.py` ≤ 300 lines (hard cap; target 200–250).
- `_handle_*` methods no longer live on the Orchestrator class.
- `lifecycle.py`'s circular `get_orchestrator()._handle_*` delegation is removed.
- `Orchestrator.run_loop` uses `beckman.next_task()` / `beckman.tick()` for the dispatch pump.
- Full test suite baseline no worse than current 248 pre-existing failures. No new failures in the touched modules.
- Manual runtime smoke: `/task`, `/shop`, a mission run end-to-end, a scheduled mission fires, a DLQ retry, a clarification round-trip.
- Scheduling behaviors preserved: age-based priority, swap-aware deferral, paused-pattern filtering. Whatever verification strategy the brainstorm picks, the evidence is in the PR.

---

## How to start the fresh session

Open a new Claude Code session. Paste this as the first message:

> Read `docs/superpowers/plans/2026-04-19-phase2b-task13-handoff.md` first. That's the starting point. It tells you what Phase 2b already shipped, what Task 13 needs to do, and why the prior session stopped short. Follow the brainstorm → spec → plan → execute flow; do not jump to code.
