# Session Summary — 2026-05-04

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries.
Code/commits/security: write normal.

---

## Arc

Started with `docs/handoff/2026-05-04-unify-non-beckman-paths.md` open. User's
core insight cut through volume audits: "any task or subtask not authorized
by Beckman is an antipattern." That principle drove the rest of the session.

Three architectural pivots happened mid-design:

1. **From "second admission API" to "single API"** — user rejected forking
   admission, even with cosmetic differences. Beckman gets ONE enqueue
   contract; every kind of work flows through it.
2. **From "agents await Beckman" to "agents enqueue continuations"** —
   user clarified agents are helpers, not workforce; idle is fine. Only
   high-urgency direct conversations (telegram) await inline. Pump cadence
   issues (wake-on-enqueue, drain) explicitly out of scope — current is
   fine.
3. **From "Phase 3 agent checkpoint state" to "kill agents entirely"** —
   investigating the agent checkpoint task surfaced that 20 BaseAgent
   subclasses are pure config (sys_prompt + allowed_tools, zero methods).
   They're not workers; they're presets disguised as classes. ReAct loop
   should live in dispatcher, not in agent code.

Spec and plan written for the singular admission migration. Six phases
become five (Phase 3 replaced by kill-agents direction). Implementation
shipped through eleven commits across Phase 1, 2, 5, and 4 (partial).

---

## What shipped today (commit-ordered)

| SHA | Phase | What |
|-----|-------|------|
| `89ea7a8` | 1.1 | feat(db): tasks.kind column |
| `82a0e3c` | 1.2 | feat(beckman): enqueue gains kind/parent_id/await_inline/on_complete/next_task_spec |
| `190d9f5` | 1.3 | feat(beckman): on_complete + next_task_spec + inline-waiter terminal hook |
| `89445b8` | 2.4 | refactor(dispatcher): request() becomes alias over beckman.enqueue(await_inline=True) |
| `e777580` | (parallel user fix) | fix(pool_pressure): count provider-wide in_flight, not same-model only |
| `3d73992` | 5.14 | refactor(monitoring): cron-seeded monitoring_check, alerts via notify_user sub-tasks |
| `7030841` | 5.15 | refactor(memory): cron-seeded vector_maint via salako, fixes event-loop wedge |
| `34e6d61` | 2.5 | refactor(beckman): admission owns pool_pressure + fatih_hoca.select + KDV.pre_call + in_flight |
| `f72013e` | 2.5-fixup | test(migration): use setattr DB_PATH not setenv for isolation |
| `e2c0f4c` | 2.5-fixup | fix(in_flight): est_tokens propagation through type-conversion boundaries |
| `d84cd06` | docs | docs(beckman): singular admission spec + plan |
| `8786e99` | (parallel user fix) | fix(db): align max_worker_attempts default with retry policy (6 → 10) |
| `88f225e` | (parallel user fix) | fix(retry): extend backoff ladder to 15 steps + raise default cap |
| `5074a3d` | docs | docs(handoff): kill-agents investigation for next session |
| `5cc94c1` | 4.8 | refactor(telegram): message classifier enqueues with kind=classifier |
| `86863dd` | 4.5 | refactor(grading): grader call enqueues as overhead sub-task |
| `726b0a8` | 4.9 | refactor(telegram): casual chat enqueues with kind=chat |
| `cd30e53` | 4.10 | refactor(classifier): pre-task classifier enqueues with kind=classifier |
| `c174026` | 4.6 | refactor(workflows): summarizer hook enqueues as overhead sub-task |
| `510bfae` | 4.7 | refactor(tools): vision enqueues as tool_call sub-task |
| `34815fc` | 4-test-infra | test(phase4): autouse db-singleton reset fixture in Phase 4 enqueue tests |

**18 of these are mine, 3 are parallel user commits.** All 52 migration +
beckman + Phase 4 tests pass.

---

## Where the migration stands

### Phase 1 — schema + Beckman contract — ✓ shipped

`tasks.kind` column. `enqueue()` widened with `kind`, `parent_id`,
`await_inline`, `on_complete`, `next_task_spec`. Continuation registry
fires `on_complete` handlers + `next_task_spec` chains + resolves
inline waiters at terminal status.

### Phase 2 — dispatcher reshape — ✓ shipped

`dispatcher.request()` is now a thin alias routing through
`beckman.enqueue(await_inline=True)`. Dispatcher's internal `_do_dispatch`
became a Beckman-only worker — admission gates (pool_pressure +
fatih_hoca.select + KDV + in_flight) all moved into Beckman's pump path.
The `raw_dispatch` sentinel in `context.llm_call` marks tasks for direct
dispatcher dispatch (single LLM call, no agent).

Surprise: `KDV.pre_call` was never in dispatcher — already in HaLLederiz
Kadir. Spec's gate map had a stale entry. Real fix was relocating
`fatih_hoca.select` and ensuring dispatcher uses Beckman's preselected
pick instead of re-selecting.

### Phase 3 — agent checkpointable state — ❌ replaced

Investigation revealed BaseAgent subclasses are pure config. ReAct loop
shouldn't live there. **Phase 3 replaced by kill-agents direction**
(see `docs/handoff/2026-05-04-kill-agents.md`).

### Phase 4 — callsite migrations — ◐ 6 of 14 done

Done:
- Site 5: `src/core/grading.py` grader → kind=overhead
- Site 6: `src/workflows/engine/hooks.py` summarizer → kind=overhead
- Site 7: `src/tools/vision.py` vision → kind=tool_call
- Site 8: `src/app/telegram_bot.py:4145` message classifier → kind=classifier
- Site 9: `src/app/telegram_bot.py:4570` casual chat → kind=chat
- Site 10: `src/core/task_classifier.py` pre-task classifier → kind=classifier

Pending:
- Sites 1-4 (BaseAgent: ReAct main + structured_emit + alt-prompt-retry +
  self-reflection) — blocked on kill-agents. After kill, "agent" calls
  vanish; ReAct lives in dispatcher.
- Sites 11-14 (shopping pipeline) — deferred. Shopping pipeline has known
  architectural issues (memory: `project_shopping_pipeline_20260414.md`,
  `project_shopping_pivot_20260420.md`). Migrate after pipeline shape
  is settled.

### Phase 5 — mechanical migration — ✓ shipped

`monitoring_check` (5min URL/GitHub poll) and `vector_maint_wal` (30min) +
`vector_maint_snapshot` (24h) now run as cron-seeded mechanical tasks via
salako executors. The old `asyncio.create_task` background loops in
`src/infra/monitoring.py` and `src/app/run.py` are deleted. Vector maint
also fixes the mission 46 incident's 120s sync I/O event-loop wedge by
delegating to ChromaDB's `asyncio.to_thread` paths.

### Phase 6 — residue cleanup — ⏸ waiting

Per spec, wait ≥1 week post-Phase-4 stability before deleting:
- `dispatcher.request()` alias
- `current_task_id` ContextVar
- `5f7f905` est_tokens shim (already partially deleted in `34e6d61`)
- Any unused imports surfacing post-migration

---

## Unfinished cliffs (next session entry points)

### Highest priority: kill-agents

Handoff at `docs/handoff/2026-05-04-kill-agents.md`. 8 investigation
tasks before code touches. Migration is big but valuable — ~1700 LOC of
preset config, plus ~5800 LOC of BaseAgent.execute() to disentangle. The
end state:

- `src/core/profiles.py` — registry mapping `agent_type` name → Profile
  (sys_prompt, allowed_tools, iteration_cap, min_tier, etc.)
- `src/core/react.py` (or inlined into dispatcher) — ReAct iteration loop
- `src/core/tool_executor.py` — tool exec extracted from BaseAgent
- `src/agents/` — deleted
- Orchestrator's `get_agent(agent_type).execute(task)` branch dies;
  every LLM-typed task flows through `dispatcher.dispatch()` with
  profile-aware iteration

Migration sites 1-4 (today's deferred BaseAgent overhead calls) become
trivial after this kill.

### Second priority: shopping pipeline architecture

Memory items flag persistent issues:
- `project_shopping_pipeline_20260414.md` — relevance filtering, scraper
  reliability, intelligence modules not wired
- `project_shopping_pivot_20260420.md` — stop reinventing product
  matching, trust site ordering, review synthesis is the real value

Migration sites 11-14 wait for this architectural decision. Shopping
pipeline currently has its own coordination layer that mostly bypasses
Beckman. Could become a workflow chain (next_task_spec) or could collapse
entirely if "trust sites + synthesize reviews" wins.

### Third priority: pump throughput

Out of scope for current spec. But several places will hit the 1-task /
3s pump cap as overhead admissions grow:

- Heavy concurrent overhead (grader + structured_emit + summarizer all
  firing per agent task)
- Telegram chat under load (await_inline eats 3s pump-tick latency)

Solutions deferred but documented in spec §6:
- Wake-on-enqueue (`asyncio.Event` signal from `enqueue` to pump)
- Drain loop (admit until capacity full per tick, not single task per
  tick)

If user-perceived latency degrades after Phase 4 is fully landed, this
moves up.

### Fourth priority: pool_pressure for sleeping parents

Spec §6 pitfall #3: when parent task awaits child via `await_inline`
(today only telegram + classifier; soon overhead in many places), parent
holds its in_flight slot but isn't consuming GPU. pool_pressure must
discount sleeping parents or it'll prematurely block child admissions
on saturated GPU.

Cleanest fix: Beckman tracks `in_flight_entry.is_sleeping` flag, set by
the await machinery, cleared on resume. pool_pressure subtracts sleepers
from active count.

---

## Memory entries written this session

- `project_beckman_singular_admission_20260504.md` (parent project)
- `project_kill_agents_20260504.md` (kill-agents direction)

Both indexed in `MEMORY.md`.

---

## DON'T

- Don't revert any Phase 1/2/4/5 commit. Each was tested standalone +
  combined. Reverts cascade into the alias chain.
- Don't try to migrate sites 1-4 without first doing kill-agents — those
  sites live in BaseAgent, which dies entirely. Migrating them
  pre-kill is wasted work.
- Don't migrate sites 11-14 without architectural decision on shopping.
- Don't ship Phase 6 cleanup until Phase 4 is fully landed AND stable
  for ≥1 week. The alias path is the safety net during transition.
- Don't run pytest without `timeout` prefix.
- Don't run pytest without `.venv/Scripts/python -m pytest` (salako
  not in conftest's `_PACKAGE_SRCS`).
- Don't use `monkeypatch.setenv("DB_PATH", ...)` — silently fails when
  module's already imported. Use `monkeypatch.setattr(_db_mod, "DB_PATH",
  ...)` and reset `_db_connection` between tests.

---

## Open questions for next session

1. Kill-agents: dict registry vs YAML? (handoff §"Open questions" #1)
2. ReAct location: dispatcher inlined vs `src/core/react.py`?
3. Shopping pipeline shape: workflow chain vs single-call collapse?
4. Pump throughput: revisit out-of-scope items if Phase 4 latency hurts?
5. `await_inline` timeout strategy: 600s default OK or per-call override?
6. Phase 6 cleanup timing: 1-week minimum, but trigger needs to be
   measurable — define a "stable" criterion.

---

## Files touched (final list)

Created:
- `docs/superpowers/specs/2026-05-04-beckman-singular-admission-design.md`
- `docs/superpowers/plans/2026-05-04-beckman-singular-admission.md`
- `docs/handoff/2026-05-04-kill-agents.md`
- `docs/handoff/2026-05-04-session-summary.md` (this file)
- `tests/beckman/test_kind_column.py`
- `tests/beckman/test_enqueue_contract.py`
- `tests/beckman/test_continuations.py`
- `tests/migration/test_dispatcher_alias_compat.py`
- `tests/migration/test_admission_gates_run_once.py`
- `tests/integration/test_monitoring_check_executor.py`
- `tests/integration/test_vector_maint_executor.py`
- `tests/core/test_grading_enqueue.py`
- `tests/core/test_task_classifier_enqueue.py`
- `tests/app/test_telegram_classifier_enqueue.py`
- `tests/app/test_telegram_casual_chat_enqueue.py`
- `tests/workflows/engine/test_hooks_enqueue.py`
- `tests/tools/test_vision_enqueue.py`
- `packages/general_beckman/src/general_beckman/continuations.py`
- `packages/salako/src/salako/executors/monitoring_check.py`
- `packages/salako/src/salako/executors/vector_maint.py`

Modified (significant):
- `src/infra/db.py` — `tasks.kind` column add
- `packages/general_beckman/src/general_beckman/__init__.py` — enqueue
  widening, terminal hook, inline waiters
- `src/core/llm_dispatcher.py` — request() alias, dispatch() body,
  admission relocation
- `src/core/orchestrator.py:262-279` — raw_dispatch sentinel branch
- `packages/general_beckman/src/general_beckman/cron_seed.py` — new
  cron markers (monitoring_check, vector_maint_wal/snapshot)
- `packages/salako/src/salako/__init__.py` — executor registration
- `src/infra/monitoring.py` — `run_monitoring_loop` deleted
- `src/app/run.py` — `_vector_maint_loop` deleted
- `src/core/grading.py:305` — site 5 migration
- `src/workflows/engine/hooks.py:46` — site 6 migration
- `src/tools/vision.py:29` — site 7 migration
- `src/app/telegram_bot.py:4145, 4570` — sites 8, 9 migration
- `src/core/task_classifier.py:258` — site 10 migration
- `packages/fatih_hoca/src/fatih_hoca/selector.py` — est_tokens
  propagation fix
- `packages/nerd_herd/src/nerd_herd/{client,exposition}.py` — same

Deleted: nothing (alias path keeps existing callers safe through Phase 4
landing).
