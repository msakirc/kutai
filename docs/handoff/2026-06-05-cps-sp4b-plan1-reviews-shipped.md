# SP4b Plan 1 (reviews CPS) — SHIPPED to branch, merge-ready

**Date:** 2026-06-05
**Branch:** `worktree-cps-sp4b` (worktree `.claude/worktrees/cps-sp4b`), 6 commits `f40b5017`→`ba7a0387` on base `516e4ab4`.
**Spec:** `docs/superpowers/specs/2026-06-05-cps-sp4b-design.md`. **Plan:** `docs/superpowers/plans/2026-06-05-cps-sp4b-reviews.md`.

## What shipped (Plan 1 of 2)

The `reviews_classify` + `reviews_draft_reply` LLM calls left mr_roboto. Each LLM hop is now an admitted Beckman task on the pump via CPS:

- **Producer** `src/reviews/producers.py` (OUTSIDE mr_roboto) — `enqueue_classify` / `enqueue_draft_reply`: load review, build prompt + raw_dispatch overhead spec, `enqueue(..., lane=oneshot, on_complete=, on_error=, cont_state=)`. Holds the prompts + platform conventions.
- **Mechanical sink** `packages/mr_roboto/src/mr_roboto/executors/reviews_continuations.py` — `_classify_resume`/`_classify_resume_err` (enum-validate → UPDATE external_reviews → low-star founder_action + bug-investigation side-effects; heuristic fallback on `on_error`) and `_draft_reply_resume`/`_err` (surface draft via founder_action, **never auto-posts**; canned fallback). Registered via `register_continuations()`.
- **Restart-recovery**: `mr_roboto.executors.reviews_continuations` added to `general_beckman.continuations._HANDLER_MODULES`.
- **Router** branches `reviews/classify` + `reviews/draft_reply` (`mr_roboto/__init__.py`) now enqueue the producer → `Action(completed, {enqueued: tid})`.
- **Cron** `src/app/jobs/reviews_poll_daily.py` enqueues a producer per unclassified review (`total_enqueued`).
- **Verbs stripped**: `reviews_classify.py` / `reviews_draft_reply.py` lost their `_call_llm_*` (−433 LOC); keep only mechanical helpers the sink reuses; `run()` delegates to the producer (legacy shim).

## Tests (all green on branch HEAD)

- New slice `tests/reviews/test_reviews_cps.py` — 11/11 (producer specs, sink persist/side-effects, never-auto-post, on_error fallbacks, `_HANDLER_MODULES`, router, cron).
- `tests/z7/test_b8_reviews_harvest.py` — 35/35 (10 legacy synchronous-contract tests rewritten to drive the sink/producer against the real `db` fixture).
- `packages/mr_roboto/tests` — 799 pass, **1 pre-existing unrelated** fail: `test_every_dispatcher_action_is_in_registry` → `publish_preview_pages` missing from `VERB_REVERSIBILITY` (someone else's action; my reviews actions have slashes and don't match the test regex).
- `packages/general_beckman/tests` — 241 pass, **3 pre-existing unrelated** fails (`test_admission_cache` ×2, `test_posthook_llm_child::test_emit_child_spec_is_raw_dispatch`). **Proven pre-existing**: all 3 fail at pure base `516e4ab4` (before any SP4b commit). They PASS on the main checkout only because main's *uncommitted* schema-gate/materializer work fixes them — the clean worktree lacks those edits.

## Not merged — and why

`main` checkout has **uncommitted in-flight work** (schema-gate/materializer; `M src/app/telegram_bot.py` + docs). Merging this branch into a dirty tree risks entangling that work. The branch is committed + clean + merge-ready. **To ship:** once main's tree is clean, `git merge --no-ff worktree-cps-sp4b` and re-verify the slice on the merged tree. Push to main (repo convention).

## Hard gate — substrate verified

Done before building (gate step 3 via live DB, read-only): all tasks `lane=oneshot`; **156 post-hook children (constrained_emit/self_reflect/grade/critic) all completed, zero orphaned** — the SP3b orphaning bug is absent live. `husam` is `pip install -e`'d. **Founder `/restart` still required** to pick up husam + run the new reviews code (until restart, live runs the old synchronous verbs from main).

## ⚠️ Environment incidents (founder, please look)

1. **16 zombie pytest** (venv + global-Python310 pairs) had accumulated over 3h, mutually deadlocking on the shared `kutai.db` WAL (the documented crash-loop hazard). Killed all 16; KutAI/llama-server untouched. The parallel shopping session running pytest concurrently with this one re-triggers the deadlock — **don't run pytest in both at once.**
2. **TWO KutAI wrappers running**: PID 31628 (venv python) + PID 40528 (global Python310), both since May 23. Likely the parallel session launched a second stack with the system Python. Two stacks on one DB = contention. Worth reconciling to one.
3. **0 llama-server processes** currently — no local model loaded.

## Plan 2 — PENDING (separate plan, distinct subsystem = workflow engine)

Workflow splits for **demo_storyboard, incident_draft_update, press_kit_assemble, crisis_draft_holding**. demo+incident already workflow steps (split in place); press_kit (4→1 fan-in) + crisis become new workflows. Needs workflow-engine investigation (agent-step prompt build, expander fallback/degraded-emit so producer DLQ doesn't block the sink, workflow launch from `/`-command). Spec §4-5 (matrix + 3-tier sink) + §7 (fallback) + §12 (open questions).

## Related

Shopping LLM-dispatch handoff (for the parallel session): `docs/handoff/2026-06-05-shopping-llm-dispatch-handoff.md` — carrier-by-job-shape + the "is the consumer waiting?" axis.
