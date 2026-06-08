# SP4b Plan 3 — SHIPPED to branch (crisis/incident/press_kit off `await_inline`)

**Date:** 2026-06-08. **Branch:** `worktree-cps-sp4b-plan3` (NOT merged). **Restart-gated.**
**Spec:** `docs/superpowers/specs/2026-06-08-cps-sp4b-plan3-design.md`. **Plan:** `docs/superpowers/plans/2026-06-08-cps-sp4b-plan3.md`.

## What shipped
The 3 Plan-3 `await_inline=True` mechanical-verb callers are removed. Verified by grep — zero `await_inline=True` code remains in `crisis_draft_holding.py`, `incident_draft_update.py`, `press_kit_assemble.py`, `src/comms/producers.py`.

Approach: **CPS continuation pattern (`on_complete`/`on_error`), mirroring Plan 1's reviews migration — NOT `degrade_on_exhaustion`.** The handoff assumed a producer→workflow-sink split (which needs `degrade_on_exhaustion`, a Beckman core change). But these 3 are standalone mechanical dispatches, so the existing continuation substrate fits: the producer enqueues the LLM hop with `on_complete`/`on_error`; the mechanical sink does the parse + canned fallback + founder card. `fire_for_task` fires `on_error` on terminal failure and `reconcile_continuations` re-fires after restart, so the canned fallback is preserved on the failure path. **Zero General Beckman core change. No `degrade_on_exhaustion` built.** (Founder cleared scope to plumbing-only; press_kit grounding / `press_kit_freshness` / demo-pipeline gap all left out of scope.)

## New files
- `src/comms/__init__.py`, `src/comms/producers.py` — CPS producers (prompts live here, out of mr_roboto): `enqueue_crisis_holding`, `enqueue_incident_update`, `enqueue_press_kit` + `_enqueue_press_kit_audience`. Plus `_AUDIENCE_INSTR` (moved verbatim from the press_kit verb).
- `packages/mr_roboto/src/mr_roboto/executors/comms_continuations.py` — mechanical sinks: `_crisis_resume/_err`, `_incident_resume/_err`, `_press_kit_resume/_err` + emitters; registered via `register_continuations()` and added to `continuations._HANDLER_MODULES` (restart-recovery).
- `packages/mr_roboto/tests/test_comms_continuations.py` — 10 tests (mocked, no DB).

## Per-verb notes
- **crisis/draft_holding** — single producer + sink. Verb keeps `parse_variants` + `canned_variants`. Router branch repointed to enqueue the producer.
- **incident/draft_update** — `run()` kept as the PREP step (validation + incidents-table fetch + `_redact_alert`), tail swapped to enqueue the producer. **SAFETY:** the final 3-pass redaction (`finalize_redaction` = `redact_internal`→`redact_secrets`→`redact_user_pii`) moved into the sink; BOTH `_incident_resume` and `_incident_resume_err` apply it before emitting — no path to the founder card skips redaction (reviewer-verified, adversarial IP/hostname/stack-trace checks pass). The `incident_update_review` post-hook is now bypassed (card emitted from sink); its code is left in place. Router branch unchanged.
- **press_kit/assemble** — serial chain of 4 producers (investor→journalist→partner→candidate), per-audience prompting preserved. Sink stages each draft into `cont_state["staged"]` and enqueues the next audience; the final hop calls `assemble_from_drafts` (= the old `run()` body, byte-for-byte except version-as-param + the LLM line). `on_error` stubs the failed audience and continues the chain. Router branch computes version + builds the source dict + enqueues.

## IMPORTANT — these 3 verbs are DEAD (built, never triggered)
No production code enqueues `crisis/draft_holding`, `incident/draft_update`, or `press_kit/assemble`. `/crisis open` (`telegram_bot.py:5704`) opens the event and points the founder at the playbook — it does NOT enqueue the draft verb. So there was no live trigger to repoint (only the router branches), and no live integration to test. The producer is the future entry point if/when a feature wires these. Live verification is N/A until then.

## Tests / verification
- 10/10 green: `packages/mr_roboto/tests/test_comms_continuations.py` (run with the MAIN-repo venv `…/.venv/Scripts/python.exe`, `--timeout=120`). Includes 2 structural tests (all 6 handlers registered + module in `_HANDLER_MODULES`).
- The full `mr_roboto` suite was deliberately NOT run — DB-touching suites hit the live `kutai.db` WAL and risk crash-looping KutAI. Targeted + import verification only.
- Bare `python -c "import mr_roboto…"` FAILS in the worktree because editable installs point at the MAIN repo, not the worktree. Not a defect — pytest's root conftest injects worktree paths. **Consequence: this goes live only on merge to main + restart.**

## SP5 ledger after Plan 3 (re-grep `await_inline\s*=\s*True`)
Remaining callers:
- `mr_roboto/reviews_classify.py:97`, `reviews_draft_reply.py:123` — **Plan 1**, branch `worktree-cps-sp4b`, NOT merged.
- `src/app/jobs/investor_bullets.py:211`, `src/core/task_classifier.py:284` — SP5 carve-outs (SP5 migrates).
- `src/core/llm_dispatcher.py:273` — shopping `request()` shim (T11 delete, evidence-gated).

**To delete `await_inline`:** merge Plan 1 + Plan 3, retire the shopping shim (T11), then SP5 migrates the 2 carve-outs and deletes the primitive + shim. Plan 3 cleared 3 of the 5 mechanical-verb callers.

## Commits (branch `worktree-cps-sp4b-plan3`)
spec → plan → Task 1 scaffold → Task 2 crisis → Task 3 incident → Task 4 press_kit → Task 5 verify/handoff. Each task: TDD + two-stage review (spec compliance + code quality), all APPROVE.

## Merge note
Branch off local HEAD (8 ahead of origin at branch time). NOT merged. Merge when bot-quiet (live bot does `git add -A` commit-storms on main — that's why this ran in a worktree). Restart required to load (editable installs → main).
