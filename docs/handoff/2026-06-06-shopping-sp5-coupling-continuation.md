# Handoff — Shopping SP5-coupling: continue from Task 7

**Date:** 2026-06-06
**For:** the session continuing the shopping `request()`/`await_inline` retirement.
**From:** the session that shipped Tasks 1-6.
**Read first:** spec `docs/superpowers/specs/2026-06-05-shopping-workflow-step-migration-design.md` (rev2), plan `docs/superpowers/plans/2026-06-05-shopping-workflow-step-migration.md`, memory `project_shopping_sp5_coupling_20260605`. Carrier rules: `docs/handoff/2026-06-05-shopping-llm-dispatch-handoff.md`.

---

## TL;DR — is SP5 unblocked from shopping? NO.

The v3 producer-agent machinery is **built, wired, and unit-tested green — but dormant.** v2 is still the live workflow; **zero `request()` callers retired.** SP5 stays blocked until T8-T11 land AND the separate `src/shopping/` spec lands.

```
request() live callers (5) — ALL still active:
  1 single_shot.run (shopping_clarifier=single_shot)   → killed by T9
  2 pipeline_v2.py:363 _grouping_llm_call               → dead once v3 live (T10) + v2 retired (T11)
  3 pipeline_v2.py:487 _synthesis_llm_call              → "
  4 labels.py:22 _label_llm_call                        → "
  5 src/shopping/intelligence/_llm.py:42                → SEPARATE SUBSYSTEM, own spec (not this plan)
dead-legacy (2): reflection.py:89, constrained_emit.py:147 → delete in T11
SP5 deletes await_inline only after request() fully gone AND both shopping subsystems migrated.
```

---

## What shipped (Tasks 1-6, all on `main`, all green, ADDITIVE/dormant)

| Commit | Task | What |
|--------|------|------|
| `23e98554` | E1 | compare-all keyboard → restart-proof `vc:{mid}:{tid}:{gid}` (bug fix, LIVE) |
| `c9d16560` | T1-3 | 3 producer agents `src/agents/shopping_{grouper,labeler,synthesizer}.py` (prompt-only react; registered in `src/agents/__init__.py`; in `test_prompt_quality` LOW_TRAFFIC) |
| `50e9c811` | T4 | group/label prep+apply handlers in `pipeline_v2.py` |
| `ad3e52d3` | T5 | synth prep+apply handlers in `pipeline_v2.py` |
| `3b690c4b` | T6 | `shopping_v3.json` (chosen + variant paths; NOT compare-all) |

**Shared pure helpers extracted (legacy `step_*` refactored to reuse — behavior-equivalent):**
- `pipeline_v2.build_group_view`, `pipeline_v2._parse_grouping_raw`
- `labels.build_label_view`, `labels.apply_labels`
- `pipeline_v2.gather_review_snippets`, `pipeline_v2._parse_synthesis_raw`

**New step handlers** (registered in `_STEP_HANDLERS_V2`, dispatched by `step.name` via `ShoppingPipelineV2.run`): `handler_group_prep`, `handler_group_apply_label_prep`, `handler_label_apply_filter_gate`, `handler_synth_prep`, `handler_synth_apply`.

**Tests:** `tests/agents/test_shopping_producers.py`, `tests/shopping/test_v3_handlers.py` (8), `tests/shopping/test_v3_wiring.py` (6). 744 shopping suite green (1 deselected = `test_performance.py::test_record_1000_requests_total_time`, a wall-clock flake — environmental, not ours).

### Key facts established (don't re-derive)
- **Agent-context mechanism:** `runner.build_step_description(instruction, input_artifacts, artifact_contents, done_when)` (`src/workflows/engine/runner.py:66`) appends `## Input Artifacts\n\n{content}` to the step instruction → becomes the agent's user message. So a producer step's `input_artifacts` content IS surfaced to the LLM. (Spec risk 3 — RESOLVED.)
- **synth split shape:** `synth_prep` emits TWO artifacts — `synth_input` (title + snippets → the synthesizer's input_artifact) and `synth_meta` (group + candidates + snippet_count → the apply step's input_artifact, NOT given to the producer).
- **raw_dispatch routing exists already:** pump routes any task with `context.llm_call.raw_dispatch==True` → `husam.run` (`orchestrator.py:355-382`). The producer-agent path does NOT use this — it routes `agent:<type>` → `coulson.execute` → react one-pass. (raw_dispatch is only relevant if you choose husam-inline for compare-all; see T8.)
- **`_task_result_to_request_response` is NO LONGER shared on main** — vision/brand_voice already moved off it in SP4a. Only `request()` calls it now → can die with `request()` (re-grep to confirm at T11).
- **skip_when operator:** `should_skip_workflow_step` (`hooks.py:1047`) supports only `<artifact>.<dot.path> == '<lit>'` / `!= '<lit>'`. v3 uses `group_input.has_residuals == 'false'` (prep emits the STRING `"false"`/`"true"`) and `gate_result.gate.kind != 'chosen'`. No `!field` / bare-bool support.

---

## Remaining work

### Task 7 — wiring/deadlock integration test (additive, SAFE)
`tests/shopping/test_v3_wiring.py` already covers structural wiring. The plan's T7 adds two integration tests whose drivers were left as `...`:
1. **prep → stub producer → apply through the real `ArtifactStore`** — instantiate `ArtifactStore(use_db=False)`, run `handler_group_prep` → store `group_input`/`groups_state` → write a stub producer `final_answer` JSON to `group_raw` → run `handler_group_apply_label_prep` → assert 2 groups in `groups_state`. (Shapes: copy from `test_v3_handlers.py`.)
2. **deadlock regression** — mirror the SP3b deadlock-regression test (drive real `enqueue → next_task` with the oneshot lane reservation maxed; assert a producer-shaped task still admits because no parent holds a slot awaiting it). Find the SP3b test: `rg -n "deadlock|ONESHOT_CONCURRENCY|reserve" packages/general_beckman/tests`.

Low priority — the unit tests already prove the handlers; this proves the seam. Do it if you want belt-and-suspenders before the live switch.

### Task 8 — compare-all sequential CPS chain (HAS A DESIGN GAP — decide first)
compare-all (`2.3`) is NOT in `shopping_v3.json` yet. Plan §Task 8 specifies a CPS continuation chain (`src/workflows/shopping/compare_continuations.py`): `compare_init` (workflow step) enqueues the synth producer for line 0 with `on_complete="shopping.compare_next"`; the continuation appends the card and enqueues the next line or finalizes. ≤5 lines (`MAX_CLARIFY_OPTIONS`), sequential, no fan-out join, no engine loop (the engine has NO loop primitive — confirmed).

**THE GAP (do not guess — founder flagged "no guesses"):** two spots need tracing before coding:
1. `_finalize_compare(mission_id, header, cards)` — writes `shopping_response` then must **advance the mission past the compare branch** so delivery proceeds. Trace `_resume_mission_at_step` (`telegram_bot.py`) + how a mission advances when a step completes off-graph. The continuation fires from Beckman's CPS, OUTSIDE the workflow step graph — so finalizing must bridge back to mission advance. This CPS→workflow bridge is the unknown.
2. `_producer_spec(line, mission_id)` — build the synthesizer producer spec the continuation enqueues. Either (a) a `raw_dispatch` spec routed to husam (`context.llm_call={raw_dispatch:True, call_category:"main_work", agent_type:"shopping_synthesizer"-equivalent, messages:[SYNTHESIS_PROMPT data], difficulty, estimated_*}` — mirror `reviews_continuations`/`posthook_continuations` spec-build), OR (b) an admitted `agent:shopping_synthesizer` task. (a) is the established CPS shape; prefer it.

CPS mechanics reference (copy patterns): `mr_roboto/executors/reviews_continuations.py` (SP4b) + `general_beckman/posthook_continuations.py` (SP3). Register `compare_continuations` in `general_beckman.continuations._HANDLER_MODULES` or the handler is absent after restart (silent bug). lane=`oneshot` ONLY. Use the dual-shape `_extract_content` (normal `result['result']['content']` vs reconcile top-level `result['content']`).

**Also in T8:** telegram re-route — delete the direct `_handler_format_compare` call in `_run_compare_all_and_reply` (`telegram_bot.py:~10960-10985`), resume the mission at the compare path instead so `2.3*` runs in the pump. Keep the `vc:` keyboard (already E1-fixed) + final render.

**Alternative worth considering:** the chosen/variant paths already work without CPS. If the CPS bridge proves messy, compare-all could instead be a fixed cap of `MAX_CLARIFY_OPTIONS` conditional synth triads (`2.3_0..2.3_4` with `skip_when` on a per-index flag) + a `compare_assemble` with `depends_on` all (the engine's native `depends_on` IS the join — no CPS, no loop). Founder earlier called fixed-steps "ugly" and chose sequential CPS, but under the workflow carrier the join is free, so reconsider the tradeoff with the founder if the bridge is costly.

### Task 9 — `shopping_clarifier` → react (CHANGES LIVE v2 BEHAVIOR)
One-line: drop `execution_pattern = "single_shot"` from `src/agents/shopping_clarifier.py:21`. Kills `request()` caller #1. BUT this changes the **live v2** 0.2 step from single_shot → react (a tool loop). Do it **near the v3 switch** (T10), and live-test the vague-query clarify UX parity (the `needs_clarification` pause + the questions). Test (already drafted, removed from this session's commit to avoid a red on the pre-flip tree): assert `execution_pattern == 'react_loop'`.

### Task 10 — switch launch to v3 (LIVE — gated on restart triage)
Flip `src/app/telegram_bot.py:8490-8498` (`_create_shopping_mission` `wf_map`): `deep_research`/`research` values + the default → `shopping_v3`. Repoint `quick_search_v2.json:8` `escalation_target` → `shopping_v3`. Update `tests/integration/test_e2e_llm_pipeline.py:142`.
**Then LIVE-validate via Telegram** (founder, after `/restart`): vague query (clarifier react), specific query (chosen path), multi-line → variant pick (variant path), "compare all" (needs T8). Check `SELECT lane,status,COUNT(*) FROM tasks WHERE mission_id=<m> GROUP BY 1,2` — producer children complete on `oneshot`, none orphaned.

⚠️ **DO NOT live-validate until the 2026-06-05 restart-error triage is clear** (schema-gate 240-tighten / SP4a / materializer / S7-S6 all went live together — see `docs/handoff/2026-06-05-deterministic-materializer-handoff.md` §0 and `[Restart triage 2026-06-05]` memory: that triage found only 4 DLQs + fixed 2 regressions, so it may already be clear — confirm before relying on shopping live signal).

### Task 11 — delete the shim (LIVE — final SP5 unblock for THIS subsystem)
Only after T10 live-validated + no v2 missions in flight. Order:
1. Re-grep: `rg -n "\.request\(|_request_kwargs_to_spec|_task_result_to_request_response|single_shot|execution_pattern\s*=\s*.single_shot" src packages --glob '!*/tests/*'` — STOP if any unexpected live caller.
2. Delete dead callers `reflection.py:89`, `constrained_emit.py:147`.
3. Delete `single_shot.py` + the `single_shot` branch in `coulson/__init__.py:88-93`.
4. Delete `LLMDispatcher.request()` + `_request_kwargs_to_spec` (+ `_task_result_to_request_response` after the re-grep confirms 0 other callers).
5. Retire `shopping_v2.json` + its v2-only handlers (`_handler_group_label_filter_gate`, `_handler_synth_one`, `_handler_format_compare`, `_grouping_llm_call`, `_synthesis_llm_call`, `labels._label_llm_call`). KEEP the deterministic functions v3 reuses (`step_filter`, `step_variant_gate`, `format_group_card`, `step_compare_all`, `_fetch_community_reviews`, all the `*_view`/`_parse_*`/`gather_*` helpers).

### The OTHER SP5 prerequisite — `src/shopping/` (NOT in this plan)
`src/shopping/intelligence/_llm.py:42` is `request()` caller #5, in the `ShoppingPipeline` / `src/shopping/` subsystem (25.9k LOC — the simple-query/two-tier path, NOT the `src/workflows/shopping/` v2/v3 pipeline). Per the dispatch handoff it gets its **own brainstorm + spec**; its carrier may be **interactive CPS resume / husam-inline** (a user waiting synchronously), NOT workflow steps. **SP5 deletes `await_inline` only after BOTH this plan AND that spec land.** Scope it separately.

---

## Build/test discipline (carry over)
- `timeout 120 pytest` always; NEVER concurrent pytest (SQLite lock crash-loops live KutAI); `tests/` and `packages/*/tests/` in SEPARATE invocations (colliding conftest).
- Bare `python -c` misses the editable packages (`fatih_hoca` etc.) — validate via pytest (conftest injects paths), not `python -c`.
- Commit per task; push to `main`. Live KutAI runs from the main checkout; founder restarts via Telegram (never `taskkill`).

## Suggested order for next session
1. T8 design decision (CPS bridge vs fixed-cap join) — 5-min founder call, then build.
2. T7 (optional, safe) alongside.
3. Confirm restart triage clear → T9 + T10 together → live-validate.
4. T11 (delete shim) once validated.
5. Separate session: `src/shopping/` spec → then SP5 can delete `await_inline`.
