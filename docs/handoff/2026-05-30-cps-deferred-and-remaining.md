# CPS migration — deferred / remaining / skipped items

**As of:** 2026-05-30, after SP3b shipped (`1183f521`) + post-merge review + fixes (`132afa3d`, `f2321ef3`, `59f38478`).
**Scope:** everything NOT done across the SP3b ship + review + fix arc, plus carried-forward migration scope (SP4/SP5) and pre-existing reds surfaced along the way.
**Legend:** 🔴 blocking/founder-action · 🟠 deferred fix (real, non-blocking) · 🟡 minor/cleanup · 📋 remaining migration scope · 🔵 pre-existing red (not CPS).

---

## 🔴 Founder actions — gate trusting grading + SP4

| # | Item | Why | Owner |
|---|------|-----|-------|
| 1 | `.venv\Scripts\pip install -e packages\husam` | Orchestrator pump + critic_gate do `import husam`; absent in `.venv` → `ModuleNotFoundError` on first raw_dispatch (ungraceful, may DLQ). No install script — manual. | founder |
| 2 | `/restart` via Telegram | Load SP3b + the fixes into the live orchestrator (running pre-SP3 code since 05-27). Never `taskkill`. | founder |
| 3 | **Validate the post-hook substrate e2e** | The review proved post-hook children were orphaned on a phantom lane → the system has **never dispatched in prod**. Run ONE graded mission; confirm `constrained_emit→self_reflect→grade` children dispatch on `lane=oneshot`, rewrite the result, and complete. Check `SELECT lane,status,COUNT(*) FROM tasks GROUP BY 1,2` (want oneshot children completing, not piling pending). | founder |

---

## 🟠 Deferred fixes — real, non-blocking

| # | Item | File | Notes | Severity |
|---|------|------|-------|----------|
| 4 | **critic_gate shape-(b)** | `packages/mr_roboto/src/mr_roboto/critic_gate.py` | `produce_verdict` calls `husam.run` INLINE from the mechanical (shape-a) — removes the *dispatcher* call but a mechanical still reaches a worker un-admitted. The LLM-free `confirm_gate` exists but is **UNUSED** (dead seam). End state = separate admitted producer step + mechanical confirm sibling. **Your architectural call.** | MEDIUM |
| 5 | **add_task lane validation** | `src/infra/db.py:4464` | `_lane = lane or "oneshot"` persists ANY lane string verbatim — no guard. This is what let the phantom `lane="overhead"` bug stay silent. Defensive fix: validate `lane ∈ {oneshot, ongoing}`, coerce-to-oneshot + WARN otherwise. Prevents the whole bug class. | MEDIUM (hardening) |
| 6 | **brand_voice_lint nested await_inline** | `packages/general_beckman/src/general_beckman/posthook_handlers/brand_voice_lint.py` (`_run_llm_tone_pass`, ~411) | A post-hook handler making a NESTED `await_inline` LLM call — sibling deadlock pattern. Folds into SP4 or a sweep. Use the spawned-child path. | MEDIUM |
| 7 | **e2e through-real-DB chain test** | new test | No single test drives `emit→reflect→grade` through a real DB with mocked LLM children, nor chain + independent post-hooks (grounding/verify_artifacts) co-existing. The pump-driven test covers dispatch; this would lock rewrite-before-grade ordering + the co-existence path. (Item 3's mission covers it manually for now.) | LOW |
| 8 | **Dead code removal** | `coulson/reflection.py::self_reflect`, `src/workflows/engine/constrained_emit.py::maybe_apply` | Zero prod callers after SP3b Task 7. Delete + relocate their behavioral coverage (`tests/test_constrained_emit.py` ×14 tests the dead `maybe_apply`; coulson reflection tests). Scheduled for the SP5 sweep. | LOW |

---

## ⚠️ Investigate (surfaced by the review, OUT of CPS scope but possibly systemic)

| # | Item | Notes |
|---|------|-------|
| 9 | **`ongoing` lane may also be unpumped** | The orchestrator pump (`run_loop` → `next_task()`) only ever uses `LANE_ONESHOT`. The review found NO production caller of `next_task("ongoing")`/`("overhead")`. Webhook/alert_triage/cron/support_ticket tasks are enqueued `lane="ongoing"` (`webhook_listener.py:360`). If nothing pumps the ongoing lane, those are orphaned too (same class as the overhead bug). **Verify:** is there a separate ongoing-lane pump, or is the Z8 ongoing lane dead? Live DB shows 57k tasks, ALL `lane=oneshot` — zero `ongoing` rows ever, which is suspicious. Not CPS, but worth a look. | POTENTIALLY HIGH |

---

## 🟡 Minor / cleanup

| # | Item | File |
|---|------|------|
| 10 | Stale comment "ctx above" (no `ctx` defined above post-refactor) | `apply.py` (Task-6 residual) |
| 11 | `_POSTHOOK_CHAIN_ORDER` (rewrite.py) ↔ `_REWRITE_POSTHOOK_KINDS` (apply.py) — overlapping constants, no sync guard; a future rewriter kind must update both | rewrite.py / apply.py |
| 12 | `_source_verdict_locks` dict grows unbounded (one empty `asyncio.Lock` per source id; eviction intentionally removed to fix a race). Cheap, but note for a future bounded-LRU if it ever matters | apply.py |

---

## 📋 Remaining migration scope

| # | Phase | Scope | Pointer |
|---|-------|-------|---------|
| 13 | **SP4 — tools + mechanicals** | vision tool (riskiest — mid-ReAct inline-result design fork, like `dispatcher.request` was); mr_roboto LLM executors (`reviews_*`, `incident_draft_update`, `press_kit_assemble`, `demo_storyboard`, `crisis_draft_holding`); yalayut `synthesize`; brand_voice_lint (#6). | `docs/handoff/2026-05-30-sp4-kickoff.md` |
| 14 | **Shopping migration off `request()`** | `coulson/single_shot.py` (shopping_clarifier) + `src/workflows/shopping/{pipeline_v2,labels}.py` + `src/shopping/intelligence/_llm.py` still call the shopping-only `request()` shim. Shopping = workflow *definitions*; migrate its LLM calls to workflow-step tasks → husam. **Blocks `request()`/`await_inline` final deletion.** Separate effort, before SP5. | (no handoff yet) |
| 15 | **SP5 — delete the primitive** | Remove `await_inline`/`resolve_inline`/`_inline_waiters`/`INLINE_TIMEOUT`; delete the `request()` shim; delete dead `self_reflect`/`maybe_apply` (#8); fix conftest/tests; guard test that the primitive is gone. **Carve-outs that must migrate first:** #2 `task_classifier`, #6 `investor_bullets` (SP2-deferred), + shopping (#14). | umbrella spec, SP5 bullet |

---

## 🔵 Pre-existing reds (NOT CPS — track separately, never blamed on SP3b)

| # | Test | Cause |
|---|------|-------|
| 16 | `mr_roboto test_reversibility_registry::test_every_dispatcher_action_is_in_registry` | missing `publish_preview_pages` in registry; baseline-confirmed pre-existing |
| 17 | `tests/test_orchestrator_routing.py` ×9 | `ImportError` on `_parse_task_difficulty`/`_reorder_by_model_affinity` — deleted in the Task-13 orchestrator trim. Stale dead tests → delete or port. |
| 18 | `tests/core/test_dispatcher_in_flight.py` ×2 | stale `src.core.llm_dispatcher.kuleden_donen_var` mock patch target (tests `request()`); pre-existing |
| 19 | `general_beckman test_admission_cache.py` ×2 | "no such table: tasks" fixture flake — FAILED in the SP3b worktree, **PASSED clean on main** (169/0). Run-order/fixture-dependent; may resurface. |

---

## Status snapshot (what IS done)

- SP3b merged (`1183f521`); review CRITICAL (phantom lane) + the lost-update race + restart-strand + silent-reflection-bridge + post-hook budget all **fixed + verified** (beckman 169 / husam 38 / coulson 56 / orch 5, zero failures).
- Items 1-3 (founder validation) gate everything downstream — the substrate is correct but **unexercised**.
