# Modularization Finish Plan
*2026-05-31. Companion to `docs/2026-05-31-root-debt-map.md`. Goal: `src/core/` + `src/models/` hold ONLY thin orchestration + genuine shims. All real logic → its owning package. Each phase is independently shippable; ordered low-risk → high so you can stop at any boundary.*

## End-state definition (what "done" means)
- `src/core/` = pump (orchestrator) + load/call loop (dispatcher) + plumbing (decisions, heartbeat, state_machine, in_flight, startup_recovery, exceptions). No LLM-prompt logic, no scoring, no validation libs.
- `src/models/` = only thin re-export shims (`<60 LOC`, mostly `from pkg import *`).
- No dead code paths. No symbols the docs reference but that don't exist.
- A **guardrail test** keeps it that way (Phase 7) — else it regrows (orchestrator already went 366→793).

## Dependency graph (do in this order)
P1 (dead delete) → P2 (relocations) → P3 (LLM-children) → P4 (dispatcher) → P5 (orchestrator) → P6 (LMM split) → P7 (docs+guardrail).
P1–P3 are independent of each other; P4/P5/P6 touch hot paths so land them after the cheap wins build confidence.

---

## Phase 1 — Delete dead code (LOWEST risk, prod-safe)
**Why first:** zero prod callers, pure subtraction, can't break runtime.
1. Delete `router.select_model()` (L131–687), `select_for_task()` (L692–694), `check_cost_budget()` (L700–705).
2. Relocate the 2 LIVE symbols out of the fossil:
   - `ModelCallFailed` (L16–28) → new `src/core/exceptions.py`. Re-export from router for back-compat.
   - `get_kdv()` (L48–128) → `src/infra/rate_limiter.py`. Re-export from router.
3. Port the 4 dead-path tests to the live ranker or delete: `test_scoring_layers`, `test_sibling_rebalancing`, `test_real_pipeline`, `test_gap_fixes` → target `fatih_hoca/ranking.py::rank_candidates`.
4. Delete `grading.apply_grade_result()` (L326) if verified callerless.
- **Verify:** `python -c "from src.core.router import get_kdv, ModelCallFailed"`; `timeout 120 pytest tests/ -q`; boot smoke.
- **Done:** router.py ≤ ~140 LOC, only KDV factory + exception (or moved + thin re-exports).

## Phase 2 — Relocate misplaced packages (pure move, no logic change)
**Why:** files are in the wrong dir but already self-contained. Move + leave a re-export shim so callers don't break, then migrate callers, then drop shim.
| Move | From → To | Caller(s) to update |
|------|-----------|---------------------|
| `fast_resolver.py` | `src/core/` → `src/tools/` | `src/tools/smart_search.py:47` |
| `models.py` (action validation) | `src/models/` → `packages/coulson/` ✅ (lives with only caller) | `coulson/react.py` |
| `auto_tuner.py` | `src/models/` → `packages/fatih_hoca/` ✅ (all model knowledge in one pkg) | `telegram_bot.py`, fatih_hoca metrics |

**`gpu_scheduler.py` — DO NOT relocate. Investigate dissolving it (founder: "ideally none").**
- Hypothesis: DaLLaMa already arbitrates GPU/inference slots via swap orchestration → gpu_scheduler may be a duplicate contention layer. Only caller is `local_model_manager.py` (`acquire_inference_slot`/`release_inference_slot`).
- Sub-task 2a (spike before any move): map DaLLaMa's existing slot/swap locking vs gpu_scheduler's priority queue. If DaLLaMa covers single-GPU mutual exclusion, fold gpu_scheduler's priority/timeout semantics into DaLLaMa and delete the module. If there's a genuine gap (priority ordering DaLLaMa lacks), the queue lives in DaLLaMa, not `src/models/`.
- Output: a kill-or-fold decision in `docs/`, then execute. Net target = `src/models/gpu_scheduler.py` ceases to exist.
- Each move = `git mv` + fix imports + thin re-export at old path + run tests. One commit per file.
- **Verify per move:** import check + `timeout 60 pytest` on the touched package + caller smoke.
- **Done:** `src/models/` contains only genuine shims (`model_registry`, `capabilities`, `quota_planner`, `model_profiles`, `gpu_monitor`, `rate_limiter`).

## Phase 3 — LLM-child builders → coulson
**Why:** architecture rule = LLM-prompt logic lives in packages, not core. All 3 are live via `general_beckman/apply.py`, build `raw_dispatch` specs, ≤2 callers, no cycles.
1. Create `packages/coulson/src/coulson/posthooks/` → move `grading.py`, `code_review.py`, `reflection_posthook.py`.
2. Update `apply.py` (L1645/1660/1696/1748) + `posthook_continuations.py` (L146/218) imports.
3. Thin re-export at old `src/core/` paths during migration; drop once green.
- **Verify:** `timeout 120 pytest packages/general_beckman packages/coulson -q`; run one mission with a grade + code_review + reflection step.
- **Done:** core has no prompt-builder modules; `reflection.py` REFLECTION_BLOCKS stay in coulson (already there).

## Phase 4 — De-accrete the dispatcher (721 → ~430)
**Why:** dispatcher should be load→call only. ~231 LOC of non-loop logic to evict.
1. `_record_pick()` (58) telemetry → `src/telemetry/pick_recorder.py`.
2. `_get_loaded_*` / `is_loaded_model_thinking` / `get_stats` (~64) introspection → `src/models/introspection.py`.
3. `_estimate_prompt_tokens()` (23) → DaLLaMa or HaLLederiz Kadir (KV sizing belongs with the loader).
4. `request()` shopping shim (86) → **delete after SP5 shopping migration** (gated on SP5, not now).
- **Verify:** dispatch path mission smoke; pick_log still written; `timeout 120 pytest`.
- **Done:** dispatcher = `execute` + `_ensure_local_model` + `_prepare_messages` + singleton. Update CLAUDE.md "thin" claim back to true.

## Phase 5 — Thin the orchestrator (793 → ~400)
**Why:** pump re-grew +427 LOC of feature logic. Extract by responsibility.
1. Periodic checks (`_check_yalayut_discovery`, `_check_source_scout`, MCP idle sweep) → `src/core/periodic_checks.py`; pump just calls a `run_due()`.
2. Workflow agent-type refresh (_dispatch L179–238, 60) → `src/workflows/engine/task_refresh.py`.
3. Self-reflection bridge (L240–277, 35) → coulson or `general_beckman` posthook wiring.
4. Founder sweep → call existing `founder_actions.sweep_*` (logic already there, inline the call).
5. Heartbeat loop → reuse `yasar_usta` HeartbeatWriter where possible.
- **Verify:** full boot + 1 multi-step mission + 1 workflow with live JSON edit (the bug the refresh guards); `timeout 120 pytest`.
- **Done:** `_dispatch` back near ~200 LOC; `run_loop` is a pump.

## Phase 6 — Split `local_model_manager` (684)
**Why:** mixed concern — thin DaLLaMa wrapper + 350 LOC orchestration.
1. Push `_need_ctx()` context-sizing → DaLLaMa (it owns load config). NOTE: this file is currently modified in your working tree (need-ctx redesign 2026-05-31) — land/settle that first.
2. Push DB scheduling + Beckman notifications inside `ensure_model()` → task layer (orchestrator/beckman callback).
3. Keep: nerd_herd push, GPU-scheduler bridge, lifecycle props → stays as the DaLLaMa-facing wrapper.
- **Verify:** model swap + load + idle-unload cycle; VRAM smoke; `timeout 120 pytest`.
- **Done:** LMM ≤ ~300 LOC of pure wrapper/bridge.

## Phase 7 — Lock it in (docs + guardrail)
1. Rewrite `docs/architecture-modularization.md` to match reality; add the now-documented LLM-child + telemetry modules.
2. **Guardrail test** `tests/test_root_stays_thin.py`:
   - assert `src/models/*.py` (shim set) each `< 80 LOC`;
   - assert `src/core/` modules do NOT import `hallederiz_kadir` / `litellm` directly (LLM execution stays in packages);
   - optional LOC ceilings on orchestrator/dispatcher with a comment "raise only with a refactor PR".
3. CI/pre-commit hook runs it → re-bloat fails loudly. This is what stops 366→793 happening again.

---

## Decisions (RESOLVED 2026-05-31)
1. **`gpu_scheduler.py`:** ❌ no relocation — **dissolve into DaLLaMa or delete** (founder: "ideally none"). Spike first (sub-task 2a above).
2. **`auto_tuner.py`:** → `packages/fatih_hoca/`. ✅
3. **`models.py`:** → `packages/coulson/`. ✅

## Effort sketch (rough)
P1 ~½d · P2 ~1d · P3 ~½d · P4 ~1d · P5 ~1–2d · P6 ~1d (after need-ctx settles) · P7 ~½d. ≈ 6–7 focused days, but P1–P3 (~2d) buy most of the truth-in-architecture back.
