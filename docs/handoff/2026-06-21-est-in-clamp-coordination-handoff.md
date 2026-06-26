# Coordination handoff — proposed `est_in` clamp in `estimate_for` (P2 / no-candidates ctx-collapse)

**Date:** 2026-06-21 (updated 2026-06-22)
**Status:** **IMPLEMENTED — committed `8f479bff` (local main, NOT pushed, restart-gated).** Heads-up so a parallel session in the same area can confirm no collision. If you're touching `estimate_for`, `requirements_builder`, the selector ctx/TPM gates, or btable estimation, check the conflict surface below.

> **Refinement vs the original proposal below:** only the LEARNED `in_tokens` is clamped. Output is NOT clamped (the `4.5b` openapi_spec override legitimately emits >100k, and output can't run away — bounded by model max generation). So there is ONE env var, `KUTAI_MAX_EST_IN_TOKENS` (default 64000); the `_MAX_EST_OUT` from the original sketch was dropped.

## The issue (one line)
`estimate_for().in_tokens` is **unbounded**; a poisoned/runaway B-table `in_p90` inflates the selection estimate and silently collapses the candidate pool → "No model candidates available".

## Mechanism (verified from logs + code, 2026-06-20 night)
- `selector.py:593` ctx gate: `model.context_length < effective_context_needed`. With `in_p90≈173k` → `effective_context_needed=222,931` → every model <223k filtered → only gemini (1M) survives → empty when gemini also out.
- Same unbounded `est_in` also trips `selector.py:643` (per-request cap) and `selector.py:662` (per-call TPM) → three gates over-filter at once.
- NOT depletion (cerebras served 88 picks the same window). Two empty-pool paths: **A** = this ctx-collapse; **B** = `local_only=True ∩ load_mode_minimal` (separate, reqs re-derivation).

## Current state (already partly healed)
- Live `step_token_stats` analyst `in_p90` now **61,863** (was ~173k) — the `d19f0051` `prompt_tokens≤64k` rollup filter drained it. Path A is dormant *right now*, but only because the stored data happens to be clean. No bound at the consumption point.

## Proposed fix — SINGLE site
Clamp the B-table-derived estimate **inside `estimate_for`** (`packages/fatih_hoca/src/fatih_hoca/estimates.py`, Level-1 return ~line 64-71), env-tunable:
```python
_MAX_EST_IN  = int(os.environ.get("KUTAI_MAX_EST_IN_TOKENS", "64000"))
_MAX_EST_OUT = int(os.environ.get("KUTAI_MAX_EST_OUT_TOKENS", "16000"))
# clamp in_tokens/out_tokens before returning Estimates(...)
```

### Why `estimate_for` and nowhere else
ALL consumers route through it — worker selection (`requirements_builder.py:213`), admission gating (`general_beckman/__init__.py:89` `_estimate_task_tokens`), ranking (`ranking.py:235`), queue profile (`queue_profile_push.py:133`). Clamping here keeps **admission and worker aligned** (requirements_builder.py:193-195 unified them on purpose to kill the admit-then-reject DLQ loop — clamping in only one of them re-splits them). Do NOT clamp at a single consumer.

### Why it's safe (not under-provisioning)
- Assembled prompt is capped at `CONTEXT_ABS_CAP=32768`; no legitimate task assembles >64k. >64k prompts are the runaway bug (`project_conversation_runaway_root`, open).
- Loading is independently correct: dispatcher `_resolve_load_ctx` (`2e23f861`) re-maxes load ctx against the **live** prompt.
- `est_in≤64k` ⇒ `effective_context_needed≈104k < cerebras 128k` ⇒ ctx gate can't collapse the pool regardless of btable hygiene (defense-in-depth atop the rollup filter).

## Conflict surface — files I'd touch
- `packages/fatih_hoca/src/fatih_hoca/estimates.py` (the clamp + env reads)
- tests: `packages/fatih_hoca/tests/test_estimates.py` (+ maybe `test_requirements_builder_estimate.py`)
- sim re-run only (no edit): `tests/sim/run_scenarios.py`, `run_swap_storm_check.py`

## The ONE open decision
The ceiling value (default 64000). Everything else is mechanical. If your parallel work changes how `in_p90` is produced/stored, or adds its own bound, we must pick ONE bound — say which and I'll defer or adapt.

## Relationship to phantom-veto
Complementary: this prevents over-*filtering* at the source; phantom-veto's Option B (least-bad fallback on empty pool) would catch any *residual* empty pool. Not conflicting. This does NOT touch S4/S5 fleet-denominator code.

— Full root analysis: `docs/handoff/2026-06-21-no-candidates-ctx-collapse-and-modality-handoff.md`
