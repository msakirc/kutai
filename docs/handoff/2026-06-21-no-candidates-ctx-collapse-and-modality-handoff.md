# Handoff — "No eligible candidates" root = ctx-gate pool collapse; modality fix shipped

**Date:** 2026-06-21
**Status:** modality fix MERGED main (`92b9f6c2` + `dc9cf586`, NOT pushed, restart-gated). Two residuals specced below for owner decision.
**Supersedes the framing of:** `2026-06-21-classA-reclassification-handoff.md` (Class A was correctly *not* a schema-gate bug — but "availability-storm residue" undersold it; the real upstream is the ctx-gate collapse traced here).

## TL;DR
Last night's mission-87 failures and the recurring **"All models failed: No model candidates available"** share ONE upstream root: the selector's **context-window eligibility gate filtered on a poisoned learned stat**, collapsing the candidate pool to gemini-only (and to empty when gemini was also out). It was NOT cloud depletion (cerebras served 88 picks the same window). Two independent empty-pool paths exist. The audio-model-wins-text-step symptom is a *separate* modality bug, now fixed.

## Evidence (live DB + logs, 2026-06-20 night)
- Pick volume 23h–00h: **168 picks/32 models, then 104/20** — heavy thrash, not exhaustion.
- Provider picks 23h–01h: **cerebras 88, openrouter 98, local 64, groq 16, gemini 6** → cloud NOT depleted.
- `writer` agent overnight: gemini 1, groq 5, local 9, openrouter 60, **cerebras 0**.
- Selector debug (smoking gun):
  ```
  model filtered: name=cerebras/gpt-oss-120b reason=ctx(131072<222931) task=analyst
  model filtered: name=cerebras/zai-glm-4.7  reason=ctx(128000<222931) task=analyst
  → selector eligibility: task=analyst candidates=8 providers=[gemini=8] fully_filtered=[groq,cerebras,openrouter]
  ```
- "no eligible candidates" count by day: **68 on 06-19, 7 on 06-20**, e.g.:
  ```
  task=analyst local_only=True filtered=78
  reasons=[55×local_only, 13×load_mode_minimal, 6×vision_variant_not_needed, 3×demoted, 1×coding_specialty_mismatch]
  ```

## Root chain
1. **needed_ctx inflated to ~223k.** `selector.py:593` filters `model.context_length < reqs.effective_context_needed`. `effective_context_needed = (estimated_input_tokens + estimated_output_tokens) × 1.3 + 512` (requirements.py:139); `estimated_input_tokens` = B-table `step_token_stats.in_p90` (requirements_builder.py:213). A runaway/poisoned p90 (~173k) → needed_ctx 222,931.
2. **Pool collapses.** Every model <223k ctx filtered: all cerebras (128–131k), groq, openrouter-free, local. Only gemini (1M) survives → 8 gemini-only candidates.
3. **Empty pool** when gemini is also unavailable (free-tier 20/day, or path 2 below) → `select()` returns None → "No model candidates available."

### Two independent empty-pool paths
- **Path A — ctx-gate collapse** (above). Driven by btable p90 inflation. Ties to `project_context_budget_cap_20260618`, `project_conversation_runaway_root_20260621`.
- **Path B — `local_only=True` ∩ `load_mode_minimal`.** Frozen stale `local_only=True` (restored from checkpoint — `project_checkpoint_freezes_reqs_20260620`) vetoes all cloud; `load_mode_minimal` vetoes all local → 55+13 filtered → 0.

## What is ALREADY healed (verify-don't-rechase)
- **Live `step_token_stats` analyst `in_p90` now = 61,863 max** (was ~173k). The `prompt_tokens≤64k` rollup filter from `d19f0051` is live → p90 settled at the 64k boundary → needed_ctx ≈ 85k → cerebras/openrouter/gemini eligible again → **pool no longer collapses to gemini-only.** Path A is effectively dead *right now*.
- Today's commits also landed: `09459c59` reviewer-routing ([1.13]/524380), `2e23f861` load-ctx sizing, deps/blackboard truncation (`16b4bb7e`/`6ccfdb11`).

## Residual 1 — modality blindness (FIXED this session)
`google/lyria-3-clip-preview` (audio) registered as a TEXT candidate and won a writer pick → empty output → surfaced as schema "missing sections". Root: `_infer_modality` id-pattern fallback knew only `{embedding,image,tts,audio,video}`; "lyria" matched none → "text" → `register_cloud_from_discovered`'s non-text gate let it through.
- **Fix `92b9f6c2`**: widened fallback vocabulary (lyria/music/speech/voice/imagen/diffusion/sdxl/flux/dall-e/veo/sora) in a shared `_modality_from_id`; gemini delegates to it (no drift).
- **Fix `dc9cf586`** (review follow-up): match tokens at **delimiter boundaries**, not bare substrings, so "flux"⊄"reflux" etc. can't false-drop text models.
- No DB surgery needed: in-memory catalog is rebuilt from discovery each cycle; after restart lyria hits the fixed gate. Stale `models` rows are registry_store dead-set bookkeeping, not the catalog.
- TDD, 68 + 146 green. **NOT pushed (restart-gated).**

## Residual 2 — no hard floor on `effective_context_needed` (PROPOSED, owner decision)
Path A is healed *only indirectly* — by the btable rollup filter holding p90 ≤64k. There is **no defense at the selection layer**: if the rollup filter ever regresses, or a single task legitimately needs >128k input, `effective_context_needed` again exceeds every non-gemini ctx window and the pool re-collapses to gemini-only → empty when gemini is out.

This is **selection policy** (overlaps `project_phantom_veto_architecture_20260617` "veto-no-fallback" and `project_context_budget_cap`), so flagged for owner rather than unilaterally changed. Two design options:

- **Option A (clamp the requirement):** cap `effective_context_needed` at a sane ceiling (e.g. `min(value, KUTAI_MAX_SELECT_CTX≈131072)`) in `requirements.py`. Simple; risk = a task that *truly* needs >131k input is under-provisioned (but the loader/window trimmer already bound the assembled prompt at `CONTEXT_ABS_CAP=32768`, so the requirement is already an over-estimate of what gets sent).
- **Option B (least-context fallback in selector):** when the ctx gate filters the pool to empty, relax to the largest-context available model instead of returning None — the "least-bad" principle from the phantom-veto spec. More robust, more invasive.

**Recommendation:** Option A first (cheap, testable, directly bounds the gate), with Option B as the durable backstop for the *generic* empty-pool case (also covers Path B and phantom-veto). Both want sim re-runs (`tests/sim/run_scenarios.py` + `run_swap_storm_check.py`) before merge.

## Residual 3 — Path B (`local_only` ∩ `load_mode_minimal` deadlock)
Not addressed today. Real fix = re-derive `reqs` each run instead of restoring frozen `local_only=True` from checkpoint (`project_checkpoint_freezes_reqs_20260620`). Band-aid = clear `task_state` for the immortal poisoned analyst tasks (459160/459220).

## Deploy
Restart-gated: modality fix is code-only and the live process still has lyria in-memory until restart. After restart + re-discovery, lyria is excluded. Push `92b9f6c2` + `dc9cf586` after verify.
