# Spec — Rejection Ledger + Conversation Reset (kill the cross-dispatch prompt runaway)

**Date:** 2026-06-22
**Status:** design / spec (no code yet)
**Reviews:** root-cause + design validated by Opus subagent (YELLOW → gaps resolved); C3 safety gate validated SAFE by a second Opus trace.
**Coordination:** complementary to the parallel `est_in` clamp (`docs/handoff/2026-06-21-est-in-clamp-coordination-handoff.md`) — disjoint files, layered defense at ~64k. That clamp bounds *consumption* of a poisoned estimate; this bounds *generation* of the bloated prompt that poisons it. No collision.

---

## 1. Problem (proven root cause)

Analyst i2p steps intermittently produced ~170k–193k-token prompts → poisoned `step_token_stats.in_p90` (learned ~173k) → estimator forced ctx_needed ~226k → all models filtered → "No model candidates" DLQ. Recurred for weeks, survived restarts.

**Root (evidence-backed, all alternatives ruled out):** the bloat is **accumulation of the model's own assistant turns across many re-dispatches of the same task**, restored from checkpoint and never trimmed.

- Live `messages state` telemetry, task 459160: prompt = 775,651 chars = `assistant 113 msgs/709,922c (91%)` + `user 150 msgs/63,927c` + `system 1,802c`.
- `conversations` for 459160: 258 assistant turns, 170 unique; one `write_file(competitive_positioning.md)` content emitted **48× identical**, another 25× — regeneration without convergence across ~dozens of dispatches over 6 days.
- `model_call_tokens`: 87 calls, ALL `success=1` → availability failures produce no turn; the accumulation is purely produced-then-rejected **quality** attempts.

**Why nothing trims it:** `window.py:88` trim threshold = `ctx_window * 0.80`; for gemini (1M) that's 800k → 775k never trims. `prune_tool_results` (window.py:155-217) drops only assistant+tool-result *pairs* — standalone `final_answer` documents survive.

**Why it's not injection:** context-layer pool is ≤~1638 tok in the react path (`context.py:1334` hardcodes `model_ctx=4096`); tool observations capped 4000 (`config.py:68`); blackboard layer capped 3000 (`blackboard.py:176`). The prior `CONTEXT_ABS_CAP=32768` fix capped the wrong array (the context-layer pool, which never binds here), not the conversation array.

**Why it persists:** `react.py:275` restores full `messages` from checkpoint on every dispatch; `react.py:291` re-derives `reqs` fresh but NOT `messages`. Only `_schema_error` (`react.py:260`) currently triggers a fresh rebuild. **[CORRECTED per spec review F1]** grade/grounding/reviewer rejections DO set `_schema_error` (via the posthook verdict appliers, `apply.py:2929/3058/3191`) → they already skip-and-rebuild → they do NOT bloat. The actual bloat re-entry paths are the quality `Failed` paths that do NOT set `_schema_error`: (a) **degenerate rejection** (`hooks.py:1536-1543`, sets `error_category="quality"`, returns before any `_schema_error`/store write); (b) **"completed with empty result"** (`result_router.py:138`); (c) the dispatch **following an availability bounce**, where `_drop_stale_retry_feedback` (`apply.py:853-856`) pops the now-stale-stamped `_schema_error`, re-exposing the checkpoint restore. Tests must exercise THESE paths.

---

## 2. Constraints (product owner; all must hold)

- **C1** No hard truncation anywhere — drop whole units (turns/attempts), never byte-slice content.
- **C2** Availability failures produce no judged output → carry nothing on availability re-dispatch. Only quality failures matter.
- **C3** 75%-progress continuation — a near-complete attempt must be continued/edited, not restarted. **[RESOLVED SAFE — see §4.3]**
- **C4** Respect the dispatch boundary — never break an in-flight tool_call→observation chain or clobber legit crash-resume; only drop COMPLETED prior dispatches.
- **C5** Distinct semantic rejections (X rejected, Y rejected → must reach Z) need the *history* of (approach, why-rejected), compactly — not N full documents, not last-only.

---

## 3. Design — rejection ledger + whole-last-attempt + bounded reset

Replace blind full-conversation-restore with:

1. **Rejection ledger** — `ctx["_rejection_ledger"] = [{attempt, category, reason}, …]`, appended (not overwritten) per quality rejection. Rendered in the retry block as an explicit "tried X→rejected(A); Y→rejected(B); avoid these, take a different path." Satisfies C5 compactly.
2. **Whole last attempt for same-step continuation** — inject the FULL prior draft from the durable store/disk (not the truncated `_prev_output`). Satisfies C3 + C1. The durable artifact IS the un-truncated last attempt.
3. **Bounded conversation reset** — on a *completed-prior-dispatch* quality re-attempt, do NOT restore the prior `messages` array (generalize the `_schema_error` skip). The ledger + durable artifact now carry the signal the conversation used to.
4. **Repeat detector** — new output ≈ last (exact/near hash) → escalate, don't re-dispatch a 49th identical attempt.

**Scaffolding that already exists** (low net-new): failure `category` taxonomy (`retry.py:124` quality / `__init__.py:469` availability); per-attempt feedback store stamped to attempt (`hooks.py:1774-1794`); `_drop_stale_retry_feedback` already drops quality feedback on availability retries (`apply.py:837`); `_schema_error` checkpoint-skip pattern (`react.py:260`).

---

## 4. Gap resolutions (from the YELLOW review)

### 4.1 C1 — enumerate ALL truncation sites
The retry-render `_prev[:4000]` (`context.py:1059`) is NOT the only one. Also: `hooks.py:1784` `canonicalize_for_retry(...)[:6000]`; `apply.py` `_prev_output[:6000]` at ~12 sites (2906, 2932, 3035, 3061, 3168, 3194, 3323, 3345, 3617, 3639, 3823, 3845).
**Resolution:** the same-step continuation no longer sources from these truncated values — it reads the FULL artifact from the store/disk (§4.3). `_prev_output` is demoted to a *fallback only* when no durable artifact exists (pure-conversation output). Each `[:6000]`/`[:4000]` site is then either (a) removed in favor of the durable read-back, or (b) explicitly retained ONLY for the no-durable-artifact fallback with a comment. Spec deliverable: a site-by-site table in the implementation plan.

### 4.2 C2 — availability carries nothing
Already built at the *feedback* level (`_drop_stale_retry_feedback`). NOT yet at the *messages* level — that is exactly Phase 2. The ledger append is gated to `category == "quality"`; availability re-dispatch appends no entry and (Phase 2) does not restore messages → fresh + durable artifacts.

### 4.3 C3 — SAFE (validated by trace)
Produced output is persisted to the durable store (`hooks.py:1564` → `artifacts.py:39-69`) and disk (`hooks.py:350-353`/`1601`) **before** the schema gate (`hooks.py:1710`); grade/grounding (apply.py) run strictly *after* the hook (`__init__.py:1298` before `:1306`). Both output modes (write_file + final_answer-as-artifact) are durable. Next-dispatch read-back via `fetch_deps`→`input_artifacts` (`context.py:578`), `workspace_snapshot` (`context.py:720`), and on-disk `produces`. **Conversation reset cannot lose the 75%.**
**Refinement:** for a SAME-step re-dispatch, the producer does not auto-read its own prior draft via `input_artifacts` (those are upstream-only). Spec adds: inject the step's own prior `produces` artifact (full, from store/disk) into the same-step retry prompt — replacing the 6k `_prev_output` as the continuation carrier.

### 4.4 C4 — explicit dispatch-boundary marker
`task_state.messages` serves both crash-resume and cross-dispatch bloat; `save_checkpoint` fires at ~10 mid-loop points (`react.py:986,1166,1197,1268,1477,1684,1747,1790,1809`). `_schema_error` alone is an insufficient discriminator (a crash mid-chain with no rejection would be wrongly reset).
**Resolution:** write an explicit `dispatch_complete: bool` (or `dispatch_id`) into the checkpoint. Reset rule:
- `dispatch_complete == False` (interrupted in-flight) → RESUME (restore messages) regardless of category — preserves the tool_call→observation chain (C4).
- `dispatch_complete == True` + quality re-attempt → fresh rebuild (drop messages); ledger + durable artifact carry forward.
- availability re-attempt → fresh (no judged output).
Set `dispatch_complete=True` when the loop exits via final_answer / terminal failure; leave False on every mid-loop checkpoint.

### 4.5 C5 — confirm every quality path emits a ledger reason
schema/grade/grounding/reviewer already write a `feedback` string into `_schema_error` (`apply.py:2903/2929/3032/3058/3165`). **Degenerate (dogru_mu_samet) path salvages in `react.py:310-318` and may NOT emit a ledger reason** — the case closest to the 48× symptom. Spec deliverable: verify; if absent, add a `(category="quality", reason="degenerate: <signature>")` ledger write on the degenerate path. Repeat-detector keys off exact/near hash + ledger-length+category-repetition (semantic near-dupes won't hash-match — acknowledged Phase-3 limit).

---

## 5. Phasing (risk-ordered)

- **Phase 1 (additive, low-risk):** ledger append + render; same-step full-draft read-back from durable store; demote `_prev_output` to fallback. Does NOT touch conversation-restore. Independently shippable; resolves C5 + C1 + C3-continuation. TDD.
- **Phase 2 (the bloat kill):** dispatch-boundary marker + bounded conversation reset on completed-prior quality re-attempts. Safe because Phase 1's ledger+durable-artifact replace what the restore provided, and C3 proved durability. TDD; sim re-run.
- **Phase 3:** repeat detector + escalation (don't re-dispatch a 49th identical).

## 6. Test plan (TDD, per phase)
- Ledger: N quality rejections → ledger has N stamped entries; availability rejection appends none; render shows all prior reasons.
- Continuation: same-step retry prompt contains the FULL prior draft from store (not 6k-truncated); no byte-slice marker.
- Reset: completed-prior quality re-attempt → messages NOT restored (assert prompt size bounded); interrupted in-flight (dispatch_complete=False) → messages RESTORED (crash-resume preserved).
- Repeat: identical output to last ledger entry → escalation path, not re-dispatch.
- Regression: existing schema-retry tests (test_prompt_noise_reduction, recency_reorder), `tests/sim/run_scenarios.py`.

## 7. Risks
- **Phase 2 crash-resume** — mitigated by the explicit `dispatch_complete` flag (§4.4); do NOT rely on `_schema_error` alone.
- **Degenerate reason gap** (§4.5) — verify before Phase 1 completes.
- **Coordination** — keep any conversation bound < 64k so the parallel `est_in` clamp and the rollup filter nest cleanly.

## Key anchors
react.py:204,260,275,291,298,986+(saves),310-318(degenerate); window.py:88,155-217; context.py:578,720,1059,1334; config.py:68; hooks.py:350,1564,1601,1710,1774-1794; apply.py:502,837,2871-2942,4919-5087,5617; retry.py:118-149; artifacts.py:39-93; __init__.py:469,1298,1306.
