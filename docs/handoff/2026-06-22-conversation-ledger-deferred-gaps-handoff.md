# Handoff — Conversation-Ledger Deferred Gaps (GAP-1 sweep, GAP-2, T8)

**Date:** 2026-06-22
**Context:** The conversation-runaway fix (rejection ledger + bounded reset + repeat-detector) is **implemented + pushed to `origin/main` (`f60d04ae`)**. Spec `docs/superpowers/specs/2026-06-22-conversation-ledger-design.md`, plan `docs/superpowers/plans/2026-06-22-conversation-ledger-plan.md`, root memory `project_conversation_runaway_root_20260621`. Two non-blocking gaps + one live-verification step were deferred (final review GREEN with these as known follow-ups). **Do these NOT-AFK** — GAP-2 touches ~16 hot-path call sites.

> The **core bloat-kill is done and safe** (`should_restore_messages` in react.py, gated on `worker_attempts`). These gaps only complete the *enhancement* coverage (ledger visibility + faster repeat termination). The repeat loop already self-bounds at `max_worker_attempts=15`, so nothing here is a safety fix.

---

## GAP-1 (sweep) — ledger populated on only 4 of ~16 quality re-pend appliers

**Where:** `packages/general_beckman/src/general_beckman/apply.py`. Every quality re-pend funnels through `_stamp_retry_feedback(ctx, attempts)` (~16 call sites). Only **4** have a preceding `_ledger_reject`:
- grounding (2960), verify_artifacts (3091), code_review (3226), grade (5062 — added in `f60d04ae`).

**Missing** `_ledger_reject` (these re-pend with no ledger entry → the "Prior attempts (do not repeat)" render in `context.py` stays empty for them): the `_stamp_retry_feedback` sites at **3004, 3135, 3270, 3399/3421, 3693/3715, 3899/3921, 4055/4077, 4489, 4641/4657, 4761/4779, 5149**. These cover mechanical-check verdicts (imports_check, test_run, pattern_lint, domain_layer_check, openapi_sync, typescript_sync, design_system_check, migration_apply, mobile_smoke, compliance_*, prior_art_min_coverage, verify_falsification_present, adr_drift_check) and review verdicts (integration/security/accessibility/contract/performance/visual_review, verify_review_verdict, launch_readiness_gate, incident_update_review, outreach_deliverability_check, documentation_gap_detect, copy_compliance_review, brand_voice_lint).

**Recommended fix (the DRY way the original review wanted): relocate the append INTO `_stamp_retry_feedback`** (apply.py:885) so all ~16 sites are covered by one change, and **remove the 4 scattered `_ledger_reject` calls** (2960/3091/3226/5062) to avoid double-append.
- Reason source: pass an explicit `reason` kwarg per caller (precise attribution like "grade: …"/"security_review: …") — preferred. Acceptable fallback: read `ctx.get("_schema_error")` inside the chokepoint (all quality re-pends set it before calling; less precise prefix). Cap stays 500 (handled by `append_rejection`).
- **No-double guarantee:** the hooks-layer paths (schema `hooks.py:1609`, degenerate `hooks.py:1866`) append via `append_rejection` directly and do NOT route through `_stamp_retry_feedback` — so the chokepoint relocation won't double-count them. Verify this still holds.
- Place the append on the re-pend path only (the chokepoint is already re-pend-only; terminal-DLQ calls `_dlq_write`, not `_stamp_retry_feedback` — a dead task needs no ledger).

---

## GAP-2 — repeat-detector doesn't cover the grade/verdict-applier paths + latent self-match

**Current state:** the T7 detector lives in `_retry_or_dlq` (apply.py ~533). But the verdict appliers (grade/grounding/verify/review/…) re-pend DIRECTLY (via `update_task(status="pending")`), **never through `_retry_or_dlq`** — so T7 never runs on the 48× grade loop. It only fires on empty-result (hash None → inert) and schema-fail (where `ledger[-1]` is the CURRENT attempt → latent false-positive self-match). T7 "fails safe" (premature DLQ is founder-visible, durable artifact preserved), so it's not harmful, just ineffective on the real path.

**Recommended fix (combine with GAP-1's chokepoint relocation):** make `_stamp_retry_feedback` return an `escalate: bool` and compute it there.
- Signature: `_stamp_retry_feedback(ctx, next_attempt, *, reason=None, prev_output=None) -> bool`.
- Logic: compute `cur = _output_hash(prev_output)`; compare to `ledger[-1].out_hash` **BEFORE appending** the new entry (never self-match); `escalate = bool(cur) and bool(prior) and cur == prior`. Then append the new entry. Return `escalate`.
- **Caller wiring (the invasive part — ~16 sites):** each re-pend branch must honor the return: `if escalate: await _dlq_write(source, error="degenerate repeat: identical output, not converging", category="quality", attempts=attempts); return` — INSTEAD of the `update_task(status="pending", …)`. This short-circuits the re-pend; without honoring the return, the signal does nothing.
- Self-match rule (test it): compare-then-append; first attempt and prior-`None`-hash never escalate; JSON round-trip safe.
- **Hash-parity caveat (carry forward):** the appliers hash `source.get("result")` (raw); the degenerate hooks path hashes `_unwrap_envelope(result["result"])`. If the ledger mixes both, a real repeat via different paths may not hash-match → under-detects (never false-DLQs). Keep one path's hashing consistent within a loop, or normalize (unwrap+canonicalize) at the hash site. Document whichever you pick.
- **Limit:** exact-hash only; semantic near-dupes won't match (acceptable — the symptom was byte-identical).

**Reference design:** a prior subagent wrote a clean test for exactly this (kwargs + `escalate` return) — file was removed (it tested unimplemented code), but the shape is: `_stamp_retry_feedback(ctx, n, reason=…, prev_output=…) -> bool`, tests `test_chokepoint_*` asserting (a) grade reason lands in ledger, (b) two identical drafts → escalate True, (c) different → False, (d) first attempt / None-prior → False, (e) never self-matches, (f) JSON round-trip. Recreate those tests (pure, no DB).

---

## T8 — live verification (post-restart, after USER pull+restart)

The fix is **restart-gated**: the live system runs from the local tree / pip-installed packages, NOT origin. After the user pulls + restarts:
1. Re-dispatch (or wait for) a previously-bloated analyst step. Query the existing `messages state` telemetry (`react.py:464`, logged per iteration) for that task across its dispatches.
2. **Assert the `assistant` char total stays FLAT across re-dispatches** (baseline: task 459160 was `assistant=113msgs/709,922c`). A correct fix shows each fresh dispatch starting small, not accumulating.
3. Confirm `step_token_stats.in_p90` for the step does not re-climb past ~32k over the next 14-day rollup window.

---

## Test discipline (every task)
- TDD: failing test first. Pure/mocked — NEVER touch the prod DB (`C:\Users\sakir\ai\kutai\kutai.db`); tmp_path sqlite if unavoidable.
- Pytest FOREGROUND only: `timeout 180 python -m pytest <file> -o addopts="" -p no:aiohttp -q`. NEVER background pytest (SQLite-orphan lock crashes the live system). NEVER kill/Stop-Process python.exe (live wrapper shares the interpreter).
- Regression set: `tests/test_rejection_ledger.py test_ledger_render.py test_degenerate_repeat.py test_dispatch_boundary.py test_grade_ledger_coverage.py test_prior_draft_readback.py`. Known pre-existing reds (NOT yours): `test_skills_called_on_low_retry`; the ~9 retry-policy tests in `test_retry*.py` (assert old quality→delayed policy; touch none of these files — verify by `git stash`).

## Risk / ordering
- GAP-1 chokepoint relocation and GAP-2 escalation are best done TOGETHER (same function + same ~16 call sites). One commit pair: (1) chokepoint append + remove 4 scattered (GAP-1), (2) escalate return + caller wiring (GAP-2).
- Keep prompts < 64k to nest under the parallel `est_in` clamp + rollup filter.
- This is on shared `main` — forward commits only, no history rewrite.
