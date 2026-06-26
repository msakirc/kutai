# Handoff — grader false-DLQ root cause was PROMPT TRUNCATION, not grader hallucination (FIXED)

**Date:** 2026-06-04
**Supersedes:** `docs/handoff/2026-06-04-grader-preempts-deterministic-check-handoff.md` — that handoff's
root-cause diagnosis ("a stochastic LLM grader can't count sections; defer structural axes to the
deterministic check") is **WRONG**. Do not implement its fix (A)/(B). It would have masked the real bug.

---

## 1. Real root cause (proven against live data)

`src/core/grading.py::build_grading_spec` fed the grader a **truncated instruction**:
```python
description=str(source.get("description", ""))[:500],   # ← the bug
```
Step 0.1 `product_charter`'s instruction is **1329 chars** and ends:
```
... EXACTLY these five `## ` sections in order:
1. Product Positioning ...
2. Brand Keywords ...
3. Core Problem / JTBD ...        ← char 500 cuts HERE
4. Goals & Mission ...            ← SEVERED from the grader's copy
5. Solutions We Own ...           ← SEVERED from the grader's copy
```
The grader was told "EXACTLY five sections," saw a list of **three**, then read an artifact containing
**Goals & Mission** + **Solutions We Own** — and correctly concluded *"extra sections (Goals & Mission,
Solutions We Own) not requested"* / *"added a sixth section."* **Those are the exact two sections the
`[:500]` cut off.** The grader was reasoning correctly from a mutilated prompt — it never hallucinated.

The 6 contradictory grade verdicts (handoff §3) are just different small overhead models each guessing
differently about the same impossible "5 promised, 3 listed, 2 unlisted in the artifact" contradiction.

### Why it surfaced "in the last few days"
The `[:500]` cap is old (relocated `5689cbfa`, 2026-05-29). The i2p instructions **grew** in the last
few days (verbose 5-section charter spec + the late-May/early-June step edits). **94 of 259 i2p steps
(36%) now exceed 500 chars** — every one was being silently severed in the grader prompt. That is the
regression: long instructions outran a fixed truncation. Blast radius = the handoff's list (ADRs,
interview, screen plans, …) for the right reason.

(The `�` in the DB description was just Windows-console rendering of U+2014 em-dashes; the DB and the
live JSON both store correct em-dashes. No encoding corruption. Red herring.)

---

## 2. Fix shipped (user rule: NO TRUNCATION of any LLM input/output, EVER)

Silent truncation of an LLM's view of the contract or the artifact guarantees false verdicts. The right
handling of an oversized prompt is model-selection/capacity (pick a bigger context window), never lopping.
Removed every truncation in the "judge/produce an artifact against its spec" path and made each estimate
derive from the ACTUAL prompt size so selection fits the context window (else the call-level cap becomes
the new silent-truncation point):

| File | Was | Now |
|------|-----|-----|
| `src/core/grading.py::build_grading_spec` | `title[:100] desc[:500] result[:30000]`, `est_input=800` | full; `est_input = max(800, chars//4)` |
| `src/core/code_review.py::build_code_review_spec` | `title[:100] desc[:500] result[:30000]`, `est_input=1500` | full; `est_input = max(1500, chars//4)` |
| `src/core/reflection_posthook.py::build_reflect_messages` | `desc[:500] result[:3000]` | full; caller (`apply.py`) derives `est_input` from msg len |
| `src/core/reflection_posthook.py::build_emit_messages` | `draft[:_EMIT_DRAFT_CAP=30000]` | full draft; `_EMIT_DRAFT_CAP` deleted; emit `est_output` tracks `len(draft)` not `len(draft[:30000])` |

Tests: `tests/core/test_build_grading_spec.py` (+2 new — full desc tail + full result/title + estimate
scales). Green: 4/4 grading-spec, 31 core (code_review/emit/reflect), 10 posthook_llm_child.
Pre-existing unrelated reds: `tests/test_grading.py::TestApplyGradeResultPass` ×2 (skill-extraction mock,
fails on clean HEAD too).

**Not live until KutAI restart** (via Telegram).

---

## 3. Follow-ups (separate class — context-window BUDGETING, not the contract bug)

These also truncate LLM-bound content but are a different problem (bounding unbounded history across many
prior steps); removing their caps blindly would blow context. They need a real budgeting strategy
(bigger-context model, chunking, or relevance selection), not a blind cap-removal — surfaced here so they
aren't forgotten:
- `src/core/context_injection.py:77,124` — prior-step results injected downstream (`[:500]`, `[:1500]`).
- `packages/coulson/src/coulson/context.py:1017` — retry feedback `_prev[:4000]`.
- `packages/coulson/src/coulson/react.py:766,1416` — tool outputs `[:2000]` fed back into the ReAct loop.
- `src/memory/rag.py` — retrieval snippet caps for context injection.
- `packages/general_beckman/.../posthook_handlers/copy_compliance_review.py:468` — `privacy_policy[:3000]`
  fed to the compliance reviewer (judge-against-contract class; tail violations could be missed — verify).
- `src/workflows/engine/constrained_emit.py:123` — `draft[:30000]`. **DEAD** (`maybe_apply` has no live
  caller; CPS-SP3 `reflection_posthook.build_emit_messages` replaced it — a coulson test asserts it's no
  longer inline). Same bug if ever revived; delete the file or fix-on-revive.

---

## 3b. Quality-failure escalation — audit + fix (shipped, not live)

User flagged: a hallucinating model burning attempts is the real problem. The
escalation mechanism (`retry.py::get_model_constraints`, read by
`fatih_hoca/requirements_builder.py:121` at `worker_attempts>=3`) is INTACT — two
arms: difficulty bump `(attempts-2)*2` (keyed on attempt count) + model exclusion
(keyed on `failed_models`). Audit of every quality re-pend in `apply.py`:

| path | retries to 3 | difficulty bump | model exclusion |
|------|------|------|------|
| grade-FAIL (~4530) | yes | ✓ | ✓ (only path that wrote `failed_models`) |
| simple_blocker checks/security/a11y/contract/perf + verify_artifacts/code_review/test_run/semgrep/pattern_lint/type_sync/migration | yes | ✓ | ✗ |
| **Z1 producer blocker `prior_art_min_coverage`** | **NO — single-shot DLQ @ attempt 1** | ✗ | ✗ |

289731 PROVES escalation works where wired (4 models tried) but truncation defeats
all of them — so escalation ≠ truncation fix; both were needed.

**Fix (TDD: `packages/general_beckman/tests/test_quality_escalation.py`):**
1. `_record_failed_model(ctx)` injected into `_stamp_retry_feedback` — the one
   chokepoint all 21 quality re-pends call. Now every quality failure excludes the
   failing model, not just grade. Quality-only chokepoint (availability rides
   `decide_retry`), so always correct.
2. `_PRODUCER_QUALITY_Z1_BLOCKERS = {"prior_art_min_coverage"}` — dispatcher routes
   it to the retry+escalate rail (`_apply_simple_blocker_verdict`) instead of
   single-shot DLQ. Deterministic-artifact Z1 blockers + critic_gate stay single-shot.

## 4. What's verified vs assumed
- **Verified:** the `[:500]` severs sections 4-5 of #289700's 1329-char instruction (read the DB row,
  sliced it, the two cut sections are exactly the ones the grader called "extra"); 94/259 i2p steps > 500;
  no encoding corruption; the four prompt-builders truncated; fix tests green.
- **Assumed / confirm on next mission run:** that with the full instruction, a compliant charter now
  PASSES the grader (re-run i2p phase 0 on a fresh mission after restart). The deterministic
  `verify_charter_shape` check stays as the structural belt-and-suspenders — it is correct and unchanged.
