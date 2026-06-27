# Handoff — Fix 3 grade-skip tradeoff: deterministic verify auto-PASS drops the LLM relevance axis

**Date:** 2026-06-27
**Status:** KNOWN TRADEOFF shipped (committed, restart-gated, not pushed). Not a defect — a deliberate coverage reduction validated by two Opus review passes. This handoff is for a FRESH session to decide whether to refine it.
**Owner decision needed:** keep auto-PASS as-is, or implement the "advisory-COMPLETE" refinement below.

## What shipped (commits `df0106c8` + `fcc8cbf8`)
In the grade gate (`packages/general_beckman/src/general_beckman/apply.py`, `_enqueue_posthook_llm_child`, `kind == "grade"` branch), after `build_grading_spec` + the empty-scope short-circuit, the step's authoritative deterministic check (`_is_grade_authoritative_check`: any `verify_*_shape` OR a kind in `_GRADE_AUTHORITATIVE_NON_SHAPE_CHECKS = {verify_adr_register}`) is run INLINE via `mr_roboto.run(...)`. On `status == "completed"` the grade is **auto-PASSed and the LLM grade child is never spawned** — mirroring the existing empty-scope short-circuit.

This fixed mission-90 567449 [5.0a] design_tokens: a shape-VALID artifact was confab-FAILed `COMPLETE:NO` by the LLM grader (whose prompt explicitly says "DO NOT JUDGE field/section presence"), then DLQ'd as "degenerate repeat" when the producer re-emitted the same correct artifact. Skipping the confab-prone grade kills that doom loop.

## The tradeoff
The LLM grade judged **two** axes: completeness/presence (which the grader was told NOT to judge but confabulated anyway) AND **topicality/relevance** (is this artifact about the RIGHT product/mission, is it coherent). The deterministic `verify_*_shape` checks validate **shape only** — they do not check topicality. By skipping the LLM grade entirely on a shape PASS, we drop the relevance check.

**Concrete risk:** a shape-valid artifact describing the WRONG product (hallucinated-but-well-formed) now auto-passes. E.g. a `user_flow.md` with correct frontmatter + one mermaid-per-surface but describing a competitor's app, or a `premortem.md` that is well-structured but about the wrong feature.

## Blast radius — 24 steps now skip the LLM grade when their verifier passes
Highest topicality-risk (narrative prose, relevance matters most):
- `[0.0z]` reverse_pitch · `[0.1]` product_charter · `[0.0c]` interview_script · `[0.6a.draft]` non_goals · `[1.4a]` competitive_positioning · `[5.0c]` user_flow · `[6.5z]` premortem · `[4.14]` adr/register

Lower risk (structured JSON/ADR/style — shape ≈ correctness, but a wrong tech-stack choice could still be shape-valid):
- `[4.1][4.2][4.2a][4.4][4.6][4.8][4.9][4.10]` ADR decisions (verify_adr_shape) · `[5.0]` taste_emphasis · `[5.0a]` design_tokens · `[5.0b]` surfaces · `[5.0d]` screen_inventory + shared_shell · `[5.20a/b]` screen_plan · `[5.30a/b]` html_prototype

(Generate the live list: grep `i2p_v3.json` for `checks[].kind` matching `verify_*_shape` or `verify_adr_register`.)

## Mitigations already in place (why this is currently acceptable)
- Matches the existing **empty-scope short-circuit** precedent (apply.py ~1917) which ALSO skips the LLM grade when completeness is deterministically proven.
- `build_grading_spec`'s **dogru_mu_samet degeneracy/trivial floor** still runs BEFORE the short-circuit (returns a `GradeResult` auto-fail at apply.py ~1900) — garbage can't auto-pass.
- The **canonical product-name pin** (`verify_contains_product_name` on 0.0z/0.1 etc.) + other checks catch some off-topic.
- Two Opus reviews called it "defensible … the one thing to watch."

## The refinement to investigate — "advisory COMPLETE, keep RELEVANT"
Instead of skipping the grade, RUN it but make a **completeness-driven FAIL advisory** while a **relevance/coherence-driven FAIL still terminal**. The grader verdict has fields `RELEVANT / COMPLETE / WELL_FORMED / COHERENT / VERDICT`. For 567449 the FAIL was driven SOLELY by `COMPLETE:NO` (RELEVANT:YES, WELL_FORMED:PASS, COHERENT:PASS). Rule: when a `verify_*_shape` check passed, a grade FAIL whose only failing axis is `COMPLETE` is overridden to PASS; a FAIL with `RELEVANT:NO` or `COHERENT:NO` stays terminal.

**Why it's harder than the current fix (the real work):**
- The grade verdict comes back from an **async LLM grade child** (`posthook.grade.resume`), NOT inline. The short-circuit decision (skip vs spawn) happens at spawn time in `_enqueue_posthook_llm_child`; the verdict is applied later in the resume handler.
- So you must: (a) at spawn time, tag the grade continuation with `shape_verify_passed=True` (the inline verify already ran — reuse its result); (b) in the grade-verdict resume/apply path, parse the verdict fields and, when the tag is set AND only `COMPLETE` failed, override to PASS instead of re-pend/DLQ.
- Find the grade-verdict applier: trace `posthook.grade.resume` → `_apply_posthook_verdict` for `kind=="grade"` → the grade-FAIL re-pend/DLQ branch (it calls `_stamp_retry_feedback`). Verdict field parsing: see `_grader_verdict_text` / the grading spec parser in `src/core/grading.py` (`build_grading_spec`, `GradeResult`) — confirm the `RELEVANT/COMPLETE/COHERENT/VERDICT` shape and how it's currently parsed on the apply side.
- Edge cases: prose graders that emit a bare `COMPLETE:NO` without per-axis fields (the apply.py:1856 comment references this); a grader that fails RELEVANT *because* it thinks something is missing (entangled axes); the degenerate-repeat interaction (if you keep the grade running, a converged artifact whose grade FAIL is advisory must COMPLETE, not degenerate-DLQ).

## Decision matrix
- **Keep auto-PASS (status quo):** simplest, kills the doom loop, loses topicality on 24 steps. Acceptable if product-name pins + reviewers (1.13 etc.) catch off-topic downstream.
- **Advisory-COMPLETE refinement:** preserves topicality, more code + risk (verdict interception, axis parsing, degenerate interaction). The "no-band-aid" choice if off-topic-but-shape-valid is a real observed failure.

## How to validate either way
- Re-run a mission through phase 0/5 and check the narrative steps (0.1 product_charter, 5.0c user_flow, 6.5z premortem) complete with on-topic content.
- Adversarial test: inject a shape-valid-but-off-topic artifact on disk for a verify-gated step and confirm whether it auto-passes (current) vs gets caught (refinement).
- `packages/general_beckman/tests/test_grade_verify_authority.py` is the test bed.

## Key files
- `packages/general_beckman/src/general_beckman/apply.py` — grade gate (`_enqueue_posthook_llm_child`), `_is_grade_authoritative_check`, `_GRADE_AUTHORITATIVE_NON_SHAPE_CHECKS`, the empty-scope precedent.
- `src/core/grading.py` — `build_grading_spec`, `GradeResult`, verdict shape.
- `packages/general_beckman/posthooks.py` — `determine_posthooks` (grade is appended BEFORE the `checks` verbs → checks run AFTER grade; this is WHY Fix 3 runs the verify inline rather than relying on ordering).

## Memory
`project_m90_three_gate_fixes_20260627`.
