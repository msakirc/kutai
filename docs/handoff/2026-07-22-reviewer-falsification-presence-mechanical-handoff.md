# Handoff — reviewer falsification-triple: presence is MECHANICAL, not LLM (2026-07-22)

## TL;DR
Killed the recurring FALSE review-halt where step **3.11 requirements_review** confabulates
"FR table empty / FR-xxx missing falsification triple" for requirements that provably carry all
three fields. Root: **presence of a triple is an objective fact**, but it was judged twice — once
mechanically (`verify_falsification_present`, a hard blocker on the producers, which PASSED) and
again by the confabulation-prone LLM reviewer re-deriving it from prose. When they disagreed, the
LLM's false "missing" (severity `major`) halted the mission.

Fix (design-reviewed + code-reviewed, both by adversarial Opus sub-agents): **presence mechanical,
quality LLM.** Two parts shipped; a third ("persist the mechanical result to the reviewer") was
DROPPED by the design review as high-effort/low-leverage.

- **Part 3 — Rule C** in `ground_review_verdict` (`packages/mr_roboto/src/mr_roboto/verify_review_verdict.py`):
  a "missing/empty triple" finding is deterministically DROPPED when the requirements_spec table
  populates the risk_if_wrong / validation_method / falsification_signal columns for the cited rows
  (parsed + fed to the SAME `verify_falsification_present` the producers gate on). **Anchored** to the
  finding's named IDs (drop only if every named FR/NFR/BR-id is in the parsed triple-table AND proven);
  no-id findings need the absence marker bound DIRECTLY to a triple token. Quality qualifiers
  (vague/specific/observable/measurable) EXCLUDE Rule C so genuine specificity findings survive.
- **Part 2 — 3.11 instruction** (`src/workflows/i2p/i2p_v3.json`, Z1 Tier 2 P4 block): presence is NOT
  the reviewer's job (hard-gated upstream + grounded); do not scan/claim missing triples for FR/NFR/BR;
  judge only QUALITY. Critical-vagueness rule scoped to items whose `risk_if_wrong` is LITERALLY
  `'critical'` (fixes a second over-claim: the reviewer flagged high/medium FRs as "critical").

**Why no quality loss:** the LLM keeps every subjective judgment (specificity, observability,
contradictions, traceability). It only loses the OBJECTIVE presence check a deterministic verifier
already does correctly 100% of the time. **Safety:** a genuinely-missing triple fails the PRODUCER
(`verify_falsification_present` is in `_Z1_BLOCKER_KINDS`, `apply.py:4372`, fail-closed on 3.1/3.2/3.3/3.7)
BEFORE 3.11 runs — stripping the reviewer's presence authority loses nothing (confirmed by the code review).

## ⭐ DO THIS FIRST (validate live)
1. **`/restart`** — Rule C is CODE (mr_roboto, editable install) → needs restart to load. The 3.11
   instruction is `i2p_v3.json` (reloads live via `(path,mtime)` — already effective, NO restart needed).
2. **Re-reset 567426** (the parked m90 review halt) — ledger-clearing SQL (rw `sqlite3.connect(db,timeout=10)`
   + `PRAGMA busy_timeout=8000`), DB `C:\Users\sakir\ai\kutai\kutai.db`:
   ```sql
   UPDATE tasks SET status='pending', task_state=NULL, result=NULL, error=NULL,
     worker_attempts=0, grade_attempts=0, next_retry_at=NULL, exhaustion_reason=NULL,
     retry_reason=NULL, sleep_state=NULL,
     context=json_remove(context,'$._rejection_ledger','$._schema_error',
       '$._schema_error_for_attempt','$._prev_output','$.failed_models','$.excluded_models',
       '$.grade_excluded_models')
     WHERE id=567426 AND status IN ('failed','waiting_human');
   ```
3. **Watch.** Expected: the reviewer re-runs with the new instruction → emits FEWER/no false findings
   (no presence claims; vagueness scoped to critical). Rule C backstops any residual presence claim.
   567426 should PASS (or halt only on a genuinely real finding), unblocking m90's requirements phase.

## What the two reviews changed (don't re-learn)
- **Design review** (pre-build): DROP Part 1 (persist result) — no persistence path, keying mismatch
  (reviewer expects 4 `*_falsification_result` artifacts; producers emit differently-named outputs),
  weak leverage (feeds mechanical truth to the same unreliable LLM). Part 2 is 95% of the value.
  Part 3 is a NEW rule, not a resolver tweak — existing Rules A/B are structurally blind (they check
  md-headers / top-level JSON keys; triples are nested table CELLS). Scope Part 2 to FR/NFR/BR
  (SQA/persona have no upstream mechanical gate).
- **Code review** (post-build): found TWO real over-drops, both FIXED (`3afa3a50`):
  - **D1** — FR table proving out dropped a real "NFR-001/002 missing triple" (NFRs render as PROSE,
    not a triple-column table; the FR table is not evidence about NFRs).
  - **D2** — absence marker + falsification token matched anywhere → "missing a traceability matrix;
    falsification signals could be stronger" wrongly dropped (real: missing matrix).
  Fix = anchor the drop to the finding's named IDs / require the absence marker to bind directly to a
  triple token. Both now have regression tests.

## Commits (branch `main`; parallel Yaşar-Usta session interleaves)
- `4398ea9d` — Rule C (presence-anchored to spec table) + 3.11 instruction rewrite.
- `3afa3a50` — anchor Rule C to named IDs + clause binding (D1/D2 over-drop fixes).
(Verify with `git log --oneline --grep falsification`.) Both restart-gated for the CODE half (Rule C);
instruction is live.

## Key mechanisms & gotchas
- **Instruction reloads live, code does not.** `i2p_v3.json` is `(path,mtime)`-keyed via
  `_refresh_workflow_step_config` → 3.11 picks up the new instruction on next run WITHOUT restart. Rule C
  (mr_roboto code) needs `/restart`.
- **Rule C parser** (`_parse_spec_requirement_triples`): finds markdown tables whose header names
  risk/validation/falsification columns, maps cells to the triple fields + `req_id`, feeds
  `verify_falsification_present`. Only the FR table has triple columns in the current 3.10a/3.10b render;
  NFR/BR/Security are prose → correctly NOT proven (D1 guard).
- **`major` is blocking** (`_NON_BLOCKING_SEVERITIES` = {minor,medium,low,info,trivial,nit}); dropping a
  confabulated `major` re-derives `pass` only if no other blocker survives.
- **Grounding is HIGH-PRECISION**: drop only on certainty; any doubt / unparsed table / unnamed section → KEEP.

## Known residuals / deferred (NOT blocking; candidates for next session)
1. **567426 finding [1]** — "Vague validation methods for critical risk requirements (FR-006, FR-010)".
   FR-006=medium, FR-010=high → NOT critical; the reviewer misapplied its own critical-only rule. The
   Part-2 instruction now tells it to scope to literally-critical, so a FRESH review should not emit this.
   If it persists, it's a QUALITY judgment (is high-risk validation specific enough?) — LLM's domain by
   design; override if you disagree. NOT worth a mechanical Rule D (that would mechanize a judgment).
2. **Rule B drops the non-goals-contradiction finding against the WRONG artifact.** In-situ, 567426's
   "Contradiction with non-goals" is dropped by the pre-existing Rule B (fabricated-quote) because it
   grounds against `target_artifact=requirements_spec.md` but the non-goal text lives in `non_goals.md`.
   Pre-existing behavior (not this fix). If non-goals findings matter, Rule B / the resolver should ground
   a non-goals finding against `non_goals.md`. Deferred.
3. **SQA / persona have no upstream `verify_falsification_present` gate** (only 3.1/3.2/3.3/3.7 = FR/NFR/BR).
   Part 2 scopes the "verified upstream" claim to FR/NFR/BR. If SQA/persona ever need triple presence
   guaranteed, add the post-hook to their producers.

## Broader session context (for the picture)
This is the LAST of a long m90 gate-DLQ saga — every DLQ this session was a CORRECT artifact
false-rejected by a broken gate (repr-serialize, fence-strip, flow-only-regex, summary-starvation,
robotics-in-fleet-pool, multi-produces-narration, and now falsification-presence-confabulation). All
fixed at root, TDD. m90 is otherwise clean/flowing. Also this session: paraflow cron DLQ fixed
(broken editable install → `pip install -e packages/c21_paraflow_diff`, restart-gated); stripe cron
still failing (needs STRIPE key — user declined root-fix); 11 stale `dead_letter_tasks` rows cleared
(resolved_at). Memory: `project_reviewer_summary_starvation_20260626`, `project_reviewer_verdict_verification_20260626`,
`project_m90_shape_verifier_yaml_parser_20260720`, `project_dlq_triage_paraflow_stripe_20260721`.

## Files
| File | Change |
|---|---|
| `packages/mr_roboto/src/mr_roboto/verify_review_verdict.py` | Rule C + helpers (`_parse_spec_requirement_triples`, `_falsification_absence_refuted`, `_TRIPLE_ABSENCE_CLAUSE`, `_REQ_ID_RE`, `_QUALITY_QUALIFIERS`) wired into `classify_issue_grounding` |
| `packages/mr_roboto/tests/test_verdict_grounding.py` | 8 Rule C tests (drop-when-present, keep-when-empty, D1 NFR-prose, D2 unrelated-absence, quality-qualifier, no-table) |
| `src/workflows/i2p/i2p_v3.json` | 3.11 Z1 Tier 2 (P4) instruction: presence-not-reviewer's-job, quality-only, critical-scoped |
| `packages/mr_roboto/src/mr_roboto/verify_falsification_present.py` | (unchanged) the producer-gate mechanical checker Rule C reuses |
