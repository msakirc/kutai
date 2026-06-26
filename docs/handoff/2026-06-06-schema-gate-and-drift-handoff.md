# Handoff — schema gate, NFR reconcile, drift backlog

**Date:** 2026-06-06
**Branch:** main
**Predecessor:** `docs/handoff/2026-06-05-schema-gate-plan.md` (still the design-of-record for #1/#3/#4)

## TL;DR

Shipped the full schema-gate plan plus the entire instruction↔schema drift
backlog. **6 commits on main, all restart-gated** (inert until KutAI restarts).
A live smoke test (mission ran post a mid-session restart) confirmed the gate
itself is safe — **zero false positives** — and surfaced three real defects, two
of which are now fixed and one deferred by founder call.

## Commits (oldest → newest)

| SHA | What |
|-----|------|
| `789ebac4` | fix(grading): remove ALL LLM-I/O truncation that severed long charters |
| `4847aa01` | fix(beckman): escalate failed model on ALL quality re-pends (not just grade) + route prior_art_min_coverage to retry rail |
| `0ea97a15` | feat(beckman): deterministic artifact-schema gate before LLM grade (#1) + drift fix 2.10 (#4) |
| `1bd6dece` | fix(emit): fire constrained re-emit on incomplete drafts, not just missing top-level keys (#3) |
| `6faca65e` | fix(i2p): reconcile 3.2/3.3 NFR schemas to single items-array contract |
| `4415fd1c` | fix(i2p): reconcile 16 genuine instruction↔schema drifts |

(`789ebac4`/`4847aa01` were tested-but-uncommitted from 2026-06-04 handoffs;
`apply.py` intermingled all three concerns as distinct hunks and was split via
`git add -p` into clean per-concern commits.)

## #3 correction (the prior plan was wrong about this)

The plan framed #3 as "build grammar/response_schema, force producer emit — NOT
STARTED". **That infra already shipped** (constrained-decode Phase A+B,
2026-04-26): `constrained_emit` auto-wires as a pre-grade result-rewriting
post-hook on every step with a constrainable `artifact_schema`
(`posthooks.determine_posthooks` → `_emit_is_constrainable`), and
`json_schema_translator.py` builds a strict json_schema.

The **real** gap was the skip predicate: `should_skip_emit`
(`src/core/reflection_posthook.py`) skipped the emit whenever the draft carried
the top-level artifact KEY — ignoring nested `required_fields` / array shape. So
the 2.8/2.10 DLQ class (object missing 3 of 6 fields; single object where an
array is required) slipped past the emit to the gate and DLQ'd. Fix: gate the
skip on FULL `validate_artifact_schema` (the same validator the #1 gate uses) —
skip iff the draft already passes, so the emit fires exactly when the gate would
reject. Strictly a superset of the old emit cases (never emits less, never
re-emits a conforming draft → the old tail-compression worry is gone). Layering:
emit = PREVENTION (forces nested fields on a json_schema-capable model), the #1
gate = deterministic BACKSTOP; they share one validator so they never disagree.

## Live smoke test (post mid-session restart)

**Gate produced ZERO false positives.** Three DLQ signals, triaged:

1. **3.1** `functional_requirements[8].source_story_ids: empty placeholder` —
   gate working correctly; a genuine producer gap with precise actionable
   feedback. No code change (producer retry fills it).
2. **3.2 / 3.3** `nfr_*.items: missing required field` — real schema drift, now
   fixed (`6faca65e`). The NFR steps carried pre-Z1 schemas requiring both flat
   metrics AND an `items` array, but the instruction + `verify_falsification_
   present` only consume `items`. Reconciled all four NFR artifacts
   (nfr_performance / nfr_scalability / nfr_availability / security_requirements)
   to a single `{items: array<min_items=3, {name, target, risk_if_wrong,
   validation_method, falsification_signal}>}` contract.
3. **5.0b** `clarify payload requires 'question'` — **DEFERRED (founder)**.
   Pre-existing wiring gap: `clarify.py` has no `surface_choice` handler, so it
   falls to the default branch which requires `question` and cannot render the
   options keyboard or persist `surfaces.json`. See "Open work" below.

## Drift backlog (commit `4415fd1c`)

The `field_drift` lint flagged 40 steps where the instruction enumerates output
fields absent from the schema. This matters because the **grader still checks
`COMPLETE: YES/NO`** against the full instruction (grading.py:32 — the plan's
"grader semantic-only" never shipped). So instruction>schema drift causes grader
`COMPLETE:NO` DLQs even though the gate (which checks the looser schema subset)
passes.

Five parallel agents triaged the 40 → **16 genuine, 24 false positives** (item
VALUES, verbs/step/package names, enum options, reviewer-read-upstream fields,
frontmatter `mission_id`, the intentionally-loose 5.0a `design_tokens` token
map). Reconciled the 16: 1.14 (scores: 5 dims), 2.7, 1.9, 1.8, 1.7, -1.6, 4.7,
3.7, 3.6, -1.4, 2.4, 1.6 (+ tiers), 6.2, 8.0, 8.sprint_ritual, 1.5.

**Two disciplines that keep this safe:**
- The gate rejects empty `[]`/`{}`/`""` for REQUIRED fields. So any field that
  can legitimately be empty (free-tier prices, `future_languages`, per-phase
  gaps, `bottleneck_tasks`, ...) was added as `optional` — never manufacturing a
  new false-DLQ.
- Pure naming mismatches (1.5 `feature_description`/`which_competitor`, 6.2
  `parallelizable_groups`) were reconciled by aligning the **instruction** to
  the stable schema field name — NOT renaming the schema — so there is zero
  downstream-consumer ripple.

Drift lint now 40 → 24 (all 24 remaining confirmed false positives). Test:
`tests/workflows/test_i2p_v3_drift_reconcile.py` (48 pass with field_drift +
nfr).

## Tests

- New: `test_should_skip_emit.py` (6), `test_i2p_v3_nfr_items_schema.py` (4),
  `test_i2p_v3_drift_reconcile.py` (24+ params). TDD throughout (RED watched).
- Regression green: posthook chain-order (12), schema-gate-at-grade (3),
  quality-escalation (6), translator (15), build_grading_spec (4).
- **Pre-existing failures (NOT introduced this session):** 6 tests in
  `tests/i2p/` KeyError on removed `.verify` / `.draft` / `.draft_confirm`
  sibling steps — `test_falsification::test_workflow_step_carries_falsification_
  post_hook` (3.1.verify), `test_adr_shape` ×3, `test_non_goals::test_workflow_
  step_0_6a_draft_confirm_split` + `::test_schema_version_carried_on_artifacts`.
  Those standalone steps were removed when verification moved to `post_hooks`;
  the tests were not updated. Verified to fail identically at HEAD with this
  session's i2p_v3.json change stashed.

## Open work (priority order)

1. **Restart KutAI + DLQ-retry.** All 6 commits are inert until restart. After
   restart, re-run the live smoke: watch for `"schema gate: X missing required
   field"` DLQs where X is genuinely OPTIONAL → that means a wrong schema to
   reconcile, NOT a gate bug. (#1 is a broad enforcement tightening — 240 schemas
   previously only soft-graded are now hard-checked.)
2. **5.0b `surface_choice` clarify handler.** Add a branch in
   `packages/mr_roboto/src/mr_roboto/clarify.py` mirroring `variant_choice`:
   render the inline `options` payload as a reply keyboard, capture the founder's
   pick, persist to `surfaces.json` with the `{_schema_version, mission_id,
   surfaces, primary_surface, founder_confirmed_at}` shape so `verify_surfaces_
   shape` passes. (Minimal "add a question" fix is insufficient — it loses the
   keyboard and never writes surfaces.json.)
3. **Pre-existing stale `.verify`/`.draft` tests.** Decide: update the 6 tests
   to match the post_hooks-based design, or restore the standalone steps.
4. **24 drift false positives** — correctly left as-is; no action unless a future
   live DLQ proves one genuine.
5. **grader semantic-only** (the plan's stated intent) is still NOT shipped. If
   adopted, instruction>schema drift becomes harmless and the backlog above is
   moot — but it can't simply drop `COMPLETE` (markdown/string artifacts have no
   field schema, so the gate is vacuous for them and the grader is their only
   completeness check). Treat as a separate design decision.

## Key files

- `packages/mr_roboto/src/mr_roboto/schema_gate.py` — #1 gate helper.
- `packages/general_beckman/src/general_beckman/apply.py` — gate at the grade
  boundary (`_enqueue_posthook_llm_child` kind=="grade" head); escalation
  (`_record_failed_model`, `_PRODUCER_QUALITY_Z1_BLOCKERS`).
- `src/core/reflection_posthook.py` — `should_skip_emit` (#3).
- `src/workflows/engine/field_drift.py` — drift lint.
- `src/workflows/engine/schema_dialect.py` — validator/translator/normalizer;
  `is_empty_required_value` is why empty-able fields must be `optional`.
- `src/workflows/i2p/i2p_v3.json` — all schema reconciles.

Memory: `project_schema_gate_shipped_20260605.md`.
