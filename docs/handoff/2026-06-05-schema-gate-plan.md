# Schema-gate plan — deterministic artifact-schema enforcement before grade

**Date:** 2026-06-05
**Trigger:** DLQs #289735 (2.8 user_stories), #289737 (2.10 monetization_strategy) — grader `COMPLETE: NO`.

## Root cause (verified, not the grader)
The LLM grader verdicts were **correct**: both artifacts were genuinely incomplete.
- #289737: artifact had 3 keys (`pricing_model, tiers, revenue_projections`); step instruction prose demanded 6.
- #289735: step requires a JSON **array** of 8–15 stories; producer emitted a **single object**.

Both retried 5/6 worker attempts on capable models (gemini-2.5-flash, gemma-31b, Qwen3.5-9B) and DLQ'd. The retry feedback (`_schema_error = "Grader rejected output: ...COMPLETE: NO..."`) carries **no actionable reason** — the grading prompt (`src/core/grading.py:24-40`) has no "what's missing" field (SITUATION = skill-capture, not a deficiency). So capable producers retried blind.

`validate_artifact_schema` (`src/workflows/engine/hooks.py:710`) + `schema_dialect.validate_value` (`schema_dialect.py:132`, normalizes legacy `required_fields`/`item_fields` at entry) are **fully capable** of enforcing both shapes — but are wired ONLY as a produces-gated file-persistence tie-breaker inside auto-persist/recanonicalize (`coulson/react.py:843, 905`). They **never gate/fail a task**. Analyst steps with no file `produces` get zero deterministic schema enforcement.

## Fix set (user approved 1 + 4 + 3)

### #1 — Schema gate (spine). Fixes 2.8.
New gating mechanical post-hook `schema_gate`, mirroring `check_grounding`:
- `packages/mr_roboto/src/mr_roboto/validate_schema.py` — `validate_artifact_schema(output_value, schema) -> {"passed", "error", ...}`, reusing `src.workflows.engine.hooks.validate_artifact_schema`.
- `packages/mr_roboto/src/mr_roboto/__init__.py` — dispatch branch `action == "schema_gate"`.
- `packages/general_beckman/src/general_beckman/posthooks.py` — `POST_HOOK_REGISTRY["schema_gate"]` (cheap, mechanical); attach in `determine_posthooks` when `artifact_schema` present, ordered BEFORE grade (parallel mechanical path, drains `_pending_posthooks` pre-grade).
- `packages/general_beckman/src/general_beckman/apply.py` — payload builder branch (`_posthook_agent_and_payload`) + `_apply_schema_gate_verdict` (mirror `_apply_grounding_verdict`): on FAIL → retry source with validator's precise `error` as `_schema_error` + escalate `failed_models`; honor attempt cap + bonus. On PASS → drain, proceed to grade.

### #4 — Drift lint + reconcile 2.10. Fixes 2.10.
- Load-time lint: fail any step whose instruction names output fields absent from its `artifact_schema` (the 2.10 class). Sweep flagged candidates (heuristic noisy — needs field-name precision).
- Reconcile step 2.10 in `src/workflows/i2p/i2p_v3.json`: instruction lists 6 fields, schema `required_fields` lists 3. Decide single contract (expand schema to 6, since `done_when` mentions free/paid split) → gate then enforces all 6.

### #3 — Constrained emit (prevention, last).
Build grammar/`response_schema` from `artifact_schema`, force producer emit. Memory: constrained-decode Phase A+B shipped but unwired for these steps. Backend-dependent.

## Notes
- Grader stays **semantic-only** (RELEVANT/COHERENT/WELL_FORMED); completeness/shape moves to the mechanical gate. Do NOT add a reason field to the grader (rejected — fragile, fights prose-leakage guard).
- Not live till KutAI restart.
- TDD throughout; `timeout` on all pytest.

## STATUS 2026-06-05

### #1 SHIPPED (working tree, uncommitted)
Realized at the **grade boundary**, not a separate post-hook kind — the chain
cursor (`_enqueue_posthook_llm_child`) is LLM-specific and a parallel mechanical
lane would race `constrained_emit`. Since `grade` is always last in the chain,
the result is post-rewrite there.
- `packages/mr_roboto/src/mr_roboto/schema_gate.py` — pure helper reusing
  `validate_artifact_schema`. Test: `packages/mr_roboto/tests/test_schema_gate.py` (5 pass).
- `apply.py` `_enqueue_posthook_llm_child` `kind=="grade"` head — schema check
  before building the grade spec; FAIL → `_apply_posthook_verdict(grade, passed=False,
  raw={"error": "schema gate: ..."})` (flows via `_grader_verdict_text` "error" key into
  the existing grade-fail retry/escalation) + `return False` (no LLM grade). PASS/no-schema
  → grade as before. Test: `packages/general_beckman/tests/test_schema_gate_at_grade.py` (3 pass).
- Regression: 38 post-hook chain tests pass. **Safety:** gate enforces the schema
  (a floor); for the instruction>schema drift below it is LOOSER than the grader,
  so introduces NO new false-DLQs. Fixes #289735 (object-vs-array) + the blind-
  retry-to-DLQ class.

### #4 core SHIPPED; broad sweep TRACKED
- 2.10 reconciled to the **6-field** contract (founder call) in `i2p_v3.json`
  (schema.required_fields + done_when). Fixes #289737. JSON valid; dep-integrity green.
- `src/workflows/engine/field_drift.py` — drift lint (comma-run snake_case
  extractor, recursive schema field names, subtracts input/output artifact refs).
  Test: `tests/workflows/test_i2p_v3_field_drift.py` (10 pass) incl. 2.10 no-drift anchor.
- **TRACKED DEBT — 40 steps with instruction>schema drift** (de-noised). Each is a
  latent 2.10-class divergence (grader may fail schema-valid output). NOT auto-fixable:
  each needs the same founder field-set decision as 2.10, and the set includes false
  positives (verbs: `create_product`, `mr_roboto`, `code_review`, `db_migration`,
  `capture_screenshots`) and intentionally-loose schemas (`5.0a design_tokens`). High-
  signal genuine candidates to reconcile first: 1.6 competitor_pricing, 1.7 sentiment,
  1.8 ux, 1.9 gap, 1.14 go_no_go, 2.9 success_metrics, 3.2/3.3 nfr, 4.1 arch_pattern,
  13.4 analytics, 14.8 app_store. Reverse direction (schema>instruction) SCANNED
  2026-06-05: 108 steps have a gate-enforced field whose exact snake_token isn't
  a substring of the instruction — but these are overwhelmingly NAMING gaps, not
  bad schemas (mission_statement vs "mission"; must_have/should_have/could_have/
  wont_have = MoSCoW; competitor_name). Conclusion: #1 is not risky via a few
  broken schemas — it is a BROAD enforcement tightening (240 schemas previously
  only soft-graded are now hard-checked). Self-correcting (gate FAIL rides the
  retry ladder with precise per-field feedback, no insta-DLQ) but NOT statically
  provable safe. **REQUIRED: live-mission smoke test post-restart.** Watch for
  `"schema gate: X missing required field"` DLQs where X is genuinely optional →
  wrong schema to reconcile, not a gate bug.
- Optional next: a **ratchet** test (snapshot current drift as allowlist, fail on NEW
  drift) so edits can't add divergence while the 40 are worked through.

### #3 constrained emit — SHIPPED (commit 1bd6dece).
The plan framed #3 as "build grammar/response_schema from artifact_schema,
force producer emit" — but that infra (Phase A+B, 2026-04-26) was ALREADY live:
`constrained_emit` auto-wires as a result-rewriting post-hook (ordered before
grade) on every step with a constrainable `artifact_schema`
(`posthooks.determine_posthooks` → `_emit_is_constrainable`), the translator
(`json_schema_translator.py`) builds a strict json_schema response_format, and
it's reachable for analyst steps with no file `produces`.

The REAL gap was the skip predicate. `should_skip_emit`
(`src/core/reflection_posthook.py`) skipped the emit whenever the draft carried
the top-level artifact KEY — ignoring nested `required_fields` and array shape.
So the 2.8/2.10 class (object with 3 of 6 fields; single object where an array
is required) was SKIPPED, flowed to grade/the #1 gate, and DLQ'd on a blind
retry. Fix: gate the skip on FULL schema validation (the same
`validate_artifact_schema` the #1 gate uses) — skip iff the draft already
passes, so the emit fires exactly when the gate would reject and the constrained
re-emit lands BEFORE the gate. Strictly a superset of the old emit cases (never
emits less; never re-emits a conforming draft, so the tail-compression worry is
gone). Test: `tests/core/test_should_skip_emit.py` (6 pass) incl. both DLQ
classes as anchors. Regression: posthook chain-order 12, schema-gate-at-grade 3,
translator+legacy-emit 29 — all green. Not live till restart.

Note on layering: emit (capable model forces nested fields via strict
json_schema) is the PREVENTION layer; the #1 gate is the deterministic BACKSTOP
for drafts the emit couldn't fix (degraded json_object models, genuinely-missing
content). The two now share one validator, so they never disagree.
