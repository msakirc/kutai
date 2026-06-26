# Handoff — Mission-90 DLQ batch: narration-clobber, self-critique write-strip blindness, prose-fallback empty-scope

**Date:** 2026-06-25
**Status:** ✅ FIXED — TDD green, regression green. **RESTART-GATED, NOT committed.** User: `/restart` → re-pend the 3 tasks → verify → commit + push.
**Trigger:** three mission-90 i2p DLQs at 21:08–21:18

| Task | Step | Agent | DLQ reason |
|---|---|---|---|
| 567379 | `[0.6a.draft] non_goals_draft` | writer | grade FAIL — "Summarized draft instead of delivering" |
| 567381 | `[1.0a] prior_art_query_plan` | query_planner | `max_iterations_reached` |
| 567396 | `[1.11a] compliance_overlay` | analyst | Schema validation: missing `monitoring_obligations`, `data_subject_rights_implementation` |

All three are **false-rejects / false-loops** — the agents' outputs were acceptable. NOT "fix not loaded": the memory-flagged fixes (`fcc14eaa` three-gates, `18694892` empty-scope producer gate, `5602e259` materialize sole-writer, `773127e4` canonical pull) are all committed at HEAD. These are distinct **new** defects in the same schema'd-`produces` subsystem.

---

## Root causes (evidenced from transcripts + disk)

### Bug A — 567379 (writer, **markdown** produces)
Disk `non_goals.md` = 423-char **narration** ("Drafted … Ready for founder review"), no real `# Non-goals` body. But the transcript `tool_calls` shows the writer **DID `write_file` a clean 831-char doc** (`ok: true`). Two compounding defects clobbered it:

1. **Predicate drift** — `materialize_produces.write_stripped` was `schema-non-empty and not _allow_write_tools` (`hooks.py:372`), TRUE for markdown. But `_apply_auto_strip` (coulson:318) strips write tools **only for `_schema_is_structured_only`** (object/array) — markdown KEEPS write_file. The drift flipped the candidate order to `[output_value, disk]` (narration first) vs the documented `[disk, output_value]`.
2. **Substring section validator** — `validate_artifact_schema` markdown check was `f"# {s}" in text` (`hooks.py:948`), a substring. The narration literally contains `` `# Non-goals` `` → spuriously passed.

→ `select_canonical` returned the narration (first + passing) → **overwrote the writer's clean file** → LLM grader read narration → FAIL.

### Bug B — 567381 (query_planner, **object** produces → write STRIPPED)
Disk `prior_art_queries.json` = **correct** (5 queries + keywords). But the **self-critique guard** (`check_self_critique_sub_iter`) did NOT receive `allowed_tools`, so it couldn't tell write_file was stripped → critic "file not created … **Call write_file**" → impossible → agent loops ("We need to call write_file") → parse-fails → `max_iterations`. The **grounding** guard already skips this case (`guards.py:374`, committed `fcc14eaa`); the **self-critique** guard never got the same treatment — an asymmetry.

### Bug C — 567396 (analyst, **object** produces → write STRIPPED)
`compliance_fingerprint.jurisdictions == []` (empty scope). Analyst emitted markdown **prose** (not JSON) → object parse fails → **prose text-fallback** (`hooks.py:877-900`) flags the exempt fields "missing content about". That fallback **never consulted the empty-scope exemption** (`_empty_exemption_granted`) that the JSON-parsed path (`hooks.py:872`) honors. The grade-branch LLM auto-pass via `is_empty_scope_artifact` already exists; the deterministic producer-gate text-fallback was the lone hole.

---

## Fixes (4 surgical changes, 2 source files + test files)

1. **A1** `src/workflows/engine/hooks.py` `materialize_produces` — `write_stripped` now uses `coulson._schema_is_structured_only(schema)` (matches `_apply_auto_strip` exactly). Markdown steps regain `[disk, output_value]` order → the agent's written file outranks narration.
2. **A2** `hooks.py` `validate_artifact_schema` markdown branch — section match is **line-anchored** (`^\s*#{1,4}\s*<s>\b`, `^\s*\*\*<s>\*\*`, bare/setext `^\s*<s>\s*$`, MULTILINE) instead of substring. A backticked/inline prose mention no longer counts; legit ATX/bold/setext headers still pass.
3. **B** `packages/coulson/src/coulson/self_critique.py` + `guards.py` — `check_self_critique_sub_iter` gains `allowed_tools` param and skips when no `WRITE_TOOLS` present (mirrors grounding guard); `guards.py:265` passes `profile.allowed_tools`. `allowed_tools=None` ("all tools") preserves prior behavior.
4. **C** `hooks.py` object prose text-fallback — excludes required fields whose `_empty_exemption_granted(frule, inputs)` is True (consults `_normalize_rule(rules)["fields"]`).
5. **A1b (found by adversarial review)** `packages/coulson/src/coulson/__init__.py` — added `"json"` to `_STRUCTURED_SCHEMA_TYPES` (`{object, array}` → `{object, array, json}`). A1 aligned `materialize_produces` with `_apply_auto_strip`, which exposed that `_apply_auto_strip` ITSELF omitted `"json"` from the structured set — so the lone `type:"json"` step (`[0.0a.draft] intake_todo_draft`) kept `write_file` despite its instruction saying *"do NOT call write_file (the tool is intentionally unavailable for this step); your returned JSON IS the artifact"* (and the intake #73 design: engine is the sole writer). Including `"json"` makes the step instruction, `_apply_auto_strip`, and `materialize_produces.write_stripped` all agree. Only 1 json step exists and it forbids write_file → safe.

## Verification
- **TDD:** 7 new tests RED→GREEN (`test_i2p_v3.py` ×5 incl. A2 + C; `test_materialize_produces.py` ×2; `test_self_critique.py` ×3).
- **Regression:** touched suites all green — coulson 138 (1 pre-existing fail), general_beckman 375, iteration_exhaustion, produces_persist (incl. the test the json fix turned green WITHOUT edit), auto_strip_schema_scope, materialize_produces, workflow_hooks, compliance_overlay_empty_scope, schema_dialect, i2p_v3. Import smoke OK.
- **Pre-existing (NOT mine — confirmed by stashing source fixes):** `test_v3_all_steps_have_artifact_schema` (i2p_v3.json data gap, step `13.demo_storyboard_draft`); `test_no_legacy_residue` + `test_pool_empty_diag` (src.infra→dabidabi / admission_forensics→kara_kutu refactor casualties).

## Adversarial review (sub-Opus)
Independently re-verified all 3 root causes from DB transcripts + disk (not the summary) and validated each fix: **A1 SOUND** (predicate now byte-identical to `_apply_auto_strip`); **A2 SOUND** (narrow watch-item: `**Section:**` with colon inside the bold, and `## 🎯 Section` with a decorator between `#` and the name, are now rejected — neither in current use); **B SOUND** (`profile.allowed_tools` is post-strip — `_apply_auto_strip` runs before `_react_run`); **C SOUND** (input-anchored, `inputs` reaches the live producer gate). It surfaced the missed `"json"` issue (fix #5 above) and the now-resolved red test.

## Deploy
Restart-gated. The 3 tasks are frozen `failed` (won't self-heal). After `/restart`:
1. Re-pend 567379, 567381, 567396.
2. Verify: 567379 keeps its written `non_goals.md` (no "Drafted…" narration on disk); 567381 completes without max_iterations; 567396 passes with empty scope.
3. Commit + push.

## Deferred (observed, not in scope)
- **Self-critique verdict parse-fail loop:** after the critic resolves `clean`, the verdict envelope is re-fed to the main ReAct parser → "could not be parsed" nag loop (567379 messages [3]-[5]). Burned iterations but didn't cause this DLQ. Candidate: have the loop consume the resolved verdict without re-parsing.
- Rewriter seam (`self_reflect`/`constrained_emit` read `source.result`) — carried from the prior handoff.
