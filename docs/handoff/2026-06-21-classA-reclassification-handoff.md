# Handoff — Class A reclassification: schema-gate "dominant class" was availability-storm residue

**Date:** 2026-06-21
**Status:** investigation note / do-not-rechase. **No live schema-gate defect found.** Purpose: stop the next session from burning time chasing a phantom, and record the gate-plumbing trace so the dismissal is auditable.

## What the original handoff claimed
`2026-06-21-remaining-content-failures-handoff.md` §A called schema/shape-gate failures the "DOMINANT remaining class" and listed six failing steps (`0.6a.draft`, `1.4a`, `1.11a`, `1.0a`, `1.3`, `1.0c`) with exact gate strings.

## What the live DB actually shows (mission 87)
Counts: **187 pending / 115 completed / 5 skipped / 4 failed.** Status of the named steps (bracket-exact title match):
| Step | Task | status | note |
|---|---|---|---|
| `[1.0a]` prior_art_query_plan | 524362 | **completed** | the "missing content about ['queries','domain_keywords']" reject did not stick |
| `[1.3]` direct_competitor_identification | 524367 | **completed** | the "~0 list items, need >=3" reject did not stick |
| `[1.4a]` competitive_positioning_lock | 524369 | **completed** | the markdown "missing sections" reject did not stick |
| `[0.6a.draft]` non_goals_draft | 524360 | failed / **availability** | result=NULL; schema string is stale; restart-gated availability class |
| `[1.11a]` compliance_overlay | 524377 | failed / quality | weak-model JSON loop, result=NULL → separate handoff |
| `[1.0c]` prior_art_synthesize | 524364 | failed / None | `prior_art_min_coverage` Rule 4 (NOT a schema gate) → separate handoff |

So **three of the six completed on retry**, and the remaining three are availability / model-quality / coverage-gate — **none is a reproducible schema-gate rejection.** The schema strings in the original handoff were the `tasks.error` column's last-error left over from earlier attempts during the overnight availability storm (when steps cycled without ever producing real content). They are not gate contract drift.

## Gate plumbing is sound (the static trace, so the dismissal is grounded)
There are **two** schema gates and both validate the right thing:
1. **Engine post-exec gate** — `src/workflows/engine/hooks.py::_post_execute_workflow_step_impl` (gate at ~:1552). Runs in the Beckman apply path via `general_beckman/__init__.py:1298` (`is_workflow_step` → `post_execute_workflow_step`). Critically it validates the **post-materialize** value: `materialize_produces` (`hooks.py:1531`/`:272`) picks the schema-best of {on-disk write, result}, and **returns the on-disk file content** as the new `output_value`, "so the schema gate validates exactly what landed on disk." For `produces`-file steps this means the gate sees the FILE, not narration.
2. **Grade-posthook gate** — `mr_roboto/schema_gate.py` invoked at `general_beckman/apply.py:1750` with `output_value = source.get("result")`. Delegates to the same `validate_artifact_schema`.

The validator itself (`hooks.py:726` + `schema_dialect.py`): structured (`object`/`array`) validation runs only when `_extract_artifact_value` parses the artifact; on parse-fail it falls back to lenient text scans (`missing content about` keyword scan / `has ~N list items` regex / markdown `missing sections` header scan). The prompts for the named steps DO instruct the required keys/shape (e.g. `[1.0a]` instruction emits `{"queries":[…],"domain_keywords":[…]}`; `[1.4a]` spells out all six markdown sections verbatim). No prompt↔gate contract drift was found — and the live completions confirm capable models satisfy the gates.

## One latent seam (separate handoff, do not conflate)
Gate site (1) targets the materialized file; site (2) targets `source.result` (final_answer text). For `produces`-file steps these can disagree in principle. It did NOT manifest in mission 87 (produces steps `[1.0a]`/`[1.4a]` completed). Tracked separately in `…-latent-grade-gate-target-seam-handoff.md` so it is neither missed nor overstated.

## Recommendation
Treat Class A as **resolved-by-availability-fixes**. After `/restart` + re-pend, re-inspect: if any step produces a genuine, reproducible schema rejection on a fresh well-fed attempt (capable model, complete output, valid file on disk, still rejected), THEN open a real schema-gate investigation. Until then, do not edit the gate.
