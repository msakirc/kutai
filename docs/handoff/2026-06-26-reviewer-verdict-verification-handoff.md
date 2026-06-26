# Handoff — Reviewer verdict verification (drop confabulated findings before halting)

**Date:** 2026-06-26
**Status:** Design + evidence ready; NOT started. This is the real fix for unreliable review halts.
**Why now:** the ctx/model/prompt fixes (separate handoff) are DONE and verified — the reviewer now gets full artifacts on a capable model — yet it STILL confabulates findings. Input fixes can't cure a single LLM making things up. The verdict itself must be verified.

## The problem (proven on mission 90 [1.13] task 567399)
Re-run with everything fixed: model `cerebras/gpt-oss-120b`, prompt 58k chars, **all 6 input artifacts full, zero truncation**, one-shot (no loop), reviewer.yaml carrying an explicit anti-hallucination evidence rule. The verdict was STILL `fail` with ~5 of 9 findings fabricated — several **echo the rubric's own illustrative examples as if observed**:

| Finding | Reality (checked against the artifact on disk) |
|---|---|
| Check 9: "placeholder text such as **'TODO: define boundaries'**" | FALSE — no such text. All 5 charter solutions have populated Boundaries + Guiding principles. Rubric says "Reject if … placeholder text" → model invented one. |
| Check 10: "headline promises **'completely free forever'**" | FALSE — headline is "Turns Your Daily Tasks and Errands Into a Game". Rubric example is literally "e.g. headline promises 'free forever'" → echoed as a finding. |
| Check 11: "does not contain all six sections" | FALSE — competitive_positioning.md has all six (Landscape/Value Thesis/Strengths-Weaknesses/Our Differentiators/Switching Costs/Notes). |
| Check 8: "contradictory statements about … pricing tiers" | FALSE — the charter has no pricing tiers. Confabulated. |
| Check 2: "no competitor unified flow … yet lists Todoist/TickTick" | OVERSTATED — a mild cross-doc wording tension, not a contradiction. |
| Check 1 / 3 / 6: market figures lack source citations; feature matrix has no pricing column | **REAL** (minor) — the only genuine gaps. |

Root: a single LLM, even capable with full artifacts, **confabulates findings (invented quotes, rubric-example echo) and nothing checks them against the artifact** before the halt blocks the mission. `verify_review_verdict` is mechanical — it only classifies pass/fail, it does not validate the findings.

## The fix — a verdict-verification pass
After the reviewer emits findings, **verify each finding against its `target_artifact`; drop any whose cited evidence is not actually present/absent as claimed.** Re-derive the verdict from the surviving findings: if no blocker/major survives → `pass`/`needs_minor_fixes` (do not halt).

### Where it hooks
- `packages/mr_roboto/src/mr_roboto/verify_review_verdict.py` — currently parses `verdict`/`status` + `issues`/`findings` and classifies pass/fail. Add a **grounding filter** on `issues` before classifying. (This is the choke point — every reviewer step, ~20 of them: 1.13, 3.8, 3.11, 4.16, 6.6, 10.x, 11.5, 12.3/12.5, 14.2, etc. — flows through it.)
- Routing of a fail is `general_beckman` `route_review_failure` (re-pends producers). With confabulated findings dropped, fewer false halts reach it.

### Verification tiers (cheap → strong; do the cheap one first)
1. **Deterministic grounding (catches most confabulation, ~free):**
   - "missing section/field X" findings → load the `target_artifact` from disk (`workspace/mission_<id>/...` via the artifact store) and check whether X *is in fact present* (heading/field). Present → finding is false → DROP. (Kills Check 9 "missing Boundaries", Check 11 "missing sections".)
   - "quote/contradiction" findings that embed a quoted string → check the quoted span actually appears in the artifact. Absent → confabulated → DROP. (Kills Check 10 "completely free forever".)
   - Reuse the existing helpers: markdown section presence (`validate_artifact_schema` markdown branch, line-anchored), `_empty_exemption_granted`/`is_empty_scope_artifact` for empty-scope, `coulson.grounding`.
2. **Adversarial 2nd-opinion (for findings with no verbatim quote — e.g. "lacks citations"):**
   - A second, independent admitted task (NOT inline) that, per finding, is asked to REFUTE: "Is this issue actually true of the artifact? Quote the evidence or say UNSUPPORTED." Default-to-refuted on uncertainty. Drop findings the refuter can't support. (Mirror the SP6 critic-gate pattern: admitted task, fail-closed, `KUTAI_*` opt-out.)
   - Keep it bounded — only run tier 2 on findings that survive tier 1 but lack a checkable quote.

### Re-derive verdict
- `blocker`/`major` findings that survive → keep `fail` (route to producers).
- Only `minor` (or none) survive → `needs_minor_fixes`/`pass`. Do NOT halt the mission on confabulated majors.
- Log every dropped finding with the reason (`[verdict-verify] dropped: <finding> — evidence not found`).

## Test corpus (use the 567399 verdict above)
- Real findings (citations, pricing) MUST survive. Confabulated ones (TODO placeholder, free-forever, missing-sections, pricing-tiers) MUST be dropped. Build fixtures from `workspace/mission_90/` artifacts + the 9-finding verdict; assert the filtered verdict keeps only Check 1/3/6 and downgrades to `needs_minor_fixes`.
- Don't over-drop: a finding that IS grounded (a real missing section in a genuinely-incomplete artifact) must survive — add a positive fixture.

## Also consider (smaller, complementary)
- **De-example the 1.13 rubric** (`src/workflows/i2p/i2p_v3.json` step 1.13 instruction): the parenthetical examples ("e.g. headline promises 'free forever'") are what get echoed. Replacing illustrative examples with abstract criteria reduces echo at the source. Cheap mitigation, but the verification pass is the robust fix — do both.

## Context
- Reviewer profile (evidence rule already present, but a prompt rule alone doesn't stop confabulation): `packages/finch/src/finch/profiles/reviewer.yaml`.
- Verdict shape: `{verdict: pass|fail|needs_minor_fixes, issues: [{target_artifact, severity (blocker|major|minor), problem}]}`.
- The earlier human audit of this exact halt: `docs/handoff/2026-06-25-*` review-halt notes + memory `project_reviewer_summary_starvation_20260626`.
