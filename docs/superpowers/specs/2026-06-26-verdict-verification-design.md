# Design — Reviewer verdict verification (drop confabulated findings before halting)

**Date:** 2026-06-26 · **Status:** implementing · **Handoff:** `docs/handoff/2026-06-26-reviewer-verdict-verification-handoff.md`

Scope chosen by founder: **Tier 1 (deterministic) + Tier 2 (adversarial refuter) + de-example 1.13 rubric.**

## Problem (recap)
A single reviewer LLM, even with full artifacts on a capable model, confabulates
findings (invented verbatim quotes, rubric-example echoes, false "missing section"
claims). Nothing checks the findings against the artifact before a `fail` halts the
mission. Proven on mission 90 `research_review_result.md` — 14 findings, ~10
fabricated (verified against the on-disk artifacts).

## Calibration corpus (real, on disk under `workspace/mission_90/`)
| Check | Claim | Reality | Tier-1 rule |
|---|---|---|---|
| 9 | placeholder `"TODO: define boundaries"` | absent | fabricated-quote → DROP |
| 10 | headline `"completely free forever"` | headline is "…Into a Game…" | fabricated-quote → DROP |
| 11 | "not all six sections" + named_competitors empty | all 6 headers present, 7 competitors | false-absence → DROP |
| 12 | diff lists `"better UI"`,`"more gamification"` | those strings absent | fabricated-quote → DROP |
| 13 | switching-costs stub `"[to be added]"` | real prose present | fabricated-quote → DROP |
| 16 | lessons e.g. `"gamification improves engagement"` | verbatim absent | fabricated-quote → DROP |
| 1/3/6 | market figures lack citations; matrix has no pricing column | REAL (minor) | no quote/section → KEEP → Tier 2 |
| 2/8 | unified-flow contradiction; charter "pricing tiers" incoherence | overstated/confab, no quote | KEEP → Tier 2 |
| 14/15/17 | interview_count/prior-art/verdict-field structured claims | confab, no quote | KEEP → Tier 2 |

## Tier 1 — deterministic grounding (high-precision, drop only when certain)
Pure logic in `packages/mr_roboto/src/mr_roboto/verify_review_verdict.py`; only runs
when the base verdict is `fail` (only `fail` halts). For each issue, resolve its
`target_artifact` to disk content and apply:

- **Rule A · false-absence:** finding asserts a named section/field/array is
  missing/empty (markers: missing|does not contain|lacks|absent|empty|without|no),
  but the named section IS present (line-anchored markdown header, reusing the
  hooks.py header regex) or the named JSON key is present & non-empty → **DROP**.
- **Rule B · fabricated-quote:** finding embeds ≥1 distinctive quoted span
  (`"…"`/`'…'`/smart/backtick, ≥6 chars & multi-word-ish) presented as evidence,
  and NONE of those spans appear (normalized) in the artifact → **DROP**. Quoted
  spans that are themselves a present section/field NAME are references, not
  evidence — ignored for this rule.
- Otherwise **KEEP**; if the kept issue is blocking (blocker/major/critical/high)
  and has no deterministic verdict → mark **tier2 candidate**.

Fail-safe: target_artifact missing/unresolvable, or ANY exception → KEEP (never
drop on doubt, never crash the gate). Re-derive verdict from survivors: no blocking
issue survives → `pass`/`needs_minor_fixes` (do not halt). Every drop is logged
`[verdict-verify] dropped: <check> — <reason>`.

**Artifact resolution:** `target_artifact` is a filename (`competitive_positioning.md`,
`prior_art_report.json`) scattered under subdirs (`.charter/`,`.prd/`,`.research/`,
`.intake/`). Resolver = recursive basename match under `get_mission_workspace(mid)`
(disk = canonical truth), then `ArtifactStore.retrieve(mid, stem)` fallback.

## Tier 2 — adversarial refuter (admitted LLM task, mirrors SP6 critic-gate)
A mechanical must never call the dispatcher. Hooked at the apply layer in
`general_beckman.apply._apply_review_verdict` (the choke point every reviewer fail
flows through), NOT inside the mechanical verifier.

- When verdict is `fail` after Tier 1 AND `tier2_candidates` non-empty AND
  `KUTAI_VERDICT_VERIFY != off`: build ONE admitted overhead child (a single
  batched refuter over all candidates — independent model, adversarial prompt),
  defer routing, park the reviewer. (One child, not N — fits the critic-gate
  single-child substrate; avoids fan-in barriers; still an independent 2nd opinion.)
- Prompt: per candidate finding, "Is this issue actually true of the artifact?
  Quote the exact supporting evidence, or answer UNSUPPORTED. Default to UNSUPPORTED
  on uncertainty." Refuter sees the finding + the artifact content.
- Resume handler (`posthook.verdict_verify.resume`): keep only findings the refuter
  marked SUPPORTED **with a quote that actually appears in the artifact** (re-ground
  the refuter's own quote — fail-closed against a confabulating refuter). Drop the
  rest. Re-derive verdict: survivors with blocking severity → route to producers via
  `route_review_failure`; none survive → complete the reviewer as pass.
- **Fail handling asymmetry:** refuter says UNSUPPORTED/uncertain → DROP the finding
  (design intent: lean away from false halts). Refuter child *fails terminally /
  unparseable* → KEEP the candidates (fall back to current halt behaviour — an
  outage must NOT silently disable the safety halt). Opt-out `KUTAI_VERDICT_VERIFY=off`
  skips Tier 2 entirely (Tier 1 still runs).

## De-example 1.13 rubric (`src/workflows/i2p/i2p_v3.json` step 1.13)
Replace the illustrative parentheticals that get echoed verbatim
(`e.g. headline promises 'free forever'`, placeholder token list `TODO, TBD,
<fill in>, lorem ipsum`) with abstract criteria. Guard test asserts the
removed example strings no longer appear in the 1.13 instruction. Cheap mitigation;
the verification pass is the robust fix — do both.

## Tests
- Tier-1 pure unit tests over the mission-90 corpus: each confab DROPPED, each real
  KEPT; positive fixture (a genuinely-absent section) must SURVIVE (no over-drop);
  fail-safe (unresolvable artifact / exception → keep all).
- Tier-2 resume tests: SUPPORTED-with-real-quote kept → routes; all UNSUPPORTED →
  pass; refuter error → keep+route (fail-closed); opt-out skips.
- De-example guard test.
- Existing `test_verify_review_verdict.py` (pure classifier) stays green unchanged.
