# INDEX — mission-87 remaining failures, categorized for separate sessions

**Date:** 2026-06-21
**Supersedes the triage in:** `docs/handoff/2026-06-21-remaining-content-failures-handoff.md`
**Method:** read-only forensics on the live DB (`file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro`) + static code trace. The 4 availability fixes are committed on local `main` (`8f7baede`, `d2038c67`, `2205f969`, `4680ef9a`) but **restart-gated — NOT loaded**. So every failing row below was produced by a build WITHOUT those fixes.

## Headline correction to the original handoff
The original handoff called **Class A (schema/shape gates) the "DOMINANT remaining class."** The live DB does not support that. Mission 87 has **4 failed tasks total** (187 pending / 115 completed / 5 skipped / 4 failed). Every Class-A step the handoff listed that reached a capable model **completed on retry** (`[1.0a]`, `[1.3]`, `[1.4a]`). The schema errors quoted in the original handoff were **stale last-error strings** left on rows during the overnight availability storm — not reproducible gate rejections. Detail + proof: the Class-A reclassification handoff below.

## The 4 current failed tasks (mission 87)
| Task | Step | status/cat | result | Real defect? | Handoff |
|---|---|---|---|---|---|
| 524380 | `[1.13]` research_quality_review | failed / None | rlen=2544 (real verdict) | **YES — root-caused** | `2026-06-21-classC-reviewer-graph-unavailable-handoff.md` |
| 524377 | `[1.11a]` compliance_overlay | failed / quality | NULL | model-selection/quality | `2026-06-21-1.11a-weak-model-json-loop-handoff.md` |
| 524364 | `[1.0c]` prior_art_synthesize | failed / None | rlen=4743 (truncated) | content/prompt (gate correct) | `2026-06-21-1.0c-prior-art-coverage-and-truncation-handoff.md` |
| 524360 | `[0.6a.draft]` non_goals_draft | failed / **availability** | NULL | availability (restart-gated) | (no new session — see below) |

## Categorized backlog (one investigation session each)

1. **Class C — reviewer-failure routing can NEVER load the workflow graph.** REAL, restart-independent, fully root-caused. Highest priority: it silently defeats `e756355c` (reviewer→producer re-pend) for *every* mission. → `…-classC-reviewer-graph-unavailable-handoff.md`

2. **`[1.11a]` weak-model JSON-emission loop.** A 9B local thinking model (`Qwen3.5-9B-…-thinking`) was selected for an operationally complex analyst step and got stuck in an 8-message "could not be parsed as valid JSON" correction loop, worker_attempts=5, never produced output. Storm-linked (cloud exhausted → weak local fallback). → `…-1.11a-weak-model-json-loop-handoff.md`

3. **`[1.0c]` synthesizer mislabels live products as dead + result truncation.** The `prior_art_min_coverage` Rule-4 gate is working correctly (anti-fabrication); the model marked Habitica/Streaks/Loop `status=dead/dormant` with no Wayback/HN evidence. Two sub-issues: model/prompt content quality, and a `result`-column truncation at 4743 chars. → `…-1.0c-prior-art-coverage-and-truncation-handoff.md`

4. **Class A reclassification (do-not-rechase note).** Documents WHY the schema-gate class is not a live defect, with the gate-plumbing trace (two gates, both sound). Prevents the next session burning time on a phantom. → `…-classA-reclassification-handoff.md`

5. **Latent grade-gate target seam (verify, do not assume).** Static finding: the `grade` posthook gate (`apply.py:1747`) validates `source.get("result")` (the agent's final_answer text), while the engine post-exec gate (`hooks.py:1552+`) validates the **post-materialize file content**. For `produces`-file steps these targets can disagree. It did **not** manifest in mission 87 (produces steps passed both), so it is LATENT — flagged so it is neither missed nor overstated. → `…-latent-grade-gate-target-seam-handoff.md`

## Not a new session (already owned)
- **`[0.6a.draft]` 524360 / availability.** `error_category=availability`, result=NULL. This is the restart-gated availability class the original handoff already fixed (`8f7baede`/`d2038c67`/`2205f969`/`4680ef9a`). Its `missing sections: ['Non-goals']` schema string is a stale artifact, not a content rejection. Action: `/restart`, re-pend, confirm it clears. Tracked by the existing availability handoffs + memory.

## Deploy/verify order (unchanged from original)
`/restart` (loads the 4 fixes) → re-pend mission 87 → re-inspect on FRESH attempts → then work the categorized backlog. Class C is the only item that is fully diagnosable and worth fixing **before** restart, because it is independent of model availability.
