# Z1 — Original additions (Claude, 2026-05-09)

Sibling to plan-v3. Items below not in v1/v2/v3 nor in 01-pre-code.md original
proposal list. Each is a gap I think the three prior planning passes missed
because they all started from the doc's six proposals and stayed inside that
frame.

---

## A1. Reverse-pitch / press-release artifact at phase 0

**Pattern:** Amazon "working backwards." Before any persona/JTBD/feature work,
founder writes the launch press release: headline, sub-head, customer quote,
quote from founder, FAQ. One page.

**Why all three plans missed it:** They optimized "extract more signal from
the founder pitch." This optimizes "force the founder to commit to an
outcome before phase 1." Different leverage point.

**Wiring:** new step `0.0z reverse_pitch_draft` (founder-authored, mechanical
clarify-shape). Output `reverse_pitch.md`. Step `0.1 idea_brief` *consumes*
it. Reviewer at 1.13 (`research_quality_review`) gains criterion: "does the
research support the press-release claims, or contradict them?"

**Anti-pattern caught:** founder pitches "AI tool for X." Cannot write the
customer quote. Reveals the product has no felt-need framing. Mission
self-aborts at $0.

**Cost:** human time, ~30 min. Zero LLM cost (mechanical pass-through).

**Acceptance:** mission with empty/template press release blocked at 0.1.
Mission with concrete press release sees 1.3 competitive_landscape and
1.11 regulatory_research narrow to the claims in the release.

---

## A2. Non-goals artifact + downstream contradiction check

**Pattern:** Real specs name what the product refuses to do. i2p has only
positive specs (features, requirements, value-props). Negatives are
inferable noise.

**Why missed:** The three plans treat phase 0 as "collect ambition" — they
don't treat refusal as first-class signal.

**Wiring:** new artifact `non_goals.md` at phase 0 (`0.6 non_goals_lock`,
mechanical clarify, 3-7 bullets minimum). Reviewer at 3.11 gains: "does any
listed FR/BR contradict a non-goal? Reject if yes." Reviewer at 4.16
re-checks against architectural choices (e.g., non-goal = "no real-time
features" → reject WebSocket in stack ADR).

**Why this beats P3 ADRs alone:** ADRs justify chosen options against
considered alternatives. Non-goals justify *categories* of options never
considered. Different concern.

**Cost:** founder ~10 min at intake; reviewer +1 LLM call at 3.11/4.16.

**Acceptance:** when a downstream phase introduces a feature that maps to a
non-goal, the conflict surfaces at the next reviewer gate, not at phase 12+
where it's expensive.

---

## A3. Boring-tech bias as default in stack ADRs

**Pattern:** P3 introduces ADRs with options + rationale. But agents trained
on hype-curated text default to novel choices (NextJS, edge runtimes, vector
DBs, LangChain). Founder gets phase-7 surprise: "why am I debugging Bun?"

**Why missed:** v3's P3 schema has `options_considered` but no bias on
selection. Treats all alternatives as equally weighted.

**Wiring:** add `tech_maturity_score` field to each option in stack ADRs.
Score from `tech_radar.yaml` (curated; recipes-overlap with Z2). Selection
rule embedded in 4.2 reviewer prompt: "Reject if the chosen option's
maturity_score is below the next-best by >2 unless ADR's `novelty_benefit`
field declares a falsifiable advantage worth the maturity cost."

**Anti-pattern caught:** agent picks Drizzle over Prisma because training
celebrates Drizzle. Reviewer rejects unless ADR names a concrete advantage
(e.g., "Prisma's edge runtime cost > $X/mo at 10k users") that's
verifiable.

**Cost:** small static file `packages/recipes/src/recipes/tech_radar.yaml`
with ~50 entries; reviewer prompt addition only.

**Acceptance:** ADR for stack pick that chooses non-boring option without
populated `novelty_benefit` is rejected at 4.16.

---

## A4. Interview-script generator as the bridge between "no evidence" and P1 intake

**Pattern:** P1 (structured intake) assumes founder has interview transcripts
to upload. They usually don't. Result: founder skips P1, evidence_refs all
populate as `agent_inference`, P1 degrades to a no-op.

**Why missed:** v3 covers the *receiving* side of evidence (intake +
extraction + dedup). Doesn't address that the artifact often doesn't exist
yet.

**Wiring:** new step `0.0c interview_script_generation` (analyst LLM, runs
after `0.3 assumption_identification`). Inputs: assumption list. Output:
`interview_script.md` — 5-7 questions targeting the highest-`risk_if_wrong`
assumptions, scripted neutrally (no leading questions). Founder sees it,
runs interviews, drops responses into evidence intake.

**Acceptance:** any `private_beta+` mission with `interview_count==0`
across all personas gets a soft warning at 1.13 reviewer; `public_launch`
hard-blocks.

**Cost:** one analyst LLM call per mission (~2k tokens). Saves: synthetic
persona missions that crash phase 7+ when first real user hits the product.

---

## A5. Founder attention budget as a tracked resource

**Pattern:** v3 R7 names founder fatigue as a risk; v3 N1 proposes batch
clarify. But neither tracks it as a finite budget against an explicit pool.

**Why missed:** Treating founder time as "minimize" optimizes wrong. Treat
as resource: founder declares budget at z0 (e.g., "5 hours over this
mission's spec phase"); every clarify/upload-prompt/review-ack debits it;
agent surfaces remaining budget per phase; when low, defers low-priority
clarifies to a "deferred questions log" rather than firing them.

**Why beats v3 N1 batching alone:** Batching helps phase-0 storm but doesn't
help cumulative drain across phases 0-6. Budget makes the trade explicit
to the founder.

**Wiring:** new column `missions.founder_attention_budget_minutes` (set at
z0). New table `founder_attention_log` (per debit: step_id, action, minutes,
ts). Mechanical action `attention_check(reserve_minutes=N)` returns
`{remaining, ok}`; clarify steps gate on it. Surface as `/budget` Telegram
command.

**Cost:** schema bump + one mechanical action; no LLM cost.

**Acceptance:** founder gets accurate "you have 47 minutes of attention
budget left this mission" at any point. Low-priority clarifies (e.g.,
"prefer light or dark mode for default?") move to deferred log when
budget < 30 min.

---

## A6. Premortem step before lock

**Pattern:** Phase 6 ends with `project_plan_review` (6.6) → spec locks.
Add a step at 6.5z `failure_premortem` — agent simulates: "it is 2027-05-09
and this product failed. Write 3 plausible obituaries (technical, market,
founder)." Founder ranks plausibility; high-plausibility scenarios become
new entries in assumption_identification + new monitoring rules.

**Why missed:** v3 P4 (failure-mode per requirement) is bottom-up
falsifiability. This is top-down: imagine the whole thing dying. Different
failure modes surface (e.g., "founder gets bored" rarely shows in
per-feature falsification, often shows in obituary).

**Wiring:** new step `6.5z failure_premortem` (analyst LLM). Output
`premortem.md`. Reviewer at 6.6 gains: "if premortem flagged X with
plausibility >=4/5 and there's no monitoring rule mapping to X, reject."

**Anti-pattern caught:** "founder loses interest in 4 weeks" is the most
common startup death. Surfaces here. Founder either commits explicitly or
chooses a smaller scope.

**Cost:** ~1 analyst call per mission (~3k tokens). Saves: silent mission
abandonment.

---

## A7. Idea fingerprint + cross-mission dedup at intake

**Pattern:** Founder iterates ideas. Mission 12 explores "X for Y," kills it
at phase 4. Mission 19, six weeks later, founder pitches the same X-for-Y
in different words. Today: full mission re-runs. No memory of the prior
verdict.

**Why missed:** P9 (cross-mission inheritance) inherits *successful*
patterns; doesn't surface *prior failures* to the founder before re-running.

**Wiring:** at `0.1 idea_brief`, vector-search the embedding against past
mission `idea_brief` artifacts (ChromaDB collection `mission_ideas`,
embedding via existing `multilingual-e5-base`). If similarity > 0.85 to
prior mission, surface: "this looks like mission #12 which you killed at
phase 4 with reason: '<final_status_note>'. Continue / branch / abort?"

**Why this is brutal:** founders need the friction. Saves real days.

**Cost:** one embedding + one Chroma query at 0.1; ~$0 in LLM tokens.
Storage: idea embeddings already small.

**Acceptance:** mission similarity-matched to a killed prior shows the
prior verdict before phase 0.5 clarification fires.

---

## A8. Cost ladder in tech-stack ADR

**Pattern:** P3 ADRs include `reversal_cost` (one number). Doesn't capture
that the tech choice's *operating cost* changes by orders of magnitude with
scale.

**Why missed:** ADR template borrowed from Nygard's enterprise pattern
where ops cost is invisible. For founder/solo missions, ops cost dominates.

**Wiring:** stack-related ADRs (4.2, 4.4, 4.6, 4.8, 4.9) gain
`monthly_cost_curve: {at_mvp: $X, at_1k_users: $Y, at_100k_users: $Z}`
field. Compliance-fingerprint compute uses z0's `cost_ceiling_monthly_usd`.
Reviewer at 4.16 rejects ADR if `cost_at_target_users > cost_ceiling`
without `cost_mitigation_plan`.

**Cost:** schema field + reviewer prompt addition. Estimates can be agent
heuristics; precision matters less than the order-of-magnitude flag.

**Acceptance:** Vercel-Pro-at-100k-users ADR shows $4k/mo against $200/mo
ceiling, gets rejected with explicit mitigation needed.

---

## Sequencing for these additions

These are additions, not replacements. Insert into v3's sequence:

| After | New | Reason |
|---|---|---|
| P7 | A1 (reverse-pitch) | Cheapest, highest leverage; rejection rate measurable. |
| P4 | A2 (non-goals) | Reuses falsification reviewer surface. |
| P3 | A3 (boring-tech bias), A8 (cost ladder) | Both extend ADR schema; land in same merge. |
| P1 | A4 (interview-script gen) | P1's preceding step; doesn't ship without it. |
| P6 | A5 (attention budget) | Compliance + budget both touch z0 contract. |
| P9 | A7 (idea dedup) | Inheritance infra reused. |
| (parallel) | A6 (premortem) | Independent, late-phase. |

## Risks specific to these additions

- **A1 reverse-pitch:** founder ego friction. Some founders refuse. Mitigation: `prototype` ambition can skip with `acknowledgement: "I am not building for users"`.
- **A3 boring-tech:** the `tech_radar.yaml` curation becomes a source of contention. Mitigation: per-mission override allowed via founder ADR vote that creates `_local_tech_radar_override.yaml`.
- **A5 attention budget:** founder under-estimates own budget; runs out mid-spec. Mitigation: agent suggests budget based on mission ambition tier (prototype: 2h, private_beta: 5h, public_launch: 10h+).
- **A7 idea dedup:** false positives infuriate founder. Mitigation: surface verdict, never auto-abort. 0.85 similarity threshold tuned on real corpus once available.

## What I notice about the v1→v2→v3 trajectory

All three plans optimize *within* the doc's six proposals. None step back
to ask: is the proposal frame itself right? My additions break that frame:
A1/A2/A6 add new artifacts the founder writes; A4/A7 add new agent
behaviors triggered by absence of evidence (current proposals only react
to presence); A3/A8 inject opinionated bias into ADR selection (current
proposals make ADRs neutral). These represent different design axes than
the original six.

Likely there's a v4 worth of further additions if I push longer; this is
the first pass at "what didn't the framing capture."
