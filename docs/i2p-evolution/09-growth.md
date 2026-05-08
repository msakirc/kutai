# Z6 — Growth

## Frame

Post-launch iteration. Product ships; data flows back; agent + founder
decide what to build next. Today's i2p has zero of this — phase 14 ends
the workflow, mission goes silent, agent forgets the product. Real
growth requires analytics integration, hypothesis tracking, A/B
infrastructure, prioritization framework, and a feedback loop from
production-signal back into mission-seeding.

This zone closes the long-term loop. Without it, every mission is a
fresh start; with it, the product compounds.

## Current state

- No analytics intake; product analytics lives in Posthog/Mixpanel/etc., never reads back into i2p.
- No hypothesis registry; "this feature increased conversion 12%" — was it? noise? no answer.
- No A/B harness; every feature ships to all users; comparing-against-counterfactual = guessing.
- No prioritization framework; backlog grows unbounded.
- No retention loop; power users + churned users get same product.
- No automated feedback intake from tickets / NPS / reviews / mentions.
- 8.0 implementation_backlog_initialization is one-shot at start of phase 8; never re-prioritized from data.

## Gaps

### Fixable by automation

**A. Analytics-driven mission seed**
- Weekly cron mission: agent reads analytics (events, funnels, retention curves), generates "interesting findings + hypotheses" digest.
- Founder picks → mission to test the hypothesis (feature, copy change, A/B).
- Closes the "product runs in vacuum" gap.

**B. Hypothesis registry**
- Each feature shipped is logged with predicted metric impact + actual measurement window.
- Closes the loop on "did we predict right?"
- Schema:
  ```
  hypotheses
    id            INT PK
    mission_id    INT
    feature       TEXT
    predicted     JSON  {metric, direction, magnitude, window}
    actual        JSON  {metric, direction, magnitude, p_value}
    verdict       TEXT  confirmed | refuted | inconclusive
    measured_at   TIMESTAMP
  ```
- Confirmed hypotheses → reinforce (build similar features); refuted → kill (rollback or remove).

**C. A/B harness**
- Built into deploy adapter: feature flag + variant assignment + metrics segmentation.
- Mission can ship feature behind flag, measure, decide.
- Statistical-significance gate on shipping (auto-roll-back if p > 0.1 against control after N users).
- Variant lifecycle: experiment → winner ships to all → loser auto-rolls-back.

**D. Backlog from signal**
- Tickets / NPS / mentions / errors auto-classified, scored by frequency × revenue-impact, surface top-N as candidate missions.
- Re-prioritized weekly.
- Founder picks; missions spawn.

**E. Cohort & segment awareness**
- Mission requirements gain "for whom" — pricing change for paid users, onboarding tweak for new sign-ups, retention experiment for week-2 churners.
- Recipes gain segmentation hooks.

**F. Pricing experiments**
- Stripe coupon/price-id management via [06-real-world-bridge.md](06-real-world-bridge.md) Stripe adapter.
- A/B harness wires variants into checkout flow.
- Statistical-significance gating.

**G. North-star metric tracking**
- Founder declares north-star at mission 1 (or refines during).
- All mission scoring rolls up to it.
- Weekly digest: north-star trend, contributing features, anomalies.

**H. Retention / churn analysis**
- Cohort retention curves auto-generated.
- Churn-driver analysis (what events precede churn).
- Surface "users who do X within day-1 retain 3x" as recipe candidates.

**I. Feature lifecycle**
- Every feature has a lifecycle: experiment → adopted → mature → sunset.
- Sunset analysis: "this feature is used by 0.3% of users; cost to maintain = $X/month; recommend deprecate."
- Founder decides; agent executes.

### Founder territory (irreducible)

- Strategic prioritization (gut + signal mix).
- Killing pet features.
- Pricing strategy (what story to tell).
- Pivot decisions.
- Feature-to-build vs feature-to-market vs feature-to-deprecate trade-off.
- Ethical product decisions (dark patterns, data use).

## Proposed direction

### Phase A — Analytics intake
- Posthog adapter (read-only): events, funnels, cohorts, retention.
- Standard event taxonomy (page_view, signup, activation_event, conversion, churn).
- Recipe-level instrumentation: every recipe ships with instrumentation hooks.
- Weekly analytics-digest mission (cron-style, see [08-operations.md](08-operations.md)).

### Phase B — Hypothesis registry
- DB schema (hypotheses table).
- Mission template: feature gets hypothesis attached at creation time.
- Measurement window auto-scheduled (e.g. 14 days post-launch); verdict mission runs when window closes.

### Phase C — A/B harness
- Feature-flag adapter (Posthog flags / GrowthBook / custom).
- Variant assignment + segmentation in product.
- Stats integration: significance / confidence intervals; auto-rollback on losers.

### Phase D — Backlog from signal
- Daily ingest: tickets + reviews + mentions + Sentry errors.
- Classifier: bug / feature-request / churn-signal / pricing-feedback / other.
- Scorer: frequency × revenue-impact × strategic-fit.
- Top-N candidates surface in mission Telegram thread weekly.

### Phase E — Cohort/segment awareness
- Mission spec gains `target_segment` field.
- Recipes gain segment-conditional logic (only-for-paid, only-for-new, etc.).
- Analytics digest segments along same lines.

### Phase F — North-star + lifecycle
- Founder declares north-star at first mission.
- Mission scoring weights tied to north-star contribution.
- Feature-lifecycle scheduler: weekly sunset-candidate proposal.

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Analytics digest | weekly summary + hypotheses | reads, picks priorities | n/a |
| Mission spawn | drafts spec + hypothesis | confirms or rejects | full pre-spawn |
| A/B variant ship | ships behind flag | optional pre-launch review | full (auto-rollback) |
| A/B result | computes significance + recommends | confirms ship-winner / rollback | full pre-rollout |
| Sunset proposal | flags low-usage features | decides deprecate / keep / market | full pre-execute |
| Pricing experiment | configures variants | reviews pre-launch | full (auto-rollback) |
| North-star refinement | tracks + alerts on flatness | refines metric definition | full |

## Dependencies

- **Inbound:** [08-operations.md](08-operations.md) — Posthog wiring + cron mission infra. [06-real-world-bridge.md](06-real-world-bridge.md) — Stripe + analytics vendor adapters.
- **Outbound:** [03-build-review-density.md](03-build-review-density.md) — fast iteration depends on review density. [07-humanish-layers.md](07-humanish-layers.md) — investor updates pull from north-star + growth digest.
- **Cross:** [10-cross-cutting.md](10-cross-cutting.md) — cost transparency on hypothesis budgets.

## Open questions

- **Analytics vendor.** Posthog (open-source, generous free tier) vs Mixpanel (mature, expensive) vs Amplitude (enterprise) vs custom (DIY)? (Posthog v1; founder switches if needed.)
- **A/B significance threshold.** p < 0.05 vs Bayesian posterior > 95%? (Bayesian; explainable to non-stat founders.)
- **Hypothesis-window length.** Fixed (14d) or per-metric? (Per-metric default; founder override.)
- **Backlog scoring transparency.** Show formula or hide? (Show; founder needs to trust the prioritization.)
- **Sunset threshold.** Usage <X% × cost >$Y? (Per-product configurable; default 1% usage + any non-zero cost.)
- **Auto-rollback aggressiveness.** Roll back losers immediately or wait for confidence? (Wait for 95% confidence loser; conservative direction.)
- **Multi-armed bandit instead of fixed A/B?** (Fixed A/B v1; bandits later — explainability simpler.)

## Agent task brief

When picking up this doc:
1. Read 00-README + 06-real-world-bridge + 08-operations + this doc.
2. Phase A: Posthog adapter + standard event taxonomy + analytics-digest cron mission.
3. Phase B: hypothesis schema + mission integration + verdict-window scheduling.
4. Phase C: A/B harness via feature-flag adapter + statistical engine + auto-rollback.
5. Phase D: signal-classifier + scorer + weekly backlog surface.
6. Phase E: cohort/segment fields in mission spec + recipes.
7. Phase F: north-star + feature-lifecycle scheduler.
8. Resolve open questions or escalate.
9. Cross-reference outbound to [03-build-review-density.md](03-build-review-density.md), [07-humanish-layers.md](07-humanish-layers.md).
10. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc.
