# Z9 — Growth (v2)

**Supersedes:** [09-growth.md](09-growth.md) (v1, 2026-05-08)
**Date:** 2026-05-15 (rev. after pass-2 deep-dive)
**Status:** plan; T1 dispatch follows

## Why v2

v1 framed Z9 as greenfield. Two investigation passes show: ~70% of the substrate already shipped via [[project_z2_t5_complete]] / [[project_z6_complete]] / [[project_z8_complete]] / [[project_z3_closed_20260512]] / [[project_z10_complete]]. Z9 is mostly a **wiring + READ-side** zone, not a build-from-scratch zone. v2 enumerates exactly which existing surfaces Z9 claims.

## Pass-2 surface map (what's actually there)

### i2p workflow phases (full enumeration)

**Phase 13 — Production / Launch readiness (19 steps).** Includes `13.4 analytics_integration` already in the workflow as a stub. Z9 implements this step.

**Phase 14 — Launch (9 steps).** Includes `14.4 launch_monitoring_and_stability`, `14.9 launch_retrospective`.

**Phase 15 — Post-Launch Operations & Iteration (17 steps).**

| Step | Owner |
|---|---|
| 15.1 post_launch_monitoring | Z8 ops |
| 15.2 feedback_system | **Z9 — T3** |
| 15.3 daily_health_report | Z8 ops + **Z9 telemetry hook** |
| 15.4 incident_response | Z8 ops |
| 15.5 weekly_analytics_and_feedback | **Z9 — T2** |
| 15.6 weekly_summary | **Z9 — T2** |
| 15.7 bug_triage_and_fixing | **Z9 — T3 (backlog seeding)** |
| 15.8 feature_request_prioritization | **Z9 — T3 (backlog seeding)** |
| 15.9 maintenance_operations | Z8 ops |
| 15.10 infrastructure_cost_and_security | Z8 ops + **Z9 cost feedback** |
| 15.11 record_demo / 15.12 verify_demo_artifact | Z7 marketing |
| 15.13 new_feature_cycle | **Z9 — T4 (hypothesis loop close)** |
| 15.14 release_cycle | Z6 deploy |
| 15.15 technical_debt_tracking | **Z9 — T5 (lifecycle/sunset)** |
| 15.16 roadmap_update | **Z9 — T5 (north-star sync)** |
| 15.17 mission_deliverable_bundle | wrap |

Z9 claims **9 of 17 phase-15 steps** + step 13.4. None require new phase definitions.

### success_metrics artifact (step 2.9, not 2.5)

```
success_metrics {
  aarrr_metrics: [{name, formula, data_source, target_value, measurement_frequency}]
  north_star_metric: {name, justification}
}
```

**Wiring gap:** created at 2.9, never re-read in phase 8+. Z9 fixes the READ side via new `inject_north_star` verb + extending mission scoring.

### Existing data substrate

| Table | Status | Z9 use |
|---|---|---|
| `model_pick_log` | shipped | template for hypothesis_log shape |
| `recipe_pin_log` | shipped (Z2 T6F) | bind recipe→outcome for sunset scoring |
| `mission_lessons` | shipped (Z2 T4) | mirror verdict-side of hypotheses |
| `scheduled_tasks` + `registry_events` | shipped (Z8) | cron + event log |
| `cost_budgets` (per-scope) | shipped | hypothesis cost gating per mission |
| `model_call_tokens` | shipped | per-call cost; Z9 reads aggregates |
| `hypotheses` / `experiment_variants` / `growth_events` | **absent** | T1 schema |

### POST_HOOK_REGISTRY (31 kinds)

Existing kinds Z9 leverages: `find_similar_missions`, `surface_prior_mission_hints`, `prior_art_min_coverage`, `critic_gate`, `integration_review`, `verify_falsification_present`. New kinds Z9 adds: `metric_emit`, `record_hypothesis`, `verdict_check`, `backlog_score_recompute`, `sunset_score_recompute`.

### mission_cron action dispatch

`mission_cron.arm(mission_id, action: str, interval_seconds: int)` — action is free-form string routed to mechanical context payload. **No registry change needed; Z9 just defines new action strings.**

### vendor_call (Z6) + IntegrationRegistry

`vendor_call(task) -> {ok, result, service, action}` already supports stripe/sentry/github. Z9 adds posthog/intercom/zendesk via new IntegrationRegistry entries + adds `mock_mode` for offline tests.

### Reversibility (Z10)

`packages/mr_roboto/src/mr_roboto/reversibility.py::VERB_REVERSIBILITY`. Every Z9 new verb gets an explicit entry.

## Stale claims in v1 (corrected, exhaustive)

| # | v1 said | Reality | v2 response |
|---|---|---|---|
| 1 | "phase 14 ends the workflow" | Phase 15 exists with 17 stubs; Phase 13.4 has analytics_integration stub | Z9 implements existing stubs, claims 10 of them; invents no new phase |
| 2 | "step ~2.5 success_metrics" | Actually step 2.9 | corrected refs throughout |
| 3 | "no analytics" | true outward; self-analytics (model_pick_log/recipe_pin_log/mission_lessons) is the pattern to mirror | mirror outward + add "internal health" section in digest |
| 4 | "no hypothesis registry" | true — but mission_lessons is verdict store | narrow hypotheses table for prediction-side; verdicts mirror to mission_lessons (`source_kind="hypothesis_verdict"`, dedup_key=feature+metric) |
| 5 | "Posthog adapter needed" | Z6 vendor_call + IntegrationRegistry ready | one IntegrationRegistry entry + recipe template; no new infra |
| 6 | "weekly cron infra needed" | Z8 shipped mission_cron + scheduled_tasks; action dispatch is free-form | one `mission_cron.arm("analytics_digest", 604800)` call |
| 7 | "A/B harness Phase C" | premature without trust in digest loop + cohort framing | move A/B to **T5**, gate behind founder /experiment_enable opt-in |
| 8 | "cost transparency cross to Z10" | Z3 cost_band already buckets cheap/moderate/heavy in posthooks; Z10 done | thread cost_band × north_star_relevance into hypothesis scoring NOW |
| 9 | "no cohort awareness" | mission.context is free-form JSON — no migration to add target_segment | additive only; warn-only validator |
| 10 | "feature lifecycle absent" | true — but recipe_pin_log + model_pick_log are the data sources for usage analysis | sunset scorer queries existing logs, no new tracking |
| 11 | "8.0 backlog one-shot" | true; DLQ /dead /revive is manual | T3 backlog-from-signal cron rewrites at every digest cycle |
| 12 | (missing) | Telegram /digest already exists | T2 **extends** /digest with growth section; no command collision |
| 13 | (missing) | Sentry webhook intake shipped (Z8) | T3 signal classifier subscribes to same webhook stream |
| 14 | (missing) | PII redaction is secrets-only (`src/security/sensitivity.redact_secrets()`) | T3 extends redaction for user-PII before signal stored |
| 15 | (missing) | Multi-product / per-mission posthog routing not in arch | v1 = single product per workspace; multi-product deferred |
| 16 | (missing) | mission_brief has no `business_model` field | T5 adds B2B vs B2C branching to mission_brief + scorer |
| 17 | (missing) | IntegrationRegistry has no `mock_mode` | T1 adds, so all later tiers stay offline-testable |
| 18 | (missing) | success_metrics created at 2.9 then never read | **THE central Z9 wiring gap** — inject_north_star verb closes it |
| 19 | (missing) | recipe.yaml schema unvalidated (additive-safe) | T4 adds `emits_events` + `segment_predicate` fields without migration risk |
| 20 | (missing) | mission.context is free-form JSON | T4 adds `target_segment`, `north_star_relevance`, `hypothesis_id`, `business_model` as keys (no schema migration) |

## Tier plan (5 tiers; T1A canonical-first per [[feedback_canonical_first_for_tier3plus]])

### T1 — Foundation (schema + integration + reversibility)

**T1A — Schema (canonical-first; land before T1B/T1C).**
- `hypotheses` table: `id, mission_id FK, feature, predicted_json, actual_json, verdict (confirmed|refuted|inconclusive|pending), window_seconds, measured_at, dedup_key, suppressed_until, created_at`
- `experiment_variants` table: `id, mission_id FK, hypothesis_id FK, variant_name, assignment_rule, status (active|winner|loser|stopped), shipped_at, retired_at`
- `growth_events` table: `id, mission_id FK, kind, properties_json, segment, occurred_at` (separate from `registry_events` to keep Z8 oncall queries clean)
- mission.context extensions (free-form, no migration): `north_star_id`, `target_segment`, `north_star_relevance`, `hypothesis_id`, `business_model`
- Migration applied via `migration_apply` recipe (Z3 T3A); pattern: idempotent CREATE TABLE IF NOT EXISTS

**T1B — IntegrationRegistry: posthog + intercom + zendesk + mock_mode.**
- Adapter configs in `src/integrations/configs/{posthog,intercom,zendesk}.json`
- Env-var pattern: `KUTAI_<PROVIDER>_API_KEY` resolved via existing pattern
- `IntegrationRegistry.mock_mode` flag returns deterministic fake responses; default off, set on in tests
- Reuses `vendor_call` verb — no new mechanical needed

**T1C — Reversibility + new verbs.**
- New verbs registered in `VERB_REVERSIBILITY`:
  - `inject_north_star` → full
  - `emit_metric` → full
  - `record_hypothesis` → full
  - `record_verdict` → full
  - `suppress_hypothesis` → full
  - `assign_variant` → partial
  - `retire_variant` → partial
  - `score_backlog` → full
  - `score_sunset` → full

**T1D — Telegram founder-track stubs.**
- New cmd handlers (stubs returning "coming soon" until later tiers): `/northstar`, `/hypothesis`, `/backlog`, `/sunset`, `/experiment`, `/approve`. Reserves command surface; prevents collision.

Founder track: founder approves migration on first run (Z10 reversibility gate); after that, T1 invisible.
Reversibility: T1A migration full (rollback table drops); T1B configs full (env var removal); T1C verb registration full.

### T2 — Phase 13.4 analytics_integration + 15.5 weekly digest

**T2A — Phase 13.4 implementer (launch-time instrumentation).**
- Agent reads success_metrics artifact (step 2.9) at phase 13.4
- Generates platform-specific instrumentation: JS shim (web) + Python shim (backend) emitting standardized events from `aarrr_metrics` field
- Recipe `analytics_instrumentation_v1`: ships a `posthog-js` + `posthog-python` template + emit helper (`track_event(name, properties)`); auto-attaches segment + mission_id metadata
- Post-hook `verify_artifacts` checks instrumentation files committed

**T2B — Phase 15.5 weekly_analytics_and_feedback cron.**
- On Phase 14 launch completion, mission arms `mission_cron.arm(mission_id, "analytics_digest", 604800)` — idempotent
- Cron action `analytics_digest` routed to a new mechanical executor that:
  1. vendor_call posthog: events, funnels, retention curves, cohort tables for last 7 days
  2. queries `mission_lessons`, `hypotheses` (pending verdicts), `growth_events`, `model_pick_log` aggregates
  3. spawns digest synthesis agent (LLM-driven, MAIN_WORK lane)

**T2C — Digest synthesis agent + Telegram surface.**
- Reads pull from T2B, drafts Telegram-ready digest with sections:
  - **North-star trend** (read success_metrics.north_star_metric, plot delta vs last week)
  - **Funnel + retention** (AARRR cohort movement)
  - **Hypothesis verdicts ready to record** (T3 verdict-side preview)
  - **Top-N candidate missions** (T3 backlog scoring preview)
  - **Internal health** (self-analytics aggregates — model pick, recipe pick, retry rates)
- Extends existing `/digest` Telegram cmd (no new command — append new section); founder /digest_now on demand

**T2D — Anti-pattern detector (built into synthesis agent).**
- **Vanity metric guard:** if north_star_metric.name matches known vanity patterns (DAU/MAU absolute, page_views, signups absolute) → digest warns "consider tying to revenue or retention"
- **Engagement vampire:** high event_count + flat retention → flag in digest
- **Insufficient N:** A/B experiments with <100 daily-active samples → warn before stats compute

Founder track: founder reads `/digest`, picks priorities; agent never auto-spawns from digest without founder /approve. Reversibility: digest is read-only; T2B cron disarm is full (single delete from scheduled_tasks).

### T3 — Phase 15.2 feedback_system + 15.7/15.8 signal → backlog

**T3A — Signal intake (extend webhook_listener).**
- Webhook adapters for posthog (events), intercom (tickets), zendesk (tickets), sentry (errors already wired Z8). HMAC verification via existing `webhook_signing.py` pattern
- Per-mission posthog API key routed via mission.context.posthog_project_id; mission lookup at intake time
- PII redaction: extend `src/security/sensitivity.py` with `redact_user_pii()` (emails, addresses, IPs); applied before payload stored to `growth_events`

**T3B — Classifier.**
- Mechanical or low-cost LLM classifier (configurable) — labels each signal: `bug | feature_request | churn_signal | pricing_feedback | praise | spam`
- Pattern mirrors [[dogru_mu_samet]] (quality assessment) shape; reuses grader-style structured verdict
- Maps each signal to `lessons_domain` via recipe registry — anchors prioritization

**T3C — Scorer + backlog seeding.**
- Score formula (explicit, shown in digest per v1 open question resolved):
  ```
  score = freq × revenue_impact × north_star_relevance × age_decay / cost_band_weight
  ```
- `score_backlog` verb computes top-N candidates daily; results written to `growth_events` with `kind="backlog_candidate"`
- Founder /backlog command lists top-N; founder /approve <id> spawns mission (does NOT auto-spawn)

**T3D — DLQ feedback hook.**
- /dead DLQ scan: cron `dlq_signal_review` mines failed mission patterns; high-frequency failures surface as `growth_events` with `kind="dlq_pattern"`
- Closes loop: DLQ no longer dead-end manual review

Founder track: founder /approve gates ALL backlog→mission seeding; never auto-spawn. Reversibility: classifier writes are full-reversible (delete row); /approve→mission spawn pre-spawn reversible (no commit yet).

### T4 — Hypothesis verdict loop + 15.13 new_feature_cycle

**T4A — record_hypothesis verb (mission-spec time).**
- At mission spec finalization (Phase 7 review), for **every** Phase 8+ mission (founder decision: all-mission scope, not feature-only):
  - Agent extracts predicted metric impact from spec (e.g. "checkout conversion +12%", "p95 latency -200ms", "error rate -50%")
  - Window resolved via per-metric default table: activation=7d / retention=30d / revenue=14d / referral=14d / acquisition=7d / latency=3d / error_rate=3d; founder /override per hypothesis
  - Writes `hypotheses` row with `predicted_json={metric, direction, magnitude}`, `window_seconds`, `verdict="pending"`, `measured_at=NULL`
  - dedup_key = feature_slug + metric_name (refuses if suppressed_until > now)
  - Mission classes without measurable impact (pure refactor, doc-only) flagged but not blocked; agent prompts founder if confused

**T4B — inject_north_star verb (THE central wiring gap from pass-2).**
- Mirrors `inject_lessons` Z2 T4 pattern exactly
- Reads success_metrics artifact from mission.context
- Injects `north_star_metric` + `aarrr_metrics` block into agent prompt context for Phase 8+ steps
- Auto-wires on any Phase 8+ step that scores features (recipe choice, mission ranking)

**T4C — Verdict scheduler.**
- On hypothesis insert: `mission_cron.arm(mission_id, "verdict_check::<hyp_id>", interval=...)` where first fire = measured_at_default; scheduled_tasks honors one-shot via `enabled=False` after fire
- Verdict mission runs at fire time; reuses ONGOING lane

**T4D — Verdict recorder.**
- Mechanical verb `record_verdict`: pulls metric via vendor_call posthog, compares vs `predicted_json`
- Bayesian posterior > 95% → `verdict="confirmed"` or `"refuted"`; otherwise `"inconclusive"`
- Writes to `hypotheses.actual_json` + `hypotheses.verdict`
- Mirrors refuted/inconclusive to `mission_lessons` with `source_kind="hypothesis_verdict"`, `dedup_key=feature+metric` (idempotent — re-runs deduped)
- Refuted → set `suppressed_until = now + 90d` on the feature/metric pair

**T4E — Phase 15.13 implementer (new_feature_cycle close).**
- Step `15.13 new_feature_cycle` becomes "verdict review + next-iteration spawn":
  - Reads `hypotheses` + `mission_lessons` for this mission
  - Founder reviews via Telegram /hypothesis; /confirm → mission seed for follow-up feature
  - Confirmed verdicts reinforce: `model_pick_log` weight bump for the recipes that built winning features (feeds [[project_selection_intelligence_phase1_20260417]])

Founder track: /hypothesis lists pending + recent verdicts; /confirm <id> gates next-iteration spawn. Reversibility: record_verdict is full (re-runnable); suppression is full (suppressed_until expires naturally).

### T5 — Cohort + lifecycle + (optional) A/B

**T5A — target_segment + segment_predicate.**
- mission.context.target_segment (free-form: "paid_users" | "new_signups" | "week2_churners" | "any")
- recipe.yaml gains `segment_predicate` field; `metric_emit` post-hook respects it
- `mission_brief` validator warns (not blocks) when Phase 8+ mission has no target_segment

**T5B — business_model branching.**
- mission_brief gains `business_model: "b2b" | "b2c" | "hybrid"` field (asked at Phase 2 spec)
- success_metrics + scorer parameterized: B2B north-stars tilt MRR/churn/seats; B2C tilt activation/retention/referral
- Default = b2c if unset

**T5C — Phase 15.15 technical_debt_tracking + 15.16 roadmap_update (sunset + north-star sync).**
- Sunset scorer (cron `sunset_score_recompute` weekly):
  - Query `recipe_pin_log` + `growth_events` for feature usage in last 30d
  - Score: `feature_usage_rate × maintenance_cost_band`; below threshold → surface as sunset candidate
  - Telegram /sunset lists candidates; founder /approve_sunset → mission spawn for deprecation
- Roadmap sync: weekly read of success_metrics; if north-star definition changed, prompt founder for refinement

**T5D — A/B harness (default-on once T5 lands; founder /experiment_disable per mission to opt out).**
- Every Phase 8+ feature ships behind flag automatically; mission spec gains `use_ab: true` by default
- `assign_variant` verb registered in mr_roboto; integrates with posthog flags or GrowthBook (env-var routed; decision deferred to T5D dispatch)
- `experiment_variants` schema (T1) populated; metrics_emit auto-attaches variant tag
- Bayesian stats engine (re-uses T4 record_verdict scaffolding): 95% posterior gates "winner" call
- **Insufficient-N guard:** if daily-active <100, A/B disabled for that mission automatically (anti-pattern detector from T2D); 100% rollout, hypothesis still recorded
- Auto-rollback with founder gate (irreversible_money rule from [[project_z10_complete]]); founder /experiment_ship and /experiment_rollback commands
- /experiment_disable <mission_id> opts a single mission out of A/B (still records hypothesis)

**T5E — Pricing A/B (Stripe coupon/price-id via Z6 vendor_call).**
- Variant routes to Stripe price_id assignment via vendor_call(service="stripe", action="assign_price_variant")
- Founder approval gate (irreversible_money: real revenue impact); confirmation typed message per Z10 pattern
- Statistical-significance gate inherited from T5D

Founder track: /experiment_enable opts in; /experiment_ship / /experiment_rollback gate winners and losers; /sunset / /approve_sunset gate deprecation. Reversibility: T5A/B/C full; T5D variant assign partial (can stop, but exposed users already saw it); T5E pricing irreversible — founder approval mandatory.

## Telegram surface (founder-track, full enumeration)

| Cmd | Tier | Purpose |
|---|---|---|
| `/digest` (existing) | T2 | extended with growth section |
| `/digest_now` | T2 | force-fire weekly digest |
| `/northstar` | T1 stub → T4 wired | show current + history of north-star metric |
| `/hypothesis` | T1 stub → T4 wired | list pending + recent verdicts |
| `/backlog` | T1 stub → T3 wired | top-N scored signal candidates |
| `/approve <id>` | T1 stub → T3 wired | approve backlog candidate → mission |
| `/sunset` | T1 stub → T5 wired | list sunset candidates |
| `/approve_sunset <id>` | T5 | approve feature deprecation |
| `/experiment` | T1 stub → T5 wired | A/B status (gated) |
| `/experiment_disable <mission_id>` | T5 | opt single mission out of A/B (default-on otherwise) |
| `/experiment_ship <id>` | T5 | promote winner to 100% |
| `/experiment_rollback <id>` | T5 | force-rollback loser |

No collisions with existing `/mission /queue /resume /debug /dead /revive`.

## Event taxonomy (T2A instrumentation generates this)

Derived from success_metrics.aarrr_metrics. Standard event names recipe ships:

- `acquisition`: `landing_view`, `signup_started`, `signup_completed`
- `activation`: `first_value_event` (recipe-defined per product)
- `retention`: `session_started` (with day_of_cohort attached)
- `revenue`: `checkout_started`, `checkout_completed`, `subscription_created`, `subscription_cancelled`
- `referral`: `share_initiated`, `share_completed`, `invite_redeemed`

Every event auto-attaches: `mission_id`, `feature_id`, `variant` (if T5 active), `segment` (if T5 active), `business_model`.

## Multi-product / per-product config

**v1 decision: single product per workspace.** Multi-product routing (one founder running N concurrent products) deferred. If needed: mission.context.posthog_project_id + per-mission env routing layer.

## Privacy / PII boundary

- T3A signal intake redacts PII before persistence via `redact_user_pii()` extension to sensitivity module
- T2 analytics intake reads **aggregates only** from posthog (no raw user rows)
- T4 verdict computation operates on metric values, not user records
- Per-product analytics keys stored via existing IntegrationRegistry pattern (env var); not vaulted (no centralized vault shipped)

## Founder-track summary (per zone-doc contract)

| Step | Agent | Founder | Reversibility |
|---|---|---|---|
| T1 schema migration | applies | approves on first run | full (drops) |
| T2 weekly digest | composes | reads, picks priorities | n/a (read-only) |
| T3 signal classify | classifies + scores | /approve to spawn | pre-spawn full |
| T4 hypothesis verdict | computes + records | /confirm next iteration | full re-runnable |
| T5 cohort + sunset | proposes | /approve_sunset gates | full pre-execute |
| T5 A/B variant | ships behind flag | /experiment_ship/_rollback | partial (exposed users seen) |
| T5 pricing A/B | configures | typed-confirmation gate | irreversible (revenue) |

## Dependencies (real, verified)

- **Z2 (COMPLETE)** — mission_lessons, recipe library, posthook registry. Ready.
- **Z3 (COMPLETE)** — cost_band, multifile expander, integration_reviewer. Ready.
- **Z6 (COMPLETE)** — Stripe adapter, vendor_call verb, IntegrationRegistry. Ready for T1B + T5E.
- **Z8 (COMPLETE)** — cron infra, scheduled_tasks, registry_events, webhook_listener + signing. Ready for T2B + T3A.
- **Z10 (COMPLETE)** — reversibility taxonomy, cost ceiling, sandboxing. Ready for T1C + T5 gates.

## Open questions (resolved + carried)

- Analytics vendor → **posthog** v1; mock_mode for tests (resolved)
- A/B significance → **Bayesian posterior > 95%** (resolved)
- Window length → **per-metric default table** (founder decision 2026-05-15): activation=7d, retention=30d, revenue=14d, referral=14d, acquisition=7d; founder /override per hypothesis
- Backlog scoring → **show formula in /digest** (resolved)
- Backlog gate → **always /approve** (founder decision 2026-05-15); no auto-spawn even on high-confidence cheap; founder bottleneck accepted
- Sunset threshold → **usage <1% + non-zero cost** default, founder configurable (resolved)
- Bandit vs A/B → **fixed A/B v1**; bandit deferred (resolved)
- Digest cadence → **weekly fixed**; founder /digest_now on demand (resolved)
- Self-analytics in digest → **yes, separate "internal health" section** (resolved)
- Multi-product per workspace → **deferred; v1 single-product** (resolved)
- B2B vs B2C → **mission_brief field, default b2c** (resolved)
- Anti-pattern detection → **vanity metric + engagement vampire + insufficient N** v1; expand as patterns emerge (resolved)
- PII handling → **redact at intake, aggregates-only downstream** (resolved)
- A/B harness default → **default-on once T5 lands** (founder decision 2026-05-15); every feature ships behind flag; /experiment_disable to opt out per mission; supersedes earlier opt-in framing
- Hypothesis scope → **all Phase 8+ missions** (founder decision 2026-05-15); every mission predicts impact; bug-fixes/perf/refactor missions also predict (e.g. "fix conversion drop X%"); classifier-asks model rejected to avoid LLM-per-spec cost

### Secondary decisions (founder-accepted 2026-05-15)

- Vault for posthog keys → **env var pattern** (`KUTAI_<PROVIDER>_API_KEY`); central vault deferred until multi-product
- Mock mode default → **always-mock when `KUTAI_ENV != prod`**; `KUTAI_VENDOR_LIVE=1` forces real calls
- PII redaction scope → **emails + IPs + addresses + phones**; UUIDs stay (recipe-controlled if sensitive)
- DLQ feedback hook frequency → **weekly** (aligns with digest cycle)
- Self-analytics in digest → **default-visible compact**; founder `/digest_brief` hides internal-health section
- Reinforce-loop strength → confirmed verdict bumps `model_pick_log` score **+0.05 with 50%-per-30d decay** (old wins fade)
- A/B flag adapter → **posthog flags v1**; GrowthBook adapter deferred until cohort targeting needed
- Sunset threshold → **<1% usage + non-zero cost** v1; founder-configurable
- Recipe sunset → **features only v1**; recipe-template sunset deferred to future Z9-T6 if `recipe_pin_log` shows bad templates
- B2B event level → attach **both `account_id` + `user_id`** when business_model=b2b
- Pricing A/B confirmation → **full-params typed confirm** (`/confirm pricing <amount> <interval> <window>`) per [[project_z10_complete]] irreversible_money
- Cost ceiling per Z9 cron mission → **inherit Z10 mission ceiling**; founder `/budget_cap_growth` if Z9 over-burns

## Agent task brief (T1 dispatch)

T1A goes solo (schema canonical-first); T1B + T1C + T1D dispatch parallel after T1A merges to main. T2/T3/T4/T5 sequential. Per [[feedback_no_tier_pauses]], no greenlight pauses between tiers.

## Updates

- 2026-05-08 — v1 initial doc
- 2026-05-15 — v2 written (pass-1); 8 stale claims corrected, 5-tier plan
- 2026-05-15 — v2 revised (pass-2); claimed Phase 13.4 + 9 of 17 Phase 15 stubs; corrected step ref 2.5→2.9; added founder-track + reversibility per tier; Telegram surface enumerated; event taxonomy; multi-product / B2B-B2C / PII / anti-patterns; 20 stale claims (up from 8); T5 expanded to 5 substasks (T5A-E)
- 2026-05-15 — founder decisions folded: per-metric window table, always /approve backlog, A/B default-on (was opt-in), hypothesis all Phase 8+ (was feature-only); supersedes pass-2 drafting on 4 questions
- 2026-05-15 — **ALL 5 TIERS SHIPPED.** T1 (schema/registry/reversibility/telegram-stubs), T2 (instrumentation recipe + weekly digest cron + synthesis agent + anti-patterns), T3 (signal intake + classifier + backlog scorer + DLQ feedback), T4 (record_hypothesis + inject_north_star + verdict pipeline + reinforce loop), T5 (cohort/segment + B2B/B2C + sunset/roadmap + A/B harness + Stripe pricing A/B). Phase 13.4 + 11 Phase 15-area stubs implemented. Two stale-branch surgical merges (T3, T5AB i2p conflict). Drift: i2p new_feature_cycle=15.11 / technical_debt=15.13 / roadmap=15.14 (doc id table was stale); metric_emit post-hook never built — segment_predicate wired into instrumentation recipe instead; model_pick_log is write-only so reinforce nudge folded into fatih_hoca grading_perf_score with 30d decay. Fixed a T2 bug (get_artifact_store wrong import → digest north-star always empty).
