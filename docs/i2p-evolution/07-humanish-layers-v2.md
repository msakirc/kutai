# Z7 v2 — Humanish layers (launch + relationships, founder leverage)

> v1 (2026-05-08) framed Z7 as a flat list of A–J helpers paired into one
> zone (Z4 launch + Z8 people). Re-audit (2026-05-15) shows: ~25% of v1
> capabilities now have *infrastructure* on which Z7 should compose
> (founder_actions, support_tier1, mission_lessons, oncall_agent,
> reversibility, sandbox), the Z4/Z8 conceit is the wrong split, and
> several v1 items are antipatterns (auto-generated investor prose,
> CRM-as-graph). v2 is the operative plan: corrected audit, refactored
> framing, **5 tiers** of batched work, every pattern wired against
> shipped infra.

## Why v2

v1 wrote ten parallel helpers (A–J) at the same priority and called the
zone "lowest leverage; do after operations + growth." After Z2/Z3/Z6/Z8
shipped, that priority claim is obsolete: Z7 now sits on top of a
functioning founder-action surface, a tier-1 support agent, a
cross-mission lessons store, an on-call long-running mission shape, and
a reversibility tagger. The leverage is no longer "build all ten" — it
is **compose narrowly, measure founder-minutes-saved, and refuse the
items that look helpful but aren't.**

Specific corrections:

| v1 claim | Reality (2026-05-15) |
|---|---|
| "no customer-support automation" | `src/agents/support_tier1.py` SHIPPED (Z8 T5E) — RAG over `support_docs` Chroma collection, confidence-gated escalation via `founder_actions(kind='support_escalation')`. Tier-1 box is **already drawn** — Z7's job is the FAQ-regen flywheel on top, not the agent itself. |
| "no founder-action queue artifact" | `founder_actions` SHIPPED in Z6: card render in `src/app/founder_action_render.py`, inline rendering in `src/app/mission_view.py`, callbacks in `telegram_bot.py`. Every Z7 item that says "founder reviews + sends" must emit a `founder_actions` row, not a free-form Telegram message. |
| "Phase 13 has marketing/launch steps" (true) and "no marketing-site copy generation" (true) | Verified: 13.1–13.14 are deploy/payment/legal-review prose steps, **not copy-drafting**. The marketing-site recipe is genuinely absent from `src/workflows/recipes/` (25 recipes total, none marketing). But the missing piece isn't "a recipe" — it's a *living-surface* shape (see A1 below). |
| "demo recordings via playwright `--video on`" | Naïve. Playwright produces a raw, untrimmed 1080p webm with no narration, no chapter marks, no platform-specific cuts (HN front-page wants ≤30s; LinkedIn wants ≤90s; YouTube embed wants ≥3min). Z7 needs a **pipeline**, not a verb. `record_demo.py` + `verify_demo_artifact.py` exist but only verify; no recorder. |
| "channel-specific launch templates × 5 (HN/PH/Twitter/LinkedIn/Reddit)" | Templates are the wrong primitive. Launch is a **synchronized multi-channel mission with a 4-hour rapid-response window**, not five independent posts. Reframe as a launch *playbook mission* (see A2) consuming Z2 lessons and Z6 metrics. |
| "investor updates: agent drafts; founder edits + sends" | Antipattern. Auto-prose triggers GPT-detection on the investor side. Z7 ships **bullet generators only** (numbers + anomaly detection), founder writes prose. Reframe (see A9). |
| "CRM: relationships table; agent populates from email threads (with founder consent) + manual entry" | 6-month build with auth+OAuth+thread-parsing scope creep. Steel-thread it as **interaction log**, Telegram-native, no email integration v1 (see A10). |
| "mention monitor: Twitter+Reddit+HN+Discord+Slack" | Z8 shipped `oncall_agent.py` long-running mission shape — Z7's mention monitor is **a configured oncall_agent variant**, not a new subsystem. Free tiers + dedup + signal/noise gate are the actual design problems (see A11). |
| "Marketing site: Astro v1, integrated analytics" | Premature. Most products at this stage are better served by an outsourced Webflow/Framer site the founder edits directly — agent generates *content*, not *site*. Defer the recipe; ship the copy generator (see A12). |
| (missing in v1) | **Cold outreach has a deliverability spine** (SPF/DKIM/DMARC, warm-up, suppression list, bounce handling, CAN-SPAM/GDPR enforcement) without which "personalized email drafts" is a mass-spam footgun. v1 hand-waves "honor unsubscribes mechanically" — needs a global suppression filter on every send (see A7). |
| (missing in v1) | **Brand-voice enforcement is automatable** (lint, not arbitration). Pattern-lint shape from Z2 applies. v1 declared brand voice "founder territory forever" — half-true; *taste* is, *consistency check* isn't (see A5). |
| (missing in v1) | **Copy-compliance review** — privacy-policy ↔ marketing-claim mismatch (e.g. "we never sell your data" + analytics pixel that does). Z3 shipped `security_review` and `accessibility_review`; Z7 wants `copy_compliance_review` in the same shape (see A6). |
| (missing in v1) | **Demo distribution** — recorded MP4 → YouTube, og:video tag, Twitter video API (separate upload), HN/PH thumbnail extract, Loom embed. v1 stops at "capture"; the value lives in *publish-and-wire*. |
| (missing in v1) | **Founder briefings ARE the unifying frame**, not item J. Every zone produces a changelog the founder reads in Telegram; Z7's job is to make those changelogs *legible, prioritized, responsive*. Reframe the whole zone around this artifact (see A0). |
| (missing in v1) | **Z7 needs a KPI** — "founder-minutes-saved per mission." Without measurement, this whole zone is theater. Add `founder_minutes_saved` column to mission_events; A0 briefing surfaces weekly rollup. |
| "Pairs Z4 + Z8 because both are human-irreplaceable + helper-shaped" | Wrong split. Z4 (launch) is **bursty / event-shaped / 72-hour-window**; Z8 (people) is **continuous / log-shaped / forever**. They share *helper-shape* but nothing operational. Treat them as adjacent, not paired. Tier batches respect this. |

## Frame change

The zone is no longer "marketing + relationships, both helper-shaped."

It is now: **founder-leverage layer.** Three orthogonal sub-zones:

1. **Bursty** — launch, demo, press kit, marketing copy (event-shaped,
   72-hour windows, rapid-response gating).
2. **Continuous** — interaction log, support flywheel, mention monitor,
   investor digest (forever-running, low-rate, signal-extracting).
3. **Cross-cutting** — the founder briefing surface (A0) that every zone
   feeds into and the founder reads in Telegram every morning.

Old Z4+Z8 framing dissolves. New zone identity is: **"what saves
founder hours, measured."**

## Net consequence

Z7 v2 ships in 5 tiers (T1–T5), parallel-safe within tier:

- **T1** lays the unifying briefing surface (A0) + the KPI column +
  brand lint (A5) + copy compliance (A6). Pure infra. ~2 weeks.
- **T2** ships the launch playbook mission (A2), demo pipeline (A3),
  press kit asset store (A4). Bursty side. ~2 weeks.
- **T3** ships the FAQ flywheel (A8), CRM-as-log (A10), investor bullet
  generator (A9). Continuous side. ~2 weeks.
- **T4** ships cold-outreach with deliverability spine (A7) +
  mention-monitor as configured oncall_agent (A11). Adjacent / opt-in.
  ~1 week.
- **T5** ships marketing-copy-generator (A12) + demo distribution
  pipeline (A3-distribute) + cross-mission launch lessons. Polish +
  flywheel. ~1 week.

8 weeks total. Highest-leverage items (A0 briefing, A2 playbook, A8
FAQ flywheel) ship in T1+T2+T3 — first 6 weeks. Everything below
A11 is gated on demonstrated need (see severity).

## Patterns

Every pattern: **trigger** (when it fires) / **alt shapes**
(considered / rejected, with PICK rationale) / **wiring** (file paths +
schemas, citing shipped infra) / **founder track** / **severity**
(must / should / nice / defer).

---

### A0 — Founder briefing (the unifying surface)

**Trigger.** Every mission emits at most one briefing card on
completion (or daily digest at 09:00 founder-tz for long-running
missions). User opens Telegram, sees the morning briefing, taps
through.

**Alt shapes considered:**
- *Per-task notifications* (today's default). Rejected — drowns founder
  in noise, no prioritization.
- *Email digest*. Rejected — Telegram is already the surface; email
  fragments attention.
- *Web dashboard*. Rejected — friction; founder lives in Telegram.
- **PICK: Telegram briefing card per mission + 09:00 digest.** Reuses
  existing `founder_actions` render path; founder already trained on it.

**Wiring.**
- New table `mission_briefings`: `mission_id, kind ('completion'|'daily'),
  body_md, founder_minutes_saved_estimate, prepared_at, read_at,
  acted_on (jsonb)`.
- New posthook `briefing_compose` (kind 26 in `general_beckman/posthooks.py`)
  fires on terminal `mission.complete`. Pulls: phase summaries,
  changed-files list, deferred items, cost actuals (from Z6 cost_pull),
  failed-then-recovered events (from `mission_lessons`).
- Render via `src/app/founder_action_render.py` (extend with
  `render_briefing(briefing_row)` — section blocks, copy-as-tweet
  button per section).
- Daily 09:00 cron: new `src/app/jobs/daily_briefing.py`, registered
  in `src/app/scheduled_jobs.py`. Aggregates: in-flight missions,
  pending founder_actions, cost burn, mention-monitor signal (when
  A11 ships).

**Founder track.** Reads, taps "ack" or "act" per section. "Act"
opens the relevant founder_action card. "Ack" marks read; counted
toward weekly attention budget.

**Severity: MUST (T1).** Without this, every other Z7 helper produces
output the founder never sees.

---

### A1 — Marketing-site flow (deferred + reframed)

**v1 said:** "marketing-site recipe; agent generates draft; founder
edits in Telegram thread or web editor; iterate via mission thread
comments; output: deployable marketing site."

**Reality.** Most early products are better served by Webflow / Framer
the founder edits directly. The agent's job at this stage is *content*
(headlines, body copy, FAQ entries, screenshot captions), not *site*
(routes, layouts, build pipeline, deploy).

**Reframe.**
- T5 ships `marketing_copy_generator` mr_roboto verb that produces a
  structured `marketing_copy.json` artifact (hero × 3 variants,
  features × N, pricing tiers, FAQ entries from spec + early support
  tickets). Founder pastes into Webflow.
- A *real* marketing-site recipe (NextJS / Astro) deferred until a
  mission demands programmatic site updates (e.g. pricing-page rebuild
  on tier change, or auto-published changelog). Recipe scaffold goes
  in `src/workflows/recipes/marketing_site.json` only when first
  consumer arrives.

**Wiring (T5 only).**
- `packages/mr_roboto/src/mr_roboto/actions/marketing_copy.py`. Inputs:
  spec, brand-voice doc (from A5), early ticket clusters (from A8).
  Outputs: `artifacts/marketing_copy/{mission_id}.json`.
- New phase 13.0a step `marketing_copy_draft` (mechanical), gated to
  fire only when `compliance_overlay.product_kind in {b2c,b2b}` and
  `goals.includes('public_launch')`.

**Founder track.** Reviews structured copy, picks variant, pastes to
external site builder. founder_action card with "approve / regenerate
hero / regenerate FAQ" buttons.

**Severity: NICE (T5).** Outsourcing to Webflow + agent-supplied copy
covers 80% of cases. Recipe is gated on demonstrated need.

---

### A2 — Launch playbook (mission shape, not template fill)

**Trigger.** Founder runs `/launch` (new Telegram cmd) with date +
selected channels. System creates a long-running mission ("launch-2026-Q3")
that owns the 72-hour window.

**Alt shapes considered:**
- *Five independent template-fill steps*. Rejected — no synchronization,
  no rapid response, no cross-channel learning.
- *Single mega-step* "do_launch". Rejected — opaque, untraceable, no
  partial-progress recovery.
- **PICK: long-running mission with phase clock.** Reuses Z8
  `oncall_agent` shape (`src/agents/oncall_agent.py`). Phases: T-72h
  (asset prep), T-24h (final review founder_actions), T-0 (synchronized
  publish), T+0..72h (rapid-response monitoring), T+7d
  (lessons writeback).

**Wiring.**
- New mission template at `src/workflows/launch/launch_playbook.json`.
  Phase clock fields: `relative_to: scheduled_publish_at`,
  `offset_hours: -72|-24|0|+1|+4|+24|+72|+168`.
- T-72h step: `launch_post_drafting` mechanical action that pulls per-channel
  templates from `packages/mr_roboto/src/mr_roboto/actions/launch_drafts/`
  (`hn.py`, `ph.py`, `twitter.py`, `linkedin.py`, `reddit.py`). Each
  consumes spec + brand-voice + prior launch lessons (A2 ↔ mission_lessons).
- T-24h: founder_action card "approve drafts" — one card per channel,
  inline edit-and-approve.
- T-0: `publish_synchronized` — calls per-channel publish endpoints (HN
  via `gh api` of HN's public submit, PH via PH GraphQL, Twitter via
  X API v2, LinkedIn via LI API, Reddit via PRAW). Each behind a
  `vendor_call` (Z6 shipped) with reversibility=`partial` (post can be
  edited within window) or `irreversible` (can't unpost).
- T+0..72h: `launch_response_monitor` — sub-mission that polls each
  channel's API for replies/comments/upvotes every 15min, surfaces:
  - top-engagement comments (founder_action: "draft reply")
  - sentiment-flagged negative threads (founder_action: "review")
  - velocity anomalies ("HN slipping below page-2 threshold; consider
    response post").
- T+7d: `launch_lessons_writeback` — emits 3-5 `mission_lessons` rows
  (e.g. `dedup_key='launch.hn.timing.9am-est'`,
  `body='2.3x traction vs. 2pm-est, n=2'`).

**Founder track.** Approves drafts T-24h, hits "publish all" or
per-channel publish at T-0, taps response cards as they surface.
Agent never auto-replies.

**Severity: SHOULD (T2).** Used quarterly but high-stakes and
high-cognitive-load — biggest hour-saver per use.

---

### A3 — Demo pipeline (storyboard → record → trim → distribute)

**Trigger.** New phase 13.x step `demo_pipeline` fires when mission
goal includes `public_demo` or A2 launch playbook is active.

**Alt shapes considered:**
- *Single playwright recording*. Rejected (v1 picked this) — raw
  artifact, useless without trim.
- *External tool (Loom, Tella, Arcade)*. Rejected — founder already
  pays no manual tools beyond bot; agent leverage zero.
- **PICK: 4-stage internal pipeline.** Storyboard (LLM-drafted from
  spec) → record (playwright with `--video on` per scenario file) →
  trim (ffmpeg with cut points from storyboard) → caption (Whisper
  reverse-direction: from script + timing, generate WebVTT).

**Wiring.**
- `packages/mr_roboto/src/mr_roboto/actions/demo/storyboard.py` —
  LLM-bound (this is the one Z7 LLM call). Output: ordered scene list
  with target_seconds, viewport_state, narrator_text. Per single-caller
  rule, goes through `beckman.enqueue` not direct dispatcher.
- `actions/demo/record.py` — non-LLM, drives playwright with one
  scenario file per scene; outputs raw scene MP4s into
  `artifacts/demo/{mission_id}/raw/`.
- `actions/demo/edit.py` — non-LLM, ffmpeg concat + per-scene trim
  using storyboard cut points; produces `cuts/{30s,60s,3min}.mp4`.
- `actions/demo/caption.py` — generates `cuts/{N}.vtt` from
  storyboard.narrator_text + scene durations (no STT — script-driven).
- New posthook `demo_artifact_check` — verifies each cut file exists,
  duration within ±10% target, captions present.

**Distribution stub (T5):** `actions/demo/distribute.py` — uploads to
YouTube via Data API v3 (upload as unlisted by default; founder
flips to public via founder_action), generates og:video tag for
homepage, extracts thumbnail still per cut.

**Founder track.** Reviews storyboard at T-72h (founder_action with
edit-cut-points buttons), reviews final cuts pre-distribute (T-24h),
flips YouTube to public after launch.

**Severity: SHOULD (T2 for record/edit; T5 for distribute).**

---

### A4 — Press kit as versioned binary store

**Trigger.** Founder runs `/press_kit` or A2 launch playbook references
press kit. System assembles + publishes to permanent URL.

**Why versioned binary store, not artifact-folder.** Journalists hate
broken links. v1 said "mechanical to assemble" — half right. The
*assemble* is mechanical; the *host* + *track downloads* + *update over
time* are the real shape.

**Wiring.**
- New table `press_kits`: `kit_id, version, mission_id,
  manifest_json, published_url, created_at`.
- `actions/press_kit/assemble.py` — gathers: one-pager (LLM-drafted
  from spec, founder-approved), founder bios (from `relationships`
  table A10), screenshots (from A3 raw), logo files (from A1
  `marketing_copy.json`), product fact sheet, recent quotes from
  reviews (when A8 has a quotes set), past press mentions.
- Output: `press_kit_v{N}.zip` + manifest.
- `actions/press_kit/publish.py` — uploads to S3/R2 (config in
  `.env`: `PRESS_KIT_BUCKET`), returns CDN URL. Permanent URL pattern:
  `presskit.<domain>/v{N}/`.
- Each kit version retained; older versions return 410 Gone with
  "see latest at presskit.example.com" stub HTML.
- New posthook `press_kit_freshness` — fires monthly; if any spec
  field changed and kit is >90 days old, surfaces founder_action
  "regenerate press kit?".

**Founder track.** Approves assembled kit before publish. Updates bios
manually (from A10 founder_input flow). Never edits hosted kit
directly — re-publishes new version.

**Severity: SHOULD (T2).** Bundled with A2 launch playbook.

---

### A5 — Brand-voice lint (consistency, not taste)

**Trigger.** Any artifact produced by Z7 (marketing copy, launch
posts, press kit text, investor digest, FAQ entry) passes through
brand-voice lint pre-founder-review.

**Why this works.** *Taste* (does this voice represent us?) is founder
arbitration, irreducible. *Consistency* (does this match our committed
voice doc?) is automatable — same shape as Z2 `pattern_lint`.

**Wiring.**
- New artifact `brand_voice.md` produced once at Z1 phase 1.x via a
  founder-led step (5–10 examples + 3–5 prohibited words/phrasings +
  preferred sentence length + reading level target). Stored in
  `artifacts/brand/{project_id}/brand_voice.md`.
- New posthook `brand_voice_lint` (kind 27): runs over Z7 text
  artifacts. Checks:
  - prohibited terms (regex list from brand_voice.md)
  - average sentence length within ±25% of target
  - Flesch-Kincaid reading level within target band
  - "we" vs "you" pronoun ratio matches doc
  - LLM-bound second pass (small model, OVERHEAD lane via beckman.enqueue)
    scores tone match 0–10 with one-sentence justification per flagged
    section.
- Violations attach to founder_action review card as inline annotations
  (Z3 integration_reviewer pattern).

**Founder track.** Founder approves brand_voice.md once. Subsequent
arbitration is "accept lint suggestion" or "override + add exception"
(exception persisted to brand_voice.md as "prior precedent").

**Severity: MUST (T1).** Cheap, immediate signal, prevents shipping
copy that founder will rewrite.

---

### A6 — Copy compliance review

**Trigger.** Any externally-published text artifact (marketing copy,
launch post, press release, investor update) runs through
copy_compliance_review pre-founder-review.

**What it catches.**
- Privacy policy ↔ marketing copy mismatch ("we never sell your data"
  vs. analytics pixel that does).
- Outcome claims without disclosure ("save 10 hours/week" without
  "results vary" or methodology footnote — required in some
  jurisdictions).
- Trademark / superlative violations ("best", "guaranteed") in
  jurisdictions where these require substantiation.
- Forward-looking statements without safe-harbor language.
- Channel-specific rules (HN guidelines: no "Show HN: launching X" if
  not actually shippable today; PH rules: no "first" or "ever" in
  taglines).

**Wiring.**
- New posthook `copy_compliance_review` (kind 28). Same shape as Z3
  `security_review` / `accessibility_review`. Severity model: blocker
  (privacy mismatch) / warning (channel rule) / info (best-practice).
- Inputs: text artifact + jurisdiction (from compliance_overlay) +
  channel (from A2 launch playbook step context) + privacy_policy.md
  (from Z6 compliance_templates).
- Tooling: rule engine (small set of regex + LLM-bound semantic check
  against privacy_policy.md). Privacy ↔ copy semantic check uses LLM
  with structured output (does claim X contradict policy Y?
  yes/no/unclear, citation).
- Fix-suggest mode: for blockers, propose rewrite (one alt phrasing).

**Founder track.** Blockers gate publish. Warnings annotate review
card. Founder can override warnings; override logged.

**Severity: MUST (T1).** Compounds with A5 — both cheap, both prevent
known-bad output.

---

### A7 — Cold outreach with deliverability spine

**Trigger.** Founder uploads target list (CSV) via `/outreach upload`
or agent identifies target from CRM (A10) follow-up rule.

**v1 missed:** the deliverability + legal spine is the actual hard part.
"Personalized email drafts" without it is mass-spam-ware.

**Wiring.**
- **Sender infra contract (one-time setup founder_action):** verify
  SPF, DKIM, DMARC for outreach domain (separate from product domain to
  protect deliverability). Use a dedicated subdomain
  (e.g. `hello.example.com`) with separate sending infrastructure
  (Postmark / SendGrid / Resend).
- New table `outreach_suppression`: `email, reason
  ('unsubscribed'|'bounced'|'complained'|'manual'), added_at,
  source_mission_id`. Global filter on every send. Webhook receiver
  for ESP unsubscribe + bounce + complaint events
  (`/webhook/outreach/unsubscribe`).
- New table `outreach_warmup`: `domain, day, sent_count, target_count`
  — enforces ramp curve (day 1: 50/day, day 14: 500/day).
- New table `outreach_sends`: `send_id, list_id, target_email,
  template_id, sent_at, opened_at, replied_at, bounced_at`.
- `actions/outreach/draft.py` — LLM-bound, generates personalized
  draft from prospect data (manual entry or from `vendor_call` against
  Clearbit-style enrichment if founder pays for it).
- `actions/outreach/send.py` — non-LLM. Filters target through
  suppression + warmup + jurisdiction (no GDPR jurisdiction without
  explicit opt-in, which v1 cold outreach doesn't have).
- Every outbound message contains: List-Unsubscribe header,
  one-click unsubscribe link, postal address (CAN-SPAM).
- New posthook `outreach_deliverability_check` — flags if domain
  reputation drops, complaint rate > 0.1%, bounce rate > 5%; pauses
  campaign + surfaces founder_action.

**Founder track.** Approves each batch (founder_action with first 3
drafted emails as preview + total count + estimated send window).
Reviews replies via existing inbox; agent doesn't auto-reply.

**Severity: NICE (T4).** Founder may never need this; gate behind
opt-in. Don't ship until founder requests, but design the schema
upfront so the ESP webhook receiver doesn't get retrofitted later.

---

### A8 — Support flywheel on top of shipped tier-1

**Trigger.** Existing `support_tier1.py` (Z8 T5E) handles incoming
queries. Z7 adds:
1. **FAQ regeneration** weekly cron pulls from escalations + low-confidence
   answers.
2. **Quote extraction** monthly cron pulls from positive resolutions
   (consent-gated) for press kit (A4).
3. **Documentation gap detection** flags missing docs causing tier-1
   misses.

**Wiring.**
- New job `src/app/jobs/faq_regen.py` — weekly. Pulls last 7 days
  from `support_escalations` + `support_low_confidence` (new column on
  support_tier1's interaction log). Clusters by topic (LLM-bound, small
  model, OVERHEAD lane via beckman.enqueue). For clusters > 3
  interactions, drafts FAQ entry. Surfaces as founder_action
  "approve FAQ entry?" → on approve, appends to `faq.md` artifact +
  re-indexes `support_docs` Chroma collection.
- New job `src/app/jobs/quote_harvest.py` — monthly. Scans positive
  resolutions for quotable language. Founder_action "request quote
  consent from this user?" with pre-drafted message. On consent,
  quote enters `press_kit_quotes` table for A4.
- New posthook `documentation_gap_detect` — runs on every escalation.
  Compares user's question against existing docs via semantic search;
  if no doc matches, flags `docs_gap_log` table. Weekly digest of
  gaps surfaces in A0 briefing → founder decides which to fill.

**Founder track.** Reviews FAQ drafts (one batch/week). Approves quote
requests. Decides which docs to write from gap digest.

**Severity: SHOULD (T3).** Compounding leverage — every FAQ entry
shipped reduces escalation rate, freeing founder hours.

---

### A9 — Investor bullets (numbers + anomalies, NOT prose)

**Trigger.** Monthly cron at month-end produces `investor_bullets.md`
artifact. NOT auto-sent.

**Why bullets-only.** Investor-grade prose written by AI is detected,
correctly read as low-effort, and damages relationship. The agent's
job is to surface *what the founder hasn't noticed* — outliers,
trends, anomalies. Founder writes prose.

**Wiring.**
- New job `src/app/jobs/investor_bullets.py` — monthly. Pulls from:
  - Z6 metrics (revenue, MRR delta, churn, customer count) — via
    Stripe recipe consumer.
  - Z6 cost actuals (burn, runway months).
  - Z8 ops metrics (uptime, P95 latency, incident count).
  - Z3 review density (PRs shipped, security issues caught).
  - A8 support volume + escalation rate.
  - A11 mention monitor positive/negative count.
- Anomaly detection (vs. trailing-3-month median, ±2σ): flag each
  outlier with one-sentence "what changed" hypothesis (LLM-bound,
  OVERHEAD lane).
- Output sections (Markdown bullets only, no prose):
  - **Highlights** (top 3 positive outliers)
  - **Lowlights** (top 3 negative outliers)
  - **Numbers** (table)
  - **Anomalies needing founder explanation** (3-5 with hypothesis)
  - **Suggested asks** (gaps the founder might want investor help on
    — pulled from `mission_lessons` flagged as `needs_external_help`).
- Surfaces as founder_action "review monthly bullets" with
  edit-then-export-to-clipboard button.

**Founder track.** Reviews bullets, edits/deletes anomaly hypotheses
that are wrong, copies the relevant subset into the email they write.

**Severity: SHOULD (T3).** Compounds with Z6 metrics; blocked on Z6
Stripe wiring shipping for revenue numbers.

---

### A10 — CRM as interaction log (steel thread)

**Trigger.** Founder logs an interaction via `/log_interaction
@contact_handle "had call about partnership; circle back in 2 weeks"`.

**Why log not graph.** v1 wanted full email integration + thread
parsing + auto-population — a 6-month build with OAuth scope creep,
inbox API rate limits, and consent UX nightmares. Steel-thread it as a
*log* the founder writes to in Telegram, agent reads from for
reminders.

**Wiring.**
- New table `relationships`: `contact_id, handle, display_name,
  category ('customer'|'prospect'|'investor'|'journalist'|'partner'|
  'advisor'|'candidate'|'vendor'|'other'), email, links_json,
  notes_md, created_at`.
- New table `interactions`: `interaction_id, contact_id, kind
  ('call'|'email'|'meeting'|'message'|'event'|'other'), summary,
  next_action, follow_up_at (nullable), logged_at, mission_id (nullable)`.
- Telegram cmds:
  - `/contact add @handle [category]`
  - `/log @handle [summary; follow-up: 2w]`
  - `/contacts [category]` — list with last_interaction
  - `/follow_ups` — list pending follow_up_at <= today + 7
- New job `src/app/jobs/follow_up_reminder.py` — daily 09:00. Reads
  `interactions WHERE follow_up_at <= today() AND follow_up_done IS
  NULL`. Surfaces in A0 briefing (one section per category).
- A2 launch playbook reads `relationships WHERE category='journalist'`
  for press kit distribution suggestions.
- A4 press kit reads `relationships WHERE category in
  ('founder','advisor')` for bios.
- A9 investor bullets reads `relationships WHERE category='investor'`
  for distribution list.

**Founder track.** Logs interactions in Telegram (1-line, takes 5
seconds). Acts on reminder digest (A0). No email integration v1; if
founder later wants Gmail thread import, that's a v2 module.

**Severity: SHOULD (T3).** Cheap, high leverage — most founders have
no relationship memory beyond "I think I emailed them in March."

---

### A11 — Mention monitor as configured oncall_agent

**Trigger.** Founder runs `/mention_monitor add @product_name [channels]`.
System spawns long-running mission with oncall_agent shape.

**Why oncall_agent shape.** Z8 shipped this for ops alerts. It already
handles: poll cycle, alert dedup, escalation policy lookup, founder
notification, sub-handler dispatch. Mention monitor is the same shape
with different sources + classifier.

**Wiring.**
- New mission template
  `src/workflows/mention_monitor/mention_monitor.json` — uses
  oncall_agent with custom poll handlers:
  - `polls/twitter.py` — X API v2 search (paid; off by default; behind
    founder_action consent for cost).
  - `polls/reddit.py` — PRAW search (free, 60/min rate limit).
  - `polls/hn.py` — Algolia HN API (free, no key).
  - `polls/discord.py` — bot user in joined servers (founder OAuths).
  - `polls/google_alerts.py` — RSS of Google Alerts (free fallback for
    arbitrary blogs/news).
- New table `mentions`: `mention_id, source, source_id, url, author,
  text, sentiment ('pos'|'neg'|'neu'), signal_score (0-10), seen_at,
  acted_on`.
- Dedup: `(source, source_id)` unique; cross-source dedup via
  URL canonicalize + 24h window.
- **Signal/noise gate (the actual hard part).** Each mention scored
  by: author follower count, prior interaction with us (CRM A10
  match), keyword density vs. generic. Below score=4: silent log only.
  Above score=7: founder_action immediate. 4–7: daily digest in A0.
- Negative-sentiment cluster of >3 in 1h: triggers `crisis_comms_draft`
  founder_action — never auto-respond.

**Founder track.** Approves channel adds (especially paid Twitter).
Acts on score>=7 mentions in real time. Reviews daily digest of 4–7s.
Decides on negative clusters.

**Severity: NICE (T4).** Gate behind first launch event (A2). Don't
ship until founder demonstrates demand.

---

### A12 — Marketing copy generator (decoupled from site)

Already covered in A1. Summary: T5 ships a structured-copy artifact;
the site itself is outsourced to Webflow until programmatic site
updates become a real need.

---

## Tiered plan

### T1 — Foundation + cheap gates (~2 weeks)

Parallel-safe: 4 agents.

- **T1A — Founder briefing surface (A0).** New `mission_briefings`
  table, `briefing_compose` posthook (kind 26), `render_briefing`
  extension to `founder_action_render.py`, `daily_briefing` job at
  09:00, founder_minutes_saved column on `mission_events`.
- **T1B — Brand voice lint (A5).** Z1-side `brand_voice.md` step
  added at 1.x (founder-led prompt sequence), `brand_voice_lint`
  posthook (kind 27), wired pre-founder-review for Z7 artifacts.
- **T1C — Copy compliance review (A6).** New posthook (kind 28),
  rule engine + LLM-bound semantic check against privacy_policy.md,
  fix-suggest mode for blockers, severity model matching
  security_review.
- **T1D — Z7 KPI plumbing.** `founder_minutes_saved` per-mission
  rollup query, weekly summary in A0, Telegram cmd
  `/founder_hours_saved [period]`. Without this, the rest of Z7 has
  no signal.

Acceptance: A0 briefing renders for completed mission, A5+A6 fire on a
test marketing-copy artifact and produce expected violations, KPI
column populated for last 30 days from heuristic estimate
(time_per_step × steps_handled).

### T2 — Bursty side (~2 weeks)

Parallel-safe: 3 agents.

- **T2A — Launch playbook (A2).** Mission template, phase clock
  fields, per-channel draft mechanicals, T-24h founder approval card,
  T-0 synchronized publish via vendor_call wrappers, T+0..72h
  response monitor, T+7d lessons writeback.
- **T2B — Demo pipeline (A3 record/edit).** storyboard (LLM-bound)
  + record (playwright) + edit (ffmpeg) + caption (script-driven
  WebVTT) + `demo_artifact_check` posthook. Distribution stub deferred
  to T5.
- **T2C — Press kit binary store (A4).** `press_kits` table,
  assemble + publish actions, S3/R2 upload, permanent URL pattern,
  `press_kit_freshness` posthook.

Acceptance: `/launch` cmd creates mission with phase clock, drafts
per-channel posts, founder can approve+publish, response monitor
surfaces a synthetic engagement event. Demo pipeline produces
{30s,60s,3min}.mp4 + .vtt for a sample storyboard. Press kit publishes
to permanent URL, retains v1 when v2 generated.

### T3 — Continuous side (~2 weeks)

Parallel-safe: 3 agents.

- **T3A — FAQ flywheel (A8).** Weekly `faq_regen` job clusters
  escalations, drafts entries, surfaces founder_action approve cards,
  appends to faq.md + reindexes `support_docs`.
- **T3B — CRM as log (A10).** `relationships` + `interactions`
  tables, 4 Telegram cmds (`/contact add`, `/log`, `/contacts`,
  `/follow_ups`), daily reminder job feeding A0. Wired into A4 (bios)
  and A9 (investor list).
- **T3C — Investor bullets (A9).** Monthly job pulls Z6 metrics + Z8
  ops + Z3 review + A8 support + A11 mentions (when present), anomaly
  detection vs. trailing 3-month median, structured-bullets output.

Acceptance: FAQ regen produces a draft from synthetic escalations,
founder approval appends + reindexes Chroma. CRM cmds round-trip,
follow-up appears in A0 next morning. Investor bullets renders for
prior month with at least 1 anomaly flagged from synthetic data.

### T4 — Adjacent / opt-in (~1 week)

Parallel-safe: 2 agents.

- **T4A — Cold outreach with deliverability spine (A7).** SPF/DKIM/DMARC
  setup founder_action, suppression + warmup + sends tables, ESP
  webhook receivers, draft + send actions, deliverability_check
  posthook. Behind feature flag; off by default.
- **T4B — Mention monitor (A11).** Configured oncall_agent variant,
  4 poll handlers (HN free + Reddit free + Google Alerts free +
  Twitter paid-opt-in), mentions table with dedup, signal/noise gate,
  crisis comms draft trigger.

Acceptance: A7 sends to a test inbox, suppression list filters a
known-bad email, warmup blocks send if over day's quota. A11 polls HN
+ Reddit for a known mention, scores it, surfaces in A0 if score>=7.

### T5 — Polish + flywheel (~1 week)

Parallel-safe: 3 agents.

- **T5A — Marketing copy generator (A12 / A1).** structured copy
  artifact via `marketing_copy.py` mr_roboto verb, founder_action
  approve card.
- **T5B — Demo distribution (A3-distribute).** YouTube Data API
  upload (unlisted by default), og:video tag generation, thumbnail
  extraction per cut.
- **T5C — Cross-mission launch lessons consumer.** A2's T+7d
  lessons writeback wired into next launch's T-72h drafting via
  STACK_BLOCKS injection (Z2-T4 pattern).

Acceptance: A12 produces structured marketing copy for a sample
spec, A5 lint passes. A3 distribute uploads sample 60s.mp4 to
YouTube as unlisted, returns embed URL. A2 second launch consumes
lessons from first launch (verifiable in draft prompt context).

## Wiring summary

| Item | New tables | New posthooks | New mr_roboto verbs | New jobs | New mission templates |
|---|---|---|---|---|---|
| A0 | `mission_briefings` | `briefing_compose` (26) | — | `daily_briefing` | — |
| A2 | — | — | `launch_drafts/*` × 5, `publish_synchronized`, `launch_response_monitor` | — | `launch_playbook.json` |
| A3 | — | `demo_artifact_check` | `demo/{storyboard,record,edit,caption,distribute}` | — | — |
| A4 | `press_kits`, `press_kit_quotes` | `press_kit_freshness` | `press_kit/{assemble,publish}` | — | — |
| A5 | — | `brand_voice_lint` (27) | — | — | — |
| A6 | — | `copy_compliance_review` (28) | — | — | — |
| A7 | `outreach_suppression`, `outreach_warmup`, `outreach_sends` | `outreach_deliverability_check` | `outreach/{draft,send}` | — | — |
| A8 | `support_low_confidence`, `docs_gap_log`, `press_kit_quotes` | `documentation_gap_detect` | — | `faq_regen`, `quote_harvest` | — |
| A9 | — | — | — | `investor_bullets` | — |
| A10 | `relationships`, `interactions` | — | — | `follow_up_reminder` | — |
| A11 | `mentions` | — | `mention_polls/{hn,reddit,google,twitter,discord}` | — | `mention_monitor.json` |
| A12 | — | — | `marketing_copy` | — | — |

Net: 8 new tables, 6 new posthooks (registry 25→31), ~20 new mr_roboto
actions, 5 new jobs, 2 new mission templates.

## Dependencies

- **Inbound (must be live first):**
  - Z6 founder_actions surface — every Z7 helper card uses it.
  - Z6 vendor_call + reversibility tagger — every external publish
    (HN/PH/X/LI/RD, Stripe, ESP, YouTube Data API) goes through
    vendor_call with reversibility set.
  - Z6 Stripe recipe consumer — A9 investor bullets need revenue
    numbers.
  - Z2 mission_lessons + STACK_BLOCKS — A2 launch lessons writeback
    + T5C launch lessons consumer.
  - Z3 security_review/integration_reviewer pattern — A5/A6 follow
    same shape.
  - Z8 oncall_agent — A11 mention monitor reuses.
  - Z8 support_tier1 + support_docs — A8 FAQ flywheel composes on top.
  - Z10 cost_band, sandbox runner — A2 publish, A7 send, A11 paid
    Twitter all need cost-band thresholds + sandbox fallback for dry
    runs.
- **Outbound (Z7 produces, others consume):**
  - A0 briefing surface read by every zone for "report progress to
    founder" — replaces ad-hoc `notify_user` in many places.
  - A10 relationships read by A4 press_kit, A9 investor list.
  - mission_lessons writeback from A2 + A8 + A11 feed Z2 cross-mission
    learning loop.

## Founder track summary

| Pattern | Founder action per use | Estimated minutes |
|---|---|---|
| A0 briefing | Read morning digest + tap "act" on 0–3 cards | 2–10/day |
| A2 launch playbook | Approve drafts (T-24h), publish (T-0), respond to flagged engagement (T+0..72h) | 30–120/launch |
| A3 demo pipeline | Edit storyboard cut points once | 10–20/launch |
| A4 press kit | Approve assembled kit pre-publish; update bios occasionally | 5/launch + 5/quarter |
| A5 brand voice lint | Approve `brand_voice.md` once at Z1; "accept lint" or "override+exception" per artifact | 30 once + 10s/artifact |
| A6 copy compliance | Override warnings inline; blockers force rewrite | 10s–2min/artifact |
| A7 cold outreach | One-time DKIM setup; approve each batch | 30 once + 5/batch |
| A8 FAQ flywheel | Approve weekly FAQ drafts; decide quote consent requests | 10/week |
| A9 investor bullets | Edit anomaly hypotheses; copy relevant subset | 15/month |
| A10 CRM log | Log interactions in Telegram (`/log @x ...`) | 5s/interaction |
| A11 mention monitor | Approve channel adds; act on score>=7 mentions; review daily digest | 5/day after launch |
| A12 marketing copy | Approve variant; paste to Webflow | 15/site update |

## Open questions

- **Brand voice doc — when does it get authored?** v2 says Z1 phase
  1.x, but Z1 was already shipped. Either retrofit a Z1 step or accept
  brand_voice.md as a Z7-T1 artifact founder authors when Z7 ships.
  Lean Z7-T1 — keeps Z7 self-contained.
- **Press kit hosting — does the system own a CDN?** No today. T2
  decision: ship with required `.env` config (`PRESS_KIT_BUCKET`),
  document setup as founder_action for first run. Or skip CDN for
  v1 and serve from main app `/press-kit/v{N}/` route — uglier URL
  but zero infra.
- **Twitter API cost.** $200/mo basic tier. Default off; founder
  enables explicitly. A11 design must degrade gracefully (3 channels
  instead of 4) when off.
- **Sentiment classifier.** Off-the-shelf small model (RoBERTa-sentiment
  via HF, runs on CPU) vs. LLM-bound. Bench: 1000 mentions/day at
  small-model is free; LLM-bound is ~$2/day. Pick small model; fall
  back to LLM only for negative cluster confirmation.
- **Investor list distribution.** A9 produces bullets; founder writes
  prose; how does it get sent? v2 says "copy to clipboard" —
  intentional. Don't ship a "send investor email" button; founder
  sends from their own inbox so reply-to is correct.
- **CRM email integration** — explicitly not in scope. Defer to a
  future v3 module after observing demand.
- **Crisis comms checklist** — A11 surfaces negative cluster, surfaces
  draft; the *checklist* (notify counsel? draft holding statement?
  freeze marketing?) needs to live somewhere. Add as a static
  `playbooks/crisis_comms.md` artifact T2 ships, surfaced when A11
  triggers cluster threshold.

## Agent task brief

When picking up this doc:

1. Read 00-README + 06-real-world-bridge-v2 (founder_actions +
   vendor_call infra you'll be composing on) + 08-operations-v2
   (oncall_agent shape) + 09-growth (metrics A9 consumes) + this doc.
2. Confirm shipped infra is still as audited (founder_actions,
   support_tier1, oncall_agent, mission_lessons, reversibility) —
   grep call sites, not docstrings (per
   `feedback_audit_call_sites`).
3. Build T1 first; T1 is the foundation everything else composes on.
   Spawn 4 parallel agents (T1A/B/C/D) using the
   `superpowers:dispatching-parallel-agents` skill.
4. After T1 land + merge to main: spawn T2 (3 agents), then T3 (3
   agents), then T4 (2 agents), then T5 (3 agents). Land tier-by-tier
   per `feedback_canonical_first_for_tier3plus` — never dispatch
   tier N+1 agents while tier N is in worktrees.
5. After each tier merge: tag `z7-t{N}-shipped`, run full test
   suite with timeout, update memory.
6. After T5: tag `z7-complete-{date}`, write `project_z7_complete.md`
   memory entry, update zone map in 00-README to mark Z7 done.
7. Per `feedback_no_tier_pauses`: do not stop between tiers for
   approval; only stop on a genuine question. Open questions above
   are resolvable from this doc (defaults given).

## Updates

- 2026-05-15 — v2 written. Re-audit found 14 of 16 v1 capabilities
  scaffolded or absent (only support_tier1 partial + Z3 review infra +
  founder_actions render shipped). Zone reframed: split bursty /
  continuous / cross-cutting; founder briefing (A0) as unifying
  surface; brand-voice lint (A5) + copy compliance (A6) added as
  cheap T1 gates; investor bullets (A9) reframed bullets-not-prose;
  CRM (A10) reframed log-not-graph; mention monitor (A11) reframed
  oncall_agent variant; marketing site (A1) deferred; demo (A3) as
  4-stage pipeline not single playwright call; cold outreach (A7) gets
  deliverability spine; founder-minutes-saved KPI added. Z4+Z8 v1
  pairing dissolved. 5-tier batched plan, ~8 weeks total.
