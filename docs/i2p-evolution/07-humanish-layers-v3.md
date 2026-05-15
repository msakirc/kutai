# Z7 v3 — Humanish layers (end-to-end pass: launch / comms / relationships / lifecycle)

> v2 (2026-05-15 morning) reframed Z4+Z8 → bursty/continuous/cross-cutting
> with 12 patterns A0–A12. Second-pass audit on the same day revealed:
> (a) **9 humanish areas missing entirely** (lifecycle email engine,
> changelog-as-public, customer-facing status page, meeting-brief
> auto-generation, founder attention budget queue, crisis-comms tiered
> playbook, customer-interview note pipeline, reviews harvest, audit log
> for external comms) — added as B1–B9;
> (b) **13 sub-pattern refinements** missed in A-series
> (launch-readiness gate, demo accessibility, audience-segmented kits,
> multi-voice brand, channel-rules library, outreach reply-handling,
> multilingual FAQ regen, segmented investor templates, consent ledger,
> oncall_agent configurability extension, internal-signal mention
> source, meeting-brief CRM amplifier, attention-budget integration);
> (c) **3 wiring corrections** (tickets table already exists for A8,
> founder_attention_log already exists for B5, oncall_agent is NOT
> configurable so A11 must extend it);
> (d) **15 patterns explicitly deferred** with rationale (D1–D15) so
> the humanish gap-map is honest, not silently incomplete.
>
> v3 is the operative plan: 21 active patterns (A0–A12 + B1–B9),
> 15 deferred (D1–D15), 6 tiers ~12 weeks, every wiring claim
> re-evidenced against the audit.

## Why v3

v2 covered launch + relationships + support flywheel + investor digest
+ mention monitor + marketing copy. After writing it, the obvious
question — "if I were a founder using KutAI for the first 12 months of
a product, what *human* tasks would still cost me hours?" — surfaced 9
glaring omissions:

- Did I onboard a user? **No lifecycle email engine** covers it.
- Did I publish a changelog? **No public-changelog artifact** ships.
- Did the product break? **No customer-facing status page**.
- Am I prepping for a call? **No meeting brief** generated.
- Am I drowning in 17 founder_actions today? **No attention budget**.
- Did a journalist email me about a security breach? **No crisis tier**.
- Did I do a customer interview? **No interview pipeline**.
- Are there reviews on G2 / AppStore? **No reviews harvest**.
- What did the agent send to whom yesterday? **No audit log**.

Each is a real recurring founder hour-sink. v2 missed all 9. v3 adds.

The audit also corrected three wiring claims in v2:

| v2 claim | Reality (re-audited 2026-05-15) | Source |
|---|---|---|
| A8 needs new `support_low_confidence` table | `tickets` table already captures `confidence`, `status`, `escalated_to_founder`, `founder_action_id`, `sentiment` per interaction. A8 reads it directly. | `src/infra/db.py:2612-2627` |
| B5 / A0 need new attention-budget plumbing | `founder_attention_log` table + `founder_attention_budget_minutes` column already exist. B5 extends with priority + deferral; doesn't create. | `src/infra/db.py` migration |
| A11 reuses `oncall_agent` shape | `oncall_agent.py` has hardcoded whitelist (restart_service, rollback, scale_up/down, drain_traffic, rotate_failed_key, archive_flake_test, escalate_to_founder). NOT configurable. A11 must extend the agent (lift handlers to a registry) before plugging in mention sources. | `src/agents/oncall_agent.py:16-80` |

Plus three audit findings that change scope materially:

| Finding | Consequence |
|---|---|
| **No email-send infra anywhere** (no SMTP, sendgrid, postmark, resend, SES, mailgun). i2p_v3.json phase 4.4 mentions "email subsystem design" as artifact text only. | A7 cold outreach + B1 lifecycle email + B2 email-channel changelog + B6 crisis email all need a shared email-base layer. New T2 dedicated to **shared infra dependencies** (email + langdetect) before any email-using pattern can ship. |
| **No in-app analytics event stream** (no PostHog / Segment / event_log table). | B14 / A11.r2 internal-signal mention source is *gated on Z6 growth shipping product event ingestion*. Not Z7 scope to build the event stream — Z7 consumes when ready. |
| **No multilingual / i18n / langdetect** | A8.r1 multilingual FAQ regen requires installing langdetect + indexing per-language Chroma collections. Bundled with email base in T2. |

## Frame (carried from v2, refined)

Zone identity unchanged: **founder-leverage layer; KPI is
founder-minutes-saved per mission**. Three sub-zones:

- **Bursty** — launch, demo, press kit, status-page during incident,
  crisis comms (event-shaped, 4–72-hour windows, rapid-response gating).
- **Continuous** — interaction log, support flywheel, mention monitor,
  investor digest, lifecycle email, changelog, reviews harvest,
  customer interviews (forever-running, low-rate, signal-extracting).
- **Cross-cutting** — founder briefing surface, attention budget queue,
  audit log, brand voice + copy compliance lint, consent ledger.

## Resource model (founder decisions, 2026-05-15)

Founder confirmed: KutAI is a **non-stop pipeline producing N isolated
products**. Z7 is **per-product** — each product has its own CRM,
press kits, launches, changelog, incidents, mentions, email
templates/sequences, reviews, brand voices, consent records, and ESP
account+domain. **No cross-product CRM** (D14 stays deferred).

But **shared infrastructure, not reinvented per product**:

| Shared (KutAI infra layer, built once) | Isolated (per-product data) |
|---|---|
| `src/integrations/email/` provider-abstracted send service | ESP account + sending domain + DKIM |
| `src/util/lang.py` langdetect | per-language Chroma collections |
| brand-voice lint engine, copy-compliance rule engine | `brand_voices/{audience}.md`, `channel_rules/` instances |
| recipe library, posthook registry, mr_roboto verbs | recipe *invocations*, posthook *runs* |
| oncall_agent handler-registry, mission templates | mention sources, launch instances |
| email template *schema*, crisis playbook *templates* | template/sequence/send rows, crisis_events rows |

Rule: **engines and schemas shared; data and accounts per-product.**
This is already KutAI's package architecture — Z7 follows it.

Founder is **solo** — single attention budget (B5), B6 escalation
chain is founder + counsel only, A0 briefing to one Telegram chat.
No `assignee` routing, no per-member budgets, no incident-commander
rotation. Drops that machinery from T1/T4.

## Net consequence

v3 ships in **6 tiers**, ~12 weeks total, parallel-safe within tier:

- **T1** lays cross-cutting infra. ~2 weeks. (was T1 in v2; expanded)
- **T2** ships shared dependencies (email base + langdetect). **NEW** in
  v3. ~1 week.
- **T3** ships bursty side. ~2 weeks. (was T2 in v2; +B3 +B6)
- **T4** ships continuous-people. ~2 weeks. (was half of T3 in v2; +B4 +B7)
- **T5** ships continuous-product. ~2 weeks. (was half of T3 in v2;
  +B1 +B2 +B8)
- **T6** ships opt-in + polish. ~2 weeks. (was T4+T5 in v2; +A11 ext.)

Net new infra: **15 tables, 11 posthooks (registry 25→36), ~32
mr_roboto actions, 11 jobs, 3 mission templates**. Email-send shared
service is a new src/integrations module reused across A7/B1/B2/B6.
oncall_agent gets a handler-registry refactor consumed by A11.

---

## Newly-identified patterns (B-series)

### B1 — Lifecycle email engine (onboarding / retention / churn-rescue)

**Trigger.** User signup → onboarding sequence (day 0/1/3/7/14).
Inactivity gap → re-engagement. Cancellation signal → churn-rescue.
Product-event triggers (first-success / power-user threshold / payment
failed / approaching-limit).

**Why distinct from A7 cold outreach.** Cold outreach is to strangers
under CAN-SPAM/GDPR opt-in framework. Lifecycle is to consenting users
under Terms of Service / Privacy Policy. Different deliverability
spine (transactional sending domain vs. marketing), different
unsubscribe model (preference center, not single suppression),
different content shape (helpful + behavioral, not pitch).

**Wiring.**
- New table `email_templates`: `template_id, kind ('onboarding' |
  'retention' | 'churn_rescue' | 'transactional' | 'announcement'),
  subject, body_md, variants_json, status ('draft' | 'approved' |
  'archived'), brand_voice_lint_pass (bool), copy_compliance_pass (bool)`.
- New table `email_sequences`: `sequence_id, name, trigger_kind, steps_json
  (ordered template_ids + delays), enabled (bool)`.
- New table `email_sends`: `send_id, user_id, sequence_id, template_id,
  scheduled_for, sent_at, opened_at, clicked_at, bounced_at, unsubscribed_at`.
- Event triggers consumed from product event stream (when Z6 ships):
  `signup`, `first_action`, `inactivity_7d`, `cancellation`,
  `payment_failed`. Until then, manual trigger via `/lifecycle trigger
  user_id sequence_id` Telegram cmd.
- New cron `src/app/jobs/lifecycle_email_send.py` — every 5min, picks
  due `email_sends`, calls shared email-send (T2), updates sent_at.
- Webhook receivers (shared with A7): `/webhook/email/open`,
  `/webhook/email/click`, `/webhook/email/bounce`, `/webhook/email/unsub`.
- Preference center page: `/email/preferences/{user_token}` — user
  toggles per-sequence subscription. Backed by `email_preferences`
  table.

**Founder track.** Authors templates once via founder_action card
(LLM-drafts; founder picks variant, A5+A6 lint must pass). Approves
sequences once. Reviews weekly digest of bounces / unsub spikes / sequence
performance in A0 briefing. Never approves individual sends (transactional
volume).

**Severity: SHOULD (T5).** Compounds on Z6 product event stream;
falls back to manual trigger if stream missing.

---

### B2 — Changelog as public artifact

**Trigger.** Mission with `goal:public_release` completes → changelog
entry generated. Or `/changelog publish` cmd.

**Distribution surfaces (4):**
- **In-app banner** (web + mobile), dismissable per-user, persistent
  in /changelog page.
- **RSS feed** at `/changelog.rss` (journalists + tools subscribe).
- **Email blast** to opted-in subscribers (lifecycle B1 sequence
  `kind='announcement'`).
- **Public webpage** at `/changelog` with version history, anchor
  links per release, share-to-social buttons.

**Wiring.**
- New table `changelog_entries`: `entry_id, version, released_at,
  title, body_md, kind_breakdown_json (added/changed/fixed/deprecated/removed),
  shipped_features_json, related_mission_ids[], external_url`.
- `actions/changelog/draft.py` — pulls from mission's git commits
  (range since last entry), maps commit-message conventional-commit
  prefixes to Keep-A-Changelog buckets, runs through A5 brand voice
  lint, runs through A6 copy compliance.
- `actions/changelog/publish.py` — non-LLM. Updates DB, regenerates
  RSS, triggers in-app banner refresh, queues B1 announcement send.
- New posthook `changelog_freshness` — fires monthly; if any
  `kind:'public_release'` mission has shipped without changelog entry,
  surfaces founder_action.
- A0 briefing surfaces "ready to publish changelog?" card after each
  release-tagged mission completes.

**Founder track.** Reviews drafted entry, edits prose (lint cannot
substitute for taste), approves publish. One card per release.

**Severity: SHOULD (T5).** Public face of product velocity.

---

### B3 — Status page + customer-facing incident comms

**Trigger.** Z8 oncall_agent fires alert → if `customer_impacting:true`
(new flag on alert payload), B3 mission spawns to manage public
comms. `/incident open` manual trigger also.

**Why distinct from Z8 oncall.** Z8 owns *fixing* (restart, rollback,
scale, drain). B3 owns *informing* (status page update, email subscribers,
post-incident postmortem). Different audience (engineers fixing vs.
customers waiting), different surface (internal alerts vs. public page).

**Wiring.**
- New table `incidents`: `incident_id, opened_at, resolved_at,
  severity ('critical' | 'major' | 'minor'), affected_components_json,
  customer_impact_summary, current_status_md, postmortem_url`.
- New table `status_updates`: `update_id, incident_id, posted_at,
  body_md, status_kind ('investigating' | 'identified' | 'monitoring' | 'resolved')`.
- Public status page `/status` — current up/down per component,
  active incidents, 90-day uptime stats. Static-rendered + cached.
- RSS at `/status.rss`. Email subscribers via B1 lifecycle email,
  sequence_kind='incident'.
- `actions/incident/draft_update.py` — LLM-bound (small model, OVERHEAD lane).
  Drafts customer-friendly status update from internal alert details.
  Critical: redacts internal hostnames / stack traces / customer-PII.
- founder_action card per status update for review pre-publish (max
  4hr SLA timer; if founder unavailable >SLA, B6 crisis playbook
  takes over).
- Postmortem template auto-drafted at incident resolve; founder edits
  + publishes within 7d.

**Founder track.** Reviews each customer-facing update pre-publish.
Edits postmortem prose. Decides which incidents merit email blast
(blast on critical only by default; configurable per status_kind).

**Severity: MUST (T3).** Trust-critical; failure mode (silent outage)
is a brand event.

---

### B4 — Meeting brief auto-generation (CRM amplifier)

**Trigger.** Founder schedules meeting with contact via `/meeting
@contact_handle YYYY-MM-DD HH:MM [purpose]`. System generates brief
delivered 30min before scheduled time.

**Why this is the highest-leverage CRM use case.** v2 had A10 CRM as
log; the actual leverage is *prep*, not retrieval. Founder-minutes-saved
per use is 15–30; founder uses recurring (~3-5 meetings/week).

**Brief contents:**
- Last interactions (chronological, last 5).
- Open follow-ups owed by us / by them.
- Recent product changes relevant to their stated interests
  (pulled by matching `relationships.notes_md` keywords against
  changelog B2 + roadmap).
- Mention-monitor hits about their company (A11) since last meeting.
- Recent shipped/deferred items in our missions touching their stated
  asks.
- Suggested talking points (3-5, LLM-drafted from above).
- Suggested asks of them (1-2, e.g. "intros to X" if their network
  matches a target).

**Wiring.**
- New table `meetings`: `meeting_id, contact_id, scheduled_for,
  purpose, brief_generated_at, brief_md, outcome_logged_interaction_id (FK to interactions)`.
- `actions/meeting/brief.py` — LLM-bound (medium model, MAIN_WORK lane).
  Pulls interactions + mentions + changelog + meeting context.
  Outputs structured Markdown.
- New cron `src/app/jobs/meeting_brief_dispatch.py` — every 5min,
  picks meetings with `scheduled_for - now in [25min, 35min] AND
  brief_generated_at IS NULL`, generates brief, surfaces in
  Telegram with founder_action card.
- Post-meeting prompt: 30min after scheduled_for, surfaces
  founder_action "log meeting outcome?" with template (what discussed,
  follow-ups owed, next step). On submit, creates `interactions` row
  + closes the meeting record.

**Founder track.** Reviews brief 30min pre-meeting (3-5 min skim).
Logs outcome 30min post-meeting (1 min).

**Severity: MUST (T4).** Highest founder-minutes-saved per use of any
Z7 pattern. Should land before A2 launch playbook (T3) if scheduling
allows.

---

### B5 — Founder attention budget queue

**Trigger.** Continuous. Every founder_action emit, attention budget
checks daily-cap and priority before surfacing.

**Why this matters.** Without it, A0/A2/A4/A6/A8/A9/A10/A11 + B1/B2/B3
all dump cards on founder simultaneously. Founder hits 30 cards in a
day, ignores all of them, the whole zone's leverage collapses.

**Wiring.**
- Extend existing `founder_attention_log` table with: `card_id (FK to
  founder_actions), surfaced_at, acted_at, deferred_to (nullable),
  attention_minutes (int)`.
- Add columns to `founder_actions`: `priority ('p0_blocking' | 'p1_today' |
  'p2_this_week' | 'p3_when_idle')`, `defer_until (nullable)`,
  `expires_at (nullable, auto-cancel if not acted)`.
- New module `src/app/attention_budget.py`:
  - `check_budget(today)` → returns remaining-minutes + top-priority queue.
  - `should_surface_now(card)` → boolean; defers if budget exhausted
    AND card priority < p0.
  - `next_review_window(card)` → schedules deferred cards for next
    morning's A0 briefing.
- Default cap from env: `FOUNDER_ATTENTION_DAILY_MINUTES=60`. Configurable
  per founder via `/attention budget 90`.
- A0 briefing renders attention queue as "today's cards" (p0+p1) +
  "this week" (p2) + "deferred / when idle" (p3) sections.
- Telegram cmds: `/attention status`, `/attention defer p2 to friday`,
  `/attention budget [minutes]`.

**Founder track.** Sets daily cap once. Reviews A0 morning queue.
Defers / acts / dismisses per card. Weekly summary of attention spent
+ minutes-saved (A0 KPI rollup).

**Severity: MUST (T1).** Foundational; everything else assumes it.

---

### B6 — Crisis comms tiered playbook

**Trigger.** A11 negative-sentiment cluster threshold OR B3 critical
incident OR `/crisis [kind]` manual.

**Tiers:**
- **Tier 1: brand misstep / pile-on** (e.g. tweet went wrong).
  Holding-statement draft + delete-or-defend decision card +
  monitor-cadence config. Recovery via apology + acknowledgment +
  next-step.
- **Tier 2: outage / data issue** (e.g. extended downtime).
  Status-page update (B3 escalates), customer email, refund/credit
  policy decision card. Recovery via postmortem + remediation comms.
- **Tier 3: security incident / breach** (e.g. credentials leaked).
  Counsel-engaged-yet? founder_action. 72h-disclosure timer if GDPR
  breach. Pre-drafted regulator notice (jurisdiction-aware via
  compliance_overlay). Customer notification (legally-reviewed
  template). Press response.
- **Tier 4: existential / legal** (e.g. lawsuit, regulatory action,
  public flame from major figure). Counsel-led; agent freezes
  marketing comms (auto-pauses A2/B1/B2 sends), surfaces emergency
  founder_action with full context dump.

**Wiring.**
- New table `crisis_events`: `event_id, opened_at, tier (1-4),
  source ('mention_monitor' | 'incident' | 'manual'), summary,
  status, resolved_at, postmortem_url`.
- `playbooks/crisis_comms_tier{1-4}.md` artifacts (founder authors
  Tier 4 once; agent drafts T1-3 from templates).
- `actions/crisis/freeze_marketing.py` — pauses A2 in-flight launches,
  blocks B1 announcement sends, blocks A7 outreach. Reversible via
  `/crisis resume`.
- `actions/crisis/draft_holding.py` — LLM-bound; reads tier playbook +
  event context; outputs holding statement variants.
- 72h-disclosure timer cron for Tier 3 (jurisdiction-aware via
  compliance_overlay): every 6h, surfaces escalating reminder.
- Counsel-engaged-yet check: founder_action card with two-button
  ack ("yes, counsel notified" / "no, escalate now").

**Founder track.** Picks tier (or accepts agent's tier classification),
approves freeze + holding statement, makes counsel call (Tier 3+),
edits regulator/customer notices, owns the entire response thread.
Agent drafts + tracks + reminds.

**Severity: SHOULD (T3).** Low-frequency / high-stakes. Cheap to
build the playbook scaffolding; expensive to lack when needed.

---

### B7 — Customer interview / call notes pipeline

**Trigger.** Founder runs `/interview start @contact_handle`, records
audio (uploads file or starts in-Telegram voice memo). On stop,
pipeline runs.

**Pipeline:**
1. **Transcribe** — Whisper (local CPU model — sentence-transformers
   ecosystem already in repo). Cheap, private, no vendor.
2. **Summarize** — LLM-bound (medium model, OVERHEAD lane). Output:
   bullets per topic, quotes (verbatim), insights (founder's
   interpretation), action items (assigned to mission backlog).
3. **Tag** — extract product-area mentions (matched against current
   backlog), customer-segment indicators, competitor mentions,
   pricing-sensitivity signals.
4. **Cross-link** — append summary to A10 `interactions` row (via
   `kind='interview'`), enqueue action items as candidate `tasks`,
   add quotes to A4 press_kit_quotes (founder approves consent).

**Wiring.**
- New table `interview_notes`: `note_id, contact_id, started_at,
  duration_minutes, transcript_md (gz-blob), summary_md, quotes_json,
  insights_md, action_items_json, audio_path`.
- `actions/interview/transcribe.py` — Whisper-CPU, runs in background.
- `actions/interview/summarize.py` — LLM-bound, reads transcript,
  writes structured output.
- `actions/interview/cross_link.py` — non-LLM, populates A10/A4/backlog.
- Telegram cmds: `/interview start @handle`, `/interview stop`,
  `/interview list [@handle]`.
- A0 briefing surfaces "review interview note" card per completed
  interview (founder edits insights, approves quote consent requests).

**Founder track.** Records the call (with consent). Reviews summary +
edits insights. Approves which quotes can go to press kit (sends
consent request to interviewee via B1 transactional email).

**Severity: NICE (T4).** High leverage when founder is doing
customer-development sprints; low usage otherwise. Ship as opt-in.

---

### B8 — Reviews harvest (G2 / Trustpilot / AppStore / PlayStore / similar)

**Trigger.** Daily cron polls per-platform review APIs / scrape
endpoints. New review → ingest → classify → surface.

**Wiring.**
- New table `external_reviews`: `review_id, platform ('g2' | 'capterra'
  | 'trustpilot' | 'appstore' | 'playstore' | 'producthunt' | 'shopify' |
  'chrome_store'), external_id, posted_at, author, rating,
  body_md, sentiment, replied_at, reply_body_md`.
- `actions/reviews/poll/{g2,capterra,trustpilot,appstore,playstore,...}.py`
  — per-platform fetcher. Free APIs where available; vecihi (Z6)
  scraper fallback for platforms without free API.
- `actions/reviews/classify.py` — LLM-bound (small model, OVERHEAD).
  Sentiment + theme tag (UX / pricing / bug / feature-request /
  support / generic-positive / generic-negative).
- `actions/reviews/draft_reply.py` — LLM-bound. Generates reply per
  brand voice + platform conventions.
- A0 briefing surfaces:
  - 5-star reviews (founder reads + considers quote consent for A4).
  - 1-2-star reviews (founder reviews + decides reply).
  - Theme clusters across reviews ("3 reviews this month mention
    pricing confusion → consider B2 changelog post explaining tiers").
- Bug-tagged reviews → enqueue investigation task in mission backlog.

**Founder track.** Reads daily/weekly digest. Approves replies (never
auto-reply). Decides which to act on at product level.

**Severity: SHOULD (T5).** Compounding signal source for product
direction.

---

### B9 — Audit log for external comms

**Trigger.** Continuous. Every external publish/send/upload generates
an immutable record post-execution.

**Why distinct from Z6 action_confirmations.** action_confirmations
logs *founder approvals*, indexed by approval-time. B9 logs *agent
sends*, indexed by send-time + recipient + content-hash. Needed for:
post-incident review ("who got the bad pricing email?"), legal
discovery, deliverability investigation, dedup ("did we already send
this?"), revocation tracing.

**Wiring.**
- New table `external_comms_log`: `log_id, sent_at, channel
  ('email' | 'tweet' | 'reddit_post' | 'hn_post' | 'ph_post' |
  'linkedin_post' | 'press_release' | 'blog_post' | 'sms' | 'webhook'),
  recipient (nullable for broadcast), recipient_count (int),
  content_hash, content_md (gz-blob), source_mission_id,
  source_action_id, vendor_call_id (FK to action_confirmations),
  reversibility ('full' | 'partial' | 'irreversible'),
  revoked_at (nullable), revoke_reason`.
- New mr_roboto wrapper `audit_log.py` — every external-publish verb
  decorates with audit-write before+after the vendor_call.
- Search cmd `/audit search recipient:journalist@bigpub.com` —
  Telegram inline result.
- New posthook `audit_completeness_check` — fires hourly; for any
  `vendor_call` with `reversibility != full` and no audit-log row
  within 5min, raises ops alert (Z8 oncall).

**Founder track.** Searches via Telegram during incidents / discovery
events. Otherwise invisible.

**Severity: MUST (T1).** Compliance + post-incident review prerequisite.
Cheap to build; expensive to retrofit later.

---

## A-series sub-pattern refinements

Compact format: `pattern.refinement_id` — one-paragraph delta + wiring.

**A2.r1 — `launch_readiness_gate` posthook.** Pre-T-0 hard checks
before A2's `publish_synchronized` fires. Verifies: (a) site loads
under expected traffic (synthetic-check kit Z2 recipe), (b) payment
flow E2E test green within 24h, (c) support_tier1 has launch-FAQ
indexed, (d) A6 copy_compliance pass on all channel drafts, (e) A4
press kit published with permanent URL, (f) status page exists (B3),
(g) crisis playbook exists (B6 Tier 1+2). Blocker on any → freezes
T-0, surfaces founder_action "override or fix?".

**A3.r1 — Demo accessibility tracks.** Beyond captions: alt text per
thumbnail, audio-description track for visual-only sequences,
ASL-inset for premium launches (founder-uploaded), keyboard-nav
walkthrough variant. New `actions/demo/accessibility_pass.py`. New
posthook `demo_accessibility_check`.

**A4.r1 — Audience-segmented press kits.** Single mission produces
multiple kit variants: investor-kit (financial-leaning), journalist-kit
(news-leaning), partner-kit (integration-leaning), candidate-kit
(culture-leaning). Each pulls different sections from same source.
Same versioning + permanent URL pattern, suffix differentiator
(`/v{N}/investor/`, `/v{N}/journalist/`).

**A5.r1 — Multi-voice brand support.** `brand_voice.md` becomes
`brand_voices/{audience}.md` directory: marketing, support, investor,
technical-blog, social, recruiting. Each artifact has same shape
(examples + prohibitions + length + reading-level + pronouns + tone).
Lint reads correct voice based on artifact's `audience` metadata.

**A6.r1 — `channel_rules/` library consumer.** Per-channel rule files
(`channel_rules/{hn,ph,reddit,twitter,linkedin,press,investor_email,
support_reply}.md`) consumed by `copy_compliance_review` posthook
when artifact has `channel` metadata. Rules: max-length, banned-words,
required-disclosures, image-requirements.

**A7.r1 — Outreach reply-handling sub-pattern.** When ESP webhook
receives reply event (matched by Reply-To header + send_id):
classify reply (positive_interest / negative / unsubscribe-request /
out-of-office / bounce / question), suggest follow-up draft if
positive, auto-suppress if unsubscribe, log to `interactions` (A10)
linked to `outreach_sends.send_id`, surface in A0 briefing if
positive cluster.

**A8.r1 — Multilingual FAQ regen.** langdetect on each ticket
(installed in T2). Cluster within-language (don't cross-lingual
cluster). Per-language Chroma collections (`support_docs_en`,
`support_docs_tr`, etc.) and per-language `faq_{lang}.md` artifacts.
A8 regen runs per language.

**A8.r2 — Use existing `tickets` table (correction).** v2 proposed
`support_low_confidence` table; that's already covered by `tickets`
(`confidence`, `escalated_to_founder`, `sentiment` columns). A8
queries `tickets` directly; new `docs_gap_log` and
`press_kit_quotes` tables remain net-new.

**A9.r1 — Segmented investor templates.** Three template variants:
`pre_investor_pitch_bullets.md` (warm intro before raise),
`current_investor_update.md` (already-on-cap-table monthly),
`advisor_check_in.md` (advisor cadence). A9 emits one per
recipient-category from `relationships.category`.

**A10.r1 — Consent ledger.** New table `consent_records`: `contact_id,
purpose ('quote_use' | 'data_processing' | 'marketing_email' |
'interview_recording' | 'case_study'), granted_at, expires_at,
source_evidence_url, revoked_at`. Every Z7 surface that touches
contact data checks consent first; if expired/missing/revoked,
surfaces founder_action "request consent before proceeding?".

**A10.r2 — Meeting brief integration (B4 ↔ A10).** `/meeting`
auto-creates `interactions` row at meeting end via B4's outcome-log
prompt. CRM stays current without separate logging step.

**A11.r1 — `oncall_agent` configurability extension (correction).** v2
claimed A11 reuses oncall_agent shape. Audit found agent has
hardcoded handler whitelist. Refactor first: lift handlers to a
registry (`packages/coulson/agent_handlers/registry.py`); existing
ops handlers register via `register_handler('ops', ...)`; new mention
handlers register via `register_handler('mention', ...)`. Then A11
ships as a configured agent with mention-source + signal-gate
handlers.

**A11.r2 — Internal-signal mention source.** Beyond external polling
(HN/Reddit/X/etc.), watch *our own product event stream* for negative
in-app signals: support-message containing
"frustrated|cancel|broken|never works", pricing-page rage-clicks,
feature-request submissions tagged urgent. Gates on Z6 product event
ingestion shipping; until then, watch `tickets.sentiment='negative'`
+ `tickets.confidence < 0.5` clusters as proxy.

---

## Deferred patterns (D-series, with rationale)

These are real humanish areas that v2/v3 deliberately omits. Documented
to make the gap-map honest, not silently incomplete. Promote to active
when trigger condition met.

| ID | Pattern | Why deferred | Promotion trigger |
|---|---|---|---|
| D1 | Hiring funnel (JD drafting, candidate pipeline, scoring) | Most KutAI users are solo / pre-hire | First active hire (founder runs `/hire start`) |
| D2 | Localization beyond language (currency, date, address, regional) | English+Turkish coverage adequate v3 | First non-EN/TR market launch via A2 |
| D3 | Customer health scoring / proactive churn intervention | Z6 growth zone owns health metrics; pull when Z6 ships event stream | Z6 product event ingestion live |
| D4 | Demo environment / sandbox account provisioning | Z2 product-side concern (recipe), not Z7 | Z2 ships `demo_environment` recipe |
| D5 | Newsletter content engine (recurring publishing cadence) | Overlaps B2 changelog + A12 marketing copy until founder demonstrates separate cadence need | Founder runs `/newsletter setup` |
| D6 | Marketplace / integration listings (Slack dir, Zapier, Vercel templates) | Z6 vendor adapter territory | Z6 vendor_call ships first integration listing |
| D7 | Ambassador / power-user program | Premature; needs A11 mentions + A8 escalations + B7 interviews to surface candidates | Founder identifies ≥10 candidates manually |
| D8 | Founder personal-brand audit + posting schedule | Too founder-specific to systematize v1 | Founder runs `/personal_brand audit` |
| D9 | Advisor agreements / cap-table tracking | Z6 legal/vendor territory; cap-table needs separate primitive | Z6 ships cap-table-aware vendor_call |
| D10 | Community moderation workflow (ban/warn/engage) | Premature; agent-generated comms in community space too risky pre-trust | Founder reaches Tier 2 community size (e.g. ≥1k members) |
| D11 | In-app help / tooltips / onboarding tour | Z2 product-side concern | Z2 ships product-UI guidance recipe |
| D12 | NPS / CSAT survey infra | Subset of B1 lifecycle email; ship as B1 sequence variant | Founder requests dedicated NPS dashboard |
| D13 | Podcast / public-speaking prep (talking points, hot-takes library) | Founder-specific; low-frequency | Founder requests / books first podcast |
| D14 | Vendor relationship tracking (vs. Z6 vendor adapters) | Z6 owns vendor-as-API; Z7 vendor-as-relationship deferred until distinct from A10 CRM | Founder logs ≥5 distinct vendor contacts via A10 |
| D15 | Tax / insurance / accounting renewal calendar | Z6 territory + low-frequency; doesn't need Z7 surface | Z6 ships compliance-renewal cron |

---

## Tier plan (revised, 6 tiers)

### T1 — Cross-cutting infra (~2 weeks, 4 agents)

- **T1A — Founder briefing surface (A0).** Same as v2 plus: integrate B5
  attention-budget queue rendering into briefing layout, weekly
  founder-minutes-saved rollup section.
- **T1B — Brand voice lint multi-voice (A5 + A5.r1).** `brand_voices/`
  directory + audience-aware lint posthook (kind 27).
- **T1C — Copy compliance + channel rules (A6 + A6.r1).**
  posthook (kind 28) + `channel_rules/` library + jurisdiction-aware
  rule application.
- **T1D — Attention budget queue + audit log (B5 + B9).** Two agents
  bundled: B5 extends `founder_attention_log` + adds priority/defer
  on `founder_actions`; B9 ships `external_comms_log` table +
  audit_log mr_roboto wrapper + completeness_check posthook (kind 29).

Acceptance: A0 briefing renders with attention-queue sections + KPI
rollup; A5/A6 fire on test artifacts and produce expected violations;
attention budget defers cards when daily-cap exceeded; external
publish creates audit_log row within 5min.

### T2 — Shared-infra dependency layer (~1 week, 2 agents)

- **T2A — Email-send shared service (free-tier-first, per-product
  provider).** New `src/integrations/email/` with provider-abstracted
  `send(to, subject, body_md, headers, template_id, idempotency_key)`.
  Founder decision (2026-05-15): **no paid email until a product
  makes money** — provider is chosen per-product at setup, default to
  a free tier.
  - **Provider registry**: each provider is a config-driven adapter
    (`providers/{brevo,resend,postmark,ses}.py`) implementing the
    same `send` + webhook contract. New per-product config row
    `product_email_config`: `product_id, provider, from_domain,
    api_key_ref (credential_store), monthly_quota, tier ('free' | 'paid')`.
  - **Free defaults**: Brevo (300 emails/day, free forever — best for
    low steady volume) or Resend free (3k/mo, 100/day — best DX).
    Founder picks per product via setup founder_action; recommend
    Brevo for lifecycle/transactional (daily cap fits drip cadence),
    Resend if founder wants the modern API.
  - **Paid upgrade path**: when a product has revenue, founder flips
    `tier='paid'` via `/email upgrade <product>` — same adapter,
    higher quota, no code change. Postmark/SES become options then.
  - **Quota guard**: shared sender enforces per-product
    `monthly_quota`; over-quota queues sends to next window + surfaces
    A0 card "product X near email cap — upgrade tier?".
  SPF/DKIM/DMARC config docs + per-product setup founder_action.
  Shared webhook receiver dispatch for open/click/bounce/unsub
  events (routed by product_id). Suppression-list base, per-product
  (used by A7 + B1). Test mode (sends to founder-only inbox).
- **T2B — Multilingual base.** Install langdetect, add
  `src/util/lang.py`, per-language Chroma collection helper, per-language
  artifact path convention (`faq_{lang}.md`). Used by A8.r1.

Acceptance: T2A sends test email via the free-tier provider
(Brevo/Resend) sandbox for a sample product config, webhook records
open event into receiver routed by product_id, quota guard blocks an
over-cap send. T2B detects language of synthetic TR + EN tickets,
routes to per-language collection.

### T3 — Bursty side (~2 weeks, 5 agents)

- **T3A — Launch playbook + readiness gate (A2 + A2.r1).**
- **T3B — Demo pipeline + accessibility (A3 + A3.r1).**
- **T3C — Press kit + audience-segmented variants (A4 + A4.r1).**
- **T3D — Status page + customer-incident comms (B3).**
- **T3E — Crisis comms tiered playbook (B6).**

Acceptance: `/launch` runs full readiness gate, blocks on missing B3
status page; demo pipeline produces accessible MP4 + .vtt + alt-text
manifest; press kit publishes 4 audience variants under permanent
URLs; B3 status page renders, status update goes through founder
review pre-publish; B6 tier classification + freeze marketing works.

### T4 — Continuous people side (~2 weeks, 4 agents)

- **T4A — FAQ flywheel (A8 + A8.r1 + A8.r2).** Reads existing
  `tickets` table; multilingual regen.
- **T4B — CRM as log + consent ledger (A10 + A10.r1).** `relationships`
  + `interactions` + `consent_records` tables; 5 Telegram cmds
  including `/consent`.
- **T4C — Meeting brief auto-generation (B4).** Highest-leverage CRM
  amplifier; ship in T4 alongside A10.
- **T4D — Customer interview pipeline (B7).** Whisper-CPU + summarize
  + cross-link to A10/A4/backlog.

Acceptance: A8 regen produces FAQ from `tickets` confidence-flagged
rows, supports TR + EN; A10 cmds round-trip + consent surfaces
correctly; B4 brief generates 30min pre-meeting + outcome logs into
A10; B7 transcribes synthetic 5-min audio, summarizes, cross-links.

### T5 — Continuous product side (~2 weeks, 4 agents)

- **T5A — Investor bullets + segmented (A9 + A9.r1).**
- **T5B — Lifecycle email engine (B1).** Templates + sequences +
  preference center; manual trigger only until Z6 event stream ships.
- **T5C — Changelog as public artifact (B2).** Draft from git +
  publish to RSS/in-app/email/page.
- **T5D — Reviews harvest (B8).** Per-platform fetchers + classify +
  draft replies + theme cluster.

Acceptance: A9 produces segmented bullets per recipient category; B1
sends test sequence to founder inbox via T2 email base; B2 publishes
to all 4 surfaces with brand-lint pass; B8 polls 3+ platforms,
clusters themes, surfaces in A0.

### T6 — Opt-in + polish (~2 weeks, 5 agents)

- **T6A — Cold outreach with deliverability spine (A7 + A7.r1).**
  Reuses T2 email base; reply-handling sub-pattern.
- **T6B — `oncall_agent` handler-registry refactor + mention monitor
  (A11.r1 + A11 + A11.r2).** Refactor agent first (deps T6C T6D),
  then ship mention monitor as configured agent. Internal-signal
  proxy via tickets (until Z6 event stream).
- **T6C — Marketing copy generator (A12).**
- **T6D — Demo distribution (A3 distribute stage).**
- **T6E — Cross-mission launch lessons consumer.** A2's T+7d lessons
  feed next launch's T-72h drafting via STACK_BLOCKS.

Acceptance: A7 sends to test inbox via T2, suppression filters
known-bad email, warmup blocks over-quota; oncall_agent registry
refactor preserves Z8 ops behavior + accepts mention handlers; A12
produces copy passing A5 lint; A3 distribute uploads to YouTube
unlisted; second launch consumes first launch's lessons.

---

## Wiring summary (revised)

| Item | New tables | New posthooks | New mr_roboto verbs | New jobs | New mission templates |
|---|---|---|---|---|---|
| A0 | `mission_briefings` | `briefing_compose` (26) | — | `daily_briefing` | — |
| A2 | — | `launch_readiness_gate` (30) | `launch_drafts/*` × 5, `publish_synchronized`, `launch_response_monitor` | — | `launch_playbook.json` |
| A3 | — | `demo_artifact_check`, `demo_accessibility_check` (31) | `demo/{storyboard,record,edit,caption,accessibility_pass,distribute}` | — | — |
| A4 | `press_kits`, `press_kit_quotes` | `press_kit_freshness` | `press_kit/{assemble,publish}` (with audience variants) | — | — |
| A5 | — | `brand_voice_lint` (27) | — | — | — |
| A6 | — | `copy_compliance_review` (28) | — | — | — |
| A7 | `outreach_suppression`, `outreach_warmup`, `outreach_sends` | `outreach_deliverability_check` | `outreach/{draft,send,handle_reply}` | — | — |
| A8 | `docs_gap_log` (uses existing `tickets`) | `documentation_gap_detect` | — | `faq_regen` (multilang), `quote_harvest` | — |
| A9 | — | — | — | `investor_bullets` (segmented) | — |
| A10 | `relationships`, `interactions`, `consent_records` | — | — | `follow_up_reminder` | — |
| A11 | `mentions` | — | `mention_polls/{hn,reddit,google,twitter,discord}`, `internal_signal_poll` | — | `mention_monitor.json` |
| A12 | — | — | `marketing_copy` | — | — |
| B1 | `email_templates`, `email_sequences`, `email_sends`, `email_preferences` | — | `email/send_via_provider` | `lifecycle_email_send` | — |
| B2 | `changelog_entries` | `changelog_freshness` | `changelog/{draft,publish}` | — | — |
| B3 | `incidents`, `status_updates` | — | `incident/{draft_update,publish_status,draft_postmortem}` | — | `incident_comms.json` |
| B4 | `meetings` | — | `meeting/{brief,outcome_prompt}` | `meeting_brief_dispatch` | — |
| B5 | extends `founder_actions`, `founder_attention_log` | — | — | — | — |
| B6 | `crisis_events` | — | `crisis/{freeze_marketing,draft_holding,disclosure_timer}` | — | — |
| B7 | `interview_notes` | — | `interview/{transcribe,summarize,cross_link}` | — | — |
| B8 | `external_reviews` | — | `reviews/poll/{g2,capterra,trustpilot,appstore,playstore,producthunt,...}`, `reviews/{classify,draft_reply}` | `reviews_poll_daily` | — |
| B9 | `external_comms_log` | `audit_completeness_check` (29) | `audit_log` wrapper | — | — |

**Net:** 16 new tables, 11 new posthooks (registry 25→36), ~32 new
mr_roboto actions, 11 new jobs, 3 new mission templates.

**Per-product scoping (founder decision 2026-05-15).** Every Z7 table
above carries a `product_id` column (FK to the product's
mission-graph root). All CRM / press / launch / email / incident /
mention / review data is product-scoped — no cross-product joins.
The 16th table is `product_email_config` (per-product ESP provider +
domain + quota + tier; see T2A). Webhook receivers, suppression
lists, attention budget, and audit log all filter by `product_id`.

Plus shared infra (T2): `src/integrations/email/` (provider-abstracted,
free-tier-first, per-product provider registry — Brevo/Resend free
defaults, Postmark/SES paid upgrade), `src/util/lang.py` (langdetect
wrapper), per-language Chroma collection pattern. Plus refactor (T6):
`oncall_agent` handler-registry lift.

---

## Dependencies (revised)

**Inbound (must be live first):**
- All v2 dependencies still apply.
- T2 shared email service must land before T5 + T6 (B1, A7, B2 email
  channel, B6 crisis email).
- T2 langdetect must land before T4 (A8.r1).
- T6B oncall_agent refactor must land before A11 ships.
- B5 attention budget (T1) must land before any tier-3+ pattern
  surfaces a founder_action.
- B9 audit log (T1) must land before any external publish via T2 email.

**Soft dependencies (graceful degrade):**
- B1 lifecycle email triggers — production event stream from Z6
  growth. Falls back to manual `/lifecycle trigger` until then.
- A11.r2 internal-signal source — Z6 product event stream. Falls
  back to `tickets.sentiment` proxy until then.
- B8 reviews harvest API costs — some platforms (G2 paid). Free-tier
  + scrape-via-vecihi fallback for paid platforms.

**Outbound (Z7 produces; others consume):**
- A0 briefing surface read by every zone.
- A10 + A11 + B7 + B8 emit `mission_lessons` rows for cross-mission
  learning loop.
- B5 attention budget consumed by every founder-action emitter.
- B9 audit log queryable by Z6 compliance flows + Z8 incident
  postmortem flows.

---

## Founder track (revised + expanded)

| Pattern | Founder action per use | Estimated minutes |
|---|---|---|
| A0 briefing + B5 queue | Read morning digest with attention-prioritized sections | 5–15/day |
| A2 launch playbook | Approve drafts (T-24h), publish (T-0), respond to flagged engagement | 30–120/launch |
| A3 demo + accessibility | Edit storyboard cut points; review accessibility manifests | 10–30/launch |
| A4 press kit (4 variants) | Approve assembled kits pre-publish | 10/launch + 5/quarter |
| A5 multi-voice lint | Author voice docs once per audience; accept/override per artifact | 60 once + 10s/artifact |
| A6 copy compliance | Override warnings; blockers force rewrite | 10s–2min/artifact |
| A7 cold outreach | One-time DKIM setup; approve each batch; review reply digest | 30 once + 5/batch + 5/week replies |
| A8 FAQ flywheel | Approve weekly drafts (per-language); decide quote consent | 15/week |
| A9 segmented investor bullets | Edit bullets per audience; copy to clipboard | 20/month |
| A10 CRM log + consent | `/log @x ...`; approve consent requests | 5s/interaction; 30s/consent |
| A11 mention monitor | Approve channel adds; act on score>=7; review daily digest | 5/day after launch |
| A12 marketing copy | Approve variant; paste to Webflow | 15/site update |
| B1 lifecycle email | Author templates once; approve sequences once; weekly perf review | 60 once + 10/week |
| B2 changelog publish | Edit drafted entry per release | 10/release |
| B3 status page | Review each customer-facing update pre-publish; edit postmortem | 5–15/incident |
| B4 meeting brief | Read brief 30min pre; log outcome 30min post | 5/meeting |
| B5 attention budget | Set cap once; review daily queue (in A0) | 5 once |
| B6 crisis comms | Pick tier; approve freeze + holding statement; counsel call (T3+) | 30–240/event |
| B7 interview pipeline | Record call (with consent); review summary + edit insights | 10/interview |
| B8 reviews harvest | Read digest; approve replies | 10/week |
| B9 audit log | Search during incidents | 5/incident, otherwise 0 |

KPI rollup: across all patterns, target **30+ minutes/day saved
weighted-average** by month 3 of Z7 in production.

---

## Open questions (revised + new)

Carried from v2 (resolved with defaults in v2 — re-stated below):
- Brand voice doc — authored as Z7-T1 artifact (founder-led).
- Press kit hosting — `.env`-configured S3/R2.
- Twitter API cost — off by default.
- Sentiment classifier — small CPU model (RoBERTa-sentiment).
- Investor distribution — copy-to-clipboard only.
- CRM email integration — out of scope.
- Crisis comms checklist — promoted to B6 in v3.

**New (v3):**

- **Email provider choice (T2A). RESOLVED 2026-05-15.** Founder: no
  paid email until a product earns. Provider is per-product, free-tier
  by default (Brevo 300/day forever, or Resend free 3k/mo). Provider
  registry + `product_email_config` row; paid upgrade (Postmark/SES)
  is a per-product `tier` flip when revenue justifies.
- **Status page hosting (B3).** Self-hosted (route in main app) vs.
  external (statuspage.io, $29/mo) vs. open-source (Cachet self-host).
  Lean self-hosted route in main app — minimal infra, full control,
  no vendor.
- **Whisper model size (B7).** base.en (~150MB, real-time on CPU) vs.
  small (~500MB) vs. medium (~1.5GB). Lean small for accuracy/speed
  balance; founder can switch via env var.
- **Reviews platforms scope (B8).** Free APIs only v1 (G2 free tier,
  AppStore RSS, PlayStore unofficial) vs. paid (G2 full, Trustpilot
  Pro). Lean free + scrape-fallback v1.
- **Attention budget enforcement strictness (B5). RESOLVED 2026-05-15.**
  Soft-warn: all cards surface, over-budget p1/p2/p3 flagged
  "beyond today's budget" + pushed below the fold in A0. p0 always
  surfaces. No withholding. Re-evaluate from KPI signal after month 1.
- **B6 marketing freeze blast radius. RESOLVED 2026-05-15.** Products
  are isolated (founder decision), so a crisis freezes only the
  affected product's in-flight A2/B1/B2 sends — other products keep
  running. `crisis_events.product_id` scopes the freeze.
- **Per-product vs. per-org Z7 instances. RESOLVED 2026-05-15.**
  Founder: KutAI is a non-stop pipeline of N isolated products.
  Z7 is per-product; D14 cross-product CRM stays deferred. Shared:
  KutAI's infra layer (email service, langdetect, lint engines,
  recipes, mr_roboto verbs, oncall_agent, mission templates). See
  Resource model section above.

---

## Agent task brief

When picking up this doc:

1. Read 00-README + 06-real-world-bridge-v2 + 08-operations-v2 +
   09-growth + this doc + the v2 doc (07-humanish-layers-v2.md, kept
   for delta context, not authoritative). Also read the audit table
   above so wiring corrections are explicit.
2. Confirm shipped infra still as audited (founder_actions,
   support_tier1, oncall_agent hardcoded handlers, mission_lessons,
   tickets table, founder_attention_log, reversibility taxonomy,
   action_confirmations) — grep call sites per
   `feedback_audit_call_sites`.
3. Build T1 first (4 agents). Tier-by-tier per
   `feedback_canonical_first_for_tier3plus`. T2 is small (2 agents)
   but blocks T5+T6.
4. After each tier merge: tag `z7-t{N}-shipped`, run full test suite
   with timeout, update memory.
5. Per `feedback_no_tier_pauses`: do not stop between tiers for
   approval; only stop on a genuine question. Open questions above
   resolvable from defaults; D-series promotion gates only fire on
   founder request.
6. **Watch for v2 wiring artifacts during implementation.** v2
   pre-existed v3 by hours; some agent task briefs / prior planning
   documents may reference v2 wiring (e.g. `support_low_confidence`
   table). v3 supersedes — use this doc.
7. After T6: tag `z7-complete-{date}`, write `project_z7_complete.md`,
   update zone map in 00-README.

## Updates

- 2026-05-15 (v3, founder decisions) — 4 decisions resolved:
  (1) build order = default T1→T6;
  (2) email = free-tier-first, per-product provider registry
  (Brevo/Resend free defaults, Postmark/SES paid upgrade per-product
  when revenue justifies) — added `product_email_config` table +
  quota guard, T2A rewritten;
  (3) solo founder — single attention budget, B6 escalation =
  founder + counsel, no assignee/per-member machinery;
  (4) N isolated products sharing KutAI infra layer — added Resource
  model section, every Z7 table now `product_id`-scoped (16 tables),
  B6 freeze scoped per-product, D14 cross-product CRM stays deferred.
- 2026-05-15 (v3) — Second-pass audit. v2 had real reframe but missed
  9 humanish areas + 13 sub-patterns + 3 wiring corrections + 15
  explicitly-deferrable patterns. v3 ships 21 active (A0–A12 +
  B1–B9), 15 deferred (D1–D15), 6 tiers (~12 weeks). Net new infra:
  16 tables, 11 posthooks (25→36), ~32 mr_roboto actions, 11 jobs,
  3 mission templates, 1 shared email-base service, 1 langdetect
  base, 1 oncall_agent handler-registry refactor.
- 2026-05-15 (v2 retained for delta context) — first reframe of
  v1's flat A–J list into bursty/continuous/cross-cutting; introduced
  founder-minutes-saved KPI; rejected investor auto-prose +
  CRM-as-graph antipatterns; demo as 4-stage pipeline; cold outreach
  deliverability spine added.
- 2026-05-08 (v1) — initial doc; pairs Z4 + Z8 because both are
  human-irreplaceable + helper-shaped. Honest framing: lowest
  leverage from automation work. (Reframe rejected in v2; expanded
  in v3.)
