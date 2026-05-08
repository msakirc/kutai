# Z4 + Z8 — Humanish layers (launch + people)

## Frame

Where automation hits its ceiling. Marketing copy, brand voice, story,
press, customer relationships, investor relations, community, hiring.
Agent's job here is to **support, not replace** — drafts, summaries,
helpers, memory — while the founder does the relationship-shaped work.

Pairs Z4 (launch) + Z8 (people) because both share "human-irreplaceable"
problem space. Different from Z6 (growth) which has measurable signal
loops. Here the signal is fuzzy, taste-laden, asymmetric.

## Current state

- Phase 11 has documentation steps; phase 13 has marketing/launch steps; phase 14 has post-launch monitoring.
- No marketing-site copy generation flow.
- No press kit / pitch deck assembly.
- No customer-support automation; tickets handled (or not) outside the system.
- No investor-update generator.
- No CRM / relationship memory.
- No community / forum monitoring.
- No agent participation in launch channel selection (HN / PH / Twitter / Reddit).

## Gaps

### Fixable by automation (as helper, not driver)

**A. Marketing site as workflow output**
- Agent generates draft (hero, features, pricing, FAQ, testimonial-placeholder).
- Founder edits in Telegram thread or web editor.
- Iterate via mission thread comments.
- Output: deployable marketing site (separate from product app).

**B. Channel-specific launch templates**
- Library: HN-launch-post, ProductHunt asset bundle, Twitter thread, Reddit post by subreddit, LinkedIn post, dev.to / Medium templates.
- Agent populates with product specifics; founder picks channel + tone + timing.
- Tracks responses (clicks, upvotes, comments) for next-launch tuning.

**C. Demo / sales kit**
- Auto-generated screen-recordings via playwright.
- FAQ assembled from spec + early ticket queries.
- Objection-handling doc (drawn from competitor positioning + spec).
- Pricing comparison table.
- Useful even if founder does the call.

**D. Press kit**
- One-page summary, founder bios (founder-supplied), screenshots, logo, quotes.
- Mechanical to assemble; agent maintains over time as product evolves.

**E. Cold-outreach helper**
- Personalized email drafts based on prospect data (from CRM input, LinkedIn / company-website scrape).
- Founder reviews, sends.
- Tracks response rates; refines templates.

**F. Customer support — tiered**
- Tier 1 (agent): answers from FAQ + product docs + common questions.
- Tier 2 (escalation): complex / angry / billing → founder, with ticket summary + suggested response.
- Source: Telegram bot for early-stage; integrate Intercom/Zendesk/Help Scout when scale demands.
- Memory: every ticket logged; FAQ updates suggested when patterns emerge.

**G. Investor updates**
- Monthly digest from metrics ([09-growth.md](09-growth.md)) + qualitative signals (tickets, reviews, mentions).
- Agent drafts; founder edits + sends.
- Section template: highlights / lowlights / asks / metrics / next month.

**H. Community / mention monitoring**
- Watch mentions on Twitter / Reddit / HN / Discord / Slack (where the product appears).
- Sentiment classification.
- Surface notable mentions (positive/negative/influencer) to founder.
- Suggest response drafts when appropriate (rare; mostly founder territory).

**I. Lightweight CRM (relationship memory)**
- Per-contact memory: who said what when, what came of it.
- Categories: customer, prospect, investor, journalist, partner, advisor.
- Reminders: "haven't followed up with X in 6 weeks."
- Agent doesn't have the relationship; agent has the memory.

**J. Founder onboarding for new features**
- When agent ships a feature: generate "what changed + how to talk about it" briefing for founder.
- Includes: copy variants, screenshot, demo gif, suggested tweet.

### Founder territory (irreducible)

- The actual relationships (calls, dinners, beta groups, advisor introductions).
- Brand voice taste arbitration.
- Story / narrative for press.
- Investor pitch — the human in the room.
- Crisis comms (security breach, public flame).
- Community moderation calls (ban / warn / engage).
- Hiring / firing / advisor agreements.
- Pricing decisions.
- Strategic pivots based on customer signal.
- Saying no to features.

## Proposed direction

### Phase A — Marketing site flow
- Recipe: marketing-site (NextJS / Astro, integrated analytics, contact form).
- Phase 13.6 marketing_site step gains drafted copy + screenshots + deployable scaffold.
- Iteration loop: founder comments in mission thread → revision tasks → redeploy.

### Phase B — Launch kit
- Phase 13 channel-template library.
- New phase 13.x `launch_post_drafting` step with channel-specific templates.
- Tracks engagement post-publish (basic metrics integration with [09-growth.md](09-growth.md)).

### Phase C — Demo + press kit
- Mr. Roboto verb: `generate_demo_video(scenario)` — playwright `--video on` per scenario.
- Press kit assembly as a recipe (output: zip with one-pager / bios / images / logo / press releases).

### Phase D — Support tiering
- Telegram-integrated tier-1 bot pulling from FAQ artifact.
- Escalation flow to founder with ticket summary.
- Tickets persisted; FAQ regenerated weekly from common queries (mission triggered cron-style).

### Phase E — Investor / mention digests
- Monthly investor update template; data pulled from [09-growth.md](09-growth.md) + recent mentions.
- Mention monitor as a long-running mission (cross-ref [08-operations.md](08-operations.md) on-call agent shape).

### Phase F — CRM
- New table `relationships`: contact_id, category, last_interaction, summary, next_follow_up.
- Agent populates from email threads (with founder consent) + manual entry.
- Reminders surface in Telegram thread.

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Marketing copy | drafts hero/features/FAQ | edits, picks tone, signs off | full |
| Brand direction | proposes 3-5 mood boards (cross-ref Z1) | picks one | full pre-publish |
| Launch channel | suggests + drafts per-channel posts | picks channel + timing + sends | full |
| Demo recording | captures via playwright | trims / re-records / approves | full |
| Press outreach | drafts pitch | sends + builds relationship | n/a |
| Cold sales email | drafts personalized | reviews + sends | full |
| Ticket tier 1 | answers from FAQ | reviews escalations | full pre-send |
| Investor update | drafts | edits + sends | full pre-send |
| Crisis comms | offers draft + checklist | drives response | n/a |
| Community moderation | flags + drafts response | makes call | n/a |

## Dependencies

- **Inbound:** [01-pre-code.md](01-pre-code.md) — brand direction + compliance affect copy. [09-growth.md](09-growth.md) — metrics drive investor updates + analytics-driven launch tuning.
- **Outbound:** [10-cross-cutting.md](10-cross-cutting.md) — Telegram thread + reversibility framing apply heavily.
- **Adjacent:** [08-operations.md](08-operations.md) — on-call agent + mention monitor share the long-running-mission shape.

## Open questions

- **Marketing-site stack.** Same as product (NextJS) or separate (Astro for static perf)? (Astro v1 — better default for marketing.)
- **CRM scope creep.** Where does this stop? Email integration? Calendar? (Email integration v1 with founder consent; calendar later.)
- **Mention monitor depth.** Twitter API (paid), Reddit API (free), HN (free), others? (Free-tier first; founder decides paid when value clear.)
- **Community moderation.** What guardrails on agent-generated responses? (Drafts-only, never auto-post; founder reviews everything community-facing.)
- **Investor data privacy.** Founder might not want certain metrics in investor updates by default. (Allowlist-driven; founder approves which metrics auto-appear.)
- **Cold outreach ethics.** GDPR + CAN-SPAM. (Honor unsubscribes mechanically; respect jurisdiction; explicit opt-in for B2C.)

## Agent task brief

When picking up this doc:
1. Read 00-README + 01-pre-code (brand direction inputs) + 09-growth (metrics) + this doc.
2. Phase A: marketing-site recipe + drafting flow + revision loop integration.
3. Phase B: launch-channel templates × 5 (HN, PH, Twitter, LinkedIn, Reddit) + tracking integration.
4. Phase C: demo video mr_roboto verb + press kit recipe.
5. Phase D: support-tiering flow design; FAQ artifact + escalation pipeline.
6. Phase E: investor update template + mention monitor as long-running mission.
7. Phase F: CRM schema + reminder flow.
8. Resolve open questions or escalate.
9. Cross-reference outbound to [10-cross-cutting.md](10-cross-cutting.md), [08-operations.md](08-operations.md).
10. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc; pairs Z4 + Z8 because both are human-irreplaceable + helper-shaped. Honest framing: lowest leverage from automation work; do after operations + growth are solid.
