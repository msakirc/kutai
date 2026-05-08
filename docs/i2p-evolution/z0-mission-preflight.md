# Z0 — Mission preflight (founder readiness + system readiness contract)

## Frame

Before phase 0 of i2p starts, a contract gets established between the
founder and the system: who is this founder, what's their ambition,
what credentials exist, what's the budget, what's success, who decides
what when. Today this is implicit (founder pastes pitch into Telegram;
agent runs). When real-world ops + cost ceilings + reversibility +
human-in-loop kick in across all later zones, the implicit contract
breaks because nothing was agreed up front.

Z0 makes the contract explicit. Outputs feed every other zone.

## Current state

- Mission start = founder pastes idea into Telegram bot; agent dispatches workflow.
- Founder identity: implicit (Telegram chat_id).
- Vault: nonexistent.
- Ambition tier: nonexistent (always full i2p).
- Cost ceiling: nonexistent.
- North-star metric: nonexistent.
- Compliance fingerprint: nonexistent (collected ad-hoc later if at all).
- Founder profile: nonexistent (cross-mission memory not implemented).
- Mission Telegram thread: not provisioned per-mission.
- Kill-switch / pause / resume agreement: implicit + ad-hoc.
- Idle/unavailable handling: agent runs blind when founder is away.

## Gaps

### Fixable by automation

**A. Founder profile (cross-mission, persistent)**
- Identity (Telegram + email + name).
- Voice / brand preferences (tone, register, examples) reused across products.
- Prior products + lessons (cross-ref [02-build-foundation.md](02-build-foundation.md) memory).
- Technical-comfort baseline (so agent calibrates jargon density).
- Time-zone + working hours (sets agent's response cadence + escalation timing).
- Notification preferences (when to page, when to digest, when to silence).

**B. Mission preflight checklist (per-mission)**
- Legal entity status (none / personal / LLC / Inc / etc.) — gates which downstream actions are possible.
- Bank account + business credit card status (gates Stripe / vendor adapters).
- Vendor accounts already provisioned vs needed (cross-ref [06-real-world-bridge.md](06-real-world-bridge.md)).
- Available time/week the founder commits to.
- Hard deadlines (launch date, demo, etc.).
- Risk appetite (prototype-only / "I'll show this to people" / "I'll charge money for this").

**C. Vault provisioning (initial setup; ongoing operation lives in [06-real-world-bridge.md](06-real-world-bridge.md))**
- Per-founder encrypted credential store, key derived from Telegram identity + passphrase.
- Recovery flow agreed up front (lost passphrase recovery via signed Telegram identity).
- Vendor scope grants happen here as accounts come online; ongoing scope evolution lives in 06.

**D. Ambition tier**
- `prototype` — running locally, demo-able to small audience, no real users
- `private_beta` — deployed, real users, no payments, hand-curated
- `public_launch` — deployed, payments, marketing, support, scale concerns
- `revenue_product` — full ops, growth, multi-jurisdiction
- Tier sets defaults for: which phases run, severity-gate strictness, cost ceiling, real-tools requirements.

**E. Cost ceiling commitment**
- Declared budget per mission (tokens / dollars / hours) with the quick-vs-thorough dial (cross-ref [10-cross-cutting.md](10-cross-cutting.md)).
- Threshold alerts set up (50% / 75% / 90%).
- Escalation contract: what happens at ceiling — pause + ask, hard stop, force-completion?

**F. North-star metric declaration**
- Founder declares one north-star at preflight (or "TBD — will refine in Z1").
- Sets the lens for [09-growth.md](09-growth.md) prioritization.
- Re-confirmed at major mission boundaries.

**G. Compliance fingerprint pre-intake**
- High-level: target jurisdictions, user types (consumer/B2B/health/children), data categories.
- Refined fully in [01-pre-code.md](01-pre-code.md); flagged here so Z1 doesn't redo the question + downstream zones (06 in particular) can plan.
- "I don't know yet" is a valid answer; agent prompts for clarification when needed.

**H. Mission Telegram thread provisioning**
- One persistent thread per mission_id (cross-ref [10-cross-cutting.md](10-cross-cutting.md)).
- Thread topic + pinned-message setup (mission summary, status, ambition tier, cost gauge).
- Founder's reactions in thread map to typed events (approve / reject / comment / pause / kill).

**I. Kill-switch / pause-resume contract**
- Founder commands: `/pause_mission <id>`, `/kill_mission <id>`, `/resume_mission <id>`.
- Pause behavior: finish in-flight tasks, hold queue, keep state, await resume.
- Kill behavior: terminate gracefully, snapshot state, mark mission killed (allows restart-from-state-snapshot — cross-ref [10-cross-cutting.md](10-cross-cutting.md) reset-to-green).
- Auto-pause triggers (configurable): cost ceiling, repeated DLQs, founder unavailable > N days.

**J. Idle / unavailable founder handling**
- Per-mission policy: if founder doesn't respond to a `[confirmation_required]` post within N hours, what does the agent do?
  - `wait` — block until response (default for irreversible actions)
  - `proceed_with_default` — continue with the agent's recommendation (default for low-stakes reversible actions)
  - `pause` — auto-pause mission
- Per-action overrides honoring reversibility tag from [10-cross-cutting.md](10-cross-cutting.md).

**K. Mission template selection**
- Pick base workflow (currently i2p_v3; future may have variants per ambition tier or domain).
- Pick target_platform (web / mobile / both — cross-ref [05-build-mobile-track.md](05-build-mobile-track.md)).
- Pick recipe seed-set (which recipes from [02-build-foundation.md](02-build-foundation.md) library are likely relevant).
- Confirms tech stack approach (greenfield / existing repo / specific stack mandate).

**L. Pre-flight readiness verdict**
- Mechanical: all required preflight outputs exist + non-empty + reasonable.
- Soft: warning if obvious risks (no legal entity but ambition `revenue_product`, no time commitment but launch in 4 weeks).
- Founder explicitly accepts warnings or addresses before mission proceeds.

### Founder territory (irreducible)

- Whether to start the mission at all (taste, gut, opportunity cost).
- Risk appetite calibration (only the founder knows what they can stomach).
- Time commitment (only the founder knows their schedule).
- Identity / legal / financial decisions (legal entity, bank, KYC).
- North-star definition (what does success mean to *them*).

## Proposed direction

### Phase A — Founder profile (DB schema + management)
- Per-founder profile row: identity, voice samples, prior missions, time-zone, notification prefs.
- Surfaces in mission preflight so founder sees + edits what's known.
- Persists across missions; new missions inherit voice/brand defaults.

### Phase B — Mission preflight wizard (Telegram conversation flow)
- Step-by-step intake: ambition tier → cost ceiling → north-star → compliance fingerprint → readiness checklist → Telegram thread provision → kill-switch contract.
- Skippable for power users (saved profile auto-fills).
- Fast-path: prototype-tier with cost ceiling $20 + 8h budget can run in 5 questions.

### Phase C — Vault initial provisioning
- Passphrase setup + recovery flow.
- Initial vendor list scaffolded based on ambition tier (prototype = none required; revenue = Stripe + Vercel + email + DNS).
- First-mission vendor onboarding handed off to [06-real-world-bridge.md](06-real-world-bridge.md) founder-action queue.

### Phase D — Per-mission Telegram thread + kill-switch wiring
- Thread provisioning at mission start.
- Pinned-message scaffolding.
- Bot commands: pause/kill/resume + handlers.
- Auto-pause triggers + idle-handling policy.

### Phase E — Readiness verdict
- Mechanical preflight check returns OK / warnings / blockers.
- Blockers stop mission start.
- Warnings surface for founder acknowledgement.
- Founder can override or fix; not auto-bypassable.

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Profile load | reads existing profile or creates | confirms / edits | full |
| Ambition tier | proposes from intake | picks | mid (changes mid-mission cost+behavior) |
| Cost ceiling | suggests by tier + recipe estimates | declares | full pre-mission |
| North-star | drafts from idea | confirms or refines | full |
| Compliance fingerprint | drafts from intake | confirms / corrects | full |
| Vault setup | walks through passphrase + recovery | provides passphrase | n/a (recovery flow exists) |
| Thread provision | creates Telegram thread + pins | n/a | full (thread can be archived) |
| Kill-switch contract | proposes auto-pause triggers | accepts / customizes | full |
| Readiness verdict | mechanical check | overrides warnings or addresses | full |

## Dependencies

- **Inbound:** none. Z0 is the entry zone (before Z1).
- **Outbound:** every other zone reads Z0 outputs.
  - [01-pre-code.md](01-pre-code.md) reads founder profile + ambition tier + compliance fingerprint
  - [02-build-foundation.md](02-build-foundation.md) reads ambition tier (severity-gate strictness defaults) + recipes seed-set
  - [05-build-mobile-track.md](05-build-mobile-track.md) reads target_platform
  - [06-real-world-bridge.md](06-real-world-bridge.md) reads vault + vendor list (initial set)
  - [09-growth.md](09-growth.md) reads north-star
  - [10-cross-cutting.md](10-cross-cutting.md) reads cost ceiling, idle policy, reversibility defaults

## Open questions

- **Profile portability.** Per-Telegram-identity OR per-account-with-multiple-Telegram-bindings? (Per-Telegram v1; multi-binding when team-mode arrives.)
- **Ambition-tier defaults.** What changes between tiers concretely — list, not vague "stricter"? (Document a per-tier matrix: phases run, severity gates, cost ceiling, real-tools requirements, idle-policy default.)
- **Cost-ceiling unit.** Hours / dollars / tokens? (All three; founder picks primary; agent shows all.)
- **North-star at preflight when idea isn't fleshed.** Allow "TBD"? (Yes; bound by phase 4 latest.)
- **Vault recovery flow.** Lost passphrase + Telegram still active = signed-message recovery? Lost both = data lost? (Yes; document the irrecoverability honestly.)
- **Mission templates.** When do we have more than i2p_v3? (Out of scope for Z0; placeholder field that defaults to i2p_v3 for now.)
- **Idle threshold.** Default N hours = ? (24h v1 for irreversible; 4h for reversible-default-proceed; configurable.)
- **Concurrent missions per founder.** Allowed? Each in own thread + cost? (Yes; bound by total cost ceiling across active missions.)
- **Preflight skip path.** Power users want fewer questions. (Profile-driven defaults + saved presets.)

## Agent task brief

When picking up this doc:
1. Read 00-README + this doc + downstream consumers (01, 02, 06, 09, 10).
2. Phase A: founder-profile DB schema + load/save flows + first-mission profile-creation flow.
3. Phase B: Telegram preflight wizard conversation flow + skippable fast-path for power users.
4. Phase C: vault initial provisioning + handoff to 06's ongoing operation.
5. Phase D: per-mission Telegram thread + kill-switch / pause / resume bot commands + auto-pause triggers + idle-handling policy.
6. Phase E: mechanical readiness verdict + warning vs blocker classification.
7. Resolve open questions or escalate; the ambition-tier matrix is especially important — every later zone defaults derive from it.
8. Cross-reference outbound: every consumer doc gets a 1-line "Z0 inputs:" addition under Dependencies.
9. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc; new zone introduced as Group 0 in dispatch, runs before all other zones.
