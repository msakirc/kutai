# Z5 — Operations

## Frame

Where systems usually die. Post-launch the product runs 24/7 against
real users, real load, real bugs, real attackers. Today's i2p has no
"watch the running app" mode — once phase 14 ends, the agent forgets the
product exists. Real ops requires monitoring, on-call, incident response,
support, backup verification, security posture, capacity planning.

The agent can become a useful **on-call assistant** with bounded autonomy
(reversible actions only) + escalation protocols. Cannot replace
on-call humans for high-stakes decisions, but can absorb 80% of the
boring lift.

## Current state

- i2p_v3 ends at phase 14 launch monitoring (24h post-launch); no ongoing operations.
- 13.3 monitoring_setup is NEEDS-REAL-TOOLS marked.
- No long-running missions / cron-style background work.
- No alert intake; no incident playbook execution.
- Support tickets: nothing; founder eats them.
- Backups: nothing.
- Dependency hygiene: nothing.
- Security posture: phase 10 audits at build-time; zero ongoing.

## Gaps

### Fixable by automation

**A. Monitoring kit at launch**
- Sentry (errors) + Better Stack (uptime) + Posthog (product) wired by adapters at deploy-time.
- Founder gets accounts (cross-ref [06-real-world-bridge.md](06-real-world-bridge.md)); agent configures rules.
- Default alerts: error spike, uptime drop, p95 regression, 5xx rate, Stripe failure rate, signup drop.
- Per-mission alert rule customization.

**B. On-call agent — long-running mission mode**
- Different mode of i2p: not building, watching.
- Reads alerts, classifies severity, runs first-response runbook, pages founder if escalation needed.
- Bounded autonomy:
  - Allowed: restart, rollback to last green, scale up, drain traffic, rotate failed key, archive flake test
  - Disallowed: migrate, delete, modify schema, change architecture, deploy unreviewed code
- All actions audit-logged + reversibility-tagged (cross-ref [10-cross-cutting.md](10-cross-cutting.md)).

**C. Incident playbook generator**
- From spec + architecture, generate likely-failure playbooks at design time:
  - "Payment provider down → fallback queue + email user."
  - "DB at 80% disk → archive old data + alert."
  - "Auth provider down → maintenance mode + status page."
  - "Cert expiring → renew via certbot, fallback to acme.sh."
- On real incident, on-call agent executes matching playbook.

**D. First-line support**
- Tier 1: agent answers from FAQ + product docs.
- Tier 2: complex / billing / angry → founder, with ticket summary + suggested response (cross-ref [07-humanish-layers.md](07-humanish-layers.md)).
- Distinguishes "can answer," "needs human," "user is angry, escalate immediately."
- Memory: ticket logged; FAQ updated weekly from common queries.

**E. Backup verify cron**
- Mechanical mission, runs weekly: spin sandbox from latest backup, run smoke tests, verify data integrity, alert on fail.
- Catches "we back up but never restore" failure mode.

**F. Dependency hygiene loop**
- Weekly: dependabot-style check, security advisories, agent proposes upgrade PRs, founder approves.
- Auto-merges low-risk patch versions after CI green; surfaces minor/major for human review.

**G. Load + headroom tracking**
- Reads p95, error rate, infra cost weekly.
- Surfaces "you're at 60% of compute budget, growing 8%/wk, project breach in 4 weeks."
- Triggers capacity-plan mission when threshold hit.

**H. Security posture monitoring (post-launch, beyond build-time audit)**
- Continuous: dependency CVE feed, exposed secrets scan, container scan.
- Periodic: pen-test scaffolding (zaproxy / nuclei templates).
- Surfaces findings with severity gating (cross-ref blockers rule).

**I. Performance regression detection**
- Synthetic checks against staging + prod after every deploy.
- Lighthouse / k6 / locust profile baselines per release.
- Surfaces regression with offending commit + suggested rollback.

**J. Cost monitor**
- Per-vendor cost API integration (AWS / Vercel / Stripe / Sentry / etc.).
- Daily cost digest; weekly trend; alert on anomalies (unexpected spike).

### Founder territory (irreducible)

- Final call on incident severity (when to wake the team / status-page / notify users).
- Customer comms during outages.
- Vendor escalations (calling Vercel / AWS / etc. support during incident).
- Post-incident retrospective participation.
- Security incident response (legal + regulatory + customer notification).
- Hiring on-call rotation (when scale demands).

## Proposed direction

### Phase A — Monitoring kit (recipes)
- Per-stack monitoring recipes (FastAPI + NextJS + ...): wires Sentry SDK, Better Stack synthetic check, Posthog client.
- Salako vendor adapters (cross-ref [06-real-world-bridge.md](06-real-world-bridge.md)) configure rules at deploy-time.
- Alert rules library (per-stack defaults).

### Phase B — On-call agent (long-running mission)
- New mission shape: `ongoing` lifecycle (currently missions are one-shot).
- Subscribes to webhooks: Sentry alerts, Better Stack status changes, Stripe failures.
- Severity classifier: critical (page founder), high (handle + log), medium (log), low (digest).
- Action whitelist + audit + escalation contracts.

### Phase C — Incident playbooks
- Phase 13 generates incident_playbooks artifact from spec + arch.
- Library of common playbooks; agent customizes per mission.
- On-call agent executes; logs decisions + outcomes.

### Phase D — Support tier 1
- Telegram bot + FAQ artifact (drawn from spec + early tickets).
- Escalation flow to founder with ticket summary.
- FAQ regenerator: weekly mission analyzes ticket patterns; proposes FAQ additions; founder approves.

### Phase E — Cron-style background missions
- Backup verify (weekly).
- Dependency hygiene (weekly).
- Cost monitor (daily digest, weekly trend).
- Security posture (daily CVE check, weekly scan).
- These are long-running missions managed by the orchestrator (which already has scheduled_jobs).

### Phase F — Performance regression detection
- Post-deploy synthetic test suite via salako (cross-ref [02-build-foundation.md](02-build-foundation.md) test_run + [04-build-visual-review.md](04-build-visual-review.md) for visual regression).
- Bisect-on-break (cross-ref [03-build-review-density.md](03-build-review-density.md)) extended to prod.

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Default alert config | wires per recipe | tunes thresholds | full |
| Tier-1 alert | restart/rollback within whitelist | reviews log | yes (rollforward) |
| Tier-2 alert | classifies + pages founder + drafts comms | makes call | n/a |
| Tier-3 (security incident) | logs, escalates immediately | drives response (legal, regulatory, comms) | n/a |
| Backup restore drill | runs weekly, alerts on fail | reviews fail digest | n/a (drill) |
| Dependency upgrade | auto-merges patch (CI green); proposes minor/major | reviews minor/major | full (revert PR) |
| Customer ticket tier 1 | answers from FAQ | reviews escalations | full pre-send |
| Cost spike | digests | investigates + decides | n/a |

## Dependencies

- **Inbound:** [06-real-world-bridge.md](06-real-world-bridge.md) — vendor adapters + accounts. [02-build-foundation.md](02-build-foundation.md) — recipes + mechanical-gate framework. [03-build-review-density.md](03-build-review-density.md) — bisect-on-break extended to prod.
- **Outbound:** [09-growth.md](09-growth.md) — product analytics (Posthog) feeds into hypothesis loop. [07-humanish-layers.md](07-humanish-layers.md) — support escalations + investor updates draw from monitoring data.
- **Cross:** [10-cross-cutting.md](10-cross-cutting.md) — reversibility tagging on every on-call action; cost transparency on monitoring infra.

## Open questions

- **Long-running mission shape.** Different from one-shot missions in lifecycle (no terminal state, just "ongoing"). DB schema change required? (Yes — add `mission_kind ∈ {oneshot, ongoing}` column; orchestrator special-cases ongoing.)
- **On-call autonomy bounds.** Default whitelist; per-product overrides? (Default conservative; per-product expansion via founder approval.)
- **Webhook ingestion.** Where do alert webhooks land? (New `webhooks` adapter; routes to on-call mission via mission_id mapping.)
- **Rate limits on on-call actions.** Prevent runaway loops (rollback/redeploy/rollback). (Per-action cooldown + max-per-hour caps.)
- **Pager integration.** PagerDuty / Opsgenie? (Telegram bot v1; PagerDuty when scale demands.)
- **Crisis playbook authorship.** LLM-generated from spec, vetted by humans? (Yes; library of vetted templates + LLM-customizes per spec.)
- **Cost monitor source of truth.** Per-vendor API or single aggregator (Vantage/CloudHealth)? (Per-vendor v1; aggregator if too noisy.)

## Agent task brief

When picking up this doc:
1. Read 00-README + 02-build-foundation + 06-real-world-bridge + this doc.
2. Phase A: monitoring kit recipes per stack + alert-rule library.
3. Phase B: design ongoing-mission lifecycle (DB + orchestrator); on-call agent profile + action whitelist + escalation protocol.
4. Phase C: incident_playbooks artifact + library + execution flow.
5. Phase D: tier-1 support flow.
6. Phase E: cron missions for backup verify + dependency hygiene + cost monitor.
7. Phase F: regression-detection wiring with [02-build-foundation.md](02-build-foundation.md) test_run.
8. Resolve open questions or escalate.
9. Cross-reference outbound to [09-growth.md](09-growth.md), [07-humanish-layers.md](07-humanish-layers.md), [10-cross-cutting.md](10-cross-cutting.md).
10. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc.
