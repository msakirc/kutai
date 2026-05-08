# Z3 + Z7 — Real-world bridge (pre-launch + money)

## Frame

Where the system meets the real world: domain registration, hosting,
email infra, payment provider, app stores, legal docs, secrets,
compliance. Most of this needs **legal personhood + money + identity**
that the agent does not have. Agent's job here is to prepare, instruct,
operate within scoped credentials, and never pretend autonomy where
human delegation is required.

Pairs Z3 (pre-launch) + Z7 (money) because both share the
real-world-identity problem space — vendor accounts + KYC + tax forms +
ongoing financial ops.

## Current state

- 7 NEEDS-REAL-TOOLS steps marked in i2p_v3 (2026-05-06): 7.13 staging_environment, 13.1 production_infrastructure, 13.3 monitoring_setup, 13.11 social_preview_test, feat.13 staging_deploy. Marker is informational; no flow-control hookup.
- Phase 12 has `legal_review` (12.5) gated on `equals: [pass, approved]` — but no legal-doc generation behind it.
- No vendor adapters in mr_roboto (Vercel / Railway / Supabase / Stripe / SES / Sentry / Apple / Google).
- No credential vault; no scoped-token model.
- Phase 13 has production_infrastructure / monitoring_setup / etc. — all NEEDS-REAL-TOOLS.
- No founder-action queue artifact.
- No compliance overlay (privacy policy / cookie banner / DPA per jurisdiction).
- No payment provider integration; no billing schema in any recipe.
- No tax-handling.

## Gaps

### Fixable by automation (with scoped credentials)

**A. Vendor adapter library**
- Per-vendor adapters; structured operations within scope.
- Hosting: Vercel / Railway / Fly / Render — deploy, redeploy, env-var management, log fetch, rollback.
- Storage: Supabase / Firebase / S3-compat — bucket create, signed URL, CORS config.
- Auth providers: Google / GitHub / Apple OAuth client setup; redirect URIs.
- Email: SES / Postmark / SendGrid — domain verify (DKIM/SPF/DMARC records), template upload, send rate, suppression list.
- Sentry / observability adapters (cross-ref [08-operations.md](08-operations.md)).
- Stripe — products, prices, checkout sessions, customer portal, dispute view (read-only by default), webhook subscription.
- DNS — Cloudflare / Route53 record CRUD.
- Domain registrars — purchase by founder; transfer / DNS by adapter.

**B. Founder-action queue**
- New artifact type: `founder_actions` — list of out-of-band human tasks with explicit instructions + expected output to paste back.
- Examples: "Buy example.com from Cloudflare/Namecheap; paste registrar order ID." "Sign up Stripe; complete KYC; paste API key (test mode first)." "Configure SES production access request; await approval; paste sandbox-out date."
- Mission can't proceed past phase 13 without founder_actions complete.
- Status surfaced in mission Telegram thread: pending / in-progress / done / blocked.

**C. Credential vault**
- Encrypted at rest; key derived from founder's Telegram identity + a passphrase.
- Per-vendor scoped tokens: read-only first, escalate to write/delete on explicit approval.
- All access audit-logged (who/what/when via fatih_hoca model + agent_type + task_id).
- Rotation reminders (90-day default).
- Recovery flow for lost passphrase (signed by Telegram identity).

**D. Compliance fingerprint**
- Phase 0 collects: target jurisdictions, user types (consumer / B2B / health / children), data categories.
- Surfaces compliance overlay:
  - Privacy policy template (per jurisdiction × user-type matrix).
  - Cookie banner config (essential / analytics / marketing tiers).
  - DPA (data processing agreement) for B2B.
  - Retention policy (per data category).
  - Consent flows (granular, withdrawable).
  - Data deletion workflow (GDPR Article 17 / CCPA "do not sell").
  - Tax registration map (where to register for VAT/GST/sales tax based on customer locations).
- Each item has explicit "fix me" markers for human review (counsel + accountant).

**E. Legal templates + jurisdiction matrix**
- Library of starter templates (privacy policy, ToS, cookie banner, DPA, refund policy) per jurisdiction × feature class.
- Agent populates from spec; founder reviews; counsel reviews if budget allows.
- Honest framing: "this is a draft; run by counsel before launch."

**F. Stripe-shaped payments integration**
- Adapter wires products + subscriptions + invoicing + dispute handling + Stripe Tax.
- Pricing experiment kit: A/B harness + Stripe coupon/price-id management; mission can test "$29 vs $39" against signups (cross-ref [09-growth.md](09-growth.md)).
- Revenue dashboard: ARR / MRR / churn / LTV calculations; weekly digest.
- Hand-off threshold: anything tax/legal/audit-shaped → human + professional.

**G. App store accounts (mobile)**
- Apple Developer Program enrollment (founder; KYC, $99/yr).
- Google Play Console enrollment (founder; KYC, $25 one-time).
- App Store Connect / Play Console adapter for metadata, screenshots, builds, review status.
- (Cross-ref [05-build-mobile-track.md](05-build-mobile-track.md) — they share the credential vault + founder-action queue.)

### Founder territory (irreducible)

- Legal entity (LLC / Inc / Ltd / Anstalt / etc.) — incorporation
- Bank account
- Business address (or registered agent)
- Personal/business credit card for vendor billing
- KYC for Stripe + app stores + cloud providers
- Tax registration in operating jurisdictions
- Counsel for ToS / privacy review (high-stakes deployments)
- Accountant for tax filings
- Data-subject request handling (GDPR DSARs are humans-only)
- Crisis decisions (security incident → notify users + regulators)

## Proposed direction

### Phase A — Vault + scoping
- Encrypted credential vault (libsodium-based; key from founder identity + passphrase).
- Per-vendor scope model: define scopes; agent requests scope; founder grants once or per-session.
- Audit log integration with existing infra/audit.

### Phase B — Founder-action queue
- New artifact type + Telegram thread integration (cross-ref [10-cross-cutting.md](10-cross-cutting.md)).
- Status board view in mission dashboard.
- Agent generates + curates queue from mission progress.
- Founder marks items done with optional output (token / URL / receipt).

### Phase C — Vendor adapters (library, ship in batches)
- Wave 1: Vercel + Cloudflare DNS + Stripe (test mode) + SendGrid (sandbox) — most projects need these.
- Wave 2: Supabase / Firebase + Postmark + Sentry — frequent picks.
- Wave 3: Apple App Store + Google Play (mobile track).
- Per adapter: scoped-operation list, error-handling patterns, idempotency, cost estimate (where applicable).

### Phase D — Compliance overlay
- Phase 0 intake form (jurisdictions + user-types + data-categories).
- Compliance overlay artifact generated from intake.
- Per-jurisdiction template library + populator.
- Counsel-handoff flow: package draft + intake summary, invite counsel review (out of band).

### Phase E — Stripe integration
- Wire products + prices + subscriptions + checkout from spec's pricing model.
- Webhook handler scaffold for invoice events.
- Dispute monitoring (read-only); escalation to founder.
- Stripe Tax wired by default.

### Phase F — Real-tools-required as hard gate
- `needs_real_tools: true` flag (informational today) becomes a hard gate that requires the matching adapter.
- If adapter unavailable: mission halts at the step, surfaces "this step requires the X adapter; configure account credentials for Y."

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| Vendor account create | provides instructions + URL | enrolls (KYC, payment) | one-way mostly |
| Token paste | parses + scopes + stores in vault | pastes from vendor's UI | full (rotate) |
| Vendor adapter operation | within scope only | escalates "I need write access for X" | scope-limited |
| Privacy policy gen | drafts from compliance overlay | reviews + sends to counsel | full pre-publish |
| Stripe config | wires products + prices | reviews tax setup + payouts | full pre-go-live |
| App store enrollment | reminds + checklists | enrolls (legal entity) | one-way |
| Crisis (incident) | drafts user notification, fills regulatory form | sends + signs | n/a |

## Dependencies

- **Inbound:** [01-pre-code.md](01-pre-code.md) — compliance fingerprint feeds in here.
- **Outbound:** [08-operations.md](08-operations.md) — monitoring adapters share the vendor library. [09-growth.md](09-growth.md) — Stripe + analytics tie into hypothesis tests. [05-build-mobile-track.md](05-build-mobile-track.md) — Apple / Google accounts share the founder-action + vault flow.
- **Cross:** [10-cross-cutting.md](10-cross-cutting.md) — credential vault is cross-cutting; reversibility framing is essential here (every real-world action needs tagging).

## Open questions

- **Vault implementation.** libsodium + local file vs OS keystore (macOS Keychain / Windows Credential Manager) vs cloud KMS? (Local libsodium v1; OS keystore v2.)
- **Token rotation cadence.** 90 days default; per-vendor override? (Default 90; override via vendor adapter.)
- **Compliance template currency.** Templates go stale (jurisdictional changes). Quarterly cron + founder approval? (Yes; out of scope for autonomous flow.)
- **Stripe Connect or just Stripe?** (Just Stripe v1; Connect when marketplace recipes appear.)
- **Tax automation depth.** Stripe Tax for collection; what about filing? (Filing is human + professional; agent generates ledger export.)
- **Adapter failure modes.** When Vercel API is down, agent retries with backoff or escalates? (Escalate after 3 retries with surfaced error.)
- **Compliance overlay extensibility.** New jurisdictions added how? (Per-jurisdiction YAML template + a contribution flow; not LLM-generated for legal accuracy.)

## Agent task brief

When picking up this doc:
1. Read 00-README + 01-pre-code + this doc.
2. Phase A: design + scaffold credential vault; tests for encrypt/decrypt + scope-grant flow.
3. Phase B: founder_actions artifact schema + Telegram surfacing + status update flow.
4. Phase C: pick vendor wave 1 set (Vercel + Cloudflare + Stripe-test + SendGrid-sandbox); scaffold adapters with structured ops; integration tests against sandboxes.
5. Phase D: design compliance overlay artifact + intake form; build template library skeleton.
6. Phase E: Stripe products + prices + checkout flow per recipe.
7. Phase F: convert `needs_real_tools` flag from informational to hard gate.
8. Cross-reference outbound to [08-operations.md](08-operations.md), [09-growth.md](09-growth.md), [05-build-mobile-track.md](05-build-mobile-track.md), [10-cross-cutting.md](10-cross-cutting.md).
9. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc; pairs Z3 + Z7. Absorbs the NEEDS-REAL-TOOLS workstream + relevant theme + cross-cutting overlap.
