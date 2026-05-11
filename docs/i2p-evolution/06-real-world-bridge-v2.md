# Z6 v2 — Real-world bridge (supersedes v1)

> v1 (2026-05-08) made several stale claims after Z0/Z1/Z2 shipped. This v2
> is the operative plan: corrected audit + revised tier-batched plan.

## Why v2

v1 framed Z6 as greenfield ("no vault, no adapters, no legal-doc gen").
Re-audit (2026-05-11) shows ~60% scaffold exists; Z6 is mostly
**finish + wire**, not build-from-scratch. Specific corrections:

| v1 claim | Reality (2026-05-11) |
|---|---|
| "no credential vault" | `src/security/credential_store.py` exists: Fernet+PBKDF2-480k, `/credential add/list/remove` Telegram cmds, `expires_at` in encrypted envelope. Skeletal but functional. |
| "no vendor adapters in mr_roboto" | `src/integrations/` has BaseIntegration + IntegrationRegistry + HttpIntegration (SSRF-hardened, retry/backoff/auth-injection — production-grade). 3 configs: github.json, vercel.json, railway.json. **Orphaned — zero call sites.** |
| "no founder-action queue artifact" | True. Still missing. |
| "no compliance overlay" | Z1 step 1.11a SHIPPED `compliance_overlay.json` with required_documents matrix per jurisdiction×data_category. **But Phase 12 doesn't consume it — wiring gap.** |
| "Phase 12.5 gated on equals: [pass, approved] — no legal-doc gen behind it" | Both halves wrong. Gate is `done_when: "legal_review_result exists"`. 12.1 already generates ToS+Privacy+Cookie via writer agent. Plus 12.2 cookie_consent_impl, 12.3 GDPR+CCPA checks, 12.4 license_audit, 12.5 final review. |
| "7 NEEDS-REAL-TOOLS steps (7.13, 13.1, 13.3, 13.11, feat.13)" | **4 actual**: 7.13, 13.1, 13.3, feat.13. 13.11 mislabeled. No `feat.*` mini-workflow exists. |
| (missing) | mr_roboto has `reversibility.py` (full/partial/irreversible verbs). Step-level field not yet added (Z0 plan). |
| (missing) | mr_roboto.clarify → `tasks.status='waiting_human'` → Telegram inline buttons → callback resume works **for mechanical executor only**. LLM agents can't emit `needs_clarification`; instruction text "Do NOT autonomously execute" is unenforced advisory. |
| (missing) | `compliance_templates/default/en/` has **1 of 8** declared doc types (privacy_policy.md.j2 only). 180-day staleness flag exists but doesn't block. |
| (missing) | Z0 already plans `mission.lifecycle_state` + `/pause/resume/kill_mission` + budget pause. Z6 must coordinate with Z0 to avoid duplicate state machine. |
| (missing) | `tests/test_integrations.py` (~150 lines) + `tests/test_credential_store.py` (~150 lines) already exist — Z6 has a test foundation. |

## Frame (unchanged)

Where the system meets the real world: domain registration, hosting,
email infra, payment provider, app stores, legal docs, secrets,
compliance. Most needs **legal personhood + money + identity** the agent
doesn't have. Agent prepares, instructs, operates within scoped
credentials, never pretends autonomy where human delegation is required.

Pairs **Z3** (pre-launch) + **Z7** (money).

## Current-state map (corrected)

```
EXISTS                              ORPHANED                       MISSING
------                              --------                       -------
credential_store (Fernet)           HttpIntegration                 founder_actions surface
/credential add/list/remove         IntegrationRegistry             needs_real_tools enforcement
compliance_template_render()        github.json, vercel.json,       7 of 8 compliance templates
compliance_overlay (Z1 1.11a)       railway.json configs            jurisdiction template variants
compliance_blocker_check post-hook                                  Stripe wiring (only sandbox prose)
mr_roboto/reversibility.py verbs                                    rotation reminder
mr_roboto.clarify → waiting_human                                   audit log for vault access
Telegram inline button resume                                       per-vendor cred schema validation
12.1 legal_documents (writer)                                       12.1 ← compliance_overlay link
12.2-12.5 compliance flow                                           mobile (Apple/Google) adapters
13.1-13.14 production phase                                         reversibility tags on steps
                                                                    LLM agent → needs_clarification
```

## Real Z6 gap list (10 gaps)

**G1 — `needs_real_tools` is write-only.**
Set in expander (line 534-535), buried in `task.context` JSON. No
consumer in beckman / orchestrator / coulson / mr_roboto checks it.
Agent reads instruction text "Do NOT autonomously execute — surface a
needs_clarification" and ignores it. Keystone missing piece.

**G2 — Integration layer is dead code.**
HttpIntegration fully built (SSRF-hardened, retry, auth injection).
IntegrationRegistry auto-discovers configs at boot. Zero call sites
from any executor. Tests exist but exercise the integration in
isolation, not wired to mission runtime.

**G3 — `founder_action` artifact missing.**
No machinery for "agent surfaces this real-world task to founder, mission
parks until founder marks done with optional output." Today only ad-hoc
mechanical clarify with single-question inline button works.

**G4 — Credential vault skeletal.**
Missing: per-vendor scope model (read/write/delete differentiation),
audit trail (no mission_id/task_id/agent context on access),
rotation tracking (only passive `expires_at` in envelope), recovery
(KUTAY_MASTER_KEY loss = data loss with no migration path),
per-vendor schema validation (`/credential add stripe {"foo": "bar"}`
accepted blindly), dev-fallback base64 silently insecure.

**G5 — Compliance wiring broken.**
1.11a emits jurisdiction × data_category matrix. 12.1 input_artifacts
is `[prd_final_summary, data_requirements, integration_requirements]`
— does NOT include `compliance_overlay`. 12.1 freeform-drafts legal
docs with `[LEGAL REVIEW REQUIRED]` placeholders instead of rendering
from templates the overlay declares.

**G6 — 7 of 8 compliance templates missing.**
README declares: privacy_policy, cookie_banner, dpa, tos,
retention_policy, age_gate, accessibility_statement,
data_processing_record. Only privacy_policy.md.j2 exists. No
jurisdiction variants beyond `default/en/`.

**G7 — Stripe not wired.**
13.12 payment_flow_test prose says "sandbox mode" but never touches
real Stripe test API. No build-phase product/price scaffolding. No
webhook handler. No dispute monitoring. No tax export.

**G8 — Reversibility tags absent on steps.**
mr_roboto verb-level tags exist (full/partial/irreversible). Step-level
field on i2p_v3.json not added. `needs_real_tools` steps should be
auto-tagged `irreversible` and require explicit founder ack with cost
estimate.

**G9 — LLM agents can't emit needs_clarification.**
Only mr_roboto.clarify writes `tasks.status='waiting_human'`. Coulson
dispatch path has no detect-needs_real_tools-and-short-circuit. Agent
gets the task, ignores the flag, fabricates an answer.

**G10 — Mobile vendor track (Apple/Google) greenfield.**
App Store Connect API + Google Play Console API not adapted. KYC,
$99/yr Apple, $25 Google one-time. Cross-ref Z5 mobile track.

## Locked design decisions (from 2026-05-11 brainstorm)

1. **founder_actions storage** → **new dedicated table** (not reuse `tasks`, not artifact). Cleaner queries, columnar status, independent of task lifecycle.
2. **vendor_call exposure** → **both mechanical post-hook + LLM tool** with per-agent allowlist + cost cap.
3. **T6 scope** → **full** (T6A reversibility + T6B ack + T6C mobile).

## Schemas

### `founder_actions` table

```sql
CREATE TABLE founder_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id INTEGER NOT NULL,
    blocking_task_id INTEGER,             -- NULL if mission-wide
    blocking_step_id TEXT,                -- e.g. "13.1"
    kind TEXT NOT NULL,                   -- 'credential_paste' | 'vendor_enroll'
                                          -- | 'cost_ack' | 'legal_counsel'
                                          -- | 'kyc' | 'generic'
    title TEXT NOT NULL,
    why TEXT NOT NULL,                    -- why this is needed
    instructions_json TEXT NOT NULL,      -- list of steps
    expected_output_kind TEXT,            -- 'credential' | 'url' | 'receipt'
                                          -- | 'ack_only' | 'free_text'
    expected_output_schema_json TEXT,     -- JSON schema if structured
    cost_estimate_usd REAL,               -- nullable
    reversibility TEXT,                   -- 'full' | 'partial' | 'irreversible'
    status TEXT NOT NULL DEFAULT 'pending', -- pending | in_progress
                                          -- | done | blocked | cancelled
    response_payload_json TEXT,           -- founder's reply
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (mission_id) REFERENCES missions(id),
    FOREIGN KEY (blocking_task_id) REFERENCES tasks(id)
);

CREATE INDEX idx_founder_actions_mission ON founder_actions(mission_id);
CREATE INDEX idx_founder_actions_status ON founder_actions(status);
```

### `credential_access_log` table (audit trail)

```sql
CREATE TABLE credential_access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT NOT NULL,
    mission_id INTEGER,
    task_id INTEGER,
    agent TEXT,                  -- which agent / executor pulled
    model_id TEXT,
    action TEXT NOT NULL,        -- 'read' | 'write' | 'rotate' | 'delete'
    scope TEXT,                  -- 'read_only' | 'read_write' | 'admin'
    success INTEGER NOT NULL,    -- 1/0
    error TEXT,
    accessed_at TEXT NOT NULL
);
CREATE INDEX idx_cred_log_service ON credential_access_log(service_name);
CREATE INDEX idx_cred_log_mission ON credential_access_log(mission_id);
```

### `credentials` table additions

```sql
ALTER TABLE credentials ADD COLUMN scope TEXT DEFAULT 'read_write';
ALTER TABLE credentials ADD COLUMN rotated_at TEXT;
ALTER TABLE credentials ADD COLUMN expires_at TEXT;  -- promote from envelope
ALTER TABLE credentials ADD COLUMN key_version INTEGER DEFAULT 1;
ALTER TABLE credentials ADD COLUMN schema_id TEXT;   -- ref to credential_schemas/
```

### `tasks` table additions

```sql
ALTER TABLE tasks ADD COLUMN needs_real_tools INTEGER DEFAULT 0;
ALTER TABLE tasks ADD COLUMN reversibility TEXT;   -- 'full'|'partial'|'irreversible'
```

(Hoisted from `task.context` JSON so beckman can index-query.)

### Step JSON additions

```jsonc
{
  "id": "13.1",
  "name": "production_infrastructure",
  "needs_real_tools": true,
  "reversibility": "irreversible",                  // NEW
  "real_tool_kind": "vercel|railway|supabase",      // NEW — adapter hint
  "cost_estimate_usd": 50,                          // NEW — for founder ack
  "post_hook": {                                    // NEW (existing kind)
    "kind": "vendor_call",
    "service": "vercel",
    "action": "deploy",
    "params_from_artifact": "production_compute_plan"
  }
}
```

### `credential_schemas/<service>.json` (per-vendor validation)

```json
{
  "service_name": "stripe",
  "required_fields": ["secret_key", "publishable_key", "webhook_secret"],
  "optional_fields": ["tax_origin_country"],
  "scopes": ["read_only", "read_write", "admin"],
  "default_scope": "read_write",
  "rotation_recommended_days": 90,
  "test_endpoint": { "action": "ping", "expect_status": 200 },
  "docs_url": "https://stripe.com/docs/keys"
}
```

## Tier plan

7 tiers. T1 sequential (anchors everything). T2-T4 parallel. T5
depends on T1+T3. T6-T7 parallel after T5.

### T1 — Keystone: hard gate + founder_actions (sequential)

**T1A** — `tasks.needs_real_tools` + `tasks.reversibility` columns; hoist
from `task.context`. Migrate existing tasks. Update expander to write
both columns. **Files:** `src/infra/db.py`, `src/workflows/engine/expander.py`,
migration in `src/infra/migrations/`.

**T1B** — `founder_actions` table + repository module
`src/founder_actions/repo.py` with `create()`, `get()`, `list_by_mission()`,
`update_status()`, `resolve()`.

**T1C** — Beckman admission gate:
- Before scheduling a `needs_real_tools=true` task, resolve `real_tool_kind`
- Check (a) adapter registered in IntegrationRegistry, (b) credentials exist for service, (c) reversibility=irreversible has cost_ack founder_action resolved.
- If any missing: create founder_action(s), mark task `status='blocked_on_founder_action'`, return without dispatch.
**Files:** `packages/general_beckman/src/general_beckman/admission.py` (new), wire into `next_task()`.

**T1D** — Telegram surfacing:
- `/actions [mission_id]` lists pending founder_actions for mission (or all active missions).
- Card layout in mission thread: title, why, instructions, expected_output, inline buttons (`Mark in-progress`, `Mark done`, `Block`).
- For `kind='credential_paste'`: button starts a `/credential add <service>` mini-flow.
- For `kind='cost_ack'`: button confirms.
- For `kind='vendor_enroll'`: button + free-text paste (URL/order ID).
- `/action_done <id> [json_payload]` text command for backup.
**Files:** `src/app/telegram_bot.py`, `src/app/telegram_topics.py`, new `src/app/founder_action_render.py`.

**T1E** — Mission lifecycle coordination with Z0:
- Reuse Z0 planned `missions.lifecycle_state` if shipped, else add `blocked_on_founder_action` state.
- Wake mission when all blocking founder_actions resolved (poll loop in orchestrator).
**Files:** `src/core/orchestrator.py`, coordinate with Z0 module location.

### T2 — Credential vault hardening (parallel agent)

**T2A** — Schema migration: add `scope`, `rotated_at`, `expires_at`, `key_version`, `schema_id` columns. Promote `expires_at` from inside encrypted envelope to indexable column (still also in envelope for tamper-proof).

**T2B** — Per-vendor schema validation:
- New `credential_schemas/` directory with stripe.json, sendgrid.json, vercel.json, cloudflare.json, sentry.json, supabase.json, github.json (already used), railway.json (already used).
- `store_credential()` validates payload against schema; rejects unknown services unless `--unsafe` flag.
- `/credential schema <service>` shows required fields.

**T2C** — Credential access audit log:
- Wire every `get_credential()` call through audit logger with (mission_id, task_id, agent, model_id, scope) from context.
- `/credential log <service>` shows last 50 access events.

**T2D** — Recovery / re-encrypt:
- Add `key_version` column. `KUTAY_MASTER_KEY_v2` env supported alongside v1.
- New CLI: `python -m src.security.rekey --from-version 1 --to-version 2` migrates all rows.
- Refuse new writes if `KUTAY_MASTER_KEY` rotated without explicit migration.

**T2E** — Drop dev fallback:
- Base64 fallback emits one warning per process AND requires `KUTAY_DEV_ALLOW_INSECURE_VAULT=1` env explicitly set.
- Without that env, fallback raises instead of warning.

**Files:** `src/security/credential_store.py`, `src/security/rekey.py` (new), `credential_schemas/*.json`, `src/app/telegram_bot.py` (new commands).

### T3 — Integration wiring + Wave 1 vendors (parallel agent batch — 3 sub-agents)

**T3A — Mechanical post-hook `vendor_call`** (sub-agent A):
- New `packages/mr_roboto/src/mr_roboto/executors/vendor_call.py`
- Signature: `async def run(task) -> dict`. Reads `task.context.post_hook.{service, action, params, params_from_artifact}`. Loads adapter from registry. Executes. Returns `{ok, result, cost_estimate_usd}`.
- Cost cap from `mission.cost_budget_remaining` (cross-ref Z0).
- Error → emit `founder_action(kind='vendor_failure')` with error detail.

**T3B — LLM tool `vendor_call`** (sub-agent B):
- New tool in `src/tools/vendor_call.py`. Tool signature: `vendor_call(service: str, action: str, params: dict) -> dict`.
- Per-agent allowlist in `tool_registry` — executor agent gets vercel/railway/supabase; implementer gets stripe/sendgrid; researcher gets none.
- Per-call cost cap; rejected calls return tool error without making HTTP request.
- Surface call in `mission_events` table for cost tracing.

**T3C — Wave 1 configs** (sub-agent C):
- New `src/integrations/configs/stripe.json` — actions: list_products, create_product, create_price, create_checkout_session, list_subscriptions, retrieve_balance.
- `sendgrid.json` — actions: send_mail, list_templates, verify_domain, list_suppressions.
- `cloudflare.json` — actions: list_zones, list_dns_records, create_dns_record, delete_dns_record.
- `sentry.json` — actions: list_projects, list_issues, get_issue, list_releases.
- `supabase.json` — actions: list_projects, run_migration (via SQL endpoint), list_buckets, create_signed_url.
- Each config in test/sandbox mode by default (e.g. `base_url: "https://api.stripe.com"` but actions specify test-mode params).

**T3D — Adapter resolver for `real_tool_kind`** (built by sub-agent A, consumed by T1C):
- `src/integrations/resolver.py`: maps step `real_tool_kind` (`"vercel"`, `"vercel|railway|supabase"`) to a concrete adapter present in registry. If none match, return `None` so beckman emits founder_action to configure one.

**Files:** `packages/mr_roboto/src/mr_roboto/executors/vendor_call.py`, `src/tools/vendor_call.py`, `src/integrations/configs/*.json`, `src/integrations/resolver.py`.

### T4 — Compliance wiring fix + missing templates (parallel agent batch)

**T4A — Fix 12.1 wiring**:
- Add `compliance_overlay` to 12.1 input_artifacts.
- Split 12.1 into 12.1 (mechanical: template render pass) + new 12.1b (writer agent: placeholder fill + jurisdiction-specific clauses).
- 12.1 mechanical reads overlay.required_documents[], calls `compliance_template_render()` per entry, writes Markdown to `mission/.compliance/legal/<doc_type>.md`.
- 12.1b agent reads rendered drafts + overlay + PRD; fills `[LEGAL REVIEW REQUIRED]` blocks with project specifics; flags items that genuinely need counsel review (don't auto-fill).

**T4B — Missing template stubs** (7 templates × default/en):
- `cookie_banner.md.j2`, `dpa.md.j2`, `tos.md.j2`, `retention_policy.md.j2`, `age_gate.md.j2`, `accessibility_statement.md.j2`, `data_processing_record.md.j2`.
- Each with sibling `.meta.json` (version 1, last_reviewed 2026-05-11).
- Templates are starter drafts with Jinja variables for: project_name, jurisdiction, controller_contact, data_categories, retention_period, third_parties, lawful_basis.

**T4C — Jurisdiction variants**:
- `compliance_templates/gdpr/en/` — overrides for privacy_policy, dpa, retention_policy (EU-specific: lawful basis, DPO, ICO complaint path).
- `compliance_templates/ccpa/en/` — overrides for privacy_policy, retention_policy ("Do Not Sell" link, verification, opt-out methods).
- Resolver order in `compliance_template_render()` already supports this; templates just need to exist.

**T4D — Staleness → founder_action**:
- New cron in beckman scheduled_jobs: `compliance_template_staleness_check`, runs weekly.
- For each template with `last_reviewed` > 180 days: emit `founder_action(kind='legal_counsel', title='Review compliance template <X>')`.
- Cross-ref Z0 cron infrastructure.

**Files:** `src/workflows/i2p/i2p_v3.json` (12.1 + new 12.1b), `compliance_templates/{default,gdpr,ccpa}/en/*.md.j2`, `packages/mr_roboto/src/mr_roboto/executors/compliance_template_staleness.py`.

### T5 — Stripe recipe (sequential, depends on T1+T3)

**T5A — Build-phase scaffolding**:
- When `monetization_strategy.billing.provider == 'stripe'`, Phase 7 build steps add:
  - `/api/checkout/create_session` endpoint scaffold (lang depends on stack)
  - `/api/webhook/stripe` endpoint scaffold with signature verification stub
  - `.env.example` keys: `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PUBLISHABLE_KEY`
- New step `7.x.stripe_scaffold` (mechanical) conditional on `skip_when` ↑.

**T5B — Mechanical `stripe_provision_products`**:
- New executor reads `monetization_strategy.products[]` (name, price_cents, interval, currency).
- Calls `vendor_call(service='stripe', action='create_product', params=...)` per product.
- Then `create_price` per (product × tier).
- Idempotent: looks up existing by `metadata.kutay_id` before creating.
- Output artifact: `stripe_provisioned.json` with `product_id`, `price_id` per item.

**T5C — Deepen 13.12 payment_flow_test**:
- Currently prose-only sandbox test. Convert to real Stripe test API run via vendor_call.
- Test list: create test customer → create checkout session → confirm with test card → verify webhook fires → query subscription status → cancel → verify refund.
- Output: `payment_flow_results.json` with each test pass/fail + Stripe IDs.

**T5D — Dispute monitor + revenue digest**:
- Read-only `stripe_dispute_check` mechanical executor (scheduled, weekly).
- Emits `founder_action(kind='legal_counsel', title='Dispute filed', urgent=true)` on new disputes.
- Revenue digest cron (cross-ref Z8 ops) — weekly Telegram message with ARR/MRR/new/churn.

**T5E — Stripe Tax + tax-export ledger**:
- Enable Stripe Tax flag in product config when monetization_strategy.tax.collect == true.
- Mechanical `tax_export_ledger` (monthly cron): fetches Stripe Tax CSV → writes to `mission/.tax/<YYYY-MM>.csv` → emits `founder_action(kind='generic', title='Send tax CSV to accountant')`.

**Files:** `src/workflows/i2p/i2p_v3.json` (new steps), `packages/mr_roboto/src/mr_roboto/executors/{stripe_provision_products,stripe_dispute_check,tax_export_ledger}.py`, `src/integrations/configs/stripe.json` (extended).

### T6 — Reversibility + mobile track (parallel agent batch)

**T6A — Step reversibility tags**:
- One-time sweep of `src/workflows/i2p/i2p_v3.json` adding `reversibility` field.
- All `needs_real_tools=true` steps → `irreversible`. Production writes (13.1, 13.2, 13.6, etc.) → `irreversible`. Migration apply, git push, vendor write → `partial`. Local writes, snapshot reads, lint → `full`.
- Heuristic walker script: outputs proposals; human reviews before merge.

**T6B — Founder ack required for irreversible + cost-estimated**:
- For any step with `reversibility='irreversible'` AND `cost_estimate_usd > 0` AND first run for this mission:
- Beckman admission emits `founder_action(kind='cost_ack', title='Confirm spend $X for step Y')`.
- Subsequent runs (within mission) reuse ack.

**T6C — Apple App Store Connect + Google Play Console adapters**:
- New `src/integrations/configs/apple_appstore.json` — actions: list_apps, list_builds, list_screenshots, submit_for_review, list_review_status.
- `google_play.json` — actions: list_apps, list_release_tracks, upload_apk_metadata, list_review_status.
- Apple auth: JWT signed with private key (.p8) — adapter handles JWT mint internally. Add `credential_schemas/apple_appstore.json` with `team_id`, `key_id`, `private_key_pem` fields.
- Google Play: OAuth service account JSON.
- Founder enrollment is one-way (KYC + $99/yr Apple, $25 Google) → `founder_action(kind='vendor_enroll')` with checklist + expected paste-back of credential.
- Cross-ref Z5 mobile track for build/release flow.

**Files:** `src/workflows/i2p/i2p_v3.json` (reversibility sweep), `src/integrations/configs/{apple_appstore,google_play}.json`, `credential_schemas/{apple_appstore,google_play}.json`, new helper `src/integrations/adapters/apple_jwt.py` for JWT mint.

### T7 — Polish + cross-cutting (parallel agent batch)

**T7A — Rotation reminder cron**:
- Weekly cron `credential_rotation_reminder`: queries `credentials WHERE expires_at < now+14d OR (rotated_at IS NULL AND created_at < now-90d)`. Emits founder_action per credential.

**T7B — Founder-action UX polish**:
- `/missions` shows blocked count: `[Mission 42] active — 2 founder_actions pending`.
- Mission detail card has "Pending Actions" inline expander.
- `/actions` empty-state shows "All clear" with last resolved card.

**T7C — LLM detect-and-bail (closes G9)**:
- Coulson dispatch checks `task.needs_real_tools` BEFORE LLM call.
- If true AND no satisfied prerequisite (adapter+creds via beckman admission): coulson short-circuits, emits mechanical `clarify` action that creates founder_action.
- If true AND prereqs satisfied: coulson proceeds, BUT adds a system prompt block saying "This task has real-world side effects. Use vendor_call tool; do not fabricate results."

**T7D — Docs + cross-refs**:
- Update `00-README.md` with v2 pointer.
- Cross-refs: Z0 (lifecycle, budget, forum topics), Z1 (compliance_overlay producer), Z5 (mobile), Z8 (operations cron + dispute monitor), Z9 (Stripe + pricing tests), Z10 (reversibility framing).
- Add `docs/architecture/founder_actions.md` and `docs/architecture/vendor_call.md` short refs.

## Dependencies between tiers

```
T1 (sequential foundation)
  ├─→ T2 (vault hardening)        [parallel]
  ├─→ T3 (integration + wave 1)   [parallel]
  └─→ T4 (compliance wiring)      [parallel]
T3 + T1 → T5 (Stripe recipe)
T1 → T6A reversibility, T6B ack [parallel after T1]
T3 → T6C mobile adapters [parallel after T3]
T1 + T3 + T4 → T7 polish [parallel after all]
```

## Human-in-loop pattern (refined from v1)

| Step | Agent does | Founder does | Reversibility | Founder action kind |
|---|---|---|---|---|
| Vendor account enroll | emits founder_action with checklist + expected output schema | enrolls (KYC, $), pastes resulting key/order ID | irreversible (KYC one-way) | `vendor_enroll` |
| Token paste | parses, validates per `credential_schemas/<svc>.json`, stores in vault, audit-logs | pastes from vendor's UI in response to /actions card | full (rotate) | `credential_paste` |
| Vendor adapter operation (in scope) | vendor_call mechanical or LLM tool, audit-logs, cost-caps | reviews call in mission_events on demand | depends on adapter | (none — runs autonomously) |
| Cost-incurring irreversible step | emits cost_ack founder_action with estimate | confirms with button | irreversible | `cost_ack` |
| Privacy policy generation | renders from compliance_overlay × templates, fills placeholders, flags for review | reviews drafts, sends to counsel out-of-band | full (pre-publish) | (none — handled by 12.5 mechanism) |
| Legal counsel review | emits founder_action with bundle (drafts + intake summary) | sends to counsel, marks done when reviewed | full pre-go-live | `legal_counsel` |
| Stripe config | runs stripe_provision_products via vendor_call, scaffolds checkout endpoint | reviews tax setup + payouts in Stripe dashboard out-of-band | full pre-go-live | `vendor_enroll` (initial) |
| App store enrollment | emits checklist founder_action | enrolls (legal entity) | irreversible | `vendor_enroll` + `kyc` |
| Crisis (incident) | drafts user notification + regulatory form to artifact, emits founder_action urgent | sends + signs | n/a | `legal_counsel` (urgent=true) |

## Open questions (revised)

- **Mobile app build artifacts.** Where does the .ipa / .aab get built so the App Store adapter can upload? Likely Z5's mobile-track CI — Z6 only handles metadata/screenshots/submit, not build. *(Tracking: cross-ref Z5; not blocking Z6 T6C scaffolding.)*
- **Stripe Connect.** Marketplace recipes (platform takes %, payouts to sellers) needs Stripe Connect. *(Defer past Z6 v2; flag in T5 doc.)*
- **Tax filing depth.** Stripe Tax collects but filing is human + accountant. T5E generates CSV ledger; filing remains founder_action territory.
- **Founder_action notification urgency.** Disputes, payment failures, security incidents need immediate ping. Add `urgent=true` flag → bypass mission thread, DM founder directly. *(Implement in T1D with simple urgent path.)*
- **Cost estimate accuracy.** `cost_estimate_usd` on steps is initially manual. Build a vendor pricing table per adapter (vercel free tier, then $20/mo etc.) over time. *(Defer past T6B; manual entry in v2.)*
- **Concurrent founder_actions of same kind.** If two missions need Stripe enrollment, do we collapse? *(No — separate missions, separate cards. Founder may bundle action at their discretion.)*
- **Audit log retention.** `credential_access_log` grows unbounded. Add 180-day retention with archival? *(Defer; revisit when 100MB.)*

## Agent task brief (revised — for execution session)

1. Re-read this v2 doc and the audit findings inline above. Treat v1 as superseded.
2. **T1 first, sequential** (T1A→B→C→D→E). T1 is the keystone — every other tier depends on `tasks.needs_real_tools` column, `founder_actions` table, and beckman admission gate existing.
3. **T2, T3, T4 in parallel** after T1 merges (each can be a parallel agent — Z1/Z2 pattern).
4. **T5 sequentially** after T1+T3 (Stripe recipe needs vendor_call + founder_actions surfaces).
5. **T6 (3 sub-tasks parallel)** after T1+T3.
6. **T7 polish** in parallel, last.
7. Each tier merge: run `pytest tests/integrations`, `pytest tests/founder_actions` (new), `pytest tests/test_credential_store`. Add tier-specific tests as part of the tier.
8. Add `## Updates` entry per tier merge with date + commit hash.

## Cross-references

* **Z0 — lifecycle coordination.** `missions.lifecycle_state` reuse for
  `blocked_on_founder_action`; budget pause; per-mission Telegram
  thread topic that hosts inline action cards.
* **Z1 — compliance_overlay producer.** Step 1.11a is the canonical
  upstream artifact consumed by 12.1 (T4A) and the staleness scanner
  (T4D).
* **Z5 — mobile build pipeline.** Z5 owns .ipa / .aab build; Z6 T6C
  owns App Store Connect + Google Play submit/list adapters.
* **Z8 — operations cron.** T4D (template staleness), T7A (credential
  rotation), T5D (Stripe dispute + revenue digest) all run on the
  Beckman cron registered via ``general_beckman.cron_seed``.
* **Z9 — Stripe pricing experiments.** Stripe Connect, A/B price
  experiments and dispute analysis depend on the T5 recipe.
* **Z10 — reversibility framing.** Step-level reversibility tags
  (T6A) feed the cost-ack gate (T6B) and into Z10's mission
  irreversibility scorecard.

## Updates

- **2026-05-08** — v1 initial doc; pairs Z3+Z7; high-level "no vault/no adapters" framing.
- **2026-05-11** — v2 supersedes v1. Re-audit revealed credential_store, IntegrationRegistry, compliance_overlay (Z1) already exist. 10-gap list reframed Z6 as wiring + finish. Locked: founder_actions = dedicated table; vendor_call = both mechanical + LLM tool; T6 full scope. Tier plan revised to 7 tiers with T1 keystone sequential, T2-T4 + T6 parallel batches, T5 sequential, T7 polish.
- **2026-05-11 — T1 shipped** (commits ``74a2175`` → ``3b40d2b``): needs_real_tools+reversibility columns, founder_actions table+repo, Beckman z6_admission gate, /actions+/action_done Telegram surfaces, mission lifecycle coord (block/unblock).
- **2026-05-11 — T2 shipped** (``b6ee8c3`` → ``c094cea``): credentials.scope/rotated_at/expires_at/key_version/schema_id columns; per-vendor schema validation; credential_access_log audit trail + /credential log; versioned master-key rekey; insecure dev-vault behind explicit env gate.
- **2026-05-11 — T3 shipped** (``c478da1`` → ``153209f``): mechanical vendor_call post-hook; LLM vendor_call tool with allowlist + cost cap; wave 1 vendor configs (stripe/sendgrid/cloudflare/sentry/supabase); real_tool_kind resolver for pipe-separated adapter choice.
- **2026-05-11 — T4 shipped** (``0ae71ce`` → ``2715a24``): 12.1 split into mechanical render + 12.1b LLM fill (consumes compliance_overlay); 7 missing template stubs; gdpr+ccpa jurisdiction overrides; weekly compliance_template_staleness cron emits legal_counsel founder_actions.
- **2026-05-11 — T5 shipped** (``9a92cf6`` → ``ca7ec93``): build-phase stripe_scaffold; stripe_provision_products executor + 13.11b workflow step; 13.12 payment_flow_test deepened via vendor_call; stripe_dispute_check + revenue_digest weekly crons; stripe_tax_export monthly cron.
- **2026-05-11 — T6 shipped** (``6ee111b`` → ``837fb64``): step-level reversibility tags swept across i2p_v3 with audit script; cost_ack founder_action for irreversible+cost-estimated steps; Apple App Store Connect + Google Play Console adapters with JWT mint + service-account auth.
- **2026-05-11 — T7 shipped** (``c4a86ad`` → THIS): credential_rotation_reminder weekly cron (T7A); /missions+/mission UX polish with action count badges (T7B); coulson detect-and-bail closes G9 — short-circuits or injects warning prompt when needs_real_tools=true (T7C); docs + cross-refs + register_artifact helper + vendor_call audit context wired (T7D).
