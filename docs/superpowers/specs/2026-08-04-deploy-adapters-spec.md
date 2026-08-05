# Spec 1 — Free-tier deploy adapters + git-host prerequisite chain

**Date:** 2026-08-04
**Status:** approved design → ready for implementation plan
**Depends on:** research report `docs/research/2026-08-04-free-deploy-stack-research.md`
**Blocks:** Spec 2 (`2026-08-04-deploy-orchestrator-spec.md`)
**Adversarial review:** GO-WITH-CHANGES — this spec incorporates review findings H2, H3, H6, H8, H9.

## Purpose
Give the KutAI deploy path the **atomic building blocks** to stand up a $0/no-card staging
environment for the apps it generates (starting with HabitHub: Next.js + NestJS/Prisma +
Postgres + Redis). This spec is **additive and mock-testable** — new declarative
`HttpIntegration` configs, credential schemas, and the git-host prerequisite chain. It does
**not** contain the deploy orchestration loop (that is Spec 2).

## Scope (right-sized per review H6)
**In:** the minimum to complete m90 task 7.13 for HabitHub, once, for real.
- New adapter configs: `render.json`, `neon.json`, `upstash.json`.
- Enrich `vercel.json`: add `get_deployment` (poll) + `mocked`-tagged `mock_responses`.
- New credential schemas: `render.json`, `neon.json`, `upstash.json`.
- Git-host prerequisite chain (review H2): PAT credential + create-repo + push-scaffold +
  repo-link.
- `mocked:true` tagging on every deploy/provision action's mock payload (feeds Spec 2's
  anti-fake guard, review H3).
- Housekeeping: 7.13 `real_tool_kind` `vercel|railway|fly` → `vercel|render` (H9).
  `AGENT_ALLOWLIST["executor"]` — **RESOLVED per review → SKIP.** Spec 2 makes 7.13 pure-mechanical
  (routes through `mr_roboto`, not the `vendor_call_tool` that consumes the allowlist), so no
  inline agent path survives for the deploy DAG. Add render/neon/upstash to the allowlist ONLY if
  a separate inline researcher/implementer probe is later wanted.

**Out (deferred to a later production-deploy phase):** Cloudflare Pages/Workers/R2, Resend,
Supabase-auth, GitHub-Actions cron/keep-warm, DNS/CDN/WAF, monitoring adapters, and the
**serverless (CF Workers) backend path** (review H5 — separate codegen track, founder
confirmed Render-only now).

## Non-goals
- No poll loop, ordering, secret plumbing, or health check here — all Spec 2.
- No app-code changes (no serverless refactor).
- No engine changes to `HttpIntegration` beyond, if needed, HTTP Basic auth support for Upstash
  (see Adapter nuances).

## Architecture fit
Configs are auto-discovered by `IntegrationRegistry._auto_discover()` from
`src/integrations/configs/*.json` and wrapped as `HttpIntegration`. Auth resolves via
`src/security/credential_store.get_credential(service_name)`. Mock mode is ON when
`KUTAI_ENV != prod` unless `KUTAI_VENDOR_LIVE=1` (`registry.py:29-37`). The mock envelope is
tagged `mocked: True` (`http_integration.py:271`, `registry.py:96`) — Spec 2's guard keys off
that tag.

## Adapter configs (concrete action tables)

> ⚠️ **API-shape caveat (Opus review):** every external action table below is INDICATIVE and
> MUST be re-verified against each provider's live API reference during the implementation plan.
> The review caught concrete errors (Render `plan`/param-shape, Render env-var semantics,
> Upstash field names) — do a live-doc pass per adapter before writing the config JSON.

### `render.json` (backend host — Render, `https://api.render.com/v1`, bearer)
| action | method | path | required_params | notes |
|---|---|---|---|---|
| `create_service` | POST | `/services` | `[ownerId, type, name, repo, serviceDetails]` | **NO `plan:"free"`** — Render's `plan` enum excludes "free"; free is an *instance type* set inside `serviceDetails`. Body is **nested** (`serviceDetails.runtime` + `envSpecificDetails` build/start or Docker). `repo`=connected git URL. **Create auto-initiates the first deploy** — do NOT also call `trigger_deploy` for first boot. Returns `{service:{id}}`. Flat `required_params` can't express the nested body → the config/executor must build the nested payload explicitly. |
| `get_service` | GET | `/services/{id}` | `[id]` | |
| `trigger_deploy` | POST | `/services/{id}/deploys` | `[id]` | **RE-deploys only** (e.g. after a post-boot env-var change), not first boot. Returns `{id, status:"created"}` |
| `get_deploy` | GET | `/services/{id}/deploys/{deployId}` | `[id, deployId]` | poll target; status→`live`/`build_failed` |
| `update_env_vars` | PUT | `/services/{id}/env-vars` | `[id, envVars]` | inject DATABASE_URL/REDIS_URL. **Does NOT auto-deploy** — Render docs: *"Changes will not be deployed automatically… you must call the deploy API."* So after any post-boot env change, call `trigger_deploy`. Set env at create time (in `serviceDetails.envVars`) to avoid a second deploy. |

### `neon.json` (Postgres — Neon, `https://console.neon.tech/api/v2`, bearer)
| action | method | path | required_params | notes |
|---|---|---|---|---|
| `create_project` | POST | `/projects` | `[project]` | returns `connection_uris[].connection_uri` (the `DATABASE_URL`) |
| `get_project` | GET | `/projects/{project_id}` | `[project_id]` | |
| `list_projects` | GET | `/projects` | `[]` | idempotency check |

> **Nuance (review H4):** Neon has **no generic run-SQL REST endpoint**. Prisma migrations run
> **out of band** against the returned `DATABASE_URL` (`prisma migrate deploy` in Render's
> release command or a one-off shell step in Spec 2). `neon.json` only provisions + returns the
> connection string. Verified: `POST /projects` is fully automatable (research claim, 3-0).

### `upstash.json` (Redis — Upstash Developer API, `https://api.upstash.com/v2`)
| action | method | path | required_params | notes |
|---|---|---|---|---|
| `create_redis` | POST | `/redis/database` | `[database_name, region, primary_region]` | field names per Upstash schema (`database_name`, `region`/`primary_region`, optional `plan`) — **NOT** `name`. Returns `{endpoint, port, password, rest_token}` |
| `get_redis` | GET | `/redis/database/{id}` | `[id]` | |
| `list_redis` | GET | `/redis/databases` | `[]` | idempotency check |

> **Nuance (auth) — corrected per review:** the Upstash **management** API uses **HTTP Basic auth
> (email : API key)**, not bearer. `HttpIntegration` supports bearer/header/query/none/jwt_p8/
> oauth_service_account — **no `basic` type** (verified across all 16 configs). Two options:
> - **(b) DEFAULT, zero engine risk:** store a pre-encoded `Authorization: Basic <b64(email:key)>`
>   as a credential field and use `auth_type:"header"` with `auth_token_field` → the header is set
>   to the raw token value. **This works today unchanged** — `twilio.json` already uses exactly
>   this `auth_type:"header"` + `auth_token_field` pattern. Ship Upstash on (b).
> - **(a) OPTIONAL later:** add a `basic` auth_type to `HttpIntegration` (clean + reusable, but an
>   engine change to a shared, SSRF-sensitive file) — defer unless another provider needs it.
>
> (The research's "POST /start-redis, no signup" is the anonymous quickstart, NOT the management
> API — do not rely on it for owned provisioning.)

### `vercel.json` enrichment (frontend host)
Add:
| action | method | path | required_params | notes |
|---|---|---|---|---|
| `get_deployment` | GET | `/v13/deployments/{id}` | `[id]` | poll target; `readyState`→`READY`/`ERROR` |

Add `mock_responses` for `deploy` + `get_deployment` (see mock tagging).

## Credential schemas (new, in `credential_schemas/`)
- `render.json`: required `[api_key]`; test_endpoint `list `-style GET.
- `neon.json`: required `[api_key]`.
- `upstash.json`: required `[email, api_key]` (basic auth pair).
Follow the shape of existing `credential_schemas/vercel.json`
(required_fields / optional_fields / scopes / rotation_recommended_days / docs_url).

## Mock tagging (review H3)
Every deploy/provision action ships a `mock_responses` block whose payload, when served, is
wrapped by the registry as `{status:"ok", data:..., mocked:true}`. **The `mocked:true` flag must
survive to the caller** so Spec 2's orchestrator can refuse to write `health_check_passed:true`
on a mocked response. This spec only guarantees the tag is present and truthful; the guard lives
in Spec 2.

## Git-host prerequisite chain (review H2 — this is ~60-70% of the real work)
Deploys build **from a connected git repo**. Today `github_init_status.md = "pending:
gh_unauthenticated"` — the app is not pushed anywhere. Required, as explicit i2p steps **before**
the deploy step:
1. **github_auth** — founder provides a GitHub PAT → `credential_store` (service `github`).
2. **create_repo + push_scaffold** — create the repo (`github.json` `create_repo`) and push the
   generated `backend/` + `frontend/` trees. **Push is a git-protocol op, not REST** — implement
   via `shell` `git push` with the PAT-authenticated remote (mechanical), or the GitHub contents
   API blob-by-blob. Recommend PAT `git push` via shell (simpler, atomic).
3. **link_repo_to_provider** — connect the repo to Render/Vercel (Render `create_service` takes
   the repo URL directly; Vercel needs a project↔repo link via its API or a git import).

A deploy-specific **preflight** must verify "repo exists + reachable" so admission
(`z6_admission.py`) does not hand Spec 2 a doomed task (the current gate only checks
adapter-registered + credential-stored, blind to git state).

## Testing
- Per-adapter mock-mode e2e (offline, deterministic) — mirror `tests/tools/test_vendor_call_tool.py`
  and the config-guard tests.
- Config-guard test: every new config loads, actions well-formed, `mock_responses` present +
  `mocked`-tagged on deploy/provision actions.
- Credential-schema shape tests.
- Git-prereq chain: unit-test create-repo + push against a temp repo (no network) with the git
  ops stubbed.
- **No live calls in CI** — real smoke is Spec 2, gated by `KUTAI_VENDOR_LIVE=1` + stored creds,
  founder-run.

## Open questions / decisions for the plan
- Upstash auth: **RESOLVED → option (b)** pre-encoded `Basic` header via `auth_type:"header"`
  (twilio precedent, zero engine risk). Defer the `basic` auth_type engine change.
- Repo push: shell `git push` (recommended) vs contents API.
- Whether the git-prereq chain is new i2p steps vs a mechanical preflight inside Spec 2's
  executor — leaning new explicit steps for auditability.
- Every external API action table needs a live-doc verification pass (Render/Neon/Upstash/Vercel)
  before the config JSON is authored — the review found several indicative-but-wrong shapes.

## References
- Research: `docs/research/2026-08-04-free-deploy-stack-research.md`
- Code: `src/integrations/registry.py`, `http_integration.py`, `src/tools/vendor_call.py`,
  `src/security/credential_store.py`, existing `src/integrations/configs/*.json`.
- Precedent for a multi-step vendor executor: `packages/mr_roboto/src/mr_roboto/executors/`
  (see `stripe_provision_products.py`).
