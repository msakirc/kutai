# KutAI $0 Deployment Stack — Research & Recommendation (2026-08-04)

**Goal:** a default, guaranteed-free, reliable, REST-deployable infrastructure stack the
KutAI agent can autonomously stand up for the full-stack apps it builds (Next.js frontend +
NestJS/Node backend + Postgres + Redis; some Expo/RN mobile), at **exactly $0 at project
init**, with a documented migration-to-paid path once a project earns revenue.

**Method:** deep-research harness — 6 search angles, 31 sources fetched, 146 candidate claims,
25 sent to 3-vote adversarial verification. The run hit the account monthly spend cap during
verification, so the synthesis and ~8 verifier votes did not complete. Confidence is tagged
per claim:

- **[V]** verified this run (adversarial vote in brackets, e.g. `[V 3-0]`).
- **[P]** primary-source claim whose verifier votes were cut off by the spend cap (abstained,
  not refuted) — sourced to the provider's own docs; treat as high-confidence-pending-recheck.
- **[K]** domain knowledge, not independently verified this run — recheck before relying.
- **[REFUTED]** genuinely voted down this run — do not rely.

> ⚠️ Free-tier terms change constantly. Re-verify [P]/[K] items (especially exact limits)
> before wiring a provider. Recheck the whole doc quarterly.

---

## TL;DR — Recommended default $0 stack

| Category | Primary (default) | Backup | No card? | REST API? |
|---|---|---|---|---|
| Frontend / SSR host | **Cloudflare Pages** | Vercel Hobby | ✅ / ✅ | ✅ / ✅ |
| Backend (long-running Node/NestJS) | **Render free web service** ⚠️ sleeps | Cloud Run (needs card) | ✅ / ❌ | ✅ / ✅ |
| Managed Postgres | **Neon** | Supabase (pauses) | ✅ / ✅ | ✅ / ✅ |
| Redis / cache | **Upstash Redis** | Redis Cloud (30 MB) | ✅ / ✅ | ✅ / ✅ |
| Object storage | **Cloudflare R2** | Backblaze B2 / Supabase Storage | ✅ / ✅ | ✅ / ✅ |
| Auth | **Supabase Auth** (bundled w/ DB) | Clerk | ✅ / ✅ | ✅ / ✅ |
| Transactional email | **Resend** | Brevo | ✅ / ✅ | ✅ / ✅ |
| CI/CD | **GitHub Actions** | — | ✅ | ✅ |
| Cron / jobs | **GitHub Actions schedule** | Cloudflare Cron / Upstash QStash | ✅ / ✅ | ✅ |
| Error monitoring | **Sentry** | BetterStack | ✅ / ✅ | ✅ |
| Product analytics | **PostHog** | — | ✅ | ✅ |
| DNS / CDN | **Cloudflare** | — | ✅ | ✅ |

**The one category where $0 genuinely breaks → the backend.** See the flag below.

---

## 🚩 Headline flag: no truly-$0, no-card, always-on backend exists

There is **no managed host that runs a long-running NestJS server always-on, with zero cold
starts, no credit card, and guaranteed-free forever.** Every option trades one of those away:

- **Render free web service** — no card, real $0, full REST API `[V 3-0 API]`, but **spins down
  after ~15 min idle with a ~1 min cold start** `[P — render.com/docs/free]`. Fine for
  staging / low-traffic; a cold first request every idle period.
- **Google Cloud Run** — huge always-free allowance (2M req/mo, 180k vCPU-sec, 360k GB-sec/mo)
  `[V 3-0]` and scales to zero, but **requires a billing account + card** `[V 3-0]` with
  overage risk past the allowance. Effectively $0 at low traffic *if you accept a card*.
- **Oracle Cloud Always Free** (Ampere A1 ARM VM) — genuinely free "for the life of the
  account" `[V 3-0]`, powerful enough to self-host NestJS+Postgres+Redis on one box, but
  **requires a card at signup** `[V 3-0]`, provisions via Terraform not a simple REST call
  `[V 3-0]`, and Oracle has reclaimed idle Always-Free instances (the "permanent, never
  reclaimed" framing was **[REFUTED 1-2]** this run). Highest ops burden.
- **Fly.io** — **[AVOID $0]** the free allowance ended ("the free tier died"), card required,
  pay-as-you-go. Has a clean Machines REST API `[V 3-0]` but it is not $0.
- **Railway** — **[AVOID $0]** removed its free tier Aug 2023 and paused lowest paid tiers
  June 2025 (The Register). Not $0.

**Three honest resolutions for the backend (pick per project):**

1. **Render + keep-warm** (default $0, no card): accept the cold start; a GitHub-Actions cron
   ping every ~10 min keeps it warm during active hours. Zero cost, zero card.
2. **Cloud Run** (accept one card on Google billing): scale-to-zero, $0 at low traffic, best
   longevity of the always-free hosts, no idle sleep. Card + overage risk are the price.
3. **Serverless-native backend (long-term, best endgame):** have KutAI's i2p target a
   serverless/edge backend (Cloudflare Workers + Hono, or Next.js API routes on Vercel)
   instead of an always-on NestJS server. Then the backend is $0-native with no host to keep
   warm. Biggest change (app architecture), but removes this whole category's pain.

**Recommendation:** default to **(1) Render + cron keep-warm** for autonomous $0 init today;
open a separate track to evaluate **(3) serverless backend** as the long-term i2p default.

---

## Per-category detail

### 1. Frontend / static + SSR — **Cloudflare Pages** (primary), Vercel Hobby (backup)
- **Cloudflare Pages**: no card, unlimited sites, free CDN, never gutted its free tier
  (strongest longevity signal in the whole stack), full Cloudflare REST API. `[K]`
- **Vercel Hobby**: no card, Next.js-native, best DX, REST deploy API `[K]`. **Caveat:** Hobby
  is **non-commercial only** — the moment a project monetizes, Vercel ToS requires Pro
  ($20/mo). So Vercel is the better *dev* default but Cloudflare Pages is the better
  *revenue-safe* default. `[K]`
- Netlify / GitHub Pages: viable backups; Netlify free is generous, GH Pages is static-only.

### 2. Backend — see 🚩 flag above. Primary **Render**, backup **Cloud Run** (card).

### 3. Managed Postgres — **Neon** (primary), Supabase (backup)
- **Neon**: no card, ~0.5 GB free, **fully automatable project creation via REST**
  (`POST /projects` at `console.neon.tech/api/v2`, bearer API key) `[V 3-0]`. Scale-to-zero,
  DB branching. Best agent-friendliness of the Postgres options.
- **Supabase**: no card, 500 MB free, bundles Postgres + Auth + Storage in one project.
  **Caveat:** free projects **auto-pause after ~1 week of inactivity** `[P — supabase docs]`;
  needs a cron ping (or real traffic) to stay live. Bundling makes it attractive when you also
  want Auth+Storage from one provider.
- Google Cloud SQL: **no always-free tier**, only a 30-day trial `[V 3-0]` → not $0.

### 4. Redis / cache — **Upstash Redis** (primary)
- **Upstash**: no card `[V 3-0]`, free plan = 256 MB / **500K commands/mo** / 10 GB
  bandwidth, up to 10 DBs `[V 3-0]`. **HTTP/REST API, agent-friendly** — a POST provisions a
  DB instantly `[V 2-1]`. **Positive longevity:** *increased* the free tier in Mar 2025
  (10K/day → 500K/mo) rather than cutting it `[V 3-0]`. Clear category winner.
- Google Memorystore/Redis: **no free tier** `[V 3-0]`.
- Redis Cloud free: 30 MB — too small for most.

### 5. Object storage — **Cloudflare R2** (primary)
- **R2**: 10 GB free, **zero egress fees**, S3-compatible API, no card. `[K]` The no-egress
  policy is the differentiator vs S3/others.
- Backups: Backblaze B2 (10 GB), Supabase Storage (1 GB, bundled).

### 6. Auth — **Supabase Auth** (primary), Clerk (backup)
- **Supabase Auth**: free with the Postgres you're already running (~50k MAU free), one
  provider for DB+Auth+Storage. `[K]`
- **Clerk**: ~10k MAU free, best-in-class DX, no card. `[K]` Better if you're not on Supabase.
- Firebase Auth: very generous but pulls you into Google billing for adjacent services.

### 7. Transactional email — **Resend** (primary), Brevo (backup)
- **Resend**: 3k emails/mo (100/day) free, no card, clean REST API, current dev favorite. `[K]`
- **Brevo**: 300 emails/day free. **SendGrid** (100/day) is Twilio-owned and has tightened
  its free terms before — usable backup, not the default.

### 8. CI/CD — **GitHub Actions**
- 2,000 min/mo free on private repos, **unlimited on public repos** `[P — GitHub docs]`.
  Without a payment method it **blocks at quota rather than charging overage** `[P]` → genuinely
  $0. Microsoft-backed, strong longevity. It's the default; already where the code lives.

### 9. Cron / background jobs — **GitHub Actions schedule** (primary)
- GH Actions `schedule:` — free, but min ~5-min cadence and can be delayed under load. `[K]`
- **Cloudflare Cron Triggers** (Workers, free) and **Upstash QStash** (500 msgs/day free) are
  strong backups; QStash is nice for HTTP-callback jobs from an agent. `[K]`

### 10. Monitoring + analytics — **Sentry** + **PostHog**
- **Sentry**: 5k errors/mo, 1 project, no card. `[K]`
- **PostHog**: ~1M events/mo free — very generous, no card. `[K]`
- BetterStack: good uptime/log backup.

### 11. DNS / CDN — **Cloudflare** (undisputed)
- Free DNS + CDN, never gutted, full REST API, best longevity in the stack. Anchor the whole
  stack's DNS/CDN/edge here. `[K]`

---

## AVOID for $0 (not truly free / not no-card)

| Provider | Reason |
|---|---|
| **Railway** | Removed free tier Aug 2023; paused lowest paid tiers June 2025 (The Register). `[K]` The existing `railway.json` adapter should **not** be a $0 target. |
| **Fly.io** | Free allowance ended ("free tier died"); card required, pay-as-you-go overage. `[K]` |
| **Heroku** | Removed free dynos Nov 2022 — historical warning, still relevant. `[K]` |
| **Google Cloud / Cloud Run / Oracle Always Free** | "Always free" but **card required** `[V 3-0]` + overage/reclaim risk. Only use if the founder consciously accepts a card. |
| **SendGrid as default** | Twilio-owned, tightened free terms; prefer Resend. `[K]` |

---

## Reliability ranking (weighted toward free-tier longevity)

- **Tier A — rock-solid, never gutted a free tier, no card:** Cloudflare (Pages / R2 / DNS /
  Workers / Cron), GitHub (Actions), Upstash (*grew* its free tier), Neon, Resend, PostHog,
  Sentry.
- **Tier B — solid $0 with one caveat:** Render (idle-sleep), Supabase (1-week pause), Vercel
  Hobby (non-commercial only), Clerk.
- **Tier C — AVOID for $0:** Railway (killed free), Fly.io (killed free), Heroku (killed free),
  GCP / Oracle (card required).

**Track record is the signal.** Providers that *killed* free tiers (Railway, Fly, Heroku) are
the risk. Providers that *grew* them (Upstash) or never touched them (Cloudflare, GitHub) are
the safe long-term bets. Anchor KutAI's default stack on Tier A.

---

## Migration-to-paid triggers (only once a project earns revenue)

| Category | Free → Paid trigger | Target |
|---|---|---|
| Frontend | Commercial use / >100 GB bandwidth (Vercel) | Vercel Pro $20/mo, or stay on Cloudflare Pages (free even commercial) |
| Backend | Cold starts hurt UX / need always-on | Render Starter $7/mo (no sleep), or Cloud Run pay-as-you-go past allowance |
| Postgres | >0.5 GB / compute cap (Neon) or pausing hurts (Supabase) | Neon Launch ~$19/mo; Supabase Pro $25/mo |
| Redis | >500K commands/mo | Upstash pay-as-you-go |
| Storage | >10 GB | R2 ~$0.015/GB-mo, still no egress fee |
| Auth | >50k MAU (Supabase) / >10k (Clerk) | provider paid tier |
| Email | >3k emails/mo | Resend $20/mo |

**Principle:** stay Tier-A-free at init; migrate one category at a time only when a real usage
signal (traffic, MAU, revenue) crosses the free ceiling. Never pre-emptively pay.

---

## KutAI adapter gap-map — research → build list

How the recommended stack maps onto KutAI's existing `src/integrations/configs/*.json`
adapters + `credential_schemas/`:

| Category | Pick | Config exists? | Cred schema? | Work needed |
|---|---|---|---|---|
| Frontend | Cloudflare Pages | `cloudflare.json` ✅ | cloudflare ✅ | add Pages deploy actions |
| Frontend alt | Vercel | `vercel.json` ✅ | vercel ✅ | add `get_deployment` poll + `mock_responses` |
| Backend | **Render** | ❌ | ❌ | **build `render.json`** (create service, deploy, get status) + cred schema |
| Postgres | **Neon** | ❌ | ❌ | **build `neon.json`** (create project, run SQL) + cred schema |
| Postgres alt | Supabase | `supabase.json` ✅ (`run_migration`) | supabase ✅ | add keep-alive cron |
| Redis | **Upstash** | ❌ | ❌ | **build `upstash.json`** (create db) + cred schema |
| Object storage | Cloudflare R2 | `cloudflare.json` ✅ | cloudflare ✅ | add R2 bucket/object actions |
| Auth | Supabase Auth | `supabase.json` ✅ | supabase ✅ | add auth actions |
| Email | **Resend** | ❌ | ❌ | **build `resend.json`** + cred schema |
| Email alt | SendGrid | `sendgrid.json` ✅ | sendgrid ✅ | — |
| CI/CD | GitHub Actions | `github.json` ✅ | github ✅ | add workflow-dispatch actions |
| Cron | GH Actions / CF | `github.json` / `cloudflare.json` ✅ | ✅ | — |
| Monitoring | Sentry | `sentry.json` ✅ | sentry ✅ | — |
| Analytics | PostHog | `posthog.json` ✅ | ❌ | add posthog cred schema |
| DNS/CDN | Cloudflare | `cloudflare.json` ✅ | cloudflare ✅ | — |

**New adapters to build:** `render`, `neon`, `upstash`, `resend`.
**Enrich existing:** `cloudflare` (Pages + R2 actions), `vercel` (status-poll + mock),
`supabase` (auth actions + keep-alive), `github` (workflow dispatch).
**Retarget away from $0:** `railway.json` — keep for paid use later, not a default $0 target.

Every adapter is a declarative `HttpIntegration` config (auto-discovered from
`src/integrations/configs/`), auth via `credential_store`, reachable by the `executor` agent
through the `vendor_call` tool. Mock mode (`KUTAI_ENV != prod`) lets each adapter be
CI-tested offline via `mock_responses`; real calls need `KUTAI_VENDOR_LIVE=1` + a stored
credential.

---

## Direct relevance to m90 / task 7.13 (staging_environment)

7.13 wants `staging_environment{url,services}` + `staging_deployment_verified{deployed,
health_check_passed}` for HabitHub (Next.js + NestJS + Postgres + Redis). The $0 path:

- Frontend → Cloudflare Pages or Vercel (free).
- Backend → Render free web service (accept cold start) + cron keep-warm.
- Postgres → Neon or Supabase (free) — run Prisma migrations via the DB's SQL/migration API.
- Redis → Upstash (free).

**But** a real run still needs the founder to (a) create free accounts, (b) `/credential add`
each token, (c) set `KUTAI_VENDOR_LIVE=1`, and (d) resolve GitHub auth
(`github_init_status.md` = `pending: gh_unauthenticated`) so the app can be pushed for
deploy. The `must_be_true: health_check_passed` gate stays honest — no mock is used to satisfy
a live deploy.

---

## Sources (31 fetched; primary sources bolded)

- **oracle.com/cloud/free**, **docs.oracle.com/.../Always_Free_Resources**, **cloud.google.com/free**,
  **upstash.com/pricing/redis**, **upstash.com/blog/redis-new-pricing**, **api-docs.neon.tech/reference/createproject**,
  **render.com/docs/api**, **render.com/docs/free**, **docs.machines.dev** (Fly),
  **docs.github.com/billing/.../about-billing-for-github-actions**,
  **supabase.com/docs/guides/platform/free-project-pausing**
- Secondary/analysis: theregister.com/2025/06/16/railway_pauses_lowest_tiers, agentdeals.dev
  (hosting/database free-tier comparisons 2026, neon vendor), srvrlss.io (upstash, fly pay-as-you-go),
  saaspricepulse.com (flyio, railway), uibakery.io/blog/supabase-pricing
- Blogs: expresstech.io (Fly alternatives / "free tier died"), birjob.com (PaaS comparison 2026),
  render.com/articles (real free tier 2026), merginit.com (auth, object-storage comparisons),
  dreamlit.ai (SendGrid alternatives), vemetric.com (PostHog vs Sentry), metacto.com (Firebase Auth),
  tellmewhendown.com (Neon autosuspend), snapdeploy.dev (free deploy platforms 2026)

*Run stats: 6 angles · 31 sources · 146 claims · 25 verified · 17 confirmed · 2 refuted · 6
abstained-on-spend-cap. Synthesis completed manually after the harness hit the account spend
limit.*
