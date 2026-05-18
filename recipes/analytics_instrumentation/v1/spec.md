# analytics_instrumentation Recipe v1 — Specification

## Scope

Launch-time product analytics instrumentation. Ships a PostHog `track_event`
helper for both the web client (posthog-js) and the backend (posthog-python),
plus the canonical AARRR event taxonomy. Used by i2p Phase 13.4
`analytics_integration` so every shipped product emits standardized events the
Z9 Growth digests can read.

## When to Pick This Recipe

**Use** `analytics_instrumentation/v1` when:
- The product needs post-launch analytics (acquisition / activation /
  retention / revenue / referral)
- Stack includes a web frontend (`nextjs` / `react`) and/or a Python backend
  (`fastapi`)
- The mission has a `success_metrics` artifact (i2p step 2.9) defining
  `aarrr_metrics`

**Do NOT use** when:
- The product is internal-tooling-only with no growth funnel
- A different analytics vendor (Mixpanel/Amplitude/GA4) is mandated — this
  recipe is PostHog-specific; fork for another vendor

## Event Taxonomy

The recipe is the single source of truth for the standard event names. See
`events.template.ts::STANDARD_EVENTS`:

| Stage | Events |
|---|---|
| acquisition | `landing_view`, `signup_started`, `signup_completed` |
| activation | `first_value_event` (renamed per product) |
| retention | `session_started` |
| revenue | `checkout_started`, `checkout_completed`, `subscription_created`, `subscription_cancelled` |
| referral | `share_initiated`, `share_completed`, `invite_redeemed` |

Every event auto-attaches `mission_id`, `feature_id`, `variant`, `segment`,
`business_model`.

## RECIPE_PARAM Markers

| Marker | Default | Description |
|--------|---------|-------------|
| `POSTHOG_API_KEY_ENV` | `POSTHOG_API_KEY` | Env var holding the project API key |
| `POSTHOG_HOST_ENV` | `POSTHOG_HOST` | Env var holding the ingestion host |
| `POSTHOG_HOST_DEFAULT` | `https://us.i.posthog.com` | Fallback host |
| `ACTIVATION_EVENT` | `first_value_event` | Product-specific activation event name |

## Dependencies

**Backend (Python)**: `posthog`
**Frontend (npm)**: `posthog-js`

## Known Non-Goals (v1)

- A/B / feature-flag wiring (Z9 T5 — `variant` is auto-attached but unset here)
- Cohort/segment targeting beyond passing `segment` through (Z9 T5)
- Server-side identity stitching across anonymous → known users
- Non-PostHog vendors
