# analytics_instrumentation v1

Ships PostHog instrumentation scaffolding so a launched product emits the
standardized **AARRR event taxonomy**. Consumed by i2p Phase 13.4
`analytics_integration` and read back by Z9 weekly digests.

## What the recipe instantiates

| File | Role |
|---|---|
| `client.template.ts` | posthog-js web shim — `initAnalytics()`, `setAnalyticsContext()`, `track_event(name, properties)` |
| `server.template.py` | posthog-python backend shim — same helper for server-side events |
| `events.template.ts` | the canonical `STANDARD_EVENTS` taxonomy constant + property hints |

## Standard AARRR event taxonomy

- **acquisition**: `landing_view`, `signup_started`, `signup_completed`
- **activation**: `first_value_event` (rename per product via `ACTIVATION_EVENT`)
- **retention**: `session_started` (attach `day_of_cohort`)
- **revenue**: `checkout_started`, `checkout_completed`, `subscription_created`, `subscription_cancelled`
- **referral**: `share_initiated`, `share_completed`, `invite_redeemed`

Every event auto-attaches: `mission_id`, `feature_id`, `variant` (if A/B
active), `segment` (if cohort targeting active), `business_model`.

## How Phase 13.4 uses it

The `analytics_integration` agent reads the `success_metrics` artifact (i2p
step 2.9), iterates `success_metrics.aarrr_metrics`, and for each metric:

1. Maps the metric's funnel stage to the matching `STANDARD_EVENTS` names.
2. Inserts a `track_event(...)` call at the relevant code site (web shim for
   browser-observable events, server shim for webhook-confirmed events such as
   `checkout_completed` / `subscription_*`).
3. Wires `initAnalytics()` at app bootstrap and `setAnalyticsContext()` with
   the mission/feature metadata.

## Env contract

- `POSTHOG_API_KEY` / `NEXT_PUBLIC_POSTHOG_API_KEY` — project API key
- `POSTHOG_HOST` / `NEXT_PUBLIC_POSTHOG_HOST` — ingestion host
  (default `https://us.i.posthog.com`)

When the API key is absent both shims log a warning and `track_event` becomes
a no-op — dev/test builds never crash.
