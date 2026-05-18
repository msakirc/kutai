# Next.js — analytics_instrumentation/v1 instantiation notes

When wiring this recipe into a Next.js App Router project:

- **Init at the root layout**: call `initAnalytics()` once in a top-level Client
  Component (`app/providers.tsx` with `"use client"`) so PostHog boots before
  any `track_event` fires. Then call `setAnalyticsContext({ mission_id, feature_id, business_model })`
  immediately after.
- **`landing_view` placement**: emit `track_event("landing_view", {...})` from
  the marketing/landing route's mount effect — NOT from a middleware. Attach
  `referrer` and `utm_*` query params.
- **Signup funnel**: `signup_started` on the signup form mount; `signup_completed`
  is best emitted server-side (posthog-python shim) right after the user row
  commits, so it reflects real account creation.
- **`RECIPE_PARAM:ACTIVATION_EVENT`**: replace `first_value_event` with the
  product's real first-value moment. This is the activation funnel's anchor.
- **Env vars**: the client shim reads `NEXT_PUBLIC_POSTHOG_API_KEY` and
  `NEXT_PUBLIC_POSTHOG_HOST`. Add both to `.env.local` and the deploy
  environment. Without the key, `track_event` no-ops with a console warning.
- **Map from success_metrics**: for each `aarrr_metrics` entry, find its funnel
  stage, pull the matching `STANDARD_EVENTS` names from `events.ts`, and place a
  `track_event` call at the code site that realizes that metric's `formula`.
