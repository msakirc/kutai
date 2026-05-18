# FastAPI — analytics_instrumentation/v1 instantiation notes

When wiring this recipe into a FastAPI backend:

- **Init at startup**: call `init_analytics()` in the FastAPI `lifespan`
  startup handler, then `set_analytics_context(mission_id=..., feature_id=...,
  business_model=...)`.
- **Server-side events only**: use the posthog-python shim for events the
  browser cannot confirm reliably:
  - `signup_completed` — after the user row commits
  - `checkout_completed`, `subscription_created`, `subscription_cancelled` —
    from the Stripe webhook handler (these are the revenue funnel; never trust
    the browser for them)
- **Flush before exit**: posthog-python batches. In webhook handlers / short
  request paths call `posthog.flush()` before returning, or events are lost.
- **`distinct_id`**: pass the same stable user id the web client uses, so
  client + server events stitch into one funnel.
- **Env vars**: the server shim reads bare `POSTHOG_API_KEY` and `POSTHOG_HOST`
  (NO `NEXT_PUBLIC_` prefix — that is client-only). Add both to the backend
  environment.
- **Map from success_metrics**: for each `aarrr_metrics` entry whose
  `data_source` is server/database/webhook, emit the matching `STANDARD_EVENTS`
  event via `track_event` at the endpoint that realizes the metric.
- **Never block the request**: `track_event` enqueues; do not `await` analytics
  on the hot path and ensure a PostHog outage cannot fail a user request.
