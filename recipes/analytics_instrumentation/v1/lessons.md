# analytics_instrumentation Recipe v1 — Known Lessons

Pitfalls captured from prior implementations. Seeded into `mission_lessons` on recipe instantiation.

- **`NEXT_PUBLIC_` prefix is mandatory for client env vars**: posthog-js runs in the browser. The API key env var MUST be prefixed `NEXT_PUBLIC_` (Next.js) or it is undefined at runtime and `track_event` silently no-ops. The server shim uses the bare `POSTHOG_API_KEY` — do not cross them.
- **Server-side `flush()` before process exit**: posthog-python batches events. Short-lived processes (serverless functions, webhook handlers) must call `posthog.flush()` before returning or events are lost. The shim does not auto-flush — wire it in webhook handlers.
- **Revenue events belong server-side**: emit `checkout_completed` / `subscription_created` / `subscription_cancelled` from the Stripe webhook handler, not the browser. Browser-side revenue events are spoofable and miss async confirmation. Use the posthog-python shim for these.
- **`distinct_id` must be stable across client and server**: the web shim uses PostHog's auto-generated distinct_id; the server shim requires you to pass one explicitly. Pass the same user id on both sides (call `posthog.identify()` on login) or funnels split into two users.
- **`capture_pageview: false` is intentional**: the recipe emits `landing_view` explicitly so it carries the mission/feature metadata. Do not re-enable PostHog auto-pageviews — it double-counts and the auto events lack the metadata block.
- **Activation event must be renamed**: `first_value_event` is a placeholder. Leaving it generic makes the activation funnel meaningless. Substitute via `RECIPE_PARAM:ACTIVATION_EVENT` to the product's real first-value moment (e.g. `first_note_created`).
- **`session_started` needs `day_of_cohort`**: retention analysis is cohort-based. Always attach `day_of_cohort` (days since signup) to `session_started` or retention curves cannot be computed.
- **Do not block the request on `capture()`**: posthog-js `capture` is async/non-blocking by design; the server `capture` enqueues. Never `await` analytics in a hot path or wrap it so a PostHog outage cannot fail a user request.
- **Cookie consent gate**: in regions requiring consent, call `posthog.opt_out_capturing()` until consent is granted, then `posthog.opt_in_capturing()`. The shim does not gate on consent — wire it to the consent banner.
