# monitoring_kit_nextjs/v1 — setup notes

A free-first observability baseline for a Next.js (App Router) app. No paid APM.

## What this recipe scaffolds

| File | Purpose |
|---|---|
| `health.template.ts` | `GET /api/health` route handler (place at `app/api/health/route.ts`) |
| `observability.template.ts` | env-gated Sentry init + JSON `log()` helper |

## Wiring (3 steps)

1. Place `health.template.ts` at `app/api/health/route.ts`.
2. Call `initObservability()` from `instrumentation.ts`'s `register()` hook.
3. Fill in `checkDependencies()` with the project's real dependency pings.

For full browser+server coverage run `npx @sentry/wizard@latest -i nextjs` —
it generates the `instrumentation.ts` + client/server config; this helper is
the minimal fallback when the wizard has not been run.

## Uptime monitoring (free)

Point a free monitor at `/api/health`:

- **UptimeRobot** free tier — 50 monitors, 5-min interval.
- **cron-job.org** — free, 1-min interval.
- **Vercel** — built-in deployment health; add a cron `GET /api/health` for app-level.

## Alert rules (start here)

| Signal | Threshold | Action |
|---|---|---|
| `/api/health` non-200 | 2 consecutive failures | page founder |
| Sentry new-issue | any unhandled exception | Slack/Telegram notify |
| Sentry error rate | >5× the 24h baseline | page founder |
| Core Web Vitals (LCP) | p75 > 2.5s | review founder_action |

## Cost

$0 at MVP scale — Sentry free tier + a free uptime monitor + Vercel's
built-in analytics cover a pre-revenue product.
