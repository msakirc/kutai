# monitoring_kit_fastapi/v1 — setup notes

A free-first observability baseline for a FastAPI service. No paid APM.

## What this recipe scaffolds

| File | Purpose |
|---|---|
| `health.template.py` | `/healthz` (liveness) + `/readyz` (readiness, DB ping) |
| `observability.template.py` | env-gated Sentry init + structured JSON logging |

## Wiring (3 steps)

1. `app.include_router(health_router)` — mount the health endpoints.
2. Call `init_observability()` once at startup (FastAPI lifespan / post-`FastAPI()`).
3. Fill in `_check_database()` with the project's real `SELECT 1`.

## Uptime monitoring (free)

`/healthz` and `/readyz` are designed to be polled by a free external monitor:

- **UptimeRobot** free tier — 50 monitors, 5-min interval.
- **cron-job.org** — free, 1-min interval, hits a URL on a schedule.
- **GitHub Actions** scheduled workflow — `curl --fail $URL/healthz` every N min.

Point the monitor at `/readyz` (catches dependency outages, not just process-up).

## Alert rules (start here)

| Signal | Threshold | Action |
|---|---|---|
| `/readyz` non-200 | 2 consecutive failures | page founder |
| Sentry new-issue | any unhandled exception | Slack/Telegram notify |
| Sentry error rate | >5× the 24h baseline | page founder |
| TLS certificate | expires < 14 days | founder_action |

## Cost

$0 at MVP scale. Sentry free tier (~5k errors/mo) + a free uptime monitor
cover a pre-revenue product. Add a paid APM only when request-tracing
becomes load-bearing.
