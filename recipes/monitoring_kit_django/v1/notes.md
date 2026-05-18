# monitoring_kit_django/v1 — setup notes

A free-first observability baseline for a Django project. No paid APM.

## What this recipe scaffolds

| File | Purpose |
|---|---|
| `health.template.py` | `healthz/` (liveness) + `readyz/` (readiness, DB ping) views |
| `observability.template.py` | env-gated Sentry init + `LOGGING_JSON` config |

## Wiring (3 steps)

1. Add the two views to `urlpatterns` in the project `urls.py`.
2. From the bottom of `settings.py`: `from .observability import init_observability, LOGGING_JSON`,
   merge `LOGGING_JSON` into `LOGGING`, then call `init_observability()`.
3. `pip install sentry-sdk python-json-logger` (both free / open-source).

## Uptime monitoring (free)

Point a free external monitor at `readyz/`:

- **UptimeRobot** free tier — 50 monitors, 5-min interval.
- **cron-job.org** — free, 1-min interval.
- **GitHub Actions** scheduled workflow — `curl --fail $URL/readyz`.

`readyz/` catches DB outages; `healthz/` only confirms the process is up.

## Alert rules (start here)

| Signal | Threshold | Action |
|---|---|---|
| `readyz/` non-200 | 2 consecutive failures | page founder |
| Sentry new-issue | any unhandled exception | Slack/Telegram notify |
| Sentry error rate | >5× the 24h baseline | page founder |
| TLS certificate | expires < 14 days | founder_action |

## Cost

$0 at MVP scale. Sentry free tier + a free uptime monitor cover a
pre-revenue product. Add a paid APM only when distributed tracing is
genuinely load-bearing.
