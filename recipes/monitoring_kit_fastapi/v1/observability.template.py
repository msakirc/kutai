"""monitoring_kit_fastapi/v1 — error tracking + structured logging.

Call ``init_observability()`` once at app startup, before the app starts
serving (e.g. top of the FastAPI lifespan, or right after ``app = FastAPI()``).

Free-first: Sentry's free tier covers error tracking for a small product;
``init_observability`` is a no-op when SENTRY_DSN is unset, so local/dev
runs need no account. Structured JSON logs cost nothing and make any log
aggregator (or just ``grep``) usable.

RECIPE_PARAM markers (leave intact):
  # RECIPE_PARAM:SENTRY_DSN_ENV=SENTRY_DSN
  # RECIPE_PARAM:ENVIRONMENT_ENV=APP_ENV
"""
from __future__ import annotations

import json
import logging
import os
import sys

_SENTRY_DSN_ENV = "SENTRY_DSN"   # RECIPE_PARAM:SENTRY_DSN_ENV=SENTRY_DSN
_ENVIRONMENT_ENV = "APP_ENV"     # RECIPE_PARAM:ENVIRONMENT_ENV=APP_ENV


class _JsonFormatter(logging.Formatter):
    """One JSON object per log line — parseable by any aggregator."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _init_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def _init_sentry() -> bool:
    """Initialise Sentry when a DSN is present. Returns True when wired."""
    dsn = os.getenv(_SENTRY_DSN_ENV, "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        logging.getLogger("monitoring_kit").warning(
            "SENTRY_DSN set but sentry-sdk not installed — error tracking off"
        )
        return False
    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv(_ENVIRONMENT_ENV, "development"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
        send_default_pii=False,
    )
    return True


def init_observability() -> dict:
    """Wire structured logging + (optional) Sentry. Idempotent-safe to call once."""
    _init_logging()
    sentry_on = _init_sentry()
    logging.getLogger("monitoring_kit").info(
        "observability initialised (sentry=%s)", "on" if sentry_on else "off"
    )
    return {"logging": "json", "sentry": sentry_on}
