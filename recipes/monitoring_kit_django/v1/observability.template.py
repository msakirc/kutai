"""monitoring_kit_django/v1 — error tracking + structured logging.

Import and call ``init_observability()`` from the bottom of ``settings.py``
(after ``DEBUG`` / ``ALLOWED_HOSTS`` are defined), or merge ``LOGGING_JSON``
into the project's ``LOGGING`` setting directly.

Free-first: Sentry's free tier covers error tracking; ``init_observability``
is a no-op when SENTRY_DSN is unset, so local/dev runs need no account.

RECIPE_PARAM markers (leave intact):
  # RECIPE_PARAM:SENTRY_DSN_ENV=SENTRY_DSN
  # RECIPE_PARAM:ENVIRONMENT_ENV=DJANGO_ENV
"""
from __future__ import annotations

import os

_SENTRY_DSN_ENV = "SENTRY_DSN"   # RECIPE_PARAM:SENTRY_DSN_ENV=SENTRY_DSN
_ENVIRONMENT_ENV = "DJANGO_ENV"  # RECIPE_PARAM:ENVIRONMENT_ENV=DJANGO_ENV

# Merge this into the project's LOGGING setting for one-JSON-object-per-line
# output (parseable by any aggregator; needs python-json-logger).
LOGGING_JSON = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "json"},
    },
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
}


def init_sentry() -> bool:
    """Initialise Sentry when a DSN is present. Returns True when wired."""
    dsn = os.getenv(_SENTRY_DSN_ENV, "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.django import DjangoIntegration
    except ImportError:
        return False
    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv(_ENVIRONMENT_ENV, "development"),
        integrations=[DjangoIntegration()],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
        send_default_pii=False,
    )
    return True


def init_observability() -> dict:
    """Wire Sentry. Call from settings.py; merge LOGGING_JSON into LOGGING."""
    return {"sentry": init_sentry(), "logging": "merge LOGGING_JSON into LOGGING"}
