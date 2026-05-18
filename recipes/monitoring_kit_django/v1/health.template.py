"""monitoring_kit_django/v1 — health-check views.

Wire into the project's urls.py:

    from .health import healthz, readyz
    urlpatterns += [
        path("healthz/", healthz),   # liveness
        path("readyz/", readyz),     # readiness — DB ping
    ]

  healthz — liveness. 200 while the process serves. Poll from the external
            uptime monitor / restart probe.
  readyz  — readiness. 200 only when the DB answers; 503 otherwise. Deploy
            gates and load balancers should poll this one.

RECIPE_PARAM markers (leave intact — substituted at instantiation, must
survive ast.parse()):
  # RECIPE_PARAM:HEALTH_PATH=healthz/
  # RECIPE_PARAM:READY_PATH=readyz/
"""
from __future__ import annotations

from django.db import connection
from django.http import JsonResponse


def healthz(_request) -> JsonResponse:
    """Liveness — the process is up and serving."""
    return JsonResponse({"status": "ok"})


def readyz(_request) -> JsonResponse:
    """Readiness — the default database answers a trivial query."""
    db_ok = True
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception:  # noqa: BLE001 — any DB error is a readiness failure
        db_ok = False
    checks = {"database": db_ok}
    ok = all(checks.values())
    return JsonResponse(
        {"status": "ready" if ok else "degraded", "checks": checks},
        status=200 if ok else 503,
    )
