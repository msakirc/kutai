"""monitoring_kit_fastapi/v1 — health-check endpoints.

Mount this router on the FastAPI app: ``app.include_router(health_router)``.

  /healthz — liveness. Always 200 while the process is up. Used by the
             external uptime monitor and by any orchestrator restart probe.
  /readyz  — readiness. 200 only when dependencies (DB) answer; 503 otherwise.
             Load balancers / deploy gates should poll this, not /healthz.

RECIPE_PARAM markers (leave intact — substituted at instantiation, must
survive ast.parse()):
  # RECIPE_PARAM:HEALTH_PATH=/healthz
  # RECIPE_PARAM:READY_PATH=/readyz
"""
from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

health_router = APIRouter(tags=["monitoring"])

_STARTED_AT = time.time()


async def _check_database() -> bool:
    """Return True when the database answers a trivial query.

    Replace the body with the project's real connection check — e.g.
    ``await database.execute("SELECT 1")`` (databases),
    ``await conn.execute(text("SELECT 1"))`` (SQLAlchemy async), or an
    ``aiosqlite`` ``SELECT 1``. Kept dependency-free here so the template
    passes imports_check before the project wires its own DB handle.
    """
    return True


@health_router.get("/healthz")  # RECIPE_PARAM:HEALTH_PATH=/healthz
async def healthz() -> dict:
    """Liveness — the process is up and serving."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _STARTED_AT, 1)}


@health_router.get("/readyz")  # RECIPE_PARAM:READY_PATH=/readyz
async def readyz() -> JSONResponse:
    """Readiness — every critical dependency answers."""
    checks = {"database": await _check_database()}
    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "degraded", "checks": checks},
    )
