"""Mechanical deploy_staging orchestrator — stands up a $0 staging env, writes 7.13 artifacts.

Reaches adapters via DIRECT adapter.execute() (NOT mr_roboto vendor_call — that wrapper
strips the ``mocked`` flag the anti-fake guard depends on). A mock response can never
certify health_check_passed:true.
"""
from __future__ import annotations
from typing import Any

def _is_mocked(envelope: dict) -> bool:
    """True when a registry response carries the mock tag."""
    return bool(isinstance(envelope, dict) and envelope.get("mocked") is True)

async def _call(service: str, action: str, params: dict) -> dict:
    """Direct adapter.execute(); preserves the mocked flag."""
    from src.integrations.registry import get_integration_registry
    adapter = get_integration_registry().get(service)
    if adapter is None:
        return {"status": "error", "error": f"adapter {service} not registered"}
    return await adapter.execute(action, params)

def _fail(reason: str, **extra) -> dict:
    return {"ok": False, "reason": reason, **extra}

async def _repo_from_mission(mission_id) -> str | None:
    """Read the pushed repo URL persisted by Plan-1 git_prepare_repo (missions.github_repo_url).
    The payload's {git_repo_url} placeholder is NOT substituted at expansion (only {mission_id}
    is) — Opus review Bug 1. Resolve from the mission row instead."""
    try:
        from dabidabi import get_db
        db = await get_db()
        cur = await db.execute("SELECT github_repo_url FROM missions WHERE id = ?", (mission_id,))
        row = await cur.fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None

async def _resolve_workspace(mission_id, payload_ws) -> str | None:
    """Absolute mission workspace. A payload-relative 'mission_{id}/' would resolve against the
    process CWD (Opus review Bug 2) — re-root under WORKSPACE_DIR via get_mission_workspace."""
    import os
    if payload_ws and os.path.isabs(payload_ws):
        return payload_ws
    if mission_id is not None:
        from src.tools.workspace import get_mission_workspace
        return get_mission_workspace(int(mission_id))
    return payload_ws

async def run(task: dict) -> dict[str, Any]:
    payload = (task.get("payload") or (task.get("context") or {}).get("payload") or {})
    backend_arch = payload.get("backend_arch", "nestjs_render")
    if backend_arch != "nestjs_render":
        return _fail("serverless_not_yet_supported")
    ctx = task.get("context") or {}
    mission_id = ctx.get("mission_id") or payload.get("mission_id")

    repo = payload.get("repo")
    if (not repo or "{" in str(repo)) and mission_id is not None:
        repo = await _repo_from_mission(mission_id)
    if not repo or "{" in str(repo):
        return _fail("missing repo (no github_repo_url on mission)")

    workspace = await _resolve_workspace(mission_id, payload.get("workspace"))
    if not workspace:
        return _fail("missing workspace")

    state = {"mocked_any": False, "services": {}, "provisioned": []}
    # DAG steps 2-9 are added in later tasks; skeleton returns not-implemented for now.
    return _fail("dag_not_implemented", state=state)


async def _provision(mission_id) -> dict:
    """Provision Neon Postgres + Upstash Redis; return env vars + service descriptors.

    Idempotent: list-before-create keyed on a mission tag would go here (list_* returns []
    in mock; real impl reuses a resource named kutay_mission_{id}). Returns
    {ok, env:{DATABASE_URL,REDIS_URL}, services:{db,cache}, mocked}.
    """
    name = f"kutay_mission_{mission_id}"
    db = await _call("neon", "create_project", {"project": {"name": name}})
    if db.get("status") != "ok":
        return _fail("neon_provision_failed", detail=db.get("error"))
    conn = (db.get("data", {}).get("connection_uris") or [{}])[0].get("connection_uri")
    if not conn:
        return _fail("neon_no_connection_uri")

    cache = await _call("upstash", "create_redis", {"database_name": name, "region": "us-east-1"})
    if cache.get("status") != "ok":
        return _fail("upstash_provision_failed", detail=cache.get("error"))
    cd = cache.get("data", {})
    redis_url = f"rediss://default:{cd.get('password')}@{cd.get('endpoint')}:{cd.get('port')}"

    return {
        "ok": True,
        "env": {"DATABASE_URL": conn, "REDIS_URL": redis_url},
        "services": {"db": {"provider": "neon"}, "cache": {"provider": "upstash"}},
        "mocked": _is_mocked(db) or _is_mocked(cache),
    }
