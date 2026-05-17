"""yalayut.admin — founder ops for auth secrets + MCP process control.

Phase 3 ops: missing_auth, set_secret, mcp_status, mcp_restart, mcp_kill.
These back the /yalayut auth ... and /yalayut mcp ... Telegram commands.
"""
# ---------------------------------------------------------------------------
# Phase 3 — auth env-var + MCP process control
# ---------------------------------------------------------------------------
from typing import Any as _Any

from src.infra.logging_config import get_logger as _get_logger

_p3_logger = _get_logger("yalayut.admin.phase3")


async def _db_query(sql: str, params: tuple = ()) -> list[tuple]:
    """Run a read query against the main DB. Patched in tests."""
    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(sql, params)
    return await cur.fetchall()


async def _revet_artifacts_for_env(env_key: str) -> None:
    """Recompute env_status for every artifact whose auth depends on ``env_key``.

    An artifact depends on the key if its api ``auth_env`` equals it or its mcp
    ``env_required`` list contains it. After a founder adds the secret these
    artifacts flip from ``missing_<KEY>`` to ``ready`` and become matchable.
    """
    import yaml

    from src.infra.db import get_db
    from yalayut.secrets import compute_env_status

    db = await get_db()
    cur = await db.execute(
        "SELECT id, manifest_path, env_status FROM yalayut_index "
        "WHERE artifact_type IN ('api', 'mcp') AND env_status != 'ready'"
    )
    rows = await cur.fetchall()
    for artifact_id, manifest_path, _status in rows:
        if not manifest_path:
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = yaml.safe_load(fh) or {}
        except OSError:
            continue
        if manifest.get("artifact_type") == "api":
            required = [manifest.get("api", {}).get("auth_env")]
        else:
            required = (manifest.get("mcp", {}) or {}).get("env_required") or []
        required = [k for k in required if k]
        if env_key not in required:
            continue
        new_status = await compute_env_status(required)
        await db.execute(
            "UPDATE yalayut_index SET env_status = ? WHERE id = ?",
            (new_status, artifact_id),
        )
    await db.commit()


async def set_secret(key_name: str, value: str) -> dict[str, _Any]:
    """Encrypt + store an auth secret, then re-vet artifacts that needed it."""
    from yalayut.secrets import set_secret as _store

    try:
        await _store(key_name, value)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    try:
        await _revet_artifacts_for_env(key_name)
    except Exception as e:
        _p3_logger.warning("re-vet after set_secret failed", err=str(e))
    return {"ok": True, "key_name": key_name}


async def missing_auth() -> list[dict[str, _Any]]:
    """List api/mcp artifacts blocked by a missing auth env var."""
    rows = await _db_query(
        "SELECT id, name, env_status FROM yalayut_index "
        "WHERE env_status LIKE 'missing_%' AND enabled = 1 "
        "ORDER BY name"
    )
    out = []
    for artifact_id, name, env_status in rows:
        missing_key = env_status[len("missing_"):] if env_status else ""
        out.append({"artifact_id": artifact_id, "name": name,
                    "missing_key": missing_key})
    return out


async def mcp_status() -> list[dict[str, _Any]]:
    """Running MCP servers + health + fail counts."""
    from yalayut.mcp_manager import get_manager

    return list(get_manager().status())


async def mcp_restart(artifact_id: int) -> dict[str, _Any]:
    """Shut down then re-start an MCP server (manual founder control)."""
    import yaml

    from yalayut.mcp_manager import get_manager

    manager = get_manager()
    await manager.shutdown(artifact_id)
    rows = await _db_query(
        "SELECT manifest_path FROM yalayut_index WHERE id = ?", (artifact_id,)
    )
    if not rows or not rows[0][0]:
        return {"ok": False, "error": f"no manifest for artifact {artifact_id}"}
    try:
        with open(rows[0][0], "r", encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh) or {}
    except OSError as e:
        return {"ok": False, "error": str(e)}
    mcp = manifest.get("mcp") or {}
    handle = await manager.ensure_running(artifact_id, mcp)
    return {"ok": handle.get("health") == "ready", "health": handle.get("health"),
            "artifact_id": artifact_id}


async def mcp_kill(artifact_id: int) -> dict[str, _Any]:
    """Terminate an MCP server."""
    from yalayut.mcp_manager import get_manager

    await get_manager().shutdown(artifact_id)
    return {"ok": True, "artifact_id": artifact_id}
