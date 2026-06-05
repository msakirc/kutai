"""yalayut.admin — auth + MCP control ops.

Rewritten against the current (Phase-4) admin API. The earlier revision of
this file asserted a Phase-3 design (``admin._db_query`` /
``admin._revet_artifacts_for_env`` / ``yalayut.secrets`` delegation /
manager-based ``mcp_status``) that the Phase-4 rewrite (commit 693a1d20)
deliberately replaced. set_secret now encrypts inline + flips ``env_status``;
missing_auth / mcp_status read the DB directly; mcp_kill / mcp_restart drive the
live :class:`yalayut.mcp_manager.McpManager`.
"""
import pytest

from yalayut import admin


# ─── set_secret ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_set_secret_encrypts_and_flips_env_status(monkeypatch):
    import os

    from src.infra.db import init_db, get_db

    # This test dir has no conftest; ensure a fernet key like tests/yalayut does.
    if not os.getenv("YALAYUT_SECRET_KEY"):
        from cryptography.fernet import Fernet
        monkeypatch.setenv("YALAYUT_SECRET_KEY", Fernet.generate_key().decode())

    await init_db()
    db = await get_db()
    # An artifact blocked solely on this env var.
    await db.execute(
        "INSERT INTO yalayut_index "
        "(artifact_type, kind, source, owner, name, version, vet_tier, "
        " exposure_class, enabled, env_status, created_at) "
        "VALUES ('api', 'rest', 'x', 'o', '_t_blocked_api', '1.0.0', 1, "
        " 'tool', 1, 'missing_OPENAQ_API_KEY', datetime('now'))")
    await db.commit()
    try:
        res = await admin.set_secret("OPENAQ_API_KEY", "the-plain-value")
        assert res["ok"] is True

        cur = await db.execute(
            "SELECT encrypted_value FROM yalayut_secrets "
            "WHERE key_name = 'OPENAQ_API_KEY'")
        row = await cur.fetchone()
        await cur.close()
        assert row is not None
        # Stored encrypted — never the plaintext.
        assert b"the-plain-value" not in (row[0] or b"")

        # Artifacts blocked on this exact key are flipped to ready.
        cur = await db.execute(
            "SELECT env_status FROM yalayut_index WHERE name = '_t_blocked_api'")
        assert (await cur.fetchone())[0] == "ready"
        await cur.close()
    finally:
        await db.execute("DELETE FROM yalayut_index WHERE name = '_t_blocked_api'")
        await db.execute(
            "DELETE FROM yalayut_secrets WHERE key_name = 'OPENAQ_API_KEY'")
        await db.commit()


# ─── missing_auth ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_auth_lists_blocked():
    from src.infra.db import init_db, get_db

    await init_db()
    db = await get_db()
    await db.execute(
        "INSERT INTO yalayut_index "
        "(artifact_type, kind, source, owner, name, version, vet_tier, "
        " exposure_class, enabled, env_status, created_at) "
        "VALUES ('api', 'rest', 'x', 'o', '_t_vt_api', '1.0.0', 1, "
        " 'tool', 1, 'missing_VIRUSTOTAL_API_KEY', datetime('now'))")
    await db.commit()
    try:
        rows = await admin.missing_auth()
        match = [r for r in rows if r["name"] == "_t_vt_api"]
        assert len(match) == 1
        assert match[0]["env_status"] == "missing_VIRUSTOTAL_API_KEY"
    finally:
        await db.execute("DELETE FROM yalayut_index WHERE name = '_t_vt_api'")
        await db.commit()


# ─── mcp_status ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_status_reports_processes():
    from src.infra.db import init_db, get_db

    await init_db()
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO yalayut_index "
        "(artifact_type, kind, source, owner, name, version, vet_tier, "
        " exposure_class, enabled, created_at) "
        "VALUES ('mcp', 'tool', 'x', 'o', '_t_mcp_srv', '1.0.0', 1, "
        " 'tool', 1, datetime('now'))")
    await db.commit()
    aid = cur.lastrowid
    await db.execute(
        "INSERT INTO yalayut_mcp_processes "
        "(artifact_id, pid, port, started_at, last_used_at, idle_timeout_s, "
        " health, last_probe_at, consecutive_probe_fails) "
        "VALUES (?, 4321, 8123, datetime('now'), datetime('now'), 300, "
        " 'ready', datetime('now'), 0)", (aid,))
    await db.commit()
    try:
        rows = await admin.mcp_status()
        match = [r for r in rows if r["artifact_id"] == aid]
        assert len(match) == 1
        # The /yalayut mcp status handler reads name / health / pid.
        assert match[0]["name"] == "_t_mcp_srv"
        assert match[0]["health"] == "ready"
        assert match[0]["pid"] == 4321
    finally:
        await db.execute(
            "DELETE FROM yalayut_mcp_processes WHERE artifact_id = ?", (aid,))
        await db.execute("DELETE FROM yalayut_index WHERE id = ?", (aid,))
        await db.commit()


# ─── mcp kill / restart ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_kill_drives_manager_shutdown(monkeypatch):
    from yalayut.mcp_manager import get_manager

    killed = []

    async def fake_shutdown(aid):
        killed.append(aid)

    monkeypatch.setattr(get_manager(), "shutdown", fake_shutdown)
    res = await admin.mcp_kill(9)
    assert res["ok"] is True
    assert killed == [9]


@pytest.mark.asyncio
async def test_mcp_restart_drives_manager_shutdown(monkeypatch):
    """restart == shutdown + lazy re-spawn on next use (no_auto_connect)."""
    from yalayut.mcp_manager import get_manager

    stopped = []

    async def fake_shutdown(aid):
        stopped.append(aid)

    monkeypatch.setattr(get_manager(), "shutdown", fake_shutdown)
    res = await admin.mcp_restart(9)
    assert res["ok"] is True
    assert stopped == [9]
