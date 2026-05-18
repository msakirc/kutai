"""Z6 T7D — vendor_call writes an audit row with correct mission/task context.

The vendor_call tool wraps adapter.execute() in an audit_context so any
credential read inside the adapter (and the explicit log_access call we
emit when entering the context) lands in credential_access_log with the
caller's mission_id / task_id / agent attached.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_t7d.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_vendor_call_writes_audit_row(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)

    # Build a fake adapter that returns ok and a fake registry that
    # serves it for service='vercel'.
    fake_adapter = MagicMock()
    fake_adapter.execute = AsyncMock(return_value={"status": "ok", "id": 1})
    fake_registry = MagicMock()
    fake_registry.get = MagicMock(return_value=fake_adapter)
    fake_registry.list_services = MagicMock(return_value=["vercel"])

    import src.integrations.registry as reg_mod
    monkeypatch.setattr(
        reg_mod, "get_integration_registry",
        lambda: fake_registry, raising=True,
    )

    from src.tools.vendor_call import vendor_call_tool
    raw = await vendor_call_tool(
        service="vercel",
        action="list_projects",
        params={},
        mission_id=77,
        task_id=8888,
        agent="executor",
    )
    parsed = json.loads(raw)
    assert parsed.get("status") == "ok"
    fake_adapter.execute.assert_awaited_once_with("list_projects", {})

    # The audit log should now have a row with our context attached.
    from src.security.credential_audit import recent_events
    rows = await recent_events(service_name="vercel", limit=5)
    assert len(rows) >= 1
    row = rows[0]
    assert row["service_name"] == "vercel"
    assert row["mission_id"] == 77
    assert row["task_id"] == 8888
    assert row["agent"] == "executor"
    assert row["success"] == 1


@pytest.mark.asyncio
async def test_refused_call_does_not_audit(tmp_path, monkeypatch):
    """Allowlist refusals never touch the adapter, so no audit row is
    written either."""
    await _setup_db(tmp_path, monkeypatch)
    from src.tools.vendor_call import vendor_call_tool
    raw = await vendor_call_tool(
        service="stripe",
        action="charge",
        params={},
        agent="researcher",   # not on stripe allowlist
        mission_id=1,
        task_id=2,
    )
    parsed = json.loads(raw)
    assert parsed.get("status") == "refused"

    from src.security.credential_audit import recent_events
    rows = await recent_events(service_name="stripe", limit=5)
    assert rows == []
