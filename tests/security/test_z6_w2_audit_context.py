"""Z6 W2 — orchestrator dispatch enters audit_context per task.

Without this wrap, credential_access_log rows accumulate with NULL
mission_id/task_id/agent — making it impossible to trace which step
read which secret. Mirrors the existing vendor_call.py audit_context
pattern but at the dispatch entry.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture
async def db_env(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    from src.app import config
    from src.infra import db as db_mod

    monkeypatch.setattr(config, "DB_PATH", db_path, raising=False)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)

    monkeypatch.setenv("KUTAY_MASTER_KEY", "z6-w2-test-key-xxxxxxxxxxxx")

    import src.security.credential_store as cs_mod
    from src.security import credential_schemas as schemas

    schemas.reset_cache()
    cs_mod._fernet = None
    cs_mod._MASTER_KEY = None

    await db_mod.init_db()
    yield db_mod
    await db_mod.close_db()
    cs_mod._fernet = None
    cs_mod._MASTER_KEY = None
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(db_path + suffix)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_audit_context_inside_dispatch_wrap_stamps_log(db_env):
    """When dispatch wraps execution in audit_context, log_access picks
    up the mission_id/task_id/agent from the ambient ContextVar — even
    when called with no kwargs."""
    from src.security._audit_context import audit_context
    from src.security.credential_audit import log_access, recent_events

    async with audit_context(mission_id=42, task_id=101, agent="executor"):
        await log_access("vercel", "read", success=True)

    rows = await recent_events(service_name="vercel", limit=1)
    assert rows, "expected at least one audit row"
    row = rows[0]
    assert row["mission_id"] == 42
    assert row["task_id"] == 101
    assert row["agent"] == "executor"


@pytest.mark.asyncio
async def test_audit_context_clears_after_block(db_env):
    """Outside the async-with, fresh log_access calls have NULL provenance —
    proving the ContextVar resets cleanly (no leakage between tasks)."""
    from src.security._audit_context import audit_context
    from src.security.credential_audit import log_access, recent_events

    async with audit_context(mission_id=7, task_id=7, agent="coder"):
        pass  # enter + exit only
    await log_access("github", "read", success=True)

    rows = await recent_events(service_name="github", limit=1)
    assert rows
    assert rows[0]["mission_id"] is None
    assert rows[0]["task_id"] is None
    assert rows[0]["agent"] is None


def test_orchestrator_dispatch_wraps_run_in_audit_context():
    """Static check: the dispatch entry references audit_context. Defends
    against a refactor accidentally dropping the wrap."""
    import inspect
    from src.core import orchestrator as orch_mod

    src = inspect.getsource(orch_mod.Orchestrator._dispatch)
    assert "audit_context" in src, (
        "orchestrator._dispatch must wrap the run coroutine in "
        "audit_context so credential_access_log rows are stamped"
    )
    # And the wrap must reference task_id + mission_id + agent.
    assert "mission_id=task.get" in src or "mission_id=task[" in src
    assert "task_id=task_id" in src
    assert "agent=agent_type" in src


@pytest.mark.asyncio
async def test_dispatch_audit_wrap_propagates_to_log(db_env, monkeypatch):
    """End-to-end-ish: replicate the wrap shape from orchestrator._dispatch
    and verify a credential_audit row written from inside it has the
    right stamps."""
    from src.security._audit_context import audit_context
    from src.security.credential_audit import log_access, recent_events

    task = {"id": 999, "mission_id": 333, "agent_type": "researcher"}

    # Mirror orchestrator._dispatch's wrap shape exactly.
    async with audit_context(
        mission_id=task.get("mission_id"),
        task_id=task["id"],
        agent=task["agent_type"],
    ):
        await log_access("sentry", "read", success=True)

    rows = await recent_events(service_name="sentry", limit=1)
    assert rows
    assert rows[0]["mission_id"] == 333
    assert rows[0]["task_id"] == 999
    assert rows[0]["agent"] == "researcher"
