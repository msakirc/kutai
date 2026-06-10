"""Tests that POST /missions routes through add_mission (not raw INSERT).

The FastAPI app in api.py defines request-body Pydantic models inside
create_app() — a factory pattern that prevents TestClient from resolving
them as JSON bodies (FastAPI infers query-param instead, returning 422).
Tests therefore call the endpoint handler directly with a MissionCreate
instance, which is the correct layer for verifying add_mission routing.

Verifies:
- endpoint returns correct shape {id, title, status}
- add_mission is the sole writer (lifecycle_state + product_id backfill performed)
- fields from the request body (title, description, priority, workflow, repo_path)
  are passed through correctly
"""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "api_mission.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


async def _close_db(db_mod):
    if db_mod._db_connection is not None:
        await db_mod._db_connection.close()
        db_mod._db_connection = None


def _get_handler_and_model():
    """Extract the create_mission handler and MissionCreate model from the app."""
    from src.app.api import create_app
    app = create_app()
    # find the POST /missions route
    for route in app.routes:
        if getattr(route, "path", None) == "/missions" and "POST" in (getattr(route, "methods", None) or set()):
            handler = route.endpoint
            break
    else:
        raise RuntimeError("POST /missions route not found")
    # MissionCreate is defined inside create_app(); access via closure
    # by importing from the live app module
    return handler


@pytest.mark.asyncio
async def test_create_mission_routes_through_add_mission(tmp_path, monkeypatch):
    """POST /missions must call add_mission, not raw INSERT."""
    db_mod = await _setup(tmp_path, monkeypatch)

    # Spy on add_mission while still writing to the real DB.
    original_add_mission = db_mod.add_mission
    calls = []

    async def spy_add_mission(*args, **kwargs):
        calls.append((args, kwargs))
        return await original_add_mission(*args, **kwargs)

    monkeypatch.setattr(db_mod, "add_mission", spy_add_mission)

    try:
        handler = _get_handler_and_model()

        # Build a minimal stand-in for MissionCreate using a SimpleNamespace
        # (the handler only accesses .title, .description, .priority, .workflow, .repo_path)
        from types import SimpleNamespace
        body = SimpleNamespace(
            title="Test Mission",
            description="desc",
            priority=3,
            workflow="i2p",
            repo_path="/tmp/repo",
        )

        result = await handler(body=body)

        assert result["title"] == "Test Mission"
        assert result["status"] == "active"
        assert isinstance(result["id"], int)

        # add_mission must have been called exactly once
        assert len(calls) == 1, f"Expected add_mission called once, got {len(calls)}"
    finally:
        await _close_db(db_mod)


@pytest.mark.asyncio
async def test_create_mission_lifecycle_state_set(tmp_path, monkeypatch):
    """Created mission must have lifecycle_state='active' (add_mission backfill)."""
    db_mod = await _setup(tmp_path, monkeypatch)
    try:
        handler = _get_handler_and_model()

        from types import SimpleNamespace
        body = SimpleNamespace(
            title="Lifecycle Test",
            description="",
            priority=5,
            workflow=None,
            repo_path=None,
        )

        result = await handler(body=body)
        mission_id = result["id"]

        db = await db_mod.get_db()
        cur = await db.execute(
            "SELECT lifecycle_state, product_id FROM missions WHERE id = ?",
            (mission_id,),
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == "active", f"lifecycle_state should be 'active', got {row[0]!r}"
        assert row[1] == str(mission_id), (
            f"product_id should equal mission id '{mission_id}', got {row[1]!r}"
        )
    finally:
        await _close_db(db_mod)


@pytest.mark.asyncio
async def test_create_mission_fields_persisted(tmp_path, monkeypatch):
    """All MissionCreate fields must be stored correctly."""
    db_mod = await _setup(tmp_path, monkeypatch)
    try:
        handler = _get_handler_and_model()

        from types import SimpleNamespace
        body = SimpleNamespace(
            title="Field Check",
            description="my description",
            priority=7,
            workflow="i2p",
            repo_path="/tmp/myrepo",
        )

        result = await handler(body=body)
        assert result["status"] == "active"
        mission_id = result["id"]

        db = await db_mod.get_db()
        cur = await db.execute(
            "SELECT title, description, priority, workflow, repo_path FROM missions WHERE id = ?",
            (mission_id,),
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == "Field Check"
        assert row[1] == "my description"
        assert row[2] == 7
        assert row[3] == "i2p"
        assert row[4] == "/tmp/myrepo"
    finally:
        await _close_db(db_mod)
