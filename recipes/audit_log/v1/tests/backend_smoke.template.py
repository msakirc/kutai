"""Audit-log recipe — backend smoke tests.

Runs POST-instantiation against the instantiated recipe in the mission
workspace. Recipe sources are scaffolds (.template suffix).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest


# ---------------------------------------------------------------------------
# In-process FastAPI + aiosqlite fixture helpers
# ---------------------------------------------------------------------------

def _make_app():
    """Build a minimal in-process FastAPI app wired to the audit_log router."""
    import importlib, sys
    from fastapi import FastAPI

    # Import the instantiated backend (post-instantiation; .template suffix dropped)
    # For smoke test purposes we import the template module directly.
    import backend_template as audit  # T6: adjust import to instantiated module name
    app = FastAPI()
    app.include_router(audit.router)
    return app, audit


async def _make_db():
    """Create an in-memory aiosqlite DB and apply the audit_log schema."""
    import aiosqlite
    db = await aiosqlite.connect(":memory:")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_user_id INTEGER,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.commit()
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_record_event_returns_id():
    """record_event returns a non-zero integer id."""
    import backend_template as audit  # T6: adjust import
    db = await _make_db()
    row_id = await audit.record_event(
        actor_user_id=1,
        action="user.updated",
        resource_type="user",
        resource_id="42",
        payload={"field": "email"},
        db=db,
    )
    assert isinstance(row_id, int)
    assert row_id > 0
    await db.close()


async def test_list_events_for_resource_paginated():
    """list_events_for_resource returns paginated results with next_cursor."""
    import backend_template as audit  # T6: adjust import
    db = await _make_db()

    # Insert 15 events for the same resource
    for i in range(15):
        await audit.record_event(
            actor_user_id=None,
            action=f"action_{i}",
            resource_type="order",
            resource_id="99",
            payload={},
            db=db,
        )

    page1, next_cursor1 = await audit.list_events_for_resource(
        resource_type="order", resource_id="99", limit=10, db=db,
    )
    assert len(page1) == 10
    assert next_cursor1 is not None

    page2, next_cursor2 = await audit.list_events_for_resource(
        resource_type="order", resource_id="99",
        cursor=next_cursor1, limit=10, db=db,
    )
    assert len(page2) == 5
    assert next_cursor2 is None
    await db.close()


async def test_retention_sweep_deletes_old():
    """sweep_retention deletes rows older than RETENTION_DAYS."""
    import backend_template as audit  # T6: adjust import
    db = await _make_db()

    # Insert one old row manually with a past timestamp
    cutoff = datetime.now(timezone.utc) - timedelta(days=audit.RETENTION_DAYS + 1)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO audit_events (actor_user_id, action, resource_type, resource_id, payload, created_at) "
        "VALUES (NULL, 'old.action', 'test', 'x', '{}', ?)",
        (cutoff_str,),
    )
    await db.commit()

    # Insert one recent row
    await audit.record_event(
        actor_user_id=None, action="recent.action",
        resource_type="test", resource_id="x", payload={}, db=db,
    )

    deleted = await audit.sweep_retention(db=db)
    assert deleted == 1

    cur = await db.execute("SELECT COUNT(*) FROM audit_events")
    (count,) = await cur.fetchone()
    assert count == 1  # only recent row remains
    await db.close()


async def test_list_for_unknown_resource_returns_empty():
    """list_events_for_resource for unknown resource returns empty list, no cursor."""
    import backend_template as audit  # T6: adjust import
    db = await _make_db()
    events, next_cursor = await audit.list_events_for_resource(
        resource_type="ghost", resource_id="0", db=db,
    )
    assert events == []
    assert next_cursor is None
    await db.close()
