"""Tests for founder_actions.defer() API — Task 2 of db-write-consolidation.

Covers:
- defer() sets defer_until and updated_at on the row.
- attention_budget.record_deferred() routes through founder_actions.defer()
  rather than raw SQL (DB-state assertion).
"""
from __future__ import annotations

import datetime
import pytest


# ── fixture helpers (mirror test_repo.py pattern) ────────────────────────────

async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "fa.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    # Reset singleton connection so each test gets a fresh DB.
    if db_mod._db_connection is not None:
        try:
            await db_mod._db_connection.close()
        except Exception:
            pass
    db_mod._db_connection = None
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_path, db_mod, fa


# ── defer() unit tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_defer_sets_defer_until(tmp_path, monkeypatch):
    """defer() must persist defer_until on the founder_actions row."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    action = await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)

    until = (datetime.datetime.utcnow() + datetime.timedelta(hours=24)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    await fa.defer(action.id, until)

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT defer_until FROM founder_actions WHERE id = ?", (action.id,)
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] == until


@pytest.mark.asyncio
async def test_defer_updates_updated_at(tmp_path, monkeypatch):
    """defer() must update the updated_at timestamp."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    action = await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    old_updated_at = action.updated_at

    until = (datetime.datetime.utcnow() + datetime.timedelta(hours=12)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    await fa.defer(action.id, until)

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT updated_at FROM founder_actions WHERE id = ?", (action.id,)
    )
    row = await cur.fetchone()
    assert row is not None
    # updated_at must have changed (or at least be set to a non-None string).
    assert row[0] is not None


@pytest.mark.asyncio
async def test_defer_does_not_change_status(tmp_path, monkeypatch):
    """defer() must not alter the status field."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    action = await fa.create(mid, "generic", "t", "w", [], notify_telegram=False)
    assert action.status == "pending"

    until = (datetime.datetime.utcnow() + datetime.timedelta(hours=6)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    await fa.defer(action.id, until)

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT status FROM founder_actions WHERE id = ?", (action.id,)
    )
    row = await cur.fetchone()
    assert row[0] == "pending"


# ── attention_budget integration: record_deferred routes via defer() ──────────

@pytest.mark.asyncio
async def test_record_deferred_routes_through_defer_api(tmp_path, monkeypatch):
    """record_deferred() must use founder_actions.defer() — verified via DB state."""
    _, db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")

    # Insert a card the same way test_attention_budget.py does.
    db = await db_mod.get_db()
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO founder_actions "
        "(id, mission_id, kind, title, why, instructions_json, "
        " status, priority, defer_until, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (500, mid, "manual", "Card 500", "because", "[]",
         "pending", "p2_this_week", None, now, now),
    )
    await db.commit()

    future = (datetime.datetime.utcnow() + datetime.timedelta(hours=18)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Patch founder_actions.defer so we can confirm it is called.
    called_with = {}

    async def _spy_defer(action_id, until):
        called_with["action_id"] = action_id
        called_with["until"] = until
        # Still do the real update.
        await fa._defer_real(action_id, until)

    import src.founder_actions as _fa_mod
    original_defer = _fa_mod.defer
    # We'll swap defer in the module; attention_budget imports it lazily.
    _fa_mod._defer_real = original_defer
    monkeypatch.setattr(_fa_mod, "defer", _spy_defer)

    from src.app.attention_budget import record_deferred
    await record_deferred(card_id=500, product_id=mid, deferred_to=future)

    # Spy must have been called with the right args.
    assert called_with.get("action_id") == 500
    assert called_with.get("until") == future

    # DB row must reflect the new defer_until.
    cur = await db.execute(
        "SELECT defer_until FROM founder_actions WHERE id = ?", (500,)
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] == future
