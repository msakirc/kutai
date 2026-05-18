"""Tests for Z7 T1D (B5) — src/app/attention_budget.py.

Covers:
- Soft-warn: ALL cards surface regardless of budget.
- Over-budget p1/p2/p3 are flagged below_fold=True; p0 never below_fold.
- get_queue() partitions cards into today/this_week/deferred/when_idle.
- record_deferred() writes log row + updates founder_actions.defer_until.
- check_budget() returns remaining, over_budget, top_queue.
"""
from __future__ import annotations

import datetime
import pytest


# ── DB fixture ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
async def _db_reset():
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None


async def _setup_db(tmp_path, monkeypatch):
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db, get_db
    await init_db()
    return await get_db()


async def _create_mission(db, mission_id: int, budget: int | None = None) -> None:
    if budget is None:
        await db.execute(
            "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
            (mission_id, f"m{mission_id}", "active"),
        )
    else:
        await db.execute(
            "INSERT INTO missions (id, title, status, "
            "founder_attention_budget_minutes) VALUES (?, ?, ?, ?)",
            (mission_id, f"m{mission_id}", "active", budget),
        )
    await db.commit()


async def _create_card(
    db,
    card_id: int,
    mission_id: int,
    priority: str = "p2_this_week",
    defer_until: str | None = None,
) -> None:
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO founder_actions "
        "(id, mission_id, kind, title, why, instructions_json, "
        " status, priority, defer_until, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            card_id, mission_id, "manual", f"Card {card_id}", "because",
            "[]", "pending", priority, defer_until, now, now,
        ),
    )
    await db.commit()


# ── B5 soft-warn tests ────────────────────────────────────────────────────────

class TestShouldSurfaceNow:
    """All cards always surface; only below_fold varies."""

    def test_p0_always_above_fold(self):
        from src.app.attention_budget import should_surface_now
        card = {"priority": "p0_blocking", "id": 1, "title": "blocker"}
        result = should_surface_now(card)
        assert result["surface"] is True
        assert result["below_fold"] is False

    def test_p1_above_fold_when_in_budget(self):
        from src.app.attention_budget import should_surface_now_batch
        cards = [{"priority": "p1_today", "id": 2, "title": "p1"}]
        results = should_surface_now_batch(cards, spent=10, cap=60)
        assert results[0]["surface"] is True
        assert results[0]["below_fold"] is False

    def test_p1_below_fold_when_over_budget(self):
        from src.app.attention_budget import should_surface_now_batch
        cards = [{"priority": "p1_today", "id": 2, "title": "p1"}]
        results = should_surface_now_batch(cards, spent=70, cap=60)
        assert results[0]["surface"] is True, "p1 must still surface"
        assert results[0]["below_fold"] is True, "p1 must be below_fold when over budget"

    def test_p0_never_below_fold_even_over_budget(self):
        from src.app.attention_budget import should_surface_now_batch
        cards = [
            {"priority": "p0_blocking", "id": 1, "title": "p0"},
            {"priority": "p1_today",    "id": 2, "title": "p1"},
            {"priority": "p2_this_week", "id": 3, "title": "p2"},
            {"priority": "p3_when_idle", "id": 4, "title": "p3"},
        ]
        results = should_surface_now_batch(cards, spent=200, cap=60)
        assert results[0]["below_fold"] is False, "p0_blocking never below_fold"
        assert results[1]["below_fold"] is True
        assert results[2]["below_fold"] is True
        assert results[3]["below_fold"] is True
        # All still surface
        assert all(r["surface"] is True for r in results)

    def test_no_cards_over_budget(self):
        from src.app.attention_budget import should_surface_now_batch
        results = should_surface_now_batch([], spent=200, cap=60)
        assert results == []

    def test_missing_priority_defaults_to_p2(self):
        from src.app.attention_budget import should_surface_now_batch
        cards = [{"id": 5, "title": "no priority"}]  # no 'priority' key
        results = should_surface_now_batch(cards, spent=10, cap=60)
        assert results[0]["surface"] is True
        assert results[0]["below_fold"] is False


# ── get_queue() partitioning tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_queue_partitions_correctly(tmp_path, monkeypatch):
    """Cards are correctly partitioned into today/this_week/deferred/when_idle."""
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 10)

    future = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    await _create_card(db, 1, 10, priority="p0_blocking")
    await _create_card(db, 2, 10, priority="p1_today")
    await _create_card(db, 3, 10, priority="p2_this_week")
    await _create_card(db, 4, 10, priority="p3_when_idle")
    await _create_card(db, 5, 10, priority="p2_this_week", defer_until=future)

    from src.app.attention_budget import get_queue
    q = await get_queue(product_id=10)

    today_ids = [c["id"] for c in q["today"]]
    this_week_ids = [c["id"] for c in q["this_week"]]
    deferred_ids = [c["id"] for c in q["deferred"]]
    when_idle_ids = [c["id"] for c in q["when_idle"]]

    assert 1 in today_ids
    assert 2 in today_ids
    assert 3 in this_week_ids
    assert 4 in when_idle_ids
    assert 5 in deferred_ids, "card with future defer_until must be in deferred"

    # p0 must be first in today list
    assert q["today"][0]["id"] == 1


@pytest.mark.asyncio
async def test_get_queue_returns_required_shape(tmp_path, monkeypatch):
    """get_queue() returns all required top-level keys."""
    await _setup_db(tmp_path, monkeypatch)
    from src.app.attention_budget import get_queue
    q = await get_queue(product_id=None)
    required = {"cap", "spent", "remaining", "over_budget", "today", "this_week", "deferred", "when_idle"}
    assert required.issubset(q.keys())
    # Numeric types
    assert isinstance(q["cap"], int)
    assert isinstance(q["spent"], int)
    assert isinstance(q["remaining"], int)
    assert isinstance(q["over_budget"], bool)


@pytest.mark.asyncio
async def test_get_queue_card_has_below_fold_field(tmp_path, monkeypatch):
    """Each card in get_queue() has a 'below_fold' field."""
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 11)
    await _create_card(db, 10, 11, priority="p1_today")

    from src.app.attention_budget import get_queue
    q = await get_queue(product_id=11)
    for section in ["today", "this_week", "deferred", "when_idle"]:
        for card in q.get(section, []):
            assert "below_fold" in card, f"card in {section} missing below_fold"


# ── Defer flow tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_deferred_writes_log_row(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 20)
    await _create_card(db, 100, 20)

    future = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    from src.app.attention_budget import record_deferred
    await record_deferred(card_id=100, product_id=20, deferred_to=future)

    cur = await db.execute(
        "SELECT action, card_id, deferred_to FROM founder_attention_log "
        "WHERE card_id = ?",
        (100,),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] == "deferred"
    assert row[1] == 100
    assert row[2] == future


@pytest.mark.asyncio
async def test_record_deferred_updates_founder_actions(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 21)
    await _create_card(db, 101, 21)

    future = (datetime.datetime.utcnow() + datetime.timedelta(hours=18)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    from src.app.attention_budget import record_deferred
    await record_deferred(card_id=101, product_id=21, deferred_to=future)

    cur = await db.execute(
        "SELECT defer_until FROM founder_actions WHERE id = ?", (101,)
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] == future


@pytest.mark.asyncio
async def test_deferred_card_moves_to_deferred_bucket(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 22)
    await _create_card(db, 102, 22, priority="p1_today")

    future = (datetime.datetime.utcnow() + datetime.timedelta(hours=12)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    from src.app.attention_budget import record_deferred, get_queue
    await record_deferred(card_id=102, product_id=22, deferred_to=future)
    q = await get_queue(product_id=22)

    deferred_ids = [c["id"] for c in q["deferred"]]
    today_ids = [c["id"] for c in q["today"]]
    assert 102 in deferred_ids
    assert 102 not in today_ids


# ── check_budget() tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_budget_over_budget_flag(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    await _create_mission(db, 30)
    await _create_card(db, 200, 30)

    # Inject spent minutes via attention log
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO founder_attention_log "
        "(mission_id, step_id, action, minutes_debited, ts, attention_minutes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (30, "", "acted", 0, now, 80),
    )
    await db.commit()

    import os
    monkeypatch.setenv("FOUNDER_ATTENTION_DAILY_MINUTES", "60")
    from src.app import attention_budget as _ab
    import importlib
    importlib.reload(_ab)

    budget_info = await _ab.check_budget(product_id=30)
    assert budget_info["over_budget"] is True
    assert budget_info["remaining"] < 0


@pytest.mark.asyncio
async def test_check_budget_in_budget(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    import os
    monkeypatch.setenv("FOUNDER_ATTENTION_DAILY_MINUTES", "120")
    from src.app import attention_budget as _ab
    import importlib
    importlib.reload(_ab)

    budget_info = await _ab.check_budget(product_id=None)
    assert budget_info["over_budget"] is False
    assert budget_info["remaining"] >= 0


# ── next_review_window tests ──────────────────────────────────────────────────

def test_next_review_window_is_tomorrow_morning():
    from src.app.attention_budget import next_review_window
    nrw = next_review_window({})
    now = datetime.datetime.utcnow()
    assert nrw.hour == 9
    assert nrw.minute == 0
    # Either tomorrow or same day (edge-case near midnight)
    delta = nrw.date() - now.date()
    assert delta.days in (0, 1)
