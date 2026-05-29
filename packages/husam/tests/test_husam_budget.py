"""husam.run passes remaining_budget_usd to fatih_hoca.select() and pauses the
mission on SelectionFailure(reason='budget').

Migrated from tests/core/test_dispatcher_budget.py (SP3b Task 2). The budget
helper (_remaining_budget) and the select call moved from the dispatcher into
husam.run; mission_id now rides on the husam task spec (top-level).
"""
from __future__ import annotations

import aiosqlite
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ─── helpers ────────────────────────────────────────────────────────────────


def _reset_db(db_module, db_path: str) -> None:
    db_module.DB_PATH = db_path
    db_module._db_connection = None


def _fake_pick(model_name: str = "local-test"):
    m = MagicMock()
    m.model_name = model_name
    m.estimated_cost_usd = 0.0
    m.model = MagicMock(
        is_local=True,
        name=model_name,
        thinking_model=False,
        has_vision=False,
        provider="local",
        location="",
    )
    m.min_time_seconds = 1.0
    m.estimated_load_seconds = 0.0
    m.score = 0.5
    m.top_summary = ""
    return m


def _task(mission_id):
    return {
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "main_work",
            "task": "coder",
            "agent_type": "coder",
            "difficulty": 5,
            "messages": [],
            "tools": None,
            "failures": [],
        }},
        "kind": "main_work",
        "preselected_pick": None,
        "mission_id": mission_id,
    }


# ─── test 1: budget passed correctly ────────────────────────────────────────


@pytest.mark.asyncio
async def test_husam_passes_remaining_budget(tmp_path, monkeypatch):
    """husam.run passes remaining_budget_usd = ceiling - spent to select()."""
    import husam
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db(db_module, db_path)

    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd) VALUES ('m', 2.0, 0.5)"
        )
        cur = await db.execute("SELECT id FROM missions")
        mid = (await cur.fetchone())[0]
        await db.commit()

    captured = {}

    def fake_select(**kwargs):
        captured.update(kwargs)
        return _fake_pick()

    _reset_db(db_module, db_path)

    import hallederiz_kadir

    with patch("fatih_hoca.select", fake_select), \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value=MagicMock(spec=hallederiz_kadir.CallResult, content="ok", model="x", model_name="x", cost=0.0, usage={}, tool_calls=[], latency=0.1, thinking="", is_local=True, provider="local", task="t"))), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()), \
         patch("src.core.in_flight.begin_call", new=AsyncMock(return_value="cid")), \
         patch("src.core.in_flight.end_call", new=AsyncMock()), \
         patch("src.models.local_model_manager.get_local_manager") as mock_mgr:
        mgr = MagicMock()
        mgr.is_loaded = True
        mgr.current_model = "local-test"
        mgr.keep_alive = MagicMock()
        mock_mgr.return_value = mgr

        try:
            await husam.run(_task(mid))
        except Exception:
            pass  # downstream may raise; we only care select was called

    assert "remaining_budget_usd" in captured, "remaining_budget_usd not passed to select()"
    assert captured["remaining_budget_usd"] == pytest.approx(1.5)


# ─── test 2: None when no ceiling ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_husam_passes_none_when_no_ceiling(tmp_path, monkeypatch):
    """When mission has no cost_ceiling_usd, remaining_budget_usd=None is passed."""
    import husam
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai2.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db(db_module, db_path)

    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO missions (title) VALUES ('no-ceiling')")
        cur = await db.execute("SELECT id FROM missions")
        mid = (await cur.fetchone())[0]
        await db.commit()

    captured = {}

    def fake_select(**kwargs):
        captured.update(kwargs)
        return _fake_pick()

    _reset_db(db_module, db_path)

    import hallederiz_kadir

    with patch("fatih_hoca.select", fake_select), \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value=MagicMock(spec=hallederiz_kadir.CallResult, content="ok", model="x", model_name="x", cost=0.0, usage={}, tool_calls=[], latency=0.1, thinking="", is_local=True, provider="local", task="t"))), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()), \
         patch("src.core.in_flight.begin_call", new=AsyncMock(return_value="cid")), \
         patch("src.core.in_flight.end_call", new=AsyncMock()), \
         patch("src.models.local_model_manager.get_local_manager") as mock_mgr:
        mgr = MagicMock()
        mgr.is_loaded = True
        mgr.current_model = "local-test"
        mgr.keep_alive = MagicMock()
        mock_mgr.return_value = mgr

        try:
            await husam.run(_task(mid))
        except Exception:
            pass

    assert "remaining_budget_usd" in captured, "remaining_budget_usd key not in select() kwargs"
    assert captured["remaining_budget_usd"] is None


# ─── test 3: budget failure pauses mission ───────────────────────────────────


@pytest.mark.asyncio
async def test_husam_pauses_mission_on_budget_failure(tmp_path, monkeypatch):
    """SelectionFailure(reason='budget') causes emit_pause and raises."""
    import husam
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai3.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db(db_module, db_path)

    from src.infra.db import init_db
    from fatih_hoca import SelectionFailure
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd, lifecycle_state) "
            "VALUES ('m', 0.10, 0.05, 'active')"
        )
        cur = await db.execute("SELECT id FROM missions")
        mid = (await cur.fetchone())[0]
        await db.commit()

    _reset_db(db_module, db_path)

    def fake_select(**kwargs):
        return SelectionFailure(reason="budget", detail="no fit")

    with patch("fatih_hoca.select", fake_select), \
         patch("src.core.in_flight.begin_call", new=AsyncMock(return_value="cid")), \
         patch("src.core.in_flight.end_call", new=AsyncMock()):
        with pytest.raises(Exception, match="budget|selection failed"):
            await husam.run(_task(mid))

    _reset_db(db_module, db_path)
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,)
        )).fetchone()
    assert row[0] == "paused", f"expected 'paused', got {row[0]!r}"
