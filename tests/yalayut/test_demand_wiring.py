"""Tests for demand convenience API and shared threshold constant (Task 1).

Uses init_db() + explicit table clear, matching the pattern used by the
yalayut demand-signal test suite (asyncio_mode=auto from pytest.ini).
"""
import pytest
from unittest.mock import patch

from src.infra.db import init_db, get_db
from yalayut.discovery import demand as _demand
from src.infra.db import get_db as _get_db_for_test


def _async_return(value):
    async def _fn(*args, **kwargs):
        return value
    return _fn


@pytest.fixture
async def db():
    """Initialise the DB and wipe demand_signals for test isolation."""
    await init_db()
    db = await get_db()
    await db.execute("DELETE FROM yalayut_demand_signals")
    await db.commit()


@pytest.mark.asyncio
async def test_record_helper_inserts_row(db):
    row_id = await _demand.record(
        source_step_pattern="test:helper-pattern",
        intent_keywords=["pdf", "extract"],
        signal_type="tool_call",
        confidence=0.4,
    )
    assert row_id > 0
    stacked = await _demand.stack_confidence("test:helper-pattern")
    assert stacked == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_threshold_constant_is_half():
    assert _demand.DEMAND_DISCOVERY_THRESHOLD == 0.5


@pytest.mark.asyncio
async def test_flash_empty_query_fires_step_entry_miss(db):
    import importlib
    _flash = importlib.import_module("intersect.flash")

    task = {"id": 901, "title": "Parse the invoice CSV", "description": "",
            "context": {}}
    with patch("yalayut.query", new=_async_return([])):
        await _flash.flash(task)

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT signal_type FROM yalayut_demand_signals "
        "WHERE source_step_pattern LIKE ?", ("%Parse the invoice CSV%",))
    types = {r[0] for r in await cur.fetchall()}
    assert types == {"step_entry_miss"}


@pytest.mark.asyncio
async def test_flash_empty_query_with_recipe_hint_fires_planning_miss(db):
    import importlib
    _flash = importlib.import_module("intersect.flash")

    task = {"id": 902, "title": "Send a Slack notification",
            "description": "", "context": {"recipe_hint": "slack"}}
    with patch("yalayut.query", new=_async_return([])):
        await _flash.flash(task)

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT signal_type FROM yalayut_demand_signals "
        "WHERE source_step_pattern LIKE ?", ("%Send a Slack notification%",))
    types = {r[0] for r in await cur.fetchall()}
    assert types == {"planning_miss"}


@pytest.mark.asyncio
async def test_react_unresolved_tool_fires_tool_call_signal(db):
    from coulson import react as _react

    await _react._fire_tool_call_signal("scrape_pdf_table", task_id=903)

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT signal_type, intent_keywords_json FROM yalayut_demand_signals "
        "WHERE source_step_pattern = ?", ("tool_call:scrape_pdf_table",))
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "tool_call"
    assert "scrape_pdf_table" in rows[0][1]


@pytest.mark.asyncio
async def test_capture_repeat_fires_hint_miss(db):
    from yalayut.capture import capture_hint

    task = {"title": "Retry flaky HTTP with backoff",
            "description": "wrap requests in exponential backoff"}
    outcome = {"status": "completed", "iterations": 3}

    # Ensure yalayut_index has no pre-existing row for this slug so the first
    # capture always takes the INSERT path (not the UPDATE path).
    dbc = await _get_db_for_test()
    await dbc.execute(
        "DELETE FROM yalayut_index WHERE source = 'internal' AND name = 'internal-retry-flaky-http-with-backoff'")
    await dbc.commit()

    # First capture — inserts, no hint_miss.
    await capture_hint(task, outcome)
    dbc = await _get_db_for_test()
    cur = await dbc.execute("SELECT COUNT(*) FROM yalayut_demand_signals "
                            "WHERE signal_type = 'hint_miss'")
    assert (await cur.fetchone())[0] == 0

    # Second capture of the same task — upsert path -> hint_miss fires.
    await capture_hint(task, outcome)
    cur = await dbc.execute("SELECT source_step_pattern FROM yalayut_demand_signals "
                            "WHERE signal_type = 'hint_miss'")
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0].startswith("hint_miss:internal-")
