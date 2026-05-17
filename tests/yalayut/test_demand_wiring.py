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


@pytest.mark.asyncio
async def test_yalayut_demand_executor_records_signal(db):
    from mr_roboto.executors.yalayut_demand import run

    task = {"payload": {
        "action": "yalayut_demand",
        "source_step_pattern": "dlq:task-555",
        "intent_keywords": ["migrate", "schema"],
        "signal_type": "dlq",
        "confidence": 0.3,
    }}
    res = await run(task)
    assert res["ok"] is True

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT signal_type FROM yalayut_demand_signals "
        "WHERE source_step_pattern = ?", ("dlq:task-555",))
    rows = await cur.fetchall()
    assert len(rows) == 1 and rows[0][0] == "dlq"


@pytest.mark.asyncio
async def test_dlq_write_enqueues_yalayut_demand_task(db, monkeypatch):
    from general_beckman import apply as _apply

    enqueued = []

    async def _fake_add_task(**kwargs):
        enqueued.append(kwargs)
        return 1

    async def _fake_update_task(*a, **k):
        return None

    async def _fake_quarantine(**k):
        return None

    monkeypatch.setattr("src.infra.db.add_task", _fake_add_task)
    monkeypatch.setattr("src.infra.db.update_task", _fake_update_task)
    monkeypatch.setattr("src.infra.dead_letter.quarantine_task", _fake_quarantine)

    task = {"id": 777, "title": "Convert HEIC images to PNG",
            "agent_type": "executor", "mission_id": None}
    await _apply._dlq_write(task, error="all attempts failed",
                            category="exhausted", attempts=5)

    demand_tasks = [
        e for e in enqueued
        if (e.get("context") or {}).get("payload", {}).get("action")
        == "yalayut_demand"
    ]
    assert len(demand_tasks) == 1
    p = demand_tasks[0]["context"]["payload"]
    assert p["signal_type"] == "dlq"
    assert p["source_step_pattern"] == "dlq:777"


@pytest.mark.asyncio
async def test_repeat_pattern_scan_amplifies_recurring_pattern(db):
    from yalayut.discovery.demand_drain import _scan_repeat_patterns

    # Same pattern, 3 distinct signal types — a recurrence.
    for st in ("step_entry_miss", "tool_call", "dlq"):
        await _demand.record(source_step_pattern="recur:pdf-parse",
                             intent_keywords=["pdf"], signal_type=st,
                             confidence=0.3)
    added = await _scan_repeat_patterns()
    assert added == 1

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT COUNT(*) FROM yalayut_demand_signals "
        "WHERE source_step_pattern = 'recur:pdf-parse' "
        "AND signal_type = 'repeat_pattern'")
    assert (await cur.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_run_demand_drain_triggers_discovery_above_threshold(db, monkeypatch):
    from yalayut.discovery import demand_drain

    # Stack a pattern over 0.5: two 0.3 signals -> 1-(0.7*0.7)=0.51.
    await _demand.record(source_step_pattern="drain:slack-bot",
                         intent_keywords=["slack"], signal_type="tool_call",
                         confidence=0.3)
    await _demand.record(source_step_pattern="drain:slack-bot",
                         intent_keywords=["slack"], signal_type="dlq",
                         confidence=0.3)

    discovered = []

    async def _fake_on_demand(demand):
        discovered.append(demand["source_step_pattern"])
        await _demand.mark_discovered(demand["source_step_pattern"])
        return {"pattern": demand["source_step_pattern"], "artifacts_ingested": 0}

    monkeypatch.setattr("yalayut.discovery.on_demand.on_demand_discovery",
                        _fake_on_demand)

    summary = await demand_drain.run_demand_drain()
    assert "drain:slack-bot" in discovered
    assert summary["patterns_discovered"] >= 1

    # Drained -> no longer pending.
    pending = await _demand.pending_signals(limit=50)
    assert all(p["source_step_pattern"] != "drain:slack-bot" for p in pending)


@pytest.mark.asyncio
async def test_daily_executor_runs_demand_drain(db, monkeypatch):
    from mr_roboto.executors import yalayut_discovery as _exec

    drained = {"called": False}

    async def _fake_daily():
        return {"sources_scanned": 0, "artifacts_ingested": 0, "errors": []}

    async def _fake_drain():
        drained["called"] = True
        return {"patterns_discovered": 2, "repeat_patterns_added": 0,
                "patterns_considered": 2, "errors": []}

    monkeypatch.setattr("yalayut.daily_discovery", _fake_daily)
    monkeypatch.setattr("yalayut.run_demand_drain", _fake_drain)

    res = await _exec.run({"payload": {"mode": "daily"}})

    assert drained["called"] is True
    assert res["demand_drain"]["patterns_discovered"] == 2
    # Existing top-level keys preserved for backward compat.
    assert "sources_scanned" in res
