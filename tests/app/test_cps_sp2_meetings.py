"""SP2 Task 4: meetings._call_llm_meeting_brief CPS migration."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

import src.infra.db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "m.db"))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _seed_meeting(meeting_id=1, product_id="prod1") -> None:
    db = await _db_mod.get_db()
    await db.execute(
        "INSERT INTO meetings (meeting_id, product_id, contact_id, "
        "scheduled_for, purpose) VALUES (?, ?, ?, "
        "strftime('%Y-%m-%d %H:%M:%S','now'), 'test')",
        (meeting_id, product_id, 7),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_enqueue_meeting_brief_uses_cps(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_meeting()
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 7777

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.meetings import enqueue_meeting_brief
        ctx = {"contact": {"display_name": "X"}, "meeting": {"meeting_id": 1}}
        cid = await enqueue_meeting_brief(
            ctx, meeting_id=1, product_id="prod1"
        )
        assert cid == 7777
        assert captured["kwargs"].get("await_inline") in (False, None)
        assert captured["kwargs"]["on_complete"] == "meetings.brief_persist_resume"
        assert captured["kwargs"]["on_error"] == "meetings.brief_persist_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs["meeting_id"] == 1
        assert cs["product_id"] == "prod1"
        assert "ctx" in cs
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_brief_resume_writes_brief_md(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_meeting()
        from src.app.meetings import _brief_persist_resume
        ctx = {"contact": {"display_name": "Alice"}, "meeting": {"meeting_id": 1}}
        await _brief_persist_resume(
            child_task_id=7777,
            result={"status": "completed",
                    "result": {"content": json.dumps({
                        "talking_points": ["TP1", "TP2"],
                        "suggested_asks": ["Ask1"],
                    })}},
            state={"meeting_id": 1, "product_id": "prod1", "ctx": ctx},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT brief_md FROM meetings WHERE meeting_id=?", (1,))
        brief_md = (await cur.fetchone())[0]
        assert "TP1" in (brief_md or "")
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_brief_on_error_writes_unavailable_brief(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_meeting()
        from src.app.meetings import _brief_persist_err
        ctx = {"contact": {"display_name": "X"}, "meeting": {"meeting_id": 1}}
        await _brief_persist_err(
            child_task_id=7777,
            result={"status": "failed", "error": "timeout"},
            state={"meeting_id": 1, "product_id": "prod1", "ctx": ctx},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT brief_md FROM meetings WHERE meeting_id=?", (1,))
        brief_md = (await cur.fetchone())[0]
        assert "unavailable" in (brief_md or "").lower()
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_register_continuations_registers_meeting_handlers(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.meetings import register_continuations
        from general_beckman.continuations import _HANDLERS
        _HANDLERS.pop("meetings.brief_persist_resume", None)
        _HANDLERS.pop("meetings.brief_persist_err", None)
        register_continuations()
        assert "meetings.brief_persist_resume" in _HANDLERS
        assert "meetings.brief_persist_err" in _HANDLERS
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_meetings_module_in_handler_modules():
    from general_beckman.continuations import _HANDLER_MODULES
    assert "src.app.meetings" in _HANDLER_MODULES
