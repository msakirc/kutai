"""SP2 Task 3: interview.summarize_interview CPS migration."""
import json
import pytest
from unittest.mock import AsyncMock

import src.infra.db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "iv.db"))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _seed_note(transcript: str = "A real conversation transcript.") -> int:
    db = await _db_mod.get_db()
    await db.execute(
        "INSERT INTO interview_notes (note_id, product_id, transcript_md) "
        "VALUES (?, ?, ?)", (1, "prod1", transcript))
    await db.commit()
    return 1


@pytest.mark.asyncio
async def test_summarize_interview_enqueues_with_on_complete(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_note()
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 9001

        monkeypatch.setattr("src.app.interview.beckman_enqueue", fake_enqueue)

        from src.app.interview import summarize_interview
        res = await summarize_interview(note_id=1, product_id="prod1")
        assert res["ok"] is True
        assert res.get("queued") is True
        assert res["note_id"] == 1
        assert captured["kwargs"].get("await_inline") in (False, None)
        assert captured["kwargs"]["on_complete"] == "interview.summary_persist_resume"
        assert captured["kwargs"]["on_error"] == "interview.summary_persist_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs == {"note_id": 1, "product_id": "prod1"}
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_summary_resume_writes_db_from_result_content(tmp_path, monkeypatch):
    """Resume must extract JSON from result['result']['content'] (NOT .output)."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_note()
        from src.app.interview import _summary_persist_resume
        await _summary_persist_resume(
            child_task_id=9001,
            result={"status": "completed", "result": {"content": json.dumps({
                "bullets": ["Point A", "Point B"],
                "quotes": ["Verbatim quote"],
                "insights": "Founder-level take.",
                "action_items": ["Follow up email"],
            })}},
            state={"note_id": 1, "product_id": "prod1"},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT summary_md, quotes_json, insights_md, action_items_json "
            "FROM interview_notes WHERE note_id=?", (1,))
        row = await cur.fetchone()
        assert "Point A" in (row[0] or "")
        assert json.loads(row[1] or "[]") == ["Verbatim quote"]
        assert "Founder-level" in (row[2] or "")
        assert json.loads(row[3] or "[]") == ["Follow up email"]
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_summary_resume_tolerates_non_json_content(tmp_path, monkeypatch):
    """Regex-rescue path (content embedded in prose) must still work."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_note()
        from src.app.interview import _summary_persist_resume
        content_with_prose = (
            "Here is the structured response:\n"
            "{\"bullets\": [\"Embedded\"], \"quotes\": [], "
            "\"insights\": \"\", \"action_items\": []}\n"
            "(end of response)"
        )
        await _summary_persist_resume(
            child_task_id=9002,
            result={"status": "completed",
                    "result": {"content": content_with_prose}},
            state={"note_id": 1, "product_id": "prod1"},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT summary_md FROM interview_notes WHERE note_id=?", (1,))
        assert "Embedded" in (await cur.fetchone())[0]
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_summary_on_error_leaves_row_intact(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_note()
        from src.app.interview import _summary_persist_err
        await _summary_persist_err(
            child_task_id=9003,
            result={"status": "failed", "error": "timeout"},
            state={"note_id": 1, "product_id": "prod1"},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT summary_md FROM interview_notes WHERE note_id=?", (1,))
        assert (await cur.fetchone())[0] is None
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_register_continuations_registers_interview_handlers(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.interview import register_continuations
        from general_beckman.continuations import _HANDLERS
        _HANDLERS.pop("interview.summary_persist_resume", None)
        _HANDLERS.pop("interview.summary_persist_err", None)
        register_continuations()
        assert "interview.summary_persist_resume" in _HANDLERS
        assert "interview.summary_persist_err" in _HANDLERS
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_interview_module_in_handler_modules():
    from general_beckman.continuations import _HANDLER_MODULES
    assert "src.app.interview" in _HANDLER_MODULES
