"""Task 15 — telemetry round-trip for the image lane.

A full pump cycle (enqueue → next_task → husam.run → on_task_finished) must
write exactly ONE row to model_pick_log and ONE row to model_call_tokens, both
tagged call_category="image" / picked_model "pollinations/flux". The +1 row
counts are the real assertion; the column names are confirmed against the live
schema (model_pick_log.call_category / .picked_model, model_call_tokens
.call_category / .model).
"""
import json

import pytest

import src.infra.db as _db_mod
from src.infra.db import init_db, get_db
import general_beckman  # noqa: F401
from general_beckman import enqueue, next_task, on_task_finished
import husam


@pytest.fixture(autouse=True)
async def _close_db_conn():
    """Close the aiosqlite connection after each test. Its non-daemon background
    thread otherwise blocks interpreter exit and hangs pytest teardown for hours."""
    yield
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation (copy of
    tests/test_beckman_next_task.py::_fresh)."""
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    import general_beckman.cron_seed as cs_mod
    monkeypatch.setattr(cs_mod, "_seeded", False)
    from general_beckman import paused_patterns as _pp
    _pp._patterns.clear()
    await init_db()
    conn = await get_db()
    await conn.execute(
        "UPDATE scheduled_tasks SET next_run = datetime('now', '+1 hour')"
    )
    await conn.commit()


def _fake_pollinations(seed=3, color=(50, 60, 70)):
    class _FakeProvider:
        name = "pollinations"

        def available(self):
            return True

        async def generate(self, spec, *, base_url=None):
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), color).save(buf, "PNG")
            return buf.getvalue(), {"seed_used": seed}

    return _FakeProvider()


@pytest.mark.asyncio
async def test_image_call_writes_pick_log_and_token_rows(monkeypatch, tmp_path):
    await _fresh_db(tmp_path, monkeypatch)
    import paintress
    monkeypatch.setattr(paintress, "_PROVIDERS", {"pollinations": _fake_pollinations()})
    monkeypatch.delenv("HF_TOKEN", raising=False)

    db = await get_db()
    cur = await db.execute("SELECT COUNT(*) FROM model_pick_log")
    base_pick = (await cur.fetchone())[0]
    cur = await db.execute("SELECT COUNT(*) FROM model_call_tokens")
    base_tok = (await cur.fetchone())[0]

    spec = {
        "title": "telemetry-roundtrip", "description": "telemetry roundtrip",
        "agent_type": "image", "kind": "image", "priority": 5,
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "telemetry test",
            "out_dir": str(tmp_path), "width": 64, "height": 64, "filename_hint": "tel",
        }},
    }
    tid = await enqueue(spec)
    task = await next_task()
    assert task["id"] == tid

    from src.core.heartbeat import current_task_id
    _tok = current_task_id.set(tid)
    try:
        result = await husam.run(task)
    finally:
        current_task_id.reset(_tok)
    await on_task_finished(tid, {"status": "completed",
                                 "result": json.dumps(result), **result})

    cur = await db.execute("SELECT COUNT(*) FROM model_pick_log")
    new_pick = (await cur.fetchone())[0]
    cur = await db.execute("SELECT COUNT(*) FROM model_call_tokens")
    new_tok = (await cur.fetchone())[0]
    assert new_pick == base_pick + 1, "model_pick_log gained no row — pick_log wiring broken"
    assert new_tok == base_tok + 1, "model_call_tokens gained no row — token telemetry broken"

    cur = await db.execute(
        "SELECT picked_model, call_category FROM model_pick_log ORDER BY rowid DESC LIMIT 1"
    )
    row = await cur.fetchone()
    assert row[0] == "pollinations/flux"
    assert row[1] == "image"

    cur = await db.execute(
        "SELECT call_category, model FROM model_call_tokens ORDER BY rowid DESC LIMIT 1"
    )
    row = await cur.fetchone()
    assert row[0] == "image"
    assert row[1] == "pollinations/flux"
