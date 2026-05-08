import pytest
import aiosqlite

from src.infra.db import init_db
from general_beckman import on_task_finished
from general_beckman.lifecycle_events import emit_pause, emit_resume


def _reset_db_for(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    return db_path


@pytest.mark.asyncio
async def test_mission_pauses_at_ceiling_and_resumes(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd, lifecycle_state) "
            "VALUES ('m', 1.0, 0.0, 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    # Use the test calling convention: single dict with mission_id + cost_usd.
    await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 0.95, "status": "completed"})
    await on_task_finished({"mission_id": mid, "id": 2, "cost_usd": 0.10, "status": "completed"})

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        spent_row = await (await db.execute(
            "SELECT spent_usd FROM missions WHERE id=?", (mid,))).fetchone()
    assert spent_row[0] >= 1.0  # tracked correctly past ceiling

    # Manually pause (full auto-pause-on-overshoot is in T7/T9; this test
    # focuses on the lifecycle plumbing).
    db_module._db_connection = None
    await emit_pause(mid, reason="ceiling_reached", triggered_by="auto:budget")

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "paused"

    db_module._db_connection = None
    await emit_resume(mid)

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "active"

    # Audit log should record both transitions.
    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        log = await (await db.execute(
            "SELECT to_state FROM mission_lifecycle_log WHERE mission_id=? ORDER BY id", (mid,))).fetchall()
    states = [r[0] for r in log]
    assert "paused" in states
    assert "active" in states
