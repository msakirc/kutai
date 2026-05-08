import pytest
import aiosqlite

from general_beckman.lifecycle_events import emit_pause, dlq_cascade_check


def _reset_db_singleton(db_path: str | None = None):
    import src.infra.db as db_module
    db_module._db_connection = None
    db_module._db_connection_path = None
    if db_path is not None:
        db_module.DB_PATH = db_path


@pytest.mark.asyncio
async def test_emit_pause_transitions_state(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    _reset_db_singleton(db_path)
    await emit_pause(mid, reason="no_model_fits_budget", triggered_by="auto:budget")

    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
        assert row[0] == "paused"
        log = await (await db.execute(
            "SELECT from_state, to_state, reason, triggered_by FROM mission_lifecycle_log "
            "WHERE mission_id=?", (mid,))).fetchall()
    assert len(log) == 1
    assert log[0] == ("active", "paused", "no_model_fits_budget", "auto:budget")


@pytest.mark.asyncio
async def test_emit_pause_idempotent_when_already_paused(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'paused')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    _reset_db_singleton(db_path)
    await emit_pause(mid, reason="dup", triggered_by="test")

    async with aiosqlite.connect(db_path) as db:
        log = await (await db.execute(
            "SELECT * FROM mission_lifecycle_log WHERE mission_id=?", (mid,))).fetchall()
    assert log == []  # no log entry on no-op


@pytest.mark.asyncio
async def test_dlq_cascade_triggers_pause_at_3_consec_failures(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    _reset_db_singleton(db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        for _ in range(3):
            await db.execute(
                "INSERT INTO tasks (mission_id, title, status, completed_at) "
                "VALUES (?, 't', 'failed', CURRENT_TIMESTAMP)", (mid,),
            )
        await db.commit()

    _reset_db_singleton(db_path)
    triggered = await dlq_cascade_check(mid)
    assert triggered is True

    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "paused"
