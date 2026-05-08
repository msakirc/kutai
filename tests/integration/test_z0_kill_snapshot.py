import pytest
import aiosqlite
import json
from unittest.mock import AsyncMock, MagicMock

from src.infra.db import init_db
from src.app.telegram_bot import TelegramInterface


def _reset_db_for(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    return db_path


@pytest.mark.asyncio
async def test_kill_mission_writes_snapshot(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't1', 'completed')",
            (mid,),
        )
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()

    update = MagicMock()
    update.effective_chat.id = 1
    context = MagicMock()
    context.args = [str(mid)]

    await tg.cmd_kill_mission(update, context)

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "killed"

    # Snapshot should be in artifact store.
    try:
        from src.workflows.engine.hooks import get_artifact_store
        store = get_artifact_store()
        snap = await store.retrieve(mid, f"mission_kill_{mid}")
    except Exception as e:
        pytest.fail(f"artifact store retrieval failed: {e}")

    assert snap, "snapshot artifact missing"
    parsed = json.loads(snap) if isinstance(snap, str) else snap
    assert parsed["mission"]["id"] == mid
    assert any(t["title"] == "t1" for t in parsed["tasks"])
