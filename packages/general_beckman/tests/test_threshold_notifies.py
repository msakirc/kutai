import pytest
import aiosqlite
from unittest.mock import AsyncMock, patch


def _reset_db_for(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    return db_path


@pytest.mark.asyncio
async def test_spent_increments(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd) VALUES ('m', 5.0, 0.0)"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    from general_beckman import on_task_finished
    await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 0.50, "status": "completed"})

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT spent_usd FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == pytest.approx(0.50)


@pytest.mark.asyncio
async def test_threshold_notify_50pct(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd) VALUES ('m', 1.0, 0.0)"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    notify = AsyncMock()
    with patch("general_beckman.notify_threshold", notify):
        from general_beckman import on_task_finished
        await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 0.55, "status": "completed"})

    notify.assert_called_once()
    call = notify.call_args
    pct = call.kwargs.get("pct") if "pct" in call.kwargs else call.args[1] if len(call.args) > 1 else None
    assert pct == 50


@pytest.mark.asyncio
async def test_each_threshold_fires_once(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd, context) "
            "VALUES ('m', 1.0, 0.55, json_object('thresholds_fired', json_array(50)))"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    notify = AsyncMock()
    with patch("general_beckman.notify_threshold", notify):
        from general_beckman import on_task_finished
        # Adds 0.20 → 0.75 → crosses 75%
        await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 0.20, "status": "completed"})

    fired_pcts = []
    for c in notify.call_args_list:
        if "pct" in c.kwargs:
            fired_pcts.append(c.kwargs["pct"])
        elif len(c.args) > 1:
            fired_pcts.append(c.args[1])
    assert 75 in fired_pcts
    assert 50 not in fired_pcts


@pytest.mark.asyncio
async def test_no_notify_when_no_ceiling(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO missions (title, spent_usd) VALUES ('m', 0)")
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    notify = AsyncMock()
    with patch("general_beckman.notify_threshold", notify):
        from general_beckman import on_task_finished
        await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 100.0, "status": "completed"})

    notify.assert_not_called()
