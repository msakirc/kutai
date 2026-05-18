import pytest
import aiosqlite
from unittest.mock import AsyncMock, patch, MagicMock


def _reset_db_for(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    return db_path


@pytest.mark.asyncio
async def test_notify_threshold_posts_to_thread(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, telegram_thread_id, context) "
            "VALUES ('m', 999, json_object('chat_id', 1234))"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    fake_tg = MagicMock()
    fake_tg.bot = MagicMock()
    fake_tg.bot.send_message = AsyncMock()

    with patch("src.app.telegram_bot.get_telegram", return_value=fake_tg):
        from general_beckman import notify_threshold
        await notify_threshold(mid, pct=50, spent=0.5, ceiling=1.0)

    fake_tg.bot.send_message.assert_called_once()
    kwargs = fake_tg.bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 1234
    assert kwargs["message_thread_id"] == 999
    assert "50%" in kwargs["text"]


@pytest.mark.asyncio
async def test_notify_threshold_no_chat_id_logs_and_returns(tmp_path, monkeypatch, caplog):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, telegram_thread_id, context) VALUES ('m', 999, '{}')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    fake_tg = MagicMock()
    fake_tg.bot = MagicMock()
    fake_tg.bot.send_message = AsyncMock()

    with patch("src.app.telegram_bot.get_telegram", return_value=fake_tg):
        from general_beckman import notify_threshold
        with caplog.at_level("WARNING"):
            await notify_threshold(mid, pct=75, spent=0.75, ceiling=1.0)

    fake_tg.bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_threshold_swallows_telegram_errors(tmp_path, monkeypatch):
    db_path = _reset_db_for(tmp_path, monkeypatch)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, telegram_thread_id, context) "
            "VALUES ('m', 999, json_object('chat_id', 1234))"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    import src.infra.db as db_module
    db_module._db_connection = None

    fake_tg = MagicMock()
    fake_tg.bot = MagicMock()
    fake_tg.bot.send_message = AsyncMock(side_effect=Exception("network"))

    with patch("src.app.telegram_bot.get_telegram", return_value=fake_tg):
        from general_beckman import notify_threshold
        # Should not raise
        await notify_threshold(mid, pct=50, spent=0.5, ceiling=1.0)
