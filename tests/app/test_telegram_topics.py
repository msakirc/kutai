"""Z10 T2B (D1) — per-mission forum topic provisioning."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "topics.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_ensure_creates_then_returns_same_id(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "demo desc")

    from src.app.telegram_topics import ensure_mission_topic

    bot = MagicMock()
    forum_topic = MagicMock()
    forum_topic.message_thread_id = 42
    bot.create_forum_topic = AsyncMock(return_value=forum_topic)

    thread_id = await ensure_mission_topic(bot, mid, "Demo", chat_id=-1001)
    assert thread_id == 42
    bot.create_forum_topic.assert_awaited_once()

    # Second call: idempotent — must NOT call create_forum_topic again.
    bot.create_forum_topic.reset_mock()
    thread_id2 = await ensure_mission_topic(bot, mid, "Demo", chat_id=-1001)
    assert thread_id2 == 42
    bot.create_forum_topic.assert_not_awaited()


@pytest.mark.asyncio
async def test_falls_back_to_flat_on_non_forum_error(tmp_path, monkeypatch, caplog):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")

    from src.app.telegram_topics import ensure_mission_topic

    bot = MagicMock()
    bot.create_forum_topic = AsyncMock(
        side_effect=RuntimeError("CHAT_NOT_FORUM"),
    )

    thread_id = await ensure_mission_topic(bot, mid, "Demo", chat_id=12345)
    assert thread_id is None
    # Mission row's thread_id should remain NULL.
    m = await db_mod.get_mission(mid)
    assert m["telegram_thread_id"] is None


@pytest.mark.asyncio
async def test_post_to_mission_thread_with_thread(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")
    # Pretend a thread was already provisioned.
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE missions SET telegram_thread_id = 99 WHERE id = ?", (mid,),
    )
    await db.commit()

    from src.app.telegram_topics import post_to_mission_thread
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1234))

    await post_to_mission_thread(bot, mid, "hello", chat_id=-1001)
    bot.send_message.assert_awaited_once()
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["message_thread_id"] == 99
    assert kwargs["text"] == "hello"


@pytest.mark.asyncio
async def test_post_to_mission_thread_flat_fallback(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("Demo", "x")

    from src.app.telegram_topics import post_to_mission_thread
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))

    await post_to_mission_thread(bot, mid, "hi", chat_id=-1001)
    kwargs = bot.send_message.await_args.kwargs
    # No thread → prefixed flat-mode post.
    assert "message_thread_id" not in kwargs
    assert kwargs["text"].startswith(f"[Mission {mid}]")
