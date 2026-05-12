"""Z6 polish P2 — stripe_revenue_digest Telegram thread fallback.

Mirrors the founder_action notifier pattern: when the mission has a
``telegram_thread_id``, send with ``message_thread_id=thread_id``;
otherwise prefix ``[Mission N] ...`` and send to the admin chat.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "digest.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_digest_post_uses_thread_when_set(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("rev", "")
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE missions SET telegram_thread_id = ? WHERE id = ?",
        (777, mid),
    )
    await db.commit()

    bot = MagicMock()
    bot.send_message = AsyncMock()
    fake_tg = MagicMock()
    fake_tg.app = MagicMock()
    fake_tg.app.bot = bot

    from src.app import telegram_bot as tb_mod
    monkeypatch.setattr(tb_mod, "get_telegram", lambda: fake_tg)

    import src.app.telegram_topics as topics_mod
    monkeypatch.setattr(
        topics_mod, "TELEGRAM_ADMIN_CHAT_ID", "9001", raising=False,
    )

    from mr_roboto.executors.stripe_revenue_digest import (
        _post_to_mission_thread,
    )
    await _post_to_mission_thread(mid, "# digest body")
    bot.send_message.assert_awaited_once()
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs.get("message_thread_id") == 777
    assert kwargs.get("text") == "# digest body"


@pytest.mark.asyncio
async def test_digest_post_flat_prefix_when_no_thread(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("rev2", "")
    # No telegram_thread_id set on this mission.

    bot = MagicMock()
    bot.send_message = AsyncMock()
    fake_tg = MagicMock()
    fake_tg.app = MagicMock()
    fake_tg.app.bot = bot

    from src.app import telegram_bot as tb_mod
    monkeypatch.setattr(tb_mod, "get_telegram", lambda: fake_tg)

    import src.app.telegram_topics as topics_mod
    monkeypatch.setattr(
        topics_mod, "TELEGRAM_ADMIN_CHAT_ID", "9002", raising=False,
    )

    from mr_roboto.executors.stripe_revenue_digest import (
        _post_to_mission_thread,
    )
    await _post_to_mission_thread(mid, "weekly numbers")
    bot.send_message.assert_awaited_once()
    kwargs = bot.send_message.await_args.kwargs
    # Thread-less path: no message_thread_id, text gets flat prefix.
    assert "message_thread_id" not in kwargs
    assert kwargs.get("text", "").startswith(f"[Mission {mid}] ")
    assert "weekly numbers" in kwargs.get("text", "")


@pytest.mark.asyncio
async def test_digest_post_noop_for_system_mission(tmp_path, monkeypatch):
    """SYSTEM_MISSION_ID (cron path) must never send to Telegram."""
    await _setup(tmp_path, monkeypatch)
    bot = MagicMock()
    bot.send_message = AsyncMock()
    fake_tg = MagicMock()
    fake_tg.app = MagicMock()
    fake_tg.app.bot = bot

    from src.app import telegram_bot as tb_mod
    monkeypatch.setattr(tb_mod, "get_telegram", lambda: fake_tg)

    from mr_roboto.executors.stripe_revenue_digest import (
        _post_to_mission_thread,
        SYSTEM_MISSION_ID,
    )
    await _post_to_mission_thread(SYSTEM_MISSION_ID, "body")
    bot.send_message.assert_not_awaited()
