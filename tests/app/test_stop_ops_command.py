"""Z8 T1D — /stop_ops Telegram command tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


async def _fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "stop_ops.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


async def _make_ongoing_mission(db_mod) -> int:
    mid = await db_mod.add_mission(title="watch", description="watch", priority=5)
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE missions SET kind='ongoing', lifecycle_state='active' WHERE id=?",
        (mid,),
    )
    await db.commit()
    return mid


def _make_update(text: str, chat_id: int = 42):
    update = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = chat_id
    update.message.chat.id = chat_id
    return update


def _make_bot():
    """Build a TelegramInterface skeleton without touching python-telegram-bot init."""
    from src.app.telegram_bot import TelegramInterface
    bot = TelegramInterface.__new__(TelegramInterface)
    # Patch _reply to a simple recorder so we don't need keyboard state.
    bot._reply = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_stop_ops_revokes_ongoing_mission(tmp_path, monkeypatch):
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await _make_ongoing_mission(db_mod)

    bot = _make_bot()
    update = _make_update(f"/stop_ops {mid}")
    context = MagicMock()
    context.args = [str(mid)]

    await bot.cmd_stop_ops(update, context)

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT lifecycle_state, revoked_at FROM missions WHERE id=?", (mid,)
    )
    row = await cur.fetchone()
    assert row[0] == "revoked"
    assert row[1] is not None
    bot._reply.assert_awaited()


@pytest.mark.asyncio
async def test_stop_ops_no_args_prompts_usage(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    bot = _make_bot()
    update = _make_update("/stop_ops")
    context = MagicMock()
    context.args = []
    await bot.cmd_stop_ops(update, context)
    bot._reply.assert_awaited_once()
    msg = bot._reply.call_args.args[1]
    assert "Usage" in msg or "usage" in msg


@pytest.mark.asyncio
async def test_stop_ops_non_int_rejected(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    bot = _make_bot()
    update = _make_update("/stop_ops abc")
    context = MagicMock()
    context.args = ["abc"]
    await bot.cmd_stop_ops(update, context)
    msg = bot._reply.call_args.args[1]
    assert "int" in msg.lower() or "integer" in msg.lower()


@pytest.mark.asyncio
async def test_stop_ops_unknown_mission_reports_not_found(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    bot = _make_bot()
    update = _make_update("/stop_ops 99999")
    context = MagicMock()
    context.args = ["99999"]
    await bot.cmd_stop_ops(update, context)
    msg = bot._reply.call_args.args[1]
    assert "not" in msg.lower()


@pytest.mark.asyncio
async def test_stop_ops_oneshot_mission_not_revoked(tmp_path, monkeypatch):
    """Revoking a oneshot mission must be a no-op."""
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission(title="x", description="d", priority=5)
    # Default kind='oneshot' from migration.
    bot = _make_bot()
    update = _make_update(f"/stop_ops {mid}")
    context = MagicMock()
    context.args = [str(mid)]
    await bot.cmd_stop_ops(update, context)
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT lifecycle_state FROM missions WHERE id=?", (mid,)
    )
    state = (await cur.fetchone())[0]
    assert state != "revoked"


def test_stop_ops_handler_registered():
    """Smoke: handler is wired in _setup_handlers (without invoking it)."""
    import inspect
    from src.app import telegram_bot as tb_mod
    src = inspect.getsource(tb_mod.TelegramInterface._setup_handlers)
    assert "stop_ops" in src
    assert "cmd_stop_ops" in src
