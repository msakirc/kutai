"""Z8 T4D — `/ops_log <mission_id>` Telegram command."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


async def _fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "ops_log.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


def _make_bot():
    from src.app.telegram_bot import TelegramInterface
    bot = TelegramInterface.__new__(TelegramInterface)
    bot._reply = AsyncMock()
    return bot


def _make_update():
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


async def _seed_action(db_mod, *, mission_id, verb, reversibility, status):
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO registry_events "
        "(scope, target, event, payload_json, mission_id, task_id, "
        " verb, reversibility) "
        "VALUES ('action', ?, ?, ?, ?, NULL, ?, ?)",
        (
            verb,
            verb,
            json.dumps({"payload": {}, "status": status}),
            mission_id,
            verb,
            reversibility,
        ),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_ops_log_usage_when_no_args(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    bot = _make_bot()
    update = _make_update()
    context = MagicMock()
    context.args = []
    await bot.cmd_ops_log(update, context)
    bot._reply.assert_awaited_once()
    msg = bot._reply.await_args.args[1]
    assert "Usage" in msg


@pytest.mark.asyncio
async def test_ops_log_bad_id(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    bot = _make_bot()
    update = _make_update()
    context = MagicMock()
    context.args = ["not_an_int"]
    await bot.cmd_ops_log(update, context)
    msg = bot._reply.await_args.args[1]
    assert "integer" in msg.lower()


@pytest.mark.asyncio
async def test_ops_log_empty(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    bot = _make_bot()
    update = _make_update()
    context = MagicMock()
    context.args = ["5"]
    await bot.cmd_ops_log(update, context)
    msg = bot._reply.await_args.args[1]
    assert "No on-call actions" in msg


@pytest.mark.asyncio
async def test_ops_log_renders_recent_actions(tmp_path, monkeypatch):
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    for verb, rev, status in [
        ("restart_service", "partial", "ok"),
        ("rollback_to_last_green", "irreversible", "ok"),
        ("scale_up", "partial", "ok"),
    ]:
        await _seed_action(
            db_mod, mission_id=5, verb=verb, reversibility=rev, status=status,
        )
    bot = _make_bot()
    update = _make_update()
    context = MagicMock()
    context.args = ["5"]
    await bot.cmd_ops_log(update, context)
    msg = bot._reply.await_args.args[1]
    assert "restart_service" in msg
    assert "rollback_to_last_green" in msg
    assert "scale_up" in msg
    assert "Ops log mission 5" in msg


@pytest.mark.asyncio
async def test_ops_log_scoped_per_mission(tmp_path, monkeypatch):
    """Actions for mission 9 must not appear when querying mission 5."""
    db_mod = await _fresh_db(tmp_path, monkeypatch)
    await _seed_action(
        db_mod, mission_id=9, verb="restart_service", reversibility="partial", status="ok",
    )
    bot = _make_bot()
    update = _make_update()
    context = MagicMock()
    context.args = ["5"]
    await bot.cmd_ops_log(update, context)
    msg = bot._reply.await_args.args[1]
    assert "restart_service" not in msg
    assert "No on-call actions" in msg
