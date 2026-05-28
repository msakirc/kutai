"""SP2 Task 1: Telegram casual-reply CPS migration."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

import src.infra.db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "sp2.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_handle_casual_enqueues_with_on_complete(tmp_path, monkeypatch):
    """_handle_casual must enqueue with on_complete, NOT await_inline."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 4242  # child task id

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)

        fake_update = MagicMock()
        fake_update.effective_chat.id = 12345
        fake_update.message.chat.id = 12345

        await bot._handle_casual("Hey, how are you?", fake_update)

        assert captured["kwargs"].get("await_inline") in (False, None), (
            f"await_inline must NOT be set; got {captured['kwargs']!r}"
        )
        assert captured["kwargs"]["on_complete"] == "telegram.casual_reply_resume"
        assert captured["kwargs"]["on_error"] == "telegram.casual_reply_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs["chat_id"] == 12345
        assert cs["text"] == "Hey, how are you?"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_casual_reply_resume_sends_telegram_message(tmp_path, monkeypatch):
    """Resume must extract content from result['result']['content'] and send to chat_id."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import TelegramInterface
        # Build a fake bot with a mock app.bot.send_message.
        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        from src.app.telegram_bot import set_telegram
        set_telegram(bot)

        from src.app.telegram_bot import _casual_reply_resume  # registered handler
        await _casual_reply_resume(
            child_task_id=4242,
            result={"status": "completed",
                    "result": {"content": "I am fine, thanks!"}},
            state={"chat_id": 12345, "text": "Hey, how are you?"},
        )
        bot.app.bot.send_message.assert_awaited_once()
        assert bot.app.bot.send_message.call_args.kwargs["chat_id"] == 12345
        assert "I am fine" in bot.app.bot.send_message.call_args.kwargs["text"]
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


@pytest.mark.asyncio
async def test_casual_reply_err_sends_fallback_text(tmp_path, monkeypatch):
    """on_error sends the documented fallback string."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        from src.app.telegram_bot import set_telegram
        set_telegram(bot)

        from src.app.telegram_bot import _casual_reply_err
        await _casual_reply_err(
            child_task_id=4242,
            result={"status": "failed", "error": "timeout"},
            state={"chat_id": 12345, "text": "Hi"},
        )
        bot.app.bot.send_message.assert_awaited_once()
        sent_text = bot.app.bot.send_message.call_args.kwargs["text"]
        assert "task or mission" in sent_text.lower() or "send me" in sent_text.lower()
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


@pytest.mark.asyncio
async def test_register_continuations_registers_casual_handlers(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import register_continuations
        from general_beckman.continuations import _HANDLERS
        # Clear and re-register.
        _HANDLERS.pop("telegram.casual_reply_resume", None)
        _HANDLERS.pop("telegram.casual_reply_err", None)
        register_continuations()
        assert "telegram.casual_reply_resume" in _HANDLERS
        assert "telegram.casual_reply_err" in _HANDLERS
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_telegram_module_added_to_handler_modules():
    """Restart-reconcile import list must include telegram_bot."""
    from general_beckman.continuations import _HANDLER_MODULES
    assert "src.app.telegram_bot" in _HANDLER_MODULES, (
        f"_HANDLER_MODULES = {_HANDLER_MODULES!r}"
    )


# ─── Task 1.5: _classify_user_message + _route_classified_message ──────────


@pytest.mark.asyncio
async def test_classify_user_message_enqueues_with_cps(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 5151

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}

        # The fn used to RETURN dict; now returns None (queued).
        rv = await bot._classify_user_message(
            "How is the coffee mission going?", chat_id=999
        )
        assert rv is None, f"_classify_user_message must return None (queued), got {rv!r}"
        assert captured["kwargs"]["on_complete"] == "telegram.message_route_resume"
        assert captured["kwargs"]["on_error"] == "telegram.message_route_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs["chat_id"] == 999
        assert cs["text"] == "How is the coffee mission going?"
        assert cs["flow"] == "message_route"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_message_route_resume_routes_via_extracted_helper(tmp_path, monkeypatch):
    """Resume must call `_route_classified_message` with parsed classification."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import (
            _message_route_resume,
            TelegramInterface,
            set_telegram,
        )
        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        bot.user_last_task_id = {}
        bot._pending_clarifications = {}
        bot._pending_mission = {}
        bot._route_classified_message = AsyncMock()
        set_telegram(bot)

        await _message_route_resume(
            child_task_id=5151,
            result={"status": "completed",
                    "result": {"content": '{"type": "casual", "confidence": 0.9}'}},
            state={"chat_id": 999, "text": "hi", "flow": "message_route"},
        )

        bot._route_classified_message.assert_awaited_once()
        args = bot._route_classified_message.call_args.args
        # signature: _route_classified_message(chat_id, text, classification)
        assert args[0] == 999
        assert args[1] == "hi"
        assert args[2]["type"] == "casual"
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


# ─── Task 1.6: cmd_mission migration ───────────────────────────────────────


@pytest.mark.asyncio
async def test_cmd_mission_uses_cps_classification(tmp_path, monkeypatch):
    """cmd_mission's no-workflow path goes through the CPS classifier."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        bot._pending_mission = {}
        bot.user_last_task_id = {}
        bot._reply = AsyncMock()
        bot._classify_user_message = AsyncMock()

        fake_update = MagicMock()
        fake_update.message.chat_id = 7777
        fake_context = MagicMock()
        fake_context.args = ["Build", "a", "login", "page"]

        await bot.cmd_mission(fake_update, fake_context)
        bot._classify_user_message.assert_awaited_once()
        # chat_id keyword must be passed (otherwise resume can't reply)
        kwargs = bot._classify_user_message.call_args.kwargs
        assert kwargs.get("chat_id") == 7777
        # The mission description must be cached in _pending_mission for the resume.
        assert bot._pending_mission.get(7777) == "Build a login page"
    finally:
        await _close_db()
