"""
Tests for Site 9: Telegram casual chat reply enqueues via beckman.enqueue.

SP2: migrated from `await_inline=True` to CPS — on_complete/on_error
continuations route the eventual reply through `_casual_reply_resume`.
The bot returns immediately from `_handle_casual`.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock



@pytest.fixture(autouse=True)
async def _reset_db_singleton():
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try: await _dbmod._db_connection.close()
        except Exception: pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try: await _dbmod._db_connection.close()
        except Exception: pass
    _dbmod._db_connection = None


@pytest.mark.asyncio
async def test_casual_chat_enqueues_with_kind_chat(tmp_path, monkeypatch):
    """_handle_casual must call beckman.enqueue with kind='chat'."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return 1234  # child task id

    # Build a minimal fake Update
    fake_update = MagicMock()
    fake_update.effective_chat.id = 12345

    replies = []

    async def fake_reply(update, text, **kwargs):
        replies.append(text)

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._reply = fake_reply

        await bot._handle_casual("Hey, how are you?", fake_update)

    spec = captured["spec"]
    kwargs = captured["kwargs"]

    assert spec["kind"] == "chat", f"Expected kind='chat', got {spec.get('kind')!r}"
    # SP2: no longer await_inline — CPS via on_complete instead.
    assert kwargs.get("await_inline") in (False, None)
    assert kwargs.get("on_complete") == "telegram.casual_reply_resume"
    assert kwargs.get("on_error") == "telegram.casual_reply_err"
    assert spec["context"]["llm_call"]["raw_dispatch"] is True
    # The bot now returns immediately — no inline reply.
    assert replies == []


@pytest.mark.asyncio
async def test_casual_chat_uses_cps_on_complete(tmp_path, monkeypatch):
    """SP2: on_complete continuation name must be passed."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured_kwargs = {}

    async def fake_enqueue(spec, **kwargs):
        captured_kwargs.update(kwargs)
        return 1234

    fake_update = MagicMock()
    fake_update.effective_chat.id = 99

    async def fake_reply(update, text, **kwargs):
        pass

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._reply = fake_reply
        await bot._handle_casual("thanks", fake_update)

    assert captured_kwargs.get("on_complete") == "telegram.casual_reply_resume"
    assert captured_kwargs.get("on_error") == "telegram.casual_reply_err"
    cs = captured_kwargs.get("cont_state") or {}
    assert cs.get("chat_id") == 99
    assert cs.get("text") == "thanks"


@pytest.mark.asyncio
async def test_casual_chat_parent_id_none(tmp_path, monkeypatch):
    """parent_id must be None — not a subtask."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured_kwargs = {}

    async def fake_enqueue(spec, **kwargs):
        captured_kwargs.update(kwargs)
        return 1234

    fake_update = MagicMock()
    fake_update.effective_chat.id = 1

    async def fake_reply(update, text, **kwargs):
        pass

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._reply = fake_reply
        await bot._handle_casual("yo", fake_update)

    assert captured_kwargs.get("parent_id") in (None, 0) or "parent_id" not in captured_kwargs


@pytest.mark.asyncio
async def test_casual_chat_fallback_on_failure(tmp_path, monkeypatch):
    """On enqueue failure, fallback reply is sent (no raise)."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fail_enqueue(spec, **kwargs):
        raise RuntimeError("beckman down")

    fake_update = MagicMock()
    fake_update.effective_chat.id = 1
    replies = []

    async def fake_reply(update, text, **kwargs):
        replies.append(text)

    with patch("general_beckman.enqueue", fail_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._reply = fake_reply
        await bot._handle_casual("hello", fake_update)

    assert len(replies) == 1
    assert "task" in replies[0].lower() or "help" in replies[0].lower()


@pytest.mark.asyncio
async def test_casual_chat_spec_contains_message_text(tmp_path, monkeypatch):
    """spec.context.llm_call.messages must carry the user text."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        return 1234

    fake_update = MagicMock()
    fake_update.effective_chat.id = 1

    async def fake_reply(update, text, **kwargs):
        pass

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._reply = fake_reply
        await bot._handle_casual("What time is it?", fake_update)

    llm_call = captured["spec"]["context"]["llm_call"]
    messages = llm_call.get("messages", [])
    assert any("What time is it?" in str(m) for m in messages)
