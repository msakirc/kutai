"""
TDD tests for Site 9 migration: Telegram casual chat reply enqueues
with kind='chat' via beckman.enqueue(await_inline=True).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


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
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "Hello! How can I help?"},
            error=None,
        )

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
    assert kwargs.get("await_inline") is True
    assert kwargs.get("parent_id") is None
    assert spec["context"]["llm_call"]["raw_dispatch"] is True
    assert len(replies) == 1
    assert "Hello" in replies[0]


@pytest.mark.asyncio
async def test_casual_chat_await_inline_true(tmp_path, monkeypatch):
    """await_inline=True must be passed."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured_kwargs = {}

    async def fake_enqueue(spec, **kwargs):
        captured_kwargs.update(kwargs)
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "Sure!"},
            error=None,
        )

    fake_update = MagicMock()
    replies = []

    async def fake_reply(update, text, **kwargs):
        replies.append(text)

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._reply = fake_reply
        await bot._handle_casual("thanks", fake_update)

    assert captured_kwargs.get("await_inline") is True


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
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "ok"},
            error=None,
        )

    fake_update = MagicMock()

    async def fake_reply(update, text, **kwargs):
        pass

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._reply = fake_reply
        await bot._handle_casual("yo", fake_update)

    assert captured_kwargs.get("parent_id") is None


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
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "yep"},
            error=None,
        )

    fake_update = MagicMock()

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
