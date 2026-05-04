"""
TDD tests for Site 8 migration: Telegram message classifier enqueues
with kind='classifier' via beckman.enqueue(await_inline=True).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_message_classifier_enqueues_with_kind_classifier(tmp_path, monkeypatch):
    """_classify_user_message must call beckman.enqueue with kind='classifier'."""
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
            result={"content": '{"type": "task", "confidence": 0.9}'},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        # Import the bot but patch heavy deps so we can instantiate minimally
        from src.app.telegram_bot import TelegramInterface

        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}

        result = await bot._classify_user_message("Build me a website")

    spec = captured["spec"]
    kwargs = captured["kwargs"]

    assert spec["kind"] == "classifier", f"Expected kind='classifier', got {spec.get('kind')!r}"
    assert kwargs.get("await_inline") is True
    assert kwargs.get("parent_id") is None
    assert spec["context"]["llm_call"]["raw_dispatch"] is True
    assert result.get("type") == "task"


@pytest.mark.asyncio
async def test_message_classifier_await_inline_true(tmp_path, monkeypatch):
    """await_inline=True must be passed so caller blocks on result."""
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
            result={"content": '{"type": "casual", "confidence": 0.8}'},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        await bot._classify_user_message("hey")

    assert captured_kwargs.get("await_inline") is True


@pytest.mark.asyncio
async def test_message_classifier_parent_id_none(tmp_path, monkeypatch):
    """parent_id must be None — these classifications are not subtasks."""
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
            result={"content": '{"type": "task", "confidence": 0.7}'},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        await bot._classify_user_message("do something")

    assert captured_kwargs.get("parent_id") is None


@pytest.mark.asyncio
async def test_message_classifier_fallback_on_failure(tmp_path, monkeypatch):
    """On enqueue failure, classifier falls back to keyword-based result (no raise)."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fail_enqueue(spec, **kwargs):
        raise RuntimeError("beckman unavailable")

    with patch("general_beckman.enqueue", fail_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        result = await bot._classify_user_message("hello world")

    # Should not raise; keyword fallback returns a dict with 'type'
    assert isinstance(result, dict)
    assert "type" in result


@pytest.mark.asyncio
async def test_message_classifier_spec_context_has_messages(tmp_path, monkeypatch):
    """spec.context.llm_call must carry the messages list."""
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
            result={"content": '{"type": "task", "confidence": 0.85}'},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        await bot._classify_user_message("Fix the login bug")

    llm_call = captured["spec"]["context"]["llm_call"]
    assert "messages" in llm_call
    assert isinstance(llm_call["messages"], list)
    assert len(llm_call["messages"]) > 0
