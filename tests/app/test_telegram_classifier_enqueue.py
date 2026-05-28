"""
Tests for Site 8: Telegram message classifier enqueues via beckman.enqueue.

SP2: migrated from `await_inline=True` to CPS — on_complete/on_error
continuations route the classification result through
`_message_route_resume` and dispatch to `_route_classified_message`.
The bot returns immediately from `_classify_user_message`.
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
        return 9999  # child task id

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface

        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}

        rv = await bot._classify_user_message(
            "Build me a website", chat_id=42,
        )

    # SP2: classifier returns None (queued); reply arrives via the resume.
    assert rv is None
    spec = captured["spec"]
    kwargs = captured["kwargs"]

    assert spec["kind"] == "classifier", f"Expected kind='classifier', got {spec.get('kind')!r}"
    # SP2: no longer await_inline.
    assert kwargs.get("await_inline") in (False, None)
    assert kwargs.get("on_complete") == "telegram.message_route_resume"
    assert kwargs.get("on_error") == "telegram.message_route_err"
    assert spec["context"]["llm_call"]["raw_dispatch"] is True


@pytest.mark.asyncio
async def test_message_classifier_cps_state_carries_chat_and_text(tmp_path, monkeypatch):
    """cont_state must include chat_id, text and the flow discriminator."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured_kwargs = {}

    async def fake_enqueue(spec, **kwargs):
        captured_kwargs.update(kwargs)
        return 9999

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        await bot._classify_user_message("hey", chat_id=42)

    cs = captured_kwargs.get("cont_state") or {}
    assert cs.get("chat_id") == 42
    assert cs.get("text") == "hey"
    assert cs.get("flow") == "message_route"


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
        return 9999

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        await bot._classify_user_message("do something", chat_id=1)

    assert captured_kwargs.get("parent_id") in (None, 0) or "parent_id" not in captured_kwargs


@pytest.mark.asyncio
async def test_message_classifier_fallback_on_enqueue_failure(tmp_path, monkeypatch):
    """On enqueue failure with chat_id, fall back to keyword routing in place."""
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
        bot._route_classified_message = AsyncMock()
        # Should NOT raise; should call the routing helper with a keyword
        # classification.
        await bot._classify_user_message("hello world", chat_id=42)

    # _route_classified_message was invoked in the local-failure branch.
    bot._route_classified_message.assert_awaited_once()


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
        return 9999

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        await bot._classify_user_message("Fix the login bug", chat_id=1)

    llm_call = captured["spec"]["context"]["llm_call"]
    assert "messages" in llm_call
    assert isinstance(llm_call["messages"], list)
    assert len(llm_call["messages"]) > 0
