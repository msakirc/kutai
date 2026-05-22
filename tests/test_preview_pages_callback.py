"""Unit C — preview:pages:<mid> callback in TelegramInterface.handle_callback.

Tests mirror the Update/CallbackQuery mocking used by
``tests/test_spec_patch_callbacks.py`` and monkeypatch
``mr_roboto.publish_preview_pages.publish_preview_pages`` so neither the live
DB nor the network is touched.

Cases
-----
- ready result (ok=True, pending=False, url=<github.io>) → reply contains url.
- pending result (pending=True, reason="no_preview_root") → reply mentions pending.
- bad token (preview:pages: with no id) → error reply, publish NOT called.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot():
    with patch("src.app.telegram_bot.Application"):
        from src.app.telegram_bot import TelegramInterface
        return TelegramInterface(orchestrator=MagicMock())


def _make_callback_update(data: str, chat_id: int = 100):
    update = MagicMock()
    update.callback_query.data = data
    update.callback_query.answer = AsyncMock()
    update.callback_query.message.reply_text = AsyncMock()
    update.callback_query.message.chat_id = chat_id
    update.callback_query.message.chat.id = chat_id
    update.callback_query.edit_message_text = AsyncMock()
    update.effective_chat.id = chat_id
    return update


def _make_context():
    ctx = MagicMock()
    ctx.args = []
    return ctx


def _install_fake_publish(monkeypatch, result: dict):
    """Patch publish_preview_pages at the source so the lazy import finds it."""
    calls: list[dict] = []

    async def _fake_publish(mission_id: int, workspace_path=None, **kw):
        calls.append({"mission_id": mission_id})
        return result

    # The callback does: from mr_roboto.publish_preview_pages import publish_preview_pages
    # Patch the attribute on the already-imported module if present, and also
    # inject a fake module so the lazy import resolves.
    fake_mod = types.ModuleType("mr_roboto.publish_preview_pages")
    fake_mod.publish_preview_pages = _fake_publish  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mr_roboto.publish_preview_pages", fake_mod)

    # Also patch on the parent package if already imported.
    parent = sys.modules.get("mr_roboto")
    if parent is not None:
        monkeypatch.setattr(parent, "publish_preview_pages", fake_mod, raising=False)

    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pages_callback_ready_result_contains_url(monkeypatch):
    """preview:pages:42 with a ready result → reply includes the github.io url."""
    bot = _make_bot()
    calls = _install_fake_publish(monkeypatch, {
        "ok": True,
        "pending": False,
        "url": "https://testowner.github.io/mission-42-project/",
        "reason": "",
    })

    update = _make_callback_update("preview:pages:42")
    ctx = _make_context()
    await bot.handle_callback(update, ctx)

    assert len(calls) == 1, f"expected publish called once, got {calls}"
    assert calls[0]["mission_id"] == 42

    update.callback_query.message.reply_text.assert_called()
    reply_text = update.callback_query.message.reply_text.call_args[0][0]
    assert "https://testowner.github.io/mission-42-project/" in reply_text, (
        f"Expected github.io URL in reply, got: {reply_text!r}"
    )


@pytest.mark.asyncio
async def test_pages_callback_pending_result_mentions_pending(monkeypatch):
    """preview:pages:42 with pending=True → reply mentions 'pending' and the reason."""
    bot = _make_bot()
    calls = _install_fake_publish(monkeypatch, {
        "ok": True,
        "pending": True,
        "reason": "no_preview_root",
        "url": None,
    })

    update = _make_callback_update("preview:pages:42")
    ctx = _make_context()
    await bot.handle_callback(update, ctx)

    assert len(calls) == 1, f"expected publish called once, got {calls}"

    update.callback_query.message.reply_text.assert_called()
    reply_text = update.callback_query.message.reply_text.call_args[0][0]
    # Must mention pending AND the reason
    lower = reply_text.lower()
    assert "pending" in lower, f"Expected 'pending' in reply, got: {reply_text!r}"
    assert "no_preview_root" in reply_text, (
        f"Expected reason string in reply, got: {reply_text!r}"
    )


@pytest.mark.asyncio
async def test_pages_callback_bad_token_no_publish(monkeypatch):
    """preview:pages: (no trailing id) → error reply, publish_preview_pages NOT called."""
    bot = _make_bot()
    calls = _install_fake_publish(monkeypatch, {
        "ok": True,
        "pending": False,
        "url": "https://example.github.io/r/",
        "reason": "",
    })

    update = _make_callback_update("preview:pages:")
    ctx = _make_context()
    await bot.handle_callback(update, ctx)

    assert calls == [], f"publish must NOT be called on bad token, got {calls}"
    update.callback_query.message.reply_text.assert_called()
