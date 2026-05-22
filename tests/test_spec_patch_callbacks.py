"""Z1 — spec-patch review+apply loop: CALLBACK side.

Founder taps the Apply / Reject inline button surfaced by the proposer.
``handle_callback`` routes ``sp_apply:<mid>:<ts>`` → enqueue a ``coder``
task that applies the reviewed proposal to the upstream spec docs, and
``sp_rej:<mid>:<ts>`` → an ack with no enqueue.

These tests mirror the Update/CallbackQuery mocking used by
``tests/test_menu_flows.py::TestCallbackData`` and monkeypatch
``general_beckman.enqueue`` + ``get_mission_workspace`` so neither the live
DB nor Telegram is touched.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_bot():
    with patch("src.app.telegram_bot.Application"):
        from src.app.telegram_bot import TelegramInterface
        return TelegramInterface(orchestrator=MagicMock())


def _make_callback_update(data, chat_id=100):
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


def _install_fake_beckman(monkeypatch):
    captured: list[dict] = []

    async def _fake_enqueue(spec, *args, **kwargs):
        captured.append(spec)
        return {"task_id": len(captured)}

    fake = types.ModuleType("general_beckman")
    fake.enqueue = _fake_enqueue  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "general_beckman", fake)
    return captured


@pytest.mark.asyncio
async def test_sp_apply_enqueues_coder_with_proposal_path(monkeypatch):
    bot = _make_bot()
    captured = _install_fake_beckman(monkeypatch)

    # get_mission_workspace is imported `from src.tools.workspace import ...`
    # inside the branch — patch at the source module.
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace",
        lambda mid: rf"C:\ws\mission_{mid}",
        raising=False,
    )

    update = _make_callback_update("sp_apply:42:1234")
    ctx = _make_context()
    await bot.handle_callback(update, ctx)

    assert len(captured) == 1, captured
    spec = captured[0]
    assert spec["agent_type"] == "coder"
    assert spec["mission_id"] == 42
    arts = spec["context"]["input_artifacts"]
    assert len(arts) == 1
    # Reconstructed proposal path: spec_patch_proposal_<ts>.md under .propagation
    assert "spec_patch_proposal_1234.md" in arts[0]
    assert ".propagation" in arts[0]
    # acknowledges the founder
    update.callback_query.message.reply_text.assert_called()
    txt = update.callback_query.message.reply_text.call_args[0][0]
    assert "42" in txt


@pytest.mark.asyncio
async def test_sp_apply_bad_token_replies_error_no_enqueue(monkeypatch):
    bot = _make_bot()
    captured = _install_fake_beckman(monkeypatch)

    update = _make_callback_update("sp_apply:notanint:1234")
    ctx = _make_context()
    await bot.handle_callback(update, ctx)

    assert captured == []
    update.callback_query.message.reply_text.assert_called()


@pytest.mark.asyncio
async def test_sp_rej_acks_without_enqueue(monkeypatch):
    bot = _make_bot()
    captured = _install_fake_beckman(monkeypatch)

    update = _make_callback_update("sp_rej:42:1234")
    ctx = _make_context()
    await bot.handle_callback(update, ctx)

    assert captured == [], "reject must not enqueue anything"
    update.callback_query.message.reply_text.assert_called()
    txt = update.callback_query.message.reply_text.call_args[0][0]
    assert "42" in txt
