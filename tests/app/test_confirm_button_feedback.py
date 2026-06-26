"""Every confirm-style callback button must give the founder INSTANT, reliable
feedback (a callback toast) — and the artifact-confirm ✅ OK must never complete
silently when editing the old card fails. Founder report 2026-06-26: "pushing
OK usually doesn't give feedback".
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _toast_text(answer_mock):
    """Extract the toast string passed to query.answer (positional or text=)."""
    args, kwargs = answer_mock.call_args
    return (args[0] if args else kwargs.get("text", "")) or ""


def _rpc_update(data: str):
    update = MagicMock()
    q = update.callback_query
    q.data = data
    q.answer = AsyncMock()
    q.message.chat_id = 42
    q.edit_message_reply_markup = AsyncMock()
    return update


@pytest.mark.asyncio
async def test_artifact_confirm_ok_gives_instant_toast():
    """Tapping ✅ OK answers the callback with a visible toast — instant
    feedback that does not depend on a follow-up send landing."""
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._pending_action = {}
    iface.app = MagicMock()
    iface.app.bot.send_message = AsyncMock()
    iface._regen_step_from_task_context = MagicMock(return_value="")
    update = _rpc_update("rpc:OK:7:55")

    with patch("src.infra.db.get_db", new=AsyncMock(return_value=MagicMock())), \
         patch("src.infra.db.get_task", new=AsyncMock(return_value={})), \
         patch.object(tb, "update_task", new=AsyncMock()):
        await iface._handle_artifact_confirm_callback(update, MagicMock())

    update.callback_query.answer.assert_awaited_once()
    assert _toast_text(update.callback_query.answer)  # non-empty toast


@pytest.mark.asyncio
async def test_artifact_confirm_ok_confirmation_survives_edit_failure():
    """If editing the old card fails (message too old / already edited), the
    task is still completed AND the confirmation message still goes out — the
    OK must never succeed silently."""
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._pending_action = {}
    iface.app = MagicMock()
    iface.app.bot.send_message = AsyncMock()
    iface._regen_step_from_task_context = MagicMock(return_value="")
    update = _rpc_update("rpc:OK:7:55")
    update.callback_query.edit_message_reply_markup = AsyncMock(
        side_effect=Exception("message too old")
    )

    with patch("src.infra.db.get_db", new=AsyncMock(return_value=MagicMock())), \
         patch("src.infra.db.get_task", new=AsyncMock(return_value={})), \
         patch.object(tb, "update_task", new=AsyncMock()) as upd:
        await iface._handle_artifact_confirm_callback(update, MagicMock())

    upd.assert_awaited_once()                          # task completed
    iface.app.bot.send_message.assert_awaited_once()   # confirmation still sent
    assert 42 not in iface._pending_action             # pending cleared


@pytest.mark.asyncio
async def test_surface_choice_gives_instant_toast():
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._pending_action = {}
    update = MagicMock()
    update.effective_chat.id = 42
    update.callback_query.data = "sc:7:2:mobile,web"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()

    with patch("mr_roboto.surfaces_persist.write_surfaces_json",
               new=AsyncMock(return_value={"surfaces": ["mobile", "web"]})), \
         patch("src.infra.db.update_task", new=AsyncMock()):
        await iface._handle_surface_choice(update, MagicMock())

    update.callback_query.answer.assert_awaited_once()
    assert _toast_text(update.callback_query.answer)


@pytest.mark.asyncio
async def test_review_halt_accept_gives_instant_toast():
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    update = MagicMock()
    update.effective_chat.id = 42
    update.callback_query.data = "rr:accept:7:55"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()

    with patch("general_beckman.review_routing._repend_producer", new=AsyncMock()), \
         patch("src.infra.db.update_task", new=AsyncMock()), \
         patch("src.infra.db.record_action_event", new=AsyncMock(return_value=1)), \
         patch("general_beckman.enqueue", new=AsyncMock(return_value=99)):
        await iface._handle_review_halt(update, MagicMock())

    update.callback_query.answer.assert_awaited_once()
    assert _toast_text(update.callback_query.answer)
