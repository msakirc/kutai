import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_surface_choice_writes_surfaces_and_completes():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}
    chat_id = 42
    iface._pending_action[chat_id] = {
        "kind": "surface_choice", "mission_id": 7, "task_id": 2,
    }
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.data = "sc:7:2:mobile,web"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()

    written = {"surfaces": ["mobile", "web"], "primary_surface": "mobile"}
    with patch("mr_roboto.surfaces_persist.write_surfaces_json",
               new=AsyncMock(return_value=written)) as wsj, \
         patch("src.infra.db.update_task", new=AsyncMock()) as upd:
        await iface._handle_surface_choice(update, MagicMock())

    wsj.assert_awaited_once()
    assert wsj.call_args.kwargs["mission_id"] == 7
    assert wsj.call_args.kwargs["option_label"] == "mobile + web"
    upd.assert_awaited_once_with(2, status="completed")
    assert chat_id not in iface._pending_action


@pytest.mark.asyncio
async def test_surface_choice_malformed_callback_noop():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._pending_action = {}
    update = MagicMock()
    update.effective_chat.id = 1
    update.callback_query.data = "sc:nope"
    update.callback_query.answer = AsyncMock()

    with patch("mr_roboto.surfaces_persist.write_surfaces_json",
               new=AsyncMock()) as wsj, \
         patch("src.infra.db.update_task", new=AsyncMock()) as upd:
        await iface._handle_surface_choice(update, MagicMock())

    wsj.assert_not_awaited()
    upd.assert_not_awaited()
    update.callback_query.answer.assert_awaited_once()
