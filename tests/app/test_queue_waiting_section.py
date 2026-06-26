"""/queue surfaces waiting_human tasks in their own '⏸ Waiting on you' section
with a one-tap [Open] button per task that re-surfaces the original card.
Founder report 2026-06-26: parked tasks were invisible on the queue.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_format_waiting_human_section_lists_each_task():
    from src.app.telegram_bot import TelegramInterface
    waiting = [
        {"id": 524, "title": "charter confirm"},
        {"id": 531, "title": "platform pick"},
    ]
    out = TelegramInterface._format_waiting_human_section(waiting)
    assert "Waiting on you (2)" in out
    assert "#524 charter confirm" in out
    assert "#531 platform pick" in out


def test_format_waiting_human_section_empty_is_blank():
    from src.app.telegram_bot import TelegramInterface
    assert TelegramInterface._format_waiting_human_section([]) == ""


def test_build_waiting_open_markup_one_button_per_task():
    from src.app.telegram_bot import TelegramInterface
    markup = TelegramInterface._build_waiting_open_markup([
        {"id": 524, "title": "x"}, {"id": 531, "title": "y"},
    ])
    datas = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert datas == ["wq:open:524", "wq:open:531"]


def test_build_waiting_open_markup_empty_is_none():
    from src.app.telegram_bot import TelegramInterface
    assert TelegramInterface._build_waiting_open_markup([]) is None


@pytest.mark.asyncio
async def test_queue_open_resends_clarification_card():
    """[Open] on a parked clarification re-runs the clarify resend so the
    founder gets the original interactive card back."""
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    update = MagicMock()
    update.callback_query.data = "wq:open:524"
    update.callback_query.answer = AsyncMock()
    task = {"id": 524, "status": "waiting_human",
            "agent_type": "planner", "context": "{}"}

    with patch("src.infra.db.get_task", new=AsyncMock(return_value=task)), \
         patch("general_beckman.sweep._resend_clarification",
               new=AsyncMock()) as rs:
        await iface._handle_queue_open(update, MagicMock())

    rs.assert_awaited_once_with(524)
    update.callback_query.answer.assert_awaited_once()
    args, kwargs = update.callback_query.answer.call_args
    assert (args[0] if args else kwargs.get("text", ""))  # toast feedback


@pytest.mark.asyncio
async def test_queue_open_review_halt_resurfaces_card_directly():
    """A parked reviewer halt re-surfaces its founder-halt card directly
    (instant) rather than via the mechanical resend path."""
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface.resurface_review_halt = AsyncMock(return_value=True)
    update = MagicMock()
    update.callback_query.data = "wq:open:55"
    update.callback_query.answer = AsyncMock()
    task = {"id": 55, "status": "waiting_human",
            "agent_type": "reviewer", "context": "{}"}

    with patch("src.infra.db.get_task", new=AsyncMock(return_value=task)), \
         patch("general_beckman.sweep._resend_clarification",
               new=AsyncMock()) as rs:
        await iface._handle_queue_open(update, MagicMock())

    iface.resurface_review_halt.assert_awaited_once()
    rs.assert_not_awaited()


@pytest.mark.asyncio
async def test_queue_open_no_longer_waiting_is_noop():
    """If the task already left waiting_human (founder answered elsewhere),
    [Open] tells them and re-sends nothing."""
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    update = MagicMock()
    update.callback_query.data = "wq:open:524"
    update.callback_query.answer = AsyncMock()
    task = {"id": 524, "status": "completed", "context": "{}"}

    with patch("src.infra.db.get_task", new=AsyncMock(return_value=task)), \
         patch("general_beckman.sweep._resend_clarification",
               new=AsyncMock()) as rs:
        await iface._handle_queue_open(update, MagicMock())

    rs.assert_not_awaited()
    update.callback_query.answer.assert_awaited_once()
    args, kwargs = update.callback_query.answer.call_args
    assert (args[0] if args else kwargs.get("text", ""))  # told something
