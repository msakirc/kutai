"""Tests for Telegram workflow commands: /wfstatus, /product, /resume.

The telegram library is not installed in the test environment, so we mock
the entire module hierarchy before importing TelegramInterface.
"""

import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Mock the telegram package so we can import telegram_bot without installing
# python-telegram-bot.
# ---------------------------------------------------------------------------
_telegram_mock = MagicMock()
_telegram_ext_mock = MagicMock()

# Provide the symbols that telegram_bot.py imports at module level.
_telegram_mock.Update = MagicMock
_telegram_mock.InlineKeyboardButton = MagicMock
_telegram_mock.InlineKeyboardMarkup = MagicMock
_telegram_ext_mock.Application = MagicMock()
_telegram_ext_mock.CommandHandler = MagicMock
_telegram_ext_mock.MessageHandler = MagicMock
_telegram_ext_mock.CallbackQueryHandler = MagicMock
_telegram_ext_mock.filters = MagicMock()
_telegram_ext_mock.ContextTypes = MagicMock()

sys.modules.setdefault("telegram", _telegram_mock)
sys.modules.setdefault("telegram.ext", _telegram_ext_mock)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_update_context(args=None):
    """Create mock Telegram Update and Context objects."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.chat_id = 12345

    context = MagicMock()
    context.args = args or []
    return update, context


def _make_bot():
    """Create a TelegramInterface with mocked Application builder."""
    mock_app = MagicMock()
    mock_app.add_handler = MagicMock()

    mock_builder = MagicMock()
    mock_builder.token.return_value = mock_builder
    mock_builder.build.return_value = mock_app

    with patch("src.app.telegram_bot.Application") as mock_cls:
        mock_cls.builder.return_value = mock_builder
        from src.app.telegram_bot import TelegramInterface
        bot = TelegramInterface()
    return bot


# ── /wfstatus tests ──────────────────────────────────────────────────────────


class TestWfstatusCommand(unittest.TestCase):
    """Tests for /wfstatus command."""

    def test_no_args_shows_usage(self):
        bot = _make_bot()
        update, context = _make_update_context(args=[])
        run_async(bot.cmd_wfstatus(update, context))

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Usage", msg)

    def test_non_numeric_id(self):
        bot = _make_bot()
        update, context = _make_update_context(args=["abc"])
        run_async(bot.cmd_wfstatus(update, context))

        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("number", msg.lower())

    @patch("src.app.telegram_bot.get_mission", new_callable=AsyncMock)
    @patch("src.app.telegram_bot.get_tasks_for_mission", new_callable=AsyncMock)
    def test_mission_not_found(self, mock_tasks, mock_mission):
        bot = _make_bot()
        update, context = _make_update_context(args=["999"])
        mock_mission.return_value = None

        run_async(bot.cmd_wfstatus(update, context))

        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("not found", msg)

    @patch("src.app.telegram_bot.get_mission", new_callable=AsyncMock)
    @patch("src.app.telegram_bot.get_tasks_for_mission", new_callable=AsyncMock)
    def test_no_tasks(self, mock_tasks, mock_mission):
        bot = _make_bot()
        update, context = _make_update_context(args=["1"])
        mock_mission.return_value = {"id": 1, "title": "T", "context": "{}"}
        mock_tasks.return_value = []

        run_async(bot.cmd_wfstatus(update, context))

        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("No tasks found", msg)

    @patch("src.app.telegram_bot.get_mission", new_callable=AsyncMock)
    @patch("src.app.telegram_bot.get_tasks_for_mission", new_callable=AsyncMock)
    def test_valid_mission_returns_status(self, mock_tasks, mock_mission):
        bot = _make_bot()
        update, context = _make_update_context(args=["1"])
        mock_mission.return_value = {
            "id": 1,
            "title": "Test",
            "context": '{"workflow_name": "idea_to_product_v2"}',
        }
        mock_tasks.return_value = [
            {"id": 10, "status": "completed",
             "context": '{"workflow_phase": "phase_0"}'},
            {"id": 11, "status": "pending",
             "context": '{"workflow_phase": "phase_1"}'},
        ]

        run_async(bot.cmd_wfstatus(update, context))

        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Workflow Status", msg)
        self.assertIn("mission #1", msg)


# ── /product tests ───────────────────────────────────────────────────────────


class TestProductCommand(unittest.TestCase):
    """Tests for /product command."""

    def test_no_args_shows_usage(self):
        bot = _make_bot()
        update, context = _make_update_context(args=[])
        run_async(bot.cmd_product(update, context))

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Usage", msg)

    @patch("src.workflows.engine.runner.WorkflowRunner.start",
           new_callable=AsyncMock)
    def test_starts_workflow(self, mock_start):
        bot = _make_bot()
        update, context = _make_update_context(args=["Build", "a", "chat", "app"])
        mock_start.return_value = 42

        run_async(bot.cmd_product(update, context))

        # Two messages: "starting..." then success
        self.assertEqual(update.message.reply_text.call_count, 2)
        final_msg = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn("42", final_msg)
        self.assertIn("/wfstatus 42", final_msg)

        mock_start.assert_called_once()
        call_kwargs = mock_start.call_args[1]
        self.assertEqual(call_kwargs["initial_input"]["raw_idea"],
                         "Build a chat app")

    @patch("src.workflows.engine.runner.WorkflowRunner.start",
           new_callable=AsyncMock)
    def test_handles_error(self, mock_start):
        bot = _make_bot()
        update, context = _make_update_context(args=["Bad", "idea"])
        mock_start.side_effect = RuntimeError("Workflow not found")

        run_async(bot.cmd_product(update, context))

        final_msg = update.message.reply_text.call_args_list[-1][0][0]
        self.assertIn("Failed to start", final_msg)


# ── /resume tests ────────────────────────────────────────────────────────────


class TestResumeCommand(unittest.TestCase):
    """Tests for /resume command."""

    def test_no_args_shows_usage(self):
        bot = _make_bot()
        update, context = _make_update_context(args=[])
        run_async(bot.cmd_resume(update, context))

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Usage", msg)

    def test_non_numeric_id(self):
        bot = _make_bot()
        update, context = _make_update_context(args=["abc"])
        run_async(bot.cmd_resume(update, context))

        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("number", msg.lower())

    @patch("src.workflows.engine.runner.WorkflowRunner.resume",
           new_callable=AsyncMock)
    def test_valid_mission_resumes(self, mock_resume):
        bot = _make_bot()
        update, context = _make_update_context(args=["5"])
        mock_resume.return_value = 5

        run_async(bot.cmd_resume(update, context))

        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("resumed", msg.lower())
        self.assertIn("5", msg)
        self.assertIn("/wfstatus 5", msg)

    @patch("src.workflows.engine.runner.WorkflowRunner.resume",
           new_callable=AsyncMock)
    def test_mission_not_found(self, mock_resume):
        bot = _make_bot()
        update, context = _make_update_context(args=["999"])
        mock_resume.side_effect = ValueError("Mission #999 not found")

        run_async(bot.cmd_resume(update, context))

        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Cannot resume", msg)

    @patch("src.workflows.engine.runner.WorkflowRunner.resume",
           new_callable=AsyncMock)
    def test_no_resumable_tasks(self, mock_resume):
        bot = _make_bot()
        update, context = _make_update_context(args=["3"])
        mock_resume.side_effect = ValueError(
            "Mission #3 has no failed or paused tasks to resume"
        )

        run_async(bot.cmd_resume(update, context))

        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Cannot resume", msg)


if __name__ == "__main__":
    unittest.main()
