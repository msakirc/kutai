"""Tests for TelegramAlertHandler — logging.Handler that sends ERROR+ to Telegram."""
import asyncio
import logging
import unittest
from unittest.mock import patch, MagicMock, AsyncMock


class TestTelegramAlertHandler(unittest.TestCase):

    @patch("src.infra.notifications._cfg")
    def test_handler_level_is_error(self, mock_cfg):
        from src.infra.notifications import TelegramAlertHandler
        handler = TelegramAlertHandler()
        self.assertEqual(handler.level, logging.ERROR)

    @patch("src.infra.notifications._cfg")
    def test_handler_formats_error_record(self, mock_cfg):
        """emit() should fire a background thread to send the alert."""
        cfg = MagicMock()
        cfg.TELEGRAM_BOT_TOKEN = "fake:token"
        cfg.TELEGRAM_ADMIN_CHAT_ID = "12345"
        mock_cfg.return_value = cfg

        from src.infra.notifications import TelegramAlertHandler
        handler = TelegramAlertHandler()

        record = logging.LogRecord(
            name="test.component",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Something broke",
            args=(),
            exc_info=None,
        )

        with patch("threading.Thread") as mock_thread:
            mock_instance = MagicMock()
            mock_thread.return_value = mock_instance
            handler.emit(record)
            mock_thread.assert_called_once()
            mock_instance.start.assert_called_once()

    @patch("src.infra.notifications._cfg")
    def test_handler_skips_when_no_token(self, mock_cfg):
        """emit() should not crash when Telegram is not configured."""
        cfg = MagicMock()
        cfg.TELEGRAM_BOT_TOKEN = ""
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg

        from src.infra.notifications import TelegramAlertHandler
        handler = TelegramAlertHandler()

        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="", lineno=0, msg="err", args=(), exc_info=None,
        )
        handler.emit(record)


if __name__ == "__main__":
    unittest.main()
