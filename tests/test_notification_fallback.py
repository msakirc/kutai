"""Tests for the notification fallback chain (Telegram -> file log)."""
import asyncio
import json
import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestFileNotificationFallback(unittest.TestCase):

    @patch("src.infra.notifications._cfg")
    def test_notify_writes_to_file_when_no_telegram(self, mock_cfg):
        cfg = MagicMock()
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg
        from src.infra.notifications import notify
        result = notify("Test Title", "Test message body", priority=3)
        self.assertEqual(result, "file")

    @patch("src.infra.notifications._cfg")
    def test_notification_log_is_valid_json(self, mock_cfg):
        cfg = MagicMock()
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg
        from src.infra.notifications import _write_notification_to_file
        _write_notification_to_file("JSON Test", "Check format", priority=2, level="warning")
        log_path = os.path.join("logs", "notifications.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertTrue(len(lines) > 0)
            entry = json.loads(lines[-1].strip())
            self.assertIn("timestamp", entry)
            self.assertIn("title", entry)
            self.assertEqual(entry["title"], "JSON Test")

    @patch("src.infra.notifications._cfg")
    def test_notify_does_not_crash_with_no_services(self, mock_cfg):
        cfg = MagicMock()
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg
        from src.infra.notifications import notify
        result = notify("No services", "Still works", priority=5, level="error")
        self.assertEqual(result, "file")


class TestTelegramFallback(unittest.TestCase):

    @patch("src.infra.notifications._cfg")
    def test_notify_uses_telegram_when_configured(self, mock_cfg):
        cfg = MagicMock()
        cfg.TELEGRAM_ADMIN_CHAT_ID = "12345"
        cfg.TELEGRAM_BOT_TOKEN = "fake:token"
        mock_cfg.return_value = cfg
        from src.infra.notifications import notify
        with patch("src.infra.notifications.notify_telegram", new_callable=AsyncMock, return_value=True):
            result = notify("TG Test", "Goes to Telegram", priority=3)
            self.assertEqual(result, "telegram")


class TestNotifyTelegram(unittest.TestCase):

    @patch("src.infra.notifications._cfg")
    def test_notify_telegram_returns_false_when_no_chat_id(self, mock_cfg):
        cfg = MagicMock()
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg
        from src.infra.notifications import notify_telegram
        result = run_async(notify_telegram("Test", "Body"))
        self.assertFalse(result)

    @patch("src.infra.notifications._cfg")
    def test_notify_telegram_returns_false_when_no_token(self, mock_cfg):
        cfg = MagicMock()
        cfg.TELEGRAM_ADMIN_CHAT_ID = "12345"
        cfg.TELEGRAM_BOT_TOKEN = ""
        mock_cfg.return_value = cfg
        from src.infra.notifications import notify_telegram
        result = run_async(notify_telegram("Test", "Body"))
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
