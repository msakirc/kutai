"""Tests for the notification fallback chain (file log, Telegram, ntfy)."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, AsyncMock


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestFileNotificationFallback(unittest.TestCase):
    """When NTFY_URL is not set, notifications go to file."""

    @patch("src.infra.notifications._cfg")
    def test_notify_writes_to_file_when_no_ntfy(self, mock_cfg):
        """notify() returns 'file' and writes to log when ntfy is not configured."""
        cfg = MagicMock()
        cfg.NTFY_URL = ""
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg

        from src.infra.notifications import notify

        result = notify("Test Title", "Test message body", priority=3)
        self.assertEqual(result, "file")

    @patch("src.infra.notifications._cfg")
    def test_notification_log_is_valid_json(self, mock_cfg):
        """Each line in the notification log must be valid JSON."""
        cfg = MagicMock()
        cfg.NTFY_URL = ""
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg

        from src.infra.notifications import _write_notification_to_file

        # Write a test notification
        _write_notification_to_file("JSON Test", "Check format", priority=2, level="warning")

        # Read the log file and verify last line is valid JSON
        log_path = os.path.join("logs", "notifications.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            self.assertTrue(len(lines) > 0, "Notification log should have entries")

            last_line = lines[-1].strip()
            entry = json.loads(last_line)  # Raises if not valid JSON

            self.assertIn("timestamp", entry)
            self.assertIn("level", entry)
            self.assertIn("title", entry)
            self.assertIn("message", entry)
            self.assertIn("priority", entry)
            self.assertEqual(entry["title"], "JSON Test")
            self.assertEqual(entry["message"], "Check format")
            self.assertEqual(entry["level"], "warning")
            self.assertEqual(entry["priority"], 2)

    @patch("src.infra.notifications._cfg")
    def test_notify_does_not_crash_with_no_services(self, mock_cfg):
        """notify() should never raise, even with nothing configured."""
        cfg = MagicMock()
        cfg.NTFY_URL = ""
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg

        from src.infra.notifications import notify

        # Should not raise
        result = notify("No services", "This should still work", priority=5, level="error")
        self.assertEqual(result, "file")


class TestNtfyFallback(unittest.TestCase):
    """When NTFY_URL is set, ntfy is tried first."""

    @patch("src.infra.notifications.requests.post")
    @patch("src.infra.notifications._cfg")
    def test_notify_uses_ntfy_when_configured(self, mock_cfg, mock_post):
        cfg = MagicMock()
        cfg.NTFY_URL = "http://localhost:8083"
        cfg.NTFY_USER = ""
        cfg.NTFY_PASS = ""
        cfg.NTFY_TOPIC_LOGS = "kutai-logs"
        cfg.NTFY_TOPIC_ERRORS = "kutai-errors"
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        mock_cfg.return_value = cfg

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        from src.infra.notifications import notify

        result = notify("Ntfy Test", "Goes to ntfy", priority=3)
        self.assertEqual(result, "ntfy")
        mock_post.assert_called_once()


class TestTelegramFallback(unittest.TestCase):
    """When ntfy is not set but Telegram is, use Telegram."""

    @patch("src.infra.notifications._cfg")
    def test_notify_uses_telegram_when_ntfy_missing(self, mock_cfg):
        cfg = MagicMock()
        cfg.NTFY_URL = ""
        cfg.TELEGRAM_ADMIN_CHAT_ID = "12345"
        cfg.TELEGRAM_BOT_TOKEN = "fake:token"
        mock_cfg.return_value = cfg

        from src.infra.notifications import notify

        # Patch notify_telegram to avoid real Telegram API calls
        with patch("src.infra.notifications.notify_telegram", new_callable=AsyncMock, return_value=True) as mock_tg:
            result = notify("TG Test", "Goes to Telegram", priority=3)
            self.assertEqual(result, "telegram")


class TestNotifyTelegram(unittest.TestCase):
    """Unit tests for notify_telegram async function."""

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
