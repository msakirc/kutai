"""Verify ntfy is fully removed from the codebase."""
import importlib
import unittest


class TestNtfyRemoved(unittest.TestCase):

    def _read_source(self, module_name: str) -> str:
        source = importlib.util.find_spec(module_name).origin
        with open(source, encoding="utf-8") as f:
            return f.read()

    def test_notifications_no_send_ntfy(self):
        text = self._read_source("src.infra.notifications")
        self.assertNotIn("def send_ntfy", text)

    def test_config_no_ntfy_vars(self):
        text = self._read_source("src.app.config")
        self.assertNotIn("NTFY_URL", text)
        self.assertNotIn("NTFY_USER", text)
        self.assertNotIn("NTFY_PASS", text)
        self.assertNotIn("NTFY_TOPIC", text)

    def test_alerting_no_ntfy(self):
        text = self._read_source("src.infra.alerting")
        self.assertNotIn("send_ntfy", text)
        self.assertNotIn("NTFY_TOPIC", text)

    def test_run_no_ntfy(self):
        text = self._read_source("src.app.run")
        self.assertNotIn("send_ntfy", text)
        self.assertNotIn("NTFY_TOPIC", text)

    def test_notify_returns_telegram_or_file(self):
        from unittest.mock import patch, MagicMock
        cfg = MagicMock()
        cfg.NTFY_URL = ""
        cfg.TELEGRAM_ADMIN_CHAT_ID = ""
        with patch("src.infra.notifications._cfg", return_value=cfg):
            from src.infra.notifications import notify
            result = notify("test", "test")
            self.assertIn(result, ("telegram", "file"))


if __name__ == "__main__":
    unittest.main()
