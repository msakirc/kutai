"""Verify ntfy handlers are removed and TelegramAlertHandler is wired in."""
import unittest


class TestNoNtfyHandlers(unittest.TestCase):

    def test_ntfy_alert_handler_removed(self):
        import src.infra.notifications as n
        self.assertFalse(hasattr(n, "NtfyAlertHandler"), "NtfyAlertHandler still exists")

    def test_ntfy_batch_handler_removed(self):
        import src.infra.notifications as n
        self.assertFalse(hasattr(n, "NtfyBatchHandler"), "NtfyBatchHandler still exists")

    def test_telegram_alert_handler_exists(self):
        from src.infra.notifications import TelegramAlertHandler
        self.assertTrue(callable(TelegramAlertHandler))


if __name__ == "__main__":
    unittest.main()
