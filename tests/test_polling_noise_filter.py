"""WS-2 (handoff 2026-05-25) — PollingNetworkNoiseFilter.

python-telegram-bot's Updater logs every transient getUpdates DNS/connect blip
(``[Errno 11001] getaddrinfo failed``, wrapped as telegram.error.NetworkError) at
ERROR with a full traceback. The bot auto-retries and recovers, but the ERROR
spammed the 🟠 Telegram alert feed (TelegramAlertHandler, ERROR+). The filter
rewrites those records in place to a single WARNING line so the handler's
ERROR-level gate skips them; genuine non-network Updater errors stay ERROR.
"""
import logging
import sys
import unittest

from telegram.error import NetworkError, TimedOut

from src.infra.notifications import (
    PollingNetworkNoiseFilter,
    install_polling_noise_filter,
)


def _record(name: str, exc: BaseException | None) -> logging.LogRecord:
    exc_info = None
    if exc is not None:
        try:
            raise exc
        except BaseException:
            exc_info = sys.exc_info()
    return logging.LogRecord(
        name=name,
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="Exception happened while polling for updates.",
        args=(),
        exc_info=exc_info,
    )


class TestPollingNetworkNoiseFilter(unittest.TestCase):

    def setUp(self):
        self.f = PollingNetworkNoiseFilter()

    def test_network_error_downgraded_to_warning(self):
        rec = _record("telegram.ext.Updater",
                      NetworkError("httpx.ConnectError: [Errno 11001] getaddrinfo failed"))
        kept = self.f.filter(rec)
        self.assertTrue(kept)  # never drop, only downgrade
        self.assertEqual(rec.levelno, logging.WARNING)
        self.assertEqual(rec.levelname, "WARNING")
        self.assertIsNone(rec.exc_info)
        self.assertIsNone(rec.exc_text)
        # Below ERROR → TelegramAlertHandler (level=ERROR) skips it.
        self.assertLess(rec.levelno, logging.ERROR)
        # Message still names the cause as one line.
        self.assertIn("getaddrinfo failed", rec.getMessage())

    def test_timeout_subclass_also_downgraded(self):
        rec = _record("telegram.ext.Updater", TimedOut("Timed out"))
        self.f.filter(rec)
        self.assertEqual(rec.levelno, logging.WARNING)

    def test_non_network_error_stays_error(self):
        rec = _record("telegram.ext.Updater", ValueError("real bug"))
        self.f.filter(rec)
        self.assertEqual(rec.levelno, logging.ERROR)
        self.assertIsNotNone(rec.exc_info)

    def test_other_logger_untouched(self):
        rec = _record("kutai.core.orchestrator", NetworkError("blip"))
        self.f.filter(rec)
        self.assertEqual(rec.levelno, logging.ERROR)

    def test_no_exc_info_untouched(self):
        rec = _record("telegram.ext.Updater", None)
        self.f.filter(rec)
        self.assertEqual(rec.levelno, logging.ERROR)

    def test_install_is_idempotent(self):
        lg = logging.getLogger("telegram.ext.Updater")
        before = [f for f in lg.filters if isinstance(f, PollingNetworkNoiseFilter)]
        for f in before:
            lg.removeFilter(f)
        install_polling_noise_filter()
        install_polling_noise_filter()
        count = sum(isinstance(f, PollingNetworkNoiseFilter) for f in lg.filters)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
