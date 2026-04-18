# notifications.py
"""
Notification handlers with fallback chain.

Delivery chain (tried in order):
  1. Telegram   — if TELEGRAM_ADMIN_CHAT_ID is configured (async)
  2. File log   — always writes to logs/notifications.log

Logging handlers for the Python logging system:
  TelegramAlertHandler  — immediate push to admin chat (ERROR+)
"""

import asyncio
import json
import logging
import logging.handlers
import os
import shutil
import threading
import time
from datetime import datetime, timezone

_pending_notification_tasks: set[asyncio.Task] = set()

from .times import db_now

import requests

# Import lazily to avoid circular imports at module load time
def _cfg():
    from src.app import config as c
    return c


# ─── File-only fallback logger (used by handlers to report their own errors) ──

_handler_error_logger = logging.getLogger("infra.notifications._internal")
_handler_error_logger.propagate = False  # never re-enter ourselves


def _safe_rotator(source: str, dest: str) -> None:
    """Rename with retry — survives Dropbox/antivirus holding the file on Windows."""
    for attempt in range(5):
        try:
            if os.path.exists(dest):
                os.remove(dest)
            os.rename(source, dest)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    try:
        shutil.copy2(source, dest)
        with open(source, "w"):
            pass
    except Exception:
        pass


def _attach_file_sink():
    """Attach a bare file handler to the internal logger (called once)."""
    os.makedirs("logs", exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        "logs/notification_errors.log", maxBytes=5_000_000,
        backupCount=2, encoding="utf-8")
    fh.rotator = _safe_rotator
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _handler_error_logger.addHandler(fh)
    _handler_error_logger.setLevel(logging.WARNING)


_attach_file_sink()


# ─── File-based notification log ──────────────────────────────────────────────

_notification_logger = logging.getLogger("infra.notifications.file")
_notification_logger.propagate = False


def _attach_notification_file_sink():
    """Attach a JSON-line file handler to the notification file logger."""
    os.makedirs("logs", exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        "logs/notifications.log", maxBytes=10_000_000,
        backupCount=3, encoding="utf-8")
    fh.rotator = _safe_rotator
    fh.setFormatter(logging.Formatter("%(message)s"))  # raw JSON lines
    _notification_logger.addHandler(fh)
    _notification_logger.setLevel(logging.INFO)


_attach_notification_file_sink()


def _write_notification_to_file(
    title: str,
    message: str,
    priority: int = 3,
    level: str = "info",
) -> None:
    """Write a single JSON-line notification to logs/notifications.log."""
    entry = {
        "timestamp": db_now(),
        "level": level,
        "title": title,
        "message": message,
        "priority": priority,
    }
    _notification_logger.info(json.dumps(entry, ensure_ascii=False))


# ─── Telegram DM notification ────────────────────────────────────────────────

async def notify_telegram(
    title: str,
    message: str,
    priority: int = 3,
) -> bool:
    """
    Send a notification directly to the admin via Telegram bot.

    Returns True if the message was sent, False otherwise.
    Requires a running bot Application with a valid bot token.
    Never raises — failures are logged internally.
    """
    try:
        cfg = _cfg()
        chat_id = cfg.TELEGRAM_ADMIN_CHAT_ID
        if not chat_id:
            return False

        from telegram import Bot
        token = cfg.TELEGRAM_BOT_TOKEN
        if not token:
            return False

        bot = Bot(token=token)
        priority_icons = {5: "🔴", 4: "🟠", 3: "🟡", 2: "🔵", 1: "⚪"}
        icon = priority_icons.get(priority, "🟡")
        text = f"{icon} *{title}*\n{message}"

        await bot.send_message(
            chat_id=int(chat_id),
            text=text,
            parse_mode="Markdown",
        )
        return True
    except Exception as exc:
        _handler_error_logger.warning(
            "notify_telegram failed: %s", exc,
        )
        return False


# ─── Unified notify() — fallback chain ───────────────────────────────────────

_standard_logger = logging.getLogger("infra.notifications")


def notify(
    title: str,
    message: str,
    priority: int = 3,
    tags: list[str] | None = None,
    level: str = "info",
) -> str:
    """
    Send a notification through the fallback chain.
    Tries: 1. Telegram  2. File log (always)
    Returns "telegram" or "file". Never raises.
    """
    _write_notification_to_file(title, message, priority, level)
    _standard_logger.info("[notification] %s: %s", title, message)

    cfg = _cfg()
    if cfg.TELEGRAM_ADMIN_CHAT_ID:
        try:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(notify_telegram(title, message, priority))
                _pending_notification_tasks.add(task)
                task.add_done_callback(_pending_notification_tasks.discard)
            except RuntimeError:
                asyncio.run(notify_telegram(title, message, priority))
            return "telegram"
        except Exception as exc:
            _handler_error_logger.warning("notify: telegram failed: %s", exc)

    return "file"


# ─── TelegramAlertHandler — immediate, ERROR+ ────────────────────────────────

class TelegramAlertHandler(logging.Handler):
    """Immediately sends every ERROR/CRITICAL record to admin via Telegram."""

    PRIORITY_MAP = {
        logging.CRITICAL: 5,
        logging.ERROR: 4,
    }

    _COOLDOWN_SECS = 30
    _last_sent: float = 0

    def __init__(self):
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            import time
            now = time.monotonic()
            if now - TelegramAlertHandler._last_sent < self._COOLDOWN_SECS:
                return
            TelegramAlertHandler._last_sent = now

            cfg = _cfg()
            token = cfg.TELEGRAM_BOT_TOKEN
            chat_id = cfg.TELEGRAM_ADMIN_CHAT_ID
            if not token or not chat_id:
                return

            priority = self.PRIORITY_MAP.get(record.levelno, 4)
            icons = {5: "\U0001f534", 4: "\U0001f7e0"}
            icon = icons.get(priority, "\U0001f7e0")

            component = record.name
            title = f"{icon} [{record.levelname}] {component}"

            task_id = getattr(record, "task_id", None)
            if task_id:
                title += f" task={task_id}"

            body = self.format(record)
            if record.exc_info:
                import traceback
                body += "\n" + traceback.format_exception(*record.exc_info)[-1].strip()

            text = f"*{title}*\n```\n{body[:3500]}\n```"

            threading.Thread(
                target=self._send_sync,
                args=(token, chat_id, text),
                daemon=True,
            ).start()
        except Exception as exc:
            _handler_error_logger.error("TelegramAlertHandler.emit failed: %s", exc)

    @staticmethod
    def _send_sync(token: str, chat_id: str, text: str) -> None:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=5,
            )
        except Exception as exc:
            _handler_error_logger.error("TelegramAlertHandler._send_sync failed: %s", exc)


# ─── Shopping Notification Helpers ────────────────────────────────────────────

async def send_price_drop_alert(
    user_id: int,
    product: str,
    old_price: float,
    new_price: float,
    url: str | None = None,
) -> None:
    """Send a price drop notification via Telegram."""
    pct = round((1 - new_price / old_price) * 100) if old_price > 0 else 0
    title = f"Price Drop: {product}"
    body = f"{product}: {old_price:.2f} TL -> {new_price:.2f} TL ({pct}% off)"
    if url:
        body += f"\n{url}"
    try:
        from src.app.telegram_bot import _send_telegram_message
        await _send_telegram_message(user_id, f"\U0001f4c9 *{title}*\n{body}")
    except Exception:
        pass


async def send_deal_alert(
    user_id: int,
    product: str,
    discount_pct: float,
    url: str | None = None,
) -> None:
    """Send a deal/discount notification via Telegram."""
    title = f"Deal Alert: {product}"
    body = f"{product}: {discount_pct:.0f}% discount found!"
    if url:
        body += f"\n{url}"
    try:
        from src.app.telegram_bot import _send_telegram_message
        await _send_telegram_message(user_id, f"\U0001f3f7\ufe0f *{title}*\n{body}")
    except Exception:
        pass


async def send_back_in_stock_alert(
    user_id: int,
    product: str,
    price: float,
    url: str | None = None,
) -> None:
    """Send a back-in-stock notification via Telegram."""
    title = f"Back in Stock: {product}"
    body = f"{product} is back in stock at {price:.2f} TL"
    if url:
        body += f"\n{url}"
    try:
        from src.app.telegram_bot import _send_telegram_message
        await _send_telegram_message(user_id, f"\U0001f4e6 *{title}*\n{body}")
    except Exception:
        pass
