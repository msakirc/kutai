# Replace ntfy with Telegram Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ntfy dependency entirely, use Telegram for ERROR+ alerts, add `/logs` command to both bot and wrapper for on-demand log viewing.

**Architecture:** Telegram replaces ntfy as the sole push notification channel (ERROR+ only). Batch log push is deleted — logs stay on disk and are pulled via `/logs` command. The wrapper handles `/logs` when KutAI is crashed, the bot handles it when running. File sink (JSON-lines) is unchanged.

**Tech Stack:** python-telegram-bot (existing), aiohttp (existing for wrapper), stdlib logging

---

### Task 1: Create TelegramAlertHandler

**Files:**
- Create: `tests/test_telegram_alert_handler.py`
- Modify: `src/infra/notifications.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_telegram_alert_handler.py`:

```python
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
        # Should not raise
        handler.emit(record)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_telegram_alert_handler.py -v`
Expected: FAIL — `ImportError: cannot import name 'TelegramAlertHandler'`

- [ ] **Step 3: Implement TelegramAlertHandler**

In `src/infra/notifications.py`, replace the `NtfyAlertHandler` class (lines 243-288) with:

```python
# ─── TelegramAlertHandler — immediate, ERROR+ ────────────────────────────────

class TelegramAlertHandler(logging.Handler):
    """Immediately sends every ERROR/CRITICAL record to admin via Telegram."""

    PRIORITY_MAP = {
        logging.CRITICAL: 5,
        logging.ERROR: 4,
    }

    # Cooldown to avoid flooding Telegram during error storms
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
            icons = {5: "🔴", 4: "🟠"}
            icon = icons.get(priority, "🟠")

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

            # Fire-and-forget on daemon thread — never block asyncio loop
            threading.Thread(
                target=self._send_sync,
                args=(token, chat_id, text),
                daemon=True,
            ).start()
        except Exception as exc:
            _handler_error_logger.error("TelegramAlertHandler.emit failed: %s", exc)

    @staticmethod
    def _send_sync(token: str, chat_id: str, text: str) -> None:
        """Synchronous Telegram sendMessage via requests."""
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=5,
            )
        except Exception as exc:
            _handler_error_logger.error("TelegramAlertHandler._send_sync failed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_telegram_alert_handler.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_telegram_alert_handler.py src/infra/notifications.py
git commit -m "feat(notifications): add TelegramAlertHandler to replace NtfyAlertHandler"
```

---

### Task 2: Remove NtfyBatchHandler and NtfyAlertHandler

**Files:**
- Modify: `src/infra/notifications.py` — delete `NtfyAlertHandler`, `NtfyBatchHandler` classes
- Modify: `src/infra/logging_config.py` — swap ntfy handlers for Telegram handler

- [ ] **Step 1: Write the failing test**

Create `tests/test_no_ntfy_handlers.py`:

```python
"""Verify ntfy handlers are removed and TelegramAlertHandler is wired in."""
import unittest


class TestNoNtfyHandlers(unittest.TestCase):

    def test_ntfy_alert_handler_removed(self):
        """NtfyAlertHandler should no longer exist in notifications module."""
        import src.infra.notifications as n
        self.assertFalse(hasattr(n, "NtfyAlertHandler"), "NtfyAlertHandler still exists")

    def test_ntfy_batch_handler_removed(self):
        """NtfyBatchHandler should no longer exist in notifications module."""
        import src.infra.notifications as n
        self.assertFalse(hasattr(n, "NtfyBatchHandler"), "NtfyBatchHandler still exists")

    def test_telegram_alert_handler_exists(self):
        from src.infra.notifications import TelegramAlertHandler
        self.assertTrue(callable(TelegramAlertHandler))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_no_ntfy_handlers.py -v`
Expected: FAIL — `NtfyAlertHandler still exists`

- [ ] **Step 3: Delete NtfyAlertHandler and NtfyBatchHandler from notifications.py**

In `src/infra/notifications.py`, delete the entire `NtfyAlertHandler` class (the old one, lines ~243-288) and the entire `NtfyBatchHandler` class (lines ~293-418). Keep `TelegramAlertHandler` which was added in Task 1.

- [ ] **Step 4: Update logging_config.py to use TelegramAlertHandler**

In `src/infra/logging_config.py`, replace sinks 3 and 4 (lines 182-200):

OLD:
```python
    # Sink 3: ntfy batch (INFO+)
    try:
        from src.infra.notifications import NtfyBatchHandler
        batch = NtfyBatchHandler()
        root.addHandler(batch)
    except Exception as e:
        logging.getLogger("infra.logging_config").warning(
            "Could not attach NtfyBatchHandler: %s", e
        )

    # Sink 4: ntfy alert (ERROR+)
    try:
        from src.infra.notifications import NtfyAlertHandler
        alert = NtfyAlertHandler()
        root.addHandler(alert)
    except Exception as e:
        logging.getLogger("infra.logging_config").warning(
            "Could not attach NtfyAlertHandler: %s", e
        )
```

NEW:
```python
    # Sink 3: Telegram alert (ERROR+)
    try:
        from src.infra.notifications import TelegramAlertHandler
        alert = TelegramAlertHandler()
        root.addHandler(alert)
    except Exception as e:
        logging.getLogger("infra.logging_config").warning(
            "Could not attach TelegramAlertHandler: %s", e
        )
```

Also update the module docstring (lines 1-12) — replace:
```python
# Log sinks (after init_logging()):
#   1. Console    — StreamHandler(stdout), DEBUG, colorized key=value
#   2. File       — RotatingFileHandler("logs/orchestrator.jsonl"), DEBUG, JSON-lines
#   3. ntfy batch — NtfyBatchHandler, INFO, flushed every 30s
#   4. ntfy alert — NtfyAlertHandler, ERROR, immediate push
```
With:
```python
# Log sinks (after init_logging()):
#   1. Console    — StreamHandler(stdout), DEBUG, colorized key=value
#   2. File       — RotatingFileHandler("logs/orchestrator.jsonl"), DEBUG, JSON-lines
#   3. Telegram   — TelegramAlertHandler, ERROR, immediate push to admin
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_no_ntfy_handlers.py tests/test_telegram_alert_handler.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/infra/notifications.py src/infra/logging_config.py tests/test_no_ntfy_handlers.py
git commit -m "refactor(notifications): remove ntfy handlers, wire TelegramAlertHandler"
```

---

### Task 3: Remove ntfy from notifications.py, alerting.py, run.py, config.py

**Files:**
- Modify: `src/infra/notifications.py` — delete `send_ntfy()`, remove ntfy from `notify()` fallback chain
- Modify: `src/infra/alerting.py` — remove ntfy path from `_send_alert()`
- Modify: `src/app/run.py` — remove ntfy send attempts on startup failure
- Modify: `src/app/config.py` — remove NTFY_* config vars

- [ ] **Step 1: Write the failing test**

Create `tests/test_ntfy_removed.py`:

```python
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
        """notify() should only return 'telegram' or 'file', never 'ntfy'."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ntfy_removed.py -v`
Expected: FAIL — `send_ntfy` still exists

- [ ] **Step 3: Remove send_ntfy and ntfy from notify() in notifications.py**

In `src/infra/notifications.py`:

1. Delete the entire `send_ntfy()` function (lines 55-108).

2. Replace the `notify()` function (lines 194-238) with:

```python
def notify(
    title: str,
    message: str,
    priority: int = 3,
    tags: list[str] | None = None,
    level: str = "info",
) -> str:
    """
    Send a notification through the fallback chain.

    Tries in order:
      1. Telegram (if TELEGRAM_ADMIN_CHAT_ID is set)
      2. File log (always)

    Returns the method used: "telegram" or "file".
    Never raises.
    """
    # Always write to file log
    _write_notification_to_file(title, message, priority, level)
    _standard_logger.info("[notification] %s: %s", title, message)

    # Try Telegram
    cfg = _cfg()
    if cfg.TELEGRAM_ADMIN_CHAT_ID:
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(notify_telegram(title, message, priority))
            except RuntimeError:
                asyncio.run(notify_telegram(title, message, priority))
            return "telegram"
        except Exception as exc:
            _handler_error_logger.warning("notify: telegram failed: %s", exc)

    return "file"
```

3. In the shopping notification helpers (`send_price_drop_alert`, `send_deal_alert`, `send_back_in_stock_alert`), remove all `send_ntfy(...)` calls. They already have Telegram fallbacks. For example, `send_price_drop_alert` becomes:

```python
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
        await _send_telegram_message(user_id, f"📉 *{title}*\n{body}")
    except Exception:
        pass
```

Apply the same pattern to `send_deal_alert` and `send_back_in_stock_alert` — remove `send_ntfy()` calls, keep only the Telegram path.

4. Remove the `import requests` at line 20 (no longer needed in this module — check if anything else uses it first; `TelegramAlertHandler._send_sync` uses requests, so KEEP it).

5. Update the module docstring (lines 1-16) to remove ntfy references:

```python
# notifications.py
"""
Notification handlers with fallback chain.

Delivery chain (tried in order):
  1. Telegram   — if TELEGRAM_ADMIN_CHAT_ID is configured (async)
  2. File log   — always writes to logs/notifications.log

Logging handlers for the Python logging system:
  TelegramAlertHandler  — immediate push to admin chat (ERROR+)
"""
```

- [ ] **Step 4: Remove ntfy from alerting.py**

In `src/infra/alerting.py`, replace `_send_alert` (lines 92-121):

```python
async def _send_alert(title: str, message: str, priority: int = 3) -> None:
    """Send alert via Telegram."""
    logger.warning(f"ALERT: {title} — {message}")

    try:
        from ..infra.runtime_state import runtime_state
        if runtime_state.get("telegram_available"):
            from ..app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID
            import aiohttp
            if TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_CHAT_ID:
                icons = {5: "🔴", 4: "🟠", 3: "🟡", 2: "🔵", 1: "⚪"}
                icon = icons.get(priority, "🟡")
                text = f"{icon} *{title}*\n\n{message}"
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                async with aiohttp.ClientSession() as s:
                    await s.post(url, json={
                        "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                        "text": text,
                        "parse_mode": "Markdown",
                    }, timeout=aiohttp.ClientTimeout(total=5))
    except Exception:
        pass
```

- [ ] **Step 5: Remove ntfy from run.py**

In `src/app/run.py`, find the two `send_ntfy` blocks (around lines 293-307 and 312-319) and delete them. Replace with a simple log. For the Docker services block (lines 293-307):

```python
    if not docker_ok:
        _log.warning("Docker services unavailable — sandbox, monitoring will not work")
```

For the critical health check block (lines 312-319):

```python
    critical_ok = await startup_health_check()
    if not critical_ok:
        _log.critical("Critical health checks failed — aborting")
        sys.exit(1)
```

- [ ] **Step 6: Remove ntfy config from config.py**

In `src/app/config.py`, delete lines 19-27:

```python
# ─── Notifications (ntfy) ────────────────────────────────────────────────────

NTFY_URL = os.getenv("NTFY_URL", "")
NTFY_USER = os.getenv("NTFY_USER", "")
NTFY_PASS = os.getenv("NTFY_PASS", "")

# Two topics: errors get phone alerts, logs are browsable
NTFY_TOPIC_ERRORS = "kutai-errors"   # ERROR/CRITICAL only, phone ON
NTFY_TOPIC_LOGS   = "kutai-logs"     # INFO/WARNING/ERROR, phone OFF
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_ntfy_removed.py tests/test_notification_fallback.py tests/test_telegram_alert_handler.py -v`
Expected: `test_ntfy_removed` all pass, some `test_notification_fallback` tests need updating (next task)

- [ ] **Step 8: Commit**

```bash
git add src/infra/notifications.py src/infra/alerting.py src/app/run.py src/app/config.py tests/test_ntfy_removed.py
git commit -m "refactor: remove ntfy from notifications, alerting, run, and config"
```

---

### Task 4: Update existing notification tests

**Files:**
- Modify: `tests/test_notification_fallback.py` — remove ntfy test, update remaining tests

- [ ] **Step 1: Rewrite test_notification_fallback.py**

Replace the entire file with:

```python
"""Tests for the notification fallback chain (Telegram → file log)."""
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
    """When Telegram is not configured, notifications go to file."""

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
```

- [ ] **Step 2: Run all notification tests**

Run: `python -m pytest tests/test_notification_fallback.py tests/test_telegram_alert_handler.py tests/test_no_ntfy_handlers.py tests/test_ntfy_removed.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_notification_fallback.py
git commit -m "test: update notification tests for Telegram-only delivery"
```

---

### Task 5: Add /logs command to telegram_bot.py

**Files:**
- Create: `tests/test_logs_command.py`
- Modify: `src/app/telegram_bot.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_logs_command.py`:

```python
"""Test /logs command reads and formats orchestrator.jsonl."""
import json
import os
import tempfile
import unittest


class TestFormatLogEntries(unittest.TestCase):

    def test_format_last_n_lines(self):
        """_format_log_entries should return last N lines, most recent last."""
        from src.app.telegram_bot import _format_log_entries

        lines = [
            json.dumps({"timestamp": "2026-03-31T10:00:00", "level": "INFO", "component": "core", "message": "started"}),
            json.dumps({"timestamp": "2026-03-31T10:00:01", "level": "ERROR", "component": "agent", "message": "failed"}),
            json.dumps({"timestamp": "2026-03-31T10:00:02", "level": "INFO", "component": "core", "message": "recovered"}),
        ]

        result = _format_log_entries(lines, n=2)
        self.assertIn("ERROR", result)
        self.assertIn("recovered", result)
        self.assertNotIn("started", result)

    def test_format_empty_log(self):
        from src.app.telegram_bot import _format_log_entries
        result = _format_log_entries([], n=10)
        self.assertIn("No log entries", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_logs_command.py -v`
Expected: FAIL — `cannot import name '_format_log_entries'`

- [ ] **Step 3: Add _format_log_entries helper and /logs command**

In `src/app/telegram_bot.py`, add this module-level helper function (near the top, after imports):

```python
def _format_log_entries(lines: list[str], n: int = 20) -> str:
    """Format the last N log lines for Telegram display.

    Reads JSON-line entries and formats them as a compact, readable message.
    Used by both /logs command and the wrapper's log viewer.
    """
    if not lines:
        return "📋 No log entries found."

    last_n = lines[-n:]
    formatted = []
    for line in last_n:
        line = line.strip()
        if not line:
            continue
        try:
            entry = __import__("json").loads(line)
            ts = entry.get("timestamp", "?")
            # Show only HH:MM:SS from timestamp
            if "T" in ts:
                ts = ts.split("T")[1][:8]
            elif " " in ts:
                ts = ts.split(" ")[1][:8]
            level = entry.get("level", "?")[:4]
            comp = entry.get("component", "?").split(".")[-1]
            msg = entry.get("message", "")[:120]
            # Level icons
            icon = {"ERRO": "🔴", "CRIT": "🔴", "WARN": "🟡", "INFO": "⚪", "DEBU": "⚫"}.get(level, "⚪")
            formatted.append(f"{icon} `{ts}` *{comp}*: {msg}")
        except (ValueError, KeyError):
            formatted.append(f"⚫ {line[:120]}")

    if not formatted:
        return "📋 No log entries found."

    return "\n".join(formatted)
```

Then add the command handler. In the `TelegramInterface` class, add the `/logs` method:

```python
    async def cmd_logs(self, update, context):
        """Show recent orchestrator log entries."""
        args = context.args
        n = 20
        if args:
            try:
                n = min(int(args[0]), 50)  # cap at 50 for Telegram message size
            except ValueError:
                pass

        log_path = os.path.join("logs", "orchestrator.jsonl")
        if not os.path.exists(log_path):
            await self._reply(update, "📋 No log file found.")
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                # Read last chunk efficiently — seek near end
                f.seek(0, 2)  # end
                size = f.tell()
                # Read last 100KB (enough for ~50 log lines)
                f.seek(max(0, size - 100_000))
                chunk = f.read()
                lines = chunk.strip().split("\n")
        except Exception as e:
            await self._reply(update, f"❌ Error reading logs: {e}")
            return

        text = _format_log_entries(lines, n=n)
        await self._reply(update, text, parse_mode="Markdown")
```

Register the handler in `_setup_handlers()`:

```python
        self.app.add_handler(CommandHandler("logs", self.cmd_logs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_logs_command.py -v`
Expected: 2 passed

- [ ] **Step 5: Verify import works**

Run: `python -c "from src.app.telegram_bot import _format_log_entries; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add tests/test_logs_command.py src/app/telegram_bot.py
git commit -m "feat(telegram): add /logs command for on-demand log viewing"
```

---

### Task 6: Add /logs to wrapper's Telegram poller

**Files:**
- Create: `tests/test_wrapper_logs.py`
- Modify: `kutai_wrapper.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_wrapper_logs.py`:

```python
"""Verify the wrapper handles /logs as a wrapper command."""
import unittest


class TestWrapperLogsCommand(unittest.TestCase):

    def test_wrapper_recognizes_logs_command(self):
        """The wrapper poll loop should treat /logs as a wrapper command."""
        # Read the wrapper source and verify /logs is in the wrapper command list
        with open("kutai_wrapper.py", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("/logs", text, "/logs not found in wrapper source")
        # Check it's in the command matching section
        self.assertIn('"/logs"', text, "/logs not properly quoted in command matching")

    def test_format_log_entries_importable_standalone(self):
        """_format_log_entries must be importable without the full bot stack."""
        # The wrapper can't import telegram_bot (circular), so it needs its own formatter
        # or a shared utility. Check that the wrapper has log formatting capability.
        with open("kutai_wrapper.py", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("orchestrator.jsonl", text, "Wrapper should read orchestrator.jsonl for /logs")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wrapper_logs.py -v`
Expected: FAIL — `/logs` not found in wrapper source

- [ ] **Step 3: Add /logs handling to the wrapper**

In `kutai_wrapper.py`, find the `_telegram_poll_loop` method. In the command classification section (around line 591-598), add `/logs` to the wrapper commands:

Change:
```python
                    is_wrapper_cmd = (
                        chat_id == str(TELEGRAM_ADMIN_CHAT_ID)
                        and (
                            text.startswith(("/kutai_start", "/kutai_status"))
                            or is_baslat
                            or is_sistem_unhealthy
                        )
                    )
```

To:
```python
                    is_logs_cmd = (
                        chat_id == str(TELEGRAM_ADMIN_CHAT_ID)
                        and text.startswith("/logs")
                    )
                    is_wrapper_cmd = (
                        chat_id == str(TELEGRAM_ADMIN_CHAT_ID)
                        and (
                            text.startswith(("/kutai_start", "/kutai_status"))
                            or is_baslat
                            or is_sistem_unhealthy
                            or is_logs_cmd
                        )
                    )
```

Then in the handler section (after the `elif text.startswith("/kutai_status"):` block), add:

```python
                            elif is_logs_cmd:
                                await self._send_logs(text)
```

Add the `_send_logs` method to the `KutAIWrapper` class (near `_send_status`):

```python
    async def _send_logs(self, text: str):
        """Read and send last N lines of orchestrator.jsonl."""
        import json as _json

        # Parse optional count: /logs 30
        parts = text.strip().split()
        n = 20
        if len(parts) > 1:
            try:
                n = min(int(parts[1]), 50)
            except ValueError:
                pass

        log_path = Path("logs/orchestrator.jsonl")
        if not log_path.exists():
            await self._send_telegram("📋 No log file found.")
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 100_000))
                chunk = f.read()
                lines = chunk.strip().split("\n")

            last_n = lines[-n:]
            formatted = []
            for line in last_n:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                    ts = entry.get("timestamp", "?")
                    if "T" in ts:
                        ts = ts.split("T")[1][:8]
                    elif " " in ts:
                        ts = ts.split(" ")[1][:8]
                    level = entry.get("level", "?")[:4]
                    comp = entry.get("component", "?").split(".")[-1]
                    msg = entry.get("message", "")[:120]
                    icon = {"ERRO": "🔴", "CRIT": "🔴", "WARN": "🟡", "INFO": "⚪", "DEBU": "⚫"}.get(level, "⚪")
                    formatted.append(f"{icon} `{ts}` *{comp}*: {msg}")
                except (ValueError, KeyError):
                    formatted.append(f"⚫ {line[:120]}")

            if not formatted:
                await self._send_telegram("📋 No log entries found.")
                return

            # Telegram message limit is 4096 chars — truncate if needed
            msg = "\n".join(formatted)
            if len(msg) > 4000:
                msg = msg[-4000:]
                msg = "...(truncated)\n" + msg[msg.index("\n") + 1:]

            await self._send_telegram(msg)
        except Exception as e:
            await self._send_telegram(f"❌ Error reading logs: {e}")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_wrapper_logs.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add kutai_wrapper.py tests/test_wrapper_logs.py
git commit -m "feat(wrapper): handle /logs command when KutAI is down"
```

---

### Task 7: Remove ntfy from docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Remove the ntfy service block**

In `docker-compose.yml`, delete the entire `ntfy:` service block (lines 24-42):

```yaml
  ntfy:
    image: binwiederhier/ntfy
    container_name: ntfy
    restart: unless-stopped
    ports:
      - "8083:80"
    volumes:
      - ${NTFY_CACHE_DIR:-./docker/data/ntfy/cache}:/var/cache/ntfy
      - ${NTFY_ETC_DIR:-./docker/data/ntfy/etc}:/etc/ntfy
    env_file:
      - .env
    entrypoint: ["/bin/sh", "-c"]
    command:
      - "NTFY_PASSWORD=$$NTFY_PASS ntfy user add --role=admin --ignore-exists $$NTFY_USER 2>/dev/null; ntfy serve"
    deploy:
      resources:
        limits:
          memory: 128M
          cpus: "0.25"
```

- [ ] **Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: remove ntfy service from docker-compose"
```

---

### Task 8: Run full regression

- [ ] **Step 1: Run all notification-related tests**

Run: `python -m pytest tests/test_notification_fallback.py tests/test_telegram_alert_handler.py tests/test_no_ntfy_handlers.py tests/test_ntfy_removed.py tests/test_logs_command.py tests/test_wrapper_logs.py -v`
Expected: All pass

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -x -q --timeout=30 2>&1 | tail -30`
Expected: No new failures

- [ ] **Step 3: Verify imports**

Run: `python -c "from src.infra.notifications import TelegramAlertHandler, notify, notify_telegram; from src.infra.logging_config import init_logging; from src.infra.alerting import check_alerts; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 4: Verify no ntfy references remain in active code**

Run: `grep -rn "ntfy" src/ --include="*.py" | grep -v "__pycache__"`
Expected: No matches (or only comments/docstrings if any)
