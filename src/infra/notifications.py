# notifications.py
"""
ntfy-based notification handlers.

Two handlers:
  NtfyAlertHandler  — immediate push to orchestrator-errors (ERROR+)
  NtfyBatchHandler  — buffered flush to orchestrator-logs (INFO+)

Neither handler ever raises or silently swallows — failures go to a
dedicated file-only logger so we never recurse into ourselves.
"""

import atexit
import logging
import threading
import time
from datetime import datetime, timezone

import requests

# Import lazily to avoid circular imports at module load time
def _cfg():
    from src.app import config as c
    return c


# ─── File-only fallback logger (used by handlers to report their own errors) ──

_handler_error_logger = logging.getLogger("infra.notifications._internal")
_handler_error_logger.propagate = False  # never re-enter ourselves


def _attach_file_sink():
    """Attach a bare file handler to the internal logger (called once)."""
    import os
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/notification_errors.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _handler_error_logger.addHandler(fh)
    _handler_error_logger.setLevel(logging.WARNING)


_attach_file_sink()


# ─── Core send function ────────────────────────────────────────────────────────

def send_ntfy(
    topic: str,
    title: str,
    message: str,
    priority: int = 3,
    tags: list[str] | None = None,
) -> int | None:
    """
    POST a notification to {NTFY_URL}/{topic}.

    Returns the HTTP status code, or None on network failure.
    Never raises.
    """
    cfg = _cfg()
    base_url = cfg.NTFY_URL
    if not base_url:
        _handler_error_logger.warning("send_ntfy: NTFY_URL not configured, skipping")
        return None

    # Strip any path from NTFY_URL — ntfy topics must be at the root.
    # e.g. "http://localhost:8083/kutai" → "http://localhost:8083"
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(base_url)
    base_url = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    url = f"{base_url}/{topic}"
    headers = {
        "Title": title[:250],
        "Priority": str(priority),
    }
    if tags:
        headers["Tags"] = ",".join(tags)

    auth = None
    if cfg.NTFY_USER:
        auth = (cfg.NTFY_USER, cfg.NTFY_PASS)

    try:
        resp = requests.post(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            auth=auth,
            timeout=3,
        )
        if resp.status_code == 401:
            _handler_error_logger.error(
                "send_ntfy: auth failure (401) for topic=%s", topic
            )
        return resp.status_code
    except Exception as exc:
        _handler_error_logger.error(
            "send_ntfy: network failure sending to topic=%s: %s", topic, exc
        )
        return None


# ─── NtfyAlertHandler — immediate, ERROR+ ─────────────────────────────────────

class NtfyAlertHandler(logging.Handler):
    """Immediately POSTs every ERROR/CRITICAL record to orchestrator-errors."""

    PRIORITY_MAP = {
        logging.CRITICAL: 5,
        logging.ERROR: 4,
    }

    def __init__(self):
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            cfg = _cfg()
            topic = cfg.NTFY_TOPIC_ERRORS

            priority = self.PRIORITY_MAP.get(record.levelno, 4)
            tags = [record.levelname.lower()]

            # Extract task_id if bound to the record
            task_id = getattr(record, "task_id", None)

            component = record.name
            title = f"[{record.levelname}] {component}"
            if task_id:
                title += f" task={task_id}"

            lines = [self.format(record)]
            if record.exc_info:
                import traceback
                lines.append(traceback.format_exception(*record.exc_info)[-1].strip())

            body = "\n".join(lines)[:4000]  # ntfy message size limit

            # Fire-and-forget on a daemon thread so the synchronous HTTP
            # call in send_ntfy() never blocks the asyncio event loop.
            threading.Thread(
                target=send_ntfy,
                args=(topic, title, body),
                kwargs={"priority": priority, "tags": tags},
                daemon=True,
            ).start()
        except Exception as exc:
            _handler_error_logger.error(
                "NtfyAlertHandler.emit failed: %s", exc
            )


# ─── NtfyBatchHandler — buffered, INFO+ ───────────────────────────────────────

class NtfyBatchHandler(logging.Handler):
    """
    Buffers log lines and flushes periodically (every 30s) or when >50 records
    accumulate.  Errors in this handler go to the file-only internal logger.
    """

    FLUSH_INTERVAL = 30       # seconds between automatic flushes
    MAX_BUFFER     = 20       # flush early if buffer hits this
    RETRY_DELAY    = 2        # seconds before one retry on flush failure

    PRIORITY_FOR = {
        "error":   3,
        "warning": 2,
        "info":    1,
    }

    def __init__(self):
        super().__init__(level=logging.INFO)
        self._buffer: list[tuple[int, str]] = []   # (levelno, formatted_line)
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._emit_count = 0
        self._flush_count = 0
        self._start_timer()
        atexit.register(self._flush)

    # ── timer management ──

    def _start_timer(self):
        self._timer = threading.Timer(self.FLUSH_INTERVAL, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timer_flush(self):
        try:
            self._flush()
        except Exception as exc:
            _handler_error_logger.error("NtfyBatchHandler._timer_flush failed: %s", exc)
        self._start_timer()

    # ── logging.Handler contract ──

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit_count += 1
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%H:%M:%S"
            )
            line = f"[{ts}] {record.levelname:8} {record.name}: {record.getMessage()}"
            with self._lock:
                self._buffer.append((record.levelno, line))
                if len(self._buffer) >= self.MAX_BUFFER:
                    self._flush_locked()
        except Exception as exc:
            _handler_error_logger.error("NtfyBatchHandler.emit failed: %s", exc)

    def close(self):
        if self._timer:
            self._timer.cancel()
        super().close()

    # ── flush logic ──

    def _flush(self):
        # Drain buffer under lock (fast), then send outside lock so
        # emit() on the asyncio thread never blocks on HTTP.
        with self._lock:
            if not self._buffer:
                return
            self._flush_count += 1
            items = list(self._buffer)
            self._buffer.clear()

        self._send_batch(items)

    def _flush_locked(self):
        """Flush while lock is already held (called from emit overflow)."""
        if not self._buffer:
            return
        self._flush_count += 1
        items = list(self._buffer)
        self._buffer.clear()
        # Send on a background thread to avoid blocking the caller
        threading.Thread(target=self._send_batch, args=(items,), daemon=True).start()

    def _send_batch(self, items: list[tuple[int, str]]) -> None:
        """Send a batch of log lines to ntfy. Safe to call without lock."""
        max_level = max(lvl for lvl, _ in items)
        if max_level >= logging.ERROR:
            priority = self.PRIORITY_FOR["error"]
        elif max_level >= logging.WARNING:
            priority = self.PRIORITY_FOR["warning"]
        else:
            priority = self.PRIORITY_FOR["info"]

        message = "\n".join(line for _, line in items)[:4000]

        try:
            cfg = _cfg()
            status = send_ntfy(
                cfg.NTFY_TOPIC_LOGS,
                title=f"Orchestrator logs ({len(items)} lines)",
                message=message,
                priority=priority,
            )
            if status is None:
                raise RuntimeError("send_ntfy returned None")
        except Exception as exc:
            _handler_error_logger.warning(
                "NtfyBatchHandler: flush failed (%s), retrying in %ds",
                exc, self.RETRY_DELAY,
            )
            time.sleep(self.RETRY_DELAY)
            try:
                cfg = _cfg()
                send_ntfy(
                    cfg.NTFY_TOPIC_LOGS,
                    title=f"Orchestrator logs (retry, {len(items)} lines)",
                    message=message,
                    priority=priority,
                )
            except Exception as exc2:
                _handler_error_logger.warning(
                    "NtfyBatchHandler: retry also failed (%s), dropping %d lines",
                    exc2, len(items),
                )


# ─── Shopping Notification Helpers ────────────────────────────────────────────

async def send_price_drop_alert(
    user_id: int,
    product: str,
    old_price: float,
    new_price: float,
    url: str | None = None,
) -> None:
    """Send a price drop notification via ntfy and/or Telegram."""
    pct = round((1 - new_price / old_price) * 100) if old_price > 0 else 0
    title = f"Price Drop: {product}"
    body = f"{product}: {old_price:.2f} TL -> {new_price:.2f} TL ({pct}% off)"
    if url:
        body += f"\n{url}"

    send_ntfy(
        topic=_cfg().NTFY_TOPIC_ERRORS,  # reuse high-priority topic for alerts
        title=title,
        message=body,
        priority=4,
        tags=["shopping", "price_drop"],
    )

    # Also try Telegram direct message
    try:
        from src.app.telegram_bot import _send_telegram_message
        await _send_telegram_message(user_id, f"📉 *{title}*\n{body}")
    except Exception:
        pass  # Telegram send is best-effort


async def send_deal_alert(
    user_id: int,
    product: str,
    discount_pct: float,
    url: str | None = None,
) -> None:
    """Send a deal/discount notification."""
    title = f"Deal Alert: {product}"
    body = f"{product}: {discount_pct:.0f}% discount found!"
    if url:
        body += f"\n{url}"

    send_ntfy(
        topic=_cfg().NTFY_TOPIC_ERRORS,
        title=title,
        message=body,
        priority=3,
        tags=["shopping", "deal"],
    )

    try:
        from src.app.telegram_bot import _send_telegram_message
        await _send_telegram_message(user_id, f"🏷️ *{title}*\n{body}")
    except Exception:
        pass


async def send_back_in_stock_alert(
    user_id: int,
    product: str,
    price: float,
    url: str | None = None,
) -> None:
    """Send a back-in-stock notification."""
    title = f"Back in Stock: {product}"
    body = f"{product} is back in stock at {price:.2f} TL"
    if url:
        body += f"\n{url}"

    send_ntfy(
        topic=_cfg().NTFY_TOPIC_ERRORS,
        title=title,
        message=body,
        priority=4,
        tags=["shopping", "restock"],
    )

    try:
        from src.app.telegram_bot import _send_telegram_message
        await _send_telegram_message(user_id, f"📦 *{title}*\n{body}")
    except Exception:
        pass
