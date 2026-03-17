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

    url = f"{base_url.rstrip('/')}/{topic}"
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
            timeout=10,
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

            send_ntfy(topic, title, body, priority=priority, tags=tags)
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
    MAX_BUFFER     = 50       # flush early if buffer hits this
    RETRY_DELAY    = 5        # seconds before one retry on flush failure

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
        self._start_timer()
        atexit.register(self._flush)

    # ── timer management ──

    def _start_timer(self):
        self._timer = threading.Timer(self.FLUSH_INTERVAL, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timer_flush(self):
        self._flush()
        self._start_timer()

    # ── logging.Handler contract ──

    def emit(self, record: logging.LogRecord) -> None:
        try:
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
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        """Must be called with self._lock held."""
        if not self._buffer:
            return

        items = list(self._buffer)
        self._buffer.clear()

        # Determine priority from highest severity
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
            # Retry once, then keep buffer for next cycle
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
                    "NtfyBatchHandler: retry also failed (%s), keeping buffer",
                    exc2,
                )
                # Restore buffer so we don't lose messages
                with self._lock:
                    self._buffer = [(lvl, l) for lvl, l in items] + self._buffer
