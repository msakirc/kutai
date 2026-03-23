# logging_config.py
"""
Structured logging configuration.

# Log sinks (after init_logging()):
#   1. Console    — StreamHandler(stdout), DEBUG, colorized key=value
#   2. File       — RotatingFileHandler("logs/orchestrator.jsonl"), DEBUG, JSON-lines
#   3. ntfy batch — NtfyBatchHandler, INFO, flushed every 30s
#   4. ntfy alert — NtfyAlertHandler, ERROR, immediate push

# Rotation policy: 5 files × 50 MB = 250 MB max disk usage
# Encoding: utf-8
#
# Usage:
#   from src.infra.logging_config import get_logger
#   logger = get_logger("core.orchestrator")
#   logger.info("task dispatched", task_id="42", agent="coder")
"""

import logging
import logging.handlers
import os
import sys

# ─── Fallback before init_logging() runs ─────────────────────────────────────
# Any module that imports at the top level and logs at import time will use
# this StreamHandler until init_logging() replaces it.

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiosqlite").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.INFO)

_initialized = False

# ─── Public API ───────────────────────────────────────────────────────────────

class _ContextLogger:
    """
    Thin wrapper around a stdlib logger that supports keyword-arg context fields.

    Usage:
        logger.info("msg", task_id="5", duration_ms=120)
    """

    def __init__(self, name: str):
        self._log = logging.getLogger(name)
        self.name = name

    def _fmt(self, msg: str, **ctx) -> str:
        if not ctx:
            return msg
        kv = " ".join(f"{k}={v!r}" for k, v in ctx.items())
        return f"{msg} | {kv}"

    def _inject(self, record_kwargs: dict, **ctx):
        """Inject context fields as LogRecord extras for JSON formatter."""
        return ctx

    def debug(self, msg: str, *args, **ctx):
        self._log.debug(self._fmt(msg, **ctx), *args, extra=ctx)

    def info(self, msg: str, *args, **ctx):
        self._log.info(self._fmt(msg, **ctx), *args, extra=ctx)

    def warning(self, msg: str, *args, **ctx):
        self._log.warning(self._fmt(msg, **ctx), *args, extra=ctx)

    def error(self, msg: str, *args, **ctx):
        self._log.error(self._fmt(msg, **ctx), *args, extra=ctx)

    def critical(self, msg: str, *args, **ctx):
        self._log.critical(self._fmt(msg, **ctx), *args, extra=ctx)

    def exception(self, msg: str, *args, **ctx):
        self._log.exception(self._fmt(msg, **ctx), *args, extra=ctx)

    def bind(self, **ctx) -> "_BoundLogger":
        return _BoundLogger(self, ctx)


class _BoundLogger:
    """Logger with pre-bound context fields."""

    def __init__(self, parent: _ContextLogger, bound: dict):
        self._parent = parent
        self._bound = bound

    def _merge(self, **ctx):
        return {**self._bound, **ctx}

    def debug(self, msg, **ctx): self._parent.debug(msg, **self._merge(**ctx))
    def info(self, msg, **ctx): self._parent.info(msg, **self._merge(**ctx))
    def warning(self, msg, **ctx): self._parent.warning(msg, **self._merge(**ctx))
    def error(self, msg, **ctx): self._parent.error(msg, **self._merge(**ctx))
    def critical(self, msg, **ctx): self._parent.critical(msg, **self._merge(**ctx))
    def exception(self, msg, **ctx): self._parent.exception(msg, **self._merge(**ctx))
    def bind(self, **ctx): return _BoundLogger(self._parent, self._merge(**ctx))


def get_logger(component: str) -> _ContextLogger:
    """
    Return a structured logger for the given component name.
    This is the only way any module should obtain a logger.
    """
    return _ContextLogger(component)


# ─── JSON formatter ───────────────────────────────────────────────────────────

import json
from datetime import datetime, timezone

CONTEXT_FIELDS = {"task_id", "goal_id", "agent_type", "model", "action", "duration_ms"}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        for field in CONTEXT_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                doc[field] = val
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)


# ─── init_logging() ──────────────────────────────────────────────────────────

def init_logging() -> None:
    """
    Configure all four sinks on the root logger.
    Call this as the very first action in run.py before any other imports that log.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs("logs", exist_ok=True)

    root = logging.getLogger()
    # Clear any basicConfig handlers
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    # Sink 1: console — human-readable
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    root.addHandler(console)

    # Sink 2: file — JSON-lines, rotating
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/orchestrator.jsonl",
        maxBytes=50_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)

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

    logging.getLogger("infra.logging_config").info(
        "Logging initialized, all sinks active"
    )
