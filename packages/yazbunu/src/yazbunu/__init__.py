"""
Yazbunu — structured JSONL logging for the KutAI ecosystem.

Usage:
    from yazbunu import get_logger, init_logging

    init_logging(log_dir="./logs", project="kutai")
    logger = get_logger("core.orchestrator")
    logger.info("task dispatched", task="42", mission="m-7")
"""

import logging
import logging.handlers
import os
import sys

from yazbunu.formatter import YazFormatter

__all__ = ["get_logger", "init_logging", "YazFormatter"]

_project_prefix: str = ""
_initialized_projects: set[str] = set()

# ─── Reserved LogRecord attributes ───────────────────────────────────────────
_RESERVED = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "process", "processName", "message",
})


class _ContextLogger:
    """
    Thin wrapper around stdlib logger supporting keyword-arg context fields.

    Usage:
        logger.info("msg", task="5", duration_ms=120)
    """

    def __init__(self, name: str):
        self._log = logging.getLogger(name)
        self.name = name

    def _emit(self, level: int, msg: str, args: tuple, ctx: dict):
        safe = {k: v for k, v in ctx.items() if k not in _RESERVED}
        extra = {**safe, "_yaz_extra": safe}
        self._log.log(level, msg, *args, extra=extra)

    def debug(self, msg: str, *args, **ctx):
        self._emit(logging.DEBUG, msg, args, ctx)

    def info(self, msg: str, *args, **ctx):
        self._emit(logging.INFO, msg, args, ctx)

    def warning(self, msg: str, *args, **ctx):
        self._emit(logging.WARNING, msg, args, ctx)

    def error(self, msg: str, *args, **ctx):
        self._emit(logging.ERROR, msg, args, ctx)

    def critical(self, msg: str, *args, **ctx):
        self._emit(logging.CRITICAL, msg, args, ctx)

    def exception(self, msg: str, *args, **ctx):
        safe = {k: v for k, v in ctx.items() if k not in _RESERVED}
        extra = {**safe, "_yaz_extra": safe}
        self._log.exception(msg, *args, extra=extra)

    def bind(self, **ctx) -> "_BoundLogger":
        return _BoundLogger(self, ctx)


class _BoundLogger:
    """Logger with pre-bound context fields."""

    def __init__(self, parent: _ContextLogger, bound: dict):
        self._parent = parent
        self._bound = bound

    def _merge(self, ctx: dict) -> dict:
        return {**self._bound, **ctx}

    def debug(self, msg, *args, **ctx): self._parent.debug(msg, *args, **self._merge(ctx))
    def info(self, msg, *args, **ctx): self._parent.info(msg, *args, **self._merge(ctx))
    def warning(self, msg, *args, **ctx): self._parent.warning(msg, *args, **self._merge(ctx))
    def error(self, msg, *args, **ctx): self._parent.error(msg, *args, **self._merge(ctx))
    def critical(self, msg, *args, **ctx): self._parent.critical(msg, *args, **self._merge(ctx))
    def exception(self, msg, *args, **ctx): self._parent.exception(msg, *args, **self._merge(ctx))
    def bind(self, **ctx): return _BoundLogger(self._parent, self._merge(ctx))


def get_logger(component: str) -> _ContextLogger:
    """
    Return a structured logger for the given component name.

    If init_logging(project="foo") was called, the logger name becomes
    "foo.component". Otherwise it's just "component".
    """
    name = f"{_project_prefix}.{component}" if _project_prefix else component
    return _ContextLogger(name)


def init_logging(
    log_dir: str = "./logs",
    project: str = "app",
    console: bool = True,
    level: str = "DEBUG",
    max_bytes: int = 50_000_000,
    backup_count: int = 5,
) -> None:
    """
    Configure logging sinks for a project.

    Args:
        log_dir: Directory for JSONL log files.
        project: Project name prefix — becomes the log filename and logger prefix.
        console: Enable console (stdout) output.
        level: Minimum log level.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated backup files to keep.
    """
    global _project_prefix

    if project in _initialized_projects:
        return
    _initialized_projects.add(project)
    _project_prefix = project

    os.makedirs(log_dir, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(log_level)

    # JSONL file sink — rotating
    log_path = os.path.join(log_dir, f"{project}.jsonl")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(YazFormatter())
    root.addHandler(file_handler)

    # Console sink — human-readable
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(console_handler)

    # Quiet noisy libraries
    for lib in ("httpcore", "httpx", "aiosqlite", "asyncio", "urllib3",
                "telegram.ext", "aiohttp.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)
