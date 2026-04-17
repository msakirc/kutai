# logging_config.py
"""
Structured logging configuration — thin re-export from yazbunu.

All modules continue to import from here:
    from src.infra.logging_config import get_logger
    logger = get_logger("core.orchestrator")

The implementation now lives in the yazbunu package.
"""

from yazbunu import get_logger, init_logging, YazFormatter  # noqa: F401

# Note: no basicConfig() here — init_logging() in run.py sets up all handlers.
# Adding basicConfig would create a duplicate StreamHandler on the root logger,
# causing every log line to appear twice on stdout (and twice in guard.jsonl).
