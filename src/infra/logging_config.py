# logging_config.py
"""
Structured logging configuration — thin re-export from yazbunu.

All modules continue to import from here:
    from src.infra.logging_config import get_logger
    logger = get_logger("core.orchestrator")

The implementation now lives in the yazbunu package.
"""

import logging
import sys

from yazbunu import get_logger, init_logging, YazFormatter  # noqa: F401

# ─── Fallback before init_logging() runs ─────────────────────────────────────
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
