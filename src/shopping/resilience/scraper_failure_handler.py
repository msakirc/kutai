"""Shopping-specific error recovery.

Classifies errors and recommends recovery actions so the orchestration
layer can react appropriately to scraper and LLM failures.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.error_recovery")

# ─── Error Classification ───────────────────────────────────────────────────

_TRANSIENT_SUBSTRINGS = [
    "timeout", "timed out", "connect", "temporarily",
    "503", "502", "504", "reset by peer", "broken pipe",
]
_RATE_LIMIT_SUBSTRINGS = ["429", "rate limit", "too many requests", "quota"]
_BLOCKED_SUBSTRINGS = ["403", "forbidden", "captcha", "blocked", "challenge"]
_PARSE_SUBSTRINGS = ["json", "parse", "decode", "keyerror", "indexerror", "attributeerror"]


def classify_error(error: Exception) -> str:
    """Classify *error* into a recovery category.

    Returns one of:
    - ``"transient"`` -- temporary network / server error, safe to retry.
    - ``"permanent"`` -- fatal error that won't resolve on retry.
    - ``"rate_limit"`` -- we're being rate-limited, back off.
    - ``"blocked"`` -- the source is actively blocking us.
    - ``"parse_error"`` -- the response was received but couldn't be parsed.
    """
    msg = str(error).lower()
    err_type = type(error).__name__.lower()
    combined = f"{msg} {err_type}"

    if any(s in combined for s in _RATE_LIMIT_SUBSTRINGS):
        return "rate_limit"
    if any(s in combined for s in _BLOCKED_SUBSTRINGS):
        return "blocked"
    if any(s in combined for s in _PARSE_SUBSTRINGS):
        return "parse_error"
    if any(s in combined for s in _TRANSIENT_SUBSTRINGS):
        return "transient"

    return "permanent"


# ─── Scraper Error Recovery ─────────────────────────────────────────────────

# How many immediate retries for each error class
_RETRY_MAP: dict[str, int] = {
    "transient": 2,
    "rate_limit": 0,     # don't retry, wait for budget reset
    "blocked": 0,        # don't retry, use fallback
    "parse_error": 1,    # one retry in case page changed mid-load
    "permanent": 0,
}

# Delay before retry (seconds)
_DELAY_MAP: dict[str, float] = {
    "transient": 3.0,
    "rate_limit": 60.0,
    "blocked": 0.0,
    "parse_error": 1.0,
    "permanent": 0.0,
}


async def handle_scraper_error(
    domain: str,
    error: Exception,
    context: dict[str, Any] | None = None,
) -> dict:
    """Decide on a recovery action for a scraper failure.

    Parameters
    ----------
    domain:
        The scraper domain that failed (e.g. ``"trendyol"``).
    error:
        The exception that was raised.
    context:
        Optional dict with extra info (``retry_count``, ``query``, etc.).

    Returns
    -------
    Dict with keys:

    - ``action`` -- one of ``"retry"``, ``"skip"``, ``"fallback"``, ``"abort"``.
    - ``reason`` -- human-readable explanation.
    - ``delay`` -- seconds to wait before retry (0 if not retrying).
    - ``error_class`` -- the classified error category.
    """
    context = context or {}
    error_class = classify_error(error)
    retries_done = context.get("retry_count", 0)
    max_retries = _RETRY_MAP.get(error_class, 0)

    logger.warning(
        "Scraper error on %s [%s]: %s (retry %d/%d)",
        domain, error_class, error, retries_done, max_retries,
    )

    if error_class == "blocked":
        return {
            "action": "fallback",
            "reason": f"{domain} is blocking requests; switching to fallback source",
            "delay": 0,
            "error_class": error_class,
        }

    if error_class == "rate_limit":
        return {
            "action": "skip",
            "reason": f"Rate limited by {domain}; skipping until budget resets",
            "delay": _DELAY_MAP["rate_limit"],
            "error_class": error_class,
        }

    if retries_done < max_retries:
        delay = _DELAY_MAP.get(error_class, 1.0)
        return {
            "action": "retry",
            "reason": f"Transient error on {domain}, retrying in {delay:.0f}s",
            "delay": delay,
            "error_class": error_class,
        }

    if error_class == "permanent":
        return {
            "action": "abort",
            "reason": f"Permanent error on {domain}: {error}",
            "delay": 0,
            "error_class": error_class,
        }

    # Exhausted retries for transient / parse errors
    return {
        "action": "fallback",
        "reason": f"Retries exhausted for {domain} ({error_class}); switching to fallback",
        "delay": 0,
        "error_class": error_class,
    }


# ─── LLM Error Recovery ────────────────────────────────────────────────────

async def handle_llm_error(
    error: Exception,
    context: dict[str, Any] | None = None,
) -> dict:
    """Decide on a recovery action for an LLM failure.

    Parameters
    ----------
    error:
        The exception raised by the LLM call.
    context:
        Optional dict (``model``, ``retry_count``, ``prompt_tokens``, etc.).

    Returns
    -------
    Dict with keys ``action``, ``reason``, ``delay``, ``error_class``.
    """
    context = context or {}
    error_class = classify_error(error)
    retries_done = context.get("retry_count", 0)
    msg = str(error).lower()

    logger.warning("LLM error [%s]: %s (retry %d)", error_class, error, retries_done)

    # Token limit exceeded -- reduce input
    if "token" in msg and ("limit" in msg or "length" in msg):
        return {
            "action": "retry",
            "reason": "Token limit exceeded; caller should truncate input and retry",
            "delay": 0,
            "error_class": "permanent",
        }

    # Rate limited by API
    if error_class == "rate_limit":
        delay = 30.0 if retries_done < 2 else 60.0
        return {
            "action": "retry" if retries_done < 3 else "abort",
            "reason": "LLM rate limited; backing off",
            "delay": delay,
            "error_class": error_class,
        }

    # Transient errors (timeout, connection)
    if error_class == "transient" and retries_done < 2:
        return {
            "action": "retry",
            "reason": "Transient LLM error; retrying",
            "delay": 5.0,
            "error_class": error_class,
        }

    return {
        "action": "abort",
        "reason": f"LLM error unrecoverable: {error}",
        "delay": 0,
        "error_class": error_class,
    }
