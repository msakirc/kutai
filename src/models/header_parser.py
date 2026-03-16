# src/models/header_parser.py
"""
Provider-specific rate limit header parsing.

Each provider returns rate limit info in different header formats.
This module normalizes them into a common RateLimitSnapshot.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitSnapshot:
    """Normalized rate limit state from one API response."""
    # Per-minute limits (None = not reported by this provider)
    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    rpm_reset_at: float | None = None       # absolute timestamp

    tpm_limit: int | None = None
    tpm_remaining: int | None = None
    tpm_reset_at: float | None = None

    # Daily limits (Cerebras, SambaNova, Gemini)
    rpd_limit: int | None = None
    rpd_remaining: int | None = None
    rpd_reset_at: float | None = None

    def has_any_data(self) -> bool:
        return any(v is not None for v in [
            self.rpm_limit, self.rpm_remaining,
            self.tpm_limit, self.tpm_remaining,
            self.rpd_limit, self.rpd_remaining,
        ])


def parse_rate_limit_headers(
    provider: str,
    headers: dict,
) -> RateLimitSnapshot | None:
    """
    Parse rate limit headers from an API response.

    Args:
        provider: Provider name (openai, anthropic, groq, gemini, cerebras, sambanova)
        headers: Raw response headers dict

    Returns:
        RateLimitSnapshot with parsed values, or None if no rate limit headers found
    """
    if not headers:
        return None

    # Normalize header keys to lowercase
    h = {k.lower(): v for k, v in headers.items()}

    # Strip litellm's "llm_provider-" prefix if present
    cleaned = {}
    for k, v in h.items():
        if k.startswith("llm_provider-"):
            cleaned[k[len("llm_provider-"):]] = v
        else:
            cleaned[k] = v
    h = cleaned

    parser = _PROVIDER_PARSERS.get(provider, _parse_openai_style)
    snap = parser(h)

    if snap and snap.has_any_data():
        return snap
    return None


# ─── Reset time parsing helpers ─────────────────────────────────────────────

def _parse_reset_duration(value: str) -> float | None:
    """
    Parse a reset duration string into an absolute timestamp.
    Handles: "12ms", "6s", "1.5s", "2m", "1h30m", ISO timestamps.
    """
    if not value:
        return None

    value = value.strip()
    now = time.time()

    # ISO 8601 timestamp (Anthropic style: "2026-01-27T12:00:30Z")
    if "T" in value and ("Z" in value or "+" in value or "-" in value[10:]):
        try:
            from datetime import datetime, timezone
            v = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(v)
            return dt.timestamp()
        except (ValueError, OSError):
            pass

    # Pure float seconds (Cerebras style: "33011.382867")
    try:
        secs = float(value)
        if secs > 1e9:
            # Already an epoch timestamp
            return secs
        return now + secs
    except ValueError:
        pass

    # Duration string: "12ms", "6s", "2m30s", "1h"
    total_seconds = 0.0
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)")
    matches = pattern.findall(value)
    if matches:
        for amount_str, unit in matches:
            amount = float(amount_str)
            if unit == "ms":
                total_seconds += amount / 1000
            elif unit == "s":
                total_seconds += amount
            elif unit == "m":
                total_seconds += amount * 60
            elif unit == "h":
                total_seconds += amount * 3600
        return now + total_seconds

    return None


def _safe_int(value) -> int | None:
    """Safely convert a header value to int."""
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return None


# ─── Provider-Specific Parsers ──────────────────────────────────────────────

def _parse_openai_style(h: dict) -> RateLimitSnapshot:
    """
    Parse OpenAI-style x-ratelimit headers.
    Used by: OpenAI, Groq, Gemini (partially).
    """
    return RateLimitSnapshot(
        rpm_limit=_safe_int(h.get("x-ratelimit-limit-requests")),
        rpm_remaining=_safe_int(h.get("x-ratelimit-remaining-requests")),
        rpm_reset_at=_parse_reset_duration(h.get("x-ratelimit-reset-requests", "")),
        tpm_limit=_safe_int(h.get("x-ratelimit-limit-tokens")),
        tpm_remaining=_safe_int(h.get("x-ratelimit-remaining-tokens")),
        tpm_reset_at=_parse_reset_duration(h.get("x-ratelimit-reset-tokens", "")),
    )


def _parse_anthropic(h: dict) -> RateLimitSnapshot:
    """
    Parse Anthropic's anthropic-ratelimit-* headers.
    Format: anthropic-ratelimit-{requests,tokens}-{limit,remaining,reset}
    """
    return RateLimitSnapshot(
        rpm_limit=_safe_int(h.get("anthropic-ratelimit-requests-limit")),
        rpm_remaining=_safe_int(h.get("anthropic-ratelimit-requests-remaining")),
        rpm_reset_at=_parse_reset_duration(h.get("anthropic-ratelimit-requests-reset", "")),
        tpm_limit=_safe_int(h.get("anthropic-ratelimit-tokens-limit")),
        tpm_remaining=_safe_int(h.get("anthropic-ratelimit-tokens-remaining")),
        tpm_reset_at=_parse_reset_duration(h.get("anthropic-ratelimit-tokens-reset", "")),
    )


def _parse_cerebras(h: dict) -> RateLimitSnapshot:
    """
    Parse Cerebras headers with daily request + per-minute token limits.
    Format: x-ratelimit-{limit,remaining,reset}-{requests-day,tokens-minute}
    Reset values are float seconds.
    """
    return RateLimitSnapshot(
        tpm_limit=_safe_int(h.get("x-ratelimit-limit-tokens-minute")),
        tpm_remaining=_safe_int(h.get("x-ratelimit-remaining-tokens-minute")),
        tpm_reset_at=_parse_reset_duration(h.get("x-ratelimit-reset-tokens-minute", "")),
        rpd_limit=_safe_int(h.get("x-ratelimit-limit-requests-day")),
        rpd_remaining=_safe_int(h.get("x-ratelimit-remaining-requests-day")),
        rpd_reset_at=_parse_reset_duration(h.get("x-ratelimit-reset-requests-day", "")),
    )


def _parse_sambanova(h: dict) -> RateLimitSnapshot:
    """
    Parse SambaNova headers with RPM + RPD limits.
    Uses epoch timestamps for reset times.
    """
    snap = RateLimitSnapshot()

    for prefix in ["x-ratelimit", "ratelimit"]:
        rpm_limit = _safe_int(h.get(f"{prefix}-limit-requests-minute"))
        if rpm_limit is not None:
            snap.rpm_limit = rpm_limit
            snap.rpm_remaining = _safe_int(h.get(f"{prefix}-remaining-requests-minute"))
            snap.rpm_reset_at = _parse_reset_duration(h.get(f"{prefix}-reset-requests-minute", ""))
            break

    for prefix in ["x-ratelimit", "ratelimit"]:
        rpd_limit = _safe_int(h.get(f"{prefix}-limit-requests-day"))
        if rpd_limit is not None:
            snap.rpd_limit = rpd_limit
            snap.rpd_remaining = _safe_int(h.get(f"{prefix}-remaining-requests-day"))
            snap.rpd_reset_at = _parse_reset_duration(h.get(f"{prefix}-reset-requests-day", ""))
            break

    if not snap.has_any_data():
        return _parse_openai_style(h)

    return snap


def _parse_gemini(h: dict) -> RateLimitSnapshot:
    """
    Parse Gemini headers. Uses generic x-ratelimit-* names.
    May include daily limits (RPD).
    """
    snap = _parse_openai_style(h)

    rpd_limit = _safe_int(h.get("x-ratelimit-limit-requests-day"))
    if rpd_limit is not None:
        snap.rpd_limit = rpd_limit
        snap.rpd_remaining = _safe_int(h.get("x-ratelimit-remaining-requests-day"))
        snap.rpd_reset_at = _parse_reset_duration(h.get("x-ratelimit-reset-requests-day", ""))

    return snap


_PROVIDER_PARSERS = {
    "openai": _parse_openai_style,
    "groq": _parse_openai_style,
    "anthropic": _parse_anthropic,
    "cerebras": _parse_cerebras,
    "sambanova": _parse_sambanova,
    "gemini": _parse_gemini,
}
