# packages/kuleden_donen_var/src/kuleden_donen_var/header_parser.py
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

    # Daily limits (Cerebras, SambaNova, Gemini, Groq paid tiers)
    rpd_limit: int | None = None
    rpd_remaining: int | None = None
    rpd_reset_at: float | None = None

    # Token-day limits (Groq paid tiers expose these on some models)
    tpd_limit: int | None = None
    tpd_remaining: int | None = None
    tpd_reset_at: float | None = None

    # Input-token splits (Anthropic exposes these per minute; some tiers per day)
    itpm_limit: int | None = None
    itpm_remaining: int | None = None
    itpm_reset_at: float | None = None
    itpd_limit: int | None = None
    itpd_remaining: int | None = None
    itpd_reset_at: float | None = None

    # Output-token splits (Anthropic; symmetric to input)
    otpm_limit: int | None = None
    otpm_remaining: int | None = None
    otpm_reset_at: float | None = None
    otpd_limit: int | None = None
    otpd_remaining: int | None = None
    otpd_reset_at: float | None = None

    # HTTP `Retry-After` header (RFC 7231). Authoritative provider hint —
    # "don't issue another request to this model before N seconds from now".
    # Stricter than bucket reset_at fields (which describe bucket recovery,
    # not the caller's permission to retry). Populated in seconds-from-now;
    # update_from_snapshot raises a floor on rpm_reset_at so admission gate
    # honors it without conflating buckets.
    retry_after_seconds: float | None = None

    def has_any_data(self) -> bool:
        return any(v is not None for v in [
            self.rpm_limit, self.rpm_remaining,
            self.tpm_limit, self.tpm_remaining,
            self.rpd_limit, self.rpd_remaining,
            self.tpd_limit, self.tpd_remaining,
            self.itpm_limit, self.itpm_remaining,
            self.itpd_limit, self.itpd_remaining,
            self.otpm_limit, self.otpm_remaining,
            self.otpd_limit, self.otpd_remaining,
            self.retry_after_seconds,
        ])


def parse_429_body(provider: str, error_message: str) -> RateLimitSnapshot | None:
    """Extract rate-limit state from a 429 RESOURCE_EXHAUSTED body.

    Some providers (Gemini in particular) don't surface daily-axis rate
    limit info via response headers — it only appears in the error body
    of a 429:

        "Quota exceeded for metric: ...generate_content_free_tier_requests,
         limit: 0, model: gemini-2.0-flash"
        "Please retry in 40.121029375s."

    Without parsing this, RateLimitMatrix.rpd stays empty for tier-locked
    models and the selector keeps rediscovering the limit by failing.
    Returns a snapshot with rpd_limit/remaining/reset_at populated so
    update_from_snapshot can write through to the matrix and S1 depletion
    fires negative pressure on the next selection.

    Currently implemented: gemini RESOURCE_EXHAUSTED. Other providers can
    add their own body-shape clauses here.
    """
    if not error_message:
        return None
    msg = str(error_message)
    if provider != "gemini":
        return None
    if "RESOURCE_EXHAUSTED" not in msg and "Quota exceeded" not in msg:
        return None
    # `limit: N` pattern. Pick the smallest non-negative limit across all
    # quotaMetric occurrences — represents the most-restrictive axis the
    # caller can satisfy without 429ing.
    limit_match = re.findall(r"limit:\s*(\d+)", msg)
    limit_val = min((int(x) for x in limit_match), default=None)
    # `Please retry in 40.5s` or `Please retry in 86400s`.
    retry_match = re.search(r"retry in ([\d\.]+)\s*s", msg, re.IGNORECASE)
    retry_secs: float
    if retry_match:
        try:
            retry_secs = float(retry_match.group(1))
        except ValueError:
            retry_secs = 60.0
    else:
        retry_secs = 86400.0  # fall back to daily-rollover window
    snap = RateLimitSnapshot()
    # Always at least one daily-axis quota-failure → write rpd cell.
    if limit_val is not None:
        snap.rpd_limit = max(1, limit_val)  # 0 → 1 marker so cell isn't dropped
        snap.rpd_remaining = 0
        snap.rpd_reset_at = time.time() + max(60.0, retry_secs)
    return snap if snap.has_any_data() else None


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

    # `Retry-After` is provider-agnostic per RFC 7231 — every cloud (groq,
    # cerebras, openai, anthropic) returns it on 429. Parse centrally so
    # provider parsers stay focused on bucket-shape headers.
    retry_after = _parse_retry_after(h.get("retry-after", ""))
    if retry_after is not None:
        if snap is None:
            snap = RateLimitSnapshot()
        snap.retry_after_seconds = retry_after

    if snap and snap.has_any_data():
        return snap
    return None


# ─── Reset time parsing helpers ─────────────────────────────────────────────

def _parse_retry_after(value: str) -> float | None:
    """Parse RFC 7231 `Retry-After` header. Two legal forms:
        delta-seconds:  "5", "12.5", "120"
        HTTP-date:      "Wed, 21 Oct 2026 07:28:00 GMT"
    Returns seconds-from-now (>= 0) or None if absent/malformed.
    JSON LLM APIs use delta-seconds in practice; the IMF-fixdate branch
    is defensive and rarely exercised.
    """
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        n = float(s)
        return max(0.0, n)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(s)
        if dt is None:
            return None
        return max(0.0, dt.timestamp() - time.time())
    except (ValueError, TypeError):
        return None


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
    """Parse Anthropic's anthropic-ratelimit-* headers.

    Header families (all anthropic-ratelimit-{family}-{limit,remaining,reset}):

      requests        → rpm   (combined request budget)
      tokens          → tpm   (combined input+output token budget)
      input-tokens    → itpm  (input-only sub-budget; some tiers cap separately)
      output-tokens   → otpm  (output-only sub-budget; expensive completions)

    Anthropic also exposes input-tokens-day / output-tokens-day on certain
    tiers — parsed when present so itpd / otpd matrix cells populate. Without
    these, the admission gate can only see the combined tpm budget and
    misses an output-burst exhaustion the headers actually warn about.

    `reset` values are ISO 8601 timestamps (e.g. "2026-01-27T12:00:30Z").
    """
    snap = RateLimitSnapshot(
        rpm_limit=_safe_int(h.get("anthropic-ratelimit-requests-limit")),
        rpm_remaining=_safe_int(h.get("anthropic-ratelimit-requests-remaining")),
        rpm_reset_at=_parse_reset_duration(h.get("anthropic-ratelimit-requests-reset", "")),
        tpm_limit=_safe_int(h.get("anthropic-ratelimit-tokens-limit")),
        tpm_remaining=_safe_int(h.get("anthropic-ratelimit-tokens-remaining")),
        tpm_reset_at=_parse_reset_duration(h.get("anthropic-ratelimit-tokens-reset", "")),
    )

    # Input-token split
    itpm_limit = _safe_int(h.get("anthropic-ratelimit-input-tokens-limit"))
    if itpm_limit is not None:
        snap.itpm_limit = itpm_limit
        snap.itpm_remaining = _safe_int(h.get("anthropic-ratelimit-input-tokens-remaining"))
        snap.itpm_reset_at = _parse_reset_duration(h.get("anthropic-ratelimit-input-tokens-reset", ""))

    itpd_limit = _safe_int(h.get("anthropic-ratelimit-input-tokens-day-limit"))
    if itpd_limit is not None:
        snap.itpd_limit = itpd_limit
        snap.itpd_remaining = _safe_int(h.get("anthropic-ratelimit-input-tokens-day-remaining"))
        snap.itpd_reset_at = _parse_reset_duration(h.get("anthropic-ratelimit-input-tokens-day-reset", ""))

    # Output-token split
    otpm_limit = _safe_int(h.get("anthropic-ratelimit-output-tokens-limit"))
    if otpm_limit is not None:
        snap.otpm_limit = otpm_limit
        snap.otpm_remaining = _safe_int(h.get("anthropic-ratelimit-output-tokens-remaining"))
        snap.otpm_reset_at = _parse_reset_duration(h.get("anthropic-ratelimit-output-tokens-reset", ""))

    otpd_limit = _safe_int(h.get("anthropic-ratelimit-output-tokens-day-limit"))
    if otpd_limit is not None:
        snap.otpd_limit = otpd_limit
        snap.otpd_remaining = _safe_int(h.get("anthropic-ratelimit-output-tokens-day-remaining"))
        snap.otpd_reset_at = _parse_reset_duration(h.get("anthropic-ratelimit-output-tokens-day-reset", ""))

    return snap


def _parse_cerebras(h: dict) -> RateLimitSnapshot:
    """
    Parse Cerebras rate-limit headers. The API exposes four bucket families
    (minute/hour/day) on two axes (requests/tokens). KDV tracks rpm/tpm/rpd/tpd
    only — the hour bucket has no field and is dropped (the day bucket alone
    is enough to gate admission). Reset headers are not emitted on success
    responses; on 429 they appear as float seconds, parsed via
    `_parse_reset_duration`. Format:
        x-ratelimit-{limit,remaining,reset}-{requests,tokens}-{minute,day}
    """
    return RateLimitSnapshot(
        rpm_limit=_safe_int(h.get("x-ratelimit-limit-requests-minute")),
        rpm_remaining=_safe_int(h.get("x-ratelimit-remaining-requests-minute")),
        rpm_reset_at=_parse_reset_duration(h.get("x-ratelimit-reset-requests-minute", "")),
        tpm_limit=_safe_int(h.get("x-ratelimit-limit-tokens-minute")),
        tpm_remaining=_safe_int(h.get("x-ratelimit-remaining-tokens-minute")),
        tpm_reset_at=_parse_reset_duration(h.get("x-ratelimit-reset-tokens-minute", "")),
        rpd_limit=_safe_int(h.get("x-ratelimit-limit-requests-day")),
        rpd_remaining=_safe_int(h.get("x-ratelimit-remaining-requests-day")),
        rpd_reset_at=_parse_reset_duration(h.get("x-ratelimit-reset-requests-day", "")),
        tpd_limit=_safe_int(h.get("x-ratelimit-limit-tokens-day")),
        tpd_remaining=_safe_int(h.get("x-ratelimit-remaining-tokens-day")),
        tpd_reset_at=_parse_reset_duration(h.get("x-ratelimit-reset-tokens-day", "")),
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
    """Parse Gemini headers. Uses generic x-ratelimit-* names.

    Free tier exposes RPD (requests-day) prominently — without per-day
    parsing, only the per-minute axis lands and the admission gate
    can't see a daily exhaustion until it 429s. Paid tiers have added
    TPD (tokens-day) on selected models — captured when present.
    """
    snap = _parse_openai_style(h)

    rpd_limit = _safe_int(h.get("x-ratelimit-limit-requests-day"))
    if rpd_limit is not None:
        snap.rpd_limit = rpd_limit
        snap.rpd_remaining = _safe_int(h.get("x-ratelimit-remaining-requests-day"))
        snap.rpd_reset_at = _parse_reset_duration(h.get("x-ratelimit-reset-requests-day", ""))

    tpd_limit = _safe_int(h.get("x-ratelimit-limit-tokens-day"))
    if tpd_limit is not None:
        snap.tpd_limit = tpd_limit
        snap.tpd_remaining = _safe_int(h.get("x-ratelimit-remaining-tokens-day"))
        snap.tpd_reset_at = _parse_reset_duration(h.get("x-ratelimit-reset-tokens-day", ""))

    return snap


def _parse_groq(h: dict) -> RateLimitSnapshot:
    """Parse Groq's x-ratelimit-* headers.

    Groq's standard response headers carry per-minute axes only:
        x-ratelimit-limit-requests          (RPM)
        x-ratelimit-remaining-requests
        x-ratelimit-reset-requests
        x-ratelimit-limit-tokens            (TPM)
        x-ratelimit-remaining-tokens
        x-ratelimit-reset-tokens

    Some paid-tier Groq models additionally expose daily axes
    (forward-compat, observed on selected accounts):
        x-ratelimit-limit-requests-day      (RPD)
        x-ratelimit-remaining-requests-day
        x-ratelimit-reset-requests-day
        x-ratelimit-limit-tokens-day        (TPD)
        x-ratelimit-remaining-tokens-day
        x-ratelimit-reset-tokens-day

    Without daily-axis parsing, RPD/TPD only ever land via static
    config in models.yaml — admission gate misses real provider
    quota churn until 429. This parser populates day-axis cells
    when the provider sends them, otherwise behaves like openai_style.
    """
    snap = _parse_openai_style(h)

    rpd_limit = _safe_int(h.get("x-ratelimit-limit-requests-day"))
    if rpd_limit is not None:
        snap.rpd_limit = rpd_limit
        snap.rpd_remaining = _safe_int(h.get("x-ratelimit-remaining-requests-day"))
        snap.rpd_reset_at = _parse_reset_duration(h.get("x-ratelimit-reset-requests-day", ""))

    tpd_limit = _safe_int(h.get("x-ratelimit-limit-tokens-day"))
    if tpd_limit is not None:
        snap.tpd_limit = tpd_limit
        snap.tpd_remaining = _safe_int(h.get("x-ratelimit-remaining-tokens-day"))
        snap.tpd_reset_at = _parse_reset_duration(h.get("x-ratelimit-reset-tokens-day", ""))

    return snap


_PROVIDER_PARSERS = {
    "openai": _parse_openai_style,
    "groq": _parse_groq,
    "anthropic": _parse_anthropic,
    "cerebras": _parse_cerebras,
    "sambanova": _parse_sambanova,
    "gemini": _parse_gemini,
}
