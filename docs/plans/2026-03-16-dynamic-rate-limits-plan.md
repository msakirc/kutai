# Dynamic Rate Limits + Quota Planner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hardcoded rate limits with dynamic header-driven discovery and add a quota planner that maximizes expensive model usage while reserving capacity for hard tasks.

**Architecture:** Two new modules (`header_parser.py`, `quota_planner.py`) plus enhancements to existing `rate_limiter.py` and `router.py`. Headers from API responses are parsed per-provider into a common `RateLimitSnapshot`, which updates the existing `RateLimitState`. The `QuotaPlanner` peeks at the task queue to dynamically set a difficulty threshold for paid model usage.

**Tech Stack:** Python 3.11+, litellm, asyncio, unittest.IsolatedAsyncioTestCase

---

### Task 1: Header Parser — RateLimitSnapshot dataclass + OpenAI/Groq parser

**Files:**
- Create: `src/models/header_parser.py`
- Test: `tests/test_header_parser.py`

**Step 1: Write the failing test**

```python
# tests/test_header_parser.py
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.header_parser import RateLimitSnapshot, parse_rate_limit_headers


class TestRateLimitSnapshot(unittest.TestCase):

    def test_openai_standard_headers(self):
        headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-limit-tokens": "200000",
            "x-ratelimit-remaining-requests": "490",
            "x-ratelimit-remaining-tokens": "195000",
            "x-ratelimit-reset-requests": "12ms",
            "x-ratelimit-reset-tokens": "6s",
        }
        snap = parse_rate_limit_headers("openai", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 500)
        self.assertEqual(snap.tpm_limit, 200000)
        self.assertEqual(snap.rpm_remaining, 490)
        self.assertEqual(snap.tpm_remaining, 195000)
        self.assertIsNotNone(snap.rpm_reset_at)
        self.assertIsNotNone(snap.tpm_reset_at)

    def test_groq_headers(self):
        """Groq uses same format as OpenAI."""
        headers = {
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-limit-tokens": "131072",
            "x-ratelimit-remaining-requests": "28",
            "x-ratelimit-remaining-tokens": "120000",
            "x-ratelimit-reset-requests": "2s",
            "x-ratelimit-reset-tokens": "1.5s",
        }
        snap = parse_rate_limit_headers("groq", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 30)
        self.assertEqual(snap.rpm_remaining, 28)

    def test_empty_headers_returns_none(self):
        snap = parse_rate_limit_headers("openai", {})
        self.assertIsNone(snap)

    def test_partial_headers_still_parse(self):
        headers = {"x-ratelimit-remaining-requests": "10"}
        snap = parse_rate_limit_headers("openai", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_remaining, 10)
        self.assertIsNone(snap.rpm_limit)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_header_parser.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
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
            # Handle Z suffix
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
        # Cerebras uses per-minute tokens but per-day requests
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

    # Try per-minute request headers
    for prefix in ["x-ratelimit", "ratelimit"]:
        rpm_limit = _safe_int(h.get(f"{prefix}-limit-requests-minute"))
        if rpm_limit is not None:
            snap.rpm_limit = rpm_limit
            snap.rpm_remaining = _safe_int(h.get(f"{prefix}-remaining-requests-minute"))
            snap.rpm_reset_at = _parse_reset_duration(h.get(f"{prefix}-reset-requests-minute", ""))
            break

    # Try daily request headers
    for prefix in ["x-ratelimit", "ratelimit"]:
        rpd_limit = _safe_int(h.get(f"{prefix}-limit-requests-day"))
        if rpd_limit is not None:
            snap.rpd_limit = rpd_limit
            snap.rpd_remaining = _safe_int(h.get(f"{prefix}-remaining-requests-day"))
            snap.rpd_reset_at = _parse_reset_duration(h.get(f"{prefix}-reset-requests-day", ""))
            break

    # Fall back to generic OpenAI-style if SambaNova-specific not found
    if not snap.has_any_data():
        return _parse_openai_style(h)

    return snap


def _parse_gemini(h: dict) -> RateLimitSnapshot:
    """
    Parse Gemini headers. Uses generic x-ratelimit-* names.
    May include daily limits (RPD).
    """
    snap = _parse_openai_style(h)

    # Gemini may also have daily limits
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_header_parser.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/models/header_parser.py tests/test_header_parser.py
git commit -m "feat: add rate limit header parser with per-provider normalization"
```

---

### Task 2: Header Parser — Anthropic + Cerebras + SambaNova + Gemini tests

**Files:**
- Modify: `tests/test_header_parser.py`

**Step 1: Write the failing tests**

Append to `tests/test_header_parser.py` inside the `TestRateLimitSnapshot` class:

```python
    def test_anthropic_headers(self):
        headers = {
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "45",
            "anthropic-ratelimit-requests-reset": "2026-01-27T12:00:30Z",
            "anthropic-ratelimit-tokens-limit": "80000",
            "anthropic-ratelimit-tokens-remaining": "72000",
            "anthropic-ratelimit-tokens-reset": "2026-01-27T12:00:30Z",
        }
        snap = parse_rate_limit_headers("anthropic", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 50)
        self.assertEqual(snap.rpm_remaining, 45)
        self.assertEqual(snap.tpm_limit, 80000)
        self.assertEqual(snap.tpm_remaining, 72000)
        self.assertIsNotNone(snap.rpm_reset_at)

    def test_cerebras_headers_daily_requests_minute_tokens(self):
        headers = {
            "x-ratelimit-limit-tokens-minute": "131072",
            "x-ratelimit-remaining-tokens-minute": "120000",
            "x-ratelimit-reset-tokens-minute": "45.5",
            "x-ratelimit-limit-requests-day": "1000",
            "x-ratelimit-remaining-requests-day": "950",
            "x-ratelimit-reset-requests-day": "33011.382867",
        }
        snap = parse_rate_limit_headers("cerebras", headers)
        self.assertIsNotNone(snap)
        # Cerebras: per-minute tokens, per-day requests
        self.assertEqual(snap.tpm_limit, 131072)
        self.assertEqual(snap.tpm_remaining, 120000)
        self.assertEqual(snap.rpd_limit, 1000)
        self.assertEqual(snap.rpd_remaining, 950)
        # No per-minute RPM for cerebras
        self.assertIsNone(snap.rpm_limit)

    def test_sambanova_headers(self):
        headers = {
            "x-ratelimit-limit-requests-minute": "20",
            "x-ratelimit-remaining-requests-minute": "18",
            "x-ratelimit-reset-requests-minute": "30",
            "x-ratelimit-limit-requests-day": "5000",
            "x-ratelimit-remaining-requests-day": "4900",
            "x-ratelimit-reset-requests-day": "43200",
        }
        snap = parse_rate_limit_headers("sambanova", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 20)
        self.assertEqual(snap.rpm_remaining, 18)
        self.assertEqual(snap.rpd_limit, 5000)
        self.assertEqual(snap.rpd_remaining, 4900)

    def test_gemini_with_daily_limits(self):
        headers = {
            "x-ratelimit-limit-requests": "15",
            "x-ratelimit-remaining-requests": "12",
            "x-ratelimit-reset-requests": "30s",
            "x-ratelimit-limit-tokens": "1000000",
            "x-ratelimit-remaining-tokens": "950000",
            "x-ratelimit-reset-tokens": "30s",
            "x-ratelimit-limit-requests-day": "1500",
            "x-ratelimit-remaining-requests-day": "1400",
            "x-ratelimit-reset-requests-day": "43200",
        }
        snap = parse_rate_limit_headers("gemini", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 15)
        self.assertEqual(snap.tpm_limit, 1000000)
        self.assertEqual(snap.rpd_limit, 1500)

    def test_llm_provider_prefix_stripped(self):
        """litellm proxy adds llm_provider- prefix."""
        headers = {
            "llm_provider-x-ratelimit-limit-requests": "100",
            "llm_provider-x-ratelimit-remaining-requests": "90",
        }
        snap = parse_rate_limit_headers("openai", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 100)
        self.assertEqual(snap.rpm_remaining, 90)

    def test_unknown_provider_uses_openai_style(self):
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "55",
        }
        snap = parse_rate_limit_headers("unknown_provider", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 60)
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_header_parser.py -v`
Expected: All 10 tests PASS (the parser from Task 1 already handles all providers)

**Step 3: Commit**

```bash
git add tests/test_header_parser.py
git commit -m "test: add header parser tests for all providers"
```

---

### Task 3: Enhanced RateLimitState — header-derived fields + daily limits

**Files:**
- Modify: `src/models/rate_limiter.py`
- Test: `tests/test_rate_limiter.py`

**Step 1: Write the failing test**

```python
# tests/test_rate_limiter.py
import sys, os, unittest, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rate_limiter import RateLimitState
from src.models.header_parser import RateLimitSnapshot


class TestRateLimitStateHeaders(unittest.TestCase):

    def test_update_from_snapshot_sets_limits(self):
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        snap = RateLimitSnapshot(
            rpm_limit=50, rpm_remaining=45,
            tpm_limit=200000, tpm_remaining=180000,
        )
        state.update_from_snapshot(snap)
        # Limits updated to header values
        self.assertEqual(state.rpm_limit, 50)
        self.assertEqual(state.tpm_limit, 200000)
        self.assertEqual(state._original_rpm, 50)
        self.assertEqual(state._original_tpm, 200000)
        self.assertTrue(state._limits_discovered)

    def test_update_from_snapshot_lower_limit_accepted(self):
        state = RateLimitState(rpm_limit=100, tpm_limit=500000)
        snap = RateLimitSnapshot(rpm_limit=30, tpm_limit=100000)
        state.update_from_snapshot(snap)
        self.assertEqual(state.rpm_limit, 30)
        self.assertEqual(state.tpm_limit, 100000)

    def test_has_capacity_uses_header_remaining_when_fresh(self):
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        snap = RateLimitSnapshot(
            rpm_remaining=5, tpm_remaining=50000,
        )
        state.update_from_snapshot(snap)
        self.assertTrue(state.has_capacity(1000))

    def test_has_capacity_detects_exhausted_from_headers(self):
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        snap = RateLimitSnapshot(rpm_remaining=0, tpm_remaining=50000)
        state.update_from_snapshot(snap)
        self.assertFalse(state.has_capacity(0))

    def test_daily_limit_exhaustion_blocks_capacity(self):
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        snap = RateLimitSnapshot(
            rpm_remaining=10, tpm_remaining=50000,
            rpd_remaining=0, rpd_reset_at=time.time() + 3600,
        )
        state.update_from_snapshot(snap)
        self.assertFalse(state.has_capacity(0))

    def test_update_from_snapshot_restores_adaptive_reduction(self):
        """If headers show higher limit than our adapted limit, restore."""
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        # Simulate 429 reduction
        state.record_429()
        self.assertLess(state.rpm_limit, 30)
        reduced_rpm = state.rpm_limit

        # Now headers say limit is actually 50
        snap = RateLimitSnapshot(rpm_limit=50, rpm_remaining=48)
        state.update_from_snapshot(snap)
        self.assertEqual(state.rpm_limit, 50)
        self.assertEqual(state._rate_limit_hits, 0)  # reset hits


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rate_limiter.py -v`
Expected: FAIL with "AttributeError: 'RateLimitState' object has no attribute 'update_from_snapshot'"

**Step 3: Write the implementation**

In `src/models/rate_limiter.py`, modify `RateLimitState`:

Add new fields after `_original_tpm`:

```python
    _header_rpm_remaining: int | None = field(default=None, repr=False)
    _header_tpm_remaining: int | None = field(default=None, repr=False)
    _header_rpm_reset_at: float | None = field(default=None, repr=False)
    _header_tpm_reset_at: float | None = field(default=None, repr=False)
    _limits_discovered: bool = field(default=False, repr=False)
    _last_header_update: float = field(default=0.0, repr=False)

    # Daily limits (Cerebras, SambaNova, Gemini)
    rpd_limit: int | None = field(default=None, repr=False)
    rpd_remaining: int | None = field(default=None, repr=False)
    rpd_reset_at: float | None = field(default=None, repr=False)
```

Add import at top of file:

```python
from .header_parser import RateLimitSnapshot
```

Add `update_from_snapshot` method to `RateLimitState`:

```python
    def update_from_snapshot(self, snap: RateLimitSnapshot) -> None:
        """Update state from parsed response headers."""
        now = time.time()
        self._last_header_update = now

        # Update limits if provider reports them
        if snap.rpm_limit is not None and snap.rpm_limit != self.rpm_limit:
            logger.info(
                f"Rate limit discovered: RPM {self.rpm_limit}→{snap.rpm_limit}"
            )
            self.rpm_limit = snap.rpm_limit
            self._original_rpm = snap.rpm_limit
        if snap.tpm_limit is not None and snap.tpm_limit != self.tpm_limit:
            logger.info(
                f"Rate limit discovered: TPM {self.tpm_limit}→{snap.tpm_limit}"
            )
            self.tpm_limit = snap.tpm_limit
            self._original_tpm = snap.tpm_limit

        # If we discovered real limits, clear any adaptive reductions
        if snap.rpm_limit is not None or snap.tpm_limit is not None:
            self._limits_discovered = True
            if self._rate_limit_hits > 0:
                self._rate_limit_hits = 0
                self._last_429_at = 0.0

        # Store remaining counts (ground truth from provider)
        if snap.rpm_remaining is not None:
            self._header_rpm_remaining = snap.rpm_remaining
        if snap.tpm_remaining is not None:
            self._header_tpm_remaining = snap.tpm_remaining

        # Store reset timestamps
        if snap.rpm_reset_at is not None:
            self._header_rpm_reset_at = snap.rpm_reset_at
        if snap.tpm_reset_at is not None:
            self._header_tpm_reset_at = snap.tpm_reset_at

        # Daily limits
        if snap.rpd_limit is not None:
            self.rpd_limit = snap.rpd_limit
        if snap.rpd_remaining is not None:
            self.rpd_remaining = snap.rpd_remaining
        if snap.rpd_reset_at is not None:
            self.rpd_reset_at = snap.rpd_reset_at
```

Modify `has_capacity` to use headers when fresh:

```python
    def has_capacity(self, estimated_tokens: int = 0) -> bool:
        """Check if a request can be made without waiting."""
        # Daily limit exhaustion is absolute
        if self.rpd_remaining is not None and self.rpd_remaining <= 0:
            if self.rpd_reset_at and time.time() < self.rpd_reset_at:
                return False

        now = time.time()
        header_fresh = (now - self._last_header_update) < 5.0

        # Use header-derived remaining when fresh
        if header_fresh and self._header_rpm_remaining is not None:
            rpm_ok = self._header_rpm_remaining > 1
        else:
            rpm_ok = self.rpm_headroom > 1

        if header_fresh and self._header_tpm_remaining is not None:
            tpm_ok = self._header_tpm_remaining > estimated_tokens
        else:
            tpm_ok = self.tpm_headroom > estimated_tokens

        return rpm_ok and tpm_ok
```

Modify `wait_if_needed` to use header reset times:

```python
    async def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """Wait until rate limit allows a request."""
        waited = 0.0
        now = time.time()
        self._cleanup(now)

        # Daily limit check — if exhausted, wait until reset
        if self.rpd_remaining is not None and self.rpd_remaining <= 0:
            if self.rpd_reset_at and self.rpd_reset_at > now:
                # Don't actually wait hours — signal no capacity
                logger.warning(
                    f"Rate limiter: daily limit exhausted, "
                    f"resets in {self.rpd_reset_at - now:.0f}s"
                )
                return -1.0  # signal: skip this model

        header_fresh = (now - self._last_header_update) < 10.0

        # RPM check — prefer header reset time if available
        if header_fresh and self._header_rpm_remaining is not None and self._header_rpm_remaining <= 1:
            if self._header_rpm_reset_at and self._header_rpm_reset_at > now:
                rpm_wait = self._header_rpm_reset_at - now + 0.5
                logger.info(
                    f"Rate limiter: RPM wait {rpm_wait:.1f}s (from headers, "
                    f"remaining={self._header_rpm_remaining})"
                )
                await asyncio.sleep(rpm_wait)
                waited += rpm_wait
                self._header_rpm_remaining = None  # stale after wait
            elif len(self._request_timestamps) >= self.rpm_limit:
                oldest = self._request_timestamps[0]
                rpm_wait = 60 - (now - oldest) + 0.5
                if rpm_wait > 0:
                    logger.info(
                        f"Rate limiter: RPM wait {rpm_wait:.1f}s "
                        f"({self.current_rpm}/{self.rpm_limit})"
                    )
                    await asyncio.sleep(rpm_wait)
                    waited += rpm_wait
                    self._cleanup()
        elif len(self._request_timestamps) >= self.rpm_limit:
            oldest = self._request_timestamps[0]
            rpm_wait = 60 - (now - oldest) + 0.5
            if rpm_wait > 0:
                logger.info(
                    f"Rate limiter: RPM wait {rpm_wait:.1f}s "
                    f"({self.current_rpm}/{self.rpm_limit})"
                )
                await asyncio.sleep(rpm_wait)
                waited += rpm_wait
                self._cleanup()

        # TPM check
        if estimated_tokens > 0 and self.current_tpm + estimated_tokens > self.tpm_limit:
            if header_fresh and self._header_tpm_reset_at and self._header_tpm_reset_at > time.time():
                tpm_wait = self._header_tpm_reset_at - time.time() + 0.5
            elif self._token_log:
                oldest_token_ts = self._token_log[0][0]
                tpm_wait = 60 - (time.time() - oldest_token_ts) + 0.5
            else:
                tpm_wait = 0
            if tpm_wait > 0:
                logger.info(
                    f"Rate limiter: TPM wait {tpm_wait:.1f}s "
                    f"({self.current_tpm}/{self.tpm_limit})"
                )
                await asyncio.sleep(tpm_wait)
                waited += tpm_wait
                self._cleanup()

        self._request_timestamps.append(time.time())
        return waited
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rate_limiter.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/models/rate_limiter.py tests/test_rate_limiter.py
git commit -m "feat: enhance RateLimitState with header-derived limits and daily limits"
```

---

### Task 4: RateLimitManager.update_from_headers + rename hardcoded constants

**Files:**
- Modify: `src/models/rate_limiter.py`
- Modify: `src/models/model_registry.py`
- Modify: `tests/test_rate_limiter.py`

**Step 1: Write the failing test**

Append to `tests/test_rate_limiter.py`:

```python
class TestRateLimitManagerHeaders(unittest.TestCase):

    def _make_manager(self):
        from src.models.rate_limiter import RateLimitManager
        mgr = RateLimitManager()
        mgr.register_model(
            litellm_name="gpt-4o", provider="openai",
            rpm=500, tpm=200000,
            provider_aggregate_rpm=500, provider_aggregate_tpm=2000000,
        )
        mgr.register_model(
            litellm_name="groq/llama-3.3-70b-versatile", provider="groq",
            rpm=30, tpm=131072,
            provider_aggregate_rpm=30, provider_aggregate_tpm=131072,
        )
        return mgr

    def test_update_from_headers_updates_model_state(self):
        mgr = self._make_manager()
        snap = RateLimitSnapshot(
            rpm_limit=600, rpm_remaining=590,
            tpm_limit=300000, tpm_remaining=290000,
        )
        mgr.update_from_headers("gpt-4o", "openai", snap)
        state = mgr.model_limits["gpt-4o"]
        self.assertEqual(state.rpm_limit, 600)
        self.assertEqual(state.tpm_limit, 300000)

    def test_update_from_headers_updates_provider_state(self):
        mgr = self._make_manager()
        snap = RateLimitSnapshot(
            rpm_limit=600, rpm_remaining=590,
            tpm_limit=3000000, tpm_remaining=2900000,
        )
        mgr.update_from_headers("gpt-4o", "openai", snap)
        prov = mgr.provider_limits["openai"]
        self.assertEqual(prov.rpm_limit, 600)
        self.assertEqual(prov.tpm_limit, 3000000)

    def test_update_from_headers_model_and_provider_differ(self):
        """Provider limit can be higher than individual model limit."""
        mgr = self._make_manager()
        # First call: model-specific limits
        snap1 = RateLimitSnapshot(rpm_limit=30, tpm_limit=100000)
        mgr.update_from_headers("groq/llama-3.3-70b-versatile", "groq", snap1)
        # Second call: different model shows provider has higher aggregate
        mgr.register_model(
            litellm_name="groq/llama-3.1-8b-instant", provider="groq",
            rpm=30, tpm=131072,
        )
        snap2 = RateLimitSnapshot(rpm_limit=30, rpm_remaining=28, tpm_remaining=125000)
        mgr.update_from_headers("groq/llama-3.1-8b-instant", "groq", snap2)
        # Each model has own state
        self.assertIn("groq/llama-3.3-70b-versatile", mgr.model_limits)
        self.assertIn("groq/llama-3.1-8b-instant", mgr.model_limits)

    def test_get_status_includes_header_info(self):
        mgr = self._make_manager()
        snap = RateLimitSnapshot(rpm_remaining=490, tpm_remaining=195000)
        mgr.update_from_headers("gpt-4o", "openai", snap)
        status = mgr.get_status()
        model_status = status["models"]["gpt-4o"]
        self.assertIn("discovered", model_status)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rate_limiter.py::TestRateLimitManagerHeaders -v`
Expected: FAIL with "AttributeError: 'RateLimitManager' object has no attribute 'update_from_headers'"

**Step 3: Write the implementation**

Add `update_from_headers` method to `RateLimitManager` in `src/models/rate_limiter.py`:

```python
    def update_from_headers(
        self,
        litellm_name: str,
        provider: str,
        snapshot: RateLimitSnapshot,
    ) -> None:
        """
        Update rate limit state from parsed response headers.
        Updates both model-level and provider-level state.
        """
        # Update model-level state
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            model_state.update_from_snapshot(snapshot)

        # Update provider-level state
        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state.update_from_snapshot(snapshot)
```

Update `get_status` to include discovery info — replace the model loop body:

```python
            models[name] = {
                "rpm": f"{state.current_rpm}/{state.rpm_limit}",
                "tpm": f"{state.current_tpm}/{state.tpm_limit}",
                "utilization_pct": round(state.utilization_pct(), 1),
                "429_hits": state._rate_limit_hits,
                "discovered": state._limits_discovered,
            }
```

Rename `PROVIDER_AGGREGATE_LIMITS` to `_INITIAL_PROVIDER_LIMITS` with updated docstring:

```python
# ─── Initial Provider Limits (fallback before header discovery) ──────────────
# Used only until the first API response with rate limit headers arrives.
# After that, headers are authoritative.
_INITIAL_PROVIDER_LIMITS: dict[str, dict[str, int]] = {
    "groq": {"rpm": 30, "tpm": 131072},
    "gemini": {"rpm": 15, "tpm": 1000000},
    "cerebras": {"rpm": 30, "tpm": 131072},
    "sambanova": {"rpm": 20, "tpm": 100000},
    "openai": {"rpm": 500, "tpm": 2000000},
    "anthropic": {"rpm": 50, "tpm": 80000},
}
```

Update `_init_from_registry` to use the new name:

```python
            agg = _INITIAL_PROVIDER_LIMITS.get(model.provider, {})
```

In `src/models/model_registry.py`, add clarifying comment to `_FREE_TIER_DEFAULTS`:

```python
# Initial rate limit defaults per provider — used as seed values until
# runtime header discovery provides actual limits.
_FREE_TIER_DEFAULTS: dict[str, dict] = {
```

**Step 4: Run all tests**

Run: `python -m pytest tests/test_rate_limiter.py tests/test_header_parser.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/models/rate_limiter.py src/models/model_registry.py tests/test_rate_limiter.py
git commit -m "feat: add RateLimitManager.update_from_headers, rename hardcoded limits to initial fallbacks"
```

---

### Task 5: Quota Planner

**Files:**
- Create: `src/models/quota_planner.py`
- Test: `tests/test_quota_planner.py`

**Step 1: Write the failing test**

```python
# tests/test_quota_planner.py
import sys, os, unittest, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.quota_planner import QuotaPlanner


class TestQuotaPlanner(unittest.TestCase):

    def _make_planner(self):
        return QuotaPlanner()

    def test_default_threshold_is_conservative(self):
        planner = self._make_planner()
        self.assertGreaterEqual(planner.expensive_threshold, 7)

    def test_healthy_quota_lowers_threshold(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 20.0, reset_in=3600)
        planner.update_paid_utilization("anthropic", 15.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(4)
        planner.recalculate()
        self.assertLessEqual(planner.expensive_threshold, 4)

    def test_exhausted_quota_raises_threshold(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 90.0, reset_in=3600)
        planner.update_paid_utilization("anthropic", 85.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(5)
        planner.recalculate()
        self.assertGreaterEqual(planner.expensive_threshold, 8)

    def test_hard_task_queued_reserves_capacity(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 40.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(9)
        planner.recalculate()
        # Threshold should be at least 8 (difficulty-1) to reserve for the 9
        self.assertGreaterEqual(planner.expensive_threshold, 8)

    def test_quota_reset_soon_lowers_threshold(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 70.0, reset_in=120)  # 2 min
        planner.set_max_upcoming_difficulty(4)
        planner.recalculate()
        # Should be more generous since reset is imminent
        self.assertLessEqual(planner.expensive_threshold, 6)

    def test_recent_429s_raise_threshold(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 50.0, reset_in=3600)
        for _ in range(5):
            planner.record_429("openai")
        planner.set_max_upcoming_difficulty(4)
        planner.recalculate()
        self.assertGreaterEqual(planner.expensive_threshold, 7)

    def test_on_quota_restored_triggers_recalculate(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 90.0, reset_in=3600)
        planner.recalculate()
        high = planner.expensive_threshold

        planner.on_quota_restored("openai", new_remaining_pct=80.0)
        self.assertLessEqual(planner.expensive_threshold, high)

    def test_threshold_never_below_1(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 0.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(1)
        planner.recalculate()
        self.assertGreaterEqual(planner.expensive_threshold, 1)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_quota_planner.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# src/models/quota_planner.py
"""
Quota Planner — dynamically adjusts when expensive (paid) models are used.

Decides the minimum difficulty threshold for paid model selection based on:
- Current quota utilization (from response headers)
- Upcoming task difficulty in the queue
- Time until quota resets
- Recent 429 frequency

Never blocks work — just adjusts scoring weights so free models are preferred
when expensive capacity should be reserved.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# How long a 429 event stays relevant for threshold calculation
_429_DECAY_SECONDS = 600  # 10 minutes


class QuotaPlanner:
    """
    Manages the dynamic difficulty threshold for expensive model usage.

    The threshold is an integer 1-10. Tasks with difficulty >= threshold
    get full access to paid models. Tasks below it see paid models
    penalized in scoring (but not blocked).
    """

    def __init__(self):
        self._expensive_threshold: int = 8  # conservative default
        self._paid_utilization: dict[str, float] = {}  # provider → 0-100
        self._paid_reset_in: dict[str, float] = {}  # provider → seconds until reset
        self._max_upcoming_difficulty: int = 0
        self._429_timestamps: list[tuple[str, float]] = []  # (provider, timestamp)
        self._last_recalc: float = 0.0

    @property
    def expensive_threshold(self) -> int:
        return self._expensive_threshold

    def update_paid_utilization(
        self,
        provider: str,
        utilization_pct: float,
        reset_in: float,
    ) -> None:
        """Update current utilization for a paid provider."""
        self._paid_utilization[provider] = utilization_pct
        self._paid_reset_in[provider] = reset_in

    def set_max_upcoming_difficulty(self, difficulty: int) -> None:
        """Set the max difficulty among upcoming queued tasks."""
        self._max_upcoming_difficulty = difficulty

    def record_429(self, provider: str) -> None:
        """Record a rate limit hit on a paid provider."""
        self._429_timestamps.append((provider, time.time()))

    def on_quota_restored(
        self,
        provider: str,
        new_remaining_pct: float,
    ) -> None:
        """Called when headers show quota has been restored."""
        self._paid_utilization[provider] = 100.0 - new_remaining_pct
        logger.info(
            f"Quota restored for {provider} — "
            f"utilization now {100.0 - new_remaining_pct:.0f}%"
        )
        self.recalculate()

    def _recent_429_rate(self) -> int:
        """Count of 429s in the last decay window."""
        cutoff = time.time() - _429_DECAY_SECONDS
        self._429_timestamps = [
            (p, t) for p, t in self._429_timestamps if t > cutoff
        ]
        return len(self._429_timestamps)

    def recalculate(self) -> int:
        """
        Recalculate the expensive model difficulty threshold.

        Returns the new threshold value (1-10).
        """
        now = time.time()
        self._last_recalc = now

        # 1. Overall paid utilization (worst-case across providers)
        if self._paid_utilization:
            paid_util = max(self._paid_utilization.values())
        else:
            paid_util = 50.0  # unknown → moderate assumption

        # 2. Upcoming task difficulty
        max_diff = self._max_upcoming_difficulty

        # 3. Time until reset (minimum across providers)
        if self._paid_reset_in:
            min_reset = min(self._paid_reset_in.values())
        else:
            min_reset = 3600  # unknown → assume 1 hour

        # 4. Recent 429 rate
        recent_429s = self._recent_429_rate()

        # ── Decision logic ──

        if paid_util < 30 and recent_429s == 0:
            threshold = 3
        elif paid_util < 50 and recent_429s <= 1:
            threshold = 5
        elif paid_util < 70:
            threshold = 6
        elif paid_util < 85:
            threshold = 7
        else:
            threshold = 9

        # 429 penalty: each recent 429 pushes threshold up
        if recent_429s >= 3:
            threshold = max(threshold, 8)
        elif recent_429s >= 1:
            threshold = max(threshold, 7)

        # Reserve capacity for hard upcoming tasks
        if max_diff >= 8:
            threshold = max(threshold, max_diff - 1)

        # Quota reset imminent (<5 min) — be more generous
        if min_reset < 300 and paid_util > 40:
            threshold = max(1, threshold - 2)

        threshold = max(1, min(10, threshold))

        if threshold != self._expensive_threshold:
            logger.info(
                f"Quota planner: threshold {self._expensive_threshold}→{threshold} "
                f"(util={paid_util:.0f}%, upcoming_max={max_diff}, "
                f"429s={recent_429s}, reset_in={min_reset:.0f}s)"
            )

        self._expensive_threshold = threshold
        return threshold

    def get_status(self) -> dict:
        """Status for diagnostics."""
        return {
            "expensive_threshold": self._expensive_threshold,
            "paid_utilization": dict(self._paid_utilization),
            "max_upcoming_difficulty": self._max_upcoming_difficulty,
            "recent_429s": self._recent_429_rate(),
        }


# ─── Singleton ───────────────────────────────────────────────
_planner: QuotaPlanner | None = None


def get_quota_planner() -> QuotaPlanner:
    global _planner
    if _planner is None:
        _planner = QuotaPlanner()
    return _planner
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_quota_planner.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/models/quota_planner.py tests/test_quota_planner.py
git commit -m "feat: add QuotaPlanner for dynamic expensive model threshold"
```

---

### Task 6: Wire header parsing into router.py call_model

**Files:**
- Modify: `src/core/router.py` (lines 17-19, 746-756)

**Step 1: Enable litellm response headers**

At top of `src/core/router.py`, after line 19 (`litellm.suppress_debug_info = True`), add:

```python
litellm.return_response_headers = True
```

**Step 2: Add header parsing after successful response**

In `call_model()`, after the token recording block (after line 756 `rl_manager.record_tokens(...)`) and before the "Update measured speed" block, add:

```python
                    # Parse rate limit headers from response
                    if not model.is_local:
                        _update_limits_from_response(response, model, rl_manager)
```

Add the helper function before `call_model`:

```python
def _update_limits_from_response(response, model: ModelInfo, rl_manager) -> None:
    """Extract rate limit headers from litellm response and update state."""
    try:
        from ..models.header_parser import parse_rate_limit_headers
        from ..models.quota_planner import get_quota_planner

        hidden = getattr(response, '_hidden_params', None)
        if not hidden:
            return

        raw_headers = (
            hidden.get('_response_headers')
            or hidden.get('additional_headers')
            or {}
        )
        if not raw_headers:
            return

        # Convert to plain dict if needed
        if hasattr(raw_headers, 'items'):
            header_dict = dict(raw_headers)
        else:
            return

        snapshot = parse_rate_limit_headers(model.provider, header_dict)
        if not snapshot:
            return

        rl_manager.update_from_headers(
            model.litellm_name, model.provider, snapshot,
        )

        # Update quota planner with paid model utilization
        if not model.is_free:
            planner = get_quota_planner()
            # Compute utilization from remaining/limit
            if snapshot.rpm_remaining is not None and snapshot.rpm_limit:
                util = (1 - snapshot.rpm_remaining / snapshot.rpm_limit) * 100
            elif snapshot.tpm_remaining is not None and snapshot.tpm_limit:
                util = (1 - snapshot.tpm_remaining / snapshot.tpm_limit) * 100
            else:
                util = rl_manager.get_provider_utilization(model.provider)

            reset_in = 60.0  # default 1 minute
            if snapshot.rpm_reset_at:
                reset_in = max(1.0, snapshot.rpm_reset_at - time.time())

            planner.update_paid_utilization(model.provider, util, reset_in)

            # Check if quota was restored (remaining jumped up)
            model_state = rl_manager.model_limits.get(model.litellm_name)
            if (model_state and model_state._limits_discovered
                    and snapshot.rpm_remaining is not None
                    and snapshot.rpm_limit is not None
                    and snapshot.rpm_remaining > snapshot.rpm_limit * 0.7):
                planner.on_quota_restored(
                    model.provider,
                    new_remaining_pct=(snapshot.rpm_remaining / snapshot.rpm_limit) * 100,
                )

    except Exception as e:
        logger.debug(f"Header parsing failed: {e}")
```

**Step 3: Also parse headers on 429 errors**

In `call_model()`, in the rate limit error handling block (around the `is_rate_limit` section), after `rl_manager.record_429(...)`, add:

```python
                        from ..models.quota_planner import get_quota_planner
                        get_quota_planner().record_429(model.provider)
```

**Step 4: Run existing tests to ensure no regression**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All existing tests still PASS

**Step 5: Commit**

```bash
git add src/core/router.py
git commit -m "feat: wire header parsing and quota planner updates into call_model"
```

---

### Task 7: Integrate quota planner into model selection scoring

**Files:**
- Modify: `src/core/router.py` (select_model function, cost scoring section ~line 380-403)

**Step 1: Add quota planner penalty for paid models below threshold**

In `select_model()`, replace the cost scoring section for paid models. After the `reasons.append(f"cost=${est_cost:.4f}")` line (line 403), add a quota planner check:

```python
            # Quota planner: penalize paid models for low-difficulty tasks
            from ..models.quota_planner import get_quota_planner
            planner = get_quota_planner()
            if reqs.difficulty < planner.expensive_threshold:
                cost_score = max(5, cost_score - 40)
                reasons.append(f"quota_reserved(thr={planner.expensive_threshold})")
```

This goes inside the `else` block (paid models), right after the cost tier scoring.

**Step 2: Add queue peeking for upcoming task difficulty**

Add a function in `router.py` to peek at the task queue and update the quota planner. This runs at the start of `select_model()`:

```python
_queue_peek_cache: dict[str, float] = {"max_diff": 0, "last_check": 0.0}


def _maybe_update_queue_difficulty() -> None:
    """Peek at task queue to update quota planner (cached 30s)."""
    now = time.time()
    if now - _queue_peek_cache["last_check"] < 30:
        return
    _queue_peek_cache["last_check"] = now

    try:
        import asyncio
        from ..infra.db import get_db

        async def _peek():
            db = await get_db()
            cursor = await db.execute(
                "SELECT context FROM tasks WHERE status = 'pending' LIMIT 20"
            )
            rows = await cursor.fetchall()
            max_diff = 0
            for row in rows:
                try:
                    import json
                    ctx = json.loads(row[0]) if row[0] else {}
                    cls = ctx.get("classification", {})
                    diff = int(cls.get("difficulty", 0))
                    max_diff = max(max_diff, diff)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            return max_diff

        # Try to run in current event loop if available
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context but select_model is sync.
            # Schedule and don't wait — use cached value.
            # The task will update the cache for next call.
            asyncio.ensure_future(_update_queue_diff_async())
        except RuntimeError:
            pass  # No event loop — use cached value

    except Exception as e:
        logger.debug(f"Queue peek failed: {e}")


async def _update_queue_diff_async() -> None:
    """Async helper to peek queue difficulty."""
    try:
        from ..infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT context FROM tasks WHERE status = 'pending' LIMIT 20"
        )
        rows = await cursor.fetchall()
        max_diff = 0
        for row in rows:
            try:
                ctx = json.loads(row[0]) if row[0] else {}
                cls = ctx.get("classification", {})
                diff = int(cls.get("difficulty", 0))
                max_diff = max(max_diff, diff)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        from ..models.quota_planner import get_quota_planner
        get_quota_planner().set_max_upcoming_difficulty(max_diff)
        _queue_peek_cache["max_diff"] = max_diff
    except Exception as e:
        logger.debug(f"Queue difficulty peek failed: {e}")
```

At the top of `select_model()`, add the call:

```python
    _maybe_update_queue_difficulty()
```

**Step 3: Run tests**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/core/router.py
git commit -m "feat: integrate quota planner into model selection scoring with queue peeking"
```

---

### Task 8: Signal backpressure queue on quota restoration

**Files:**
- Modify: `src/models/quota_planner.py`
- Modify: `src/infra/backpressure.py`

**Step 1: Add wake signal to BackpressureQueue**

In `src/infra/backpressure.py`, add a method to `BackpressureQueue`:

```python
    def signal_capacity_available(self) -> None:
        """Signal that model capacity may have been restored."""
        if self._queue:
            # Move all items' next_retry_at to now so they retry immediately
            now = time.time()
            for entry in self._queue:
                if not entry.result_future.done():
                    entry.next_retry_at = now
            self._has_items.set()
            logger.info(
                f"Backpressure: capacity signal received, "
                f"{len(self._queue)} items fast-tracked for retry"
            )
```

**Step 2: Call signal from quota planner's on_quota_restored**

In `src/models/quota_planner.py`, update `on_quota_restored` to signal the backpressure queue:

```python
    def on_quota_restored(
        self,
        provider: str,
        new_remaining_pct: float,
    ) -> None:
        """Called when headers show quota has been restored."""
        self._paid_utilization[provider] = 100.0 - new_remaining_pct
        logger.info(
            f"Quota restored for {provider} — "
            f"utilization now {100.0 - new_remaining_pct:.0f}%"
        )
        self.recalculate()

        # Signal backpressure queue to retry waiting calls
        try:
            from ..infra.backpressure import get_backpressure_queue
            get_backpressure_queue().signal_capacity_available()
        except Exception:
            pass  # Queue may not be initialized yet
```

**Step 3: Run tests**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/models/quota_planner.py src/infra/backpressure.py
git commit -m "feat: signal backpressure queue on quota restoration"
```

---

### Task 9: Handle daily limit exhaustion in call_model

**Files:**
- Modify: `src/core/router.py` (call_model, rate limiter wait section)

**Step 1: Handle -1 return from wait_and_acquire**

In `call_model()`, the rate limiter `wait_and_acquire` section (around line 681), modify to handle daily limit exhaustion. The `wait_if_needed` method now returns -1.0 when daily limits are exhausted:

In `RateLimitManager.wait_and_acquire`, add handling:

```python
    async def wait_and_acquire(
        self,
        litellm_name: str,
        provider: str,
        estimated_tokens: int = 0,
    ) -> float:
        """
        Wait for both model and provider limits, then record request.
        Returns total seconds waited, or -1.0 if daily limit exhausted.
        """
        total_waited = 0.0

        model_state = self.model_limits.get(litellm_name)
        if model_state:
            result = await model_state.wait_if_needed(estimated_tokens)
            if result < 0:
                return -1.0  # daily limit exhausted
            total_waited += result

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            result = await provider_state.wait_if_needed(estimated_tokens)
            if result < 0:
                return -1.0
            total_waited += result

        return total_waited
```

In `call_model()`, after the `wait_and_acquire` call, check the return value:

```python
            wait_time = await rl_manager.wait_and_acquire(
                litellm_name=model.litellm_name,
                provider=model.provider,
                estimated_tokens=estimated_tokens,
            )
            if wait_time < 0:
                # Daily limit exhausted — skip this model
                logger.warning(
                    f"Daily limit exhausted for {model.name}, skipping"
                )
                last_error = f"Daily limit exhausted for {model.name}"
                continue
            if wait_time > 0:
                logger.info(
                    f"Rate limiter waited {wait_time:.1f}s for "
                    f"{model.name}"
                )
```

**Step 2: Run tests**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/core/router.py src/models/rate_limiter.py
git commit -m "feat: handle daily rate limit exhaustion by skipping to next model"
```

---

### Task 10: Integration test — end-to-end header flow

**Files:**
- Create: `tests/test_rate_limit_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_rate_limit_integration.py
"""Integration test: header parsing → rate limiter update → quota planner."""
import sys, os, unittest, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.header_parser import parse_rate_limit_headers
from src.models.rate_limiter import RateLimitManager
from src.models.quota_planner import QuotaPlanner


class TestRateLimitIntegration(unittest.TestCase):

    def test_full_flow_openai_headers(self):
        """Simulate: API response → header parse → limiter update → planner update."""
        mgr = RateLimitManager()
        mgr.register_model("gpt-4o", "openai", rpm=500, tpm=200000,
                           provider_aggregate_rpm=500, provider_aggregate_tpm=2000000)
        planner = QuotaPlanner()

        # Simulate response headers from OpenAI
        headers = {
            "x-ratelimit-limit-requests": "5000",
            "x-ratelimit-limit-tokens": "800000",
            "x-ratelimit-remaining-requests": "4990",
            "x-ratelimit-remaining-tokens": "790000",
            "x-ratelimit-reset-requests": "12ms",
            "x-ratelimit-reset-tokens": "6s",
        }
        snap = parse_rate_limit_headers("openai", headers)
        self.assertIsNotNone(snap)

        # Limits should be updated (provider tier upgrade detected)
        mgr.update_from_headers("gpt-4o", "openai", snap)
        self.assertEqual(mgr.model_limits["gpt-4o"].rpm_limit, 5000)
        self.assertTrue(mgr.model_limits["gpt-4o"]._limits_discovered)

        # Planner should lower threshold (plenty of capacity)
        util = (1 - 4990 / 5000) * 100
        planner.update_paid_utilization("openai", util, reset_in=60)
        planner.set_max_upcoming_difficulty(4)
        planner.recalculate()
        self.assertLessEqual(planner.expensive_threshold, 4)

    def test_full_flow_anthropic_near_limit(self):
        """Near rate limit → planner raises threshold."""
        mgr = RateLimitManager()
        mgr.register_model("claude-sonnet-4-20250514", "anthropic",
                           rpm=50, tpm=80000,
                           provider_aggregate_rpm=50, provider_aggregate_tpm=80000)
        planner = QuotaPlanner()

        headers = {
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "3",
            "anthropic-ratelimit-requests-reset": "2026-03-16T12:00:30Z",
            "anthropic-ratelimit-tokens-limit": "80000",
            "anthropic-ratelimit-tokens-remaining": "5000",
            "anthropic-ratelimit-tokens-reset": "2026-03-16T12:00:30Z",
        }
        snap = parse_rate_limit_headers("anthropic", headers)
        mgr.update_from_headers("claude-sonnet-4-20250514", "anthropic", snap)

        util = (1 - 3 / 50) * 100  # 94% used
        planner.update_paid_utilization("anthropic", util, reset_in=30)
        planner.set_max_upcoming_difficulty(5)
        planner.recalculate()
        self.assertGreaterEqual(planner.expensive_threshold, 8)

    def test_cerebras_daily_limit_exhaustion(self):
        """Cerebras daily limit hits 0 → has_capacity returns False."""
        mgr = RateLimitManager()
        mgr.register_model("cerebras/llama3.3-70b", "cerebras",
                           rpm=30, tpm=131072,
                           provider_aggregate_rpm=30, provider_aggregate_tpm=131072)

        headers = {
            "x-ratelimit-limit-tokens-minute": "131072",
            "x-ratelimit-remaining-tokens-minute": "100000",
            "x-ratelimit-reset-tokens-minute": "45.5",
            "x-ratelimit-limit-requests-day": "1000",
            "x-ratelimit-remaining-requests-day": "0",
            "x-ratelimit-reset-requests-day": "33011",
        }
        snap = parse_rate_limit_headers("cerebras", headers)
        mgr.update_from_headers("cerebras/llama3.3-70b", "cerebras", snap)

        # Daily limit exhausted → no capacity
        self.assertFalse(
            mgr.has_capacity("cerebras/llama3.3-70b", "cerebras", 1000)
        )

    def test_limit_discovery_overrides_hardcoded(self):
        """Headers showing higher limit replace hardcoded defaults."""
        mgr = RateLimitManager()
        mgr.register_model("groq/llama-3.3-70b-versatile", "groq",
                           rpm=30, tpm=131072,
                           provider_aggregate_rpm=30, provider_aggregate_tpm=131072)

        # First: verify initial hardcoded limits
        state = mgr.model_limits["groq/llama-3.3-70b-versatile"]
        self.assertEqual(state.rpm_limit, 30)

        # Simulate headers showing higher limit (user upgraded tier)
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "500000",
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-remaining-tokens": "499000",
        }
        snap = parse_rate_limit_headers("groq", headers)
        mgr.update_from_headers("groq/llama-3.3-70b-versatile", "groq", snap)

        # Limits updated to what headers say
        self.assertEqual(state.rpm_limit, 100)
        self.assertEqual(state.tpm_limit, 500000)
        self.assertTrue(state._limits_discovered)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_rate_limit_integration.py -v`
Expected: All 4 tests PASS

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS, no regressions

**Step 4: Commit**

```bash
git add tests/test_rate_limit_integration.py
git commit -m "test: add end-to-end integration tests for dynamic rate limits"
```
