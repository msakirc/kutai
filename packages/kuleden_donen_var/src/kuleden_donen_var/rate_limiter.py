# rate_limiter.py
"""
Two-tier rate limiting: per-model + per-provider.

Providers like Groq enforce per-model RPM/TPM limits AND
per-account aggregate limits. We track both layers.

Supports:
- Adaptive limit discovery from 429 errors
- Dynamic limit updates from API response headers
- Daily limit tracking (Cerebras, SambaNova, Gemini)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .header_parser import RateLimitSnapshot

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Tracks request and token usage within a sliding window."""
    rpm_limit: int
    tpm_limit: int
    _request_timestamps: list[float] = field(default_factory=list)
    _token_log: list[tuple[float, int]] = field(default_factory=list)
    _rate_limit_hits: int = 0
    _last_429_at: float = 0.0
    _original_rpm: int = 0
    _original_tpm: int = 0

    # Header-derived live state
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

    def __post_init__(self):
        self._original_rpm = self.rpm_limit
        self._original_tpm = self.tpm_limit

    def _cleanup(self, now: float | None = None) -> None:
        """Remove entries older than 60 seconds."""
        now = now or time.time()
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < 60
        ]
        self._token_log = [
            (t, c) for t, c in self._token_log if now - t < 60
        ]

    @property
    def current_rpm(self) -> int:
        self._cleanup()
        return len(self._request_timestamps)

    @property
    def current_tpm(self) -> int:
        self._cleanup()
        return sum(c for _, c in self._token_log)

    @property
    def rpm_headroom(self) -> int:
        return max(0, self.rpm_limit - self.current_rpm)

    @property
    def tpm_headroom(self) -> int:
        return max(0, self.tpm_limit - self.current_tpm)

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

    def utilization_pct(self) -> float:
        """How close to limits we are (0-100)."""
        rpm_pct = (self.current_rpm / self.rpm_limit * 100) if self.rpm_limit else 0
        tpm_pct = (self.current_tpm / self.tpm_limit * 100) if self.tpm_limit else 0
        return max(rpm_pct, tpm_pct)

    async def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """
        Wait until rate limit allows a request.
        Returns seconds waited (0 if no wait needed).
        Returns -1.0 if daily limit exhausted (caller should skip model).
        """
        waited = 0.0
        now = time.time()
        self._cleanup(now)

        # Daily limit check — if exhausted, signal skip
        if self.rpd_remaining is not None and self.rpd_remaining <= 0:
            if self.rpd_reset_at and self.rpd_reset_at > now:
                logger.warning(
                    f"Rate limiter: daily limit exhausted, "
                    f"resets in {self.rpd_reset_at - now:.0f}s"
                )
                return -1.0

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

    def record_tokens(self, token_count: int) -> None:
        """Record actual token usage after a call completes."""
        self._token_log.append((time.time(), token_count))

    def record_429(self) -> None:
        """
        Record a rate limit hit. Adaptively lower limits.

        If we're getting 429s below our configured limit, the provider's
        actual limit is lower than what we think. Reduce by 20% each hit,
        minimum 50% of original.
        """
        self._rate_limit_hits += 1
        self._last_429_at = time.time()

        # Adaptive reduction
        min_rpm = max(1, self._original_rpm // 2)
        min_tpm = max(1000, self._original_tpm // 2)

        new_rpm = max(min_rpm, int(self.rpm_limit * 0.8))
        new_tpm = max(min_tpm, int(self.tpm_limit * 0.8))

        if new_rpm < self.rpm_limit or new_tpm < self.tpm_limit:
            logger.warning(
                f"Rate limit adapted: RPM {self.rpm_limit}→{new_rpm}, "
                f"TPM {self.tpm_limit}→{new_tpm} "
                f"(hit #{self._rate_limit_hits})"
            )
            self.rpm_limit = new_rpm
            self.tpm_limit = new_tpm

    def maybe_restore_limits(self) -> None:
        """
        If we haven't hit a 429 in 10 minutes, gradually restore limits.
        Called periodically by the watchdog.
        """
        if self._rate_limit_hits == 0:
            return

        now = time.time()
        if now - self._last_429_at < 600:  # 10 minutes
            return

        # Restore 10% toward original
        if self.rpm_limit < self._original_rpm:
            self.rpm_limit = min(
                self._original_rpm,
                self.rpm_limit + max(1, self._original_rpm // 10),
            )
        if self.tpm_limit < self._original_tpm:
            self.tpm_limit = min(
                self._original_tpm,
                self.tpm_limit + max(100, self._original_tpm // 10),
            )

        logger.debug(
            f"Rate limits partially restored: "
            f"RPM={self.rpm_limit}/{self._original_rpm}, "
            f"TPM={self.tpm_limit}/{self._original_tpm}"
        )

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


class RateLimitManager:
    """
    Two-tier rate limiting manager.

    Tier 1: Per-model limits (e.g., groq/llama-8b has its own RPM)
    Tier 2: Per-provider aggregate (e.g., all Groq models share account quota)

    A request must pass BOTH tiers to proceed.
    """

    def __init__(self):
        self.model_limits: dict[str, RateLimitState] = {}
        self._provider_limits: dict[str, RateLimitState] = {}

    def register_model(
        self,
        litellm_name: str,
        provider: str,
        rpm: int,
        tpm: int,
        provider_aggregate_rpm: int | None = None,
        provider_aggregate_tpm: int | None = None,
    ) -> None:
        """
        Register rate limits for a model.

        Args:
            litellm_name: unique model identifier
            provider: provider name (for aggregate tracking)
            rpm: per-model requests per minute
            tpm: per-model tokens per minute
            provider_aggregate_rpm: shared provider-level RPM (optional)
            provider_aggregate_tpm: shared provider-level TPM (optional)
        """
        if litellm_name not in self.model_limits:
            self.model_limits[litellm_name] = RateLimitState(
                rpm_limit=rpm, tpm_limit=tpm,
            )

        # Provider aggregate — only create once per provider
        if provider not in self._provider_limits:
            agg_rpm = provider_aggregate_rpm or rpm * 3  # heuristic
            agg_tpm = provider_aggregate_tpm or tpm * 3
            self._provider_limits[provider] = RateLimitState(
                rpm_limit=agg_rpm, tpm_limit=agg_tpm,
            )

    def has_capacity(
        self,
        litellm_name: str,
        provider: str,
        estimated_tokens: int = 0,
    ) -> bool:
        """Check if both model and provider have capacity."""
        model_state = self.model_limits.get(litellm_name)
        provider_state = self._provider_limits.get(provider)

        model_ok = model_state.has_capacity(estimated_tokens) if model_state else True
        provider_ok = provider_state.has_capacity(estimated_tokens) if provider_state else True

        return model_ok and provider_ok

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

        # Wait on model limit first
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            result = await model_state.wait_if_needed(estimated_tokens)
            if result < 0:
                return -1.0
            total_waited += result

        # Then wait on provider aggregate
        provider_state = self._provider_limits.get(provider)
        if provider_state:
            result = await provider_state.wait_if_needed(estimated_tokens)
            if result < 0:
                return -1.0
            total_waited += result

        return total_waited

    def record_tokens(
        self,
        litellm_name: str,
        provider: str,
        token_count: int,
    ) -> None:
        """Record actual token usage after a call completes."""
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            model_state.record_tokens(token_count)

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state.record_tokens(token_count)

    def record_429(
        self,
        litellm_name: str,
        provider: str,
    ) -> None:
        """Record a rate limit error — adaptively reduces limits."""
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            model_state.record_429()

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state.record_429()

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
        model_state = self.model_limits.get(litellm_name)
        if model_state:
            model_state.update_from_snapshot(snapshot)

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state.update_from_snapshot(snapshot)

    def restore_limits(self) -> None:
        """Periodically called to gradually restore adapted limits."""
        for state in self.model_limits.values():
            state.maybe_restore_limits()
        for state in self._provider_limits.values():
            state.maybe_restore_limits()

    def get_utilization(self, litellm_name: str) -> float:
        """Get utilization percentage for a model (0-100)."""
        state = self.model_limits.get(litellm_name)
        return state.utilization_pct() if state else 0.0

    def get_provider_utilization(self, provider: str) -> float:
        """Get utilization percentage for a provider (0-100)."""
        state = self._provider_limits.get(provider)
        return state.utilization_pct() if state else 0.0

    def is_daily_exhausted(self, litellm_name: str) -> bool:
        """Check if a model's daily request limit is exhausted."""
        state = self.model_limits.get(litellm_name)
        if not state:
            return False
        if state.rpd_remaining is not None and state.rpd_remaining <= 0:
            if state.rpd_reset_at and time.time() < state.rpd_reset_at:
                return True
        return False

    def get_status(self) -> dict:
        """Full status for diagnostics."""
        models = {}
        for name, state in self.model_limits.items():
            models[name] = {
                "rpm": f"{state.current_rpm}/{state.rpm_limit}",
                "tpm": f"{state.current_tpm}/{state.tpm_limit}",
                "utilization_pct": round(state.utilization_pct(), 1),
                "429_hits": state._rate_limit_hits,
                "discovered": state._limits_discovered,
            }

        providers = {}
        for name, state in self._provider_limits.items():
            providers[name] = {
                "rpm": f"{state.current_rpm}/{state.rpm_limit}",
                "tpm": f"{state.current_tpm}/{state.tpm_limit}",
                "utilization_pct": round(state.utilization_pct(), 1),
                "429_hits": state._rate_limit_hits,
                "discovered": state._limits_discovered,
            }

        return {"models": models, "providers": providers}

    @property
    def provider_limits(self):
        return self._provider_limits
