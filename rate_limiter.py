# rate_limiter.py
"""
Two-tier rate limiting: per-model + per-provider.

Providers like Groq enforce per-model RPM/TPM limits AND
per-account aggregate limits. We track both layers.

Also supports adaptive limit discovery: if we get a 429 at
a rate below our configured limit, we automatically lower
the limit.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

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
        return (
            self.rpm_headroom > 1
            and self.tpm_headroom > estimated_tokens
        )

    def utilization_pct(self) -> float:
        """How close to limits we are (0-100)."""
        rpm_pct = (self.current_rpm / self.rpm_limit * 100) if self.rpm_limit else 0
        tpm_pct = (self.current_tpm / self.tpm_limit * 100) if self.tpm_limit else 0
        return max(rpm_pct, tpm_pct)

    async def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """
        Wait until rate limit allows a request.
        Returns seconds waited (0 if no wait needed).
        """
        waited = 0.0
        now = time.time()
        self._cleanup(now)

        # RPM check
        if len(self._request_timestamps) >= self.rpm_limit:
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
            # Wait for oldest token entries to expire
            if self._token_log:
                oldest_token_ts = self._token_log[0][0]
                tpm_wait = 60 - (time.time() - oldest_token_ts) + 0.5
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


class RateLimitManager:
    """
    Two-tier rate limiting manager.

    Tier 1: Per-model limits (e.g., groq/llama-8b has its own RPM)
    Tier 2: Per-provider aggregate (e.g., all Groq models share account quota)

    A request must pass BOTH tiers to proceed.
    """

    def __init__(self):
        self._model_limits: dict[str, RateLimitState] = {}
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
        if litellm_name not in self._model_limits:
            self._model_limits[litellm_name] = RateLimitState(
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
        model_state = self._model_limits.get(litellm_name)
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
        Returns total seconds waited.
        """
        total_waited = 0.0

        # Wait on model limit first
        model_state = self._model_limits.get(litellm_name)
        if model_state:
            total_waited += await model_state.wait_if_needed(estimated_tokens)

        # Then wait on provider aggregate
        provider_state = self._provider_limits.get(provider)
        if provider_state:
            total_waited += await provider_state.wait_if_needed(estimated_tokens)

        return total_waited

    def record_tokens(
        self,
        litellm_name: str,
        provider: str,
        token_count: int,
    ) -> None:
        """Record actual token usage after a call completes."""
        model_state = self._model_limits.get(litellm_name)
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
        model_state = self._model_limits.get(litellm_name)
        if model_state:
            model_state.record_429()

        provider_state = self._provider_limits.get(provider)
        if provider_state:
            provider_state.record_429()

    def restore_limits(self) -> None:
        """Periodically called to gradually restore adapted limits."""
        for state in self._model_limits.values():
            state.maybe_restore_limits()
        for state in self._provider_limits.values():
            state.maybe_restore_limits()

    def get_utilization(self, litellm_name: str) -> float:
        """Get utilization percentage for a model (0-100)."""
        state = self._model_limits.get(litellm_name)
        return state.utilization_pct() if state else 0.0

    def get_provider_utilization(self, provider: str) -> float:
        """Get utilization percentage for a provider (0-100)."""
        state = self._provider_limits.get(provider)
        return state.utilization_pct() if state else 0.0

    def get_status(self) -> dict:
        """Full status for diagnostics."""
        models = {}
        for name, state in self._model_limits.items():
            models[name] = {
                "rpm": f"{state.current_rpm}/{state.rpm_limit}",
                "tpm": f"{state.current_tpm}/{state.tpm_limit}",
                "utilization_pct": round(state.utilization_pct(), 1),
                "429_hits": state._rate_limit_hits,
            }

        providers = {}
        for name, state in self._provider_limits.items():
            providers[name] = {
                "rpm": f"{state.current_rpm}/{state.rpm_limit}",
                "tpm": f"{state.current_tpm}/{state.tpm_limit}",
                "utilization_pct": round(state.utilization_pct(), 1),
                "429_hits": state._rate_limit_hits,
            }

        return {"models": models, "providers": providers}


# ─── Provider-Specific Aggregate Limits ──────────────────────────────────────
# These are account-level limits for free tiers (as of mid-2025).
# Per-model limits come from models.yaml via the registry.

PROVIDER_AGGREGATE_LIMITS: dict[str, dict[str, int]] = {
    "groq": {"rpm": 30, "tpm": 131072},
    "gemini": {"rpm": 15, "tpm": 1000000},
    "cerebras": {"rpm": 30, "tpm": 131072},
    "sambanova": {"rpm": 20, "tpm": 100000},
    "openai": {"rpm": 500, "tpm": 2000000},
    "anthropic": {"rpm": 50, "tpm": 80000},
}


# ─── Singleton ───────────────────────────────────────────────
_manager: RateLimitManager | None = None


def get_rate_limit_manager() -> RateLimitManager:
    global _manager
    if _manager is None:
        _manager = RateLimitManager()
        _init_from_registry()
    return _manager


def _init_from_registry() -> None:
    """Auto-register all cloud models from the model registry."""
    try:
        from model_registry import get_registry
        registry = get_registry()
        manager = get_rate_limit_manager()

        for model in registry.cloud_models():
            agg = PROVIDER_AGGREGATE_LIMITS.get(model.provider, {})
            manager.register_model(
                litellm_name=model.litellm_name,
                provider=model.provider,
                rpm=model.rate_limit_rpm,
                tpm=model.rate_limit_tpm,
                provider_aggregate_rpm=agg.get("rpm"),
                provider_aggregate_tpm=agg.get("tpm"),
            )

        logger.info(
            f"Rate limits initialized: "
            f"{len(manager._model_limits)} models, "
            f"{len(manager._provider_limits)} providers"
        )
    except Exception as e:
        logger.warning(f"Rate limit initialization failed: {e}")
