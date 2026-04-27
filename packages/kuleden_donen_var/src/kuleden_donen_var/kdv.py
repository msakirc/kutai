"""KuledenDonenVar — main class composing all modules."""
from __future__ import annotations

import logging
import time
from typing import Any

from .circuit_breaker import CircuitBreaker
from .config import (
    CapacityEvent,
    KuledenConfig,
    ModelStatus,
    PreCallResult,
    ProviderStatus,
)
from .header_parser import parse_rate_limit_headers
from .rate_limiter import RateLimitManager

logger = logging.getLogger(__name__)


class KuledenDonenVar:
    def __init__(self, config: KuledenConfig):
        self._config = config
        self._rate_limiter = RateLimitManager()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._providers: dict[str, set[str]] = {}  # provider → {model_ids}
        self._provider_enabled_at: dict[str, float] = {}
        self._provider_call_count: dict[str, int] = {}

    def _get_cb(self, provider: str) -> CircuitBreaker:
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreaker(
                failure_threshold=self._config.circuit_breaker_threshold,
                window_seconds=self._config.circuit_breaker_window_seconds,
                cooldown_seconds=self._config.circuit_breaker_cooldown_seconds,
            )
        return self._circuit_breakers[provider]

    def _fire(self, provider: str, model_id: str | None, event_type: str) -> None:
        if self._config.on_capacity_change is None:
            return
        snapshot = self._build_provider_status(provider)
        evt = CapacityEvent(
            provider=provider,
            model_id=model_id,
            event_type=event_type,
            snapshot=snapshot,
        )
        try:
            self._config.on_capacity_change(evt)
        except Exception:
            logger.exception("on_capacity_change callback failed")

    def register(
        self,
        model_id: str,
        provider: str,
        rpm: int,
        tpm: int,
        provider_aggregate_rpm: int | None = None,
        provider_aggregate_tpm: int | None = None,
    ) -> None:
        self._rate_limiter.register_model(
            litellm_name=model_id,
            provider=provider,
            rpm=rpm,
            tpm=tpm,
            provider_aggregate_rpm=provider_aggregate_rpm,
            provider_aggregate_tpm=provider_aggregate_tpm,
        )
        self._providers.setdefault(provider, set()).add(model_id)

    def mark_provider_enabled(self, provider: str, at_unix: float | None = None) -> None:
        """Record when a provider was first enabled (boot or first auth_ok refresh).

        Idempotent: re-marking the same provider does NOT update the timestamp
        — the periodic 6h refresh would otherwise reset the clock and the
        no-data warning could never fire. Trade-off: a provider that goes
        auth_fail then recovers will appear "idle" for the original duration,
        not just the post-recovery duration. Bridge callers that detect a
        recovery transition and want a fresh clock should call
        ``reset_provider_enabled`` first.
        """
        if provider in self._provider_enabled_at:
            return
        self._provider_enabled_at[provider] = (
            at_unix if at_unix is not None else time.time()
        )

    def reset_provider_enabled(self, provider: str) -> None:
        """Drop the enabled-at timestamp + observation count for a provider.

        Use on auth_fail→ok recovery to restart the no-data-warning clock.
        Caller is responsible for re-marking afterward.
        """
        self._provider_enabled_at.pop(provider, None)
        self._provider_call_count.pop(provider, None)

    def record_call_observation(self, provider: str) -> None:
        """Bump per-provider observation count. Wired internally from post_call.

        Public surface so external callers (tests, manual reconciliation) can
        inject observations too.
        """
        self._provider_call_count[provider] = self._provider_call_count.get(provider, 0) + 1

    def no_data_warnings(self, min_age_hours: float = 24.0) -> list[dict]:
        """Return list of providers enabled longer than ``min_age_hours`` with
        zero observations.

        Each entry: ``{"provider": str, "enabled_at_unix": float, "age_hours": float}``.
        Caller (status command, scheduled task) uses this to surface "defaults
        still in use" warnings to the operator.
        """
        now = time.time()
        out: list[dict] = []
        for provider, enabled_at in self._provider_enabled_at.items():
            age_hours = (now - enabled_at) / 3600.0
            if age_hours < min_age_hours:
                continue
            if self._provider_call_count.get(provider, 0) > 0:
                continue
            out.append({
                "provider": provider,
                "enabled_at_unix": enabled_at,
                "age_hours": age_hours,
            })
        return out

    def pre_call(
        self,
        model_id: str,
        provider: str,
        estimated_tokens: int = 0,
    ) -> PreCallResult:
        # Circuit breaker check
        cb = self._get_cb(provider)
        if cb.is_degraded:
            return PreCallResult(allowed=False, wait_seconds=0.0, daily_exhausted=False)

        # Daily limit check
        if self._rate_limiter.is_daily_exhausted(model_id):
            return PreCallResult(allowed=False, wait_seconds=0.0, daily_exhausted=True)

        # Rate limit capacity check
        if not self._rate_limiter.has_capacity(model_id, provider, estimated_tokens):
            state = self._rate_limiter.model_limits.get(model_id)
            wait = 0.0
            if state and state._request_timestamps:
                oldest = state._request_timestamps[0]
                wait = max(0, 60 - (time.time() - oldest) + 0.5)
            return PreCallResult(allowed=False, wait_seconds=wait, daily_exhausted=False)

        return PreCallResult(allowed=True, wait_seconds=0.0, daily_exhausted=False)

    def post_call(
        self,
        model_id: str,
        provider: str,
        headers: dict[str, Any] | None,
        token_count: int,
    ) -> None:
        # Track real-call observation count for no_data_warnings.
        self.record_call_observation(provider)
        # Record request (RPM tracking) and tokens (TPM tracking)
        self._rate_limiter.record_request(model_id, provider)
        self._rate_limiter.record_tokens(model_id, provider, token_count)

        # Parse and apply response headers
        if headers:
            snapshot = parse_rate_limit_headers(provider, headers)
            if snapshot is not None:
                prev_state = self._rate_limiter.model_limits.get(model_id)
                prev_rpm_remaining = prev_state._header_rpm_remaining if prev_state else None

                self._rate_limiter.update_from_headers(model_id, provider, snapshot)

                # Fire capacity_restored if significant improvement
                if (prev_rpm_remaining is not None
                        and snapshot.rpm_remaining is not None
                        and prev_rpm_remaining <= 1
                        and snapshot.rpm_remaining > 5):
                    self._fire(provider, model_id, "capacity_restored")

        # Circuit breaker success
        cb = self._get_cb(provider)
        was_degraded = cb.is_degraded
        cb.record_success()
        if was_degraded:
            self._fire(provider, model_id, "circuit_breaker_reset")

    def record_failure(
        self,
        model_id: str,
        provider: str,
        error_type: str,
    ) -> None:
        if error_type == "rate_limit":
            self._rate_limiter.record_429(model_id, provider)
            self._fire(provider, model_id, "limit_hit")
        elif error_type in ("server_error", "timeout"):
            cb = self._get_cb(provider)
            was_degraded = cb.is_degraded
            cb.record_failure()
            if cb.is_degraded and not was_degraded:
                self._fire(provider, model_id, "circuit_breaker_tripped")
        # auth errors: not tracked (permanent, not transient)

    def _build_provider_status(self, provider: str) -> ProviderStatus:
        cb = self._get_cb(provider)
        model_ids = self._providers.get(provider, set())

        models: dict[str, ModelStatus] = {}
        worst_util = 0.0
        earliest_reset: float | None = None

        for mid in model_ids:
            state = self._rate_limiter.model_limits.get(mid)
            if state is None:
                models[mid] = ModelStatus(model_id=mid)
                continue

            util = state.utilization_pct()
            worst_util = max(worst_util, util)

            daily_exhausted = (
                state.rpd_remaining is not None
                and state.rpd_remaining <= 0
                and state.rpd_reset_at is not None
                and time.time() < state.rpd_reset_at
            )

            for reset_at in (state._header_rpm_reset_at, state._header_tpm_reset_at, state.rpd_reset_at):
                if reset_at is not None:
                    remaining = reset_at - time.time()
                    if remaining > 0:
                        if earliest_reset is None or remaining < earliest_reset:
                            earliest_reset = remaining

            models[mid] = ModelStatus(
                model_id=mid,
                utilization_pct=util,
                has_capacity=state.has_capacity(),
                daily_exhausted=daily_exhausted,
                rpm_remaining=state._header_rpm_remaining,
                tpm_remaining=state._header_tpm_remaining,
                rpd_remaining=state.rpd_remaining,
            )

        prov_state = self._rate_limiter._provider_limits.get(provider)
        prov_util = prov_state.utilization_pct() if prov_state else worst_util

        return ProviderStatus(
            provider=provider,
            circuit_breaker_open=cb.is_degraded,
            utilization_pct=max(worst_util, prov_util),
            rpm_remaining=prov_state._header_rpm_remaining if prov_state else None,
            tpm_remaining=prov_state._header_tpm_remaining if prov_state else None,
            rpd_remaining=None,
            reset_in_seconds=earliest_reset,
            models=models,
        )

    def restore_limits(self) -> None:
        """Gradually restore adaptive rate limit reductions. Called by watchdog."""
        self._rate_limiter.restore_limits()

    @property
    def status(self) -> dict[str, ProviderStatus]:
        return {
            provider: self._build_provider_status(provider)
            for provider in self._providers
        }
