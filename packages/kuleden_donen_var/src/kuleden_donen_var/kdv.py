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
        # Two counters with deliberately different semantics:
        #   _provider_call_count    — successes only (post_call). Drives
        #     downstream observability (success rate per provider). Kept
        #     unchanged for backward compatibility with existing call sites.
        #   _provider_attempt_count — every outgoing API call regardless of
        #     outcome (record_attempt). Drives no_data_warnings: a provider
        #     getting hammered with 4xx is "live", not "no data".
        self._provider_call_count: dict[str, int] = {}
        self._provider_attempt_count: dict[str, int] = {}

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
        """Drop the enabled-at timestamp + both counters for a provider.

        Use on auth_fail→ok recovery to restart the no-data-warning clock.
        Caller is responsible for re-marking afterward.
        """
        self._provider_enabled_at.pop(provider, None)
        self._provider_call_count.pop(provider, None)
        self._provider_attempt_count.pop(provider, None)

    def record_call_observation(self, provider: str) -> None:
        """Bump per-provider success count. Wired internally from post_call.

        Public surface so external callers (tests, manual reconciliation) can
        inject observations too.
        """
        self._provider_call_count[provider] = self._provider_call_count.get(provider, 0) + 1

    def record_attempt(
        self,
        model_id: str,
        provider: str,
        headers: dict[str, Any] | None = None,
    ) -> None:
        """Record an outgoing API call attempt — fires BEFORE LiteLLM POST and
        again from the failure path so RPM bookkeeping reflects what actually
        hit the provider (not just what succeeded).

        Effects:
          * ``_provider_attempt_count[provider]`` += 1
          * ``RateLimitManager.record_request`` bumps the per-minute sliding
            window for both per-model and per-provider-aggregate buckets.
          * BurnLog gets a ``calls=1, tokens=0`` entry (S7 burn-rate signal).
          * If headers are provided (e.g. captured from a 429 response),
            ``update_from_headers`` parses x-ratelimit-* into authoritative
            ``_header_rpm/tpm/rpd_remaining`` + reset clocks.

        Tokens are NOT counted here — actual usage is unknown until response.
        post_call() bumps tokens on success. Failed attempts contribute zero
        tokens but still consume the per-call slot, matching what Groq counts.
        """
        self._provider_attempt_count[provider] = (
            self._provider_attempt_count.get(provider, 0) + 1
        )
        self._rate_limiter.record_request(model_id, provider)
        try:
            from nerd_herd.burn_log import get_burn_log
            get_burn_log().record(provider=provider, model=model_id,
                                  tokens=0, calls=1)
        except Exception:
            pass
        if headers:
            try:
                snapshot = parse_rate_limit_headers(provider, headers)
                if snapshot is not None:
                    self._rate_limiter.update_from_headers(model_id, provider,
                                                          snapshot)
            except Exception:
                pass

    def no_data_warnings(self, min_age_hours: float = 24.0) -> list[dict]:
        """Return providers enabled longer than ``min_age_hours`` with zero
        attempts. A provider returning 4xx/5xx on every call is NOT "no data" —
        it's getting plenty of data, the data is just bad. Use attempt count,
        not success count, so the warning fires only when the provider is
        truly unused.

        Each entry: ``{"provider": str, "enabled_at_unix": float, "age_hours": float}``.
        """
        now = time.time()
        out: list[dict] = []
        for provider, enabled_at in self._provider_enabled_at.items():
            age_hours = (now - enabled_at) / 3600.0
            if age_hours < min_age_hours:
                continue
            if self._provider_attempt_count.get(provider, 0) > 0:
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
        # Track success count (different from attempt count — see __init__ doc).
        self.record_call_observation(provider)
        # Tokens (TPM tracking). RPM was already counted at record_attempt
        # time so we do NOT call record_request here — would double-count.
        self._rate_limiter.record_tokens(model_id, provider, token_count)
        # Feed S7 burn-rate signal with actual volume. Calls were already
        # counted at attempt time (calls=1); now add the realised tokens
        # (calls=0 to avoid double-counting).
        try:
            from nerd_herd.burn_log import get_burn_log
            get_burn_log().record(provider=provider, model=model_id,
                                  tokens=token_count, calls=0)
        except Exception:
            pass

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
        # Accept both "rate_limit" (legacy) and "rate_limited" (the string
        # hallederiz_kadir.classify_error actually emits). Without the
        # latter, real Groq 429 responses never triggered adaptive
        # reduction — _rate_limit_hits stayed 0 forever.
        if error_type in ("rate_limit", "rate_limited"):
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

    # ── Persistence ──────────────────────────────────────────────────────
    # snapshot_state / restore_state allow a host process to persist KDV
    # state across reboots. State is keyed by (scope, scope_key) so the
    # storage layer can write per-row without parsing the whole blob.
    #
    # Returned shape:
    #   {
    #     "models":         {model_id: {field: value, ...}, ...},
    #     "providers":      {provider: {field: value, ...}, ...},
    #     "breakers":       {provider: {field: value, ...}, ...},
    #     "enabled_at":     {provider: float},
    #     "call_count":     {provider: int},   # successes
    #     "attempt_count":  {provider: int},   # successes + failures
    #   }
    # Per-minute counters (_request_timestamps, _token_log) are intentionally
    # NOT persisted — see RateLimitState._PERSISTED_FIELDS.
    def snapshot_state(self) -> dict:
        return {
            "models": {
                mid: state.snapshot_state()
                for mid, state in self._rate_limiter.model_limits.items()
            },
            "providers": {
                prov: state.snapshot_state()
                for prov, state in self._rate_limiter._provider_limits.items()
            },
            "breakers": {
                prov: cb.snapshot_state()
                for prov, cb in self._circuit_breakers.items()
            },
            "enabled_at": dict(self._provider_enabled_at),
            "call_count": dict(self._provider_call_count),
            "attempt_count": dict(self._provider_attempt_count),
        }

    def restore_state(self, snap: dict) -> None:
        """Apply a previously-captured snapshot. Only known model_ids and
        providers are restored — anything that has since been deregistered
        is silently skipped. The caller (persistence layer) is responsible
        for staleness checks (e.g., dropping snapshots older than 24h).
        """
        for mid, model_snap in snap.get("models", {}).items():
            state = self._rate_limiter.model_limits.get(mid)
            if state is not None:
                state.restore_state(model_snap)
        for prov, prov_snap in snap.get("providers", {}).items():
            state = self._rate_limiter._provider_limits.get(prov)
            if state is not None:
                state.restore_state(prov_snap)
        for prov, cb_snap in snap.get("breakers", {}).items():
            cb = self._get_cb(prov)
            cb.restore_state(cb_snap)
        for prov, ts in snap.get("enabled_at", {}).items():
            self._provider_enabled_at.setdefault(prov, ts)
        for prov, count in snap.get("call_count", {}).items():
            self._provider_call_count[prov] = (
                self._provider_call_count.get(prov, 0) + int(count)
            )
        for prov, count in snap.get("attempt_count", {}).items():
            self._provider_attempt_count[prov] = (
                self._provider_attempt_count.get(prov, 0) + int(count)
            )
