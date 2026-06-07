"""KuledenDonenVar (KDV) factory — the live cloud rate-limit / capacity tracker.

Relocated 2026-06-07 out of src/core/router.py (which retained it only as a
fossil alongside the now-deleted select_model scorer). src/core/router.py
re-exports get_kdv from here for back-compat.
"""

from __future__ import annotations

from src.infra.logging_config import get_logger
from kuleden_donen_var import KuledenDonenVar, KuledenConfig, CapacityEvent

logger = get_logger("infra.rate_limiter")


_kdv: KuledenDonenVar | None = None


def get_kdv() -> KuledenDonenVar:
    global _kdv
    if _kdv is None:
        from src.models.rate_limiter import _INITIAL_PROVIDER_LIMITS
        from src.models.model_registry import get_registry as _get_registry
        from fatih_hoca.requirements import get_quota_planner

        def _on_capacity_change(evt: CapacityEvent) -> None:
            planner = get_quota_planner()
            snap = evt.snapshot
            if snap.utilization_pct > 0:
                reset_in = snap.reset_in_seconds or 3600
                planner.update_paid_utilization(evt.provider, snap.utilization_pct, reset_in)

            if evt.event_type in ("capacity_restored", "circuit_breaker_reset"):
                try:
                    from src.infra.db import schedule_accelerate_retries
                    schedule_accelerate_retries("capacity_restored")
                except Exception:
                    pass

        cfg = KuledenConfig(on_capacity_change=_on_capacity_change)
        _kdv = KuledenDonenVar(cfg)

        try:
            registry = _get_registry()
            for model in registry.cloud_models():
                agg = _INITIAL_PROVIDER_LIMITS.get(model.provider, {})
                _kdv.register(
                    model_id=model.litellm_name,
                    provider=model.provider,
                    rpm=model.rate_limit_rpm,
                    tpm=model.rate_limit_tpm,
                    provider_aggregate_rpm=agg.get("rpm"),
                    provider_aggregate_tpm=agg.get("tpm"),
                )
                # Propagate the daily-axis quota when known. KDV.register
                # only accepts rpm/tpm; rpd lives on RateLimitState as a
                # separate field. Static seeds (Gemini free tier per AI
                # Studio quota table) populate this on every registration
                # — without it, S1's time_bucketed depletion arm has no
                # rpd cell to compute frac on, and exhausted models stay
                # invisible to pool pressure.
                if model.rate_limit_rpd is not None:
                    state = _kdv._rate_limiter.model_limits.get(model.litellm_name)
                    if state is not None:
                        state.rpd_limit = int(model.rate_limit_rpd)
                        state.rpd_remaining = int(model.rate_limit_rpd)
            # Mark each cloud provider as enabled so KDV can surface
            # "no observations after Nh" warnings later.
            for provider in {m.provider for m in registry.cloud_models()}:
                _kdv.mark_provider_enabled(provider)
        except Exception:
            pass

        # Wire the in-flight tracker so begin_call / end_call push a
        # CloudProviderState (with overlaid in_flight) into nerd_herd.
        # Without this, the tracker counts handles in-process but the
        # signal never reaches pool_pressure computation.
        try:
            import nerd_herd
            from kuleden_donen_var import configure_in_flight_push
            from kuleden_donen_var.nerd_herd_adapter import make_state_getter
            configure_in_flight_push(nerd_herd, make_state_getter(_kdv))
        except Exception:
            pass

        # Restore persisted KDV state synchronously here so the first
        # pre_call after boot sees real adapted limits / 429 history /
        # daily counters / header reset clocks. Uses plain sqlite3 (not
        # aiosqlite) so it works whether or not an event loop is active.
        # Best-effort: failures degrade to cold-start state. Skipped
        # silently when DB_PATH is unset (CLI tools, tests).
        try:
            import os
            db_path = os.environ.get("DB_PATH")
            if db_path:
                from src.infra import kdv_persistence
                kdv_persistence.load_sync(_kdv, db_path)
        except Exception:
            pass
    return _kdv
