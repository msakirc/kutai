"""Adapter: KDV internal provider state -> nerd_herd.CloudProviderState.

KDV stores rate-limit records as `ModelLimits` inside `rate_limiter`; nerd_herd
consumes `CloudProviderState` (with `RateLimits.rpd`). This adapter walks KDV's
registered providers and produces a fresh `CloudProviderState` on demand.

Wired at app startup via `configure_in_flight_push(nerd_herd_module, getter)`.
The in-flight tracker calls `getter(provider)` from `_push()` so nerd_herd
receives a snapshot-ready state with in_flight counts overlaid.

Only `rpd` is forwarded today — that's the field pool_pressure needs. rpm/tpm
can be added if a consumer actually reads them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from nerd_herd.types import CloudProviderState

    from .kdv import KuledenDonenVar


def build_cloud_provider_state(
    kdv: "KuledenDonenVar",
    provider: str,
) -> "CloudProviderState | None":
    from nerd_herd.types import (
        CloudModelState,
        CloudProviderState,
        RateLimit,
        RateLimits,
    )

    model_ids = kdv._providers.get(provider)
    if not model_ids:
        return None

    def _rl(state) -> RateLimit:
        if state is None:
            return RateLimit()
        return RateLimit(
            limit=state.rpd_limit,
            remaining=state.rpd_remaining,
            reset_at=int(state.rpd_reset_at) if state.rpd_reset_at else None,
        )

    models = {}
    for mid in model_ids:
        mstate = kdv._rate_limiter.model_limits.get(mid)
        limits = RateLimits(rpd=_rl(mstate))
        models[mid] = CloudModelState(model_id=mid, limits=limits)

    prov_state = kdv._rate_limiter._provider_limits.get(provider)
    prov_limits = RateLimits(rpd=_rl(prov_state))

    return CloudProviderState(
        provider=provider,
        models=models,
        limits=prov_limits,
    )


def make_state_getter(kdv: "KuledenDonenVar") -> Callable[[str], "CloudProviderState | None"]:
    """Factory returning a closure suitable for configure_in_flight_push()."""
    return lambda provider: build_cloud_provider_state(kdv, provider)
