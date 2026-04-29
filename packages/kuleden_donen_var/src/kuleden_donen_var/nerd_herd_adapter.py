"""Adapter: KDV internal provider state -> nerd_herd.CloudProviderState.

Forwards all populated rate-limit cells (request, token, cost axes).
KDV's RateLimitState exposes per-axis attributes; missing axes return RateLimit()
with no limit set — those cells stay invisible to signal consumers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from nerd_herd.types import CloudProviderState

    from .kdv import KuledenDonenVar


# Axes the adapter forwards. Matches RateLimitMatrix field names.
_ADAPTER_AXES = (
    "rpm", "rph", "rpd", "rpw", "rpmonth",
    "tpm", "tph", "tpd", "tpw", "tpmonth",
    "itpm", "itpd", "otpm", "otpd",
    "cpd", "cpmonth",
)


def _rl(state, axis: str):
    """Build a RateLimit for axis from KDV state. Empty if state lacks the axis."""
    from nerd_herd.types import RateLimit
    if state is None:
        return RateLimit()
    return RateLimit(
        limit=getattr(state, f"{axis}_limit", None),
        remaining=getattr(state, f"{axis}_remaining", None),
        reset_at=int(getattr(state, f"{axis}_reset_at", 0) or 0) or None,
    )


def build_cloud_provider_state(
    kdv: "KuledenDonenVar",
    provider: str,
) -> "CloudProviderState | None":
    from nerd_herd.types import (
        CloudModelState,
        CloudProviderState,
        RateLimitMatrix,
    )
    model_ids = kdv._providers.get(provider)
    if not model_ids:
        return None

    def _matrix(state):
        m = RateLimitMatrix()
        for axis in _ADAPTER_AXES:
            setattr(m, axis, _rl(state, axis))
        return m

    models = {}
    for mid in model_ids:
        mstate = kdv._rate_limiter.model_limits.get(mid)
        models[mid] = CloudModelState(model_id=mid, limits=_matrix(mstate))

    prov_state = kdv._rate_limiter._provider_limits.get(provider)
    return CloudProviderState(
        provider=provider,
        models=models,
        limits=_matrix(prov_state),
    )


def make_state_getter(kdv: "KuledenDonenVar") -> Callable[[str], "CloudProviderState | None"]:
    """Factory returning a closure suitable for configure_in_flight_push()."""
    return lambda provider: build_cloud_provider_state(kdv, provider)
