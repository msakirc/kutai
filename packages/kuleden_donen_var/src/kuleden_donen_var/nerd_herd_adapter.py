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

    def _worst_of(model_state, prov_state):
        """Per-cell, return the more-constraining axis between model and
        provider-aggregate state. KDV's has_capacity gates on BOTH —
        pool pressure must see the binding one to compute correct
        depletion. Without this, per-model rpm=15 with 5 calls looks
        abundant while provider-aggregate rpm=15 with 14 calls is
        nearly exhausted; selector picked freely and KDV refused at
        pre_call (production triage 2026-05-01).
        """
        from nerd_herd.types import RateLimit
        m = RateLimitMatrix()
        for axis in _ADAPTER_AXES:
            mr = _rl(model_state, axis)
            pr = _rl(prov_state, axis)
            # If only one side has a limit, that wins. If both, take the
            # one with smaller `remaining` (or smaller `limit` when
            # remaining ties). Provider-aggregate rpm/tpm typically
            # binds; per-model rpd/tpd typically binds.
            if mr.limit is None and pr.limit is None:
                setattr(m, axis, RateLimit())
                continue
            if mr.limit is None:
                setattr(m, axis, pr); continue
            if pr.limit is None:
                setattr(m, axis, mr); continue
            # Both populated. Pick worse.
            m_rem = mr.remaining if mr.remaining is not None else mr.limit
            p_rem = pr.remaining if pr.remaining is not None else pr.limit
            chosen = pr if p_rem < m_rem else mr
            setattr(m, axis, chosen)
        return m

    prov_state = kdv._rate_limiter._provider_limits.get(provider)
    models = {}
    for mid in model_ids:
        mstate = kdv._rate_limiter.model_limits.get(mid)
        # Per-model matrix BLENDED with provider aggregate so pool
        # pressure sees the binding constraint.
        if prov_state is not None:
            matrix = _worst_of(mstate, prov_state)
        else:
            matrix = _matrix(mstate)
        models[mid] = CloudModelState(model_id=mid, limits=matrix)

    cb = kdv._circuit_breakers.get(provider)
    return CloudProviderState(
        provider=provider,
        models=models,
        limits=_matrix(prov_state),
        circuit_breaker_open=bool(cb.is_degraded) if cb is not None else False,
    )


def make_state_getter(kdv: "KuledenDonenVar") -> Callable[[str], "CloudProviderState | None"]:
    """Factory returning a closure suitable for configure_in_flight_push()."""
    return lambda provider: build_cloud_provider_state(kdv, provider)
