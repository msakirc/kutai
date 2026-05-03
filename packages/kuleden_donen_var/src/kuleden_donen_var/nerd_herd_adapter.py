"""Adapter: KDV internal provider state -> nerd_herd.CloudProviderState.

Forwards all populated rate-limit cells (request, token, cost axes).
KDV's RateLimitState exposes per-axis attributes; missing axes return RateLimit()
with no limit set — those cells stay invisible to signal consumers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from nerd_herd.signals.s10_failure import MIN_SAMPLES as S10_MIN_SAMPLES

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


def _prior_key(provider: str, model_id: str) -> str | None:
    """Identifier for provider_prior aggregation.

    Openrouter is structurally an aggregator, not a provider — per-id
    failure modes vary wildly across upstream vendors (anthropic vs
    tencent vs meta-llama backends behave very differently when reached
    via openrouter). Aggregating across all openrouter ids would smear
    the signal and let one healthy vendor's prior cover for another
    vendor's broken endpoint. Split the prior by sub-vendor, parsed from
    the litellm_name path: ``openrouter/<vendor>/<model>:free`` →
    ``"openrouter::<vendor>"``.

    Other providers aggregate at provider level. Returns None when the
    model has no usable grouping key (shouldn't happen for registered
    ids, but be defensive).
    """
    if provider == "openrouter":
        parts = model_id.split("/")
        if len(parts) >= 3:
            return f"openrouter::{parts[1]}"
        return None
    return provider


def _build_prior_groups(
    provider: str,
    model_ids: set[str],
) -> dict[str, set[str]]:
    """Group ``model_ids`` by their _prior_key for aggregation.

    For non-openrouter providers, this is a single group keyed on the
    provider name. For openrouter, one group per sub-vendor.
    """
    groups: dict[str, set[str]] = {}
    for mid in model_ids:
        key = _prior_key(provider, mid)
        if key is None:
            continue
        groups.setdefault(key, set()).add(mid)
    return groups


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

    # Pre-compute provider-prior groups + their aggregate rates once
    # per build (each model in a group reads the same value). For
    # non-openrouter providers there's a single group; for openrouter,
    # one per sub-vendor — failure modes diverge enough that smearing
    # across all openrouter ids would hide vendor-specific outages.
    prior_groups = _build_prior_groups(provider, set(model_ids))
    prior_by_key: dict[str, float | None] = {}
    for key, members in prior_groups.items():
        try:
            prior_by_key[key] = kdv.provider_prior_rate(
                provider, member_ids=members,
            )
        except Exception:
            prior_by_key[key] = None

    models = {}
    for mid in model_ids:
        mstate = kdv._rate_limiter.model_limits.get(mid)
        # Per-model matrix BLENDED with provider aggregate so pool
        # pressure sees the binding constraint.
        if prov_state is not None:
            matrix = _worst_of(mstate, prov_state)
        else:
            matrix = _matrix(mstate)
        # Reliability signal: rolling success rate + sample count
        # plumbed for the S10_failure pressure signal. samples_n gates
        # the signal — below MIN_SAMPLES (imported from s10_failure)
        # S10 returns 0 (no data, no opinion), preventing freshly-
        # revived models from ranking as "perfectly reliable" on an
        # empty window.
        #
        # The adapter ALSO gates success_rate at the same threshold so
        # downstream consumers don't see KDV's no-data sentinel
        # (`recent_success_rate(mid)` returns 1.0 when samples < 5,
        # which would lie if anyone ever read it without consulting
        # samples_n). When below MIN_SAMPLES, leave success_rate at
        # its CloudModelState default. Both sides import the same
        # MIN_SAMPLES so the threshold can't drift between layers.
        try:
            samples_n = int(kdv.recent_samples_n(mid))
        except Exception:
            samples_n = 0
        if samples_n >= S10_MIN_SAMPLES:
            try:
                success_rate = float(kdv.recent_success_rate(mid))
            except Exception:
                success_rate = 1.0
        else:
            success_rate = 1.0  # ignored by S10; matches CloudModelState default
        # Daily-exhausted: surface KDV's per-model rpd-exhausted state
        # so selector eligibility can reject before ranking. Pre-this,
        # selector saw stale rpd_remaining (providers like gemini don't
        # return rpd headers) and admitted tasks that KDV.pre_call
        # would immediately refuse with daily_exhausted reason.
        try:
            daily_out = bool(kdv._rate_limiter.is_daily_exhausted(mid))
        except Exception:
            daily_out = False
        # Surface KDV's rpm cooldown (Retry-After / x-ratelimit-reset
        # floor with remaining=0) so selector eligibility can reject
        # before ranking. Reads raw _header_* fields under the hood —
        # bypasses the 5s freshness window that gates the public
        # rpm_remaining property, since retry-after horizons routinely
        # exceed it.
        try:
            rpm_cool = bool(kdv._rate_limiter.is_rpm_cooldown(mid))
        except Exception:
            rpm_cool = False
        # Provider-level success-rate prior, looked up via the same
        # grouping key the adapter used to compute the aggregate. S10
        # uses this only when the model's own samples are below
        # MIN_SAMPLES — fills the cold-start gap for new / revived ids.
        prior_key_for_mid = _prior_key(provider, mid)
        provider_prior = (
            prior_by_key.get(prior_key_for_mid) if prior_key_for_mid else None
        )

        models[mid] = CloudModelState(
            model_id=mid, limits=matrix,
            recent_success_rate=success_rate,
            recent_samples_n=samples_n,
            provider_prior_rate=provider_prior,
            daily_exhausted=daily_out,
            rpm_cooldown=rpm_cool,
        )

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
