"""S10 — per-model reliability as a pressure signal.

Pre-2026-05-03: S10 was a per-provider consecutive_failures step
(0 → -0.2 → -0.5). The provider-level consecutive_failures field had
no live writer in cloud paths (only api_reliability table for free APIs
wrote to it), so S10 was effectively dead weight. Reliability was
bolted on as a post-composite multiplier in ranking.py — bypassing the
signal infrastructure (M1/M2/M3 weighting, bucket worst-wins,
abundance gate) and using a 0.1 floor that let fully-broken models
stay competitive.

Now: S10 takes per-model rolling success_rate and samples_n from KDV,
returns a continuous negative scalar that flows through OTHER_BUCKET
worst-wins and M3 difficulty weights. The reliability multiplier in
ranking.py is deleted — single source of truth.

Step 6 (2026-05-04): provider_prior_rate is consumed as a fallback
when the model's own samples are below MIN_SAMPLES. Filled by the
adapter from KDV.provider_prior_rate, with openrouter aggregated by
sub-vendor (per-vendor failure modes diverge too much to share a
single openrouter aggregate).

Signal shape (priority order):
    samples_n  >= MIN_SAMPLES  → use own success_rate (curve below)
    provider_prior_rate set    → use prior (same curve)
    else                       → 0.0 (no data anywhere, no opinion)

Curve:
    rate >= 0.95 → 0.0 (healthy)
    rate <= 0.20 → -1.0 (broken)
    else         → linear interp

User design 2026-05-03: "S12 = nothing different from S10. Wire S10
properly with rate, delete the multiplier." Provider prior added
2026-05-04 to close the cold-start gap.
"""
from __future__ import annotations


MIN_SAMPLES = 5
HEALTHY_THRESHOLD = 0.95
BROKEN_THRESHOLD = 0.20


def _curve(rate: float) -> float:
    if rate >= HEALTHY_THRESHOLD:
        return 0.0
    if rate <= BROKEN_THRESHOLD:
        return -1.0
    span = HEALTHY_THRESHOLD - BROKEN_THRESHOLD
    return -((HEALTHY_THRESHOLD - rate) / span)


def s10_failure(
    *,
    success_rate: float = 1.0,
    samples_n: int = 0,
    provider_prior_rate: float | None = None,
    consecutive_failures: int = 0,
) -> float:
    """Per-model reliability scalar in [-1, 0].

    Args:
        success_rate: rolling rate from KDV outcome window. Ignored
            when samples_n < MIN_SAMPLES.
        samples_n: count of outcomes in the rolling window. Below
            MIN_SAMPLES, falls back to provider_prior_rate if set.
        provider_prior_rate: aggregate success rate across the model's
            siblings on the same provider (or openrouter sub-vendor),
            from KDV.provider_prior_rate. Used ONLY when own samples
            are insufficient — when own data is good, prior is ignored
            (don't blend, don't double-count). None means the prior
            also lacks enough data → fall to neutral.
        consecutive_failures: legacy per-provider streak counter,
            kept as a fallback for callers that haven't migrated to
            success_rate plumbing. Steps -0.2 / -0.5 like the old
            implementation. When both are supplied, takes the worst
            (most negative).

    Returns:
        Scalar in [-1, 0]. Zero means neutral / no data.
    """
    rate_signal = 0.0
    if samples_n >= MIN_SAMPLES:
        rate_signal = _curve(success_rate)
    elif provider_prior_rate is not None:
        rate_signal = _curve(provider_prior_rate)

    streak_signal = 0.0
    if consecutive_failures > 0:
        streak_signal = -0.2 if consecutive_failures <= 2 else -0.5

    return min(rate_signal, streak_signal)
