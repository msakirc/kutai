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

Signal shape:
    samples_n  < MIN_SAMPLES → 0.0 (no data, neutral — provider-prior
                                   carries the signal until data lands)
    success_rate >= 0.95     → 0.0 (healthy)
    success_rate <= 0.20     → -1.0 (broken)
    else                     → linear interp from -0.0 at 0.95 to -1.0
                               at 0.20

User design 2026-05-03: "S12 = nothing different from S10. Wire S10
properly with rate, delete the multiplier."
"""
from __future__ import annotations


MIN_SAMPLES = 5
HEALTHY_THRESHOLD = 0.95
BROKEN_THRESHOLD = 0.20


def s10_failure(
    *,
    success_rate: float = 1.0,
    samples_n: int = 0,
    consecutive_failures: int = 0,
) -> float:
    """Per-model reliability scalar in [-1, 0].

    Args:
        success_rate: rolling rate from KDV outcome window. Ignored
            when samples_n < MIN_SAMPLES.
        samples_n: count of outcomes in the rolling window. Below
            MIN_SAMPLES, signal returns 0 (no opinion).
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
        if success_rate >= HEALTHY_THRESHOLD:
            rate_signal = 0.0
        elif success_rate <= BROKEN_THRESHOLD:
            rate_signal = -1.0
        else:
            # Linear interp from (HEALTHY, 0) to (BROKEN, -1)
            span = HEALTHY_THRESHOLD - BROKEN_THRESHOLD
            rate_signal = -((HEALTHY_THRESHOLD - success_rate) / span)

    streak_signal = 0.0
    if consecutive_failures > 0:
        streak_signal = -0.2 if consecutive_failures <= 2 else -0.5

    return min(rate_signal, streak_signal)
