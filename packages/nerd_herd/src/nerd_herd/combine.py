"""Combination logic: signals → scalar.

Worst-wins per bucket; weighted sum across buckets; gated abundance.
"""
from __future__ import annotations

from nerd_herd.breakdown import PressureBreakdown


W_BURDEN = 0.5
W_QUEUE = 0.7
W_OTHER = 1.0
ABUNDANCE_GATE = -0.2

BURDEN_BUCKET = ("S2", "S3")
QUEUE_BUCKET = ("S4", "S5", "S6")
OTHER_BUCKET = ("S1", "S7", "S9", "S10", "S11")
POSITIVE_ARM_SIGNALS = ("S1", "S9")


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def combine_signals(*, signals: dict[str, float], weights: dict[str, float]) -> PressureBreakdown:
    """Combine ten weighted signals into a single scalar plus diagnostic struct."""
    weighted = {k: signals.get(k, 0.0) * weights.get(k, 1.0) for k in OTHER_BUCKET + BURDEN_BUCKET + QUEUE_BUCKET}

    burden_neg = min((weighted[k] for k in BURDEN_BUCKET if weighted[k] < 0), default=0.0)
    queue_neg = min((weighted[k] for k in QUEUE_BUCKET if weighted[k] < 0), default=0.0)
    other_neg = min((weighted[k] for k in OTHER_BUCKET if weighted[k] < 0), default=0.0)

    bucket_totals = {
        "burden": W_BURDEN * burden_neg,
        "queue": W_QUEUE * queue_neg,
        "other": W_OTHER * other_neg,
    }

    negative_total = sum(bucket_totals.values())

    positive_total = 0.0
    if negative_total > ABUNDANCE_GATE:
        positives = [weighted[k] for k in POSITIVE_ARM_SIGNALS if weighted[k] > 0]
        positive_total = max(positives, default=0.0)

    scalar = _clamp(negative_total + positive_total)

    return PressureBreakdown(
        scalar=scalar,
        signals=dict(signals),
        modifiers={"weights": dict(weights)},
        bucket_totals=bucket_totals,
        positive_total=positive_total,
        negative_total=negative_total,
    )
