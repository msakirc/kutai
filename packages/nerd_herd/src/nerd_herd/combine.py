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
OTHER_BUCKET = ("S1", "S7", "S9", "S10", "S11", "S12", "S13", "S14")
# Positive (abundance) arm. S1 dropped 2026-06-04: its free-cloud abundance
# is now 0 (scale-invariant frac removed — see s1_remaining), so it only ever
# contributed 0 here. Positive pull = S9 (timing) noisy-OR S12 (fleet under-use).
POSITIVE_ARM_SIGNALS = ("S9", "S12")


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
        # Noisy-OR composition over POSITIVE_ARM signals (S1, S9). Each
        # signal is clamped to [0, 1] (M3 weights can push >1) then
        # combined as P(any fires) = 1 - prod(1 - p_i). Properties:
        #   - Single strong signal preserved exactly: (0.9, 0) → 0.9
        #   - Two moderates compose without inflation: (0.4, 0.5) → 0.7
        #   - Both strong → near cap: (0.8, 0.9) → 0.98
        #   - Zero signals contribute nothing — no spurious activation
        # Replaces the prior max() which discarded reinforcement when
        # both stock (S1) and timing (S9) signals fired together.
        # User design 2026-05-03: noisy-OR for positive sum.
        clamped = [
            min(1.0, max(0.0, weighted[k]))
            for k in POSITIVE_ARM_SIGNALS
        ]
        complement = 1.0
        for p in clamped:
            complement *= (1.0 - p)
        positive_total = 1.0 - complement

    scalar = _clamp(negative_total + positive_total)

    return PressureBreakdown(
        scalar=scalar,
        signals=dict(signals),
        modifiers={"weights": dict(weights)},
        bucket_totals=bucket_totals,
        positive_total=positive_total,
        negative_total=negative_total,
    )
