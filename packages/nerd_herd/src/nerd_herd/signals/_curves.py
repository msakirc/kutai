"""Shared continuous shaping curves for signals (no gates, no kinks).

Single source of truth so S7 / S6 / S12 cannot drift apart.
"""
from __future__ import annotations


def smoothstep(x: float) -> float:
    """Hermite 3x^2 - 2x^3 clamped to [0, 1]. Zero slope at both ends."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)
