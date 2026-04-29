"""S2 — per-call token burden.

For each token-axis cell with remaining > 0:
    bite_frac = est_per_call_tokens / remaining
    pressure = -clip(bite_frac - 0.30, 0, 0.70) / 0.70
Fold: largest bite (most-stressed window) wins.
"""
from __future__ import annotations

from nerd_herd.types import RateLimitMatrix


BITE_THRESHOLD = 0.30
BITE_RANGE = 0.70  # 0.30 → 0, 1.00 → -1


def s2_call_burden(matrix: RateLimitMatrix, *, est_per_call_tokens: int) -> float:
    if est_per_call_tokens <= 0:
        return 0.0
    worst = 0.0
    for _, rl in matrix.token_cells():
        remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        bite = est_per_call_tokens / remaining
        excess = max(0.0, bite - BITE_THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess / BITE_RANGE)
        if pressure < worst:
            worst = pressure
    return worst
