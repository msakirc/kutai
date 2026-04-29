"""S3 — cumulative per-task token burden across iterations.

Same shape as S2 but uses `est_per_task_tokens = est_per_call_tokens × est_iterations`.
Captures the full task's hit on each token-axis budget.
"""
from __future__ import annotations

from nerd_herd.signals.s2_call_burden import BITE_RANGE, BITE_THRESHOLD
from nerd_herd.types import RateLimitMatrix


def s3_task_burden(matrix: RateLimitMatrix, *, est_per_task_tokens: int) -> float:
    if est_per_task_tokens <= 0:
        return 0.0
    worst = 0.0
    for _, rl in matrix.token_cells():
        remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        bite = est_per_task_tokens / remaining
        excess = max(0.0, bite - BITE_THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess / BITE_RANGE)
        if pressure < worst:
            worst = pressure
    return worst
