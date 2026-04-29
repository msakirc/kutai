"""S4 — queue token pressure.

Sum est_per_task_tokens over unblocked + pending + dep-resolved tasks (queue
projection done at queue_profile_push). Compare to remaining token budget
across windows. Fold: most-stressed window wins.

Threshold: 70% projected → 0; 100% → -0.5; 120%+ → -1.0.
"""
from __future__ import annotations

from nerd_herd.types import QueueProfile, RateLimitMatrix


THRESHOLD = 0.70
SLOPE = 2.0  # demand 70% → 0; 95% → -0.5; 120% → -1.0


def s4_queue_tokens(matrix: RateLimitMatrix, *, queue: QueueProfile) -> float:
    projected = queue.projected_tokens
    if projected <= 0:
        return 0.0
    worst = 0.0
    for _, rl in matrix.token_cells():
        remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        ratio = projected / remaining
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
