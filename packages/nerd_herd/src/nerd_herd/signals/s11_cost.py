"""S11 — cost burden.

Compare est_call_cost against daily/monthly cost remaining. Same shape as
S2 but on cost axis. Zero when no cost cap configured.
"""
from __future__ import annotations


THRESHOLD = 0.30
SLOPE = 1.0 / 0.70


def s11_cost(*, est_call_cost: float, daily_cost_remaining: float) -> float:
    if est_call_cost <= 0 or daily_cost_remaining <= 0:
        return 0.0
    bite = est_call_cost / daily_cost_remaining
    excess = max(0.0, bite - THRESHOLD)
    if excess <= 0:
        return 0.0
    return -min(1.0, excess * SLOPE)
