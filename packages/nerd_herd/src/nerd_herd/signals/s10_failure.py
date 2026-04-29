"""S10 — failure state. Step from 0 → -0.2 → -0.5 with consecutive failures."""
from __future__ import annotations


def s10_failure(*, consecutive_failures: int) -> float:
    if consecutive_failures <= 0:
        return 0.0
    if consecutive_failures <= 2:
        return -0.2
    return -0.5
