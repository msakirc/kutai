"""Capability-needed-per-difficulty curve (Phase 2d).

Plain dict mapping task difficulty (1-10) to the minimum `cap_score_100`
a model should have to serve that difficulty without over-qualification.

Used by ranking._apply_utilization_layer via:
    fit_excess = (cap_score_100 - cap_needed_for_difficulty(d)) / 100

Hand-tuned starting curve; graduation to empirical derivation from
`model_stats` is deferred until sample counts per (model, d) warrant it.
"""
from __future__ import annotations


CAP_NEEDED_BY_DIFFICULTY: dict[int, float] = {
    1: 30.0, 2: 30.0, 3: 30.0,
    4: 45.0, 5: 45.0,
    6: 60.0, 7: 60.0,
    8: 75.0,
    9: 88.0, 10: 88.0,
}


def cap_needed_for_difficulty(d: int) -> float:
    """Return the cap_score_100 floor for difficulty `d`.

    Clamps `d` to [1, 10] so out-of-range inputs degrade gracefully.
    """
    d_clamped = max(1, min(10, int(d)))
    return CAP_NEEDED_BY_DIFFICULTY[d_clamped]
