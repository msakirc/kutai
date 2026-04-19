"""Metrics for simulator runs (Phase 2d spec §7)."""
from __future__ import annotations

from dataclasses import dataclass

from fatih_hoca.capability_curve import cap_needed_for_difficulty
from sim.runner import SimRun


@dataclass
class SimMetrics:
    hard_task_satisfaction: float = 0.0     # fraction of d>=7 picks meeting cap_needed
    easy_task_waste: float = 0.0            # fraction of d<=4 picks with fit_excess>0.4
    free_quota_utilization: float = 0.0     # avg fraction of time_bucketed capacity consumed
    max_local_idle: float = 0.0             # max idle_seconds across run (future)
    exhaustion_crashes: int = 0             # runner exceptions (0 if clean)


def compute_metrics(run: SimRun) -> SimMetrics:
    m = SimMetrics()

    hard = [p for p in run.picks if p.task_difficulty >= 7]
    if hard:
        passed = sum(
            1 for p in hard
            if p.cap_score_100 >= cap_needed_for_difficulty(p.task_difficulty)
        )
        m.hard_task_satisfaction = passed / len(hard)

    easy = [p for p in run.picks if p.task_difficulty <= 4]
    if easy:
        wasted = 0
        for p in easy:
            fit_excess = (p.cap_score_100 - cap_needed_for_difficulty(p.task_difficulty)) / 100.0
            if fit_excess > 0.4:
                wasted += 1
        m.easy_task_waste = wasted / len(easy)

    tb = run.final_state.time_bucketed
    if tb:
        ratios = []
        for counter in tb.values():
            if counter.limit > 0:
                used = counter.limit - counter.remaining
                ratios.append(max(0.0, used / counter.limit))
        if ratios:
            m.free_quota_utilization = sum(ratios) / len(ratios)

    locals_idle = [l.idle_seconds for l in run.final_state.locals.values()]
    m.max_local_idle = max(locals_idle) if locals_idle else 0.0

    return m
