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

    # Easy-task waste: over-qualified picks that actually cost something.
    # A free-pool pick on an easy task isn't waste — it's burning quota we
    # won't get refunded. Per-call (paid) is the only pool where
    # over-qualification is a real opportunity cost. Locals being
    # over-qualified on easy tasks also isn't "waste" — loaded-local is
    # the fastest tool available, cost is already sunk.
    easy = [p for p in run.picks if p.task_difficulty <= 4]
    if easy:
        wasted = 0
        for p in easy:
            if p.pool != "per_call":
                continue  # free or local — no waste semantics
            fit_excess = (p.cap_score_100 - cap_needed_for_difficulty(p.task_difficulty)) / 100.0
            if fit_excess > 0.4:
                wasted += 1
        m.easy_task_waste = wasted / len(easy)

    # Free-quota utilization: count picks per time_bucketed pool across the
    # full run and divide by the pool's limit. Using final-state `remaining`
    # under-counts when resets fire mid-sim (the counter gets refilled, erasing
    # the pre-reset picks from the accounting). Pick counts are reset-proof.
    tb = run.final_state.time_bucketed
    if tb:
        picks_by_pool: dict[str, int] = {}
        for p in run.picks:
            if p.pool == "time_bucketed":
                picks_by_pool[p.model_name] = picks_by_pool.get(p.model_name, 0) + 1
        ratios = []
        for pool_name, counter in tb.items():
            if counter.limit > 0:
                picks = picks_by_pool.get(pool_name, 0)
                # Cap at 1.0 per pool — utilization >100% means we hit reset
                # cycles, which is a good thing, but the metric is "did we
                # fill each pool's budget." Clamp so one heavily-used pool
                # doesn't mask a neglected one in the average.
                ratios.append(min(1.0, picks / counter.limit))
        if ratios:
            m.free_quota_utilization = sum(ratios) / len(ratios)

    locals_idle = [l.idle_seconds for l in run.final_state.locals.values()]
    m.max_local_idle = max(locals_idle) if locals_idle else 0.0

    return m
