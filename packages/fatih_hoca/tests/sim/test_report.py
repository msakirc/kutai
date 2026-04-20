"""Tests for simulator report (Phase 2d)."""
from sim.runner import SimPick, SimRun
from sim.state import SimState, SimPoolCounter
from sim.report import compute_metrics


def _picks(*specs):
    out = []
    for i, (d, pool, cap) in enumerate(specs):
        out.append(SimPick(
            task_idx=i, task_difficulty=d, model_name=f"m{i}",
            pool=pool, cap_score_100=cap, elapsed_seconds=10.0,
        ))
    return out


def test_hard_task_satisfaction_100pct():
    picks = _picks((8, "local", 80), (9, "local", 90))
    run = SimRun(picks=picks, final_state=SimState())
    m = compute_metrics(run)
    assert m.hard_task_satisfaction == 1.0


def test_hard_task_satisfaction_50pct():
    # d=8 needs 75; one meets, one doesn't
    picks = _picks((8, "local", 80), (8, "local", 50))
    run = SimRun(picks=picks, final_state=SimState())
    m = compute_metrics(run)
    assert m.hard_task_satisfaction == 0.5


def test_easy_task_waste_rate():
    # Waste semantics (updated): only per_call (paid) picks count as waste
    # when over-qualified on easy tasks. Free-pool picks aren't waste —
    # they're "burn quota we won't be refunded on." Local picks aren't
    # waste either — sunk cost, fastest tool available.
    # d=2 needs 30; cap 95 → fit_excess = 0.65 > 0.4 → waste (if per_call)
    picks = [
        SimPick(task_idx=0, task_difficulty=2, model_name="claude",
                pool="per_call", cap_score_100=95, elapsed_seconds=1.0),
        SimPick(task_idx=1, task_difficulty=2, model_name="claude",
                pool="per_call", cap_score_100=40, elapsed_seconds=1.0),
        SimPick(task_idx=2, task_difficulty=2, model_name="local",
                pool="local", cap_score_100=35, elapsed_seconds=1.0),
    ]
    run = SimRun(picks=picks, final_state=SimState())
    m = compute_metrics(run)
    # 1 wasted (per_call cap=95) out of 3 easy picks
    assert m.easy_task_waste == 1.0 / 3.0


def test_easy_task_waste_ignores_free_pool_over_qualification():
    # Free pool burning is not waste, even if grossly over-qualified.
    picks = [
        SimPick(task_idx=0, task_difficulty=2, model_name="groq",
                pool="time_bucketed", cap_score_100=72, elapsed_seconds=1.0),
        SimPick(task_idx=1, task_difficulty=2, model_name="local",
                pool="local", cap_score_100=90, elapsed_seconds=1.0),
    ]
    run = SimRun(picks=picks, final_state=SimState())
    m = compute_metrics(run)
    assert m.easy_task_waste == 0.0


def test_free_quota_utilization():
    # Metric counts picks per time_bucketed pool (reset-proof), not final
    # remaining (under-counts when resets fire mid-run).
    final = SimState()
    final.time_bucketed["groq"] = SimPoolCounter(remaining=200, limit=1000, reset_at=0.0)
    picks = [
        SimPick(task_idx=i, task_difficulty=3, model_name="groq",
                pool="time_bucketed", cap_score_100=72, elapsed_seconds=1.0)
        for i in range(800)
    ]
    run = SimRun(picks=picks, final_state=final)
    m = compute_metrics(run)
    # 800 picks / 1000 limit = 80%
    assert m.free_quota_utilization == 0.8


def test_free_quota_utilization_caps_at_100pct_per_pool():
    # If we pick beyond a pool's limit (reset cycles), ratio caps at 1.0 per
    # pool so a single hot pool doesn't skew the average.
    final = SimState()
    final.time_bucketed["groq"] = SimPoolCounter(remaining=0, limit=10, reset_at=0.0)
    final.time_bucketed["gemini"] = SimPoolCounter(remaining=5, limit=10, reset_at=0.0)
    picks = (
        [SimPick(task_idx=i, task_difficulty=3, model_name="groq",
                 pool="time_bucketed", cap_score_100=72, elapsed_seconds=1.0)
         for i in range(50)]  # way over limit
        + [SimPick(task_idx=i, task_difficulty=3, model_name="gemini",
                   pool="time_bucketed", cap_score_100=68, elapsed_seconds=1.0)
           for i in range(2)]
    )
    run = SimRun(picks=picks, final_state=final)
    m = compute_metrics(run)
    # groq: min(50/10, 1.0) = 1.0, gemini: 2/10 = 0.2, avg = 0.6
    assert m.free_quota_utilization == 0.6
