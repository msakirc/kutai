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
    # d=2 needs 30; cap 95 → fit_excess = 0.65 → > 0.4 → waste
    # d=2 cap 40 → fit_excess = 0.10 → not waste
    picks = _picks((2, "local", 95), (2, "local", 40), (2, "local", 35))
    run = SimRun(picks=picks, final_state=SimState())
    m = compute_metrics(run)
    assert m.easy_task_waste == 1.0 / 3.0


def test_free_quota_utilization():
    final = SimState()
    final.time_bucketed["groq"] = SimPoolCounter(remaining=200, limit=1000, reset_at=0.0)
    run = SimRun(picks=[], final_state=final)
    m = compute_metrics(run)
    # 800 used / 1000 limit = 80%
    assert m.free_quota_utilization == 0.8
