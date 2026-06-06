"""Mid-task urgency helper — a started task finishes on (at least) its
admission urgency plus a finish-bias, so it is never judged stricter than
a fresh task. Design: docs/superpowers/specs/2026-06-04-mid-task-urgency-asymmetry-design.md
"""
from fatih_hoca.urgency import mid_task_urgency, FINISH_BIAS, FAILURE_BUMP


def test_baseline_no_failures_adds_finish_bias():
    # admission urgency 0.5 (priority-5 baseline) → 0.5 + 0.1 finish-bias
    assert mid_task_urgency(0.5, has_failures=False) == 0.6


def test_baseline_with_failures_stacks_failure_bump():
    # 0.5 + 0.1 finish + 0.1 failure
    assert abs(mid_task_urgency(0.5, has_failures=True) - 0.7) < 1e-9


def test_high_admission_urgency_is_honored():
    # a priority>5 / aged task admitted at 0.8 keeps that band mid-task
    assert abs(mid_task_urgency(0.8, has_failures=False) - 0.9) < 1e-9


def test_clamped_at_one():
    assert mid_task_urgency(0.95, has_failures=True) == 1.0
    assert mid_task_urgency(1.0, has_failures=False) == 1.0


def test_none_base_falls_back_to_half():
    assert mid_task_urgency(None, has_failures=False) == 0.6


def test_constants_are_tenths():
    assert FINISH_BIAS == 0.1
    assert FAILURE_BUMP == 0.1
