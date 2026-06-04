"""husam mid-task re-selection routes urgency through the shared
fatih_hoca.mid_task_urgency helper, so the finish-bias applies on the
raw_dispatch path too. Boundary check on the helper contract husam relies on.
"""
from fatih_hoca.urgency import mid_task_urgency


def test_husam_base_gets_finish_bias():
    # urgency_in 0.5 (default) → 0.6 mid-task
    assert mid_task_urgency(0.5, has_failures=False) == 0.6


def test_husam_failures_stack():
    assert abs(mid_task_urgency(0.5, has_failures=True) - 0.7) < 1e-9
