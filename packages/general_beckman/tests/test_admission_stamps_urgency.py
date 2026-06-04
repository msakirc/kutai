"""Admission must stamp the urgency it used onto the task dict so mid-task
re-selection (coulson/husam) can reuse it. Guards the contract between
admission and fatih_hoca.mid_task_urgency.
"""
import time

from general_beckman.admission import compute_urgency
from general_beckman import _stamp_admission_urgency


def _task(priority=8, age_s=0, unblocks=0):
    return {
        "id": 1,
        "priority": priority,
        "created_at": time.time() - age_s,
        "downstream_unblocks_count": unblocks,
    }


def test_stamp_matches_compute_urgency():
    task = _task(priority=8)
    expected = compute_urgency(task)
    _stamp_admission_urgency(task)
    assert task["_admission_urgency"] == expected


def test_stamp_baseline_is_half():
    task = _task(priority=5)
    _stamp_admission_urgency(task)
    assert abs(task["_admission_urgency"] - 0.5) < 0.01
