"""Task urgency helpers — Task 17."""
import time

from general_beckman.types import (
    task_age_seconds,
    task_preselected_pick,
    task_unblocks_count,
)


def test_task_age_seconds_derived_from_created_at():
    t = {"id": 1, "created_at": time.time() - 120, "priority": 5, "difficulty": 3}
    assert 110 < task_age_seconds(t) < 130


def test_task_age_seconds_missing_created_at_returns_zero():
    assert task_age_seconds({"id": 1}) == 0.0


def test_task_preselected_pick_defaults_none():
    assert task_preselected_pick({"id": 1}) is None


def test_task_preselected_pick_returns_attached_value():
    pick_obj = object()
    assert task_preselected_pick({"id": 1, "preselected_pick": pick_obj}) is pick_obj


def test_task_downstream_unblocks_count_default_zero():
    assert task_unblocks_count({"id": 1}) == 0


def test_task_downstream_unblocks_count_reads_field():
    assert task_unblocks_count({"id": 1, "downstream_unblocks_count": 7}) == 7
