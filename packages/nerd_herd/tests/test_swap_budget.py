import pytest
from nerd_herd.swap_budget import SwapBudget


def test_record_swap_increments():
    sb = SwapBudget(window_seconds=300)
    assert sb.recent_count() == 0
    sb.record_swap()
    assert sb.recent_count() == 1


def test_expired_swaps_pruned(monkeypatch):
    current = [1000.0]
    monkeypatch.setattr("nerd_herd.swap_budget.time.monotonic", lambda: current[0])
    sb = SwapBudget(window_seconds=1)
    sb.record_swap()
    current[0] += 1.1
    sb.record_swap()
    assert sb.recent_count() == 1


def test_multiple_records_accumulate():
    sb = SwapBudget(window_seconds=300)
    for _ in range(5):
        sb.record_swap()
    assert sb.recent_count() == 5
