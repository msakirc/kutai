import pytest
from nerd_herd.swap_budget import SwapBudget


def test_record_swap_increments():
    sb = SwapBudget(max_swaps=3, window_seconds=300)
    assert sb.recent_count() == 0
    sb.record_swap()
    assert sb.recent_count() == 1


def test_can_swap_false_after_limit():
    sb = SwapBudget(max_swaps=2, window_seconds=300)
    sb.record_swap()
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=5) is False


def test_expired_swaps_pruned(monkeypatch):
    current = [1000.0]
    monkeypatch.setattr("nerd_herd.swap_budget.time.monotonic", lambda: current[0])
    sb = SwapBudget(max_swaps=3, window_seconds=1)
    sb.record_swap()
    current[0] += 1.1
    sb.record_swap()
    assert sb.recent_count() == 1


def test_local_only_exemption():
    """local_only=True bypasses the swap limit."""
    sb = SwapBudget(max_swaps=1, window_seconds=300)
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=5) is False
    assert sb.can_swap(local_only=True, priority=5) is True


def test_priority_exemption():
    """priority >= 9 bypasses the swap limit."""
    sb = SwapBudget(max_swaps=1, window_seconds=300)
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=8) is False
    assert sb.can_swap(local_only=False, priority=9) is True


def test_remaining_property():
    sb = SwapBudget(max_swaps=3, window_seconds=300)
    assert sb.remaining == 3
    sb.record_swap()
    assert sb.remaining == 2


def test_exhausted_property():
    sb = SwapBudget(max_swaps=2, window_seconds=300)
    assert sb.exhausted is False
    sb.record_swap()
    sb.record_swap()
    assert sb.exhausted is True
