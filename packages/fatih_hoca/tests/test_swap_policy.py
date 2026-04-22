from fatih_hoca.swap_policy import can_swap


def test_under_budget_allows():
    assert can_swap(recent_count=0, priority=5) is True
    assert can_swap(recent_count=2, priority=5) is True


def test_at_budget_blocks():
    assert can_swap(recent_count=3, priority=5) is False


def test_over_budget_blocks():
    assert can_swap(recent_count=10, priority=5) is False


def test_local_only_bypasses():
    assert can_swap(recent_count=10, local_only=True, priority=5) is True


def test_high_priority_bypasses():
    assert can_swap(recent_count=10, priority=9) is True
    assert can_swap(recent_count=100, priority=10) is True


def test_priority_just_below_floor_still_blocked():
    assert can_swap(recent_count=5, priority=8) is False


def test_custom_max_swaps():
    assert can_swap(recent_count=4, priority=5, max_swaps=5) is True
    assert can_swap(recent_count=5, priority=5, max_swaps=5) is False
