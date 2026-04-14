from fatih_hoca.types import Pick, Failure, SwapBudget


def test_pick_fields():
    pick = Pick(model=None, min_time_seconds=30.0)
    assert pick.min_time_seconds == 30.0


def test_failure_fields():
    f = Failure(model="qwen3-30b", reason="timeout", latency=120.0)
    assert f.model == "qwen3-30b"
    assert f.reason == "timeout"
    assert f.latency == 120.0


def test_failure_no_latency():
    f = Failure(model="groq/llama-8b", reason="rate_limit")
    assert f.latency is None


def test_swap_budget_allows_first_swap():
    sb = SwapBudget(max_swaps=3, window_seconds=300)
    assert sb.can_swap(local_only=False, priority=5) is True
    assert sb.remaining == 3


def test_swap_budget_blocks_after_max():
    sb = SwapBudget(max_swaps=2, window_seconds=300)
    sb.record_swap()
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=5) is False
    assert sb.remaining == 0


def test_swap_budget_exempt_local_only():
    sb = SwapBudget(max_swaps=1, window_seconds=300)
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=5) is False
    assert sb.can_swap(local_only=True, priority=5) is True


def test_swap_budget_exempt_high_priority():
    sb = SwapBudget(max_swaps=1, window_seconds=300)
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=9) is True
    assert sb.can_swap(local_only=False, priority=8) is False


def test_swap_budget_exhausted_property():
    sb = SwapBudget(max_swaps=1, window_seconds=300)
    assert sb.exhausted is False
    sb.record_swap()
    assert sb.exhausted is True
