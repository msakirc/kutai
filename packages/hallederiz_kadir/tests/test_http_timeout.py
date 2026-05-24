"""Bug F (intake #73) — the local "no wall-clock cap" timeout sentinel.

`llm_dispatcher.execute` passes timeout=0.0 for local models to mean "no HTTP
deadline — let the stream watchdog govern hangs". The old
`max(10.0, timeout - 5.0)` collapsed that into a 10s HTTP timeout, so a slow
CPU-offloaded/thinking local model (selector-estimated ~100s) got aborted at
10s with APITimeoutError. `_http_timeout` must return None for the sentinel.
"""
from hallederiz_kadir.caller import _http_timeout


def test_no_cap_sentinel_returns_none():
    # 0.0 / negative = "no wall-clock cap" -> no HTTP timeout (was 10s before).
    assert _http_timeout(0.0) is None
    assert _http_timeout(-1.0) is None


def test_positive_budget_leaves_5s_headroom():
    assert _http_timeout(35.0) == 30.0
    assert _http_timeout(605.0) == 600.0


def test_positive_budget_floored_at_10s():
    # Tiny positive budgets still get a usable floor, never < 10s.
    assert _http_timeout(12.0) == 10.0
    assert _http_timeout(1.0) == 10.0
