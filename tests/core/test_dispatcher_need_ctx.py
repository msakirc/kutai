from src.core.llm_dispatcher import _resolve_load_ctx


def test_prefers_pick_need_ctx():
    assert _resolve_load_ctx(need_ctx=18432, min_context=0, est_in=0, est_out=0) == 18432


def test_falls_back_to_min_context():
    assert _resolve_load_ctx(need_ctx=0, min_context=16384, est_in=0, est_out=0) == 16384


def test_falls_back_to_heuristic():
    assert _resolve_load_ctx(need_ctx=0, min_context=0, est_in=1000, est_out=1000) == 3112


def test_all_zero_returns_zero():
    assert _resolve_load_ctx(need_ctx=0, min_context=0, est_in=0, est_out=0) == 0
