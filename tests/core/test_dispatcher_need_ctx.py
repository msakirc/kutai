from src.core.llm_dispatcher import _resolve_load_ctx


def test_prefers_pick_need_ctx():
    assert _resolve_load_ctx(need_ctx=18432, min_context=0, est_in=0, est_out=0) == 18432


def test_falls_back_to_min_context():
    assert _resolve_load_ctx(need_ctx=0, min_context=16384, est_in=0, est_out=0) == 16384


def test_falls_back_to_heuristic():
    assert _resolve_load_ctx(need_ctx=0, min_context=0, est_in=1000, est_out=1000) == 3112


def test_all_zero_returns_zero():
    assert _resolve_load_ctx(need_ctx=0, min_context=0, est_in=0, est_out=0) == 0


def test_grown_prompt_est_overrides_stale_need_ctx():
    """pick.need_ctx is computed once at admission and frozen across ReAct
    iterations (pick_for_iter reuses the admitted pick on no-failure turns).
    When the live conversation grows, react recomputes est_in =
    count_tokens(messages) fresh each iteration, but a need_ctx-first resolver
    returns the STALE small admission value — so the dispatcher's
    loaded_ctx_insufficient guard compares against a stale target, reuses the
    small-loaded local model, and the grown prompt overflows it
    (context_overflow recurrence: Qwen3.5-9B / gemma-26B / analyst, 2026-06).
    The load ctx must track the LARGER of the frozen need_ctx and the fresh
    prompt estimate so the resize-on-growth machinery actually fires.
    (ensure_model clamps to the model's trained ceiling, so over-asking is safe.)"""
    fresh = int((30000 + 2000) * 1.3) + 512  # 42112
    result = _resolve_load_ctx(need_ctx=8192, min_context=0, est_in=30000, est_out=2000)
    assert result == fresh
    assert result > 8192
