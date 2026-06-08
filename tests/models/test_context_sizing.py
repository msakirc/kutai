"""need-ctx local-load sizing policy (replaces the retired VRAM-floor model).

A local model loads at the call's REAL need, floored to MIN_CTX, rounded up to
a 2048-block, capped at the model's trained window. No VRAM math — `--fit` owns
layer fitting. (Supersedes test_local_ctx_floor.py / _floored_baseline_ctx,
deleted 2026-06-07 with the dead code.)
"""
from src.models.context_sizing import MIN_CTX, need_ctx


def test_floors_to_min_ctx_when_task_need_is_small():
    # A tiny / zero task need is floored to MIN_CTX, not below it.
    assert need_ctx(0, 32768) == MIN_CTX
    assert need_ctx(100, 32768) == MIN_CTX


def test_rounds_up_to_2048_block():
    # A need above the floor rounds up to the next 2048-block (KV alignment).
    assert need_ctx(10000, 32768) == 10240  # ceil_2048(10000)
    assert need_ctx(16384, 32768) == 16384  # already aligned


def test_capped_by_model_ceiling():
    # Never exceed the model's trained window.
    assert need_ctx(30000, 8192) == 8192


def test_no_ceiling_when_model_ctx_unknown():
    # model_ctx_ceiling <= 0 → no cap applied (defensive: registry returned 0).
    assert need_ctx(20000, 0) == 20480
