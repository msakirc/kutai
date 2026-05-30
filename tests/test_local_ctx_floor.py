"""VRAM-aware baseline context floor (regression: mission_79, 2026-05-30).

The intake-#73 anti-churn floor (`BASELINE_LOCAL_CTX`) was raising the
dynamically-sized context window back up to 16384 *without re-checking VRAM*.
Under a transient VRAM-pressure spike the dynamic calc correctly collapsed to
~4096, the floor bumped it to 16384, and llama-server OOM'd loading the 9B
(weights+KV no longer fit in the ~4.2GB free) → circuit breaker → with cloud
exhausted, researcher starved into a DLQ-storm.

The floor must override the *volatile RAM* bound (what #73 was for) but NEVER
the *hard VRAM* bound (exceeding it = CUDA OOM, not a swappable limit).
"""


def test_baseline_floor_respects_vram_ceiling():
    # dynamic calc returned 4096 (VRAM-bound); baseline wants 16384 but the
    # GPU-tier VRAM ceiling is only 4096 → stay at 4096, do NOT bump to 16384.
    from src.models.local_model_manager import _floored_baseline_ctx
    out = _floored_baseline_ctx(
        4096, baseline=16384, model_ctx_ceiling=32768, vram_ctx_ceiling=4096,
    )
    assert out == 4096


def test_baseline_floor_applies_when_vram_ample():
    # A transient *RAM* dip collapsed the dynamic value, but VRAM is ample →
    # the #73 floor still applies (bump 4096 → 16384) so prompts fit first load.
    from src.models.local_model_manager import _floored_baseline_ctx
    out = _floored_baseline_ctx(
        4096, baseline=16384, model_ctx_ceiling=32768, vram_ctx_ceiling=30000,
    )
    assert out == 16384


def test_baseline_floor_capped_by_model_ceiling():
    # Never exceed the model's trained window even when VRAM/baseline would.
    from src.models.local_model_manager import _floored_baseline_ctx
    out = _floored_baseline_ctx(
        2048, baseline=16384, model_ctx_ceiling=8192, vram_ctx_ceiling=30000,
    )
    assert out == 8192


def test_baseline_floor_never_lowers_dynamic_value():
    # A dynamic context already above the baseline is left untouched.
    from src.models.local_model_manager import _floored_baseline_ctx
    out = _floored_baseline_ctx(
        20480, baseline=16384, model_ctx_ceiling=32768, vram_ctx_ceiling=30000,
    )
    assert out == 20480
