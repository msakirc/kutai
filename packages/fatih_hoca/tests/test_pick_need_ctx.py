"""TDD test: Pick.need_ctx field computed in select()."""
from fatih_hoca.types import Pick
from fatih_hoca.need_ctx import compute_need_ctx, MIN_CTX


def test_pick_has_need_ctx_default_zero():
    p = Pick(model=object(), min_time_seconds=1.0)
    assert p.need_ctx == 0


def test_select_uses_compute_need_ctx_formula_for_local():
    # The value select() would compute for a local pick with a 20k min-context
    # task on a 128k model: ceil-2048(20000)=20480, floored>=MIN_CTX, capped<=128000.
    assert compute_need_ctx(min_context=20000, est_in=0, est_out=0, model_ctx=128000) == 20480
    # And a tiny task floors to MIN_CTX.
    assert compute_need_ctx(min_context=1000, est_in=0, est_out=0, model_ctx=128000) == MIN_CTX
