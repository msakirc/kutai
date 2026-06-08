"""TDD test: Pick.need_ctx field computed in select()."""
from fatih_hoca.types import Pick


def test_pick_has_need_ctx_default_zero():
    p = Pick(model=object(), min_time_seconds=1.0)
    assert p.need_ctx == 0
