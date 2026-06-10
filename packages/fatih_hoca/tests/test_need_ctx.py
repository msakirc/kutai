from fatih_hoca.need_ctx import compute_need_ctx, MIN_CTX


def test_floors_to_min_ctx_when_unknown():
    assert compute_need_ctx(min_context=0, est_in=0, est_out=0, model_ctx=128000) == MIN_CTX


def test_small_need_floored():
    assert compute_need_ctx(min_context=4412, est_in=0, est_out=0, model_ctx=128000) == MIN_CTX


def test_ceils_to_2048_block():
    assert compute_need_ctx(min_context=18000, est_in=0, est_out=0, model_ctx=128000) == 18432


def test_estimate_used_when_no_min_context():
    # (4000+4000)*1.3+512 = 10912 → ceil2048 → 12288
    assert compute_need_ctx(min_context=0, est_in=4000, est_out=4000, model_ctx=128000) == 12288


def test_clamped_to_model_ceiling():
    assert compute_need_ctx(min_context=40000, est_in=0, est_out=0, model_ctx=8192) == 8192
