from nerd_herd.types import RateLimit, RateLimitMatrix


def test_matrix_has_all_axes():
    m = RateLimitMatrix()
    # Request axes
    for axis in ("rpm", "rph", "rpd", "rpw", "rpmonth"):
        assert isinstance(getattr(m, axis), RateLimit)
    # Token axes
    for axis in ("tpm", "tph", "tpd", "tpw", "tpmonth", "itpm", "itpd", "otpm", "otpd"):
        assert isinstance(getattr(m, axis), RateLimit)
    # Cost axes
    for axis in ("cpd", "cpmonth"):
        assert isinstance(getattr(m, axis), RateLimit)


def test_populated_cells_iterator_returns_only_filled():
    m = RateLimitMatrix(
        rpm=RateLimit(limit=30),
        tpm=RateLimit(limit=6000),
    )
    populated = dict(m.populated_cells())
    assert "rpm" in populated
    assert "tpm" in populated
    assert "rpd" not in populated


def test_token_cells_filters_correctly():
    m = RateLimitMatrix(
        rpm=RateLimit(limit=30),
        tpm=RateLimit(limit=6000),
        tpd=RateLimit(limit=1_000_000),
    )
    token_axes = {n for n, _ in m.token_cells()}
    assert token_axes == {"tpm", "tpd"}


def test_request_cells_filters_correctly():
    m = RateLimitMatrix(
        rpm=RateLimit(limit=30),
        tpm=RateLimit(limit=6000),
        rpd=RateLimit(limit=14_400),
    )
    req_axes = {n for n, _ in m.request_cells()}
    assert req_axes == {"rpm", "rpd"}
