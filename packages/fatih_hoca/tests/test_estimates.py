from fatih_hoca.estimates import Estimates


def test_estimates_total_tokens_computed():
    e = Estimates(in_tokens=1000, out_tokens=2000, iterations=5)
    assert e.total_tokens == (1000 + 2000) * 5


def test_estimates_per_call_tokens():
    e = Estimates(in_tokens=1000, out_tokens=2000, iterations=5)
    assert e.per_call_tokens == 3000


def test_step_token_overrides_known_step():
    from fatih_hoca.step_overrides import STEP_TOKEN_OVERRIDES
    e = STEP_TOKEN_OVERRIDES["4.5b"]
    assert e.out_tokens >= 100_000  # openapi_spec is heavy
