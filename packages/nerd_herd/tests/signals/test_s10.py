import pytest
from nerd_herd.signals.s10_failure import s10_failure


def test_s10_zero_failures():
    assert s10_failure(consecutive_failures=0) == 0.0


def test_s10_one_failure():
    assert s10_failure(consecutive_failures=1) == pytest.approx(-0.2, abs=0.01)


def test_s10_three_failures():
    assert s10_failure(consecutive_failures=3) == pytest.approx(-0.5, abs=0.01)


def test_s10_clamps_at_minus_half():
    assert s10_failure(consecutive_failures=10) == pytest.approx(-0.5, abs=0.01)
