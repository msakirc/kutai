import time
import pytest

from nerd_herd.burn_log import BurnLog
from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s7_burn_rate import s7_burn_rate


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s7_zero_when_no_history():
    log = BurnLog(window_secs=300)
    m = _matrix(tpm=RateLimit(limit=6000, remaining=6000, reset_at=int(time.time())+60))
    p = s7_burn_rate(m, provider="groq", model="x", burn_log=log, now=time.time())
    assert p == 0.0


def test_s7_negative_when_burn_extrapolates_over_remaining():
    log = BurnLog(window_secs=300)
    now = time.time()
    log.record(provider="groq", model="x", tokens=1000, calls=5, now=now - 60)
    log.record(provider="groq", model="x", tokens=2000, calls=8, now=now - 10)
    m = _matrix(tpm=RateLimit(limit=6000, remaining=2000,
                              reset_at=int(now)+30))
    p = s7_burn_rate(m, provider="groq", model="x", burn_log=log, now=now)
    assert p < -0.3
