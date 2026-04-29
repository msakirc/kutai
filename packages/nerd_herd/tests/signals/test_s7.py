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
    # 30k tok consumed in last 5 min → 6000 tok/min sustained rate.
    # 30s to reset → extrapolated = 3000 tok, vs 1000 remaining → ratio 3.0
    # Excess over 0.70 = 2.30; clamped to 1.0 → pressure -1.0
    log.record(provider="groq", model="x", tokens=15000, calls=20, now=now - 200)
    log.record(provider="groq", model="x", tokens=15000, calls=20, now=now - 30)
    m = _matrix(tpm=RateLimit(limit=6000, remaining=1000,
                              reset_at=int(now)+30))
    p = s7_burn_rate(m, provider="groq", model="x", burn_log=log, now=now)
    assert p < -0.3
