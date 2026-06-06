import time
from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.burn_log import BurnLog
from nerd_herd.signals.s7_burn_rate import s7_burn_rate, SAT


def _matrix(remaining, limit, reset_in):
    now = time.time()
    return RateLimitMatrix(rpd=RateLimit(limit=limit, remaining=remaining,
                                         reset_at=int(now + reset_in))), now


def test_cold_start_zero():
    mtx, now = _matrix(10, 20, 3600)
    bl = BurnLog(300.0)
    assert s7_burn_rate(mtx, provider="p", model="m", burn_log=bl, now=now) == 0.0


def test_ramp_is_continuous_from_zero_no_deadband():
    # Light burn that yields ratio well under the OLD 0.70 gate must now be
    # nonzero (de-blinded) — proves the dead-band is gone.
    mtx, now = _matrix(remaining=10_000, limit=14_400, reset_in=3600)
    bl = BurnLog(300.0)
    # ~12 calls in window → calls_per_min = 12*60/300 = 2.4; extrapolated over
    # 60min = 144; ratio = 144/10000 = 0.0144 → old gate gave exactly 0.
    for i in range(12):
        bl.record(provider="p", model="m", tokens=100, calls=1, now=now - i)
    s7 = s7_burn_rate(mtx, provider="p", model="m", burn_log=bl, now=now)
    assert s7 < 0.0          # de-blinded: was 0 under the 0.70 gate
    assert s7 > -0.2         # but still a whisper, not a shout


def test_monotonic_in_burn_and_saturates():
    mtx, now = _matrix(remaining=20, limit=20, reset_in=3600)
    mags = []
    for n in (1, 3, 6, 12, 30):
        bl = BurnLog(300.0)
        for i in range(n):
            bl.record(provider="p", model="m", tokens=100, calls=1, now=now - i)
        mags.append(-s7_burn_rate(mtx, provider="p", model="m", burn_log=bl, now=now))
    assert all(b >= a - 1e-9 for a, b in zip(mags, mags[1:]))
    assert mags[-1] == 1.0   # heavy burn on a tiny tank saturates at -1


def test_sat_constant_exists():
    assert SAT > 0
