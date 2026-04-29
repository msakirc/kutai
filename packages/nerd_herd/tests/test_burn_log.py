import time

from nerd_herd.burn_log import BurnLog


def test_burn_log_records_and_extrapolates():
    log = BurnLog(window_secs=300)
    now = time.time()
    log.record(provider="groq", model="llama-8b", tokens=1000, calls=1, now=now - 60)
    log.record(provider="groq", model="llama-8b", tokens=2000, calls=1, now=now - 30)
    log.record(provider="groq", model="llama-8b", tokens=1500, calls=1, now=now - 10)
    rate = log.rate(provider="groq", model="llama-8b", now=now)
    assert rate.tokens_per_min > 0
    assert rate.calls_per_min > 0


def test_burn_log_drops_old_entries():
    log = BurnLog(window_secs=60)
    log.record(provider="x", model="y", tokens=100, calls=1, now=time.time() - 3600)
    rate = log.rate(provider="x", model="y", now=time.time())
    assert rate.tokens_per_min == 0
    assert rate.calls_per_min == 0
