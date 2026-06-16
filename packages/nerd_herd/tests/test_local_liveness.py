"""Local-inference liveness tracker.

When llama-server cannot bring ANY model up (port collision, crashed server,
VRAM wall, missing GGUF, driver fault), the app must lay off ALL local at
selection instead of admitting every task against a dead server (live
2026-06-16: hours with not a single task processed). The per-model circuit
breaker can't express "the whole local server is down" — it resets its counter
whenever a different model is attempted.

This tracker counts consecutive local load failures across ANY model and trips
a process-level "down" flag for a cooldown, with half-open recovery: after the
cooldown a single probe is allowed; one more failure re-trips immediately, a
success fully recovers.
"""
from __future__ import annotations

from nerd_herd.local_liveness import LocalLivenessTracker


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _tracker(clock):
    return LocalLivenessTracker(threshold=5, cooldown_s=300.0, clock=clock)


def test_below_threshold_not_down():
    t = _tracker(_Clock())
    for _ in range(4):
        t.record_load(False)
    assert t.is_down() is False


def test_at_threshold_trips_down():
    t = _tracker(_Clock())
    for _ in range(5):
        t.record_load(False)
    assert t.is_down() is True


def test_success_resets_streak():
    t = _tracker(_Clock())
    for _ in range(4):
        t.record_load(False)
    t.record_load(True)
    for _ in range(4):
        t.record_load(False)
    assert t.is_down() is False


def test_cooldown_expiry_is_half_open():
    clk = _Clock()
    t = _tracker(clk)
    for _ in range(5):
        t.record_load(False)
    assert t.is_down() is True
    clk.t += 301  # cooldown elapsed
    assert t.is_down() is False  # half-open: a probe is allowed


def test_half_open_single_failure_retrips_immediately():
    clk = _Clock()
    t = _tracker(clk)
    for _ in range(5):
        t.record_load(False)
    clk.t += 301
    assert t.is_down() is False  # half-open probe allowed
    t.record_load(False)         # the probe failed
    assert t.is_down() is True   # re-tripped on a SINGLE failure, not 5


def test_half_open_success_recovers_fully():
    clk = _Clock()
    t = _tracker(clk)
    for _ in range(5):
        t.record_load(False)
    clk.t += 301
    assert t.is_down() is False
    t.record_load(True)          # probe succeeded
    assert t.is_down() is False
    # full threshold required again to trip
    for _ in range(4):
        t.record_load(False)
    assert t.is_down() is False
