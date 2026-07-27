"""TDD for the 2026-07-27 daemon-thread heartbeat + loop-tick watchdog.

The liveness heartbeat must be written by an OS thread so a legitimately-blocked
event loop (first-agent cold imports ~250s) can't starve it and trigger a false
hung-kill. But a TRULY wedged loop must still be caught: the loop bumps a tick;
if it hasn't ticked within wedge_threshold, the thread withholds the heartbeat so
the hub restarts it.
"""
import time

from src.core.threaded_heartbeat import ThreadedHeartbeat


def test_loop_alive_true_when_recently_ticked():
    clk = [1000.0]
    hb = ThreadedHeartbeat(["x"], wedge_threshold=480.0, _clock=lambda: clk[0])
    hb.bump()                 # tick at t=1000
    clk[0] = 1000.0 + 250.0   # 250s later — a long but legitimate block
    assert hb.loop_alive() is True


def test_loop_alive_false_when_tick_exceeds_wedge_threshold():
    clk = [1000.0]
    hb = ThreadedHeartbeat(["x"], wedge_threshold=480.0, _clock=lambda: clk[0])
    hb.bump()                 # tick at t=1000
    clk[0] = 1000.0 + 481.0   # past the wedge threshold → loop is wedged
    assert hb.loop_alive() is False


def test_write_once_writes_a_fresh_timestamp(tmp_path):
    p = tmp_path / "hb"
    hb = ThreadedHeartbeat([str(p)])
    hb._write_once()
    assert p.exists()
    written = float(p.read_text().strip())
    assert abs(written - time.time()) < 5


def test_thread_writes_while_loop_alive(tmp_path):
    p = tmp_path / "hb"
    hb = ThreadedHeartbeat([str(p)], interval=0.03, wedge_threshold=100.0)
    hb.bump()  # loop alive
    hb.start_thread()
    try:
        time.sleep(0.2)
        assert p.exists(), "daemon thread must write the heartbeat while alive"
        first = float(p.read_text().strip())
        time.sleep(0.15)
        assert float(p.read_text().strip()) >= first
    finally:
        hb.stop()


def test_thread_withholds_when_wedged(tmp_path):
    p = tmp_path / "hb"
    # wedge_threshold tiny + last tick far in the past → loop considered wedged
    hb = ThreadedHeartbeat([str(p)], interval=0.03, wedge_threshold=0.01)
    hb._last_tick = time.time() - 100.0
    hb.start_thread()
    try:
        time.sleep(0.2)
        assert not p.exists(), "wedged loop must NOT get a fresh heartbeat"
    finally:
        hb.stop()
