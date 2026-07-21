"""Outer hub-liveness watchdog decision logic.

The hub writes `hub.alive` (a timestamp) on a fixed cadence. A small Task-
Scheduler task runs this every few minutes: if `hub.alive` is stale AND a hub
process is still alive (hung), it kills the hub → the main task's
restart-on-failure relaunches it. A dead hub is left alone (the scheduler's
exit-code handling owns that). Threshold > max backoff so a legit 300s backoff
sleep is never read as a hang.
"""

import tempfile
from pathlib import Path

from yasar_usta.watchdog import (
    read_alive_ts, is_stale, decide_kill, run_once, DEFAULT_STALE_SECONDS,
)


class TestReadAliveTs:
    def test_reads_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hub.alive"
            p.write_text("1234.5")
            assert read_alive_ts(p) == 1234.5

    def test_missing_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert read_alive_ts(Path(tmp) / "nope.alive") is None

    def test_corrupt_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hub.alive"
            p.write_text("not-a-number")
            assert read_alive_ts(p) is None


class TestIsStale:
    def test_fresh_not_stale(self):
        assert is_stale(1000.0, now=1000.0 + 10) is False

    def test_stale_when_past_threshold(self):
        assert is_stale(1000.0, now=1000.0 + DEFAULT_STALE_SECONDS + 1) is True

    def test_none_never_stale(self):
        assert is_stale(None, now=9_999_999.0) is False


class TestDecideKill:
    def test_stale_with_live_hub_returns_pids(self):
        ts = 1000.0
        now = ts + DEFAULT_STALE_SECONDS + 5
        assert decide_kill(ts, now, [111, 222]) == [111, 222]

    def test_fresh_kills_nothing(self):
        assert decide_kill(1000.0, 1000.0 + 5, [111]) == []

    def test_missing_alive_kills_nothing(self):
        assert decide_kill(None, 9_999_999.0, [111]) == []

    def test_stale_but_no_hub_process_kills_nothing(self):
        ts = 1000.0
        now = ts + DEFAULT_STALE_SECONDS + 5
        assert decide_kill(ts, now, []) == []  # dead hub → scheduler's job


class TestRunOnce:
    def test_kills_hung_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hub.alive"
            p.write_text("1000.0")
            killed = []
            out = run_once(
                p, now=1000.0 + DEFAULT_STALE_SECONDS + 5,
                find_pids=lambda: [777], kill=killed.append)
            assert out == [777]
            assert killed == [777]

    def test_leaves_fresh_hub_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hub.alive"
            p.write_text("1000.0")
            killed = []
            out = run_once(p, now=1005.0, find_pids=lambda: [777], kill=killed.append)
            assert out == []
            assert killed == []
