"""Tests for yasar_usta.backoff."""

from yasar_usta.backoff import BackoffTracker


class TestBackoffTracker:
    def test_initial_backoff(self):
        bt = BackoffTracker(steps=[5, 15, 60, 300])
        assert bt.get_delay() == 5

    def test_escalation(self):
        bt = BackoffTracker(steps=[5, 15, 60, 300])
        bt.record_crash()
        assert bt.get_delay() == 15
        bt.record_crash()
        assert bt.get_delay() == 60

    def test_clamps_at_max(self):
        bt = BackoffTracker(steps=[5, 15])
        bt.record_crash()
        bt.record_crash()
        bt.record_crash()
        assert bt.get_delay() == 15

    def test_reset_after_stability(self):
        bt = BackoffTracker(steps=[5, 15], reset_after=10)
        bt.record_crash()
        bt.record_crash()
        assert bt.get_delay() == 15
        # Simulate stable run
        bt.mark_started()
        import time
        bt._start_time = time.time() - 11  # pretend 11s ago
        bt.maybe_reset()
        assert bt.get_delay() == 5
        assert bt.crash_count == 0

    def test_total_crashes_not_reset(self):
        bt = BackoffTracker(steps=[5, 15], reset_after=10)
        bt.record_crash()
        bt.record_crash()
        assert bt.total_crashes == 2
        bt.mark_started()
        import time
        bt._start_time = time.time() - 11
        bt.maybe_reset()
        assert bt.total_crashes == 2  # total preserved
        assert bt.crash_count == 0  # window reset
