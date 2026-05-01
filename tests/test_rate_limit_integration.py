# tests/test_rate_limit_integration.py
"""
Integration tests for the dynamic rate limit pipeline:
header parsing → rate limiter update → quota planner adjustment → model scoring.
"""

import sys, os, unittest, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuleden_donen_var.header_parser import parse_rate_limit_headers
from src.models.rate_limiter import RateLimitManager, RateLimitState
from fatih_hoca.requirements import QuotaPlanner


class TestEndToEndHeaderFlow(unittest.TestCase):
    """Headers → RateLimitManager → QuotaPlanner threshold adjustment."""

    def test_openai_headers_update_manager_and_planner(self):
        """Full pipeline: parse OpenAI headers, update manager, feed planner."""
        manager = RateLimitManager()
        manager.register_model("openai/gpt-4o", "openai", rpm=500, tpm=200000)
        planner = QuotaPlanner()

        # Simulate healthy response headers
        headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "480",
            "x-ratelimit-reset-requests": "6s",
            "x-ratelimit-limit-tokens": "200000",
            "x-ratelimit-remaining-tokens": "190000",
            "x-ratelimit-reset-tokens": "6s",
        }
        snap = parse_rate_limit_headers("openai", headers)
        self.assertIsNotNone(snap)

        manager.update_from_headers("openai/gpt-4o", "openai", snap)

        # Manager should have discovered real limits
        status = manager.get_status()
        self.assertTrue(status["models"]["openai/gpt-4o"]["discovered"])

        # Feed utilization to planner (4% used)
        util_pct = (1.0 - 480 / 500) * 100
        planner.update_paid_utilization("openai", util_pct, reset_in=6)
        planner.set_max_upcoming_difficulty(4)
        planner.recalculate()

        # Healthy quota → low threshold
        self.assertLessEqual(planner.expensive_threshold, 5)

    def test_exhausted_headers_raise_planner_threshold(self):
        """Near-exhausted headers should push threshold high."""
        manager = RateLimitManager()
        manager.register_model("anthropic/claude-3.5-sonnet", "anthropic", rpm=50, tpm=80000)
        planner = QuotaPlanner()

        headers = {
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "3",
            "anthropic-ratelimit-requests-reset": "2026-01-27T12:00:30Z",
            "anthropic-ratelimit-tokens-limit": "80000",
            "anthropic-ratelimit-tokens-remaining": "5000",
            "anthropic-ratelimit-tokens-reset": "2026-01-27T12:00:30Z",
        }
        snap = parse_rate_limit_headers("anthropic", headers)
        manager.update_from_headers("anthropic/claude-3.5-sonnet", "anthropic", snap)

        util_pct = (1.0 - 3 / 50) * 100  # 94%
        planner.update_paid_utilization("anthropic", util_pct, reset_in=3600)
        planner.set_max_upcoming_difficulty(5)
        planner.recalculate()

        self.assertGreaterEqual(planner.expensive_threshold, 8)

    def test_daily_limit_tracked_through_pipeline(self):
        """Cerebras daily limits should be stored in manager state."""
        manager = RateLimitManager()
        manager.register_model("cerebras/llama3.1-8b", "cerebras", rpm=30, tpm=131072)

        headers = {
            "x-ratelimit-limit-tokens-minute": "131072",
            "x-ratelimit-remaining-tokens-minute": "100000",
            "x-ratelimit-reset-tokens-minute": "45.5",
            "x-ratelimit-limit-requests-day": "1000",
            "x-ratelimit-remaining-requests-day": "50",
            "x-ratelimit-reset-requests-day": "33011.382867",
        }
        snap = parse_rate_limit_headers("cerebras", headers)
        manager.update_from_headers("cerebras/llama3.1-8b", "cerebras", snap)

        # Check daily limits are stored
        model_state = manager.model_limits["cerebras/llama3.1-8b"]
        self.assertEqual(model_state.rpd_limit, 1000)
        self.assertEqual(model_state.rpd_remaining, 50)
        self.assertIsNotNone(model_state.rpd_reset_at)

    def test_429_recording_feeds_planner(self):
        """429 errors should raise planner threshold via record_429."""
        planner = QuotaPlanner()
        planner.update_paid_utilization("openai", 60.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(5)
        planner.recalculate()
        initial = planner.expensive_threshold

        # Record multiple 429s
        for _ in range(4):
            planner.record_429("openai")
        planner.recalculate()

        self.assertGreater(planner.expensive_threshold, initial)
        self.assertGreaterEqual(planner.expensive_threshold, 8)


class TestBackpressureSignal(unittest.TestCase):
    """Test that signal_capacity_available makes queued calls retry-eligible."""

    def test_signal_sets_retry_times_to_now(self):
        """signal_capacity_available should reset next_retry_at for all queued entries."""
        import asyncio
        from src.infra.backpressure import BackpressureQueue, QueuedCall

        queue = BackpressureQueue()

        # Manually add entries with future retry times
        entry1 = QueuedCall(
            call_id="test1", priority=5,
            next_retry_at=time.time() + 300,  # 5 min in future
        )
        entry2 = QueuedCall(
            call_id="test2", priority=3,
            next_retry_at=time.time() + 600,  # 10 min in future
        )
        queue._queue = [entry1, entry2]

        before = time.time()
        queue.signal_capacity_available()
        after = time.time()

        # Both entries should now be eligible
        for entry in queue._queue:
            self.assertGreaterEqual(entry.next_retry_at, before)
            self.assertLessEqual(entry.next_retry_at, after)

    def test_signal_on_empty_queue_is_noop(self):
        from src.infra.backpressure import BackpressureQueue
        queue = BackpressureQueue()
        queue.signal_capacity_available()  # should not raise


if __name__ == "__main__":
    unittest.main()
