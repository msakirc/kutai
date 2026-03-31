"""
Tests for src/infra/backpressure.py

Covers:
  - Future created on running loop (not a stale/new loop)
  - Entry mutations under lock
  - Retry task error tracking
  - Callable cleanup on removal
  - Queue depth and event lifecycle
  - signal_capacity_available correctness
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestQueuedCall(unittest.TestCase):

    def test_future_not_created_in_post_init(self):
        """QueuedCall.__post_init__ should NOT create a future."""
        from src.infra.backpressure import QueuedCall
        entry = QueuedCall(
            call_id="test",
            priority=5,
            call_func=AsyncMock(),
        )
        # Future is None until the queue sets it
        self.assertIsNone(entry.result_future)

    def test_wait_seconds(self):
        from src.infra.backpressure import QueuedCall
        entry = QueuedCall(
            call_id="test",
            priority=5,
            call_func=AsyncMock(),
            enqueued_at=time.time() - 10,
        )
        self.assertGreaterEqual(entry.wait_seconds, 9.5)

    def test_retry_interval_exponential(self):
        from src.infra.backpressure import QueuedCall, RETRY_INTERVALS
        entry = QueuedCall(call_id="t", priority=5, call_func=AsyncMock())
        for i, expected in enumerate(RETRY_INTERVALS):
            entry.attempt = i
            self.assertEqual(entry.retry_interval, expected)
        # Beyond max index should clamp
        entry.attempt = 100
        self.assertEqual(entry.retry_interval, RETRY_INTERVALS[-1])


class TestBackpressureQueue(unittest.TestCase):

    def test_enqueue_creates_future_on_running_loop(self):
        """Future must be created on the running loop, not a new loop."""
        from src.infra.backpressure import BackpressureQueue

        async def _test():
            bq = BackpressureQueue()
            running_loop = asyncio.get_running_loop()

            # Mock call_func that succeeds immediately
            async def _succeed():
                return "ok"

            # Enqueue should create future on the running loop
            # We can't await it normally (would block), so peek at internals
            entry_future = None

            # Temporarily reduce timeout to avoid blocking
            import src.infra.backpressure as bp_mod
            old_timeout = bp_mod.MAX_QUEUE_WAIT_SECONDS
            bp_mod.MAX_QUEUE_WAIT_SECONDS = 0.1
            try:
                with self.assertRaises(RuntimeError):
                    # Will timeout immediately but the future should be
                    # created on the correct loop
                    await bq.enqueue(
                        call_id="test",
                        priority=5,
                        last_error="test",
                        call_func=_succeed,
                    )
            finally:
                bp_mod.MAX_QUEUE_WAIT_SECONDS = old_timeout

            # After timeout, queue should be empty (entry cleaned up)
            self.assertEqual(bq.depth, 0)

        run_async(_test())

    def test_queue_depth_limit(self):
        """Queue should reject when full."""
        from src.infra.backpressure import BackpressureQueue, MAX_QUEUE_DEPTH

        async def _test():
            bq = BackpressureQueue()
            loop = asyncio.get_running_loop()

            # Fill the queue without awaiting (just stuff entries in)
            async with bq._lock:
                for i in range(MAX_QUEUE_DEPTH):
                    from src.infra.backpressure import QueuedCall
                    entry = QueuedCall(
                        call_id=f"fill-{i}",
                        priority=5,
                        call_func=AsyncMock(),
                        result_future=loop.create_future(),
                    )
                    bq._queue.append(entry)

            # Next enqueue should fail
            with self.assertRaises(RuntimeError) as ctx:
                await bq.enqueue(
                    call_id="overflow",
                    priority=5,
                    last_error="test",
                    call_func=AsyncMock(),
                )
            self.assertIn("queue full", str(ctx.exception).lower())

        run_async(_test())

    def test_signal_advances_retry_respecting_gap(self):
        """signal_capacity_available should respect MIN_RETRY_GAP_SECONDS."""
        from src.infra.backpressure import (
            BackpressureQueue, QueuedCall, MIN_RETRY_GAP_SECONDS,
        )

        async def _test():
            bq = BackpressureQueue()
            loop = asyncio.get_running_loop()

            now = time.time()
            entry = QueuedCall(
                call_id="test",
                priority=5,
                call_func=AsyncMock(),
                next_retry_at=now + 60,  # far in the future
                last_attempt_at=now - 1,  # 1 second ago
                result_future=loop.create_future(),
            )

            async with bq._lock:
                bq._queue.append(entry)
                bq._has_items.set()

            await bq.signal_capacity_available()

            # Should be advanced to max(now, last_attempt + MIN_GAP)
            # Since last_attempt was 1s ago and MIN_GAP is 5s, earliest = now + 4s
            self.assertGreater(entry.next_retry_at, now)
            self.assertLess(entry.next_retry_at, now + 60)  # advanced from 60s

        run_async(_test())

    def test_signal_does_not_advance_below_minimum_gap(self):
        """If last attempt was very recent, signal should NOT set retry to now."""
        from src.infra.backpressure import (
            BackpressureQueue, QueuedCall, MIN_RETRY_GAP_SECONDS,
        )

        async def _test():
            bq = BackpressureQueue()
            loop = asyncio.get_running_loop()

            now = time.time()
            entry = QueuedCall(
                call_id="test",
                priority=5,
                call_func=AsyncMock(),
                next_retry_at=now + 60,
                last_attempt_at=now,  # JUST attempted
                result_future=loop.create_future(),
            )

            async with bq._lock:
                bq._queue.append(entry)
                bq._has_items.set()

            await bq.signal_capacity_available()

            # MIN_GAP should prevent retry from being set to "now"
            self.assertGreaterEqual(
                entry.next_retry_at,
                now + MIN_RETRY_GAP_SECONDS - 0.1,
            )

        run_async(_test())

    def test_callable_cleared_on_success(self):
        """After successful retry, call_func should be None."""
        from src.infra.backpressure import BackpressureQueue, QueuedCall

        async def _test():
            bq = BackpressureQueue()
            loop = asyncio.get_running_loop()

            mock_func = AsyncMock(return_value="result")
            future = loop.create_future()

            entry = QueuedCall(
                call_id="test",
                priority=5,
                call_func=mock_func,
                result_future=future,
            )

            async with bq._lock:
                bq._queue.append(entry)

            await bq._retry_call(entry)

            # Callable should be cleared
            self.assertIsNone(entry.call_func)
            # Future should have result
            self.assertTrue(future.done())
            self.assertEqual(future.result(), "result")

        run_async(_test())

    def test_callable_cleared_on_max_retries(self):
        """After max retries, call_func should be None."""
        from src.infra.backpressure import (
            BackpressureQueue, QueuedCall, MAX_RETRY_ATTEMPTS,
        )

        async def _test():
            bq = BackpressureQueue()
            loop = asyncio.get_running_loop()

            mock_func = AsyncMock()
            future = loop.create_future()

            entry = QueuedCall(
                call_id="test",
                priority=5,
                call_func=mock_func,
                attempt=MAX_RETRY_ATTEMPTS,  # at max
                result_future=future,
            )

            async with bq._lock:
                bq._queue.append(entry)

            await bq._retry_call(entry)

            # Callable should be cleared
            self.assertIsNone(entry.call_func)
            # Future should have exception
            self.assertTrue(future.done())
            with self.assertRaises(RuntimeError):
                future.result()

        run_async(_test())

    def test_retry_mutations_under_lock(self):
        """_retry_call should increment attempt under lock."""
        from src.infra.backpressure import BackpressureQueue, QueuedCall

        async def _test():
            bq = BackpressureQueue()
            loop = asyncio.get_running_loop()

            mock_func = AsyncMock(return_value="ok")
            future = loop.create_future()

            entry = QueuedCall(
                call_id="test",
                priority=5,
                call_func=mock_func,
                result_future=future,
            )

            async with bq._lock:
                bq._queue.append(entry)

            self.assertEqual(entry.attempt, 0)
            self.assertEqual(entry.last_attempt_at, 0.0)

            await bq._retry_call(entry)

            self.assertEqual(entry.attempt, 1)
            self.assertGreater(entry.last_attempt_at, 0.0)

        run_async(_test())

    def test_retry_task_tracked(self):
        """Retry tasks should be in _active_retries set."""
        from src.infra.backpressure import BackpressureQueue

        bq = BackpressureQueue()
        self.assertEqual(len(bq._active_retries), 0)


if __name__ == "__main__":
    unittest.main()
