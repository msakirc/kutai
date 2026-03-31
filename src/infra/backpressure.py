# backpressure.py
"""
Backpressure queue for model calls that can't be served immediately.

When all models are rate-limited, down, or busy, instead of failing,
requests are queued and retried with exponential backoff.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from src.infra.logging_config import get_logger

logger = get_logger("infra.backpressure")

# Maximum time a request can sit in the backpressure queue
MAX_QUEUE_WAIT_SECONDS: float = 300  # 5 minutes

# Maximum queue depth before rejecting new requests
MAX_QUEUE_DEPTH: int = 20

# Maximum retry attempts before giving up (prevents 43-attempt spirals)
MAX_RETRY_ATTEMPTS: int = 5

# Retry intervals (exponential backoff)
RETRY_INTERVALS: list[float] = [5, 10, 20, 40, 60]

# Minimum time between retries — capacity signals can advance next_retry_at
# but never below this floor relative to the last attempt.
MIN_RETRY_GAP_SECONDS: float = 5.0


@dataclass
class QueuedCall:
    """A model call waiting to be retried."""
    call_id: str
    priority: int
    call_func: Callable[..., Awaitable[Any]]
    attempt: int = 0
    enqueued_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0
    last_error: str = ""
    last_attempt_at: float = 0.0
    result_future: asyncio.Future | None = field(default=None, repr=False)

    def __post_init__(self):
        # Future is created lazily by the queue on enqueue, not here.
        # This avoids the event loop mismatch bug where __post_init__
        # might run outside the running loop.
        pass

    @property
    def wait_seconds(self) -> float:
        return time.time() - self.enqueued_at

    @property
    def is_expired(self) -> bool:
        return self.wait_seconds > MAX_QUEUE_WAIT_SECONDS

    @property
    def retry_interval(self) -> float:
        idx = min(self.attempt, len(RETRY_INTERVALS) - 1)
        return RETRY_INTERVALS[idx]


class BackpressureQueue:
    """
    Holds model calls that failed due to transient issues
    (rate limits, timeouts, all providers busy).

    The queue processor runs as a background task, retrying
    calls with exponential backoff.

    Integration with call_model:
        Instead of raising RuntimeError, call_model can
        submit to this queue and await the result.
    """

    def __init__(self):
        self._queue: list[QueuedCall] = []
        self._lock = asyncio.Lock()
        self._has_items = asyncio.Event()
        self._running = False
        self._active_retries: set[asyncio.Task] = set()

        # Stats
        self.total_queued: int = 0
        self.total_retried: int = 0
        self.total_succeeded: int = 0
        self.total_expired: int = 0

    async def enqueue(
        self,
        call_id: str,
        priority: int,
        last_error: str,
        call_func: Callable[..., Awaitable[Any]],
    ) -> Any:
        """
        Enqueue a failed model call for retry.

        Returns the eventual result (blocks until success or expiry).
        Raises RuntimeError if the call expires without success.

        Args:
            call_id: identifier for logging
            priority: task priority (higher = retry sooner)
            last_error: the error that caused the initial failure
            call_func: async callable (no args) that retries the model call
        """
        # Create the future on the RUNNING loop — guaranteed correct.
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        entry = QueuedCall(
            call_id=call_id,
            priority=priority,
            call_func=call_func,
            last_error=last_error,
            next_retry_at=time.time() + RETRY_INTERVALS[0],
            result_future=future,
        )

        async with self._lock:
            if len(self._queue) >= MAX_QUEUE_DEPTH:
                raise RuntimeError(
                    f"Backpressure queue full ({MAX_QUEUE_DEPTH} items). "
                    f"System is overloaded. Last error: {last_error}"
                )
            self._queue.append(entry)
            self._queue.sort(
                key=lambda e: (-e.priority, e.enqueued_at)
            )
            self.total_queued += 1
            self._has_items.set()

        logger.info(
            f"Backpressure: queued '{call_id}' "
            f"(priority={priority}, depth={len(self._queue)}, "
            f"error: {last_error[:80]})"
        )

        # Wait for the result
        try:
            result = await asyncio.wait_for(
                future,
                timeout=MAX_QUEUE_WAIT_SECONDS,
            )
            return result
        except asyncio.TimeoutError:
            async with self._lock:
                self._queue = [e for e in self._queue if e is not entry]
                self.total_expired += 1
            # Clear callable reference to release captured objects
            entry.call_func = None  # type: ignore[assignment]
            raise RuntimeError(
                f"Backpressure queue timeout after "
                f"{MAX_QUEUE_WAIT_SECONDS}s for '{call_id}'. "
                f"Last error: {entry.last_error}"
            )

    async def run_processor(self) -> None:
        """
        Background task: processes queued calls with backoff.
        Run as: asyncio.create_task(queue.run_processor())
        """
        self._running = True
        logger.info("Backpressure queue processor started")

        while self._running:
            # Wait for items
            try:
                await asyncio.wait_for(
                    self._has_items.wait(), timeout=10,
                )
            except asyncio.TimeoutError:
                continue

            now = time.time()

            async with self._lock:
                # Remove expired entries
                expired = [e for e in self._queue if e.is_expired]
                for entry in expired:
                    if entry.result_future and not entry.result_future.done():
                        entry.result_future.set_exception(
                            RuntimeError(
                                f"Backpressure expired: {entry.last_error}"
                            )
                        )
                    entry.call_func = None  # type: ignore[assignment]
                    self.total_expired += 1
                self._queue = [
                    e for e in self._queue if not e.is_expired
                ]

                # Find entries ready for retry
                ready = [
                    e for e in self._queue
                    if now >= e.next_retry_at
                    and e.result_future is not None
                    and not e.result_future.done()
                ]

                # Clear event ONLY if queue is truly empty — check while
                # still holding the lock so no enqueue can slip in between.
                if not self._queue:
                    self._has_items.clear()

            # Process ready entries (outside lock) — tracked tasks
            for entry in ready[:3]:
                task = asyncio.create_task(self._retry_call(entry))
                self._active_retries.add(task)
                task.add_done_callback(self._on_retry_done)

            # Sleep between processor cycles
            await asyncio.sleep(5)

    def _on_retry_done(self, task: asyncio.Task) -> None:
        """Callback when a retry task completes. Logs unhandled exceptions."""
        self._active_retries.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(
                f"Backpressure retry task crashed: {exc}",
                exc_info=exc,
            )

    async def _retry_call(self, entry: QueuedCall) -> None:
        """Attempt to retry a queued call.

        All mutations to entry fields happen under the lock to prevent
        concurrent reads from signal_capacity_available() seeing torn state.
        """
        async with self._lock:
            entry.attempt += 1
            entry.last_attempt_at = time.time()
            self.total_retried += 1
            current_attempt = entry.attempt

        # Give up after max attempts
        if current_attempt > MAX_RETRY_ATTEMPTS:
            logger.warning(
                f"Backpressure: giving up on '{entry.call_id}' "
                f"after {current_attempt} attempts "
                f"({entry.wait_seconds:.1f}s). "
                f"Last error: {entry.last_error[:80]}"
            )
            if entry.result_future and not entry.result_future.done():
                entry.result_future.set_exception(
                    RuntimeError(
                        f"Backpressure exhausted after {current_attempt} retries "
                        f"for '{entry.call_id}'. Last error: {entry.last_error}"
                    )
                )
            async with self._lock:
                self._queue = [e for e in self._queue if e is not entry]
                self.total_expired += 1
            entry.call_func = None  # type: ignore[assignment]
            return

        logger.info(
            f"Backpressure retry: '{entry.call_id}' "
            f"(attempt={current_attempt}/{MAX_RETRY_ATTEMPTS}, "
            f"waited={entry.wait_seconds:.1f}s)"
        )

        try:
            result = await entry.call_func()

            # Success
            if entry.result_future and not entry.result_future.done():
                entry.result_future.set_result(result)
            self.total_succeeded += 1

            async with self._lock:
                self._queue = [e for e in self._queue if e is not entry]

            entry.call_func = None  # type: ignore[assignment]

            logger.info(
                f"Backpressure success: '{entry.call_id}' "
                f"after {current_attempt} retries "
                f"({entry.wait_seconds:.1f}s total wait)"
            )

        except Exception as e:
            async with self._lock:
                entry.last_error = str(e)[:200]
                entry.next_retry_at = time.time() + entry.retry_interval

            logger.debug(
                f"Backpressure retry failed: '{entry.call_id}' — "
                f"{e} — next retry in {entry.retry_interval:.0f}s"
            )

    async def signal_capacity_available(self) -> None:
        """Signal that capacity has been restored (rate limit reset or model swap).

        Advances next_retry_at to be sooner, but never below MIN_RETRY_GAP_SECONDS
        after the last attempt.
        """
        async with self._lock:
            if not self._queue:
                return
            now = time.time()
            signaled = 0
            for entry in self._queue:
                if entry.result_future and entry.result_future.done():
                    continue
                earliest = entry.last_attempt_at + MIN_RETRY_GAP_SECONDS
                new_retry = max(now, earliest)
                if new_retry < entry.next_retry_at:
                    entry.next_retry_at = new_retry
                    signaled += 1
            if signaled:
                self._has_items.set()

        if signaled:
            logger.info(f"Backpressure: capacity signal advanced {signaled} calls")

    @property
    def depth(self) -> int:
        """Current queue depth. Approximate — no lock for performance."""
        return len(self._queue)

    def stop(self) -> None:
        self._running = False

    def get_status(self) -> dict:
        return {
            "queue_depth": len(self._queue),
            "active_retries": len(self._active_retries),
            "total_queued": self.total_queued,
            "total_retried": self.total_retried,
            "total_succeeded": self.total_succeeded,
            "total_expired": self.total_expired,
            "entries": [
                {
                    "call_id": e.call_id,
                    "priority": e.priority,
                    "attempts": e.attempt,
                    "waiting_seconds": round(e.wait_seconds, 1),
                    "last_error": e.last_error[:60],
                }
                for e in self._queue[:10]
            ],
        }


# ─── Singleton ───────────────────────────────────────────────
_queue: BackpressureQueue | None = None


def get_backpressure_queue() -> BackpressureQueue:
    global _queue
    if _queue is None:
        _queue = BackpressureQueue()
    return _queue
