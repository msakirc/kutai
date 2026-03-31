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
from typing import Any

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
    attempt: int = 0
    enqueued_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0
    last_error: str = ""
    result_future: asyncio.Future = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.result_future is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            self.result_future = loop.create_future()

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
        call_func,       # async callable that retries the model call
        *call_args,
        **call_kwargs,
    ) -> Any:
        """
        Enqueue a failed model call for retry.

        Returns the eventual result (blocks until success or expiry).
        Raises RuntimeError if the call expires without success.

        Args:
            call_id: identifier for logging
            priority: task priority (higher = retry sooner)
            last_error: the error that caused the initial failure
            call_func: the async function to retry
            *call_args, **call_kwargs: arguments to pass to call_func
        """
        entry = QueuedCall(
            call_id=call_id,
            priority=priority,
            last_error=last_error,
            next_retry_at=time.time() + RETRY_INTERVALS[0],
        )

        # Store the retry callable on the entry
        entry._call_func = call_func
        entry._call_args = call_args
        entry._call_kwargs = call_kwargs

        # Single lock acquisition: depth check + append are atomic to
        # prevent exceeding MAX_QUEUE_DEPTH under concurrent enqueues.
        async with self._lock:
            if len(self._queue) >= MAX_QUEUE_DEPTH:
                raise RuntimeError(
                    f"Backpressure queue full ({MAX_QUEUE_DEPTH} items). "
                    f"System is overloaded. Last error: {last_error}"
                )
            self._queue.append(entry)
            # Sort: higher priority first, then earlier enqueue time
            self._queue.sort(
                key=lambda e: (-e.priority, e.enqueued_at)
            )
            self.total_queued += 1

        self._has_items.set()

        logger.info(
            f"Backpressure: queued call '{call_id}' "
            f"(priority={priority}, queue_depth={len(self._queue)}, "
            f"error: {last_error[:80]})"
        )

        # Wait for the result
        try:
            result = await asyncio.wait_for(
                entry.result_future,
                timeout=MAX_QUEUE_WAIT_SECONDS,
            )
            return result
        except asyncio.TimeoutError:
            # Remove from queue
            async with self._lock:
                self._queue = [e for e in self._queue if e is not entry]
                self.total_expired += 1
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
                    if not entry.result_future.done():
                        entry.result_future.set_exception(
                            RuntimeError(
                                f"Backpressure expired: {entry.last_error}"
                            )
                        )
                    self.total_expired += 1
                self._queue = [
                    e for e in self._queue if not e.is_expired
                ]

                # Find entries ready for retry
                ready = [
                    e for e in self._queue
                    if now >= e.next_retry_at
                    and not e.result_future.done()
                ]

                if not self._queue:
                    self._has_items.clear()

            # Process ready entries (outside lock)
            for entry in ready[:3]:  # max 3 concurrent retries
                asyncio.create_task(self._retry_call(entry))

            # Sleep between processor cycles — long enough to let retries
            # complete and backoff timers advance, short enough to be responsive.
            await asyncio.sleep(5)

    async def _retry_call(self, entry: QueuedCall) -> None:
        """Attempt to retry a queued call."""
        entry.attempt += 1
        entry._last_attempt_at = time.time()
        self.total_retried += 1

        # Give up after max attempts — stop the 43-retry spirals
        if entry.attempt > MAX_RETRY_ATTEMPTS:
            logger.warning(
                f"Backpressure: giving up on '{entry.call_id}' "
                f"after {entry.attempt} attempts "
                f"({entry.wait_seconds:.1f}s). "
                f"Last error: {entry.last_error[:80]}"
            )
            if not entry.result_future.done():
                entry.result_future.set_exception(
                    RuntimeError(
                        f"Backpressure exhausted after {entry.attempt} retries "
                        f"for '{entry.call_id}'. Last error: {entry.last_error}"
                    )
                )
            async with self._lock:
                self._queue = [e for e in self._queue if e is not entry]
                self.total_expired += 1
            return

        logger.info(
            f"Backpressure retry: '{entry.call_id}' "
            f"(attempt={entry.attempt}/{MAX_RETRY_ATTEMPTS}, "
            f"waited={entry.wait_seconds:.1f}s)"
        )

        try:
            result = await entry._call_func(
                *entry._call_args,
                **entry._call_kwargs,
            )

            # Success
            if not entry.result_future.done():
                entry.result_future.set_result(result)
            self.total_succeeded += 1

            async with self._lock:
                self._queue = [e for e in self._queue if e is not entry]

            logger.info(
                f"Backpressure success: '{entry.call_id}' "
                f"after {entry.attempt} retries "
                f"({entry.wait_seconds:.1f}s total wait)"
            )

        except Exception as e:
            entry.last_error = str(e)[:200]
            # Schedule next retry with backoff
            entry.next_retry_at = time.time() + entry.retry_interval

            logger.debug(
                f"Backpressure retry failed: '{entry.call_id}' — "
                f"{e} — next retry in {entry.retry_interval:.0f}s"
            )

    async def signal_capacity_available(self) -> None:
        """Signal that capacity has been restored (rate limit reset or model swap).

        Advances next_retry_at to be sooner, but never below MIN_RETRY_GAP_SECONDS
        after the last attempt. This prevents the spam loop where capacity signals
        cause immediate retries that fail instantly and re-enter the queue.
        """
        async with self._lock:
            if not self._queue:
                return
            now = time.time()
            signaled = 0
            for entry in self._queue:
                if entry.result_future.done():
                    continue
                # Respect minimum gap since last attempt
                last_attempt = getattr(entry, '_last_attempt_at', 0.0)
                earliest = last_attempt + MIN_RETRY_GAP_SECONDS
                new_retry = max(now, earliest)
                if new_retry < entry.next_retry_at:
                    entry.next_retry_at = new_retry
                    signaled += 1
        if signaled:
            self._has_items.set()
            logger.info(
                f"Backpressure: capacity signal — "
                f"{signaled}/{len(self._queue)} calls advanced"
            )

    def stop(self) -> None:
        self._running = False

    def get_status(self) -> dict:
        return {
            "queue_depth": len(self._queue),
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
