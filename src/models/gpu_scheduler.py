# gpu_scheduler.py
"""
Priority-aware GPU access scheduler.

Since we have a single GPU that can only do one inference at a time,
this module ensures:
  - High-priority tasks (user waiting) get GPU access before low-priority ones
  - Background tasks yield to critical requests
  - Fairness: same-priority tasks are FIFO
  - Timeout: callers don't wait forever
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("models.gpu_scheduler")


@dataclass(order=True)
class GPURequest:
    """A request for GPU inference time."""
    # Negative priority so higher priority = sorted first in min-heap
    sort_key: tuple = field(compare=True, repr=False)
    priority: int = field(compare=False, default=5)
    task_id: str = field(compare=False, default="?")
    agent_type: str = field(compare=False, default="")
    model_needed: str = field(compare=False, default="")
    event: asyncio.Event = field(
        compare=False, default_factory=asyncio.Event,
    )
    granted: bool = field(compare=False, default=False)
    enqueued_at: float = field(compare=False, default=0.0)
    cancelled: bool = field(compare=False, default=False)

    @staticmethod
    def make(
        priority: int,
        task_id: str = "?",
        agent_type: str = "",
        model_needed: str = "",
    ) -> GPURequest:
        """Create a request with proper sort key."""
        return GPURequest(
            sort_key=(-priority, time.time()),  # higher priority first, then FIFO
            priority=priority,
            task_id=task_id,
            agent_type=agent_type,
            model_needed=model_needed,
            enqueued_at=time.time(),
        )


class GPUScheduler:
    """
    Priority queue for GPU access.

    Usage:
        request = GPURequest.make(priority=10, task_id="42")
        granted = await scheduler.acquire(request, timeout=60)
        if granted:
            try:
                # do inference
            finally:
                scheduler.release()
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._queue: list[GPURequest] = []  # sorted by priority
        self._current: Optional[GPURequest] = None
        self._gpu_free = asyncio.Event()
        self._gpu_free.set()  # starts as free

        # Stats
        self.total_requests: int = 0
        self.total_preemptions: int = 0
        self.total_timeouts: int = 0

    async def acquire(
        self,
        request: GPURequest,
        timeout: float = 120,
    ) -> bool:
        """
        Request GPU access. Blocks until granted or timeout.

        Returns True if access granted, False if timed out.
        Higher priority requests are served first.
        """
        self.total_requests += 1

        async with self._lock:
            # If GPU is free and no queue, grant immediately
            if self._current is None and not self._queue:
                self._current = request
                request.granted = True
                request.event.set()
                self._gpu_free.clear()
                logger.debug(
                    f"GPU granted immediately to task #{request.task_id} "
                    f"(priority={request.priority})"
                )
                return True

            # Check if we should preempt notification
            # (We can't interrupt current inference, but we log it)
            if self._current and request.priority > self._current.priority + 3:
                logger.info(
                    f"⚡ High-priority GPU request from task "
                    f"#{request.task_id} (p={request.priority}) "
                    f"— current task #{self._current.task_id} "
                    f"(p={self._current.priority}) will finish first"
                )
                self.total_preemptions += 1

            # Add to priority queue
            self._queue.append(request)
            self._queue.sort()  # sort by sort_key (priority DESC, time ASC)

        # Wait for our turn
        try:
            await asyncio.wait_for(request.event.wait(), timeout=timeout)
            return request.granted
        except asyncio.TimeoutError:
            self.total_timeouts += 1
            # Remove from queue
            async with self._lock:
                request.cancelled = True
                self._queue = [r for r in self._queue if r is not request]
            logger.warning(
                f"GPU request timeout for task #{request.task_id} "
                f"after {timeout:.0f}s (priority={request.priority}, "
                f"queue_depth={len(self._queue)})"
            )
            return False

    def release(self) -> None:
        """
        Release GPU access. Grants to next highest-priority waiter.

        MUST be called after inference completes (use try/finally).
        """
        # We need to schedule the grant in the event loop since
        # release() might be called from a non-async context
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                asyncio.ensure_future, self._do_release()
            )
        except RuntimeError:
            # No running loop — likely during shutdown
            pass

    async def _do_release(self) -> None:
        """Internal: process release and grant next waiter."""
        async with self._lock:
            old = self._current
            self._current = None

            # Clean cancelled requests
            self._queue = [r for r in self._queue if not r.cancelled]

            if self._queue:
                # Grant to highest priority waiter
                next_req = self._queue.pop(0)
                self._current = next_req
                next_req.granted = True
                next_req.event.set()

                wait_time = time.time() - next_req.enqueued_at
                logger.info(
                    f"GPU granted to task #{next_req.task_id} "
                    f"(priority={next_req.priority}, "
                    f"waited={wait_time:.1f}s, "
                    f"remaining_queue={len(self._queue)})"
                )
            else:
                self._gpu_free.set()

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def is_busy(self) -> bool:
        return self._current is not None

    @property
    def current_task_priority(self) -> int:
        return self._current.priority if self._current else 0

    def get_status(self) -> dict:
        return {
            "is_busy": self.is_busy,
            "queue_depth": self.queue_depth,
            "current_task": self._current.task_id if self._current else None,
            "current_priority": self.current_task_priority,
            "queued_tasks": [
                {
                    "task_id": r.task_id,
                    "priority": r.priority,
                    "waiting_seconds": round(time.time() - r.enqueued_at, 1),
                }
                for r in self._queue[:5]
            ],
            "total_requests": self.total_requests,
            "total_preemptions": self.total_preemptions,
            "total_timeouts": self.total_timeouts,
        }


# ─── Singleton ───────────────────────────────────────────────
_scheduler: GPUScheduler | None = None


def get_gpu_scheduler() -> GPUScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = GPUScheduler()
    return _scheduler
