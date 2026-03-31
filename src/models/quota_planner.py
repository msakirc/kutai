# quota_planner.py
"""
Quota Planner — dynamically adjusts when expensive (paid) models are used.

Decides the minimum difficulty threshold for paid model selection based on:
- Current quota utilization (from response headers)
- Upcoming task difficulty in the queue
- Time until quota resets
- Recent 429 frequency

Never blocks work — just adjusts scoring weights so free models are preferred
when expensive capacity should be reserved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.infra.logging_config import get_logger

logger = get_logger("models.quota_planner")

# How long a 429 event stays relevant for threshold calculation
_429_DECAY_SECONDS = 600  # 10 minutes


@dataclass
class QueueProfile:
    """Summary of the upcoming task queue's capability needs.

    Built by the orchestrator's forward-looking scan and fed to the QuotaPlanner
    so it can reserve cloud quota for tasks that genuinely need it (e.g. vision
    tasks that only cloud models can handle, or thinking-heavy tasks needing CoT).
    """
    total_tasks: int = 0
    max_difficulty: int = 0
    needs_vision_count: int = 0      # tasks requiring vision capability
    needs_tools_count: int = 0       # tasks requiring function calling
    needs_thinking_count: int = 0    # tasks explicitly requesting CoT
    hard_tasks_count: int = 0        # tasks with difficulty >= 7
    cloud_only_count: int = 0        # tasks that MUST use cloud (vision, etc.)


class QuotaPlanner:
    """
    Manages the dynamic difficulty threshold for expensive model usage.

    The threshold is an integer 1-10. Tasks with difficulty >= threshold
    get full access to paid models. Tasks below it see paid models
    penalized in scoring (but not blocked).
    """

    def __init__(self):
        self._expensive_threshold: int = 8  # conservative default
        self._paid_utilization: dict[str, float] = {}  # provider → 0-100
        self._paid_reset_in: dict[str, float] = {}  # provider → seconds until reset
        self._max_upcoming_difficulty: int = 0
        self._queue_profile: QueueProfile = QueueProfile()
        self._429_timestamps: list[tuple[str, float]] = []  # (provider, timestamp)
        self._last_recalc: float = 0.0

    @property
    def expensive_threshold(self) -> int:
        return self._expensive_threshold

    def update_paid_utilization(
        self,
        provider: str,
        utilization_pct: float,
        reset_in: float,
    ) -> None:
        """Update current utilization for a paid provider."""
        self._paid_utilization[provider] = utilization_pct
        self._paid_reset_in[provider] = reset_in

    def set_max_upcoming_difficulty(self, difficulty: int) -> None:
        """Set the max difficulty among upcoming queued tasks."""
        self._max_upcoming_difficulty = difficulty

    def set_queue_profile(self, profile: "QueueProfile") -> None:
        """Provide full capability analysis of the upcoming task queue.

        This is the richer version of set_max_upcoming_difficulty() — it also
        captures how many tasks need vision, tool use, thinking, and the
        distribution of difficulty tiers so the planner can better anticipate
        which models will be needed.
        """
        self._max_upcoming_difficulty = profile.max_difficulty
        self._queue_profile = profile

    def record_429(self, provider: str) -> None:
        """Record a rate limit hit on a paid provider."""
        self._429_timestamps.append((provider, time.time()))

    def on_quota_restored(
        self,
        provider: str,
        new_remaining_pct: float,
    ) -> None:
        """Called when headers show quota has been restored."""
        self._paid_utilization[provider] = 100.0 - new_remaining_pct
        logger.info(
            f"Quota restored for {provider} — "
            f"utilization now {100.0 - new_remaining_pct:.0f}%"
        )
        self.recalculate()

        # Signal backpressure queue to retry waiting calls
        try:
            import asyncio
            from ..infra.backpressure import get_backpressure_queue
            bp = get_backpressure_queue()
            if bp._queue:
                asyncio.ensure_future(bp.signal_capacity_available())
        except Exception:
            pass  # Queue may not be initialized yet

    def _recent_429_rate(self) -> int:
        """Count of 429s in the last decay window."""
        cutoff = time.time() - _429_DECAY_SECONDS
        self._429_timestamps = [
            (p, t) for p, t in self._429_timestamps if t > cutoff
        ]
        return len(self._429_timestamps)

    def recalculate(self) -> int:
        """
        Recalculate the expensive model difficulty threshold.

        Returns the new threshold value (1-10).
        """
        now = time.time()
        self._last_recalc = now

        # 1. Overall paid utilization (worst-case across providers)
        if self._paid_utilization:
            paid_util = max(self._paid_utilization.values())
        else:
            paid_util = 50.0  # unknown → moderate assumption

        # 2. Upcoming task difficulty
        max_diff = self._max_upcoming_difficulty

        # 3. Time until reset (minimum across providers)
        if self._paid_reset_in:
            min_reset = min(self._paid_reset_in.values())
        else:
            min_reset = 3600  # unknown → assume 1 hour

        # 4. Recent 429 rate
        recent_429s = self._recent_429_rate()

        # ── Decision logic ──

        if paid_util < 30 and recent_429s == 0:
            threshold = 3
        elif paid_util < 50 and recent_429s <= 1:
            threshold = 5
        elif paid_util < 70:
            threshold = 6
        elif paid_util < 85:
            threshold = 7
        else:
            threshold = 9

        # 429 penalty: each recent 429 pushes threshold up
        if recent_429s >= 3:
            threshold = max(threshold, 8)
        elif recent_429s >= 1:
            threshold = max(threshold, 7)

        # Reserve capacity for hard upcoming tasks
        if max_diff >= 8:
            threshold = max(threshold, max_diff - 1)

        # Cloud-only tasks (vision, etc.) need cloud quota reserved — if many
        # are pending, tighten threshold so overhead doesn't consume it.
        qp = self._queue_profile
        if qp.cloud_only_count >= 3:
            threshold = max(threshold, 6)
        elif qp.cloud_only_count >= 1 and paid_util > 50:
            threshold = max(threshold, 5)

        # Thinking-heavy queues: thinking models are often paid; reserve.
        if qp.needs_thinking_count >= 2 and paid_util > 40:
            threshold = max(threshold, 6)

        # Quota reset imminent (<5 min) — be more generous
        if min_reset < 300 and paid_util > 40:
            threshold = max(1, threshold - 2)

        threshold = max(1, min(10, threshold))

        if threshold != self._expensive_threshold:
            logger.info(
                f"Quota planner: threshold {self._expensive_threshold}→{threshold} "
                f"(util={paid_util:.0f}%, upcoming_max={max_diff}, "
                f"429s={recent_429s}, reset_in={min_reset:.0f}s)"
            )

        self._expensive_threshold = threshold
        return threshold

    def get_status(self) -> dict:
        """Status for diagnostics."""
        qp = self._queue_profile
        return {
            "expensive_threshold": self._expensive_threshold,
            "paid_utilization": dict(self._paid_utilization),
            "max_upcoming_difficulty": self._max_upcoming_difficulty,
            "recent_429s": self._recent_429_rate(),
            "queue_profile": {
                "total_tasks": qp.total_tasks,
                "cloud_only": qp.cloud_only_count,
                "needs_vision": qp.needs_vision_count,
                "needs_thinking": qp.needs_thinking_count,
                "hard_tasks": qp.hard_tasks_count,
            },
        }


# ─── Singleton ───────────────────────────────────────────────
_planner: QuotaPlanner | None = None


def get_quota_planner() -> QuotaPlanner:
    global _planner
    if _planner is None:
        _planner = QuotaPlanner()
    return _planner
