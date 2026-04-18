# requirements.py
"""
ModelRequirements, AGENT_REQUIREMENTS, CAPABILITY_TO_TASK, and QuotaPlanner.

Extracted from src/core/router.py and src/models/quota_planner.py.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

from fatih_hoca.capabilities import TASK_PROFILES, ALL_CAPABILITIES, Cap

logger = logging.getLogger("fatih_hoca.requirements")

# How long a 429 event stays relevant for threshold calculation
_429_DECAY_SECONDS = 600  # 10 minutes


# ─── Capability ↔ Task Mapping ───────────────────────────────────────────────

CAPABILITY_TO_TASK: dict[str, str] = {
    # Map primary_capability values to TASK_PROFILES keys
    "reasoning":             "planner",
    "planning":              "planner",
    "analysis":              "analyst",
    "code_generation":       "coder",
    "code_reasoning":        "fixer",
    "system_design":         "architect",
    "prose_quality":         "writer",
    "instruction_adherence": "executor",
    "domain_knowledge":      "researcher",
    "context_utilization":   "summarizer",
    "structured_output":     "executor",
    "tool_use":              "executor",
    "vision":                "visual_reviewer",
    "conversation":          "assistant",
    "general":               "assistant",
    "shopping":              "shopping_advisor",
}


def _make_adhoc_profile(primary_cap: str) -> dict[str, float]:
    """Create a task profile on the fly for unknown capability requests."""
    profile = {cap: 0.3 for cap in ALL_CAPABILITIES}
    for c in Cap:
        if c.value == primary_cap or primary_cap in c.value:
            profile[c.value] = 1.0
            return profile
    profile[Cap.REASONING.value] = 0.8
    profile[Cap.INSTRUCTION_ADHERENCE.value] = 0.8
    return profile


# ─── Model Requirements ─────────────────────────────────────────────────────

@dataclass
class ModelRequirements:
    """
    Structured description of what a task needs from a model.

    Uses difficulty (1-10) to express how capable the model must be.
    """
    # ── Task identity (preferred path) ──
    task: str = ""                            # Key into TASK_PROFILES

    # ── Capability path (auto-maps to task) ──
    primary_capability: str = "general"
    secondary_capabilities: list[str] = field(default_factory=list)

    # ── Difficulty (1-10) — drives model quality selection ──
    difficulty: int = 5
    min_score: float = 0.0                     # Override; if 0, computed from difficulty

    # ── Context requirements ──
    estimated_input_tokens: int = 2000
    estimated_output_tokens: int = 1000
    min_context_length: int = 0

    # ── Feature requirements ──
    needs_function_calling: bool = False
    needs_json_mode: bool = False
    needs_thinking: bool = False
    needs_vision: bool = False

    # ── Constraints ──
    local_only: bool = False
    prefer_speed: bool = False
    prefer_quality: bool = False
    prefer_local: bool = False
    max_cost: float = 0.0

    # ── Priority ──
    priority: int = 5

    # ── Exclusion ──
    exclude_models: list[str] = field(default_factory=list)

    # ── Call category ──
    # "main_work" (default) or "overhead" (classifier, grader, reflection)
    call_category: str = "main_work"

    # ── Agent context ──
    agent_type: str = ""

    @property
    def effective_task(self) -> str:
        if self.task and self.task in TASK_PROFILES:
            return self.task
        mapped = CAPABILITY_TO_TASK.get(self.primary_capability)
        if mapped and mapped in TASK_PROFILES:
            return mapped
        return ""

    @property
    def task_profile(self) -> dict[str, float]:
        task = self.effective_task
        if task:
            return TASK_PROFILES[task]
        return _make_adhoc_profile(self.primary_capability)

    @property
    def effective_context_needed(self) -> int:
        if self.min_context_length > 0:
            return self.min_context_length
        # 1.3× covers tokenizer variance; extra 512 tokens headroom for
        # tool results that grow the context during multi-iteration execution.
        # Without this, a model that barely fits the initial prompt gets
        # filtered after one tool call (e.g. ctx 8192 < 8217 after read_file).
        return int((self.estimated_input_tokens + self.estimated_output_tokens) * 1.3) + 512

    @property
    def effective_min_score(self) -> float:
        if self.min_score > 0:
            return self.min_score
        # Gentle curve: difficulty 1→0.3, 5→1.5, 7→2.5, 10→4.5
        # This ensures available models (groq-llama-70b scores ~3.3) aren't
        # filtered out for reasonable tasks. Only difficulty 10 needs top-tier.
        return max(0.0, (self.difficulty - 1) * 0.47)

    def escalate(self) -> "ModelRequirements":
        """
        Return a copy with escalated quality requirements.
        Used by base.py for mid-task escalation.
        """
        escalated = copy.copy(self)
        escalated.difficulty = min(10, self.difficulty + 2)
        escalated.min_score = 0.0  # reset so it recomputes from new difficulty
        escalated.prefer_quality = True
        return escalated


# ─── Agent Requirement Templates ─────────────────────────────────────────────

AGENT_REQUIREMENTS: dict[str, ModelRequirements] = {
    # ── Difficult / sensitive → cloud (better quality, rate-limited) ──
    "planner":        ModelRequirements(task="planner",        difficulty=5, estimated_output_tokens=2000, prefer_quality=True),
    "architect":      ModelRequirements(task="architect",      difficulty=5, estimated_output_tokens=3000, prefer_quality=True),
    "coder":          ModelRequirements(task="coder",          difficulty=6, estimated_output_tokens=4000, needs_function_calling=True),
    "fixer":          ModelRequirements(task="fixer",          difficulty=6, estimated_output_tokens=3000, needs_function_calling=True),
    "reviewer":       ModelRequirements(task="reviewer",       difficulty=6, estimated_output_tokens=2000),
    "analyst":        ModelRequirements(task="analyst",        difficulty=6, estimated_output_tokens=3000, needs_function_calling=True),
    # ── Moderate → let the scorer decide (local if free, cloud if rate OK) ──
    "implementer":    ModelRequirements(task="implementer",    difficulty=5, estimated_output_tokens=4000, needs_function_calling=True),
    "test_generator": ModelRequirements(task="test_generator", difficulty=5, estimated_output_tokens=3000, needs_function_calling=True),
    "writer":         ModelRequirements(task="writer",         difficulty=5, estimated_output_tokens=3000),
    "visual_reviewer": ModelRequirements(task="visual_reviewer", difficulty=5, estimated_output_tokens=2000, needs_vision=True),
    # ── Token-heavy / conversational → prefer local (save cloud rate limits) ──
    "researcher":     ModelRequirements(task="researcher",     difficulty=4, estimated_output_tokens=2000, needs_function_calling=True, prefer_local=True, prefer_speed=True),
    "assistant":      ModelRequirements(task="assistant",      difficulty=3, estimated_output_tokens=2000, prefer_local=True, prefer_speed=True),
    "executor":       ModelRequirements(task="executor",       difficulty=3, estimated_output_tokens=1000, needs_function_calling=True, prefer_speed=True, prefer_local=True),
    "summarizer":     ModelRequirements(task="summarizer",     difficulty=4, estimated_output_tokens=2000, prefer_speed=True, prefer_local=True),
    # ── Shopping agents → prefer local, need tool calling ──
    "shopping_advisor":    ModelRequirements(task="shopping_advisor",    difficulty=5, estimated_output_tokens=2500, needs_function_calling=True, prefer_local=True, prefer_speed=True),
    "product_researcher":  ModelRequirements(task="shopping_advisor",    difficulty=4, estimated_output_tokens=2000, needs_function_calling=True, prefer_local=True, prefer_speed=True),
    "deal_analyst":        ModelRequirements(task="shopping_advisor",    difficulty=5, estimated_output_tokens=2000, needs_function_calling=True, prefer_local=True),
    "shopping_clarifier":  ModelRequirements(task="shopping_advisor",    difficulty=3, estimated_output_tokens=1000, prefer_local=True, prefer_speed=True),
}


# ─── Queue Profile ────────────────────────────────────────────────────────────

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


# ─── Quota Planner ────────────────────────────────────────────────────────────

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

    def set_queue_profile(self, profile: QueueProfile) -> None:
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

        # Wake sleeping tasks — quota restored means cloud capacity available
        try:
            from src.infra.db import schedule_accelerate_retries
            schedule_accelerate_retries("quota_restored")
        except Exception:
            pass

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
