# llm_dispatcher.py
"""
Centralized LLM Dispatcher — the ONLY entry point for all LLM calls.

All LLM calls in KutAI go through this dispatcher. It categorizes each
request and applies appropriate routing rules:

  MAIN_WORK  — Agent execution (ReAct iterations, single-shot, shopping,
               sub-agents). The actual task work the user cares about.
               CAN trigger model swaps.

  OVERHEAD   — Classifiers, graders, self-reflection, subtask classification.
               System housekeeping. CANNOT trigger model swaps — uses
               whatever is loaded or falls back to cloud.

Key responsibilities:
  - Swap protection: only MAIN_WORK can trigger local model swaps
  - Deferred grading: non-urgent grading queued until natural swap point
  - Quota awareness: checks cloud quota before routing overhead to cloud
  - Error propagation: all failures propagate with full context, never silent
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Optional

from src.infra.logging_config import get_logger

logger = get_logger("core.llm_dispatcher")


# ─── Call Categories ─────────────────────────────────────────────────────────

class CallCategory(Enum):
    """Two categories — simple and clear."""
    MAIN_WORK = "main_work"   # Agent execution, can trigger swaps
    OVERHEAD = "overhead"     # Classifier/grader/reflection, no swaps


# ─── Swap Budget ─────────────────────────────────────────────────────────────

class SwapBudget:
    """Track recent model swaps and throttle when excessive.

    Exemptions:
      - local_only tasks (no choice — must swap)
      - priority >= 9 (urgent, bypass budget)
    """

    def __init__(self, max_swaps: int = 3, window_seconds: float = 300.0):
        self.max_swaps = max_swaps
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def can_swap(self, local_only: bool = False, priority: int = 5) -> bool:
        """Check if a swap is allowed within the budget."""
        if local_only:
            return True   # no alternative
        if priority >= 9:
            return True   # urgent bypasses
        self._prune()
        return len(self._timestamps) < self.max_swaps

    def record_swap(self):
        """Record that a swap occurred."""
        self._timestamps.append(time.time())
        logger.info(
            "swap recorded",
            recent_swaps=len(self._timestamps),
            budget_remaining=max(0, self.max_swaps - len(self._timestamps)),
        )

    def _prune(self):
        cutoff = time.time() - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    @property
    def remaining(self) -> int:
        self._prune()
        return max(0, self.max_swaps - len(self._timestamps))

    @property
    def exhausted(self) -> bool:
        self._prune()
        return len(self._timestamps) >= self.max_swaps


# ─── Deferred Grade Queue ───────────────────────────────────────────────────

@dataclass
class PendingGrade:
    """A grading request deferred until a natural opportunity."""
    task_id: str
    task_title: str
    task_description: str
    response_text: str
    generating_model: str
    task_name: str
    priority: int
    queued_at: float = field(default_factory=time.time)
    # Callback to apply grade result back to the task
    on_graded: Optional[Callable[[float], Awaitable[None]]] = None


class GradeQueue:
    """Queue for deferred grading requests.

    Grading is deferred when:
      - The loaded model IS the generating model (can't self-grade)
      - Cloud quota is tight
      - Task is non-urgent (priority < 8)

    Queue drains when:
      - A model swap happens (before: drain what old model can grade;
        after: drain what new model can grade)
      - Cloud quota has headroom (batch-grade via cheapest model)
      - Queue exceeds max_pending threshold
      - No main work tasks remain (idle drain)
    """

    def __init__(self, max_pending: int = 20):
        self.max_pending = max_pending
        self._queue: list[PendingGrade] = []
        self._lock = asyncio.Lock()

    async def enqueue(self, grade: PendingGrade):
        """Add a grading request to the deferred queue."""
        async with self._lock:
            self._queue.append(grade)
            logger.info(
                "grade deferred",
                task_id=grade.task_id,
                queue_depth=len(self._queue),
                generating_model=grade.generating_model,
            )

    async def drain(
        self,
        available_model: str | None = None,
        use_cloud: bool = False,
    ) -> int:
        """Drain as many pending grades as possible.

        Args:
            available_model: Currently loaded local model name (litellm_name).
                             Grades whose generating_model != this can be graded.
            use_cloud: If True, use cloud for remaining grades.

        Returns:
            Number of grades completed.
        """
        async with self._lock:
            if not self._queue:
                return 0

            to_process = []
            remaining = []

            for grade in self._queue:
                can_grade_locally = (
                    available_model
                    and grade.generating_model != available_model
                )
                if can_grade_locally or use_cloud:
                    to_process.append(grade)
                else:
                    remaining.append(grade)

            self._queue = remaining

        if not to_process:
            return 0

        # Process grades outside the lock
        completed = 0
        for grade in to_process:
            try:
                score = await self._execute_grade(
                    grade, use_cloud=not (
                        available_model
                        and grade.generating_model != available_model
                    ),
                )
                if score is not None and grade.on_graded:
                    await grade.on_graded(score)
                completed += 1
            except Exception as e:
                logger.warning(
                    "deferred grade failed",
                    task_id=grade.task_id,
                    error=str(e),
                )
                # Re-queue failed grades
                async with self._lock:
                    self._queue.append(grade)

        if completed > 0:
            logger.info(
                "grade queue drained",
                completed=completed,
                remaining=len(self._queue),
            )
        return completed

    async def _execute_grade(
        self,
        grade: PendingGrade,
        use_cloud: bool = False,
    ) -> float | None:
        """Execute a single grade. Uses router's grade_response internally."""
        from src.core.router import grade_response
        return await grade_response(
            task_title=grade.task_title,
            task_description=grade.task_description,
            response_text=grade.response_text,
            generating_model=grade.generating_model,
            task_name=grade.task_name,
        )

    @property
    def depth(self) -> int:
        return len(self._queue)

    @property
    def needs_drain(self) -> bool:
        return len(self._queue) >= self.max_pending

    async def get_pending_models(self) -> set[str]:
        """Get set of generating models that have pending grades."""
        async with self._lock:
            return {g.generating_model for g in self._queue}


# ─── LLM Dispatcher ─────────────────────────────────────────────────────────

class LLMDispatcher:
    """Centralized LLM call coordinator.

    The ONLY component that should trigger model swaps, acquire GPU slots,
    and route between local and cloud.

    Usage:
        dispatcher = get_dispatcher()
        result = await dispatcher.request(
            category=CallCategory.MAIN_WORK,
            reqs=model_requirements,
            messages=messages,
            tools=tools,
        )
    """

    def __init__(self):
        self.swap_budget = SwapBudget(max_swaps=3, window_seconds=300.0)
        self.grade_queue = GradeQueue(max_pending=20)
        self._total_calls = 0
        self._overhead_calls = 0
        self._swaps_prevented = 0

    async def request(
        self,
        category: CallCategory,
        reqs: "ModelRequirements",
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Route an LLM call through the dispatcher.

        Args:
            category: MAIN_WORK or OVERHEAD
            reqs: ModelRequirements describing what the call needs
            messages: Chat messages
            tools: Optional tool definitions

        Returns:
            dict with response content, model info, cost, etc.

        Raises:
            RuntimeError: If all models fail (propagated from router)
        """
        self._total_calls += 1

        if category == CallCategory.OVERHEAD:
            self._overhead_calls += 1
            return await self._route_overhead(reqs, messages, tools)
        else:
            return await self._route_main_work(reqs, messages, tools)

    def _compute_timeout(
        self,
        category: CallCategory,
        reqs: "ModelRequirements",
    ) -> float:
        """Compute adaptive request timeout (seconds).

        S9 — Adaptive Timeouts:
          OVERHEAD hard cap  20s   (classifier/grader must be fast)
          MAIN_WORK          TPS-based: (estimated_output_tokens / measured_tps) × 2.0
                             Clamped to [20, 300] seconds.
                             Falls back to difficulty heuristics when no TPS measured.

        The returned value is passed as timeout_override to call_model(),
        which uses it for both the litellm HTTP timeout (override − 5s) and
        for outer asyncio cancellation should the HTTP layer not fire in time.
        """
        # ── OVERHEAD: hard 20s cap ────────────────────────────────────────
        if category == CallCategory.OVERHEAD:
            return 20.0

        # ── MAIN_WORK: TPS-based adaptive timeout ─────────────────────────
        _MAIN_WORK_MIN = 20.0
        _MAIN_WORK_MAX = 300.0

        # Try runtime measured_tps first (most accurate)
        try:
            from src.models.local_model_manager import get_runtime_state
            runtime = get_runtime_state()
            if runtime is not None and runtime.measured_tps > 0.0:
                est_gen_secs = reqs.estimated_output_tokens / runtime.measured_tps
                return max(_MAIN_WORK_MIN, min(_MAIN_WORK_MAX, est_gen_secs * 2.0))
        except Exception:
            pass

        # Fallback: difficulty heuristic (no TPS data yet)
        d = reqs.difficulty
        if d <= 2:
            return 25.0
        elif d <= 4:
            return 60.0
        elif d <= 6:
            return 120.0
        elif d <= 8:
            return 200.0
        else:
            return _MAIN_WORK_MAX

    async def _route_main_work(
        self,
        reqs: "ModelRequirements",
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict:
        """Route a MAIN_WORK call. Can trigger model swaps.

        Before swapping, checks the swap budget. If budget exhausted,
        tries to use the loaded model (even if suboptimal) or cloud.
        """
        from src.core.router import call_model

        timeout = self._compute_timeout(CallCategory.MAIN_WORK, reqs)

        # Check swap budget — if exhausted, constrain to loaded model or cloud
        if self.swap_budget.exhausted and not reqs.local_only:
            reqs_copy = copy.copy(reqs)
            loaded = self._get_loaded_model_name()
            if loaded:
                # Try the loaded model first by pinning it
                reqs_copy.model_override = loaded
                try:
                    result = await call_model(reqs_copy, messages, tools,
                                              timeout_override=timeout)
                    return result
                except Exception:
                    # Loaded model failed — fall through to normal routing
                    # which may try cloud
                    pass
            # If no model loaded or loaded model failed, let normal routing
            # handle it (will likely pick cloud since budget is exhausted
            # and we can't swap)

        result = await call_model(reqs, messages, tools, timeout_override=timeout)
        return result

    async def _route_overhead(
        self,
        reqs: "ModelRequirements",
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict:
        """Route an OVERHEAD call. CANNOT trigger model swaps.

        Strategy:
          1. If a local model is loaded, try it (free, no swap needed)
          2. If loaded model won't work, go to cloud
          3. Never trigger a model swap for overhead
        """
        from src.core.router import call_model, select_model, ModelRequirements

        timeout = self._compute_timeout(CallCategory.OVERHEAD, reqs)

        loaded_name = self._get_loaded_model_name()
        loaded_litellm = self._get_loaded_litellm_name()

        if loaded_litellm:
            # Try loaded model first — it's free and requires no swap
            reqs_pinned = copy.copy(reqs)
            reqs_pinned.model_override = loaded_litellm
            try:
                result = await call_model(reqs_pinned, messages, tools,
                                          timeout_override=timeout)
                return result
            except Exception as e:
                logger.debug(
                    "overhead: loaded model failed, trying cloud",
                    loaded_model=loaded_name,
                    error=str(e),
                )

        # Loaded model unavailable or failed — use cloud
        # Exclude all local models to prevent swap
        reqs_cloud = copy.copy(reqs)
        reqs_cloud = self._force_cloud_only(reqs_cloud)

        try:
            result = await call_model(reqs_cloud, messages, tools,
                                      timeout_override=timeout)
            return result
        except Exception as e:
            # Cloud also failed — this is a real error, propagate it
            raise RuntimeError(
                f"OVERHEAD call failed: no loaded model and cloud unavailable. "
                f"Task: {reqs.effective_task or reqs.primary_capability}, "
                f"Error: {e}"
            ) from e

    def _force_cloud_only(self, reqs: "ModelRequirements") -> "ModelRequirements":
        """Modify requirements to exclude all local models (prevent swaps)."""
        from src.models.model_registry import get_registry
        registry = get_registry()

        local_models = [
            m.litellm_name for m in registry.all_models()
            if m.is_local
        ]
        existing_excludes = list(reqs.exclude_models) if reqs.exclude_models else []
        reqs.exclude_models = list(set(existing_excludes + local_models))
        reqs.local_only = False  # override in case it was set
        return reqs

    def _get_loaded_model_name(self) -> str | None:
        """Get the currently loaded local model's name."""
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()
            return manager.current_model
        except Exception:
            return None

    def _get_loaded_litellm_name(self) -> str | None:
        """Get the currently loaded local model's litellm_name."""
        try:
            from src.models.local_model_manager import get_local_manager
            from src.models.model_registry import get_registry
            manager = get_local_manager()
            if not manager.current_model:
                return None
            registry = get_registry()
            info = registry.get(manager.current_model)
            return info.litellm_name if info else None
        except Exception:
            return None

    # ─── Deferred Grading ────────────────────────────────────────────────

    async def request_grade(
        self,
        task_id: str,
        task_title: str,
        task_description: str,
        response_text: str,
        generating_model: str,
        task_name: str = "",
        priority: int = 5,
        on_graded: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> float | None:
        """Request grading — may execute immediately or defer.

        Immediate grading when:
          - Priority >= 8 (urgent, need quality signal now)
          - Loaded model != generating model (can grade for free)

        Deferred grading when:
          - Loaded model IS the generating model
          - Priority < 8 (non-urgent)
        """
        from src.core.router import grade_response

        loaded_litellm = self._get_loaded_litellm_name()

        # Urgent tasks: grade immediately via whatever is available
        if priority >= 8:
            return await grade_response(
                task_title=task_title,
                task_description=task_description,
                response_text=response_text,
                generating_model=generating_model,
                task_name=task_name,
            )

        # Can the loaded model grade? (loaded != generator)
        can_grade_locally = (
            loaded_litellm
            and generating_model != loaded_litellm
        )

        if can_grade_locally:
            # Grade immediately using loaded model (free, no swap)
            return await grade_response(
                task_title=task_title,
                task_description=task_description,
                response_text=response_text,
                generating_model=generating_model,
                task_name=task_name,
            )

        # Defer grading — loaded model is the generator or nothing loaded
        await self.grade_queue.enqueue(PendingGrade(
            task_id=str(task_id),
            task_title=task_title,
            task_description=task_description,
            response_text=response_text,
            generating_model=generating_model,
            task_name=task_name,
            priority=priority,
            on_graded=on_graded,
        ))
        return None  # grade will be applied retroactively

    async def on_model_swap(self, old_model: str | None, new_model: str | None):
        """Called when a model swap occurs. Drains applicable deferred grades.

        Called by ensure_model() after a successful swap.
        """
        self.swap_budget.record_swap()

        # Before the old model is gone, drain grades it could handle
        # (any grade where generating_model != old_model)
        if old_model:
            drained = await self.grade_queue.drain(available_model=old_model)
            if drained:
                logger.info(
                    "drained grades before swap",
                    old_model=old_model,
                    drained=drained,
                )

        # After new model is ready, drain grades it can handle
        if new_model:
            drained = await self.grade_queue.drain(available_model=new_model)
            if drained:
                logger.info(
                    "drained grades after swap",
                    new_model=new_model,
                    drained=drained,
                )

    async def drain_grades_if_idle(self):
        """Called from main loop when no tasks are running.

        Uses cloud to drain any remaining deferred grades.
        """
        if self.grade_queue.depth > 0:
            drained = await self.grade_queue.drain(use_cloud=True)
            if drained:
                logger.info("drained grades during idle", drained=drained)

    async def drain_grades_if_full(self):
        """Called periodically. If queue exceeds threshold, force drain via cloud."""
        if self.grade_queue.needs_drain:
            logger.info(
                "grade queue full, forcing cloud drain",
                depth=self.grade_queue.depth,
            )
            await self.grade_queue.drain(use_cloud=True)

    # ─── Proactive GPU Loading ───────────────────────────────────────────

    async def ensure_gpu_utilized(self, upcoming_tasks: list[dict]):
        """Proactively load a local model if GPU is idle and queue has work.

        Called from the orchestrator main loop. If no model is loaded and
        there are tasks in the queue that ANY local model can handle,
        load the best-fit model. Local inference is free — don't waste the GPU.

        Args:
            upcoming_tasks: List of task dicts from get_ready_tasks()
        """
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()

            if manager.current_model:
                return  # already loaded, nothing to do

            if not upcoming_tasks:
                return  # no work, stay idle (save power)

            best_model = self._find_best_local_for_batch(upcoming_tasks)
            if best_model:
                logger.info(
                    "proactive GPU load",
                    model=best_model,
                    queue_depth=len(upcoming_tasks),
                )
                await manager.ensure_model(
                    best_model,
                    reason="proactive_load",
                )
        except Exception as e:
            logger.debug(f"Proactive GPU load failed: {e}")

    def _find_best_local_for_batch(self, tasks: list[dict]) -> str | None:
        """Find the local model that can serve the most upcoming tasks.

        Considers agent_type, difficulty, capabilities needed. Returns the
        model name (not litellm_name) or None if no local model is suitable.
        """
        try:
            import json as _json
            from src.models.model_registry import get_registry
            from src.models.capabilities import (
                score_model_for_task, TASK_PROFILES,
                TaskRequirements as CapTaskReqs,
            )
            from src.core.router import CAPABILITY_TO_TASK, AGENT_REQUIREMENTS

            registry = get_registry()
            local_models = [m for m in registry.all_models() if m.is_local]

            if not local_models:
                return None

            # Score each local model by how many tasks it can handle
            model_scores: dict[str, int] = {}
            for model in local_models:
                if model.demoted:
                    continue
                match_count = 0
                for task in tasks:
                    ctx = task.get("context", {})
                    if isinstance(ctx, str):
                        try:
                            ctx = _json.loads(ctx)
                        except (Exception,):
                            ctx = {}
                    cls = ctx.get("classification", {})
                    agent_type = task.get(
                        "agent_type", cls.get("agent_type", "executor"),
                    )
                    difficulty = max(1, min(10, int(cls.get("difficulty", 5))))

                    # Resolve task key for profile lookup
                    task_key = agent_type
                    if task_key in CAPABILITY_TO_TASK:
                        task_key = CAPABILITY_TO_TASK[task_key]
                    template = AGENT_REQUIREMENTS.get(agent_type)
                    if template:
                        task_key = template.task or task_key

                    if task_key not in TASK_PROFILES:
                        continue

                    # Build minimal requirements for scoring
                    cap_reqs = CapTaskReqs(
                        task_name=task_key,
                        needs_function_calling=cls.get("needs_tools", False),
                        needs_vision=cls.get("needs_vision", False),
                        needs_thinking=cls.get("needs_thinking", False),
                    )

                    # Check if model meets minimum capability
                    min_score = max(0.0, (difficulty - 1) * 0.47)
                    cap_score = score_model_for_task(
                        model.capabilities,
                        model.operational_dict(),
                        cap_reqs,
                    )
                    if cap_score >= min_score:
                        match_count += 1

                if match_count > 0:
                    model_scores[model.name] = match_count

            if not model_scores:
                return None

            # Pick model that serves the most tasks
            best = max(model_scores, key=model_scores.get)
            logger.debug(
                "proactive load candidates",
                scores=model_scores,
                selected=best,
            )
            return best

        except Exception as e:
            logger.debug(f"_find_best_local_for_batch failed: {e}")
            return None

    # ─── Metrics ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return dispatcher statistics for monitoring."""
        return {
            "total_calls": self._total_calls,
            "overhead_calls": self._overhead_calls,
            "overhead_pct": (
                f"{self._overhead_calls / self._total_calls * 100:.1f}%"
                if self._total_calls > 0 else "0%"
            ),
            "swaps_prevented": self._swaps_prevented,
            "swap_budget_remaining": self.swap_budget.remaining,
            "grade_queue_depth": self.grade_queue.depth,
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_dispatcher: LLMDispatcher | None = None


def get_dispatcher() -> LLMDispatcher:
    """Get or create the global LLM dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = LLMDispatcher()
    return _dispatcher
