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

import logging

logger = logging.getLogger(__name__)

# Cold-start wait parameters (patchable in tests)
_COLD_START_WAIT_TIMEOUT = 15.0   # max seconds to wait for model load
_COLD_START_POLL_INTERVAL = 0.5   # poll interval while waiting


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
        logger.info(f"swap recorded | recent_swaps={len(self._timestamps)} budget_remaining={max(0, self.max_swaps - len(self._timestamps))}")

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
            logger.info(f"grade deferred | task_id={grade.task_id} queue_depth={len(self._queue)} generating_model={grade.generating_model}")

    async def drain(
        self,
        available_model: str | None = None,
        use_cloud: bool = False,
        max_batch: int = 3,
    ) -> int:
        """Drain pending grades in small batches.

        Processes at most `max_batch` grades per call so the orchestrator's
        main loop can interleave main work between drain cycles.  Remaining
        eligible grades stay in the queue for the next drain call.

        Args:
            available_model: Currently loaded local model name (litellm_name).
                             Grades whose generating_model != this can be graded.
            use_cloud: If True, use cloud for remaining grades.
            max_batch: Maximum grades to process in this call (default 3).

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
                if (can_grade_locally or use_cloud) and len(to_process) < max_batch:
                    to_process.append(grade)
                else:
                    remaining.append(grade)

            self._queue = remaining

        if not to_process:
            return 0

        # Process grades outside the lock — one at a time so GPU scheduler
        # can interleave higher-priority requests between grades.
        completed = 0
        for grade in to_process:
            try:
                score = await self._execute_grade(grade)
                if score is not None and grade.on_graded:
                    await grade.on_graded(score)
                completed += 1
            except Exception as e:
                logger.warning(f"deferred grade failed | task_id={grade.task_id} error={e}")
                # Re-queue failed grades
                async with self._lock:
                    self._queue.append(grade)

        if completed > 0:
            logger.info(f"grade queue drained | completed={completed} remaining={len(self._queue)}")
        return completed

    async def _execute_grade(
        self,
        grade: PendingGrade,
    ) -> float | None:
        """Execute a single grade via dispatcher's OVERHEAD routing."""
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
            loaded = self._get_loaded_litellm_name()
            if loaded:
                # Skip pinning if task prefers speed but loaded model is too
                # slow (e.g. web_search agents need >10 tok/s for Perplexica).
                # Let normal routing pick a better model (may trigger a swap
                # via exemption or fall back to cloud).
                loaded_speed = self.get_loaded_model_speed()
                if reqs.prefer_speed and loaded_speed > 0 and loaded_speed < 10.0:
                    logger.info(f"skip slow-model pin for speed-critical task | loaded_speed={loaded_speed} task={reqs.effective_task or reqs.primary_capability}")
                    # Fall through to normal routing below
                else:
                    # Try the loaded model first by pinning it
                    reqs_copy.model_override = loaded
                    try:
                        result = await call_model(reqs_copy, messages, tools,
                                                  timeout_override=timeout)
                        return result
                    except Exception as e:
                        # Loaded model failed — fall through to normal routing
                        # which may try cloud
                        logger.debug(
                            "main_work: pinned loaded model failed, "
                            "falling through to full routing",
                            loaded_model=loaded,
                            error=str(e),
                            task=reqs.effective_task or reqs.primary_capability,
                        )
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
          Exclude unloaded local models so select_model() can only pick:
            - The currently loaded model (free, no swap)
            - Cloud models (fallback)

          call_model() handles the rest naturally:
            1. Tries loaded model first (ranked highest: free + loaded)
            2. GPU scheduler queues request — up to 60s waiting room
            3. If GPU times out, falls through to cloud candidates
            4. If cloud also fails, backpressure retries (swap-aware)

          Swap-awareness:
            When a model swap is in progress, local model is unavailable.
            OVERHEAD calls skip local entirely and go cloud-only to avoid
            piling up in the GPU scheduler queue (cascade failure).

          Cold-start awareness:
            When no model is loaded and no cloud is available, but a
            proactive load is in progress, wait briefly for it to complete
            rather than failing immediately.
        """
        from src.core.router import call_model

        timeout = self._compute_timeout(CallCategory.OVERHEAD, reqs)

        # ── Cold-start wait: if no model loaded, no cloud, but load in progress ──
        if not self._get_loaded_model_name() and self._should_wait_for_cold_start():
            await self._wait_for_model_load(reqs)

        reqs_safe = copy.copy(reqs)

        # ── Atomic swap-state check ──
        # Snapshot swap_version + loaded model atomically. If the swap
        # version changes between our snapshot and call_model's execution,
        # the exclusion list may be stale — but that's safe because:
        #   - If swap completed: loaded model changed, but we only kept
        #     it as a candidate (select_model will see it's now different).
        #   - If swap started: we excluded unloaded models already.
        # The timestamp lets us detect these transitions for logging.
        _sv = self._swap_version()
        _loaded = self._get_loaded_model_name()

        if _sv > 0 and not _loaded:
            # Swap in progress, no model available yet → cloud only
            logger.debug(f"overhead skipping local — swap in progress | task={reqs.effective_task or reqs.primary_capability}")
            reqs_safe = self._exclude_all_local(reqs_safe)
        else:
            reqs_safe = self._exclude_unloaded_local(reqs_safe)

        try:
            result = await call_model(reqs_safe, messages, tools,
                                      timeout_override=timeout)
            return result
        except Exception as e:
            raise RuntimeError(
                f"OVERHEAD call failed: loaded model and cloud unavailable. "
                f"Task: {reqs.effective_task or reqs.primary_capability}, "
                f"Error: {e}"
            ) from e

    def _should_wait_for_cold_start(self) -> bool:
        """Check if we should wait for a model to load on cold start.

        Returns True when:
          1. No cloud models are available (local-only setup)
          2. A model swap/load is in progress (proactive load started)
        """
        # Check if cloud models exist
        if self._has_cloud_models():
            return False  # cloud available, no need to wait

        # Check if a load is in progress
        return self._is_swap_in_progress()

    def _has_cloud_models(self) -> bool:
        """Check if any cloud models are available in the registry."""
        try:
            from src.models.model_registry import get_registry
            registry = get_registry()
            return any(not m.is_local for m in registry.all_models())
        except Exception:
            return False

    async def _wait_for_model_load(self, reqs: "ModelRequirements") -> None:
        """Wait for a model to finish loading (cold-start / post-idle).

        Polls _get_loaded_model_name() until a model is available or
        the timeout expires. Used only when no cloud fallback exists.
        """
        task_desc = reqs.effective_task or reqs.primary_capability
        logger.info(f"overhead waiting for model load (cold start) | task={task_desc} timeout={_COLD_START_WAIT_TIMEOUT}")

        start = time.monotonic()
        deadline = start + _COLD_START_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(_COLD_START_POLL_INTERVAL)
            if self._get_loaded_model_name():
                elapsed = time.monotonic() - start
                logger.info(f"model loaded, overhead proceeding | task={task_desc} waited={elapsed:.1f}s")
                return
            # If swap is no longer in progress and still no model, stop waiting
            if not self._is_swap_in_progress():
                logger.debug(f"swap finished but no model loaded — stop waiting | task={task_desc}")
                return

        logger.warning(f"cold-start wait timed out, proceeding without model | task={task_desc} timeout={_COLD_START_WAIT_TIMEOUT}")

    def _is_swap_in_progress(self) -> bool:
        """Check if a model swap is currently in progress."""
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()
            return manager.swap_started_at > 0
        except Exception:
            return False

    def _swap_version(self) -> float:
        """Return the swap_started_at timestamp for staleness detection.

        Callers can snapshot this value, perform work, then compare against
        a second read to detect if a swap started or completed in between.
        Returns 0.0 if no swap in progress.
        """
        try:
            from src.models.local_model_manager import get_local_manager
            return get_local_manager().swap_started_at
        except Exception:
            return 0.0

    def _exclude_all_local(self, reqs: "ModelRequirements") -> "ModelRequirements":
        """Exclude ALL local models. Used during swaps to force cloud-only routing."""
        from src.models.model_registry import get_registry

        registry = get_registry()
        all_local = [m.litellm_name for m in registry.all_models() if m.is_local]
        existing_excludes = list(reqs.exclude_models) if reqs.exclude_models else []
        reqs.exclude_models = list(set(existing_excludes + all_local))
        reqs.local_only = False
        return reqs

    def _exclude_unloaded_local(self, reqs: "ModelRequirements") -> "ModelRequirements":
        """Exclude local models that aren't loaded. Prevents swap triggers
        while keeping the loaded model as a candidate alongside cloud.

        If no model is loaded, all local models are excluded (cloud only).
        """
        from src.models.model_registry import get_registry

        registry = get_registry()
        loaded_name = self._get_loaded_model_name()

        unloaded_local = [
            m.litellm_name for m in registry.all_models()
            if m.is_local and m.name != loaded_name
        ]

        existing_excludes = list(reqs.exclude_models) if reqs.exclude_models else []
        reqs.exclude_models = list(set(existing_excludes + unloaded_local))
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

    def get_loaded_model_speed(self) -> float:
        """Get the currently loaded model's measured tok/s. Returns 0 if unknown."""
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()
            if manager.runtime_state and manager.runtime_state.measured_tps > 0:
                return manager.runtime_state.measured_tps
            # Fall back to registry value
            if manager.current_model:
                from src.models.model_registry import get_registry
                info = get_registry().get(manager.current_model)
                if info:
                    return info.tokens_per_second
        except Exception:
            pass
        return 0.0

    def is_loaded_model_thinking(self) -> bool:
        """Check if the currently loaded model has thinking enabled."""
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()
            if manager.runtime_state:
                return manager.runtime_state.thinking_enabled
        except Exception:
            pass
        return False

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
        """Called when a model swap occurs. Drains deferred grades and
        signals backpressure that new capacity is available.

        Called by ensure_model() via asyncio.ensure_future after a successful swap.
        Note: by the time this runs, the old model's server is already stopped
        and the new model is loaded.
        """
        # Drain grades the new model can handle.
        # These are grades generated by models other than new_model.
        # (Grades generated BY new_model would need a different model,
        # and grade_response excludes generating_model automatically.)
        if new_model:
            drained = await self.grade_queue.drain(available_model=new_model)
            if drained:
                logger.info(f"drained grades after swap | new_model={new_model} drained={drained}")

        # Signal backpressure queue — a model swap means new local capacity
        # is available. Queued MAIN_WORK calls that failed because the old
        # model couldn't handle them may succeed with the new one.
        try:
            from src.infra.backpressure import get_backpressure_queue
            bp = get_backpressure_queue()
            if bp.depth > 0:
                await bp.signal_capacity_available()
                logger.info(f"signaled backpressure after swap | new_model={new_model} bp_depth={bp.depth}")
        except Exception:
            pass

    async def drain_grades_if_idle(self):
        """Called from main loop when no tasks are running.

        Uses cloud to drain any remaining deferred grades.
        """
        if self.grade_queue.depth > 0:
            drained = await self.grade_queue.drain(use_cloud=True)
            if drained:
                logger.info(f"drained grades during idle | drained={drained}")

    async def drain_grades_if_full(self):
        """Called periodically. If queue exceeds threshold, force drain via cloud."""
        if self.grade_queue.needs_drain:
            logger.info(f"grade queue full, forcing cloud drain | depth={self.grade_queue.depth}")
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
                logger.info(f"proactive GPU load | model={best_model} queue_depth={len(upcoming_tasks)}")
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
                # Skip specialty/code-only models for proactive loading —
                # they can't handle general tasks (classification, chat, etc.)
                if model.specialty in ("code", "coding") or not model.supports_function_calling:
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

            # Pick model that serves the most tasks, breaking ties by speed.
            # Prefer: (1) measured TPS if available, (2) MoE models (fast on
            # partial GPU), (3) smaller file size (more layers fit on GPU).
            def _model_priority(name):
                info = registry.get(name)
                match = model_scores[name]
                if info and info.tokens_per_second > 0:
                    speed = info.tokens_per_second
                elif info and info.model_type == "moe":
                    speed = 30.0  # MoE models typically fast on this hardware
                elif info:
                    # Estimate: smaller file = more GPU layers = faster
                    speed = max(1.0, 50.0 - info.file_size_mb / 500)
                else:
                    speed = 1.0
                return (match, speed)

            best = max(model_scores, key=_model_priority)
            logger.debug(f"proactive load candidates | scores={model_scores} selected={best}")
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
