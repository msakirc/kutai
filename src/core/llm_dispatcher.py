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
from enum import Enum
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("core.llm_dispatcher")

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
        # ── OVERHEAD: 20s default, 60s for grading ────────────────────────
        if category == CallCategory.OVERHEAD:
            # Grading generates ~100 tokens of structured output and may
            # run on slower models (MoE, large dense).  Classifiers and
            # other overhead are short-context, low-output → 20s is fine.
            if reqs.task == "reviewer":
                return 60.0
            return 20.0

        # ── MAIN_WORK: TPS-based adaptive timeout ─────────────────────────
        _MAIN_WORK_MIN = 20.0
        _MAIN_WORK_MAX = 300.0

        # Try runtime measured_tps first (most accurate)
        try:
            from src.models.local_model_manager import get_runtime_state
            runtime = get_runtime_state()
            if runtime is not None and runtime.measured_tps > 0.0:
                tps = runtime.measured_tps
                # Thinking models: /metrics reports TPS including thinking
                # tokens, which are much faster than content tokens.  The
                # effective content-only speed is ~3-4x slower, so halve
                # the measured TPS to avoid timeouts that are too aggressive.
                if runtime.thinking_enabled:
                    tps = tps * 0.5
                est_gen_secs = reqs.estimated_output_tokens / tps
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
                            f"main_work: pinned loaded model failed ({loaded}), "
                            f"falling through to full routing. "
                            f"error={str(e)[:100]} task={reqs.effective_task or reqs.primary_capability}"
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
            4. If cloud also fails, ModelCallFailed → sleeping queue

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

    async def on_model_swap(self, old_model: str | None, new_model: str | None):
        """Called when a model swap occurs. Grades ungraded tasks and
        wakes availability-delayed tasks.

        Order matters: accelerate first, then drain.
        """
        # 1. Wake availability-delayed tasks
        try:
            from src.infra.db import accelerate_retries
            woken = await accelerate_retries("model_swap")
            if woken:
                logger.info(f"accelerated {woken} task(s) after swap")
        except Exception as e:
            logger.debug(f"accelerate_retries failed: {e}")

        # 2. Grade ungraded tasks the new model can handle
        if new_model:
            try:
                from src.core.grading import drain_ungraded_tasks
                graded = await drain_ungraded_tasks(new_model)
                if graded:
                    logger.info(f"graded {graded} task(s) after swap to {new_model}")
            except Exception as e:
                logger.debug(f"drain_ungraded_tasks failed: {e}")

    # ─── Proactive GPU Loading ───────────────────────────────────────────

    async def ensure_gpu_utilized(self, upcoming_tasks: list[dict]):
        """Proactively load a local model if GPU is idle and there's work.

        Enhanced: when loaded model can't grade self-generated tasks,
        swap to a grader model instead of waiting for idle unload.
        """
        try:
            from src.models.local_model_manager import get_local_manager
            manager = get_local_manager()

            if upcoming_tasks:
                if not manager.current_model:
                    best_model = self._find_best_local_for_batch(upcoming_tasks)
                    if best_model:
                        logger.info(f"proactive GPU load | model={best_model} queue_depth={len(upcoming_tasks)}")
                        await manager.ensure_model(best_model, reason="proactive_load")
                return

            # No main work. Check overhead needs.
            if not await self._has_pending_overhead_needs():
                return

            if manager.current_model:
                can_grade = await self._loaded_model_can_grade()
                if can_grade or can_grade is None:
                    return  # gradeable or no ungraded tasks → idle path handles it
                # Loaded model can't grade any ungraded task — find best alternative
                best = await self._find_fastest_general_model()
                if best and best != manager.current_model:
                    logger.info(f"grade swap | loaded={manager.current_model} → {best}")
                    await manager.ensure_model(best, reason="grade_swap")
            else:
                best = await self._find_fastest_general_model()
                if best:
                    logger.info(f"overhead load | model={best}")
                    await manager.ensure_model(best, reason="overhead_load")

        except Exception as e:
            logger.debug(f"ensure_gpu_utilized failed: {e}")

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

    async def _has_pending_overhead_needs(self) -> bool:
        """Check if there's pending overhead work that needs a model loaded.

        Only ungraded tasks qualify — they need a grader model.
        Todos are handled by the reminder scheduler, not proactive loading.
        """
        try:
            from src.infra.db import get_db
            db = await get_db()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'ungraded'"
            )
            return (await cursor.fetchone())[0] > 0
        except Exception:
            return False

    async def _loaded_model_can_grade(self) -> bool | None:
        """Check if loaded model can grade ANY ungraded task.

        Returns:
            True  — loaded model can grade at least one task
            False — ungraded tasks exist but loaded model can't grade any
            None  — no ungraded tasks exist (nothing to grade)
        """
        loaded = self._get_loaded_litellm_name()
        if not loaded:
            return False
        try:
            import json
            from src.infra.db import get_db
            db = await get_db()
            cursor = await db.execute(
                "SELECT context FROM tasks WHERE status = 'ungraded'"
            )
            rows = await cursor.fetchall()
            if not rows:
                return None  # no grading work
            for row in rows:
                try:
                    ctx = json.loads(row["context"] or "{}")
                except (ValueError, TypeError):
                    ctx = {}
                if ctx.get("generating_model") == loaded:
                    continue
                if loaded in ctx.get("grade_excluded_models", []):
                    continue
                return True
            return False
        except Exception:
            return False

    async def _find_fastest_general_model(self) -> str | None:
        """Find the best local model for overhead work.

        When ungraded tasks exist, score = gradeable_count x speed.
        Models that can't grade anything score zero and are eliminated.
        When no ungraded tasks exist, pure speed ranking.
        """
        try:
            import json
            from src.models.model_registry import get_registry

            registry = get_registry()
            candidates = [
                m for m in registry.all_models()
                if m.is_local
                and not m.demoted
                and m.supports_function_calling
                and m.specialty not in ("code", "coding")
                # Exclude vision variants — loading mmproj wastes RAM
                # for non-vision tasks like grading
                and "vision" not in getattr(m, "variant_flags", set())
            ]
            if not candidates:
                return None

            def _speed(m):
                if m.tokens_per_second > 0:
                    return m.tokens_per_second
                if m.model_type == "moe":
                    return 30.0
                return max(1.0, 50.0 - m.file_size_mb / 500)

            # Count gradeable tasks per model (the main overhead signal)
            task_exclusions = await self._get_grade_exclusions()
            total_ungraded = len(task_exclusions)

            best_name = None
            best_score = 0.0
            for m in candidates:
                speed = _speed(m)

                if total_ungraded == 0:
                    # No grading work — pure speed
                    score = speed
                else:
                    gradeable = sum(
                        1 for excl in task_exclusions
                        if m.litellm_name not in excl
                    )
                    # Zero gradeable = zero score = eliminated.
                    # Otherwise score = gradeable_count x speed.
                    score = gradeable * speed

                if score > best_score:
                    best_score = score
                    best_name = m.name

            return best_name

        except Exception as e:
            logger.debug(f"_find_fastest_general_model failed: {e}")
            return None

    async def _get_grade_exclusions(self) -> list[set[str]]:
        """Get per-task exclusion sets for ungraded tasks.

        Returns a list where each element is the set of litellm_names
        that cannot grade that task (generating model + grade_excluded).
        Empty list if no ungraded tasks.
        """
        try:
            import json
            from src.infra.db import get_db
            db = await get_db()
            cursor = await db.execute(
                "SELECT context FROM tasks WHERE status = 'ungraded'"
            )
            rows = await cursor.fetchall()
            if not rows:
                return []

            exclusions: list[set[str]] = []
            for row in rows:
                try:
                    ctx = json.loads(row["context"] or "{}")
                except (ValueError, TypeError):
                    ctx = {}
                excluded = {ctx.get("generating_model", "")}
                excluded.update(ctx.get("grade_excluded_models", []))
                excluded.discard("")
                exclusions.append(excluded)
            return exclusions
        except Exception:
            return []

    # ─── Metrics ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "overhead_calls": self._overhead_calls,
            "overhead_pct": (
                f"{self._overhead_calls / self._total_calls * 100:.1f}%"
                if self._total_calls > 0 else "0%"
            ),
            "swaps_prevented": self._swaps_prevented,
            "swap_budget_remaining": self.swap_budget.remaining,
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_dispatcher: LLMDispatcher | None = None


def get_dispatcher() -> LLMDispatcher:
    """Get or create the global LLM dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = LLMDispatcher()
    return _dispatcher
