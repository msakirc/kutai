# agents/base.py
"""BaseAgent — profile interface for the runtime.

Phase A.11 cutover (2026-05-04). Most of the original 4092-LOC class moved
to src/runtime/. This file is now ~600 LOC and shrinking — three method
bodies still live here (_build_model_requirements, _build_context wrapper,
_maybe_constrained_emit), they relocate in phases A.12 + A.13.

Subclasses override ``get_system_prompt`` and customize attributes
(allowed_tools, max_iterations, execution_pattern, enable_self_reflection,
min_confidence, can_create_subtasks). The runtime treats subclasses as
duck-typed Profile objects.
"""
from __future__ import annotations

import copy
import json
from typing import Callable

from ..collaboration.blackboard import get_or_create_blackboard, \
    format_blackboard_for_prompt
from ..context.onboarding import get_project_profile_for_task, \
    format_project_profile
from ..memory.preferences import get_user_preferences, format_preferences
from ..memory.rag import retrieve_context
from fatih_hoca.requirements import ModelRequirements
from ..infra.db import recall_memory
from ..app.config import MAX_AGENT_ITERATIONS
from ..infra.logging_config import get_logger

logger = get_logger("agents.base")


# Re-exports for backward-compat callsites that imported these from base.py.
from src.runtime.tools import (  # noqa: F401
    SIDE_EFFECT_TOOLS,
    CACHEABLE_READ_TOOLS,
    TOOL_FAILURE_ESCALATION_THRESHOLD,
    partition_tool_calls as _partition_tool_calls,
)
from src.runtime.guards import (  # noqa: F401
    GuardCorrection,
    MAX_SUB_CORRECTIONS,
    MAX_FORMAT_CORRECTIONS,
)
from src.runtime.parsing import unwrap_final_answer as _unwrap_final_answer  # noqa: F401


class BaseAgent:
    """
    Base agent implementing a multi-turn ReAct loop.

    Subclasses override ``get_system_prompt`` and optionally set
    ``allowed_tools``, ``max_iterations``, ``min_tier``, etc.
    """

    name: str = "base"
    description: str = "General-purpose agent"
    default_tier: str = "cheap"
    min_tier: str = "cheap"

    # None → all tools allowed;  [] → no tools;  ["x","y"] → only those
    allowed_tools: list[str] | None = None

    # Default iteration budget (== MAX_AGENT_ITERATIONS from config).
    # Each agent subclass overrides this with a value tuned to its typical
    # workflow length.  See per-agent comments for rationale.
    max_iterations: int = MAX_AGENT_ITERATIONS

    can_create_subtasks: bool = False
    _suppress_clarification: bool = False

    # ── Phase 5: Execution pattern ──
    # "react_loop" (default) — multi-turn with tools
    # "single_shot" — one LLM call, no tool loop (planner, classifier)
    execution_pattern: str = "react_loop"

    # ── Phase 5: Self-reflection ──
    # If True, inject a "review your own output" prompt before accepting
    # final_answer. Costs one extra LLM call but catches obvious mistakes.
    enable_self_reflection: bool = False

    # ── Phase 5: Confidence-gated output ──
    # Minimum confidence (1-5) for final_answer. Below this → reviewer.
    min_confidence: int = 0  # 0 = disabled

    # ── Z10 T1A: confidence-gate enforcement mode ──
    # "fail_closed" (default): below-threshold output blocks the step and
    #     escalates to reviewer (status="needs_review", routed by Beckman's
    #     result_router → RequestReview).
    # "warn"        : log a structured warning and proceed (use for info-
    #     gathering agents whose downstream filters can compensate, e.g.
    #     researcher).
    # See docs/i2p-evolution/10-cross-cutting.md "Confidence gate is unplugged".
    confidence_gate: str = "fail_closed"

    # ------------------------------------------------------------------ #
    #  System prompt — override in subclasses                             #
    #                                                                     #
    #  ╔════════════════════════════════════════════════════════════════╗ #
    #  ║ READ BEFORE EDITING:                                           ║ #
    #  ║                                                                ║ #
    #  ║ The runtime SOURCE OF TRUTH for agent prompts is the           ║ #
    #  ║ `prompt_versions` DB table, not this file. Editing the         ║ #
    #  ║ get_system_prompt strings below ONLY changes a frozen          ║ #
    #  ║ reference — the running system loads from DB via               ║ #
    #  ║ get_active_prompt() and ignores this code at execute() time.   ║ #
    #  ║                                                                ║ #
    #  ║ To change a live prompt:                                       ║ #
    #  ║   1. Use `/prompt save <agent>` (or save_prompt_version()) to  ║ #
    #  ║      insert a new row, activate=True. The system picks it up   ║ #
    #  ║      on next dispatch — no restart needed.                     ║ #
    #  ║   2. Phase 13.1 auto-promotion (record_prompt_quality +        ║ #
    #  ║      _maybe_promote_candidate) will A/B and promote winners.   ║ #
    #  ║                                                                ║ #
    #  ║ Why this design:                                               ║ #
    #  ║   - Runtime evolution without git+restart                       ║ #
    #  ║   - Quality telemetry per prompt version                        ║ #
    #  ║   - A/B comparisons + auto-promotion of winning variants        ║ #
    #  ║                                                                ║ #
    #  ║ Boot-time auto-seed was removed 2026-04-25 because it silently ║ #
    #  ║ re-derived DB from possibly-stale code. Code edits to these    ║ #
    #  ║ strings are now isolated by design.                            ║ #
    #  ╚════════════════════════════════════════════════════════════════╝ #
    # ------------------------------------------------------------------ #
    def get_system_prompt(self, task: dict) -> str:
        """
        Return the base system prompt.  Override in every concrete agent.

        NOTE: This is a frozen reference, not the live prompt. See the
        block comment above. Use `/prompt save` to change the live prompt.
        """
        return (
            f"You are a helpful AI assistant named '{self.name}'.\n"
            f"Complete the given task thoroughly and accurately."
        )

    # ------------------------------------------------------------------ #
    #  Tool-description block                                             #
    # ------------------------------------------------------------------ #
    async def _build_context(self, task: dict) -> str:
        """Assemble the user message with task info and policy-gated context layers.

        Phase A.5: delegates to src.runtime.context.build_user_context.
        Applies the skill-injection bug fix: injected tools are added to a
        per-execution mutable copy of self.allowed_tools (via the existing
        _original_allowed_tools snapshot pattern) instead of mutating the
        shared class attribute.
        """
        from src.runtime.context import build_user_context

        # Resolve model context window — try dispatcher's loaded model
        model_ctx = 4096
        try:
            from src.models.introspection import get_loaded_litellm_name
            loaded = get_loaded_litellm_name()
            if loaded:
                model_ctx = self._get_context_window(loaded) or 4096
        except Exception:
            pass

        ctx_str, injected_skills = await build_user_context(
            self, task, model_ctx=model_ctx
        )

        # Apply skill-injection bug fix: mutate only the per-execution copy.
        # If _original_allowed_tools is already set (tools_hint / _strip_set
        # path already created a snapshot in execute()), self.allowed_tools is
        # already a mutable per-execution list — just append to it.
        # If not yet set, create the snapshot first so execute()'s finally
        # block can restore the original.
        if injected_skills:
            if not hasattr(self, '_original_allowed_tools'):
                self._original_allowed_tools = self.allowed_tools
                self.allowed_tools = list(self.allowed_tools or [])
            for tool in injected_skills:
                if self.allowed_tools is not None and tool not in self.allowed_tools:
                    self.allowed_tools.append(tool)

        return ctx_str

    # (Phase A.5: original ~540-LOC _build_context body moved to
    #  src/runtime/context.py::build_user_context. Edit there, not here.)

    async def execute(self, task: dict, progress_callback: Callable | None = None) -> dict:
        """Drive one task to completion. Phase A.10: delegates to src.runtime.execute."""
        from src.runtime import execute as _runtime_execute
        return await _runtime_execute(self, task, progress_callback)