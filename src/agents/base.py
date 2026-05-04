# agents/base.py
"""
Base agent with iterative ReAct loop:
  Think → Act (tool or respond) → Observe → Think again
"""
from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import re
import time
from typing import Callable

from ..collaboration.blackboard import get_or_create_blackboard, \
    format_blackboard_for_prompt
from ..context.onboarding import get_project_profile_for_task, \
    format_project_profile
from ..memory.preferences import get_user_preferences, format_preferences
from ..memory.rag import retrieve_context
from ..models.model_registry import get_registry
from fatih_hoca.requirements import ModelRequirements
from ..core.router import select_model
from ..infra.db import (
    log_conversation,
    store_memory,
    recall_memory,
    get_completed_dependency_results,
    save_task_checkpoint,
    load_task_checkpoint,
    clear_task_checkpoint,
    record_model_call,
    update_task,
    record_cost,
)
from ..tools import TOOL_REGISTRY, TOOL_SCHEMAS, get_tool_descriptions, execute_tool
from ..app.config import MAX_AGENT_ITERATIONS, MAX_TOOL_OUTPUT_LENGTH
from ..models.models import validate_action, validate_tool_args, validate_task_output
from ..infra.logging_config import get_logger
import litellm as _litellm

logger = get_logger("agents.base")


# Phase A.4: tool VM helpers extracted to src/runtime/tools.py.
# Re-export constants and free function for callsite stability.
from src.runtime.tools import (  # noqa: F401
    SIDE_EFFECT_TOOLS,
    CACHEABLE_READ_TOOLS,
    TOOL_FAILURE_ESCALATION_THRESHOLD,
    partition_tool_calls as _partition_tool_calls,
    TOOL_SCHEMAS_BY_NAME as _TOOL_SCHEMAS_BY_NAME,
)

# Phase A.6: guards extracted to src/runtime/guards.py.
# Re-export constants and dataclass for callsite stability.
from src.runtime.guards import (  # noqa: F401
    GuardCorrection,
    MAX_SUB_CORRECTIONS,
    MAX_FORMAT_CORRECTIONS,
)

# Phase A.1: parsing extracted to src/runtime/parsing.py.
# BaseAgent re-exports for callsite stability.
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
    def _get_available_tools_prompt(self) -> str:
        """Build the tools section appended to the system prompt.

        Phase A.5: delegates to src.runtime.context.get_available_tools_prompt.
        """
        from src.runtime.context import get_available_tools_prompt
        return get_available_tools_prompt(self)

    # ------------------------------------------------------------------ #
    #  Full system prompt assembly                                        #
    # ------------------------------------------------------------------ #
    def _build_full_system_prompt(self, task: dict) -> str:
        """Compose system prompt.

        Phase A.5: delegates to src.runtime.context.build_system_prompt.
        Kept sync — original was sync and callers don't await it.
        """
        from src.runtime.context import build_system_prompt
        return build_system_prompt(self, task)

    def _is_action_task(self, task: dict) -> bool:
        """Delegate to guards module. See src/runtime/guards.py."""
        from src.runtime.guards import is_action_task
        return is_action_task(task)

    @staticmethod
    def _get_search_depth(task: dict) -> str:
        """Delegate to guards module. See src/runtime/guards.py."""
        from src.runtime.guards import get_search_depth
        return get_search_depth(task)

    # ------------------------------------------------------------------ #
    #  Sub-iteration guards                                                #
    # ------------------------------------------------------------------ #

    def _check_sub_iteration_guards(
        self,
        parsed: dict,
        iteration: int,
        tools_used: bool,
        tools_used_names: set[str],
        task: dict,
        search_depth: str,
        suppress_guards: bool,
    ) -> GuardCorrection | None:
        """Delegate to guards module. See src/runtime/guards.py."""
        from src.runtime.guards import check_sub_iter_guards
        return check_sub_iter_guards(
            parsed,
            profile=self,
            iteration=iteration,
            tools_used=tools_used,
            tools_used_names=tools_used_names,
            task=task,
            search_depth=search_depth,
            suppress_guards=suppress_guards,
        )

    # ------------------------------------------------------------------ #
    #  Tier helpers                                                       #
    # ------------------------------------------------------------------ #

    def _check_tool_permission(self, tool_name: str) -> bool:
        """Check if this agent is permitted to use tool_name (Phase 8.1).

        Phase A.4: delegates to src.runtime.tools.check_tool_permission.
        """
        from src.runtime.tools import check_tool_permission
        return check_tool_permission(self.name, tool_name)

    def _trim_for_escalation(
        self, messages: list[dict], iteration: int, max_iterations: int,
    ) -> list[dict]:
        from src.runtime.escalation import trim_for_escalation
        return trim_for_escalation(messages, iteration, max_iterations)

    def _escalate_requirements(self, reqs: ModelRequirements) -> ModelRequirements:
        from src.runtime.escalation import escalate_requirements
        return escalate_requirements(reqs)

    # ------------------------------------------------------------------ #
    #  Context builder (DB + inline fallback)                             #
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
            from ..core.llm_dispatcher import get_dispatcher
            dispatcher = get_dispatcher()
            loaded = dispatcher._get_loaded_litellm_name()
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

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Rough truncation: ~4 chars per token.

        Phase A.5: delegates to src.runtime.context.truncate_to_tokens.
        """
        from src.runtime.context import truncate_to_tokens
        return truncate_to_tokens(text, max_tokens)

    async def _fetch_deps(self, task: dict, max_tokens: int) -> str:
        """Fetch dependency results, truncated to budget.

        Phase A.5: delegates to src.runtime.context.fetch_deps.
        """
        from src.runtime.context import fetch_deps
        return await fetch_deps(self, task, max_tokens)

    def _format_prior_steps(self, task_context: dict, max_tokens: int) -> str:
        """Format inline prior steps, truncated to budget.

        Phase A.5: delegates to src.runtime.context.format_prior_steps.
        """
        from src.runtime.context import format_prior_steps
        return format_prior_steps(task_context, max_tokens)

    def _format_conversation(self, task_context: dict, max_tokens: int) -> str:
        """Format recent conversation + summaries, truncated to budget.

        Phase A.5: delegates to src.runtime.context.format_conversation.
        """
        from src.runtime.context import format_conversation
        return format_conversation(task_context, max_tokens)

    # ------------------------------------------------------------------ #
    #  JSON parsing & normalisation                                       #
    # ------------------------------------------------------------------ #
    # Phase A.1: parsing extracted to src/runtime/parsing.py.
    # Method delegates kept for callsite compat (self._parse_agent_response).
    def _parse_agent_response(self, content: str) -> dict | None:
        from src.runtime.parsing import parse_action
        return parse_action(content)

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        from src.runtime.parsing import _try_parse_json
        return _try_parse_json(text)

    @staticmethod
    def _normalize_action(parsed: dict) -> dict | None:
        from src.runtime.parsing import _normalize_action
        return _normalize_action(parsed)

    # ------------------------------------------------------------------ #
    #  Context window management                                          #
    # ------------------------------------------------------------------ #
    def _count_tokens(self, messages: list[dict], model: str) -> int:
        """Estimate token count for a message list."""
        from src.runtime.window import count_tokens
        return count_tokens(messages, model)

    def _get_context_window(self, model: str, tier_or_reqs=None) -> int:
        """Return the context window size for a model."""
        from src.runtime.window import context_window_for
        return context_window_for(model, tier_or_reqs)

    def _trim_messages_if_needed(
        self, messages: list[dict], model: str, tier_or_reqs=None,
    ) -> list[dict]:
        """
        If the conversation exceeds 80% of context, compress older exchanges.
        Accepts tier string or ModelRequirements for compat.
        """
        from src.runtime.window import trim_if_needed
        return trim_if_needed(messages, model, tier_or_reqs)

    def _prune_tool_results_to_fit(
        self,
        messages: list[dict],
        ctx_window: int,
        estimated_output_tokens: int,
        task_id: int | str = "?",
    ) -> list[dict]:
        """
        Cheap char/3 prompt-size guard that runs BEFORE the heavier
        :meth:`_trim_messages_if_needed` compression.  If the current prompt
        estimate exceeds ``ctx_window - estimated_output_tokens``, drop the
        oldest *tool-result* exchanges (assistant+user pair where the user
        message is a tool result) until we fit.  System prompt (index 0),
        initial user context (index 1), and the most recent exchange
        (last 2 messages) are always preserved.

        Logs ``[Task #X] Pruned N oldest tool results to fit context`` once
        per call when pruning happened.
        """
        from src.runtime.window import prune_tool_results
        return prune_tool_results(messages, ctx_window, estimated_output_tokens, task_id)

    # ------------------------------------------------------------------ #
    #  Function calling support                                            #
    # ------------------------------------------------------------------ #
    def _build_litellm_tools(
        self, exclude: set[str] | None = None,
    ) -> list[dict] | None:
        """Build filtered tool schemas for LiteLLM function calling.

        exclude: tool names to strip from the schema list (e.g. ``read_file``
        once the agent has already read a file this task — discourages the
        LLM from re-reading content already present in the blackboard).

        Phase A.4: delegates to src.runtime.tools.build_litellm_tools.
        """
        from src.runtime.tools import build_litellm_tools
        return build_litellm_tools(self.allowed_tools, exclude)

    @staticmethod
    def _parse_function_call_response(tool_calls: list[dict]) -> dict | None:
        # Phase A.1: parsing extracted to src/runtime/parsing.py.
        from src.runtime.parsing import parse_function_call
        return parse_function_call(tool_calls)

    # ------------------------------------------------------------------ #
    #  Output validation                                                   #
    # ------------------------------------------------------------------ #
    def _validate_response(self, result: str, task: dict) -> str | None:
        """Validate a final_answer result. Delegates to src.runtime.validation."""
        from src.runtime.validation import validate_final_answer
        return validate_final_answer(result, task)

    # ------------------------------------------------------------------ #
    #  Main execution loop                                                #
    # ------------------------------------------------------------------ #
    # ── Phase 4.6: Progress streaming callback ──
    progress_callback: Callable | None = None

    # Phase 13.1: Cached prompt override from DB (set per-execution)
    _prompt_version_override: str | None = None

    async def execute(self, task: dict, progress_callback: Callable | None = None) -> dict:
        """Drive one task to completion. Phase A.10: delegates to src.runtime.execute."""
        from src.runtime import execute as _runtime_execute
        return await _runtime_execute(self, task, progress_callback)
    async def _execute_react_loop(self, task: dict) -> dict:
        """ReAct multi-call loop. Phase A.8: delegates to src.runtime.react.run.

        Profile attrs (self.allowed_tools, self.max_iterations, etc.) and
        delegate methods (self._build_full_system_prompt, self._build_context,
        self._count_tokens, etc. — already extracted in phases A.1-A.7) are
        consumed by react.run via duck-typed profile interface.
        """
        from src.runtime.react import run as _react_run
        return await _react_run(self, task)
    async def _build_model_requirements(
        self, task: dict, task_ctx: dict,
    ) -> ModelRequirements:
        """
        Build ModelRequirements from task metadata + agent properties.
        Uses the new task-based routing while preserving all existing logic.
        """
        title = task.get("title", "").lower()
        description = task.get("description", "").lower()
        priority = task.get("priority", 5)

        # ── Start from curated AGENT_REQUIREMENTS template ──
        from fatih_hoca.requirements import AGENT_REQUIREMENTS
        import copy

        classification = task_ctx.get("classification", {})
        # Workflow steps declare their agent explicitly — don't let the
        # classifier override it (e.g. "writer" step misclassified as "coder")
        if task_ctx.get("is_workflow_step"):
            agent_type = self.name
        else:
            agent_type = classification.get("agent_type", self.name)

        template = AGENT_REQUIREMENTS.get(agent_type) or AGENT_REQUIREMENTS.get(
            self.name, ModelRequirements(task=agent_type, difficulty=5)
        )
        reqs = copy.deepcopy(template)
        reqs.agent_type = self.name
        reqs.priority = priority

        # Overlay classification signals (only upgrade, never downgrade)
        cls_difficulty = classification.get("difficulty", 5)
        reqs.difficulty = max(reqs.difficulty, cls_difficulty)

        if classification.get("needs_tools"):
            reqs.needs_function_calling = True
        if classification.get("needs_vision"):
            reqs.needs_vision = True
        if classification.get("needs_thinking"):
            reqs.needs_thinking = True
        if classification.get("local_only"):
            reqs.local_only = True

        # ── Adjust for task priority ──
        if priority >= 10:
            reqs.prefer_speed = True
            reqs.difficulty = max(reqs.difficulty, 6)
        elif priority <= 2:
            reqs.difficulty = max(1, reqs.difficulty - 2)

        # ── Detect personal/sensitive data ──
        sensitivity_keywords = [
            "personal", "private", "secret", "password",
            "credential", "my ", "my_", "home",
        ]
        if any(kw in f"{title} {description}" for kw in sensitivity_keywords):
            reqs.local_only = True

        if task_ctx.get("local_only"):
            reqs.local_only = True
        if task_ctx.get("prefer_quality"):
            reqs.prefer_quality = True
        if task_ctx.get("prefer_speed"):
            reqs.prefer_speed = True
            reqs.prefer_local = False

        # ── Model diversity ──
        exclude = task_ctx.get("exclude_models", [])
        if exclude:
            reqs.exclude_models = exclude

        # ── Retry-based model exclusion and difficulty escalation ──
        task_attempts = task.get("worker_attempts", 0) or 0
        if task_attempts >= 3:
            from src.core.retry import get_model_constraints
            retry_excluded, difficulty_bump = get_model_constraints(task_ctx, task_attempts)
            if retry_excluded:
                existing = list(reqs.exclude_models) if reqs.exclude_models else []
                reqs.exclude_models = list(set(existing + retry_excluded))
            if difficulty_bump > 0:
                reqs.difficulty = min(10, reqs.difficulty + difficulty_bump)

        # ── Estimate context size ──
        desc_len = len(task.get("description", ""))
        context_json = task.get("context", "{}")
        if isinstance(context_json, str):
            ctx_len = len(context_json)
        else:
            ctx_len = len(json.dumps(context_json))

        estimated_input = (desc_len + ctx_len) // 4  # rough char-to-token
        reqs.estimated_input_tokens = max(estimated_input, 1000)
        # Template's estimated_output_tokens is a per-agent default (e.g.
        # analyst=3000, coder=4000). List-heavy workflow steps like
        # feature_prioritization need far more — a 15-25-item MoSCoW
        # breakdown with justifications runs 5-8k tokens and the default
        # caps the LLM mid-list, leaving trailing keys like 'could_have'
        # and 'wont_have' empty and the artifact failing schema validation
        # on "missing content about: [...]". Let the workflow step override
        # via context.estimated_output_tokens (clamped to [500, 12000]).
        _out_override = task_ctx.get("estimated_output_tokens")
        # Workflow-step tasks may pre-date the step's context-block edit
        # in i2p_v3 (existing DB rows captured their ctx at expansion time
        # with no estimated_output_tokens field). Re-read the live step
        # def from the workflow JSON so retries pick up newly-bumped
        # budgets without requiring row regeneration.
        if not _out_override and task_ctx.get("is_workflow_step"):
            try:
                step_id = task_ctx.get("workflow_step_id")
                mission_id = task.get("mission_id")
                if step_id and mission_id:
                    from src.infra.db import get_db
                    _db = await get_db()
                    _cur = await _db.execute(
                        "SELECT context FROM missions WHERE id = ?", (mission_id,),
                    )
                    _row = await _cur.fetchone()
                    await _cur.close()
                    _mctx = {}
                    if _row and _row[0]:
                        try:
                            _mctx = json.loads(_row[0])
                            if isinstance(_mctx, str):
                                _mctx = json.loads(_mctx)
                        except (json.JSONDecodeError, TypeError):
                            _mctx = {}
                    _wf_name = (
                        _mctx.get("workflow_name") if isinstance(_mctx, dict) else None
                    ) or "i2p_v3"
                    from src.workflows.engine.loader import load_workflow
                    _wf = load_workflow(_wf_name)
                    _step = _wf.get_step(step_id)
                    if _step:
                        _step_ctx = _step.get("context") or {}
                        if isinstance(_step_ctx, dict):
                            _out_override = _step_ctx.get("estimated_output_tokens")
                            if _out_override:
                                logger.info(
                                    f"[Task #{task.get('id','?')}] step-refresh: "
                                    f"estimated_output_tokens={_out_override} "
                                    f"(step={step_id}, wf={_wf_name})"
                                )
            except Exception as _e:
                logger.warning(
                    f"[Task #{task.get('id','?')}] step-config refresh failed: {_e}"
                )
        if _out_override:
            try:
                # 16k ceiling chosen to cover feature_brainstorm's 50-200
                # item list worst-case; still well under any local model's
                # trained ctx. Bump if a new step needs more.
                reqs.estimated_output_tokens = max(500, min(16000, int(_out_override)))
            except (TypeError, ValueError):
                pass

        # ── Tools needed? (agent-level override) ──
        # Only upgrade to function_calling, don't force it if the
        # template doesn't need it (e.g., planner, writer, reviewer).
        # The template already sets needs_function_calling=True for
        # agents that genuinely need tool use (coder, fixer, executor).
        if reqs.needs_function_calling:
            pass  # already set by template or classification
        elif self.allowed_tools and len(self.allowed_tools) > 0:
            # Agent explicitly declares tool list → it needs function calling
            reqs.needs_function_calling = True

        # ── Vision needed? (keyword override) ──
        # Skip keyword heuristic for workflow steps — they declare vision
        # need explicitly via tools_hint containing analyze_image.
        if task_ctx.get("needs_vision"):
            reqs.needs_vision = True
        elif not task_ctx.get("is_workflow_step"):
            if any(kw in f"{title} {description}" for kw in [
                "screenshot", "image", "visual", "ui review", "layout",
                "diagram", "photo", "picture",
            ]):
                reqs.needs_vision = True

        # ── Thinking needed? ──
        if task_ctx.get("needs_thinking"):
            reqs.needs_thinking = True

        # ── Workflow difficulty override ──
        wf_difficulty = task_ctx.get("difficulty")
        if wf_difficulty and isinstance(wf_difficulty, int):
            reqs.difficulty = max(reqs.difficulty, wf_difficulty)

        return reqs

    async def _maybe_constrained_emit(self, task: dict, result: dict) -> dict:
        """Post-execution structured-output guarantee for workflow steps.

        Logged failure data showed models dropping the same required JSON
        field across 5+ retries even when ``_schema_error`` injection
        named the missing field (mission 46 step 7.4 stuck on
        ``connection_verified`` 25 times). Post-hoc retry hints are
        whack-a-mole — the structural fix is to constrain decoding so the
        omission can't occur.

        Behaviour:

        * No-op unless this is a workflow step with a constrainable
          ``artifact_schema`` (object / array — markdown is unconstrainable
          and handled by the validator + writer schema-aware prompt).
        * No-op unless the upstream result is a normal completion. We do
          not rewrite ``needs_subtasks``, ``needs_clarification``,
          ``needs_review``, or already-failed results.
        * Skips when the model picked for the fix-up call doesn't support
          json_schema — caller's degradation logic handles this, but here
          we just don't waste a call when the registry has no capable
          candidate.
        * On any error, returns the original ``result`` unchanged. The
          existing schema validation hook will still flag missing fields
          and trigger the normal retry path — fix-up is a best-effort
          win, never a regression.

        Cost: one extra OVERHEAD call per artifact step. Acceptable when
        the alternative is 5 worker retries × main_work cost on a hot
        loaded model.
        """
        if not isinstance(result, dict):
            return result
        if result.get("status") not in (None, "completed"):
            # Failures, clarifies, subtasks pass through unchanged.
            return result
        draft = result.get("result")
        if not isinstance(draft, str) or not draft.strip():
            return result

        ctx = task.get("context") or {}
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
                if isinstance(ctx, str):
                    ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError, ValueError):
                return result
        if not isinstance(ctx, dict):
            return result
        if not ctx.get("is_workflow_step"):
            return result

        artifact_schema = ctx.get("artifact_schema")
        if not isinstance(artifact_schema, dict):
            return result

        from src.workflows.engine.json_schema_translator import (
            build_response_format,
        )
        step_id = ctx.get("workflow_step_id") or "artifact"
        # JSON Schema 'name' must be alphanumeric+underscore; sanitize.
        safe_name = "step_" + "".join(
            c if c.isalnum() else "_" for c in str(step_id)
        )
        response_format = build_response_format(
            artifact_schema, name=safe_name,
        )
        if response_format is None:
            # Unconstrainable (markdown / string). Skip — validator and
            # writer-schema-aware prompt cover that path.
            return result

        # Build a tight prompt that re-emits the artifact in conforming
        # JSON. The model receives the raw JSON Schema so it can see
        # exactly what fields are required, plus the draft to anchor on.
        schema_text = json.dumps(
            response_format["json_schema"]["schema"],
            ensure_ascii=False,
            indent=2,
        )
        # Skip the emit when the draft already parses as JSON with all
        # required artifact keys present. Re-emitting in that case
        # tends to COMPRESS rather than reshape — the model sees a long
        # rich draft, gets a tight token budget, and trims content from
        # tail fields to fit (mission 57 task 4441 5.4b: draft 30751
        # chars with full empty_states/error_states arrays became a
        # 12826-char emit with empty placeholder lists). The schema
        # validator runs next and catches genuine shape gaps; the emit
        # pass is only valuable when the draft is non-JSON or missing
        # top-level keys.
        try:
            _parsed = json.loads(draft)
            if isinstance(_parsed, dict):
                _need = [
                    n for n, r in artifact_schema.items()
                    if isinstance(r, dict) and r.get("type") in ("object", "array")
                ]
                if _need and all(k in _parsed for k in _need):
                    logger.info(
                        f"[Task #{task.get('id','?')}] constrained_emit skipped "
                        f"— draft parses with all required keys present "
                        f"(step={step_id}, draft={len(draft)} chars)"
                    )
                    return result
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # Draft isn't JSON — emit pass will reshape.

        # Cap draft to keep input token cost in line. Bumped from 12000
        # to 30000 so big multi-artifact drafts (form_specs +
        # empty_error_state_specs) don't lose tail content before the
        # emit even sees it. Local OVERHEAD calls have no per-token
        # cost; cap is purely a context-window guardrail.
        draft_for_prompt = draft[:30000]
        system = (
            "You are a structured-output emitter. Re-emit the artifact "
            "below as JSON conforming exactly to the provided schema. "
            "Do not add commentary. Do not wrap in envelopes. Output "
            "ONLY the JSON value.\n\n"
            "Rules:\n"
            "- Every required field must be present with a real value.\n"
            "- Do not invent fields not in the schema.\n"
            "- Preserve the draft's information; restructure into the "
            "schema, do not summarize away content."
        )
        user = (
            f"Schema:\n```json\n{schema_text}\n```\n\n"
            f"Draft to fix:\n```\n{draft_for_prompt}\n```\n\n"
            f"Emit the final JSON now."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            from src.core.llm_dispatcher import get_dispatcher, CallCategory
            resp = await get_dispatcher().request(
                CallCategory.OVERHEAD,
                task="structured_emit",
                difficulty=3,
                messages=messages,
                estimated_input_tokens=max(1000, len(user) // 4),
                # Output budget: previously min(12000, len/3) which gave
                # 4000 tokens for a 12000-char prompt — way too tight
                # for multi-artifact schemas (5.4b: form_specs +
                # empty_error_state_specs). Now floors at len/3 with
                # a higher ceiling so emits can preserve a large
                # draft instead of compressing.
                estimated_output_tokens=min(
                    16000,
                    max(2000, len(draft_for_prompt) // 3),
                ),
                prefer_speed=True,
                response_format=response_format,
                task_obj=task,
            )
        except Exception as exc:
            logger.warning(
                f"[Task #{task.get('id','?')}] constrained_emit dispatch "
                f"failed: {exc!r} — keeping draft"
            )
            return result

        emitted = resp.get("content", "") if isinstance(resp, dict) else ""
        if isinstance(emitted, list):
            emitted = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in emitted
            )
        if not isinstance(emitted, str) or not emitted.strip():
            logger.warning(
                f"[Task #{task.get('id','?')}] constrained_emit returned "
                f"empty — keeping draft"
            )
            return result

        # Cheap shape check: must parse as JSON. The schema-validation
        # hook will do the deeper required-field check.
        try:
            json.loads(emitted)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                f"[Task #{task.get('id','?')}] constrained_emit produced "
                f"non-JSON output (model={resp.get('model','?')}) — keeping draft"
            )
            return result

        logger.info(
            f"[Task #{task.get('id','?')}] constrained_emit applied "
            f"(model={resp.get('model','?')}, "
            f"draft={len(draft)} -> emit={len(emitted)} chars, "
            f"step={step_id})"
        )
        # Replace result while preserving metadata.
        new_result = dict(result)
        new_result["result"] = emitted
        new_result["constrained_emit_applied"] = True
        return new_result

    async def execute_single_shot(self, task: dict) -> dict:
        """Single LLM call. Phase A.9: delegates to src.runtime.single_shot.run."""
        from src.runtime.single_shot import run as _ss_run
        return await _ss_run(self, task)
    async def _self_reflect(
        self, task: dict, result: str,
        tier_or_reqs=None, used_model: str = "",
    ) -> dict | None:
        """Review own output for errors. Accepts tier string or ModelRequirements."""
        from src.runtime.reflection import self_reflect
        return await self_reflect(task, result, tier_or_reqs, used_model)

    # ------------------------------------------------------------------ #
    #  Idempotency helpers                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tool_idempotency_key(tool_name: str, tool_args: dict) -> str:
        """Compute a short hash key for a tool call's identity.

        Used to skip re-execution of side-effect tools (write_file, shell,
        git_commit, etc.) when resuming from a checkpoint.
        """
        # Phase A.7c: delegates to src/runtime/checkpoint.tool_idempotency_key
        from src.runtime.checkpoint import tool_idempotency_key
        return tool_idempotency_key(tool_name, tool_args)

    # ------------------------------------------------------------------ #
    #  Checkpointing helpers                                              #
    # ------------------------------------------------------------------ #
    async def _save_checkpoint(
        self,
        task_id,
        next_iteration: int,
        messages: list[dict],
        total_cost: float,
        used_model: str,
        reqs: ModelRequirements,
        tools_used: bool,
        validation_retried: bool,
        completed_tool_ops: dict[str, str] | None = None,
        format_corrections: int = 0,
        tools_used_names: set[str] | None = None,
    ) -> None:
        """Persist agent loop state so execution can resume after a crash."""
        # Phase A.7c: delegates to src/runtime/checkpoint.save_checkpoint
        from src.runtime.checkpoint import save_checkpoint
        return await save_checkpoint(
            task_id, next_iteration, messages, total_cost, used_model, reqs,
            tools_used, validation_retried, completed_tool_ops, format_corrections,
            tools_used_names,
        )

    async def _clear_checkpoint_safe(self, task_id) -> None:
        """Clear checkpoint on successful completion — never raises."""
        # Phase A.7c: delegates to src/runtime/checkpoint.clear_checkpoint_safe
        from src.runtime.checkpoint import clear_checkpoint_safe
        return await clear_checkpoint_safe(task_id)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    async def _safe_log(
        self,
        task_id,
        role: str,
        content: str,
        model: str | None,
        cost: float,
    ) -> None:
        """Fire-and-forget conversation log — never breaks the loop."""
        # Phase A.7c: delegates to src/runtime/checkpoint.safe_log_conversation
        from src.runtime.checkpoint import safe_log_conversation
        return await safe_log_conversation(task_id, role, content, model, cost, self.name)
