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
        """
        Route to appropriate execution pattern, then run.
        progress_callback: async fn(task_id, iteration, max_iter, summary)
        """
        self.progress_callback = progress_callback

        # Phase 13.1: Load active prompt version from DB (if available)
        self._prompt_version_override = None
        try:
            from ..memory.prompt_versions import get_active_prompt
            db_prompt = await get_active_prompt(self.name)
            if db_prompt:
                self._prompt_version_override = db_prompt
        except Exception:
            pass

        # ── Override allowed_tools from workflow tools_hint ──
        _task_ctx = task.get("context")
        if isinstance(_task_ctx, str):
            try:
                _task_ctx = json.loads(_task_ctx)
            except (json.JSONDecodeError, TypeError):
                _task_ctx = {}
        if not isinstance(_task_ctx, dict):
            _task_ctx = {}
        tools_hint = _task_ctx.get("tools_hint")
        if tools_hint is not None and isinstance(tools_hint, list):
            self._original_allowed_tools = self.allowed_tools
            self.allowed_tools = tools_hint

        # Strip file tools when all input artifacts are already in context.
        # Prevents wasting iterations re-reading data that's in the prompt.
        _FILE_TOOLS = {"read_file", "file_tree", "project_info"}
        _WEB_TOOLS = {"web_search", "smart_search", "extract_url"}
        # Tools that write content to disk. For steps whose output is
        # expected in the `result` field (artifact_schema = object /
        # array / string / markdown), exposing these invites small
        # models to call write_file with the output stuffed into a
        # JSON-stringified "content" arg — the resulting escape-hell
        # fails the parser (observed 2026-04-23 task 2865 DLQ'd after
        # 5 such attempts). Workflow engine already persists the
        # `result` to workspace files itself, so write tools are
        # redundant for structured-output steps.
        _WRITE_TOOLS = {"write_file", "apply_diff", "edit_file", "patch_file"}
        _strip_set = set()
        if _task_ctx.get("_strip_file_tools"):
            _strip_set |= _FILE_TOOLS
        if _task_ctx.get("_strip_web_tools"):
            _strip_set |= _WEB_TOOLS
        # Auto-strip write tools when the step has a structured-output
        # schema. Explicit opt-out via "_allow_write_tools" for the rare
        # step that legitimately needs both schema'd result AND file
        # side-effects.
        _schema = _task_ctx.get("artifact_schema")
        if (_schema and isinstance(_schema, dict)
                and not _task_ctx.get("_allow_write_tools")):
            _strip_set |= _WRITE_TOOLS
        if _strip_set:
            if self.allowed_tools is not None:
                if not hasattr(self, '_original_allowed_tools'):
                    self._original_allowed_tools = self.allowed_tools
                self.allowed_tools = [t for t in self.allowed_tools if t not in _strip_set]
            else:
                from src.tools import list_tool_names
                self._original_allowed_tools = self.allowed_tools
                self.allowed_tools = [t for t in list_tool_names() if t not in _strip_set]

        # Suppress clarification if task explicitly disallows it
        self._suppress_clarification = _task_ctx.get("may_need_clarification") is False

        # ── Workflow step: refresh live JSON-driven fields ──
        # tasks.description and context.* are frozen at expander time.
        # Edits to workflow JSON don't propagate to existing rows, so
        # retries kept running stale config (observed task 2890: step
        # 2.8 instruction reshaped from 11-field use cases → 6-field
        # stories, but task row still carried old text and grader kept
        # citing "use cases" in DLQ reasons).
        #
        # Refreshed scope (every field whose stale value would mislead
        # live execution):
        #   - description (was: instruction)
        #   - done_when, input_artifacts, output_artifacts,
        #     artifact_schema, tools_hint, difficulty,
        #     estimated_output_tokens, may_need_clarification,
        #     triggers_clarification (in _task_ctx)
        #   - any keys defined in the step's "context" sub-dict
        # NOT refreshed (would change task identity mid-flight or are
        # already evaluated at expansion time):
        #   - agent_type, skip_when, depends_on
        if _task_ctx.get("is_workflow_step"):
            try:
                _step_id = _task_ctx.get("workflow_step_id")
                _mid = task.get("mission_id")
                if _step_id and _mid:
                    from src.infra.db import get_db
                    _db = await get_db()
                    _cur = await _db.execute(
                        "SELECT context FROM missions WHERE id = ?", (_mid,),
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
                    _step = _wf.get_step(_step_id)
                    if _step:
                        _changed_fields: list[str] = []

                        _live_instr = _step.get("instruction")
                        if (_live_instr
                                and isinstance(_live_instr, str)
                                and _live_instr != task.get("description")):
                            task["description"] = _live_instr
                            _changed_fields.append("description")

                        # Top-level step fields that the engine plumbs
                        # through expander.py into task context — refresh
                        # the same set so retries see live config.
                        _CTX_FIELDS = (
                            "done_when",
                            "input_artifacts",
                            "output_artifacts",
                            "artifact_schema",
                            "tools_hint",
                            "difficulty",
                        )
                        for _f in _CTX_FIELDS:
                            _live_val = _step.get(_f)
                            if _live_val is None:
                                continue
                            if _task_ctx.get(_f) != _live_val:
                                _task_ctx[_f] = _live_val
                                _changed_fields.append(_f)

                        # The step may declare a free-form "context" dict
                        # (estimated_output_tokens, may_need_clarification,
                        # triggers_clarification, custom keys). Merge it
                        # so additions / edits flow into live tasks.
                        _step_inner_ctx = _step.get("context") or {}
                        if isinstance(_step_inner_ctx, dict):
                            for _k, _v in _step_inner_ctx.items():
                                if _task_ctx.get(_k) != _v:
                                    _task_ctx[_k] = _v
                                    _changed_fields.append(f"context.{_k}")

                        if _changed_fields:
                            task["context"] = json.dumps(_task_ctx)
                            logger.info(
                                f"[Task #{task.get('id','?')}] step-refresh: "
                                f"{', '.join(_changed_fields)} re-synced from "
                                f"live JSON (step={_step_id}, wf={_wf_name})"
                            )
                            # Persist to DB so the post-execute hook (which
                            # re-fetches the task via workflow_engine.advance)
                            # validates against the LIVE schema, not the stale
                            # snapshot stored at admission time. Without this
                            # write, mission 57 task 4450 (6.1) kept DLQ'ing on
                            # an empty-placeholder check for ``dependencies``
                            # because advance saw the old legacy schema even
                            # though base.py had refreshed in-memory.
                            try:
                                from src.infra.db import update_task as _update_task
                                await _update_task(task["id"], context=task["context"])
                            except Exception as _persist_exc:
                                logger.warning(
                                    f"[Task #{task.get('id','?')}] step-refresh "
                                    f"persist failed: {_persist_exc!r}"
                                )
            except Exception as _e:
                logger.warning(
                    f"[Task #{task.get('id','?')}] step config refresh failed: {_e}"
                )

        try:
            # ── Phase 5: execution pattern routing ──
            if self.execution_pattern == "single_shot":
                _result = await self.execute_single_shot(task)
            else:
                _result = await self._execute_react_loop(task)
            # Post-emit constrained-decoding pass: if the workflow step
            # declares a constrainable artifact_schema (object/array)
            # and the draft result is a non-empty completion, run a
            # single-shot fix-up call with response_format:json_schema
            # so required fields can't be silently dropped. Skips
            # markdown/string schemas and non-completed results.
            try:
                _result = await self._maybe_constrained_emit(task, _result)
            except Exception as _emit_exc:
                logger.warning(
                    f"[Task #{task.get('id','?')}] constrained_emit raised: "
                    f"{_emit_exc!r} — keeping draft result"
                )
            return _result
        finally:
            # Restore original allowed_tools if overridden by tools_hint
            if hasattr(self, '_original_allowed_tools'):
                self.allowed_tools = self._original_allowed_tools
                del self._original_allowed_tools

    async def _execute_react_loop(self, task: dict) -> dict:
        """ReAct loop with requirements-based model selection."""
        _start_time = time.time()
        task_id = task.get("id", "?")
        mission_id = task.get("mission_id")

        # ── Parse task context ──
        _task_ctx = task.get("context")
        if isinstance(_task_ctx, str):
            try:
                _task_ctx = json.loads(_task_ctx)
            except (json.JSONDecodeError, TypeError):
                _task_ctx = {}
        if not isinstance(_task_ctx, dict):
            _task_ctx = {}
        reqs = await self._build_model_requirements(task, _task_ctx)
        # Phase 9.2: Attach task_id for tracing in router
        reqs._task_id = int(task_id) if str(task_id).isdigit() else None

        # ── attempt checkpoint recovery ──
        start_iteration = 0
        checkpoint = None
        # Skip checkpoint when this is a validation/grader retry. Presence
        # of _schema_error means a prior attempt failed and the hooks /
        # apply layers just injected a fresh nudge into ctx. The checkpoint
        # holds messages from that prior (bad) attempt — resuming from it
        # uses the OLD user prompt and the model never sees the retry
        # feedback, so it regenerates the same truncated output every
        # iteration. Mission 46 task #2888 looped 15+ times on an
        # identical ~26k-char truncation before DLQ (2026-04-24). Force
        # a fresh _build_context so enriched retry prompt lands.
        if _task_ctx.get("_schema_error"):
            logger.info(
                f"[Task #{task_id}] retry context detected "
                f"(_schema_error present) — skipping checkpoint, "
                f"rebuilding prompt with enriched nudge"
            )
        elif task_id != "?":
            try:
                checkpoint = await load_task_checkpoint(task_id)
            except Exception as exc:
                logger.warning(
                    f"[Task #{task_id}] Checkpoint load failed: {exc}"
                )

        if checkpoint:
            messages = checkpoint.get("messages", [])
            start_iteration = checkpoint.get("iteration", 0)
            total_cost = checkpoint.get("total_cost", 0.0)
            used_model = checkpoint.get("used_model", "unknown")
            tools_used = checkpoint.get("tools_used", False)
            tools_used_names: set[str] = set(checkpoint.get("tools_used_names", []))
            _compat_retried = checkpoint.get("validation_retried", False)
            custom_validation_retried = _compat_retried
            task_type_validation_retried = _compat_retried
            format_corrections = checkpoint.get("format_corrections",
                                                checkpoint.get("format_retries", 0))
            completed_tool_ops: dict[str, str] = checkpoint.get(
                "completed_tool_ops", {}
            )

            # Restore reqs from checkpoint
            saved_reqs = checkpoint.get("reqs")
            if isinstance(saved_reqs, ModelRequirements):
                reqs = saved_reqs
            elif isinstance(saved_reqs, dict):
                # Checkpoint saved via dataclasses.asdict — reconstruct
                valid_fields = {f.name for f in dataclasses.fields(ModelRequirements)}
                reqs = ModelRequirements(
                    **{k: v for k, v in saved_reqs.items() if k in valid_fields}
                )
            else:
                # Very old checkpoint or missing — build fresh
                reqs = await self._build_model_requirements(task, _task_ctx)

            logger.info(
                f"[Task #{task_id}] Resuming from checkpoint "
                f"(iteration {start_iteration}, "
                f"{len(messages)} messages, ${total_cost:.4f} spent, "
                f"{len(completed_tool_ops)} cached tool ops)"
            )

            # Validate recovered _prev_output from checkpoint context
            _recovered_prev = _task_ctx.get("_prev_output")
            if _recovered_prev:
                from dogru_mu_samet import assess as cq_assess, salvage as cq_salvage
                _rec_cq = cq_assess(_recovered_prev)
                if _rec_cq.is_degenerate:
                    cleaned = cq_salvage(_recovered_prev)
                    if cleaned:
                        _task_ctx["_prev_output"] = cleaned
                    else:
                        _task_ctx.pop("_prev_output", None)
                    logger.info(
                        f"[Task #{task_id}] Checkpoint _prev_output was degenerate, "
                        f"{'salvaged' if cleaned else 'discarded'}"
                    )
        else:
            system_prompt = self._build_full_system_prompt(task)
            context = await self._build_context(task)

            logger.info(
                f"[Task #{task_id}] System prompt ({len(system_prompt)} chars):\n"
                f"{system_prompt}"
            )
            logger.info(
                f"[Task #{task_id}] User context ({len(context)} chars):\n"
                f"{context}"
            )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": context},
            ]

            total_cost = 0.0
            used_model = "unknown"
            tools_used = False
            tools_used_names: set[str] = set()
            custom_validation_retried = False
            task_type_validation_retried = False
            format_corrections = 0
            completed_tool_ops: dict[str, str] = {}

        consecutive_tool_failures = 0
        model_escalated = False

        _progress_last_sent = time.time()
        _search_depth = self._get_search_depth(task)
        _suppress_guards = _task_ctx.get("suppress_guards", False)

        # Dynamic iteration budget (retry boost from exhaustion handler)
        effective_max_iterations = self.max_iterations
        _boost = _task_ctx.get("iteration_budget_boost", 1.0)
        if _boost > 1.0:
            effective_max_iterations = min(int(self.max_iterations * _boost), 12)
            logger.info(
                f"[Task #{task_id}] Iteration budget boosted: "
                f"{self.max_iterations} → {effective_max_iterations}"
            )

        # Exhaustion tracking counters
        guard_burns = 0
        useful_iterations = 0
        empty_response_count = 0

        # Per-task cumulative tool-call tracker (reset per outer iteration
        # via ``_iter_tool_calls_seen``).  Used to dynamically strip tools
        # like read_file once the agent has already invoked them — prevents
        # re-reading artifacts already present in the blackboard/history.
        self._iter_tool_calls_seen: set[str] = set()

        try:
          # Budget is fresh per attempt. Checkpoint preserves messages/tool ops
          # for LLM context; it does not consume the new attempt's budget.
          for iteration in range(effective_max_iterations):
            # ── Check if task was cancelled while running ──
            if iteration > 0 and iteration % 2 == 0:
                try:
                    from ..infra.db import get_task as _get_task
                    _current = await _get_task(task_id)
                    if _current and _current.get("status") == "cancelled":
                        logger.info(f"[Task #{task_id}] Cancelled by user, aborting")
                        return {"status": "cancelled", "result": "Task cancelled by user"}
                except Exception:
                    pass

            logger.info(
                f"[Task #{task_id}] Agent '{self.name}' iteration "
                f"{iteration + 1}/{effective_max_iterations}"
            )
            # ── Accumulated messages telemetry ──
            # ReAct loop appends assistant+tool_result pairs each iteration;
            # tool outputs can be 0.1-50kB each. The pruner trims when over
            # 80% ctx, but a 90k-token call may originate from accumulated
            # messages rather than _build_context alone. Log size + role
            # breakdown so retries-after-many-iterations are visible.
            try:
                _msg_counts: dict[str, int] = {}
                _msg_chars: dict[str, int] = {}
                for _m in messages:
                    _role = _m.get("role", "?")
                    _c = _m.get("content") or ""
                    if not isinstance(_c, str):
                        _c = str(_c)
                    _msg_counts[_role] = _msg_counts.get(_role, 0) + 1
                    _msg_chars[_role] = _msg_chars.get(_role, 0) + len(_c)
                _total_chars = sum(_msg_chars.values())
                _breakdown = " ".join(
                    f"{r}={_msg_counts[r]}#/{_msg_chars[r]}c"
                    for r in sorted(_msg_chars.keys())
                )
                logger.info(
                    f"[Task #{task_id}] messages state iter "
                    f"{iteration + 1}: total={_total_chars}c "
                    f"n={len(messages)} {_breakdown}"
                )
            except Exception as _exc:
                logger.debug(f"messages-size telemetry failed: {_exc!r}")
            # Per-task progress signal — orchestrator's no-progress
            # watchdog (src/core/heartbeat.py) keys off these bumps to
            # decide whether the dispatch is wedged.
            try:
                from src.core import heartbeat as _hb
                _hb.bump(task_id)
            except Exception:
                pass

            # ── Phase 4.6: Progress streaming ──
            _now = time.time()
            if (self.progress_callback
                    and iteration > 0
                    and _now - _progress_last_sent >= 15):
                try:
                    # Summarize last action with meaningful context
                    _last_action = ""
                    for _m in reversed(messages):
                        if _m.get("role") == "assistant":
                            _content = _m.get("content") or ""
                            if _m.get("tool_calls"):
                                _tc = _m["tool_calls"][0]
                                _fn = _tc.get("function", {}).get("name", "tool")
                                _last_action = f"Using {_fn}..."
                            elif _content.lstrip().startswith(("{", "[", "```")):
                                import json as _pjson
                                import re as _re
                                # Strip markdown code fences before parsing
                                _raw = _re.sub(r"^```(?:json)?\s*|\s*```$", "", _content.strip())
                                try:
                                    _parsed = _pjson.loads(_raw)
                                    _action = _parsed.get("action", "")
                                    if _action == "tool_call":
                                        _tool = _parsed.get("tool", "tool")
                                        _last_action = f"Using {_tool}..."
                                    elif _action == "final_answer":
                                        _last_action = "Finalizing answer..."
                                    elif _action:
                                        _last_action = f"{_action.replace('_', ' ').capitalize()}..."
                                    else:
                                        _last_action = "Processing..."
                                except Exception:
                                    _last_action = "Processing..."
                            else:
                                # Show a snippet of the LLM's reasoning
                                _snippet = _content.strip()[:80]
                                _last_action = f"Thinking: {_snippet}..." if _snippet else "Processing..."
                            break
                    await self.progress_callback(
                        task_id, iteration + 1, effective_max_iterations, _last_action
                    )
                    _progress_last_sent = _now
                except Exception:
                    pass

            # ── Inner correction loop ──
            # Guards and format corrections are handled as sub-iterations
            # within the SAME outer iteration, so they don't burn iteration budget.
            sub_corrections = 0

            while sub_corrections <= MAX_SUB_CORRECTIONS:
                # ── Update token estimates ──
                estimation_model = used_model if used_model != "unknown" else "gpt-4o-mini"
                reqs.estimated_input_tokens = self._count_tokens(
                    messages, estimation_model
                )
                reqs.estimated_output_tokens = min(
                    reqs.estimated_output_tokens, 4096,
                )

                # ── Prune oldest tool-result pairs if estimate exceeds
                # context budget (cheap char/3 guard before the heavier
                # compression below).  llama-server silently truncates long
                # prompts otherwise — we prefer to drop oldest tool output
                # visibly and log it.
                try:
                    _ctx_win = self._get_context_window(estimation_model, reqs)
                    messages = self._prune_tool_results_to_fit(
                        messages,
                        ctx_window=_ctx_win,
                        estimated_output_tokens=reqs.estimated_output_tokens,
                        task_id=task_id,
                    )
                except Exception as _prune_exc:
                    logger.debug(f"[Task #{task_id}] prune skipped: {_prune_exc}")

                # ── Trim context ── (now accepts reqs directly)
                messages = self._trim_messages_if_needed(
                    messages, estimation_model, reqs,
                )

                # ── Tools ──
                # Hard guardrail: on the LAST iteration, strip all tools so the
                # LLM is forced to produce a text response (final_answer).
                # Small models ignore "LAST ITERATION" text warnings — this makes
                # it physically impossible to call tools on the final turn.
                #
                # Also strip tools when running low on time — local LLMs need
                # 120+ seconds to generate a full analysis.  Without this, the
                # agent wastes iterations on tool calls and then the task-level
                # timeout kills the final-answer LLM call mid-generation.
                is_last_iteration = (iteration + 1 >= effective_max_iterations)
                _elapsed = time.time() - _start_time
                _time_budget = getattr(self, '_task_timeout', 300)
                _remaining = _time_budget - _elapsed
                if not is_last_iteration and _remaining < 120 and iteration > 0:
                    logger.warning(
                        f"[Task #{task_id}] Forcing final answer: "
                        f"only {_remaining:.0f}s remaining (need 120s for answer)"
                    )
                    is_last_iteration = True
                if is_last_iteration:
                    litellm_tools = None
                    # Inject a system reminder that tools are gone
                    # (only on first sub-correction pass to avoid duplicates)
                    if sub_corrections == 0:
                        messages.append({
                            "role": "user",
                            "content": (
                                "FINAL ITERATION — no tools available. You MUST produce your "
                                "final answer NOW as plain text or JSON. Summarize everything "
                                "you have gathered so far."
                            ),
                        })
                else:
                    # Dynamic tool strip: if the agent already read a file
                    # earlier this task, drop ``read_file`` from the schema
                    # so the LLM reuses the blackboard/prior tool results
                    # instead of re-reading.  The cumulative set lives on
                    # ``self._iter_tool_calls_seen`` (reset per-task).
                    _exclude: set[str] = set()
                    if "read_file" in getattr(
                        self, "_iter_tool_calls_seen", set()
                    ):
                        _exclude.add("read_file")
                    litellm_tools = self._build_litellm_tools(exclude=_exclude)
                if litellm_tools:
                    reqs.needs_function_calling = True

                # ── Call LLM ──
                self._partial_content = ""
                try:
                    from src.core.llm_dispatcher import get_dispatcher, CallCategory
                    response = await get_dispatcher().request(
                        CallCategory.MAIN_WORK,
                        task=reqs.effective_task or reqs.primary_capability,
                        agent_type=reqs.agent_type,
                        difficulty=reqs.difficulty,
                        messages=messages,
                        tools=litellm_tools,
                        needs_thinking=reqs.needs_thinking,
                        needs_function_calling=reqs.needs_function_calling,
                        needs_vision=reqs.needs_vision,
                        local_only=reqs.local_only,
                        prefer_speed=reqs.prefer_speed,
                        prefer_quality=reqs.prefer_quality,
                        prefer_local=reqs.prefer_local,
                        estimated_input_tokens=reqs.estimated_input_tokens,
                        estimated_output_tokens=reqs.estimated_output_tokens,
                        min_context=reqs.effective_context_needed,
                        priority=reqs.priority,
                        exclude_models=reqs.exclude_models or [],
                        remaining_budget=max(0.0, _remaining),
                        preselected_pick=task.get("preselected_pick") if iteration == 0 else None,
                        task_obj=task,
                        iteration_n=iteration,
                    )
                except Exception as exc:
                    # Let ModelCallFailed propagate — the orchestrator handles
                    # it as an availability failure with backoff + wake signals.
                    from src.core.router import ModelCallFailed
                    _NON_RETRYABLE = (ModelCallFailed, AttributeError, TypeError,
                                      ImportError, NameError, KeyError)
                    if isinstance(exc, _NON_RETRYABLE):
                        raise
                    logger.error(f"[Task #{task_id}] Model call failed: {exc}")
                    return {
                        "status": "failed",
                        "result": f"Agent failed after {iteration} iteration(s): {exc}",
                        "error": str(exc),
                        "model": used_model,
                        "cost": total_cost,
                        "iterations": iteration,
                        "difficulty": reqs.difficulty,
                    }

                content    = response.get("content", "")
                used_model = response.get("model", used_model)
                step_cost  = response.get("cost", 0)
                step_latency = response.get("latency", 0)
                total_cost += step_cost

                try:
                    await record_model_call(
                        model=used_model,
                        agent_type=self.name,
                        success=True,
                        cost=step_cost,
                        latency=step_latency,
                    )
                except Exception:
                    pass

                if step_cost > 0:
                    try:
                        await record_cost(step_cost)
                    except Exception:
                        pass

                logger.info(f"[Task #{task_id}] Raw response ({len(content)} chars):\n{content}")

                # Skip empty responses — don't burn an iteration on nothing.
                if not content and not response.get("tool_calls"):
                    logger.warning(
                        f"[Task #{task_id}] Empty response (0 chars, no tool_calls) "
                        f"— not counting as iteration {iteration + 1}/{effective_max_iterations}"
                    )
                    empty_response_count += 1
                    if empty_response_count >= 3:
                        return {
                            "status": "failed",
                            "error": f"Model returned {empty_response_count} consecutive empty responses",
                            "model": used_model,
                            "cost": total_cost,
                        }
                    continue  # retry same iteration
                empty_response_count = 0  # reset on non-empty

                await self._safe_log(
                    task_id, "assistant", content, used_model, step_cost
                )

                # ── Parse response ──
                fc_tool_calls = response.get("tool_calls")
                parsed = None
                if fc_tool_calls:
                    parsed = self._parse_function_call_response(fc_tool_calls)
                if parsed is None:
                    parsed = self._parse_agent_response(content)

                # ── FORMAT CORRECTION (sub-iteration) ──
                if parsed is None:
                    # If the response is substantial natural-language text
                    # (not a raw tool call or envelope), accept as final
                    # answer rather than wasting a correction on format.
                    _looks_like_tool = any(
                        marker in content[:100]
                        for marker in (
                            '"tool_call"', '"tool":', "<|function_call|>",
                            "```tool_code", '"write_file"', '"read_file"',
                        )
                    )
                    if len(content) > 200 and not _looks_like_tool:
                        result_text = _unwrap_final_answer(content)
                        logger.info(
                            f"[Task #{task_id}] Accepting unparsed response "
                            f"as final answer ({len(content)} chars)"
                        )
                        parsed = {"action": "final_answer", "result": result_text}
                    elif format_corrections < MAX_FORMAT_CORRECTIONS and sub_corrections < MAX_SUB_CORRECTIONS:
                        format_corrections += 1
                        sub_corrections += 1
                        logger.warning(
                            f"[Task #{task_id}] JSON parse failed — "
                            f"format-correction {format_corrections}/{MAX_FORMAT_CORRECTIONS}"
                        )
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your response could not be parsed as valid JSON. "
                                "Please fix the formatting and try again.\n\n"
                                "You MUST respond with ONLY a valid JSON block:\n"
                                "```json\n"
                                '{"action": "tool_call", "tool": "...", "args": {...}}\n'
                                "```\nor:\n```json\n"
                                '{"action": "final_answer", "result": "..."}\n'
                                "```\nNo text before or after the JSON block."
                            ),
                        })
                        continue  # inner loop — re-prompt LLM
                    else:
                        parsed = {
                            "action": "final_answer",
                            "result": (
                                f"[Parse failure] Agent could not produce valid "
                                f"JSON after {MAX_FORMAT_CORRECTIONS} format corrections. "
                                f"Raw output:\n{content[:2000]}"
                            ),
                        }

                try:
                    parsed = validate_action(parsed)
                except ValueError as exc:
                    logger.warning(f"[Task #{task_id}] Action validation warning: {exc}")

                # ── SUB-ITERATION GUARD CHECK ──
                correction = self._check_sub_iteration_guards(
                    parsed=parsed,
                    iteration=iteration,
                    tools_used=tools_used,
                    tools_used_names=tools_used_names,
                    task=task,
                    search_depth=_search_depth,
                    suppress_guards=_suppress_guards,
                )
                if correction and sub_corrections < MAX_SUB_CORRECTIONS:
                    guard_burns += 1
                    logger.warning(
                        f"[Task #{task_id}] [{correction.guard_name}] "
                        f"sub-correction {sub_corrections + 1}/{MAX_SUB_CORRECTIONS}"
                    )
                    await self._safe_log(
                        task_id, "system",
                        f"[{correction.guard_name}] sub-correction",
                        None, 0,
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": correction.message})
                    sub_corrections += 1
                    continue  # inner loop — re-prompt LLM

                # ── CUSTOM VALIDATION (sub-iteration) ──
                action_type = parsed.get("action", "final_answer")
                if action_type == "final_answer":
                    result = parsed.get("result", content)
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False, indent=2)

                    if not custom_validation_retried and sub_corrections < MAX_SUB_CORRECTIONS:
                        validation_error = self._validate_response(result, task)
                        if validation_error:
                            custom_validation_retried = True
                            sub_corrections += 1
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user",
                                "content": f"{validation_error}\n\nPlease try again.",
                            })
                            continue  # inner loop — re-prompt LLM

                    # ── TASK-TYPE VALIDATION (sub-iteration) ──
                    task_type_errors = validate_task_output(self.name, result)
                    if task_type_errors and not task_type_validation_retried and sub_corrections < MAX_SUB_CORRECTIONS:
                        task_type_validation_retried = True
                        err_msg = "; ".join(task_type_errors)
                        sub_corrections += 1
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"Output quality issue: {err_msg}\n\nPlease revise.",
                        })
                        continue  # inner loop — re-prompt LLM

                break  # No guard/correction fired → proceed to action handling

            # Save checkpoint after inner loop completes
            await self._save_checkpoint(
                task_id, iteration + 1, messages, total_cost,
                used_model, reqs, tools_used,
                custom_validation_retried or task_type_validation_retried,
                completed_tool_ops, format_corrections,
                tools_used_names,
            )

            action_type = parsed.get("action", "final_answer")

            # ── FINAL ANSWER ──
            if action_type == "final_answer":
                result = parsed.get("result", content)
                # Ensure result is always a string — LLMs sometimes return
                # a dict/list as the result value instead of text.
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False, indent=2)

                # Memories
                raw_memories = parsed.get("memories", {})
                if raw_memories and isinstance(raw_memories, dict):
                    for key, value in raw_memories.items():
                        try:
                            await store_memory(
                                key, str(value),
                                category=self.name, mission_id=mission_id,
                            )
                        except Exception as exc:
                            logger.warning(f"store_memory failed: {exc}")

                logger.info(
                    f"[Task #{task_id}] ✅ Agent answered after "
                    f"{iteration + 1} iteration(s)"
                )

                # Debug: show what keys the parsed response has
                parsed_keys = list(parsed.keys()) if isinstance(parsed, dict) else "not-dict"
                has_subtasks = bool(parsed.get("subtasks")) if isinstance(parsed, dict) else False
                logger.debug(
                    f"[Task #{task_id}] Parsed keys: {parsed_keys}, "
                    f"has_subtasks={has_subtasks}, "
                    f"can_create_subtasks={self.can_create_subtasks}"
                )

                # Normalize subtask keys — LLMs sometimes use "tasks",
                # "steps", "plan" instead of "subtasks"
                subtasks = parsed.get("subtasks")
                if not subtasks and self.can_create_subtasks:
                    for alt_key in ("tasks", "steps", "plan", "sub_tasks"):
                        candidate = parsed.get(alt_key)
                        if isinstance(candidate, list) and candidate:
                            subtasks = candidate
                            logger.debug(
                                f"[Task #{task_id}] Found subtasks under "
                                f"alt key '{alt_key}' ({len(candidate)} items)"
                            )
                            break

                # Workflow steps must produce output directly — never delegate
                _is_wf = bool(_task_ctx.get("is_workflow_step"))
                if subtasks and not _is_wf:
                    return {
                        "status":       "needs_subtasks",
                        "subtasks":     subtasks,
                        "plan_summary": parsed.get("plan_summary", ""),
                        "model":        used_model,
                        "cost":         total_cost,
                        "difficulty":   reqs.difficulty,
                    }
                elif subtasks and _is_wf:
                    # Agent tried to create subtasks in a workflow step —
                    # treat the plan_summary or result as the actual output
                    logger.warning(
                        f"[Task #{task_id}] Planner tried to create subtasks "
                        f"in workflow step — using result directly"
                    )
                    result = parsed.get("plan_summary") or parsed.get("result", result)

                if parsed.get("needs_clarification"):
                    return {
                        "status":        "needs_clarification",
                        "clarification": parsed["needs_clarification"],
                        "model":         used_model,
                        "cost":          total_cost,
                        "difficulty":    reqs.difficulty,
                    }

                # Self-reflection
                if self.enable_self_reflection:
                    try:
                        reflection = await self._self_reflect(
                            task, result, reqs, used_model,
                        )
                        if reflection and reflection.get("verdict") == "fix":
                            corrected = reflection.get("corrected_result")
                            if corrected:
                                result = corrected
                    except Exception as exc:
                        logger.debug(f"Self-reflection error: {exc}")

                # Confidence gating
                confidence = parsed.get("confidence")
                if (
                    self.min_confidence > 0
                    and isinstance(confidence, (int, float))
                    and confidence < self.min_confidence
                ):
                    return {
                        "status":      "needs_review",
                        "result":      result,
                        "review_note": f"Agent confidence: {confidence}/5",
                        "model":       used_model,
                        "cost":        total_cost,
                        "difficulty":  reqs.difficulty,
                    }

                return {
                    "status":           "completed",
                    "result":           result,
                    "model":            used_model,
                    "cost":             total_cost,
                    "difficulty":       reqs.difficulty,
                    "iterations":       iteration + 1,
                    "tools_used_names": sorted(tools_used_names),
                    "generating_model": used_model,  # surfaced for post-hook scheduling
                }

            # ── TOOL CALL ──
            if action_type == "tool_call":
                tools_used = True
                tool_name = parsed.get("tool", "")
                tools_used_names.add(tool_name)
                try:
                    self._iter_tool_calls_seen.add(tool_name)
                except AttributeError:
                    self._iter_tool_calls_seen = {tool_name}
                tool_args = parsed.get("args", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                # ── Intercept read_file for artifacts already in context ──
                # Workflow steps inject input artifacts into the prompt.
                # If the model tries to read_file for one of them, short-
                # circuit with a nudge — saves an iteration + tool call.
                if tool_name == "read_file" and _task_ctx.get("is_workflow_step"):
                    _input_arts = _task_ctx.get("input_artifacts", [])
                    _filepath = tool_args.get("filepath", "")
                    _matched = any(
                        art in _filepath for art in _input_arts
                    ) if _input_arts and _filepath else False
                    if _matched:
                        tool_output = (
                            "This artifact is already included in your "
                            "context above under '## Context Artifacts'. "
                            "Use it directly — do NOT call read_file for it."
                        )
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"Tool Result ({tool_name}):\n{tool_output}",
                        })
                        await self._save_checkpoint(
                            task_id, iteration + 1, messages, total_cost,
                            used_model, reqs, tools_used,
                            custom_validation_retried or task_type_validation_retried,
                            completed_tool_ops, format_corrections,
                            tools_used_names,
                        )
                        continue

                if (
                    self.allowed_tools is not None
                    and tool_name not in self.allowed_tools
                ):
                    tool_output = (
                        f"❌ Tool '{tool_name}' not available. "
                        f"Allowed: {self.allowed_tools}"
                    )
                elif not self._check_tool_permission(tool_name):
                    tool_output = (
                        f"🚫 Tool '{tool_name}' not permitted for agent "
                        f"type '{self.name}' (security policy)."
                    )
                    logger.warning(
                        f"[Task #{task_id}] Permission denied: "
                        f"{self.name} → {tool_name}"
                    )
                elif tool_name not in TOOL_REGISTRY:
                    tool_output = (
                        f"❌ Unknown tool '{tool_name}'. "
                        f"Available: {list(TOOL_REGISTRY.keys())}"
                    )
                else:
                    arg_schema = _TOOL_SCHEMAS_BY_NAME.get(tool_name)
                    if arg_schema:
                        tool_args, arg_errors = validate_tool_args(
                            tool_name, tool_args, arg_schema,
                        )
                        if arg_errors:
                            err_msg = "; ".join(arg_errors)
                            tool_output = (
                                f"❌ Argument error for '{tool_name}': {err_msg}\n\n"
                                f"Expected: {json.dumps(arg_schema, indent=2)}"
                            )
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": tool_output})
                            await self._save_checkpoint(
                                task_id, iteration + 1, messages, total_cost,
                                used_model, reqs, tools_used,
                                custom_validation_retried or task_type_validation_retried,
                                completed_tool_ops, format_corrections,
                            )
                            continue

                    idem_key = self._tool_idempotency_key(tool_name, tool_args)

                    # Check caches: side-effect idempotency OR read-only result cache
                    cached = None
                    if tool_name in SIDE_EFFECT_TOOLS:
                        cached = completed_tool_ops.get(idem_key)
                    elif tool_name in CACHEABLE_READ_TOOLS:
                        cached = completed_tool_ops.get(f"rc:{idem_key}")

                    if cached is not None:
                        tool_output = cached
                        logger.debug(f"[Task #{task_id}] cache hit: {tool_name}")
                    else:
                        logger.info(
                            f"[Task #{task_id}] \U0001f527 {tool_name}("
                            f"{', '.join(f'{k}={repr(v)[:50]}' for k, v in tool_args.items())})"
                        )
                        try:
                            # Per-tool timeout: 120s for shell tools, 60s for others.
                            # Prevents a single hung tool from blocking the agent loop.
                            _tool_timeout = 120 if tool_name in (
                                "shell", "shell_stdin", "shell_sequential",
                            ) else 60
                            # Build task hints for context-aware tools
                            _hints = {
                                "agent_type": self.name,
                                "search_depth": self._get_search_depth(task),
                                "shopping_sub_intent": task.get("shopping_sub_intent"),
                                "workspace_path": _task_ctx.get("workspace_path", ""),
                            }

                            tool_output = await asyncio.wait_for(
                                execute_tool(
                                    tool_name, agent_type=self.name, task_hints=_hints, **tool_args
                                ),
                                timeout=_tool_timeout,
                            )
                        except asyncio.TimeoutError:
                            tool_output = (
                                f"\u274c Tool '{tool_name}' timed out after "
                                f"{_tool_timeout}s — try a simpler approach."
                            )
                        except Exception as exc:
                            tool_output = f"\u274c Tool execution error: {exc}"

                        # Phase 8.4: Audit log tool execution
                        try:
                            from ..infra.audit import audit, ACTOR_AGENT, ACTION_TOOL_EXEC
                            _tid = int(task_id) if str(task_id).isdigit() else None
                            await audit(
                                actor=f"{ACTOR_AGENT}:{self.name}",
                                action=ACTION_TOOL_EXEC,
                                target=tool_name,
                                details=str(tool_args)[:500],
                                task_id=_tid,
                                mission_id=mission_id,
                            )
                        except Exception:
                            pass

                        # Phase 9.2: Trace tool execution
                        try:
                            _tid = int(task_id) if str(task_id).isdigit() else None
                            if _tid:
                                from ..infra.tracing import append_trace
                                await append_trace(
                                    task_id=_tid,
                                    entry_type="tool",
                                    input_summary=f"{tool_name}({', '.join(f'{k}={repr(v)[:30]}' for k, v in tool_args.items())})",
                                    output_summary=tool_output[:200] if tool_output else "",
                                )
                        except Exception:
                            pass

                        # Phase 9.1: Record tool call metric
                        try:
                            from ..infra.metrics import record_tool_call
                            record_tool_call(tool=tool_name)
                        except Exception:
                            pass

                        # Cache results
                        if tool_name in SIDE_EFFECT_TOOLS:
                            completed_tool_ops[idem_key] = tool_output
                            # Invalidate read-only cache on side effects
                            _to_remove = [k for k in completed_tool_ops if k.startswith("rc:")]
                            for k in _to_remove:
                                del completed_tool_ops[k]

                            # Phase E: Post-tool reindexing for file-modifying tools
                            if tool_name in ("write_file", "edit_file", "patch_file", "apply_diff"):
                                _target_file = tool_args.get("filepath", tool_args.get("path", ""))
                                if _target_file:
                                    try:
                                        from ..parsing.code_embeddings import post_tool_reindex
                                        _repo = context.get("repo_path", "") if isinstance(context, dict) else ""
                                        await post_tool_reindex(_target_file, root_path=_repo)
                                    except Exception:
                                        pass
                        elif tool_name in CACHEABLE_READ_TOOLS:
                            completed_tool_ops[f"rc:{idem_key}"] = tool_output

                if len(tool_output) > MAX_TOOL_OUTPUT_LENGTH:
                    tool_output = (
                        tool_output[:MAX_TOOL_OUTPUT_LENGTH]
                        + f"\n\n... [{len(tool_output)} chars total]"
                    )

                tool_failed = (
                    tool_output.startswith("❌")
                    or tool_output.startswith("🚫")
                    or "command not found" in tool_output
                    or "No such file" in tool_output
                    or ("exit code" in tool_output and "exit code 0" not in tool_output)
                )

                if tool_failed:
                    consecutive_tool_failures += 1
                else:
                    consecutive_tool_failures = 0
                    useful_iterations += 1

                # ── Mid-task escalation ── (NOW uses reqs.escalate())
                if (
                    not model_escalated
                    and consecutive_tool_failures >= TOOL_FAILURE_ESCALATION_THRESHOLD
                    and iteration >= TOOL_FAILURE_ESCALATION_THRESHOLD
                ):
                    old_tier = reqs.difficulty
                    reqs = self._escalate_requirements(reqs)
                    new_tier = reqs.difficulty
                    if new_tier != old_tier:
                        logger.warning(
                            f"[Task #{task_id}] ⬆️ model-escalation: "
                            f"'{old_tier}' → '{new_tier}' after "
                            f"{consecutive_tool_failures} consecutive failures"
                        )
                        model_escalated = True
                        await self._safe_log(
                            task_id, "system",
                            f"[escalation] Upgraded quality after "
                            f"{consecutive_tool_failures} failures",
                            None, 0,
                        )
                        # Reset context for the better model
                        messages = self._trim_for_escalation(
                            messages, iteration, effective_max_iterations,
                        )

                if tool_failed:
                    recovery_guidance = (
                        f"## Tool Result (`{tool_name}`) — ERROR:\n\n"
                        f"```\n{tool_output}\n```\n\n"
                        f"The tool call failed. Try a DIFFERENT approach.\n"
                        f"Iteration {iteration + 2}/{effective_max_iterations}."
                    )
                else:
                    recovery_guidance = (
                        f"## Tool Result (`{tool_name}`):\n\n"
                        f"```\n{tool_output}\n```\n\n"
                        f"{'LAST ITERATION — you MUST respond with final_answer now. Do NOT call any more tools.' if iteration + 2 >= effective_max_iterations else 'Continue working.'} Iteration {iteration + 2}/{effective_max_iterations}."
                    )

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": recovery_guidance})

                await self._safe_log(
                    task_id, "tool",
                    f"[{tool_name}] {tool_output[:2000]}",
                    None, 0,
                )
                await self._save_checkpoint(
                    task_id, iteration + 1, messages, total_cost,
                    used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                    completed_tool_ops, format_corrections,
                )
                continue

            # ── MULTI TOOL CALL (parallel read-only, sequential side-effect) ──
            if action_type == "multi_tool_call":
                tools_used = True
                tool_list = parsed.get("tools", [])

                # Validate each tool call
                validated: list[tuple[str, dict, str | None]] = []
                for tc in tool_list:
                    t_name = tc.get("tool", "")
                    t_args = tc.get("args", {})
                    if not isinstance(t_args, dict):
                        t_args = {}
                    tools_used_names.add(t_name)
                    try:
                        self._iter_tool_calls_seen.add(t_name)
                    except AttributeError:
                        self._iter_tool_calls_seen = {t_name}

                    if self.allowed_tools is not None and t_name not in self.allowed_tools:
                        validated.append((t_name, t_args, f"❌ Tool '{t_name}' not available."))
                    elif not self._check_tool_permission(t_name):
                        validated.append((t_name, t_args, f"🚫 Tool '{t_name}' not permitted."))
                    elif t_name not in TOOL_REGISTRY:
                        validated.append((t_name, t_args, f"❌ Unknown tool '{t_name}'."))
                    else:
                        validated.append((t_name, t_args, None))

                to_execute = [(n, a) for n, a, err in validated if err is None]
                errors = [(n, err) for n, _, err in validated if err is not None]

                parallel_group, sequential_group = _partition_tool_calls(
                    [{"tool": n, "args": a} for n, a in to_execute]
                )

                results: list[tuple[str, dict, str]] = []

                # --- Parallel group (read-only) ---
                if parallel_group:
                    async def _exec_one(tc_item: dict) -> tuple[str, dict, str]:
                        _tn, _ta = tc_item["tool"], tc_item["args"]
                        _timeout = 120 if _tn in ("shell", "shell_stdin", "shell_sequential") else 60
                        _hints = {
                            "agent_type": self.name,
                            "search_depth": _search_depth,
                            "shopping_sub_intent": task.get("shopping_sub_intent"),
                            "workspace_path": _task_ctx.get("workspace_path", ""),
                        }
                        try:
                            out = await asyncio.wait_for(
                                execute_tool(_tn, agent_type=self.name, task_hints=_hints, **_ta),
                                timeout=_timeout,
                            )
                        except asyncio.TimeoutError:
                            out = f"❌ Tool '{_tn}' timed out after {_timeout}s"
                        except Exception as exc:
                            out = f"❌ Tool execution error: {exc}"
                        return _tn, _ta, out

                    par_results = await asyncio.gather(
                        *[_exec_one(tc_item) for tc_item in parallel_group],
                        return_exceptions=True,
                    )
                    for r in par_results:
                        if isinstance(r, Exception):
                            results.append(("unknown", {}, f"❌ Parallel error: {r}"))
                        else:
                            results.append(r)

                # --- Sequential group (side-effect) ---
                for tc_item in sequential_group:
                    _tn, _ta = tc_item["tool"], tc_item["args"]
                    _timeout = 120 if _tn in ("shell", "shell_stdin", "shell_sequential") else 60
                    _hints = {
                        "agent_type": self.name,
                        "search_depth": _search_depth,
                        "shopping_sub_intent": task.get("shopping_sub_intent"),
                        "workspace_path": _task_ctx.get("workspace_path", ""),
                    }
                    try:
                        out = await asyncio.wait_for(
                            execute_tool(_tn, agent_type=self.name, task_hints=_hints, **_ta),
                            timeout=_timeout,
                        )
                    except asyncio.TimeoutError:
                        out = f"❌ Tool '{_tn}' timed out after {_timeout}s"
                    except Exception as exc:
                        out = f"❌ Tool execution error: {exc}"
                    results.append((_tn, _ta, out))

                # Add pre-validation errors
                for n, err in errors:
                    results.append((n, {}, err))

                # Audit, metrics, caching per tool result
                for t_name, t_args, t_output in results:
                    # Audit log
                    try:
                        from ..infra.audit import audit, ACTOR_AGENT, ACTION_TOOL_EXEC
                        _tid = int(task_id) if str(task_id).isdigit() else None
                        await audit(
                            actor=f"{ACTOR_AGENT}:{self.name}",
                            action=ACTION_TOOL_EXEC,
                            target=t_name,
                            details=str(t_args)[:500],
                            task_id=_tid,
                            mission_id=mission_id,
                        )
                    except Exception:
                        pass

                    # Metrics
                    try:
                        from ..infra.metrics import record_tool_call
                        record_tool_call(tool=t_name)
                    except Exception:
                        pass

                    # Cache results
                    if t_name in SIDE_EFFECT_TOOLS:
                        idem_key = self._tool_idempotency_key(t_name, t_args)
                        completed_tool_ops[idem_key] = t_output
                        _to_remove = [k for k in completed_tool_ops if k.startswith("rc:")]
                        for k in _to_remove:
                            del completed_tool_ops[k]
                    elif t_name in CACHEABLE_READ_TOOLS:
                        idem_key = self._tool_idempotency_key(t_name, t_args)
                        completed_tool_ops[f"rc:{idem_key}"] = t_output

                # Build combined result message
                result_parts = []
                tool_failures = 0
                for t_name, t_args, t_output in results:
                    if len(t_output) > MAX_TOOL_OUTPUT_LENGTH:
                        t_output = (
                            t_output[:MAX_TOOL_OUTPUT_LENGTH]
                            + f"\n\n... [{len(t_output)} chars total]"
                        )
                    key_arg = next(iter(t_args.values()), "") if t_args else ""
                    if isinstance(key_arg, str) and len(key_arg) > 60:
                        key_arg = key_arg[:60]
                    result_parts.append(
                        f"## Tool Result (`{t_name}` → {key_arg}):\n\n"
                        f"```\n{t_output}\n```"
                    )
                    if t_output.startswith("❌") or t_output.startswith("🚫"):
                        tool_failures += 1

                if tool_failures > 0:
                    consecutive_tool_failures += tool_failures
                else:
                    consecutive_tool_failures = 0
                    useful_iterations += 1

                is_next_last = (iteration + 2 >= effective_max_iterations)
                combined = "\n\n".join(result_parts)
                combined += (
                    f"\n\n{'LAST ITERATION — you MUST respond with final_answer now.' if is_next_last else 'Continue working.'}"
                    f" Iteration {iteration + 2}/{effective_max_iterations}."
                )

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": combined})

                await self._safe_log(
                    task_id, "tool",
                    f"[multi:{len(results)} tools] {', '.join(n for n, _, _ in results)}",
                    None, 0,
                )
                await self._save_checkpoint(
                    task_id, iteration + 1, messages, total_cost,
                    used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                    completed_tool_ops, format_corrections,
                )
                continue

            # ── ASK AGENT (inter-agent query) ──
            if action_type == "ask_agent":
                import asyncio as _asyncio
                target_type = parsed.get("target", "researcher")
                question = parsed.get("question", "")
                logger.info(
                    f"[Task #{task_id}] 🤝 ask_agent → {target_type}: "
                    f"{question[:80]}"
                )
                try:
                    from ..agents import get_agent as _get_agent
                    target_agent = _get_agent(target_type)
                    inline_task = {
                        "id": f"{task_id}_inline_{iteration}",
                        "title": f"[Inline query from {self.name}]",
                        "description": question,
                        "mission_id": mission_id,
                        "context": json.dumps({"tool_depth": 1}),
                    }
                    inline_result = await _asyncio.wait_for(
                        target_agent.execute(inline_task), timeout=300
                    )
                    agent_answer = inline_result.get("result", "(no answer)")
                    agent_cost = inline_result.get("cost", 0)
                    total_cost += agent_cost
                    tool_output = (
                        f"## Answer from {target_type} agent:\n\n{agent_answer}"
                    )
                    logger.info(
                        f"[Task #{task_id}] ✅ ask_agent from {target_type} "
                        f"completed (${agent_cost:.4f})"
                    )
                except _asyncio.TimeoutError:
                    tool_output = f"❌ ask_agent timeout: {target_type} did not respond within 5 minutes"
                    logger.warning(f"[Task #{task_id}] ask_agent timeout ({target_type})")
                except Exception as exc:
                    tool_output = f"❌ ask_agent error: {exc}"
                    logger.warning(f"[Task #{task_id}] ask_agent error: {exc}")

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"{tool_output}\n\n"
                        f"{'LAST ITERATION — you MUST respond with final_answer now. Do NOT call any more tools.' if iteration + 2 >= effective_max_iterations else 'Continue working.'} Iteration {iteration + 2}/{effective_max_iterations}."
                    ),
                })
                await self._save_checkpoint(
                    task_id, iteration + 1, messages, total_cost,
                    used_model, reqs, tools_used,
                    custom_validation_retried or task_type_validation_retried,
                    completed_tool_ops, format_corrections,
                )
                continue

            # ── CLARIFY / DECOMPOSE / UNKNOWN ──
            # NOTE: Blocked clarification (suppress_clarification=True) is now
            # handled as a sub-iteration guard — see _check_sub_iteration_guards.
            # Only the non-suppressed return path remains here.
            if action_type == "clarify":
                return {
                    "status": "needs_clarification",
                    "clarification": parsed.get("question", content),
                    "model": used_model, "cost": total_cost, "difficulty": reqs.difficulty,
                }

            if action_type == "decompose":
                return {
                    "status": "needs_subtasks",
                    "subtasks": parsed.get("subtasks", []),
                    "plan_summary": parsed.get("summary", ""),
                    "model": used_model, "cost": total_cost, "difficulty": reqs.difficulty,
                }

            # Unknown action
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"ERROR: Unrecognized action '{action_type}'. "
                    f"Use tool_call or final_answer only.\n\n"
                    f"```json\n"
                    f'{{"action": "tool_call", "tool": "shell", "args": {{"command": "ls"}}}}\n'
                    f"```\nor:\n```json\n"
                    f'{{"action": "final_answer", "result": "your answer"}}\n'
                    f"```"
                ),
            })
            await self._save_checkpoint(
                task_id, iteration + 1, messages, total_cost,
                used_model, reqs, tools_used, custom_validation_retried or task_type_validation_retried,
                completed_tool_ops, format_corrections,
            )

        except asyncio.CancelledError:
            # Timeout — save checkpoint so previous work survives for retry.
            # Include any partial LLM output captured via streaming.
            try:
                partial = getattr(self, "_partial_content", "")
                if partial and len(partial) > 50:
                    messages.append({"role": "assistant", "content": partial})
                    logger.info(
                        f"[Task #{task_id}] Saved {len(partial)} chars of "
                        f"partial LLM output from interrupted generation"
                    )
                await self._save_checkpoint(
                    task_id, iteration, messages, total_cost,
                    used_model, reqs, tools_used,
                    False, completed_tool_ops, format_corrections,
                    tools_used_names,
                )
                logger.info(
                    f"[Task #{task_id}] Timeout checkpoint saved at iteration {iteration}"
                )
            except Exception:
                pass
            raise  # re-raise so orchestrator's TimeoutError handler fires

        # ── Exhausted iterations ──
        # Checkpoint is preserved here — orchestrator will clear it on final completion.

        # Classify exhaustion reason
        if guard_burns >= effective_max_iterations * 0.5:
            exhaustion_reason = "guards"
        elif consecutive_tool_failures >= TOOL_FAILURE_ESCALATION_THRESHOLD:
            exhaustion_reason = "tool_failures"
        else:
            exhaustion_reason = "budget"

        # Extract last meaningful assistant response for the result.
        # Do NOT truncate before unwrapping — truncation breaks JSON parsing.
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = msg["content"]
                break
        # Try to parse as JSON and extract "result" field — the LLM often
        # wraps its answer in {"action": "final_answer", "result": "..."}
        if last_assistant:
            parsed_final = self._parse_agent_response(last_assistant)
            if parsed_final and parsed_final.get("result"):
                last_assistant = parsed_final["result"]
            elif '"result"' in last_assistant and '"final_answer"' in last_assistant:
                # JSON parse failed (truncated by context trimming?) — regex fallback
                import re as _re
                m = _re.search(r'"result"\s*:\s*"((?:[^"\\]|\\.)*)', last_assistant)
                if m:
                    try:
                        last_assistant = m.group(1).encode().decode('unicode_escape')
                    except Exception:
                        last_assistant = m.group(1)
        # Truncate AFTER unwrapping — preserve the actual content.
        # 8000 chars is well above _SUMMARY_THRESHOLD (3000) so the post-hook
        # will always create a summary for large artifacts.
        if len(last_assistant) > 8000:
            last_assistant = last_assistant[:8000]

        logger.warning(
            f"[Task #{task_id}] Exhausted iterations | "
            f"reason={exhaustion_reason} "
            f"guard_burns={guard_burns} "
            f"useful={useful_iterations}/{effective_max_iterations}"
        )

        return {
            "status": "exhausted",
            "result": last_assistant or "",
            "exhaustion_reason": exhaustion_reason,
            "guard_burns": guard_burns,
            "useful_iterations": useful_iterations,
            "model": used_model,
            "cost": total_cost,
            "difficulty": reqs.difficulty,
            "iterations": effective_max_iterations,
            "tools_used_names": sorted(tools_used_names),
        }

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
        """Single LLM call with no tool loop. For planning/classification."""
        task_id = task.get("id", "?")

        _ss_ctx = task.get("context")
        if isinstance(_ss_ctx, str):
            try:
                _ss_ctx = json.loads(_ss_ctx)
            except (json.JSONDecodeError, TypeError):
                _ss_ctx = {}
        if not isinstance(_ss_ctx, dict):
            _ss_ctx = {}

        # Build requirements using the same method as react loop
        reqs = await self._build_model_requirements(task, _ss_ctx)

        system_prompt = self._build_full_system_prompt(task)
        context = await self._build_context(task)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": context},
        ]

        try:
            from src.core.llm_dispatcher import get_dispatcher, CallCategory
            response = await get_dispatcher().request(
                CallCategory.MAIN_WORK,
                task=reqs.effective_task or reqs.primary_capability,
                agent_type=reqs.agent_type,
                difficulty=reqs.difficulty,
                messages=messages,
                needs_thinking=reqs.needs_thinking,
                needs_function_calling=reqs.needs_function_calling,
                needs_vision=reqs.needs_vision,
                local_only=reqs.local_only,
                prefer_speed=reqs.prefer_speed,
                prefer_quality=reqs.prefer_quality,
                prefer_local=reqs.prefer_local,
                estimated_input_tokens=reqs.estimated_input_tokens,
                estimated_output_tokens=reqs.estimated_output_tokens,
                min_context=reqs.effective_context_needed,
                priority=reqs.priority,
                exclude_models=reqs.exclude_models or [],
                task_obj=task,
                iteration_n=0,
            )
        except Exception as exc:
            # Propagate non-retryable errors to the orchestrator:
            # - ModelCallFailed → availability backoff with wake signals
            # - Code bugs → immediate terminal (retrying won't help)
            from src.core.router import ModelCallFailed
            _NON_RETRYABLE = (ModelCallFailed, AttributeError, TypeError,
                              ImportError, NameError, KeyError)
            if isinstance(exc, _NON_RETRYABLE):
                raise
            logger.error(f"[Task #{task_id}] Single-shot call failed: {exc}")
            return {
                "status": "failed",
                "result": f"Agent failed: {exc}",
                "error": str(exc),
                "model": "unknown", "cost": 0, "difficulty": reqs.difficulty,
            }

        content = response.get("content", "")
        used_model = response.get("model", "unknown")
        cost = response.get("cost", 0)

        parsed = self._parse_agent_response(content)
        if parsed is None:
            parsed = {"action": "final_answer", "result": content}

        action_type = parsed.get("action", "final_answer")

        if action_type == "decompose" or parsed.get("subtasks"):
            return {
                "status": "needs_subtasks",
                "subtasks": parsed.get("subtasks", []),
                "plan_summary": parsed.get("plan_summary", ""),
                "model": used_model, "cost": cost, "difficulty": reqs.difficulty,
            }

        return {
            "status": "completed",
            "result": parsed.get("result", content),
            "model": used_model, "cost": cost,
            "difficulty": reqs.difficulty, "iterations": 1,
        }

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
