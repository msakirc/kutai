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
from ..infra.db import recall_memory, update_task
from ..app.config import MAX_AGENT_ITERATIONS
from ..infra.logging_config import get_logger

logger = get_logger("agents.base")


# Re-exports for backward-compat callsites that imported these from base.py.
from src.runtime.tools import (  # noqa: F401
    SIDE_EFFECT_TOOLS,
    CACHEABLE_READ_TOOLS,
    TOOL_FAILURE_ESCALATION_THRESHOLD,
    partition_tool_calls as _partition_tool_calls,
    TOOL_SCHEMAS_BY_NAME as _TOOL_SCHEMAS_BY_NAME,
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

    async def execute(self, task: dict, progress_callback: Callable | None = None) -> dict:
        """Drive one task to completion. Phase A.10: delegates to src.runtime.execute."""
        from src.runtime import execute as _runtime_execute
        return await _runtime_execute(self, task, progress_callback)
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
