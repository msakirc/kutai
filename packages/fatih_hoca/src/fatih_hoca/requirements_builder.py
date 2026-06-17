"""Build ModelRequirements from a task + its context.

Phase A.13 (2026-05-04): moved from BaseAgent._build_model_requirements.
Selection-vocabulary (template lookup, classification overlay, retry
escalation, sensitivity heuristics) belongs next to AGENT_REQUIREMENTS,
not in agent runtime.

Public API:
    await requirements_for(task, task_ctx, *, agent_name=None) -> ModelRequirements

agent_name override: when None, falls back to task_ctx.classification.agent_type
or "executor". Workflow-step tasks always use task_ctx.agent_type if present.
"""
from __future__ import annotations

import copy
import json
import re

from .requirements import AGENT_REQUIREMENTS, ModelRequirements

import logging
logger = logging.getLogger("fatih_hoca.requirements_builder")

# Whole-word sensitivity tokens that auto-force ``local_only`` (keep user
# PII off cloud providers). Matched with ``\b`` so a larger identifier that
# merely contains a token as a substring (e.g. the ambition-tier enum value
# ``private_beta``, or ``homepage``) does NOT trip it.
_SENSITIVITY_RE = re.compile(
    r"\b(?:personal|private|secret|password|credential|my|home)\b"
)


def resolve_local_only(task: dict, task_ctx: dict | None) -> bool:
    """Single source of truth for whether a task must run on a LOCAL model.

    Called by BOTH the worker requirements builder (``requirements_for``) and
    Beckman admission, so the two hand the selector the SAME constraint.
    Divergence here was the admit-then-refuse loop (live 2026-06-16): admission
    built its select kwargs by hand and omitted ``local_only``, so it admitted a
    sensitive/local_only task onto a cloud model that the worker's re-select
    (which DOES pass local_only) then categorically refused -> ModelCallFailed
    -> Beckman re-pend -> admission re-picks cloud -> infinite loop.

    Any source True -> local_only:
      * classifier signal (``task_ctx.classification.local_only``)
      * explicit context flag (``task_ctx.local_only``)
      * PII heuristic, scoped by task origin (see below)

    PII-heuristic scoping (live 2026-06-18):
      Workflow steps (``task_ctx.is_workflow_step``) have machine-generated
      titles/descriptions that *discuss* security/auth/UI as SUBJECT MATTER —
      ``[4.6] auth_system_design`` says "password hashing", a prototype step
      embeds the ``lucide:home`` icon, ``[12.3] compliance_checks`` mentions
      "personal info". Keyword-scanning that template text mass-false-marked
      public engineering work ``local_only`` (5 i2p steps live), starving them
      onto a busy local GPU and feeding the local-thrash crashloop.
      Sensitivity is a property of the USER'S actual input, not of the
      generated step text — so for workflow steps we scan the propagated
      mission idea (``workflow_context``: raw_idea/product_name) instead. A
      genuinely sensitive idea still marks every step; benign idea marks none.
      Non-workflow (user free-text) tasks keep the title+description scan.
    """
    ctx = task_ctx if isinstance(task_ctx, dict) else {}
    classification = ctx.get("classification") or {}
    if classification.get("local_only"):
        return True
    if ctx.get("local_only"):
        return True
    if ctx.get("is_workflow_step"):
        wc = ctx.get("workflow_context")
        if isinstance(wc, dict):
            user_idea = " ".join(
                str(v) for v in wc.values() if isinstance(v, str)
            ).lower()
            if _SENSITIVITY_RE.search(user_idea):
                return True
        return False
    title = (task.get("title") or "").lower()
    description = (task.get("description") or "").lower()
    if _SENSITIVITY_RE.search(f"{title} {description}"):
        return True
    return False


async def requirements_for(
    task: dict,
    task_ctx: dict,
    *,
    agent_name: str | None = None,
    allowed_tools: list[str] | None = None,
) -> ModelRequirements:
    """Build ModelRequirements from task metadata + agent properties + classification.

    The agent_name parameter, when provided, locks the requirement to that
    template and overrides any classifier-supplied agent_type. Workflow-step
    tasks always lock to their declared agent_name to prevent classifier
    misroutes (e.g. "writer" step misclassified as "coder").

    When agent_name is None and the task isn't a workflow step, classifier
    output (task_ctx.classification.agent_type) takes precedence; otherwise
    falls back to "executor".
    """
    title = task.get("title", "").lower()
    description = task.get("description", "").lower()
    priority = task.get("priority", 5)

    classification = task_ctx.get("classification", {})

    # Workflow steps declare their agent explicitly — don't let the
    # classifier override it (e.g. "writer" step misclassified as "coder")
    if task_ctx.get("is_workflow_step"):
        effective_agent = agent_name or classification.get("agent_type") or "executor"
    else:
        effective_agent = (
            agent_name
            or classification.get("agent_type")
            or "executor"
        )

    template = AGENT_REQUIREMENTS.get(effective_agent) or AGENT_REQUIREMENTS.get(
        agent_name or "executor",
        ModelRequirements(task=effective_agent, difficulty=5),
    )
    reqs = copy.deepcopy(template)
    reqs.agent_type = agent_name or effective_agent
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

    # ── Adjust for task priority ──
    if priority >= 10:
        reqs.prefer_speed = True
        reqs.difficulty = max(reqs.difficulty, 6)
    elif priority <= 2:
        reqs.difficulty = max(1, reqs.difficulty - 2)

    # ── Local-only constraint (classifier / context flag / PII heuristic) ──
    # Single source of truth shared with Beckman admission so both select with
    # the same constraint (see resolve_local_only). Only upgrade, never
    # downgrade a template that already set local_only=True.
    if resolve_local_only(task, task_ctx):
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

    # ── Z10 T2A D7 — propagate mission quality_mode onto reqs ──
    try:
        _mid = task.get("mission_id")
        if _mid is not None:
            from dabidabi import get_mission_quality_mode
            reqs.quality_mode = await get_mission_quality_mode(int(_mid))
    except Exception:
        # leave default "balanced"
        pass

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

    # ── Input token estimate (RC-A change B, mission 74) ──
    # Use the SAME source as Beckman admission — estimate_for (the btable
    # chain: learned step p90 → static override → agent default) — instead
    # of a char-based (desc+ctx)//4 heuristic. Ground-truth telemetry over
    # 9998 calls showed char-based floored to 1000 while the real assembled
    # prompt (RAG + system + tools, added at call time) ran ~9k tokens, so
    # the worker under-estimated input by ~9000 and the per-call/per-request
    # gates were toothless. Worse, admission (already on estimate_for) and
    # the worker disagreed → admission admitted models the worker's select()
    # then rejected → the `no_candidates` DLQ loop. Both now agree.
    #
    # Lazy import: btable_cache lives in general_beckman (which imports
    # fatih_hoca), so a module-level import here would be circular. Fall
    # back to an empty btable → estimate_for resolves the agent default.
    try:
        from general_beckman.btable_cache import get_btable
        _bt = get_btable()
    except Exception:
        _bt = {}
    from fatih_hoca.estimates import estimate_for

    class _EstShim:
        __slots__ = ("agent_type", "context")

    _shim = _EstShim()
    _shim.agent_type = reqs.agent_type
    _shim.context = task_ctx if isinstance(task_ctx, dict) else {}
    reqs.estimated_input_tokens = estimate_for(_shim, btable=_bt).in_tokens

    # Template's estimated_output_tokens is a per-agent default (e.g.
    # analyst=3000, coder=4000). List-heavy workflow steps like
    # feature_prioritization need far more — a 15-25-item MoSCoW
    # breakdown with justifications runs 5-8k tokens and the default
    # caps the LLM mid-list, leaving trailing keys like 'could_have'
    # and 'wont_have' empty and the artifact failing schema validation
    # on "missing content about: [...]". Let the workflow step override
    # via context.estimated_output_tokens (clamped to [500, 16000]).
    _out_override = task_ctx.get("estimated_output_tokens")

    # Workflow-step tasks may pre-date the step's context-block edit in
    # i2p_v3 (existing DB rows captured their ctx at expansion time with
    # no estimated_output_tokens field). Re-read the live step def from
    # the workflow JSON so retries pick up newly-bumped budgets without
    # requiring row regeneration.
    if not _out_override and task_ctx.get("is_workflow_step"):
        try:
            step_id = task_ctx.get("workflow_step_id")
            mission_id = task.get("mission_id")
            if step_id and mission_id:
                from dabidabi import get_db
                _db = await get_db()
                _cur = await _db.execute(
                    "SELECT context FROM missions WHERE id = ?", (mission_id,),
                )
                _row = await _cur.fetchone()
                await _cur.close()
                _mctx: dict = {}
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
    # Only upgrade to function_calling, don't force it if the template
    # doesn't need it (e.g., planner, writer, reviewer). The template
    # already sets needs_function_calling=True for agents that genuinely
    # need tool use (coder, fixer, executor). The agent's allowed_tools
    # list (when non-empty) is also a signal — passed in via the
    # has_allowed_tools flag in task_ctx.
    if reqs.needs_function_calling:
        pass  # already set by template or classification
    elif allowed_tools and len(allowed_tools) > 0:
        # Agent explicitly declares tool list → it needs function calling
        reqs.needs_function_calling = True

    # ── Vision needed? (keyword override) ──
    # Skip keyword heuristic for workflow steps — they declare vision need
    # explicitly via tools_hint containing analyze_image.
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
