"""Single-shot LLM call — one request, parse, return. No tool loop, no state.

Used by profiles with execution_pattern="single_shot" (planner, classifier).
"""
from __future__ import annotations

import json

from ..infra.logging_config import get_logger
from .context import build_system_prompt
from .parsing import parse_action

logger = get_logger("runtime.single_shot")


async def run(profile, task: dict) -> dict:
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
    reqs = await profile._build_model_requirements(task, _ss_ctx)

    system_prompt = build_system_prompt(profile, task)
    context = await profile._build_context(task)
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

    parsed = parse_action(content)
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
