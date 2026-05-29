"""Self-reflection — review own output for errors.

Pure async function. No agent state.

Public API
----------
self_reflect(task, result, reqs_or_tier, used_model, checklist=None) -> dict | None
    Review own output for errors. Returns verdict dict or None on error.
    Uses dogru_mu_samet to reject degenerate corrected_result.

build_reflection_prompt(agent_name, iteration) -> str
    Return a role-specific self-check checklist injected into the reviewer's
    system message. Falls back to a generic prompt for unknown agents.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger
from .parsing import _try_parse_json

logger = get_logger("coulson.reflection")


# ────────────────────────────────────────────────────────────────────────────
# Per-agent reflection checklists — MOVED to src.core.reflection_posthook
# (SP3b Task 5). These names are re-exported here for back-compat: coulson's
# inline self_reflect caller + the Z2/Z3 stack/layer tests import them from
# coulson.reflection. Task 7 removes coulson's inline caller; the shim can be
# dropped then if no external importer remains.
# ────────────────────────────────────────────────────────────────────────────

from src.core.reflection_posthook import (  # noqa: E402,F401
    STACK_BLOCKS,
    LAYER_BLOCKS,
    REFLECTION_BLOCKS,
    _GENERIC_REFLECTION_BLOCK,
    build_reflection_prompt,
    build_reflect_messages,
)


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


async def self_reflect(
    task: dict,
    result: str,
    reqs_or_tier=None,
    used_model: str = "",
    checklist: str | None = None,
) -> dict | None:
    """Review own output for errors. Returns verdict dict or None on error.

    Accepts tier string or ModelRequirements as ``reqs_or_tier``.
    Uses dogru_mu_samet to reject degenerate corrected_result.

    ``checklist`` is an optional per-agent self-check block (from
    ``build_reflection_prompt``) appended to the reviewer system message.
    """
    try:
        from fatih_hoca.requirements import ModelRequirements

        # Build requirements for the reflection call
        if isinstance(reqs_or_tier, ModelRequirements):
            reflect_reqs = ModelRequirements(
                task="reviewer",
                difficulty=reqs_or_tier.difficulty,
                agent_type="self_reflection",
                estimated_input_tokens=800,
                estimated_output_tokens=500,
                prefer_speed=True,
            )
        else:
            # Legacy fallback — tier strings no longer used
            reflect_reqs = ModelRequirements(
                task="reviewer",
                difficulty=6,
                agent_type="self_reflection",
                estimated_input_tokens=800,
                estimated_output_tokens=500,
                prefer_speed=True,
            )

        # Message build moved to src.core.reflection_posthook (SP3b Task 5) so
        # the reflection LLM post-hook child can reuse the exact same prompt.
        messages = build_reflect_messages(task, result, checklist=checklist)
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        response = await get_dispatcher().request(
            CallCategory.OVERHEAD,
            task=reflect_reqs.task,
            agent_type=reflect_reqs.agent_type,
            difficulty=reflect_reqs.difficulty,
            messages=messages,
            estimated_input_tokens=reflect_reqs.estimated_input_tokens,
            estimated_output_tokens=reflect_reqs.estimated_output_tokens,
            min_context=reflect_reqs.effective_context_needed,
            prefer_speed=reflect_reqs.prefer_speed,
            task_obj=task,
        )
        raw = response.get("content", "").strip()
        parsed = _try_parse_json(raw)
        if parsed and parsed.get("verdict") == "fix":
            corrected = parsed.get("corrected_result")
            if corrected:
                from dogru_mu_samet import assess as cq_assess
                _reflect_cq = cq_assess(corrected)
                if _reflect_cq.is_degenerate:
                    logger.warning(
                        f"Self-reflection produced degenerate corrected_result "
                        f"({_reflect_cq.summary}), keeping original"
                    )
                    return None
            return parsed
    except Exception as exc:
        logger.debug(f"Self-reflection failed: {exc}")
    return None
