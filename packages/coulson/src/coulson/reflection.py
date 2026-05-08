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

logger = get_logger("agents.base")


# ────────────────────────────────────────────────────────────────────────────
# Per-agent reflection checklists
# ────────────────────────────────────────────────────────────────────────────

REFLECTION_BLOCKS: dict[str, str] = {
    "coder": (
        "Self-check before final_answer:\n"
        "1. Did you RUN the code? Don't assume it works.\n"
        "2. Did TESTs pass (if a test suite exists)?\n"
        "3. Any TODO / pass / placeholder left in the code?\n"
        "4. Are all IMPORTs at the top of files and resolvable?\n"
        "If any 'no' — keep iterating, don't emit final_answer yet."
    ),
    "implementer": (
        "Self-check before final_answer:\n"
        "1. LINT clean (run `lint` tool)?\n"
        "2. `python -m py_compile <file>` — SYNTAX clean?\n"
        "3. Matches the SPEC / ARCHITECTURE.md interface exactly?\n"
        "4. Only your assigned file touched (no wandering)?\n"
        "If any 'no' — fix before final_answer."
    ),
    "fixer": (
        "Self-check before final_answer:\n"
        "1. Every FEEDBACK bullet addressed?\n"
        "2. TESTs run after edit, no new failures?\n"
        "3. No unintended DELETe of unrelated logic?\n"
        "If any 'no' — fix before final_answer."
    ),
    "test_generator": (
        "Self-check before final_answer:\n"
        "1. Tests run? (Don't claim they pass without running.)\n"
        "2. COVERAGE — every public function + error path + boundary tested?\n"
        "3. No FLAKy waits / sleeps / time-based asserts?\n"
        "4. ASSERT messages helpful (not bare `assert x`)?\n"
        "If any 'no' — keep iterating."
    ),
}

_GENERIC_REFLECTION_BLOCK = (
    "Review your output for errors before final_answer. "
    "Did you actually do what was asked?"
)


def build_reflection_prompt(agent_name: str, iteration: int) -> str:
    """Return a role-specific self-check checklist for *agent_name*.

    Falls back to a generic prompt for agents without a dedicated checklist
    so that all currently enabled agents (researcher, writer, shopping_advisor,
    deal_analyst, product_researcher) continue to work unchanged.
    """
    block = REFLECTION_BLOCKS.get(agent_name, _GENERIC_REFLECTION_BLOCK)
    return f"[iteration {iteration}] {block}"


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

        _system_base = (
            "You are a careful reviewer. Check this response "
            "for errors, omissions, or hallucinations. "
            "If the response is good, respond: "
            '{"verdict": "ok"}. '
            "If there are issues, respond: "
            '{"verdict": "fix", "issues": "description", '
            '"corrected_result": "the fixed version"}.'
        )
        _system_content = (
            f"{_system_base}\n\n{checklist}" if checklist else _system_base
        )
        messages = [
            {"role": "system", "content": _system_content},
            {"role": "user", "content": (
                f"Task: {task.get('title', '')}\n"
                f"Description: {(task.get('description') or '')[:500]}\n\n"
                f"Response to review:\n{result[:3000]}"
            )},
        ]
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
