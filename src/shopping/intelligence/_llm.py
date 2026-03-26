"""Shared LLM helper for shopping intelligence modules.

Routes all LLM calls through the centralized model router, which handles
model selection, rate limiting, GPU scheduling, and cost tracking.

Uses lazy imports so the module can be loaded in test environments where
litellm and the full router stack are not installed.
"""

from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence._llm")


async def _llm_call(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    task_id: int | None = None,
    mission_id: int | None = None,
) -> str:
    """Call LLM via the centralized router. Falls back gracefully.

    The router handles model selection (local llama.cpp, cloud, etc.),
    rate limiting, GPU scheduling, and cost tracking.

    Args:
        prompt: User prompt text.
        system: Optional system prompt.
        temperature: Sampling temperature.
        task_id: Optional task ID for cost attribution in the router.
        mission_id: Optional mission ID for cost attribution (future use).
    """
    try:
        from src.core.router import ModelRequirements, call_model

        reqs = ModelRequirements(
            task="shopping",
            agent_type="shopping_advisor",
            difficulty=3,
            prefer_speed=True,
            estimated_input_tokens=len(prompt) // 4,
            estimated_output_tokens=2048,
        )
        if task_id is not None:
            reqs._task_id = task_id  # type: ignore[attr-defined]
        if mission_id is not None:
            reqs._mission_id = mission_id  # type: ignore[attr-defined]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await call_model(reqs, messages)
        return response.get("content", "")
    except Exception as e:
        logger.warning("_llm_call failed", error=str(e), prompt_len=len(prompt))
        return ""
