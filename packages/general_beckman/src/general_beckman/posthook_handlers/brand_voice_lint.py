"""Z7 T1.0 stub: brand_voice_lint posthook handler.

Agent A5 (brand voice lint) owns this file. Replace the stub body with real
brand-voice linting logic against docs/templates/brand_voices/*.md.
Handler contract: return {"status": "ok"} or {"status": "skip", "reason": ...}.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.brand_voice_lint")


async def handle(task: dict, result: dict) -> dict:
    """Stub: brand_voice_lint posthook.

    Lints produced copy artifacts against brand voice rules.
    Returns {"status": "ok"} as a no-op until A5 implements the real logic.
    """
    logger.debug(
        "brand_voice_lint stub: no-op",
        task_id=task.get("id"),
        mission_id=task.get("mission_id"),
    )
    return {"status": "ok"}
