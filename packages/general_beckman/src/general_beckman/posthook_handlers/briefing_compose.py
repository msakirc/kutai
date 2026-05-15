"""Z7 T1.0 stub: briefing_compose posthook handler.

Agent A0 (briefing composition) owns this file. Replace the stub body with
real briefing-compose logic. The handler contract: receive task + result dicts,
return a dict with at minimum {"status": "ok"} (or {"status": "skip", "reason": ...}).
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.briefing_compose")


async def handle(task: dict, result: dict) -> dict:
    """Stub: briefing_compose posthook.

    Composes a mission briefing row in mission_briefings for the founder.
    Returns {"status": "ok"} as a no-op until A0 implements the real logic.
    """
    logger.debug(
        "briefing_compose stub: no-op",
        task_id=task.get("id"),
        mission_id=task.get("mission_id"),
    )
    return {"status": "ok"}
