"""Z7 T1.0 stub: copy_compliance_review posthook handler.

Agent A6 (copy compliance review) owns this file. Replace the stub body with
real compliance-check logic against docs/templates/channel_rules/*.md and
any jurisdiction-specific rules surfaced by the compliance overlay.
Handler contract: return {"status": "ok"} or {"status": "skip", "reason": ...}.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.copy_compliance_review")


async def handle(task: dict, result: dict) -> dict:
    """Stub: copy_compliance_review posthook.

    Reviews produced copy for compliance (disclaimers, banned words,
    channel-specific rules). Returns {"status": "ok"} as a no-op until
    A6 implements the real logic.
    """
    logger.debug(
        "copy_compliance_review stub: no-op",
        task_id=task.get("id"),
        mission_id=task.get("mission_id"),
    )
    return {"status": "ok"}
