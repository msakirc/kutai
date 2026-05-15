"""Z7 T6 A7 — outreach/draft: LLM-bound personalized draft dispatcher.

LLM-bound verb: dispatches a personalized cold outreach draft task via
general_beckman.enqueue. The agent receives prospect_data and the template
context; it writes a personalized draft body_md.

Public API
----------
  run_outreach_draft(
      product_id, mission_id, prospect_data, template_id, list_id
  ) -> dict

Internal hook (patched in tests)
---------------------------------
  enqueue(spec_dict, *, lane=, ...) -> dict   (beckman enqueue)
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.outreach_draft")


async def enqueue(spec: dict, **kwargs) -> dict:
    """Thin wrapper around general_beckman.enqueue for test patching."""
    from general_beckman import enqueue as _enqueue
    return await _enqueue(spec, **kwargs)


async def run_outreach_draft(
    product_id: str,
    mission_id: int,
    prospect_data: dict[str, Any],
    template_id: str,
    list_id: str,
) -> dict[str, Any]:
    """Enqueue an LLM draft task for cold outreach personalization.

    Returns:
      {"status": "enqueued", "task_id": <int>}
      {"status": "error", "error": <str>}
    """
    spec = {
        "title": f"Draft outreach for {prospect_data.get('name', 'prospect')} ({product_id})",
        "description": (
            "Generate a personalized cold outreach email body. "
            "Inputs are in context.prospect_data and context.template_id. "
            "Output the final body_md in the result artifact."
        ),
        "agent_type": "coder",  # generic LLM agent; workflow steps use agent-type routing
        "mission_id": mission_id,
        "context": {
            "product_id": product_id,
            "list_id": list_id,
            "template_id": template_id,
            "prospect_data": prospect_data,
            "action_hint": "outreach_draft",
        },
    }

    try:
        result = await enqueue(spec)
        task_id = (result or {}).get("task_id") or (result or {}).get("id")
        logger.info(
            "outreach_draft: enqueued",
            product_id=product_id,
            mission_id=mission_id,
            task_id=task_id,
        )
        return {"status": "enqueued", "task_id": task_id}
    except Exception as exc:
        logger.error(
            "outreach_draft: enqueue failed",
            product_id=product_id,
            error=str(exc),
        )
        return {"status": "error", "error": str(exc)}
