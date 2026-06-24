"""Z7 T3D — B3: incident_update_review posthook handler.

Fires after incident/draft_update completes.  Emits a founder_action card
containing the draft status update for review.  The founder approves or
edits before incident/publish_status is called.

Handler contract
----------------
``handle(task, result) -> dict``

Returns one of:
  {"status": "ok", "founder_action_id": int}  — founder_action created
  {"status": "skip", "reason": str}            — nothing to review (no draft)
  {"status": "failed", "error": str}           — unexpected error

TODO: max 4hr SLA timer — if founder is unavailable >4h the B6 crisis
      playbook should take over.  Wire this when B6 (T3E) lands.
"""
from __future__ import annotations

import json
from typing import Any

from yazbunu import get_logger

logger = get_logger("beckman.posthooks.incident_update_review")


async def _emit_founder_action(
    *,
    mission_id: int,
    incident_id: Any,
    product_id: str,
    draft: str,
    status_kind: str,
) -> Any:
    """Emit a founder_action asking for review of the draft status update."""
    try:
        from src.founder_actions import create as fa_create

        title = (
            f"Review status update draft for incident #{incident_id} "
            f"({product_id}) — [{status_kind}]"
        )
        why = (
            "A customer-facing status update has been drafted. "
            "It must be reviewed and approved before publication. "
            "The update will appear on /status and in the RSS feed."
        )
        instructions = [
            f"Draft text:\n\n{draft[:1000]}",
            "If the draft looks good, call incident/publish_status with this text.",
            "If you want to edit: call incident/draft_update again with updated details.",
            "If this is a critical incident (>4h unresolved), the B6 crisis playbook applies.",
        ]
        return await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=why,
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=True,
            urgent=(status_kind == "resolved"),
        )
    except Exception as exc:
        logger.warning(
            "incident_update_review: _emit_founder_action failed", error=str(exc)
        )
        return None


async def handle(task: dict, result: dict) -> dict[str, Any]:
    """incident_update_review posthook handler."""
    task_id = task.get("id")
    mission_id = task.get("mission_id") or 0

    # Parse task context for incident info.
    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx: dict = json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = dict(ctx_raw)
    else:
        ctx = {}

    # Try getting draft from result (set by incident/draft_update verb).
    draft = result.get("draft") or ctx.get("draft") or ""
    incident_id = result.get("incident_id") or ctx.get("incident_id")
    product_id = result.get("product_id") or ctx.get("product_id") or ""
    status_kind = result.get("status_kind") or ctx.get("status_kind") or "investigating"

    if not draft:
        logger.debug(
            "incident_update_review: no draft in result — skip",
            task_id=task_id,
        )
        return {"status": "skip", "reason": "no draft text in task result"}

    if not incident_id:
        logger.debug(
            "incident_update_review: no incident_id — skip",
            task_id=task_id,
        )
        return {"status": "skip", "reason": "no incident_id in result or context"}

    fa = await _emit_founder_action(
        mission_id=mission_id,
        incident_id=incident_id,
        product_id=product_id,
        draft=draft,
        status_kind=status_kind,
    )
    fa_id = getattr(fa, "id", None) if fa else None

    logger.info(
        "incident_update_review: founder_action emitted",
        task_id=task_id,
        incident_id=incident_id,
        product_id=product_id,
        founder_action_id=fa_id,
    )
    return {
        "status": "ok",
        "founder_action_id": fa_id,
        "incident_id": incident_id,
        "product_id": product_id,
        "status_kind": status_kind,
    }
