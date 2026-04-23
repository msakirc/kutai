"""Mechanical clarify executor: sends clarification prompt via Telegram."""
from __future__ import annotations

import logging

from src.infra.db import update_task
from src.app.telegram_bot import get_telegram

logger = logging.getLogger(__name__)


async def send_variant_keyboard(
    mission_id: int,
    task_id: int,
    base_label: str,
    options: list[dict],
) -> None:
    """Stub: send an inline-keyboard with variant buttons + 'Compare all'.

    Task 13 will wire the real Telegram send here. For now, just log.
    """
    logger.info(
        "send_variant_keyboard: mission=%s task=%s base=%r options=%s",
        mission_id, task_id, base_label, [o.get("label") for o in options],
    )


async def clarify(task: dict) -> dict:
    payload = task.get("payload") or {}
    kind = payload.get("kind")

    if kind == "variant_choice":
        payload_from = payload.get("payload_from", "gate_result")
        artifacts = task.get("artifacts") or {}
        source = artifacts.get(payload_from) or {}
        base_label = source.get("base_label", "")
        options = source.get("clarify_options", [])

        await send_variant_keyboard(
            task.get("mission_id"),
            task["id"],
            base_label,
            options,
        )
        await update_task(task["id"], status="waiting_human")
        return {
            "status": "needs_clarification",
            "kind": "variant_choice",
            "prompt": f"{base_label} için hangi model?",
        }

    # Default: plain question clarify
    question = payload.get("question")
    if not question:
        raise ValueError("clarify payload requires 'question'")
    # Register the SOURCE (blocked LLM) task with Telegram, not this
    # mechanical executor row. apply._apply_clarify set the source to
    # waiting_human and spawned us as its child (parent_task_id=source).
    # Using task["id"] (mechanical) caused the user's reply to miss its
    # target: orchestrator overwrote the mechanical row back to
    # status=completed on return, leaving no waiting_human match for
    # the reply handler (observed 2026-04-23: user's "C" answer
    # rerouted to the generic LLM classifier). parent_task_id falls
    # back to task["id"] when absent — safe for test fixtures that
    # don't model the full spawn graph.
    source_id = task.get("parent_task_id") or task["id"]
    tg = get_telegram()
    await tg.request_clarification(source_id, task.get("title", ""), question)
    return {"sent": True, "question": question, "source_task_id": source_id}
