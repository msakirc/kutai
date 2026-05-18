"""Yalayut demand-signal mechanical executor.

Lets core-loop files (general_beckman/apply.py) record a demand signal
WITHOUT importing yalayut — they enqueue a mechanical task with
action "yalayut_demand" and this leaf shim does the import.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.yalayut_demand")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    pattern = payload.get("source_step_pattern") or ""
    if not pattern:
        return {"ok": True, "recorded": False,
                "reason": "yalayut_demand needs source_step_pattern"}
    import yalayut
    try:
        row_id = await yalayut.record_demand_signal(
            source_step_pattern=pattern,
            intent_keywords=payload.get("intent_keywords") or [],
            signal_type=payload.get("signal_type") or "dlq",
            confidence=float(payload.get("confidence", 0.3)),
        )
        return {"ok": True, "row_id": row_id}
    except Exception as e:  # noqa: BLE001 — a signal failure must not DLQ
        logger.warning("yalayut_demand executor failed: %s", e)
        return {"ok": True, "recorded": False, "reason": str(e)}
