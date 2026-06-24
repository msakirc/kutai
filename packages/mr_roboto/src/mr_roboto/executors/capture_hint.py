"""Yalayut Phase 4 — capture_hint mechanical executor.

The ``capture_hint`` post-hook routes here. Payload carries the source
task dict + its outcome; the executor calls ``yalayut.capture_hint``.
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.capture_hint")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    source_task = payload.get("source_task") or {}
    outcome = payload.get("outcome") or {}
    import yalayut
    try:
        await yalayut.capture_hint(source_task, outcome)
        return {"ok": True, "captured": True}
    except Exception as e:  # noqa: BLE001 — post-hook must never DLQ the source
        logger.warning("capture_hint executor failed: %s", e)
        return {"ok": True, "captured": False, "reason": str(e)}
