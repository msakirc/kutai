"""Yalayut Phase 4 — source_scout mechanical executor.

Runs ``yalayut.source_scout_scan()`` — proposes candidate sources to the
founder. Leaf shim importing yalayut.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.source_scout")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    import yalayut
    try:
        return await yalayut.source_scout_scan()
    except Exception as e:  # noqa: BLE001
        logger.warning("source_scout executor failed: %s", e)
        return {"ok": False, "reason": str(e)}
