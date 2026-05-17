"""Yalayut Phase 4 — yalayut_discovery mechanical executor.

Dispatches the catalog-discovery pipeline. ``mode`` (payload):
  - ``daily``      → yalayut.daily_discovery()    (trusted cron-mode sources)
  - ``on_demand``  → yalayut.on_demand_discovery(demand)  (one DemandSignal)

Leaf shim — the only mr_roboto file that imports yalayut for discovery.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.yalayut_discovery")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    mode = (payload.get("mode") or "daily").lower()
    import yalayut
    try:
        if mode == "daily":
            return await yalayut.daily_discovery()
        if mode == "on_demand":
            demand = payload.get("demand") or {}
            if not demand:
                return {"ok": False, "reason": "on_demand mode needs a demand"}
            return await yalayut.on_demand_discovery(demand)
        return {"ok": False, "reason": f"unknown discovery mode: {mode!r}"}
    except Exception as e:  # noqa: BLE001
        logger.warning("yalayut_discovery executor failed: %s", e)
        return {"ok": False, "reason": str(e)}
