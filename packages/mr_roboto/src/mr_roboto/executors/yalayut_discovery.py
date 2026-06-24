"""Yalayut Phase 4 — yalayut_discovery mechanical executor.

Dispatches the catalog-discovery pipeline. ``mode`` (payload):
  - ``daily``      → yalayut.daily_discovery() + yalayut.run_demand_drain()
  - ``on_demand``  → yalayut.on_demand_discovery(demand)  (one DemandSignal)

Leaf shim — the only mr_roboto file that imports yalayut for discovery.
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.yalayut_discovery")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    mode = (payload.get("mode") or "daily").lower()
    import yalayut
    try:
        if mode == "daily":
            result = await yalayut.daily_discovery()
            # Fold the autonomous demand drain into the daily run — no new
            # orchestrator method, no new cron cadence row (handoff option 2).
            try:
                result["demand_drain"] = await yalayut.run_demand_drain()
            except Exception as e:  # noqa: BLE001 — drain must not fail the run
                logger.warning("demand drain failed inside daily discovery: %s", e)
                result["demand_drain"] = {"error": str(e)}
            return result
        if mode == "on_demand":
            demand = payload.get("demand") or {}
            if not demand:
                return {"ok": False, "reason": "on_demand mode needs a demand"}
            return await yalayut.on_demand_discovery(demand)
        return {"ok": False, "reason": f"unknown discovery mode: {mode!r}"}
    except Exception as e:  # noqa: BLE001
        logger.warning("yalayut_discovery executor failed: %s", e)
        return {"ok": False, "reason": str(e)}
