"""KDV state persistence executor.

Periodic salako handler — fired every 60s by the ``kdv_persist`` internal
cadence. Snapshots the current KuledenDonenVar state and writes it to
the ``kdv_state`` table in kutai.db so adapted rate limits, 429 history,
daily counters, and header reset clocks survive process restarts.

Boundary: this module legitimately bridges KDV with sqlite. KDV itself
holds no DB knowledge; it exposes ``snapshot_state()`` and the wiring
lives here.
"""
from __future__ import annotations

import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("salako.kdv_persist")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Save current KDV state to kutai.db. Returns counters for visibility.

    No-op when DB_PATH is unset (tests, CLI tools). Errors are logged but
    never raised — persistence failures must not crash the orchestrator.
    """
    db_path = os.environ.get("DB_PATH")
    if not db_path:
        return {"saved": False, "reason": "DB_PATH unset"}

    from src.core.router import get_kdv
    from src.infra import kdv_persistence

    kdv = get_kdv()
    snap = kdv.snapshot_state()
    try:
        await kdv_persistence.save(kdv, db_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("kdv_persist save failed: %s", e)
        return {"saved": False, "error": str(e)}

    return {
        "saved": True,
        "models": len(snap.get("models", {})),
        "providers": len(snap.get("providers", {})),
        "breakers": len(snap.get("breakers", {})),
    }
