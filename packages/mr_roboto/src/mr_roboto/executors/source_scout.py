"""Yalayut Phase 4 — source_scout mechanical executor.

Runs ``yalayut.source_scout_scan()`` — proposes candidate sources to the
founder — and then runs ``yalayut.observe_and_propose()`` to scan vetting
audit data for repeated unknown shell/domain tokens and write founder policy
proposals. Both are the same "scan + propose to founder" shape and share the
source-scout's orchestrator cadence. Leaf shim importing yalayut.
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.source_scout")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    import yalayut
    try:
        result = await yalayut.source_scout_scan()
    except Exception as e:  # noqa: BLE001
        logger.warning("source_scout executor failed: %s", e)
        return {"ok": False, "reason": str(e)}

    # Best-effort: policy observation rides the same scout cadence. Its
    # failure must NOT fail the scout run.
    try:
        proposals = await yalayut.observe_and_propose()
        if isinstance(result, dict):
            result["policy_proposals_written"] = proposals
    except Exception as e:  # noqa: BLE001
        logger.warning("policy observe_and_propose failed: %s", e)

    return result
