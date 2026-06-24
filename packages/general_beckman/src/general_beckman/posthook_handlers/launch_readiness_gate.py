"""Z7 T3A (A2.r1) — launch_readiness_gate posthook handler.

Pre-T-0 hard gate: runs 7 readiness checks before publish_synchronized executes.
Called by the beckman apply layer as a mechanical posthook task.

Delegates to ``mr_roboto.launch_readiness_gate.handle(task, result)``.

Contract
--------
``handle(task, result) -> dict``

Returns one of:
- ``{"passed": True, "status": "ready", ...}``
- ``{"passed": True, "status": "ready_with_warnings", "warnings": [...]}``
- ``{"passed": False, "status": "blocked", "failing_checks": [...]}``

Handler is registered in posthooks.py as kind='launch_readiness_gate',
verb='launch_readiness_gate', default_severity='blocker'.
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("beckman.posthooks.launch_readiness_gate")


async def handle(task: dict, result: dict) -> dict[str, Any]:
    """launch_readiness_gate posthook handler.

    Delegates to the mr_roboto verb module so the gate logic lives in one
    canonical place (mr_roboto.launch_readiness_gate).
    """
    try:
        from mr_roboto.launch_readiness_gate import handle as _gate_handle
        return await _gate_handle(task, result)
    except Exception as exc:
        logger.error(
            "launch_readiness_gate posthook: unexpected error",
            error=str(exc),
        )
        # Fail closed: if the gate handler crashes, treat as blocked
        return {
            "passed": False,
            "status": "blocked",
            "failing_checks": ["gate_handler_error"],
            "warnings": [],
            "check_details": {"gate_handler_error": {"ok": False, "reason": str(exc)}},
        }
