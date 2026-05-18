"""Z7 T6 A7 — outreach_deliverability_check posthook handler.

Thin shim that delegates to mr_roboto.outreach_deliverability_check.handle.

Handler contract
----------------
``handle(task, result) -> dict``

Returns:
  {"status": "ok"}                             — metrics within bounds
  {"status": "paused", "bounce_rate": float}   — threshold exceeded; FA emitted
  {"status": "skip", "reason": str}            — not enough data or missing params
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.outreach_deliverability_check")


async def handle(task: dict, result: dict) -> dict:
    """Delegate to the mr_roboto module implementation."""
    try:
        from mr_roboto.outreach_deliverability_check import handle as _handle
        return await _handle(task, result)
    except Exception as exc:
        logger.warning(
            "outreach_deliverability_check posthook failed: %s", exc
        )
        return {"status": "skip", "reason": f"handler unavailable: {exc}"}
