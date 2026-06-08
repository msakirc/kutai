"""SP4b Plan 3 — crisis/incident/press_kit CPS mechanical sinks.

MECHANICAL: no LLM call, no dispatcher import. Receive the already-produced
LLM output (*.resume) or fire on_error with the verb's canned fallback
(*.resume_err), then perform the founder-facing side-effect. Registered in
``general_beckman.continuations._HANDLER_MODULES`` so handlers survive restart.

Handler signature: async def handler(child_task_id: int, result: dict, state: dict) -> None
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.executors.comms_continuations")


def _extract_content(result: dict) -> str:
    """Dual-shape decode (normal terminal vs restart-reconcile)."""
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = inner.get("content", "")
    elif inner is not None:
        content = inner
    else:
        content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


# Stubs — fully implemented in Tasks 2-4. Must exist now for registration.
async def _crisis_resume(child_task_id, result, state): pass
async def _crisis_resume_err(child_task_id, result, state): pass
async def _incident_resume(child_task_id, result, state): pass
async def _incident_resume_err(child_task_id, result, state): pass
async def _press_kit_resume(child_task_id, result, state): pass
async def _press_kit_resume_err(child_task_id, result, state): pass


def register_continuations() -> None:
    """Register Plan-3 comms CPS sinks. Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("comms.crisis_holding.resume", _crisis_resume)
        register_resume("comms.crisis_holding.resume_err", _crisis_resume_err)
        register_resume("comms.incident_update.resume", _incident_resume)
        register_resume("comms.incident_update.resume_err", _incident_resume_err)
        register_resume("comms.press_kit.resume", _press_kit_resume)
        register_resume("comms.press_kit.resume_err", _press_kit_resume_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("comms continuation registration deferred: %s", exc)


register_continuations()
