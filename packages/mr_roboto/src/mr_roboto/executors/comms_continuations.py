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


async def _emit_crisis_card(*, event_id, product_id, tier, variants):
    """Surface holding-statement variants to the founder. Never auto-posts."""
    try:
        from src.founder_actions import create as fa_create
        await fa_create(
            mission_id=None, kind="generic",
            title=f"Crisis holding statements ready (event #{event_id}, Tier {tier}) — pick one",
            why=("KutAI drafted holding-statement variants for the crisis. "
                 "NEVER auto-posted — select/edit and post manually."),
            instructions=[f"Variant {chr(65+i)}:\n\n{v}" for i, v in enumerate(variants)]
                         + ["Pick one, edit as needed, post manually."],
            expected_output_kind="ack_only", notify_telegram=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("crisis card emit failed: %s", exc)


async def _crisis_resume(child_task_id, result, state):
    from mr_roboto.crisis_draft_holding import parse_variants, canned_variants
    variants = parse_variants(_extract_content(result))
    if not variants:
        variants = canned_variants(int(state.get("tier") or 1), state.get("product_id") or "")
    await _emit_crisis_card(event_id=state.get("event_id"), product_id=state.get("product_id") or "",
                            tier=int(state.get("tier") or 1), variants=variants)


async def _crisis_resume_err(child_task_id, result, state):
    from mr_roboto.crisis_draft_holding import canned_variants
    logger.warning("crisis holding child failed (%s) — canned fallback", (result or {}).get("error"))
    variants = canned_variants(int(state.get("tier") or 1), state.get("product_id") or "")
    await _emit_crisis_card(event_id=state.get("event_id"), product_id=state.get("product_id") or "",
                            tier=int(state.get("tier") or 1), variants=variants)


# Stubs — fully implemented in Tasks 3-4. Must exist now for registration.
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
