"""Z8 T4B — oncall_action mechanical executor.

Pre-execute path for every on-call action verb. The on-call agent emits a
final_answer with ``{"verb": ..., "params": ...}``; the orchestrator wraps
that into a mechanical task with ``payload.action == "oncall_action"``.

Order of operations
-------------------
1. Validate the verb is in the whitelist.
2. Check :func:`src.ops.action_cooldowns.check` for the (mission, verb).
   On block → return ``{"status": "blocked_by_cooldown", ...}`` WITHOUT
   invoking the sub-handler. No record() — blocks don't count as invocations.
3. Delegate to the verb-specific sub-handler (stubs for v1; vendor-specific
   real implementations ride on the ``vendor_call`` executor from Z6).
4. Call :func:`record` with the sub-handler's status.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger
from src.ops.action_cooldowns import check, record

logger = get_logger("mr_roboto.oncall_action")

WHITELISTED_VERBS = frozenset(
    {
        "restart_service",
        "rollback_to_last_green",
        "scale_up",
        "scale_down",
        "drain_traffic",
        "rotate_failed_key",
        "archive_flake_test",
        "escalate_to_founder",
    }
)


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Entry point invoked from ``mr_roboto.run`` dispatch."""
    payload = task.get("payload") or {}
    verb = str(payload.get("verb") or "")
    params = payload.get("params") or {}
    mission_id = task.get("mission_id")

    if not verb:
        return {"status": "failed", "error": "oncall_action: missing 'verb'"}
    if verb not in WHITELISTED_VERBS:
        logger.warning("oncall_action: verb not whitelisted", verb=verb)
        return {
            "status": "refused_not_whitelisted",
            "verb": verb,
            "whitelist": sorted(WHITELISTED_VERBS),
        }
    if mission_id is None:
        return {"status": "failed", "error": "oncall_action: missing mission_id"}

    if not await check(int(mission_id), verb):
        logger.info(
            "oncall_action blocked by cooldown",
            mission_id=mission_id,
            verb=verb,
        )
        return {
            "status": "blocked_by_cooldown",
            "verb": verb,
            "mission_id": mission_id,
        }

    outcome = await _execute_verb(verb, dict(params), int(mission_id))
    await record(int(mission_id), verb, str(outcome.get("status") or "unknown"))
    return outcome


async def _execute_verb(verb: str, params: dict, mission_id: int) -> dict:
    """Dispatch to the verb-specific sub-handler.

    For T4 v1, each sub-handler is a stub that logs + returns a synthetic
    ``ok`` result so the cooldown + audit-trail wiring can be tested end-to-
    end without depending on real vendor adapters. Real implementations
    plug in through the Z6 ``vendor_call`` executor — those land in T5.
    """
    handler = _SUB_HANDLERS.get(verb, _stub_handler)
    return await handler(verb, params, mission_id)


async def _stub_handler(verb: str, params: dict, mission_id: int) -> dict:
    logger.info(
        "oncall verb stub fired (no vendor adapter wired yet)",
        verb=verb,
        params=params,
        mission_id=mission_id,
    )
    return {"status": "ok", "verb": verb, "params": params, "stub": True}


# Verb-specific sub-handlers; all stubs in v1 so behaviour is uniform.
# Real adapters (rollback via git tag, restart via Render API, scale via
# fly.io, etc.) land in T5 once vendor_call has the targeted methods.
_SUB_HANDLERS: dict[str, Any] = {
    "restart_service": _stub_handler,
    "rollback_to_last_green": _stub_handler,
    "scale_up": _stub_handler,
    "scale_down": _stub_handler,
    "drain_traffic": _stub_handler,
    "rotate_failed_key": _stub_handler,
    "archive_flake_test": _stub_handler,
    "escalate_to_founder": _stub_handler,
}
