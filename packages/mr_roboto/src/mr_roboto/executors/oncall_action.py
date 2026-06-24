"""Z8 T4B — oncall_action mechanical executor.

Pre-execute path for every on-call action verb. The on-call agent emits a
final_answer with ``{"verb": ..., "params": ...}``; the orchestrator wraps
that into a mechanical task with ``payload.action == "oncall_action"``.

Order of operations
-------------------
1. Validate the verb is in the whitelist (ops domain + any registered domain).
2. Check :func:`src.ops.action_cooldowns.check` for the (mission, verb).
   On block → return ``{"status": "blocked_by_cooldown", ...}`` WITHOUT
   invoking the sub-handler. No record() — blocks don't count as invocations.
3. Delegate to the verb-specific sub-handler via handler registry (A11.r1).
4. Call :func:`record` with the sub-handler's status.

A11.r1 refactor
---------------
The hardcoded ``WHITELISTED_VERBS`` frozenset and ``_SUB_HANDLERS`` dict
are now backed by the ``coulson.agent_handlers.registry`` so new domains
(e.g. ``'mention'``) can plug in at import time without editing this file.
Existing Z8 ops handlers register themselves below; all prior behavior is
preserved.
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger
from src.ops.action_cooldowns import check, record

logger = get_logger("mr_roboto.oncall_action")


# ---------------------------------------------------------------------------
# Registry bootstrap — ops domain (Z8 T4B originals)
# ---------------------------------------------------------------------------

def _bootstrap_ops_handlers() -> None:
    """Register all Z8 ops verb handlers into the coulson handler registry.

    Called once at module load time.  Idempotent: re-importing this module
    replaces with identical handlers (no observable difference).

    Also eagerly imports the other A11 domains (currently ``mention``) so
    their import-time ``register_handler`` calls fire as soon as the
    on-call dispatcher is set up — otherwise the ``mention`` domain stays
    empty until the first cron poll and ``oncall_action.run()``'s mention
    lookup misses every agent-dispatched mention event.
    """
    from coulson.agent_handlers.registry import register_handler

    register_handler("ops", "restart_service", _stub_handler)
    register_handler("ops", "rollback_to_last_green", _stub_handler)
    register_handler("ops", "scale_up", _stub_handler)
    register_handler("ops", "scale_down", _stub_handler)
    register_handler("ops", "drain_traffic", _stub_handler)
    register_handler("ops", "rotate_failed_key", _stub_handler)
    register_handler("ops", "archive_flake_test", _stub_handler)
    register_handler("ops", "escalate_to_founder", _escalate_handler)

    # A11.r1 — trigger the 'mention' domain's import-time registration.
    # mention_polls registers its handler at module import; importing it
    # here makes the 'mention' domain live without waiting for a cron poll.
    try:
        import mr_roboto.mention_polls  # noqa: F401  (import for side effect)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("oncall_action: mention domain bootstrap skipped: %s", exc)


# ---------------------------------------------------------------------------
# Sub-handlers (same implementations as original)
# ---------------------------------------------------------------------------

async def _stub_handler(verb: str, params: dict, mission_id: int) -> dict:
    msg = f"{verb} not implemented — needs vendor cloud API wiring"
    logger.warning(
        "oncall verb not implemented — escalate to founder",
        verb=verb,
        params=params,
        mission_id=mission_id,
    )
    return {
        "status": "not_implemented",
        "verb": verb,
        "error": msg,
        "stub": True,
    }


async def _escalate_handler(verb: str, params: dict, mission_id: int) -> dict:
    """Z8 T5G — real wire-through for ``escalate_to_founder``.

    Delegates to the dedicated ``escalate_to_founder`` mechanical executor
    which resolves the channel against ``escalation_policy`` and dispatches
    Telegram / SMS / log accordingly.
    """
    from .escalate_to_founder import run as ef_run
    result = await ef_run({
        "mission_id": mission_id,
        "payload": params,
    })
    return {
        "status": "ok" if result.get("ok") else "failed",
        "verb": verb,
        "channel": result.get("channel"),
        "tier": result.get("tier"),
        "founder_action_id": result.get("founder_action_id"),
        "sms_sid": result.get("sms_sid"),
        "stub": False,
    }


# Bootstrap ops handlers at import time
_bootstrap_ops_handlers()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Entry point invoked from ``mr_roboto.run`` dispatch.

    Verb lookup uses the handler registry (all domains) so mention/ and other
    future domain handlers are accepted alongside ops verbs.
    """
    from coulson.agent_handlers.registry import lookup_handler, list_verbs

    payload = task.get("payload") or {}
    verb = str(payload.get("verb") or "")
    params = payload.get("params") or {}
    mission_id = task.get("mission_id")
    domain = str(payload.get("domain") or "ops")

    if not verb:
        return {"status": "failed", "error": "oncall_action: missing 'verb'"}
    if mission_id is None:
        return {"status": "failed", "error": "oncall_action: missing mission_id"}

    # Check registry: try specified domain first, then fall back to any domain.
    handler = lookup_handler(domain, verb)
    if handler is None:
        # Fallback: search all domains so ops verbs still work when domain
        # isn't specified in the payload.
        for d in ("ops", "mention"):
            handler = lookup_handler(d, verb)
            if handler is not None:
                domain = d
                break

    if handler is None:
        all_verbs = sorted(list_verbs())
        logger.warning(
            "oncall_action: verb not in any registered domain",
            verb=verb,
            registered_verbs=all_verbs,
        )
        return {
            "status": "refused_not_whitelisted",
            "verb": verb,
            "whitelist": all_verbs,
        }

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

    outcome = await handler(verb, dict(params), int(mission_id))
    await record(int(mission_id), verb, str(outcome.get("status") or "unknown"))
    return outcome


# ---------------------------------------------------------------------------
# Legacy compat — keep WHITELISTED_VERBS for any external code that imports it
# ---------------------------------------------------------------------------

WHITELISTED_VERBS: frozenset[str] = frozenset({
    "restart_service",
    "rollback_to_last_green",
    "scale_up",
    "scale_down",
    "drain_traffic",
    "rotate_failed_key",
    "archive_flake_test",
    "escalate_to_founder",
})
