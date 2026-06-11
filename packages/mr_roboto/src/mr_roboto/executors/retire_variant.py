"""Z9 Growth T5D/T5E — retire_variant mechanical executor.

Closes out an A/B experiment: marks the winning variant ``winner`` and the
losing variant(s) ``loser`` via ``update_variant_status``, and flips the
PostHog feature-flag — winner → 100% rollout, loser → 0%.

Routed via ``mr_roboto.run`` when ``payload["action"] == "retire_variant"``.
Invoked by the founder-gated ``/experiment_ship`` and ``/experiment_rollback``
Telegram commands — never auto-fired. The A/B *winner* call itself is
computed mechanically by ``record_verdict`` (Bayesian posterior in
``src/growth/verdict_stats.py``); this executor only enacts the decision.

Architecture contract
---------------------
**Mechanical, no LLM.** Status transitions + flag flips are deterministic.
Nothing here touches the ``LLMDispatcher``.

Payload (``task.context.payload`` or ``task.payload``)::

    {
        "action": "retire_variant",
        "mission_id": 12,            # OR hypothesis_id
        "winner": "treatment",       # variant_name to promote (default treatment)
        "decision": "ship"           # ship | rollback — audit label
    }

``ship``    — promote ``winner`` to 100%, retire the other arm at 0%.
``rollback`` — force the *control* to 100% (treatment is the loser),
regardless of stats. Both go through the same status + flag machinery.
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.retire_variant")


def _parse_context(task: dict) -> dict:
    raw = task.get("context")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _pick_payload(task: dict, ctx: dict) -> dict:
    spec = ctx.get("payload")
    if isinstance(spec, dict) and spec:
        return spec
    payload = task.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _decode_rule(rule: Any) -> dict:
    if isinstance(rule, dict):
        return rule
    if isinstance(rule, str) and rule.strip():
        try:
            return json.loads(rule) or {}
        except json.JSONDecodeError:
            return {}
    return {}


async def _flip_posthog_flag(
    task: dict, flag_id: Any, flag_key: str, rollout: int
) -> bool:
    """Set a PostHog feature-flag's rollout percentage via vendor_call.

    ``rollout`` is 0 (loser) or 100 (winner). Returns True on success.
    Mock-mode safe — never hits the network in a non-prod env.
    """
    if not flag_id and not flag_key:
        return False
    from mr_roboto.executors.vendor_call import run as vendor_call_run

    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": "posthog",
                "action": "update_feature_flag",
                "params": {
                    "project_id": "default",
                    "flag_id": flag_id,
                    "key": flag_key,
                    "active": rollout > 0,
                    "filters": {
                        "groups": [
                            {"properties": [], "rollout_percentage": rollout}
                        ]
                    },
                },
            }
        },
    }
    try:
        env = await vendor_call_run(sub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("posthog update_feature_flag raised", error=str(exc))
        return False
    return bool(isinstance(env, dict) and env.get("ok"))


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Retire an A/B experiment's variants. Never raises."""
    from src.infra.db import (
        get_variants,
        update_variant_status,
    )
    from general_beckman import record_growth_event

    ctx = _parse_context(task)
    payload = _pick_payload(task, ctx)
    mission_id = payload.get("mission_id") or task.get("mission_id")
    hypothesis_id = payload.get("hypothesis_id")
    winner_name = str(payload.get("winner") or "treatment").lower()
    decision = str(payload.get("decision") or "ship").lower()

    if mission_id is None and hypothesis_id is None:
        return {"ok": False, "reason": "missing_mission_or_hypothesis"}

    # rollback always crowns control; ship honours the requested winner.
    if decision == "rollback":
        winner_name = "control"

    variants = await get_variants(
        mission_id=int(mission_id) if mission_id is not None else None,
        hypothesis_id=int(hypothesis_id)
        if hypothesis_id is not None else None,
    )
    # Only act on still-active variants — idempotent re-runs.
    active = [v for v in variants if v.get("status") == "active"]
    if not active:
        logger.info(
            "retire_variant: no active variants for mission=%s hyp=%s",
            mission_id, hypothesis_id,
        )
        return {"ok": True, "retired": 0, "reason": "no_active_variants",
                "mission_id": mission_id}

    retired: list[dict] = []
    for v in active:
        vid = int(v.get("id") or 0)
        vname = str(v.get("variant_name") or "").lower()
        is_winner = vname == winner_name
        new_status = "winner" if is_winner else "loser"
        rule = _decode_rule(v.get("assignment_rule"))
        flag_id = rule.get("posthog_flag_id")
        flag_key = rule.get("posthog_flag_key") or ""

        try:
            await update_variant_status(vid, new_status)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "retire_variant: status update failed", variant_id=vid,
                error=str(exc),
            )
            continue

        # Feature variants carry a posthog flag — flip it. Pricing
        # variants have no flag (rule.kind == 'pricing'); Stripe price
        # routing is handled by the founder /confirm pricing flow.
        flag_flipped = False
        if rule.get("kind", "feature") != "pricing":
            rollout = 100 if is_winner else 0
            flag_flipped = await _flip_posthog_flag(
                task, flag_id, flag_key, rollout,
            )

        retired.append({
            "variant_id": vid,
            "variant_name": vname,
            "status": new_status,
            "flag_flipped": flag_flipped,
            "rollout": 100 if is_winner else 0,
        })

    try:
        await record_growth_event(
            mission_id, "ab_retired",
            {
                "decision": decision,
                "winner": winner_name,
                "hypothesis_id": hypothesis_id,
                "retired": retired,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ab_retired event failed", error=str(exc))

    logger.info(
        "retire_variant complete mission=%s decision=%s winner=%s retired=%d",
        mission_id, decision, winner_name, len(retired),
    )
    return {
        "ok": True,
        "retired": len(retired),
        "decision": decision,
        "winner": winner_name,
        "mission_id": mission_id,
        "variants": retired,
    }


__all__ = ["run"]
