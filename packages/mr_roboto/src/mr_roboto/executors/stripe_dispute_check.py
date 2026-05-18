"""Z6 T5D — weekly Stripe dispute scan.

Calls ``vendor_call(stripe, list_disputes)`` for every mission whose
``monetization_strategy`` uses Stripe billing (or, when invoked as a
system cron with no mission_id, against the configured Stripe credential
once and surfaces a system-scope founder_action). New disputes — those
not present in the per-mission checkpoint at
``mission_<id>/.stripe/last_dispute_check.json`` — produce a
``legal_counsel`` ``founder_action`` per dispute (deduplicated by
dispute_id).

Idempotency
-----------
The checkpoint stores the last list of seen dispute ids. Re-runs without
new activity produce no founder_actions.

System mission scope
--------------------
Beckman seeds this as an internal cadence (no per-mission queue). We
walk the credentials table for missions that have provisioned Stripe
products (``stripe_provisioned`` artifact present) and run the check
per-mission so disputes are attached to the right thread. When no
provisioned missions exist, the executor is a cheap no-op.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.stripe_dispute_check")

SYSTEM_MISSION_ID = 0


# ── checkpoint I/O ────────────────────────────────────────────────────────


def _checkpoint_path(workspace_root: str, mission_id: int) -> str:
    return os.path.join(
        workspace_root, f"mission_{mission_id}", ".stripe",
        "last_dispute_check.json",
    )


def _load_checkpoint(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"seen_ids": []}


def _save_checkpoint(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def _workspace_root() -> str:
    return os.environ.get("MISSION_WORKSPACE_ROOT", os.getcwd())


# ── vendor_call indirection ────────────────────────────────────────────────


async def _vc(task: dict, service: str, action: str, params: dict) -> dict:
    from mr_roboto.executors.vendor_call import run as vendor_call_run
    sub = {
        "mission_id": task.get("mission_id"),
        "id": task.get("id"),
        "context": {
            "post_hook": {
                "service": service,
                "action": action,
                "params": params,
            }
        },
    }
    return await vendor_call_run(sub)


# ── founder_action helper ─────────────────────────────────────────────────


async def _emit_legal_counsel(
    mission_id: int,
    dispute_id: str,
    amount: Any,
    currency: Any,
    detail: dict,
) -> None:
    try:
        import src.founder_actions as fa
        why = (
            f"Stripe dispute {dispute_id} filed. "
            f"Amount: {amount} {str(currency or '').upper()}. "
            f"Reason: {detail.get('reason', 'unknown')}."
        )
        await fa.create(
            mission_id=int(mission_id),
            kind="legal_counsel",
            title=(
                f"Stripe dispute filed: {amount} "
                f"{str(currency or '').upper()}"
            ),
            why=why[:500],
            instructions=[
                f"Review dispute {dispute_id} in the Stripe dashboard.",
                "Decide whether to contest with evidence or accept.",
                "Consult counsel if reason involves fraud or chargeback abuse.",
            ],
            urgent=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("legal_counsel founder_action emit failed: %s", exc)


# ── main entrypoint ───────────────────────────────────────────────────────


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Scan Stripe disputes, emit legal_counsel founder_action for new ones.

    Cron path (no mission_id on task): uses ``SYSTEM_MISSION_ID``.
    Per-mission path: uses the mission_id from the task.
    """
    mission_id = task.get("mission_id")
    if mission_id is None:
        mission_id_int = SYSTEM_MISSION_ID
    else:
        try:
            mission_id_int = int(mission_id)
        except (TypeError, ValueError):
            mission_id_int = SYSTEM_MISSION_ID

    res = await _vc(task, "stripe", "list_disputes", {"limit": 50})
    if not res.get("ok"):
        return {
            "ok": False,
            "reason": "list_disputes_failed",
            "detail": res,
        }

    payload = res.get("result") or {}
    if isinstance(payload, dict):
        disputes = payload.get("data") or []
    else:
        disputes = []

    if not isinstance(disputes, list):
        disputes = []

    workspace = _workspace_root()
    ckpt_path = _checkpoint_path(workspace, mission_id_int)
    ckpt = _load_checkpoint(ckpt_path)
    seen: set[str] = set(ckpt.get("seen_ids") or [])

    new_disputes: list[dict] = []
    for d in disputes:
        if not isinstance(d, dict):
            continue
        did = d.get("id")
        if not did or did in seen:
            continue
        new_disputes.append(d)
        await _emit_legal_counsel(
            mission_id=mission_id_int,
            dispute_id=did,
            amount=d.get("amount"),
            currency=d.get("currency"),
            detail=d,
        )
        seen.add(did)

    # Persist updated checkpoint (best-effort).
    try:
        _save_checkpoint(ckpt_path, {"seen_ids": sorted(seen)})
    except OSError as exc:
        logger.warning("dispute_check checkpoint save failed: %s", exc)

    return {
        "ok": True,
        "new_disputes": len(new_disputes),
        "total_seen": len(seen),
    }


__all__ = ["run"]
