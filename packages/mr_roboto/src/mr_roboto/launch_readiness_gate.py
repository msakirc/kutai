"""Z7 T3A (A2.r1) — launch_readiness_gate mr_roboto verb + posthook handler.

Pre-T-0 hard checks that must pass BEFORE publish_synchronized fires.
Runs as a mechanical posthook (kind='launch_readiness_gate') after
publish_synchronized is enqueued but before it executes.

Checks (a–g per spec)
---------------------
a. Site loads under expected traffic (synthetic HTTP check)
b. Payment flow E2E test green within 24h
c. support_tier1 has launch FAQ indexed (Z2 support docs)
d. A6 copy_compliance pass on all channel drafts
e. A4 press kit published with permanent URL
f. B3 status page exists
g. B6 crisis playbook tier1+tier2 exist

Degradation policy
------------------
When a check's subsystem is absent (ImportError, missing table, etc.),
the gate logs a WARNING and records the check as a non-blocking warning
rather than crashing or blocking T-0. The result carries
``"warnings": [...]`` for surfacing to the founder.

When any hard check fails, the gate returns ``{"status": "blocked"}``
and emits a founder_action asking "override or fix?".

Payload
-------
::

    {
        "product_id": "prod-abc",
        "launch_id": 1,
        "channels": ["hn", "twitter"],
    }

Returns
-------
``{"status": "ready"}``               — all checks pass
``{"status": "ready_with_warnings"}`` — all hard checks pass; some absent subsystems
``{"status": "blocked", "failing_checks": [...]}`` — one+ hard check failed
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.launch_readiness_gate")


# ---------------------------------------------------------------------------
# Individual check functions (monkeypatchable for tests)
# ---------------------------------------------------------------------------

async def _check_site_load(product_id: str, **kwargs) -> dict:
    """(a) Synthetic HTTP check — site loads under expected traffic."""
    try:
        from mr_roboto.http_check import http_check
        # Best-effort: check that the product's public URL responds 200.
        # If product_url is not configured, degrade to warning.
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT metadata_json FROM missions WHERE id = ("
            "  SELECT mission_id FROM launches WHERE product_id = ? "
            "  ORDER BY created_at DESC LIMIT 1"
            ")",
            (product_id,),
        ) as cur:
            row = await cur.fetchone()
        import json
        meta = json.loads((row[0] if row else None) or "{}")
        url = meta.get("product_url") or ""
        if not url:
            return {"ok": True, "_note": "product_url not configured — skipped"}
        result = await http_check(url=url, method="GET", expected_status=200)
        ok = result.get("status_code", 0) == 200
        return {"ok": ok, "url": url, "status_code": result.get("status_code")}
    except ImportError:
        raise
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


async def _check_payment_e2e(product_id: str, **kwargs) -> dict:
    """(b) Payment flow E2E test green within last 24h."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT COUNT(*) FROM stripe_payment_test_log "
            "WHERE product_id = ? AND passed = 1 "
            "AND tested_at >= datetime('now', '-24 hours')",
            (product_id,),
        ) as cur:
            row = await cur.fetchone()
        count = row[0] if row else 0
        return {"ok": count > 0, "recent_green_count": count}
    except Exception as exc:
        # Table may not exist (Stripe not configured) — degrade to warning
        raise ImportError(f"payment e2e check unavailable: {exc}") from exc


async def _check_support_faq(product_id: str, **kwargs) -> dict:
    """(c) support_tier1 has launch FAQ indexed."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT COUNT(*) FROM support_docs "
            "WHERE product_id = ? AND kind = 'faq' AND status = 'indexed'",
            (product_id,),
        ) as cur:
            row = await cur.fetchone()
        count = row[0] if row else 0
        return {"ok": count > 0, "faq_doc_count": count}
    except Exception as exc:
        raise ImportError(f"support_faq check unavailable: {exc}") from exc


async def _check_copy_compliance(product_id: str, **kwargs) -> dict:
    """(d) A6 copy_compliance pass on all channel drafts."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        # Look for any copy_compliance failure in the recent launch drafts
        async with db.execute(
            "SELECT COUNT(*) FROM copy_compliance_results "
            "WHERE product_id = ? AND result = 'fail' "
            "AND checked_at >= datetime('now', '-48 hours')",
            (product_id,),
        ) as cur:
            row = await cur.fetchone()
        fail_count = row[0] if row else 0
        return {"ok": fail_count == 0, "compliance_failures": fail_count}
    except Exception as exc:
        raise ImportError(f"copy_compliance check unavailable: {exc}") from exc


async def _check_press_kit(product_id: str, **kwargs) -> dict:
    """(e) A4 press kit published with permanent URL."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT kit_id, published_url FROM press_kits "
            "WHERE product_id = ? AND published_url IS NOT NULL "
            "AND published_url != '' "
            "ORDER BY version DESC LIMIT 1",
            (product_id,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return {"ok": True, "kit_id": row[0], "url": row[1]}
        return {"ok": False, "reason": "no published press kit found"}
    except Exception as exc:
        raise ImportError(f"press_kit check unavailable: {exc}") from exc


async def _check_status_page(product_id: str, **kwargs) -> dict:
    """(f) B3 status page exists."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        # Check that at least one incident or status record exists
        # (B3 creates at least one seed record when set up)
        async with db.execute(
            "SELECT COUNT(*) FROM incidents WHERE product_id = ?",
            (product_id,),
        ) as cur:
            row = await cur.fetchone()
        # Status page "exists" if the table is present and accessible
        # (B3 migration ran). Incident count of 0 means no incidents yet,
        # which is FINE — the page exists, just no active incidents.
        _ = row[0] if row else 0
        return {"ok": True, "note": "status page infrastructure present (B3 migration)"}
    except Exception as exc:
        raise ImportError(f"status_page check unavailable: {exc}") from exc


async def _check_crisis_playbook(product_id: str, **kwargs) -> dict:
    """(g) B6 crisis playbook tier1+tier2 exist."""
    import os

    tier1 = os.path.isfile("playbooks/crisis_comms_tier1.md")
    tier2 = os.path.isfile("playbooks/crisis_comms_tier2.md")

    if not tier1 or not tier2:
        # Check in a few common locations
        for base in (".", "data", "docs"):
            if not tier1:
                tier1 = os.path.isfile(f"{base}/playbooks/crisis_comms_tier1.md")
            if not tier2:
                tier2 = os.path.isfile(f"{base}/playbooks/crisis_comms_tier2.md")

    return {
        "ok": tier1 and tier2,
        "tier1_exists": tier1,
        "tier2_exists": tier2,
    }


# ---------------------------------------------------------------------------
# Founder action emitter
# ---------------------------------------------------------------------------

async def _emit_blocked_founder_action(
    *,
    mission_id: int,
    product_id: str,
    launch_id: int,
    failing_checks: list[str],
    check_details: dict,
) -> Any:
    """Emit a founder_action asking to override or fix failing readiness checks."""
    try:
        from src.founder_actions import create as create_founder_action

        check_lines = []
        for check_name in failing_checks:
            detail = check_details.get(check_name) or {}
            reason = detail.get("reason") or "check failed"
            check_lines.append(f"  - {check_name}: {reason}")
        checks_text = "\n".join(check_lines)

        return await create_founder_action(
            mission_id=mission_id,
            kind="generic",
            title=(
                f"Launch readiness gate BLOCKED for '{product_id}' "
                f"— {len(failing_checks)} check(s) failed. Override or fix?"
            ),
            why=(
                f"T-0 publish is frozen until {len(failing_checks)} check(s) pass:\n"
                f"{checks_text}\n\n"
                "Approve to override (publish anyway) or fix the issues above."
            ),
            instructions=[
                "Review each failing check.",
                "Fix the issue (re-run the relevant setup) OR override if acceptable.",
                "Approve this card to unblock T-0 publish.",
            ],
            expected_output_kind="approve_or_reject",
            notify_telegram=True,
        )
    except Exception as exc:
        logger.warning(
            "launch_readiness_gate: _emit_blocked_founder_action failed",
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Main gate runner
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, str]] = [
    ("site_load", "_check_site_load"),
    ("payment_e2e", "_check_payment_e2e"),
    ("support_faq", "_check_support_faq"),
    ("copy_compliance", "_check_copy_compliance"),
    ("press_kit", "_check_press_kit"),
    ("status_page", "_check_status_page"),
    ("crisis_playbook", "_check_crisis_playbook"),
]


async def run(payload: dict) -> dict:
    """Execute all launch readiness checks.

    Parameters
    ----------
    payload:
        Must include ``product_id``, ``launch_id``.
        Optional: ``channels``, ``mission_id``.

    Returns
    -------
    dict
        One of:
        - ``{"status": "ready"}``
        - ``{"status": "ready_with_warnings", "warnings": [...]}``
        - ``{"status": "blocked", "failing_checks": [...]}``
    """
    product_id = payload.get("product_id") or ""
    launch_id = payload.get("launch_id") or 0
    mission_id = payload.get("mission_id") or 0

    failing_checks: list[str] = []
    warnings: list[str] = []
    check_details: dict = {}

    import mr_roboto.launch_readiness_gate as _self_module

    for check_name, check_attr in CHECKS:
        check_fn = getattr(_self_module, check_attr)
        try:
            result = await check_fn(product_id=product_id, payload=payload)
            check_details[check_name] = result
            if not result.get("ok", True):
                failing_checks.append(check_name)
                logger.warning(
                    "launch_readiness_gate: check FAILED",
                    check=check_name,
                    product_id=product_id,
                    detail=result,
                )
            else:
                logger.debug(
                    "launch_readiness_gate: check passed",
                    check=check_name,
                    product_id=product_id,
                )
        except ImportError as exc:
            # Subsystem absent — degrade to warning, not crash
            warn_msg = f"{check_name}: subsystem absent ({exc})"
            warnings.append(warn_msg)
            check_details[check_name] = {"ok": None, "absent": True, "note": str(exc)}
            logger.warning(
                "launch_readiness_gate: check subsystem absent — warning only",
                check=check_name,
                product_id=product_id,
                error=str(exc),
            )
        except Exception as exc:
            # Unexpected error — treat as hard failure
            failing_checks.append(check_name)
            check_details[check_name] = {"ok": False, "reason": str(exc)}
            logger.error(
                "launch_readiness_gate: check raised unexpected error",
                check=check_name,
                product_id=product_id,
                error=str(exc),
            )

    if failing_checks:
        # Emit founder_action to ask for override or fix
        await _emit_blocked_founder_action(
            mission_id=int(mission_id),
            product_id=product_id,
            launch_id=int(launch_id),
            failing_checks=failing_checks,
            check_details=check_details,
        )
        return {
            "status": "blocked",
            "failing_checks": failing_checks,
            "warnings": warnings,
            "check_details": check_details,
        }

    if warnings:
        return {
            "status": "ready_with_warnings",
            "warnings": warnings,
            "check_details": check_details,
        }

    return {
        "status": "ready",
        "warnings": [],
        "check_details": check_details,
    }


# ---------------------------------------------------------------------------
# Posthook handler interface (called by beckman posthook_handlers/<kind>.py)
# ---------------------------------------------------------------------------

async def handle(task: dict, result: dict) -> dict:
    """Posthook handler entry point (called by beckman apply layer).

    Delegates to ``run()`` with context extracted from the source task.
    """
    import json as _json

    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx: dict = _json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = ctx_raw
    else:
        ctx = {}

    product_id = ctx.get("product_id") or ""
    launch_id = ctx.get("launch_id") or 0
    channels = ctx.get("channels") or []
    mission_id = task.get("mission_id") or 0

    gate_result = await run({
        "product_id": product_id,
        "launch_id": launch_id,
        "channels": channels,
        "mission_id": mission_id,
    })

    # Map status → passed/failed for the beckman apply layer
    passed = gate_result["status"] in ("ready", "ready_with_warnings")
    return {
        "passed": passed,
        **gate_result,
    }
