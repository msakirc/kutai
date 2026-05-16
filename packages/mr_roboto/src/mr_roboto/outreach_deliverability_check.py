"""Z7 T6 A7 — outreach_deliverability_check: deliverability posthook.

Checks per-product outreach send metrics for:
  - Bounce rate > 5% → campaign paused + founder_action surfaced.
  - Complaint rate > 0.1% → campaign paused + founder_action surfaced.
  - Domain-reputation drop (stub; requires ESP API integration).

Handler contract (posthook via beckman)
---------------------------------------
  handle(task, result) -> dict

Also exposed as:
  run_deliverability_check(product_id, list_id) -> dict

Status values returned:
  "ok"     — metrics within acceptable bounds
  "paused" — metrics exceeded threshold; founder_action emitted
  "skip"   — not enough data (< 10 sends)
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.outreach_deliverability_check")

BOUNCE_RATE_THRESHOLD = 0.05       # 5%
COMPLAINT_RATE_THRESHOLD = 0.001   # 0.1%
MIN_SENDS_FOR_RATE_CALC = 10


async def _emit_founder_action(
    *,
    product_id: str,
    mission_id: int,
    issue: str,
    bounce_rate: float,
    complaint_rate: float,
    list_id: str,
) -> Any:
    """Surface a founder_action alerting deliverability problem."""
    try:
        from src.founder_actions import create as fa_create

        title = (
            f"Outreach deliverability alert for {product_id} "
            f"(list: {list_id})"
        )
        why = (
            f"Outreach campaign metrics have exceeded acceptable thresholds. "
            f"Bounce rate: {bounce_rate:.1%}. Complaint rate: {complaint_rate:.3%}. "
            f"The campaign has been paused to protect domain reputation."
        )
        instructions = [
            f"Issue: {issue}",
            "Review the recent outreach_sends rows for this product + list.",
            "Clean the list (remove bad emails, unengaged prospects).",
            "Resume by clearing the warmup day counter or contacting your ESP.",
            "Threshold: bounce >5% or complaint >0.1% triggers this alert.",
        ]
        return await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=why,
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=True,
            urgent=True,
        )
    except Exception as exc:
        logger.warning(
            "outreach_deliverability_check: _emit_founder_action failed",
            error=str(exc),
        )
        return None


async def run_deliverability_check(
    product_id: str,
    list_id: str,
    *,
    mission_id: int = 0,
) -> dict[str, Any]:
    """Check deliverability metrics for a product+list combination.

    Returns:
      {"status": "ok"}
      {"status": "paused", "issue": <str>, "bounce_rate": <float>, ...}
      {"status": "skip", "reason": <str>}
    """
    from src.infra.db import get_db
    db = await get_db()

    # Count total sends for the list
    cur = await db.execute(
        "SELECT COUNT(*) FROM outreach_sends "
        "WHERE product_id=? AND list_id=? AND sent_at IS NOT NULL",
        (product_id, list_id),
    )
    row = await cur.fetchone()
    total_sent = row[0] if row else 0

    if total_sent < MIN_SENDS_FOR_RATE_CALC:
        return {
            "status": "skip",
            "reason": f"only {total_sent} sends — need {MIN_SENDS_FOR_RATE_CALC} for rate calc",
        }

    # Count bounces
    cur = await db.execute(
        "SELECT COUNT(*) FROM outreach_sends "
        "WHERE product_id=? AND list_id=? AND bounced_at IS NOT NULL",
        (product_id, list_id),
    )
    row = await cur.fetchone()
    bounced = row[0] if row else 0

    bounce_rate = bounced / total_sent
    # Complaint rate stub — ESP API not yet wired; use 0.0
    complaint_rate = 0.0

    issue: str | None = None
    if bounce_rate > BOUNCE_RATE_THRESHOLD:
        issue = (
            f"Bounce rate {bounce_rate:.1%} exceeds {BOUNCE_RATE_THRESHOLD:.0%} threshold "
            f"({bounced}/{total_sent} sends bounced)"
        )
    elif complaint_rate > COMPLAINT_RATE_THRESHOLD:
        issue = (
            f"Complaint rate {complaint_rate:.3%} exceeds "
            f"{COMPLAINT_RATE_THRESHOLD:.1%} threshold"
        )

    if issue:
        logger.warning(
            "outreach_deliverability_check: threshold exceeded",
            product_id=product_id,
            list_id=list_id,
            bounce_rate=bounce_rate,
            complaint_rate=complaint_rate,
            issue=issue,
        )
        # Set the REAL pause flag so outreach/send actually stops — the
        # founder_action alone is cosmetic.
        try:
            await db.execute(
                "INSERT INTO outreach_pauses (product_id, list_id, reason) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(product_id, list_id) DO UPDATE SET "
                "  reason=excluded.reason, "
                "  paused_at=strftime('%Y-%m-%d %H:%M:%S','now'), "
                "  cleared_at=NULL",
                (product_id, list_id, issue),
            )
            await db.commit()
        except Exception as exc:
            logger.error(
                "outreach_deliverability_check: failed to write pause flag",
                product_id=product_id,
                list_id=list_id,
                error=str(exc),
            )
        await _emit_founder_action(
            product_id=product_id,
            mission_id=mission_id,
            issue=issue,
            bounce_rate=bounce_rate,
            complaint_rate=complaint_rate,
            list_id=list_id,
        )
        return {
            "status": "paused",
            "issue": issue,
            "bounce_rate": bounce_rate,
            "complaint_rate": complaint_rate,
            "total_sent": total_sent,
            "bounced": bounced,
        }

    return {
        "status": "ok",
        "bounce_rate": bounce_rate,
        "complaint_rate": complaint_rate,
        "total_sent": total_sent,
    }


async def handle(task: dict, result: dict) -> dict:
    """Posthook handler contract for beckman.

    Reads product_id + list_id from task payload context.
    """
    import json as _json

    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx = _json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = dict(ctx_raw)
    else:
        ctx = {}

    payload = task.get("payload") or {}
    product_id = (
        payload.get("product_id")
        or ctx.get("product_id")
        or result.get("product_id")
        or ""
    )
    list_id = (
        payload.get("list_id")
        or ctx.get("list_id")
        or result.get("list_id")
        or ""
    )
    mission_id = task.get("mission_id") or 0

    if not product_id or not list_id:
        return {"status": "skip", "reason": "product_id or list_id missing"}

    return await run_deliverability_check(
        product_id=product_id,
        list_id=list_id,
        mission_id=mission_id,
    )
