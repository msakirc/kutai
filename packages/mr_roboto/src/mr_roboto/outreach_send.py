"""Z7 T6 A7 — outreach/send: cold outreach email send verb.

Non-LLM mechanical executor. Runs through a 4-gate filter chain before
calling T2A send_email:

  1. Feature flag (OUTREACH_ENABLED env) — disabled → return status='disabled'.
  2. Global suppression (email_suppression table) — known-bad email → suppressed.
  2b. Campaign pause (outreach_pauses table) — un-cleared row → campaign_paused.
  3. Warmup quota (outreach_warmup table) — sent_count >= target_count → quota exceeded.
  4. Jurisdiction check — GDPR target without explicit opt-in → gdpr_blocked.

On pass:
  - Injects mandatory CAN-SPAM / GDPR delivery headers:
      List-Unsubscribe: <{unsub_url}>, mailto:{unsub_email}
      List-Unsubscribe-Post: List-Unsubscribe=One-Click
  - Appends postal address + one-click unsubscribe link to body_md.
  - Calls T2A send_email.
  - Inserts an outreach_sends row.
  - Increments outreach_warmup.sent_count for the target domain.

Public API
----------
  run_outreach_send(
      product_id, list_id, target_email, template_id,
      subject, body_md, postal_address, unsubscribe_base_url,
      *,
      jurisdiction=None,
      has_explicit_opt_in=False,
  ) -> dict
"""
from __future__ import annotations

import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.outreach_send")

# Day-1 warmup quota (50/day) — the most restrictive ramp bucket. A domain
# with no warmup row is treated as day-1 to prevent zero-throttle sends.
_DAY1_QUOTA = 50

# GDPR-jurisdiction TLDs and country codes (non-exhaustive; expand as needed)
_GDPR_TLDS = frozenset({
    ".de", ".fr", ".it", ".es", ".nl", ".be", ".at", ".pl", ".se", ".dk",
    ".fi", ".no", ".pt", ".ie", ".cz", ".hu", ".ro", ".sk", ".bg", ".hr",
    ".si", ".lt", ".lv", ".ee", ".lu", ".mt", ".cy", ".gr",
    ".eu",  # catch-all EU TLD
})


def _is_gdpr_email(email: str) -> bool:
    """Heuristic: email domain ends with a known GDPR-jurisdiction TLD."""
    if not email or "@" not in email:
        return False
    domain = email.rsplit("@", 1)[-1].lower()
    # Check for exact TLD match
    for tld in _GDPR_TLDS:
        if domain.endswith(tld):
            return True
    return False


def _outreach_enabled() -> bool:
    """Return True when OUTREACH_ENABLED is set to '1' or 'true' (case-insensitive)."""
    val = os.getenv("OUTREACH_ENABLED", "0").strip().lower()
    return val in ("1", "true", "yes")


def _build_unsub_url(unsubscribe_base_url: str, send_id: int | None) -> str:
    base = unsubscribe_base_url.rstrip("/")
    if send_id:
        return f"{base}?sid={send_id}"
    return base


def _inject_can_spam(
    body_md: str,
    postal_address: str,
    unsubscribe_url: str,
) -> str:
    """Append CAN-SPAM required disclosures to the email body."""
    footer = (
        "\n\n---\n"
        f"To unsubscribe click here: {unsubscribe_url}\n\n"
        f"{postal_address}"
    )
    return body_md + footer


async def run_outreach_send(
    product_id: str,
    list_id: str,
    target_email: str,
    template_id: str,
    subject: str,
    body_md: str,
    postal_address: str,
    unsubscribe_base_url: str,
    *,
    jurisdiction: str | None = None,
    has_explicit_opt_in: bool = False,
) -> dict[str, Any]:
    """Cold outreach send with full gate chain.

    Returns a dict with at minimum {"status": <str>}.
    status values: disabled | suppressed | campaign_paused |
                   warmup_quota_exceeded | gdpr_blocked | sent | error
    """
    # ── Gate 1: Feature flag ────────────────────────────────────────────────
    if not _outreach_enabled():
        logger.info("outreach_send: feature flag off", product_id=product_id)
        return {
            "status": "disabled",
            "reason": "outreach is disabled (OUTREACH_ENABLED not set to 1)",
        }

    from src.infra.db import get_db
    db = await get_db()

    # ── Gate 2: Suppression check ───────────────────────────────────────────
    cur = await db.execute(
        "SELECT reason FROM email_suppression WHERE product_id=? AND email=?",
        (product_id, target_email),
    )
    sup_row = await cur.fetchone()
    if sup_row is not None:
        logger.info(
            "outreach_send: suppressed",
            product_id=product_id,
            email=target_email,
            reason=sup_row[0],
        )
        return {
            "status": "suppressed",
            "reason": sup_row[0],
            "email": target_email,
        }

    # ── Gate 2b: Campaign pause flag ────────────────────────────────────────
    # outreach_deliverability_check sets an outreach_pauses row when bounce /
    # complaint thresholds are exceeded. Refuse to send while it is un-cleared.
    cur = await db.execute(
        "SELECT reason, paused_at FROM outreach_pauses "
        "WHERE product_id=? AND list_id=? AND cleared_at IS NULL",
        (product_id, list_id),
    )
    pause_row = await cur.fetchone()
    if pause_row is not None:
        logger.warning(
            "outreach_send: campaign paused",
            product_id=product_id,
            list_id=list_id,
            reason=pause_row[0],
        )
        return {
            "status": "campaign_paused",
            "reason": pause_row[0] or "campaign paused by deliverability check",
            "paused_at": pause_row[1],
        }

    # ── Gate 3: Warmup quota ────────────────────────────────────────────────
    # Determine the sending domain from the email
    send_domain = target_email.rsplit("@", 1)[-1] if "@" in target_email else ""
    if send_domain:
        # Find the highest-day warmup row for this product+domain
        cur = await db.execute(
            "SELECT sent_count, target_count, day FROM outreach_warmup "
            "WHERE product_id=? AND domain=? "
            "ORDER BY day DESC LIMIT 1",
            (product_id, send_domain),
        )
        warmup_row = await cur.fetchone()
        if warmup_row is None:
            # No warmup row → seed a day-1 row and enforce the day-1 limit.
            await db.execute(
                "INSERT OR IGNORE INTO outreach_warmup "
                "(product_id, domain, day, sent_count, target_count) "
                "VALUES (?, ?, 1, 0, ?)",
                (product_id, send_domain, _DAY1_QUOTA),
            )
            await db.commit()
            # Re-fetch so the rest of the path works uniformly.
            cur = await db.execute(
                "SELECT sent_count, target_count, day FROM outreach_warmup "
                "WHERE product_id=? AND domain=? AND day=1",
                (product_id, send_domain),
            )
            warmup_row = await cur.fetchone()
        if warmup_row is not None:
            sent_count, target_count, day = warmup_row
            if sent_count >= target_count:
                logger.info(
                    "outreach_send: warmup quota exceeded",
                    product_id=product_id,
                    domain=send_domain,
                    day=day,
                    sent_count=sent_count,
                    target_count=target_count,
                )
                return {
                    "status": "warmup_quota_exceeded",
                    "domain": send_domain,
                    "day": day,
                    "sent_count": sent_count,
                    "target_count": target_count,
                }

    # ── Gate 4: Jurisdiction / GDPR check ──────────────────────────────────
    effective_jurisdiction = jurisdiction
    if effective_jurisdiction is None and _is_gdpr_email(target_email):
        effective_jurisdiction = "GDPR"
    if effective_jurisdiction == "GDPR" and not has_explicit_opt_in:
        logger.info(
            "outreach_send: GDPR jurisdiction blocked — no explicit opt-in",
            product_id=product_id,
            email=target_email,
        )
        return {
            "status": "gdpr_blocked",
            "reason": "GDPR jurisdiction requires explicit opt-in consent",
            "email": target_email,
        }

    # ── Insert outreach_sends row to get send_id for headers ────────────────
    cur = await db.execute(
        "INSERT INTO outreach_sends "
        "(product_id, list_id, target_email, template_id) "
        "VALUES (?, ?, ?, ?)",
        (product_id, list_id, target_email, template_id),
    )
    await db.commit()
    send_id = cur.lastrowid

    # ── Build CAN-SPAM compliant headers and body ────────────────────────────
    unsub_url = _build_unsub_url(unsubscribe_base_url, send_id)
    headers = {
        "List-Unsubscribe": f"<{unsub_url}>, <mailto:unsub@{product_id}.kutai.internal>",
        "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
        "X-Send-ID": str(send_id),
    }
    enriched_body = _inject_can_spam(body_md, postal_address, unsub_url)

    # ── Send via T2A service ─────────────────────────────────────────────────
    from src.integrations.email.service import send_email

    idempotency_key = f"outreach:{product_id}:{send_id}"
    result = await send_email(
        product_id,
        target_email,
        subject,
        enriched_body,
        headers=headers,
        idempotency_key=idempotency_key,
    )

    if result.get("status") == "sent":
        # Update sent_at on the outreach_sends row
        await db.execute(
            "UPDATE outreach_sends SET sent_at=datetime('now') WHERE send_id=?",
            (send_id,),
        )
        # Increment warmup counter for the target domain
        if send_domain:
            await db.execute(
                "UPDATE outreach_warmup "
                "SET sent_count = sent_count + 1, updated_at = datetime('now') "
                "WHERE product_id=? AND domain=? "
                "AND day = (SELECT MAX(day) FROM outreach_warmup "
                "           WHERE product_id=? AND domain=?)",
                (product_id, send_domain, product_id, send_domain),
            )
        await db.commit()
        logger.info(
            "outreach_send: sent",
            product_id=product_id,
            send_id=send_id,
            email=target_email,
        )

    result["send_id"] = send_id
    return result
