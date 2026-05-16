"""Z7 T2A — Email-send shared service.

Public API:
    send_email(product_id, to, subject, body_md, *, headers, idempotency_key)
        -> {"status": "sent"|"quota_blocked"|"suppressed"|"error", ...}

    handle_webhook_event(product_id, provider, raw_payload)
        -> None  (persists events + updates suppression list)

Flow for send_email:
  1. Load product_email_config from DB.
  2. Check suppression list — return {"status": "suppressed", ...} if hit.
  3. Count this-month sends from email_events — return {"status": "quota_blocked"}
     if >= monthly_quota.
  4. Resolve provider adapter + credential.
  5. If EMAIL_TEST_MODE=1 → redirect to FOUNDER_EMAIL.
  6. Call provider.send(); on success: persist a 'sent' email_events row.
  7. Return provider result dict.

Return-value contract (callers: T5 lifecycle-email, B2 changelog):
    {
        "status":     "sent" | "quota_blocked" | "suppressed" | "error",
        "provider":   str | None,
        "message_id": str | None,
        # quota_blocked only:
        "quota":      int,          # monthly_quota cap
        "sent_count": int,          # sends so far this month
        # suppressed only:
        "reason":     str,          # suppression reason from DB
        # error only:
        "error":      str,
    }
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("integrations.email.service")


async def send_email(
    product_id: str,
    to: str,
    subject: str,
    body_md: str,
    *,
    headers: dict[str, str] | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Send an email on behalf of a product.

    See module docstring for the full flow and return-value contract.
    """
    from src.infra.db import get_db

    db = await get_db()

    # 1. Load product config
    cur = await db.execute(
        "SELECT provider, from_domain, api_key_ref, monthly_quota, tier "
        "FROM product_email_config WHERE product_id = ?",
        (product_id,),
    )
    row = await cur.fetchone()
    if row is None:
        logger.warning("email config not found", product_id=product_id)
        return {
            "status": "error",
            "provider": None,
            "message_id": None,
            "reason": f"config not found for product '{product_id}'",
        }

    provider_name = row[0]
    from_domain = row[1]
    api_key_ref = row[2]
    monthly_quota = row[3]
    # tier = row[4]  # reserved for future paid-gating logic

    # 2. Suppression check
    sup_cur = await db.execute(
        "SELECT reason FROM email_suppression "
        "WHERE product_id = ? AND email = ?",
        (product_id, to),
    )
    sup_row = await sup_cur.fetchone()
    if sup_row is not None:
        reason = sup_row[0]
        logger.info(
            "email suppressed",
            product_id=product_id,
            to=to,
            reason=reason,
        )
        return {
            "status": "suppressed",
            "provider": provider_name,
            "message_id": None,
            "reason": reason,
        }

    # 3. Quota check — count this calendar month's sends
    now = datetime.now(timezone.utc)
    month_start = now.strftime("%Y-%m-01 00:00:00")
    cnt_cur = await db.execute(
        "SELECT COUNT(*) FROM email_events "
        "WHERE product_id = ? AND event_type = 'sent' "
        "  AND occurred_at >= ?",
        (product_id, month_start),
    )
    cnt_row = await cnt_cur.fetchone()
    sent_count = cnt_row[0] if cnt_row else 0

    if monthly_quota is not None and sent_count >= monthly_quota:
        logger.warning(
            "email quota blocked",
            product_id=product_id,
            sent_count=sent_count,
            monthly_quota=monthly_quota,
        )
        return {
            "status": "quota_blocked",
            "provider": provider_name,
            "message_id": None,
            "quota": monthly_quota,
            "sent_count": sent_count,
        }

    # 4. Resolve API key
    api_key = await _resolve_api_key(api_key_ref)

    # 5. Test-mode redirect
    actual_to = to
    test_mode = os.getenv("EMAIL_TEST_MODE", "").strip() in ("1", "true", "yes")
    if test_mode:
        founder_email = os.getenv("FOUNDER_EMAIL", "")
        if founder_email:
            logger.info(
                "EMAIL_TEST_MODE: redirecting to founder inbox",
                original_to=to,
                redirect_to=founder_email,
            )
            actual_to = founder_email

    # 6. Instantiate provider + send
    from src.integrations.email.registry import get_provider_class

    provider_cls = get_provider_class(provider_name)
    provider = provider_cls(api_key=api_key or "", from_domain=from_domain or "")

    result = await provider.send(
        to=actual_to,
        subject=subject,
        body_md=body_md,
        headers=headers,
        idempotency_key=idempotency_key,
    )

    # 7. Persist sent event on success
    if result.get("status") == "sent":
        await _insert_email_event(
            db,
            product_id=product_id,
            event_type="sent",
            recipient=to,
            provider=provider_name,
            message_id=result.get("message_id"),
        )

    return result


async def handle_webhook_event(
    product_id: str,
    provider: str,
    raw_payload: dict[str, Any],
) -> None:
    """Parse + persist a provider webhook event; add to suppression if needed.

    Called by the FastAPI webhook listener route for email provider webhooks.
    """
    from src.infra.db import get_db
    from src.integrations.email.registry import get_provider_class

    db = await get_db()

    try:
        provider_cls = get_provider_class(provider)
        adapter = provider_cls(api_key="", from_domain="")
        event = adapter.parse_webhook_event(raw_payload)
    except (KeyError, Exception) as exc:
        logger.error(
            "webhook parse failed",
            product_id=product_id,
            provider=provider,
            exc=str(exc),
        )
        return

    event_type = event.get("event_type", "unknown")
    recipient = event.get("recipient", "")
    message_id = event.get("message_id")
    should_suppress = event.get("should_suppress", False)

    # Persist the event
    await _insert_email_event(
        db,
        product_id=product_id,
        event_type=event_type,
        recipient=recipient,
        provider=provider,
        message_id=message_id,
    )

    # Suppress on bounce / complaint / unsub
    if should_suppress and recipient:
        reason_map = {
            "bounce": "bounce",
            "complaint": "complaint",
            "unsub": "unsub",
        }
        reason = reason_map.get(event_type, event_type)
        await db.execute(
            """INSERT OR IGNORE INTO email_suppression
               (product_id, email, reason, added_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (product_id, recipient, reason),
        )
        await db.commit()
        logger.info(
            "email suppressed after webhook",
            product_id=product_id,
            recipient=recipient,
            reason=reason,
            event_type=event_type,
        )

    # B1 — propagate unsubscribe / spam-complaint to the preference center so
    # email_preferences (not just the suppression list) reflects the opt-out.
    # The preference center is what broadcast fan-out + the lifecycle send job
    # consult; the suppression list alone would not stop new sends being
    # queued (only block them at the provider boundary).
    if event_type in ("unsub", "complaint") and recipient:
        try:
            from src.app.lifecycle_email import handle_email_event_for_lifecycle

            await handle_email_event_for_lifecycle(
                product_id=product_id,
                event_type=event_type,
                recipient=recipient,
            )
        except Exception as exc:
            logger.error(
                "preference-center update failed after webhook",
                product_id=product_id,
                recipient=recipient,
                event_type=event_type,
                exc=str(exc),
            )


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _resolve_api_key(api_key_ref: str | None) -> str | None:
    """Retrieve the API key value from the credential store."""
    if not api_key_ref:
        return None
    try:
        from src.security.credential_store import get_credential

        cred = await get_credential(api_key_ref)
        if cred is None:
            logger.warning("api_key_ref not found in credential store", ref=api_key_ref)
            return None
        # Credential payload may be {"api_key": "..."} or {"token": "..."} etc.
        return (
            cred.get("api_key")
            or cred.get("token")
            or cred.get("key")
            or next(iter(cred.values()), None)
        )
    except Exception as exc:
        logger.error("credential lookup failed", ref=api_key_ref, exc=str(exc))
        return None


async def _insert_email_event(
    db,
    product_id: str,
    event_type: str,
    recipient: str,
    provider: str,
    message_id: str | None = None,
) -> None:
    """Insert a row into email_events."""
    await db.execute(
        """INSERT INTO email_events
           (product_id, event_type, recipient, provider, message_id, occurred_at)
           VALUES (?, ?, ?, ?, ?, datetime('now'))""",
        (product_id, event_type, recipient, provider, message_id),
    )
    await db.commit()
