"""Z7 T5 B1 — Lifecycle email send job.

Runs every 5 minutes (registered in beckman cron_seed as
``_executor='lifecycle_email_send'``).

Picks email_sends rows where:
  - scheduled_for <= datetime('now')
  - sent_at IS NULL
  - unsubscribed_at IS NULL  (deliberately suppressed rows are skipped)

For each due row:
  1. Load the template (subject + body_md).
  2. Call send_email(product_id, user_id, subject, body_md, ...).
  3. On "sent": mark sent_at = now.
  4. On "suppressed" | "quota_blocked": leave sent_at NULL but log.
  5. On "error": log and leave for next tick (natural retry).

Public API
----------
- ``_pick_due_sends()`` — query helper exposed for tests.
- ``run_lifecycle_email_send()`` — entry point called by mr_roboto
  for the ``lifecycle_email_send`` executor.  Returns {"ok": True} on
  overall success (partial failures are logged, not propagated).
"""
from __future__ import annotations

from src.infra.logging_config import get_logger
from src.infra.times import db_now

logger = get_logger("app.jobs.lifecycle_email_send")

# Maximum sends per tick to avoid overwhelming a free-tier quota guard.
_MAX_PER_TICK = 50


async def _pick_due_sends() -> list[dict]:
    """Return email_sends rows due for sending.

    Filters out both already-sent rows (``sent_at`` set) and deliberately
    suppressed rows (``unsubscribed_at`` set) so suppressed sends are not
    re-polled every tick and are never counted as "sent".
    """
    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT send_id, product_id, user_id, sequence_id, template_id, "
        "scheduled_for "
        "FROM email_sends "
        "WHERE scheduled_for <= strftime('%Y-%m-%d %H:%M:%S','now') "
        "AND sent_at IS NULL "
        "AND unsubscribed_at IS NULL "
        f"LIMIT {_MAX_PER_TICK}"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def run_lifecycle_email_send() -> dict:
    """Main entry point called by mr_roboto ``lifecycle_email_send`` executor.

    Returns {"ok": True, "sent": N, "skipped": M, "errors": [...]} on success.
    Never raises — partial failures are logged and the cron cadence continues.
    """
    from src.infra.db import get_db

    # Resolve send_email lazily so tests can monkeypatch
    # src.app.lifecycle_email.send_email before calling this function.
    import src.app.lifecycle_email as _le_mod
    _get_template = _le_mod._get_template
    _send_email = _le_mod.send_email
    _is_subscribed = _le_mod.is_subscribed

    sent_count = 0
    skipped_count = 0
    errors: list[str] = []

    try:
        due = await _pick_due_sends()
    except Exception as exc:
        logger.error("lifecycle_email_send: failed to pick due sends", error=str(exc))
        return {"ok": True, "sent": 0, "skipped": 0, "errors": [str(exc)]}

    db = await get_db()

    for row in due:
        send_id = row["send_id"]
        product_id = row["product_id"]
        user_id = row["user_id"]
        sequence_id = row.get("sequence_id")
        template_id = row.get("template_id")

        try:
            # Preference gate (B1) — never send to an unsubscribed recipient.
            # email_sends.user_id carries the preference-center user_token.
            # An explicit `false` for this sequence in email_preferences means
            # the user opted out (via unsubscribe link / webhook). We stamp
            # ``unsubscribed_at`` (NOT ``sent_at``) so the cron does not
            # re-evaluate the row every tick AND the send is not miscounted as
            # delivered by any "emails sent" metric reading sent_at.
            if sequence_id is not None and not await _is_subscribed(
                product_id, str(user_id), sequence_id
            ):
                await db.execute(
                    "UPDATE email_sends SET unsubscribed_at=? WHERE send_id=?",
                    (db_now(), send_id),
                )
                await db.commit()
                skipped_count += 1
                logger.info(
                    "lifecycle_email_send: skipped — recipient unsubscribed",
                    send_id=send_id,
                    product_id=product_id,
                    sequence_id=sequence_id,
                )
                continue

            # Load template
            tmpl = await _get_template(db, template_id) if template_id else None
            if tmpl is None:
                logger.warning(
                    "lifecycle_email_send: template not found",
                    send_id=send_id,
                    template_id=template_id,
                )
                skipped_count += 1
                continue

            subject = tmpl.get("subject") or "(no subject)"
            body_md = tmpl.get("body_md") or ""

            result = await _send_email(
                product_id=product_id,
                to=user_id,
                subject=subject,
                body_md=body_md,
                idempotency_key=f"lifecycle-send-{send_id}",
            )

            status = result.get("status", "error")
            if status == "sent":
                await db.execute(
                    "UPDATE email_sends SET sent_at=? WHERE send_id=?",
                    (db_now(), send_id),
                )
                await db.commit()
                sent_count += 1
                logger.info(
                    "lifecycle_email_send: sent",
                    send_id=send_id,
                    product_id=product_id,
                )
            elif status in ("suppressed", "quota_blocked"):
                skipped_count += 1
                logger.info(
                    "lifecycle_email_send: skipped",
                    send_id=send_id,
                    status=status,
                )
            else:
                errors.append(f"send/{send_id}: status={status} error={result.get('error')}")
                logger.warning(
                    "lifecycle_email_send: send error",
                    send_id=send_id,
                    status=status,
                    error=result.get("error"),
                )

        except Exception as exc:
            err = f"send/{send_id}: {exc}"
            errors.append(err)
            logger.error("lifecycle_email_send: exception", send_id=send_id, error=err)

    logger.info(
        "lifecycle_email_send: tick complete",
        sent=sent_count,
        skipped=skipped_count,
        errors=len(errors),
    )
    return {
        "ok": True,
        "sent": sent_count,
        "skipped": skipped_count,
        "errors": errors or None,
    }
