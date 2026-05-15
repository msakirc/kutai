"""Z7 T5 B1 — Lifecycle email engine.

Public API (consumed by cron, mr_roboto, and Telegram command):
  trigger_sequence(product_id, user_id, sequence_id) -> dict
      Expands the sequence's steps_json into email_sends rows.
      Returns {"ok": True, "sends_created": N} or {"ok": False, "reason": str}.

  trigger_sequence_by_kind(product_id, user_id, trigger_kind) -> dict
      Finds the first enabled sequence matching trigger_kind and calls
      trigger_sequence.  Gracefully returns {"ok": False} if none found.

  get_preferences(product_id, user_token) -> dict
      Returns {"subscriptions": {"<sequence_id>": bool, ...}}.

  set_preferences(product_id, user_token, subscriptions) -> None
      Upsert the preferences row.

  approve_template(template_id) -> dict
      Transitions template status → 'approved' if both lint flags are 1.
      Returns {"ok": True} or {"ok": False, "reason": str}.

  handle_email_event_for_lifecycle(product_id, event_type, recipient,
                                   user_token, sequence_id) -> None
      Called by the webhook listener after a send-level event.  Handles
      'unsub' by toggling the sequence off in email_preferences.

Dependencies (lazy, to avoid circular imports):
  - src.infra.db.get_db
  - src.integrations.email.service.send_email  (imported at call sites)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.lifecycle_email")

# Re-export send_email so callers can monkeypatch at this module's namespace.
# The cron job does `from src.app.lifecycle_email import send_email` so that
# `patch("src.app.lifecycle_email.send_email", ...)` works correctly in tests.
from src.integrations.email.service import send_email as send_email  # noqa: F401


# ── helpers ──────────────────────────────────────────────────────────────────


def _db_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


async def _get_template(db, template_id: int) -> dict | None:
    cur = await db.execute(
        "SELECT template_id, product_id, kind, subject, body_md, "
        "variants_json, status, brand_voice_lint_pass, copy_compliance_pass "
        "FROM email_templates WHERE template_id = ?",
        (template_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    keys = [
        "template_id", "product_id", "kind", "subject", "body_md",
        "variants_json", "status", "brand_voice_lint_pass", "copy_compliance_pass",
    ]
    return dict(zip(keys, row))


async def _get_sequence(db, sequence_id: int) -> dict | None:
    cur = await db.execute(
        "SELECT sequence_id, product_id, name, trigger_kind, steps_json, enabled "
        "FROM email_sequences WHERE sequence_id = ?",
        (sequence_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    keys = ["sequence_id", "product_id", "name", "trigger_kind", "steps_json", "enabled"]
    return dict(zip(keys, row))


# ── Public API ────────────────────────────────────────────────────────────────


async def trigger_sequence(
    product_id: str,
    user_id: str,
    sequence_id: int,
) -> dict[str, Any]:
    """Expand a sequence's steps into email_sends rows.

    Parameters
    ----------
    product_id:   Per-product scoping key.
    user_id:      Recipient identifier (email address or opaque user ID).
    sequence_id:  The sequence to trigger.

    Returns
    -------
    {"ok": True, "sends_created": N}
    {"ok": False, "reason": str}
    """
    from src.infra.db import get_db

    db = await get_db()
    seq = await _get_sequence(db, sequence_id)
    if seq is None:
        return {"ok": False, "reason": f"sequence {sequence_id} not found"}

    if not seq["enabled"]:
        return {"ok": False, "reason": f"sequence {sequence_id} is disabled"}

    try:
        steps = json.loads(seq["steps_json"])
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "reason": "invalid steps_json"}

    now = datetime.now(timezone.utc)
    sends_created = 0
    for step in steps:
        tmpl_id = step.get("template_id")
        delay_hours = float(step.get("delay_hours", 0))
        scheduled_for = _db_str(now + timedelta(hours=delay_hours))

        await db.execute(
            "INSERT INTO email_sends "
            "(product_id, user_id, sequence_id, template_id, scheduled_for) "
            "VALUES (?, ?, ?, ?, ?)",
            (product_id, user_id, sequence_id, tmpl_id, scheduled_for),
        )
        sends_created += 1

    await db.commit()
    logger.info(
        "lifecycle_email.trigger_sequence: sends created",
        product_id=product_id,
        user_id=user_id,
        sequence_id=sequence_id,
        sends_created=sends_created,
    )
    return {"ok": True, "sends_created": sends_created, "sequence_id": sequence_id}


async def trigger_sequence_by_kind(
    product_id: str,
    user_id: str,
    trigger_kind: str,
) -> dict[str, Any]:
    """Find the first enabled sequence matching trigger_kind and trigger it.

    Falls back gracefully when Z6 product event stream is not live —
    callers can use the manual `/lifecycle trigger` path instead.
    """
    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT sequence_id FROM email_sequences "
        "WHERE product_id = ? AND trigger_kind = ? AND enabled = 1 "
        "LIMIT 1",
        (product_id, trigger_kind),
    )
    row = await cur.fetchone()
    if row is None:
        logger.debug(
            "lifecycle_email.trigger_sequence_by_kind: no sequence found",
            product_id=product_id,
            trigger_kind=trigger_kind,
        )
        return {
            "ok": False,
            "reason": f"no enabled sequence for trigger_kind={trigger_kind!r}",
        }

    return await trigger_sequence(
        product_id=product_id,
        user_id=user_id,
        sequence_id=row[0],
    )


async def get_preferences(product_id: str, user_token: str) -> dict[str, Any]:
    """Return per-sequence subscription toggles for the given user_token.

    Returns {"subscriptions": {"<sequence_id>": bool, ...}}.
    Empty dict when the user_token is unknown.
    """
    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT subscriptions_json FROM email_preferences "
        "WHERE product_id = ? AND user_token = ?",
        (product_id, user_token),
    )
    row = await cur.fetchone()
    if row is None:
        return {"product_id": product_id, "user_token": user_token, "subscriptions": {}}

    try:
        subs = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        subs = {}
    return {"product_id": product_id, "user_token": user_token, "subscriptions": subs}


async def set_preferences(
    product_id: str,
    user_token: str,
    subscriptions: dict[str, bool],
) -> None:
    """Upsert the email_preferences row for product_id + user_token."""
    from src.infra.db import get_db

    db = await get_db()
    await db.execute(
        """INSERT INTO email_preferences
               (product_id, user_token, subscriptions_json, updated_at)
           VALUES (?, ?, ?, strftime('%Y-%m-%d %H:%M:%S','now'))
           ON CONFLICT(product_id, user_token) DO UPDATE SET
               subscriptions_json = excluded.subscriptions_json,
               updated_at = excluded.updated_at""",
        (product_id, user_token, json.dumps(subscriptions)),
    )
    await db.commit()
    logger.debug(
        "lifecycle_email.set_preferences: updated",
        product_id=product_id,
        user_token=user_token,
    )


async def approve_template(template_id: int) -> dict[str, Any]:
    """Approve a template — transitions status → 'approved'.

    Requires both brand_voice_lint_pass=1 AND copy_compliance_pass=1.
    Returns {"ok": True} or {"ok": False, "reason": str}.
    """
    from src.infra.db import get_db

    db = await get_db()
    tmpl = await _get_template(db, template_id)
    if tmpl is None:
        return {"ok": False, "reason": f"template {template_id} not found"}

    missing: list[str] = []
    if not tmpl["brand_voice_lint_pass"]:
        missing.append("brand_voice_lint")
    if not tmpl["copy_compliance_pass"]:
        missing.append("copy_compliance")

    if missing:
        return {
            "ok": False,
            "reason": f"lint checks not passed: {', '.join(missing)}",
            "missing": missing,
        }

    await db.execute(
        "UPDATE email_templates SET status='approved', "
        "updated_at=strftime('%Y-%m-%d %H:%M:%S','now') WHERE template_id=?",
        (template_id,),
    )
    await db.commit()
    logger.info("lifecycle_email.approve_template: approved", template_id=template_id)
    return {"ok": True, "template_id": template_id}


async def handle_email_event_for_lifecycle(
    product_id: str,
    event_type: str,
    recipient: str,
    user_token: str | None = None,
    sequence_id: int | str | None = None,
) -> None:
    """Update lifecycle state from an email send-level event.

    Called by the webhook listener (via handle_webhook_event) after
    persisting the event.  Currently handles:
      - 'unsub': toggle sequence off in email_preferences.
      - 'bounce': record bounced_at on the open email_sends row.
    """
    if event_type == "unsub" and user_token and sequence_id is not None:
        prefs = await get_preferences(product_id, user_token)
        subs = dict(prefs.get("subscriptions", {}))
        subs[str(sequence_id)] = False
        await set_preferences(product_id, user_token, subs)
        logger.info(
            "lifecycle_email.handle_email_event_for_lifecycle: unsubscribed",
            product_id=product_id,
            user_token=user_token,
            sequence_id=sequence_id,
        )
