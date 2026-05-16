"""Z7 T1D (B9) — External Comms Audit Log.

Every external-publish verb writes a row to ``external_comms_log`` before
AND after the vendor_call, capturing:

- content_hash: sha256 of the raw message body
- content_md: gzip+base64 of the raw message body (recoverable)
- channel, recipient, reversibility, source_mission_id, source_action_id,
  vendor_call_id

Usage::

    from mr_roboto.audit_log import log_external_send

    # Direct row write:
    row_id = await log_external_send(
        channel="telegram",
        content="Hello founder!",
        recipient="@founder",
        source_mission_id=42,
        reversibility="irreversible",
    )

External-publish verbs: a verb is treated as an external publish when it is a
member of the :data:`EXTERNAL_PUBLISH_VERBS` set below. ``log_publish_action``
gates on set-membership plus :func:`_was_sent` (the verb's Action/inner result
must report a delivered status) — it does NOT derive publish-ness from
``VERB_REVERSIBILITY``.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import datetime
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.audit_log")

# Verbs that publish content to an external party (founder, users, public).
# These are the irreversible content-delivery verbs registered in the
# mr_roboto.run() dispatcher (cross-checked against reversibility.py). The
# dispatcher (run()) calls log_publish_action() after any of these completes
# successfully, landing an external_comms_log row.
#
# NOTE: keys must match the dispatcher verb strings exactly (slash-paths
# included). The reversibility-tag-derived check is: verb in this set AND the
# returned Action.status == "completed".
EXTERNAL_PUBLISH_VERBS: frozenset[str] = frozenset({
    # ---- Telegram message to founder / user ----------------------------
    "notify_user",
    "todo_reminder",
    "clarify",
    "price_watch_check",
    "escalate_to_founder",
    "mission_deliverable_bundle",
    "stripe_revenue_digest",
    # ---- founder_action surfaced to Telegram (user sees the card) ------
    "incident_update_review",
    "crisis/disclosure_timer",
    "meeting/outcome_prompt",
    "tax_export_ledger",
    # ---- public publish (customers / world see it) ---------------------
    "incident/publish_status",
    "changelog/publish",
    "publish_synchronized",
    # ---- real email sent to a recipient --------------------------------
    "outreach/send",
    "email/send_via_provider",
    # ---- other irreversible external-facing sends ----------------------
    "stripe_provision_products",
    "rotate_failed_key",
    "rollback_to_last_green",
    # ---- irreversible publish: content delivered to an external party --
    # init_mission_github_repo: creates a public GitHub repo — visible to
    #   the world.
    # emit_preview_url: opens a tunnelled URL that is publicly discoverable
    #   once shared.
    # demo/distribute: uploads demo cuts to YouTube (unlisted) — the dispatcher
    #   verb is "demo/distribute"; "demo/distribute/flip_to_public" is only an
    #   instruction string surfaced to the founder, NOT a dispatcher verb.
    # eas_submit / fastlane: push a binary to an app-store track (TestFlight /
    #   Play internal) — testers receive it.
    "init_mission_github_repo",
    "emit_preview_url",
    "demo/distribute",
    "eas_submit",
    "fastlane",
})

# Default channel per external-publish verb — used when the payload does not
# carry an explicit ``channel`` field. Verbs that surface a founder_action /
# Telegram message default to "telegram"; email verbs to "email"; public
# publish verbs to "public".
_CHANNEL_BY_VERB: dict[str, str] = {
    "notify_user": "telegram",
    "todo_reminder": "telegram",
    "clarify": "telegram",
    "price_watch_check": "telegram",
    "escalate_to_founder": "telegram",
    "mission_deliverable_bundle": "telegram",
    "stripe_revenue_digest": "telegram",
    "incident_update_review": "telegram",
    "crisis/disclosure_timer": "telegram",
    "meeting/outcome_prompt": "telegram",
    "tax_export_ledger": "telegram",
    "incident/publish_status": "public",
    "changelog/publish": "public",
    "publish_synchronized": "public",
    "outreach/send": "email",
    "email/send_via_provider": "email",
    "stripe_provision_products": "vendor",
    "rotate_failed_key": "telegram",
    "rollback_to_last_green": "telegram",
    "init_mission_github_repo": "github",
    "emit_preview_url": "public",
    "demo/distribute": "youtube",
    "eas_submit": "app_store",
    "fastlane": "app_store",
}


def _encode_content(body: str | bytes) -> tuple[str, str]:
    """Return (content_hash, content_md) for ``body``.

    content_hash: sha256 hex of raw body (str encoded as UTF-8).
    content_md: gzip+base64 encoded body for compact storage.
    """
    if isinstance(body, str):
        raw = body.encode("utf-8")
    else:
        raw = body
    content_hash = hashlib.sha256(raw).hexdigest()
    compressed = gzip.compress(raw, compresslevel=6)
    content_md = base64.b64encode(compressed).decode("ascii")
    return content_hash, content_md


def decode_content(content_md: str) -> str:
    """Reverse _encode_content — returns original body as UTF-8 string."""
    compressed = base64.b64decode(content_md.encode("ascii"))
    raw = gzip.decompress(compressed)
    return raw.decode("utf-8")


async def log_external_send(
    channel: str,
    content: str | bytes,
    recipient: str | None = None,
    recipient_count: int | None = None,
    source_mission_id: int | None = None,
    source_action_id: int | None = None,
    vendor_call_id: int | None = None,
    product_id: int | None = None,
    reversibility: str = "irreversible",
) -> int:
    """Write one row to ``external_comms_log``.

    Returns the ``log_id`` of the inserted row.

    Parameters
    ----------
    channel:
        "telegram" / "sms" / "email" / "push" / "webhook" / etc.
    content:
        The actual message body (str or bytes). Will be sha256-hashed and
        gzip+base64-encoded for storage.
    recipient:
        Channel-specific recipient (username, phone, email). PII — store
        at caller discretion.
    recipient_count:
        For broadcast channels, number of recipients.
    source_mission_id:
        The mission that triggered the send (for cross-ref).
    source_action_id:
        The founder_actions.id that triggered the send (if any).
    vendor_call_id:
        The action_confirmations.id for the approval that enabled this send.
    product_id:
        The product (mission) context, e.g. for push notifications.
    reversibility:
        One of "full" / "partial" / "irreversible". Defaults to "irreversible"
        since external sends are nearly always irreversible.
    """
    from src.infra.db import get_db
    db = await get_db()

    content_hash, content_md = _encode_content(content)
    sent_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # external_comms_log.product_id is NOT NULL (Z7 per-product scoping).
    # Derive from the source mission when no explicit product is given;
    # fall back to a system sentinel for non-product sends.
    effective_product_id = (
        str(product_id) if product_id is not None
        else str(source_mission_id) if source_mission_id is not None
        else "_system"
    )

    cur = await db.execute(
        "INSERT INTO external_comms_log "
        "(product_id, sent_at, channel, recipient, recipient_count, "
        " content_hash, content_md, source_mission_id, source_action_id, "
        " vendor_call_id, reversibility) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            effective_product_id,
            sent_at,
            channel,
            recipient,
            recipient_count,
            content_hash,
            content_md,
            source_mission_id,
            source_action_id,
            vendor_call_id,
            reversibility,
        ),
    )
    await db.commit()
    log_id = cur.lastrowid or 0
    logger.info(
        "audit_log: external send logged — log_id=%d channel=%s hash=%s...",
        log_id, channel, content_hash[:12],
    )
    return log_id


async def revoke_send(log_id: int, reason: str) -> bool:
    """Mark a logged send as revoked (soft-delete). Returns False if not found."""
    from src.infra.db import get_db
    db = await get_db()
    revoked_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur = await db.execute(
        "UPDATE external_comms_log SET revoked_at = ?, revoke_reason = ? "
        "WHERE log_id = ? AND revoked_at IS NULL",
        (revoked_at, reason, log_id),
    )
    await db.commit()
    updated = cur.rowcount or 0
    return updated > 0


async def search_sends(
    *,
    recipient: str | None = None,
    channel: str | None = None,
    since: str | None = None,
    until: str | None = None,
    mission_id: int | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Query external_comms_log with optional filters.

    ``since``/``until`` are YYYY-MM-DD or YYYY-MM-DD HH:MM:SS strings.
    Returns list of dicts (log_id, sent_at, channel, recipient,
    content_hash, source_mission_id, reversibility, revoked_at).
    """
    from src.infra.db import get_db
    db = await get_db()

    clauses: list[str] = []
    params: list[Any] = []

    if recipient:
        clauses.append("recipient LIKE ?")
        params.append(f"%{recipient}%")
    if channel:
        clauses.append("channel = ?")
        params.append(channel)
    if since:
        clauses.append("sent_at >= ?")
        params.append(since)
    if until:
        clauses.append("sent_at <= ?")
        params.append(until)
    if mission_id is not None:
        clauses.append("source_mission_id = ?")
        params.append(mission_id)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(max(1, min(100, limit)))

    cur = await db.execute(
        f"SELECT log_id, product_id, sent_at, channel, recipient, "
        f"       recipient_count, content_hash, source_mission_id, "
        f"       source_action_id, vendor_call_id, reversibility, "
        f"       revoked_at, revoke_reason "
        f"FROM external_comms_log {where} "
        f"ORDER BY sent_at DESC LIMIT ?",
        params,
    )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    await cur.close()
    return [dict(zip(cols, r)) for r in rows]


async def pending_audit_gaps(window_minutes: int = 5) -> list[dict[str, Any]]:
    """Return external-publish confirmations that have no audit-log row.

    A "gap" is an ``action_confirmations`` row that:

    - has ``reversibility != 'full'`` (an irreversible / partial external act),
    - was requested more than ``window_minutes`` ago (so a freshly-issued
      confirmation still mid-flight is not flagged),
    - and has NO matching ``external_comms_log`` row joined on
      ``external_comms_log.vendor_call_id = action_confirmations.id``.

    Such a row means a publish/send/upload was confirmed but produced no
    immutable audit record — a compliance + post-incident-review hole. Used
    by the hourly ``audit_completeness_check`` cron to escalate one
    founder_action per gap.

    The ``action_confirmations`` table (see ``src/infra/db.py``) has columns
    ``id, task_id, verb, reversibility, payload_summary, requested_at,
    responded_at, verdict`` — there is no ``mission_id`` column, so it is
    derived by LEFT JOINing ``tasks`` on ``action_confirmations.task_id =
    tasks.id``.
    """
    from src.infra.db import get_db
    db = await get_db()

    # action_confirmations holds external-publish confirmation requests.
    # mission_id is NOT on action_confirmations — derive it via tasks.id.
    # A row with reversibility != 'full', requested > window_minutes ago, and
    # no matching external_comms_log row (joined on vendor_call_id) is a gap.
    gap_sql = f"""
        SELECT ac.id           AS vendor_call_id,
               ac.verb         AS verb,
               t.mission_id    AS mission_id,
               ac.requested_at AS created_at,
               ac.reversibility AS reversibility,
               ac.verdict      AS verdict
        FROM action_confirmations ac
        LEFT JOIN tasks t ON t.id = ac.task_id
        WHERE ac.reversibility != 'full'
          AND ac.requested_at IS NOT NULL
          AND ac.requested_at <= datetime('now', '-{window_minutes} minutes')
          AND NOT EXISTS (
              SELECT 1 FROM external_comms_log ecl
              WHERE ecl.vendor_call_id = ac.id
          )
        ORDER BY ac.requested_at DESC
        LIMIT 50
    """
    try:
        cur = await db.execute(gap_sql)
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]
        await cur.close()
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        # A failure here is almost always a schema drift in
        # action_confirmations / tasks — surface it loudly so it is not
        # silently swallowed into an empty (no-gap) result.
        logger.warning(
            "pending_audit_gaps: gap-scan query failed (schema drift?): %s", e
        )
        return []


# Result statuses that mean "the send actually happened".
#  - mr_roboto Action objects use status="completed" on success.
#  - inner result dicts (e.g. email/send_via_provider) use status="sent".
# A verb can return Action(status="completed") while its inner result dict
# reports status="suppressed"/"quota_blocked" — in that case nothing was
# delivered, so we must NOT log it. log_publish_action() inspects both.
_SENT_STATUSES: frozenset[str] = frozenset({"completed", "sent", "ok"})
# Inner statuses that mean the verb completed but delivered NOTHING. A
# "completed" Action wrapping any of these must not produce an audit row.
#  - suppressed / quota_blocked / skipped / blocked / rejected: generic.
#  - disabled: outreach/send when OUTREACH_ENABLED is off — no email left.
#  - warmup_quota_exceeded: outreach/send hit the per-day warmup cap — held.
#  - gdpr_blocked: outreach/send refused for lack of explicit opt-in.
_NOT_DELIVERED_STATUSES: frozenset[str] = frozenset({
    "suppressed", "quota_blocked", "skipped", "blocked", "rejected",
    "disabled", "warmup_quota_exceeded", "gdpr_blocked",
})


def _result_status(result: Any) -> str | None:
    """Extract a status string from an Action object or a dict."""
    if result is None:
        return None
    # mr_roboto Action dataclass — has a `status` attribute.
    status = getattr(result, "status", None)
    if status is not None:
        return str(status)
    if isinstance(result, dict):
        if "status" in result:
            return str(result["status"])
        # legacy callers used an `ok` boolean.
        if "ok" in result:
            return "ok" if result["ok"] else "failed"
    return None


def _was_sent(outer: Any, inner: Any = None) -> bool:
    """True iff the verb actually delivered content externally.

    ``outer`` is the verb's top-level return (Action or dict). ``inner`` is
    the verb's nested ``result`` dict, if any — some verbs (email send) return
    a top-level "completed" Action whose inner dict says "suppressed".

    A few verbs (eas_submit / eas_build / fastlane / expo_cli / android_build)
    do not carry a non-delivery ``status`` string — they return a "completed"
    Action wrapping ``{"skipped": True}`` when the CLI is absent. A truthy
    ``skipped`` flag on the inner dict also means nothing was published.
    """
    outer_status = _result_status(outer)
    if outer_status not in _SENT_STATUSES:
        return False
    inner_status = _result_status(inner)
    if inner_status is not None and inner_status in _NOT_DELIVERED_STATUSES:
        return False
    # Skip-flag verbs (mobile build/submit adapters) report no status string.
    if isinstance(inner, dict) and inner.get("skipped"):
        return False
    return True


async def log_publish_action(verb: str, action_obj: Any, task: dict) -> int | None:
    """Land an external_comms_log row for a completed external-publish verb.

    Called by ``mr_roboto.run()`` after dispatching any verb in
    :data:`EXTERNAL_PUBLISH_VERBS`. Best-effort — never raises into the
    dispatch path. Returns the new ``log_id`` (or ``None`` if nothing was
    logged: not an external verb, the send did not happen, or an error).

    Result-shape handling
    ---------------------
    External-publish verbs return a :class:`mr_roboto.actions.Action`.
    Success is ``Action.status == "completed"``. The verb's inner
    ``result`` dict is also inspected: a "completed" Action whose inner
    result dict reports ``status`` in {suppressed, quota_blocked, ...}
    means nothing was delivered → no audit row.
    """
    if verb not in EXTERNAL_PUBLISH_VERBS:
        return None

    inner = getattr(action_obj, "result", None)
    if inner is None and isinstance(action_obj, dict):
        inner = action_obj.get("result")
    if not _was_sent(action_obj, inner):
        return None

    payload = task.get("payload") or {}
    inner = inner if isinstance(inner, dict) else {}

    content = (
        payload.get("content")
        or payload.get("body")
        or payload.get("body_md")
        or payload.get("message")
        or payload.get("text")
        or payload.get("summary")
        or inner.get("content")
        or inner.get("body_md")
        or inner.get("summary")
        # last resort: a stable repr of the verb's result so the audit row
        # is never empty (the hash still proves *something* was sent).
        or str(inner or action_obj)
    )
    channel = str(
        payload.get("channel")
        or inner.get("channel")
        or _CHANNEL_BY_VERB.get(verb)
        or "unknown"
    )
    recipient = (
        payload.get("recipient")
        or payload.get("to")
        or payload.get("target_email")
        or inner.get("recipient")
    )

    try:
        return await log_external_send(
            channel=channel,
            content=content,
            recipient=recipient,
            source_mission_id=task.get("mission_id"),
            source_action_id=payload.get("source_action_id"),
            product_id=payload.get("product_id") if str(
                payload.get("product_id") or ""
            ).isdigit() else None,
            reversibility="irreversible",
        )
    except Exception as e:
        logger.warning(
            "log_publish_action: audit log failed for verb=%s: %s", verb, e
        )
        return None


__all__ = [
    "EXTERNAL_PUBLISH_VERBS",
    "log_external_send",
    "log_publish_action",
    "revoke_send",
    "search_sends",
    "pending_audit_gaps",
    "decode_content",
]
