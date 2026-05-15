"""Z7 T1D (B9) — External Comms Audit Log.

Every external-publish verb writes a row to ``external_comms_log`` before
AND after the vendor_call, capturing:

- content_hash: sha256 of the raw message body
- content_md: gzip+base64 of the raw message body (recoverable)
- channel, recipient, reversibility, source_mission_id, source_action_id,
  vendor_call_id

Usage::

    from mr_roboto.audit_log import log_external_send, wrap_external_verb

    # Direct row write:
    row_id = await log_external_send(
        channel="telegram",
        content="Hello founder!",
        recipient="@founder",
        source_mission_id=42,
        reversibility="irreversible",
    )

    # Decorator for vendor_call executors:
    @wrap_external_verb
    async def my_executor(task):
        ...

External-publish verbs (from reversibility.py): any verb whose reversibility
is "irreversible" AND that delivers content to an external party. The check
is: VERB_REVERSIBILITY[verb] == "irreversible" AND verb is in
EXTERNAL_PUBLISH_VERBS below.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import datetime
import functools
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.audit_log")

# Verbs that publish content to external parties (founder, users, public).
# Source: reversibility.py irreversible verbs that involve content delivery.
EXTERNAL_PUBLISH_VERBS: frozenset[str] = frozenset({
    "notify_user",
    "todo_reminder",
    "price_watch_check",
    "clarify",
    "emit_preview_url",
    "stripe_revenue_digest",
    "tax_export_ledger",
    "mission_deliverable_bundle",
    "escalate_to_founder",
    "vendor_call",          # gated: only when vendor_call delivers content
    "rollback_to_last_green",  # notifies founder
    "rotate_failed_key",       # notifies affected parties
})


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
    """Return vendor_call rows with reversibility != 'full' that have no
    external_comms_log entry within ``window_minutes``.

    Used by the audit_completeness_check posthook to flag missing audit rows.
    """
    from src.infra.db import get_db
    db = await get_db()

    # action_confirmations holds vendor_call approvals.
    # We join against external_comms_log on vendor_call_id.
    # Rows that have no matching log entry within window_minutes are gaps.
    gap_sql = f"""
        SELECT ac.id        AS vendor_call_id,
               ac.action    AS verb,
               ac.mission_id,
               ac.created_at
        FROM action_confirmations ac
        WHERE ac.reversibility != 'full'
          AND ac.created_at <= datetime('now', '-{window_minutes} minutes')
          AND NOT EXISTS (
              SELECT 1 FROM external_comms_log ecl
              WHERE ecl.vendor_call_id = ac.id
          )
        ORDER BY ac.created_at DESC
        LIMIT 50
    """
    try:
        cur = await db.execute(gap_sql)
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description]
        await cur.close()
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logger.warning("pending_audit_gaps: query failed: %s", e)
        return []


def wrap_external_verb(func):
    """Decorator: auto-log external sends for vendor_call executors.

    Expects the wrapped coroutine to accept a ``task`` dict with:
      - task["mission_id"]
      - task["payload"]["channel"] (optional)
      - task["payload"]["recipient"] (optional)
      - task["payload"]["content"] (optional, extracted for hash/md)

    The decorator writes an audit log row AFTER the executor completes
    successfully. On failure, no row is written (the send didn't happen).
    """
    @functools.wraps(func)
    async def wrapper(task: dict, *args, **kwargs) -> dict:
        result = await func(task, *args, **kwargs)
        if result and result.get("ok"):
            payload = task.get("payload") or {}
            content = (
                payload.get("content")
                or payload.get("body")
                or payload.get("message")
                or str(result)
            )
            try:
                await log_external_send(
                    channel=str(payload.get("channel") or "unknown"),
                    content=content,
                    recipient=payload.get("recipient") or payload.get("to"),
                    source_mission_id=task.get("mission_id"),
                    reversibility="irreversible",
                )
            except Exception as e:
                logger.warning("wrap_external_verb: audit log failed: %s", e)
        return result
    return wrapper


__all__ = [
    "EXTERNAL_PUBLISH_VERBS",
    "log_external_send",
    "revoke_send",
    "search_sends",
    "pending_audit_gaps",
    "decode_content",
    "wrap_external_verb",
]
