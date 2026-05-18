"""Z7 T4 A10 — CRM-as-interaction-log + A10.r1 Consent ledger.

This module is the data layer for:
  - relationships: Telegram-native contact directory (NOT a relationship graph)
  - interactions: structured interaction log (calls, meetings, messages, etc.)
  - consent_records: per-purpose consent tracking

Public API (imported by B4/B7 and Telegram commands):
------------------------------------------------------
  add_contact(product_id, handle, display_name, category, **kwargs) -> int
  get_contact_by_handle(product_id, handle) -> dict | None
  list_contacts(product_id, category=None) -> list[dict]

  log_interaction(product_id, contact_id, kind, summary,
                  follow_up=None, next_action=None, mission_id=None) -> int
  list_interactions(product_id, contact_id=None) -> list[dict]
  get_pending_follow_ups(product_id, within_days=7) -> list[dict]
  mark_follow_up_done(interaction_id) -> None

  grant_consent(product_id, contact_id, purpose,
                source_evidence_url, expires_at=None) -> int
  revoke_consent(product_id, contact_id, purpose) -> None
  check_consent(product_id, contact_id, purpose) -> dict | None
  has_consent(product_id, contact_id, purpose) -> bool

Relative follow-up window syntax supported by log_interaction follow_up param:
  Nd  — N days   (e.g. "3d")
  Nw  — N weeks  (e.g. "2w")
  Nm  — N months (approximate: N*30 days) (e.g. "1m")
  YYYY-MM-DD — absolute date string

B4 (meeting brief) writes interactions rows directly.
B7 (interview pipeline) writes interactions rows with kind='interview'.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.crm")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CATEGORIES = frozenset({
    "customer", "prospect", "investor", "journalist",
    "partner", "advisor", "candidate", "vendor", "other",
})

VALID_INTERACTION_KINDS = frozenset({
    "call", "email", "meeting", "message", "event", "interview", "other",
})

VALID_CONSENT_PURPOSES = frozenset({
    "quote_use", "data_processing", "marketing_email",
    "interview_recording", "case_study",
})


def _parse_follow_up(follow_up: str | None) -> str | None:
    """Parse a relative follow-up window to a DB-compatible datetime string.

    Accepts:
      "Nd"  → now + N days
      "Nw"  → now + N weeks
      "Nm"  → now + N*30 days
      "YYYY-MM-DD" → that date at 09:00 UTC
      None  → None

    Returns: "%Y-%m-%d %H:%M:%S" formatted string, or None.
    """
    if not follow_up:
        return None
    follow_up = follow_up.strip().lower()
    now = datetime.now(timezone.utc)

    m = re.fullmatch(r"(\d+)d", follow_up)
    if m:
        delta = timedelta(days=int(m.group(1)))
        return (now + delta).strftime("%Y-%m-%d %H:%M:%S")

    m = re.fullmatch(r"(\d+)w", follow_up)
    if m:
        delta = timedelta(weeks=int(m.group(1)))
        return (now + delta).strftime("%Y-%m-%d %H:%M:%S")

    m = re.fullmatch(r"(\d+)m", follow_up)
    if m:
        delta = timedelta(days=int(m.group(1)) * 30)
        return (now + delta).strftime("%Y-%m-%d %H:%M:%S")

    # Absolute date
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})", follow_up)
    if m:
        return f"{m.group(1)} 09:00:00"

    logger.warning("crm: unrecognised follow_up format", follow_up=follow_up)
    return None


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


async def add_contact(
    product_id: str,
    handle: str,
    display_name: str,
    category: str = "other",
    *,
    email: str | None = None,
    links_json: str | None = None,
    notes_md: str | None = None,
) -> int:
    """Upsert a contact in the relationships table.

    Returns the contact_id (new or existing).
    """
    from src.infra.db import get_db
    db = await get_db()

    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {sorted(VALID_CATEGORIES)}")

    # Check for existing contact with same product_id + handle
    cur = await db.execute(
        "SELECT contact_id FROM relationships WHERE product_id=? AND handle=?",
        (product_id, handle),
    )
    existing = await cur.fetchone()

    if existing:
        contact_id = existing[0]
        await db.execute(
            "UPDATE relationships SET display_name=?, category=?, email=?, "
            "links_json=?, notes_md=? WHERE contact_id=?",
            (display_name, category, email, links_json, notes_md, contact_id),
        )
        await db.commit()
        logger.info("crm: updated contact", contact_id=contact_id, handle=handle)
        return contact_id

    cur = await db.execute(
        "INSERT INTO relationships "
        "(product_id, handle, display_name, category, email, links_json, notes_md, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%S','now'))",
        (product_id, handle, display_name, category, email, links_json, notes_md),
    )
    await db.commit()
    contact_id = cur.lastrowid
    logger.info("crm: created contact", contact_id=contact_id, handle=handle)
    return contact_id


async def get_contact_by_handle(product_id: str, handle: str) -> dict | None:
    """Return a contact dict or None if not found."""
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT contact_id, product_id, handle, display_name, category, "
        "email, links_json, notes_md, created_at "
        "FROM relationships WHERE product_id=? AND handle=?",
        (product_id, handle),
    )
    row = await cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


async def get_contact_by_id(contact_id: int) -> dict | None:
    """Return a contact dict by ID or None."""
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT contact_id, product_id, handle, display_name, category, "
        "email, links_json, notes_md, created_at "
        "FROM relationships WHERE contact_id=?",
        (contact_id,),
    )
    row = await cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


async def list_contacts(
    product_id: str,
    category: str | None = None,
) -> list[dict]:
    """List contacts for a product, optionally filtered by category.

    Each returned dict includes a ``last_interaction`` key (datetime string or None).
    """
    from src.infra.db import get_db
    db = await get_db()

    if category:
        cur = await db.execute(
            "SELECT r.contact_id, r.product_id, r.handle, r.display_name, "
            "r.category, r.email, r.created_at, "
            "MAX(i.logged_at) AS last_interaction "
            "FROM relationships r "
            "LEFT JOIN interactions i ON i.contact_id = r.contact_id AND i.product_id = r.product_id "
            "WHERE r.product_id=? AND r.category=? "
            "GROUP BY r.contact_id "
            "ORDER BY r.display_name",
            (product_id, category),
        )
    else:
        cur = await db.execute(
            "SELECT r.contact_id, r.product_id, r.handle, r.display_name, "
            "r.category, r.email, r.created_at, "
            "MAX(i.logged_at) AS last_interaction "
            "FROM relationships r "
            "LEFT JOIN interactions i ON i.contact_id = r.contact_id AND i.product_id = r.product_id "
            "WHERE r.product_id=? "
            "GROUP BY r.contact_id "
            "ORDER BY r.display_name",
            (product_id,),
        )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in await cur.fetchall()]


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


async def log_interaction(
    product_id: str,
    contact_id: int,
    kind: str,
    summary: str,
    *,
    follow_up: str | None = None,
    next_action: str | None = None,
    mission_id: int | None = None,
) -> int:
    """Write one interaction row.

    follow_up accepts relative shorthand: "2w", "3d", "1m", or "YYYY-MM-DD".
    Returns interaction_id.
    """
    from src.infra.db import get_db
    db = await get_db()

    if kind not in VALID_INTERACTION_KINDS:
        raise ValueError(f"Invalid interaction kind '{kind}'. Must be one of: {sorted(VALID_INTERACTION_KINDS)}")

    follow_up_at = _parse_follow_up(follow_up)

    cur = await db.execute(
        "INSERT INTO interactions "
        "(product_id, contact_id, kind, summary, next_action, follow_up_at, "
        "logged_at, mission_id, done) "
        "VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%S','now'), ?, 0)",
        (product_id, contact_id, kind, summary, next_action, follow_up_at, mission_id),
    )
    await db.commit()
    interaction_id = cur.lastrowid
    logger.info(
        "crm: logged interaction",
        interaction_id=interaction_id,
        contact_id=contact_id,
        kind=kind,
        follow_up_at=follow_up_at,
    )
    return interaction_id


async def list_interactions(
    product_id: str,
    contact_id: int | None = None,
    limit: int = 50,
) -> list[dict]:
    """List interactions, optionally filtered by contact_id.

    Returns most-recent-first.
    """
    from src.infra.db import get_db
    db = await get_db()

    if contact_id is not None:
        cur = await db.execute(
            "SELECT interaction_id, product_id, contact_id, kind, summary, "
            "next_action, follow_up_at, logged_at, mission_id, done "
            "FROM interactions "
            "WHERE product_id=? AND contact_id=? "
            "ORDER BY logged_at DESC LIMIT ?",
            (product_id, contact_id, limit),
        )
    else:
        cur = await db.execute(
            "SELECT interaction_id, product_id, contact_id, kind, summary, "
            "next_action, follow_up_at, logged_at, mission_id, done "
            "FROM interactions "
            "WHERE product_id=? "
            "ORDER BY logged_at DESC LIMIT ?",
            (product_id, limit),
        )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def get_pending_follow_ups(
    product_id: str | None = None,
    within_days: int = 7,
) -> list[dict]:
    """Return pending interactions with follow_up_at <= now + within_days days.

    Only returns rows where done=0 (not yet marked complete).
    If product_id is None, returns across all products.
    """
    from src.infra.db import get_db
    db = await get_db()

    cutoff = (datetime.now(timezone.utc) + timedelta(days=within_days)).strftime("%Y-%m-%d %H:%M:%S")

    if product_id is not None:
        cur = await db.execute(
            "SELECT i.interaction_id, i.product_id, i.contact_id, i.kind, i.summary, "
            "i.next_action, i.follow_up_at, i.logged_at, i.mission_id, "
            "r.handle, r.display_name "
            "FROM interactions i "
            "LEFT JOIN relationships r ON r.contact_id = i.contact_id "
            "WHERE i.product_id=? AND i.done=0 "
            "AND i.follow_up_at IS NOT NULL AND i.follow_up_at <= ? "
            "ORDER BY i.follow_up_at",
            (product_id, cutoff),
        )
    else:
        cur = await db.execute(
            "SELECT i.interaction_id, i.product_id, i.contact_id, i.kind, i.summary, "
            "i.next_action, i.follow_up_at, i.logged_at, i.mission_id, "
            "r.handle, r.display_name "
            "FROM interactions i "
            "LEFT JOIN relationships r ON r.contact_id = i.contact_id "
            "WHERE i.done=0 "
            "AND i.follow_up_at IS NOT NULL AND i.follow_up_at <= ? "
            "ORDER BY i.follow_up_at",
            (cutoff,),
        )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def mark_follow_up_done(interaction_id: int) -> None:
    """Mark an interaction's follow-up as done (done=1)."""
    from src.infra.db import get_db
    db = await get_db()
    await db.execute(
        "UPDATE interactions SET done=1 WHERE interaction_id=?",
        (interaction_id,),
    )
    await db.commit()
    logger.info("crm: marked follow-up done", interaction_id=interaction_id)


# ---------------------------------------------------------------------------
# Consent (A10.r1)
# ---------------------------------------------------------------------------


async def grant_consent(
    product_id: str,
    contact_id: int,
    purpose: str,
    source_evidence_url: str,
    *,
    expires_at: str | None = None,
) -> int:
    """Grant consent for a purpose. Upserts if a revoked record exists.

    Returns consent_id.
    """
    from src.infra.db import get_db
    db = await get_db()

    if purpose not in VALID_CONSENT_PURPOSES:
        raise ValueError(
            f"Invalid purpose '{purpose}'. Must be one of: {sorted(VALID_CONSENT_PURPOSES)}"
        )

    # Check for existing record (any state)
    cur = await db.execute(
        "SELECT consent_id FROM consent_records "
        "WHERE product_id=? AND contact_id=? AND purpose=?",
        (product_id, contact_id, purpose),
    )
    existing = await cur.fetchone()

    if existing:
        consent_id = existing[0]
        await db.execute(
            "UPDATE consent_records "
            "SET granted_at=strftime('%Y-%m-%d %H:%M:%S','now'), "
            "expires_at=?, source_evidence_url=?, revoked_at=NULL "
            "WHERE consent_id=?",
            (expires_at, source_evidence_url, consent_id),
        )
        await db.commit()
        logger.info("crm: re-granted consent", consent_id=consent_id, purpose=purpose)
        return consent_id

    cur = await db.execute(
        "INSERT INTO consent_records "
        "(product_id, contact_id, purpose, granted_at, expires_at, source_evidence_url, revoked_at) "
        "VALUES (?, ?, ?, strftime('%Y-%m-%d %H:%M:%S','now'), ?, ?, NULL)",
        (product_id, contact_id, purpose, expires_at, source_evidence_url),
    )
    await db.commit()
    consent_id = cur.lastrowid
    logger.info("crm: granted consent", consent_id=consent_id, purpose=purpose)
    return consent_id


async def revoke_consent(
    product_id: str,
    contact_id: int,
    purpose: str,
) -> None:
    """Revoke a previously granted consent by setting revoked_at = now."""
    from src.infra.db import get_db
    db = await get_db()

    await db.execute(
        "UPDATE consent_records "
        "SET revoked_at=strftime('%Y-%m-%d %H:%M:%S','now') "
        "WHERE product_id=? AND contact_id=? AND purpose=? AND revoked_at IS NULL",
        (product_id, contact_id, purpose),
    )
    await db.commit()
    logger.info("crm: revoked consent", contact_id=contact_id, purpose=purpose)


async def check_consent(
    product_id: str,
    contact_id: int,
    purpose: str,
) -> dict | None:
    """Return the consent record dict, or None if none exists (any state)."""
    from src.infra.db import get_db
    db = await get_db()

    cur = await db.execute(
        "SELECT consent_id, product_id, contact_id, purpose, "
        "granted_at, expires_at, source_evidence_url, revoked_at "
        "FROM consent_records "
        "WHERE product_id=? AND contact_id=? AND purpose=? "
        "ORDER BY granted_at DESC LIMIT 1",
        (product_id, contact_id, purpose),
    )
    row = await cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


async def has_consent(
    product_id: str,
    contact_id: int,
    purpose: str,
) -> bool:
    """Return True iff a valid (not revoked, not expired) consent record exists.

    Used by every Z7 surface that touches contact data before acting.
    B4 (meeting brief) and B7 (interviews) call this before creating
    quote_use or interview_recording consent-gated artifacts.
    """
    record = await check_consent(product_id, contact_id, purpose)
    if record is None:
        return False
    if record["revoked_at"] is not None:
        return False
    if record["expires_at"] is not None:
        # Compare as strings; both are stored as "%Y-%m-%d %H:%M:%S"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if record["expires_at"] < now_str:
            return False
    return True
