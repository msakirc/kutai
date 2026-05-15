"""Z7 T4 B4 — Meeting brief auto-generation.

Public API
----------
  create_meeting(product_id, contact_id, scheduled_for, purpose) -> int
  list_meetings(product_id, contact_id=None) -> list[dict]
  build_brief_context(meeting_id, product_id) -> dict
  compose_brief_md(ctx, talking_points) -> str
  emit_outcome_prompt(meeting_id, product_id) -> dict
  log_meeting_outcome(meeting_id, product_id, contact_id, summary,
                      next_action=None, follow_up=None) -> int

B4 hooks into:
  - crm.list_interactions(product_id, contact_id, limit=5) — recent history
  - crm.get_pending_follow_ups(product_id) — open follow-ups
  - crm.log_interaction(... kind='meeting') — outcome logging
  - founder_actions.create — outcome_prompt card
  - Optional A11 (mentions table): degrade gracefully when absent
  - Optional B2 (changelog_entries table): degrade gracefully when absent
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.meetings")

_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime(_DATETIME_FMT)


def _parse_scheduled_for(s: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM'. Raises ValueError on bad input."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(
        f"scheduled_for must be YYYY-MM-DD HH:MM[:SS], got: {s!r}"
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def create_meeting(
    product_id: str,
    contact_id: int,
    scheduled_for: str,
    purpose: str = "",
) -> int:
    """Create a meetings row. Returns meeting_id.

    Raises ValueError when scheduled_for cannot be parsed.
    """
    _parse_scheduled_for(scheduled_for)  # validate; raises on bad input

    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO meetings (product_id, contact_id, scheduled_for, purpose) "
        "VALUES (?, ?, ?, ?)",
        (product_id, contact_id, scheduled_for, purpose),
    )
    await db.commit()
    meeting_id = cur.lastrowid
    logger.info(
        "meetings: created",
        meeting_id=meeting_id,
        product_id=product_id,
        scheduled_for=scheduled_for,
    )
    return meeting_id


async def list_meetings(
    product_id: str,
    contact_id: int | None = None,
) -> list[dict]:
    """List meetings for a product, optionally filtered by contact_id.

    Returns most-recent-scheduled-for-first.
    """
    from src.infra.db import get_db
    db = await get_db()
    if contact_id is not None:
        cur = await db.execute(
            "SELECT meeting_id, product_id, contact_id, scheduled_for, purpose, "
            "brief_generated_at, brief_md, outcome_logged_interaction_id "
            "FROM meetings WHERE product_id=? AND contact_id=? "
            "ORDER BY scheduled_for DESC",
            (product_id, contact_id),
        )
    else:
        cur = await db.execute(
            "SELECT meeting_id, product_id, contact_id, scheduled_for, purpose, "
            "brief_generated_at, brief_md, outcome_logged_interaction_id "
            "FROM meetings WHERE product_id=? "
            "ORDER BY scheduled_for DESC",
            (product_id,),
        )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def _get_meeting(meeting_id: int) -> dict | None:
    """Return a single meeting row dict or None."""
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT meeting_id, product_id, contact_id, scheduled_for, purpose, "
        "brief_generated_at, brief_md, outcome_logged_interaction_id "
        "FROM meetings WHERE meeting_id=?",
        (meeting_id,),
    )
    row = await cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Brief context assembly
# ---------------------------------------------------------------------------


async def _fetch_mentions(product_id: str, contact_id: int) -> list[dict]:
    """Pull mention-monitor hits for the contact's company.

    Degrades gracefully if A11 is not built (mentions table absent).
    Returns empty list on any error.
    """
    try:
        from src.infra.db import get_db
        db = await get_db()
        # A11 stores mentions in 'mentions' table with company_handle or contact_id.
        # Try contact_id filter first; fall back to empty if column not present.
        cur = await db.execute(
            "SELECT mention_id, source, url, snippet, score, detected_at "
            "FROM mentions "
            "WHERE product_id=? AND contact_id=? "
            "ORDER BY detected_at DESC LIMIT 5",
            (product_id, contact_id),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in await cur.fetchall()]
    except Exception as exc:
        logger.debug("meetings: mentions lookup skipped (A11 absent)", error=str(exc))
        return []


async def _fetch_changelog(product_id: str) -> list[dict]:
    """Pull recent changelog entries.

    Degrades gracefully if B2 is not built (changelog_entries table absent).
    Returns empty list on any error.
    """
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT entry_id, version, released_at, title "
            "FROM changelog_entries "
            "WHERE product_id=? "
            "ORDER BY released_at DESC LIMIT 3",
            (product_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in await cur.fetchall()]
    except Exception as exc:
        logger.debug("meetings: changelog lookup skipped (B2 absent)", error=str(exc))
        return []


async def _fetch_mission_items(product_id: str) -> list[dict]:
    """Pull recent shipped/deferred mission items.

    Returns empty list on any error.
    """
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT id, title, COALESCE(status,'unknown') AS status "
            "FROM missions "
            "WHERE COALESCE(status,'') IN ('completed','failed','cancelled') "
            "ORDER BY id DESC LIMIT 5",
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in await cur.fetchall()]
    except Exception as exc:
        logger.debug("meetings: mission_items lookup failed", error=str(exc))
        return []


async def build_brief_context(meeting_id: int, product_id: str) -> dict:
    """Assemble all context needed for brief generation.

    Keys returned:
      contact   — dict from relationships (or minimal stub)
      meeting   — dict from meetings row
      interactions — last 5 interactions for the contact
      follow_ups   — open follow-ups for the product
      mentions     — A11 mention hits (empty list if A11 absent)
      changelog    — B2 changelog entries (empty list if B2 absent)
      mission_items — recent shipped/deferred missions
    """
    meeting = await _get_meeting(meeting_id)
    if not meeting:
        raise ValueError(f"Meeting {meeting_id} not found")

    contact_id = meeting.get("contact_id")

    # Load contact
    contact: dict = {}
    if contact_id:
        from src.app.crm import get_contact_by_id
        contact = await get_contact_by_id(contact_id) or {}

    # Recent interactions (last 5)
    interactions: list[dict] = []
    if contact_id:
        from src.app.crm import list_interactions
        interactions = await list_interactions(product_id, contact_id, limit=5)

    # Pending follow-ups
    from src.app.crm import get_pending_follow_ups
    follow_ups = await get_pending_follow_ups(product_id, within_days=30)

    # Optional subsystems
    mentions = await _fetch_mentions(product_id, contact_id or 0)
    changelog = await _fetch_changelog(product_id)
    mission_items = await _fetch_mission_items(product_id)

    return {
        "contact": contact,
        "meeting": meeting,
        "interactions": interactions,
        "follow_ups": follow_ups,
        "mentions": mentions,
        "changelog": changelog,
        "mission_items": mission_items,
    }


def compose_brief_md(
    ctx: dict,
    talking_points: list[str] | None = None,
    suggested_asks: list[str] | None = None,
) -> str:
    """Compose a structured Markdown brief from context dict.

    Used both by tests (with stub ctx) and by the LLM verb (after LLM drafts
    talking_points / suggested_asks from ctx).

    Sections:
      - Last Interactions
      - Open Follow-Ups
      - Recent Product Changes
      - Recent Mentions
      - Talking Points (3-5, LLM-drafted or caller-supplied)
      - Suggested Asks (1-2)
    """
    contact = ctx.get("contact") or {}
    meeting = ctx.get("meeting") or {}
    name = contact.get("display_name") or contact.get("handle") or "Contact"
    purpose = meeting.get("purpose") or "(no purpose stated)"
    scheduled = meeting.get("scheduled_for") or "?"

    sections: list[str] = [
        f"# Meeting Brief: {name}",
        f"**Scheduled:** {scheduled}  \n**Purpose:** {purpose}",
    ]

    # Last interactions
    interactions = ctx.get("interactions") or []
    if interactions:
        lines = ["## Last Interactions"]
        for ix in interactions[:5]:
            when = ix.get("logged_at") or "?"
            kind = ix.get("kind") or "other"
            summary = ix.get("summary") or "(no summary)"
            lines.append(f"- **{when}** [{kind}] {summary}")
        sections.append("\n".join(lines))
    else:
        sections.append("## Last Interactions\n_(none on record)_")

    # Open follow-ups
    follow_ups = ctx.get("follow_ups") or []
    if follow_ups:
        lines = ["## Open Follow-Ups"]
        for fu in follow_ups[:5]:
            handle = fu.get("handle") or f"contact#{fu.get('contact_id', '?')}"
            kind = fu.get("kind") or "?"
            summary = fu.get("summary") or "(no summary)"
            due = fu.get("follow_up_at") or "?"
            lines.append(f"- [{kind}] {summary} — due {due} ({handle})")
        sections.append("\n".join(lines))
    else:
        sections.append("## Open Follow-Ups\n_(none pending)_")

    # Recent product changes (B2 changelog)
    changelog = ctx.get("changelog") or []
    if changelog:
        lines = ["## Recent Product Changes"]
        for entry in changelog[:3]:
            version = entry.get("version") or "?"
            released = entry.get("released_at") or "?"
            title = entry.get("title") or "(no title)"
            lines.append(f"- **v{version}** ({released}): {title}")
        sections.append("\n".join(lines))
    else:
        sections.append("## Recent Product Changes\n_(no changelog entries)_")

    # Mentions (A11)
    mentions = ctx.get("mentions") or []
    if mentions:
        lines = ["## Recent Mentions"]
        for m in mentions[:5]:
            source = m.get("source") or "?"
            snippet = (m.get("snippet") or "")[:120]
            detected = m.get("detected_at") or "?"
            lines.append(f"- [{source}] {snippet} _(detected {detected})_")
        sections.append("\n".join(lines))
    else:
        sections.append("## Recent Mentions\n_(none detected)_")

    # Talking points
    tps = talking_points or []
    if tps:
        lines = ["## Talking Points"]
        for i, tp in enumerate(tps[:5], 1):
            lines.append(f"{i}. {tp}")
        sections.append("\n".join(lines))
    else:
        sections.append(
            "## Talking Points\n"
            "_(LLM-drafted talking points will appear here after brief generation)_"
        )

    # Suggested asks
    asks = suggested_asks or []
    if asks:
        lines = ["## Suggested Asks"]
        for i, ask in enumerate(asks[:2], 1):
            lines.append(f"{i}. {ask}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Outcome prompt + logging
# ---------------------------------------------------------------------------


async def emit_outcome_prompt(meeting_id: int, product_id: str) -> dict:
    """Surface a founder_action card asking for meeting outcome.

    Returns {"ok": True} on success, {"ok": False, "reason": ...} on failure.
    """
    meeting = await _get_meeting(meeting_id)
    if not meeting:
        return {"ok": False, "reason": f"meeting {meeting_id} not found"}

    try:
        from src.founder_actions import create as fa_create
        contact_id = meeting.get("contact_id") or 0
        purpose = meeting.get("purpose") or "(no purpose)"
        scheduled = meeting.get("scheduled_for") or "?"

        # Use mission_id=0 sentinel for non-mission-bound actions
        # (consistent with other B-series patterns)
        await fa_create(
            mission_id=0,
            kind="generic",
            title=f"Log meeting outcome — {purpose} ({scheduled})",
            why=(
                "30 minutes have passed since the scheduled meeting. "
                "Logging the outcome keeps the CRM current and surfaces follow-ups."
            ),
            instructions=[
                "What did you discuss?",
                "Any follow-ups owed (by you or them)?",
                "What is the next step?",
            ],
            blocking_task_id=None,
            urgent=False,
            notify_telegram=True,
        )
        logger.info(
            "meetings: outcome_prompt emitted",
            meeting_id=meeting_id,
            product_id=product_id,
        )
        return {"ok": True, "meeting_id": meeting_id}
    except Exception as exc:
        logger.error("meetings: emit_outcome_prompt failed", error=str(exc))
        return {"ok": False, "reason": str(exc)}


async def log_meeting_outcome(
    meeting_id: int,
    product_id: str,
    contact_id: int,
    summary: str,
    *,
    next_action: str | None = None,
    follow_up: str | None = None,
    mission_id: int | None = None,
) -> int:
    """Create an interactions row (kind='meeting') and link it to the meeting.

    Returns the new interaction_id.
    """
    from src.app.crm import log_interaction
    iid = await log_interaction(
        product_id,
        contact_id,
        "meeting",
        summary,
        next_action=next_action,
        follow_up=follow_up,
        mission_id=mission_id,
    )

    from src.infra.db import get_db
    db = await get_db()
    await db.execute(
        "UPDATE meetings SET outcome_logged_interaction_id=? WHERE meeting_id=?",
        (iid, meeting_id),
    )
    await db.commit()

    logger.info(
        "meetings: outcome logged",
        meeting_id=meeting_id,
        interaction_id=iid,
        product_id=product_id,
    )
    return iid
