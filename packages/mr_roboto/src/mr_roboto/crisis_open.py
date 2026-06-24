"""Z7 T3E — B6: Crisis comms — open, resolve, trigger helpers.

Provides:
  - open_crisis_event()       — insert crisis_events row, emit founder_actions
  - resolve_crisis_event()    — mark row resolved
  - trigger_crisis_from_incident() — B3 bridge: critical incident → Tier 2 crisis
  - parse_crisis_cmd()        — parse /crisis subcommand text
  - _create_founder_action()  — thin wrapper, mockable in tests
"""
from __future__ import annotations

import re
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.crisis_open")

# Severity → tier mapping for incident-triggered crises.
_INCIDENT_TIER: dict[str, int] = {
    "critical": 2,  # outage / data issue (Tier 2)
    "major": 2,
    "minor": 0,  # not a crisis
}


async def _create_founder_action(**kwargs) -> Any:
    """Thin wrapper around src.founder_actions.create — mockable in tests."""
    from src.founder_actions import create as fa_create
    return await fa_create(**kwargs)


async def open_crisis_event(
    *,
    product_id: str,
    tier: int,
    source: str = "manual",
    summary: str = "",
    mission_id: int = 0,
) -> dict:
    """Open a new crisis_events row for *product_id* at *tier*.

    Emits:
    - Tier-classification founder_action (always).
    - Counsel-engaged? founder_action (Tier 3+).

    Returns the event dict.
    """
    from dabidabi import get_db

    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO crisis_events "
        "(product_id, tier, source, summary, status) "
        "VALUES (?, ?, ?, ?, 'active')",
        (product_id, tier, source, summary),
    )
    await db.commit()
    event_id = cursor.lastrowid

    row_cur = await db.execute(
        "SELECT event_id, product_id, opened_at, tier, source, summary, status, "
        "resolved_at, postmortem_url FROM crisis_events WHERE event_id = ?",
        (event_id,),
    )
    row = await row_cur.fetchone()
    event: dict = dict(row) if row else {
        "event_id": event_id,
        "product_id": product_id,
        "tier": tier,
        "source": source,
        "summary": summary,
        "status": "active",
        "opened_at": None,
        "resolved_at": None,
        "postmortem_url": None,
    }

    # ── Emit tier-classification founder_action ──────────────────────────────
    tier_labels = {
        1: "Brand misstep / pile-on",
        2: "Outage / data issue",
        3: "Security incident / breach",
        4: "Existential / legal",
    }
    label = tier_labels.get(tier, f"Tier {tier}")
    try:
        await _create_founder_action(
            mission_id=mission_id,
            kind="generic",
            title=f"[Crisis Tier {tier}] Confirm tier classification",
            why=(
                f"A Tier {tier} crisis has been opened for product '{product_id}'. "
                f"Tier {tier} = {label}. "
                f"Source: {source}. Summary: {summary[:200]}"
            ),
            instructions=[
                f"Review the crisis summary above.",
                f"Confirm or adjust the tier (1=brand misstep, 2=outage, "
                f"3=security breach, 4=existential/legal).",
                f"If tier is wrong, run `/crisis open <correct_tier> {product_id}` "
                f"and resolve this event.",
                f"Review playbooks/crisis_comms_tier{tier}.md for next steps.",
            ],
            urgent=(tier >= 3),
            notify_telegram=True,
        )
    except Exception as exc:
        logger.warning(
            "crisis_open: tier-classification founder_action failed",
            event_id=event_id,
            error=str(exc),
        )

    # ── Emit counsel-engaged? founder_action for Tier 3+ ────────────────────
    if tier >= 3:
        try:
            await _create_founder_action(
                mission_id=mission_id,
                kind="legal_counsel",
                title=f"[Crisis Tier {tier}] Counsel engaged?",
                why=(
                    f"Tier {tier}+ crisis requires immediate legal counsel engagement. "
                    f"Product: '{product_id}'. Summary: {summary[:200]}"
                ),
                instructions=[
                    "Confirm: is your legal counsel already notified?",
                    "If YES: click 'Done' — proceed per counsel's guidance.",
                    "If NO: contact counsel immediately before any public statement.",
                    "Do NOT draft or publish any external communication without counsel review.",
                    "See playbooks/crisis_comms_tier3.md (or tier4.md) for jurisdiction matrix.",
                ],
                urgent=True,
                notify_telegram=True,
            )
        except Exception as exc:
            logger.warning(
                "crisis_open: counsel founder_action failed",
                event_id=event_id,
                error=str(exc),
            )

    logger.info(
        "crisis_open: crisis event opened",
        event_id=event_id,
        product_id=product_id,
        tier=tier,
        source=source,
    )
    return event


async def resolve_crisis_event(*, event_id: int, product_id: str) -> dict:
    """Mark a crisis_events row as resolved.

    Returns the updated event dict.
    """
    from dabidabi import get_db

    db = await get_db()
    await db.execute(
        "UPDATE crisis_events "
        "SET status='resolved', "
        "resolved_at=strftime('%Y-%m-%d %H:%M:%S','now') "
        "WHERE event_id=? AND product_id=?",
        (event_id, product_id),
    )
    await db.commit()

    row_cur = await db.execute(
        "SELECT event_id, product_id, opened_at, tier, source, summary, status, "
        "resolved_at, postmortem_url FROM crisis_events WHERE event_id=?",
        (event_id,),
    )
    row = await row_cur.fetchone()
    event = dict(row) if row else {"event_id": event_id, "status": "resolved"}

    logger.info("crisis_open: crisis event resolved", event_id=event_id, product_id=product_id)
    return event


async def trigger_crisis_from_incident(
    *,
    product_id: str,
    incident_id: int,
    severity: str,
    summary: str,
    mission_id: int = 0,
) -> dict:
    """B3 bridge: open a crisis event from a B3 critical/major incident.

    Returns {"opened": bool, "event_id": int|None, "tier": int|None}.
    Only opens for critical or major severity. Minor = no crisis.
    """
    tier = _INCIDENT_TIER.get(severity.lower(), 0)
    if tier == 0:
        logger.info(
            "crisis_open: incident severity=%s below crisis threshold — skipped",
            severity,
            incident_id=incident_id,
            product_id=product_id,
        )
        return {"opened": False, "event_id": None, "tier": None}

    event = await open_crisis_event(
        product_id=product_id,
        tier=tier,
        source="incident",
        summary=f"[incident #{incident_id}] {summary}",
        mission_id=mission_id,
    )
    return {"opened": True, "event_id": event["event_id"], "tier": tier}


def parse_crisis_cmd(text: str) -> dict:
    """Parse a /crisis command string.

    Supported forms:
      /crisis open [tier] [summary...]
      /crisis resume <product_id>
      /crisis status

    Returns a dict with at minimum {"subcommand": str}.
    """
    # Strip leading slash-command prefix
    text = text.strip()
    # Remove the "/crisis" prefix if present
    text = re.sub(r"^/crisis\s*", "", text, flags=re.IGNORECASE).strip()

    parts = text.split(None, 1)  # split on first whitespace
    if not parts:
        return {"subcommand": "status"}

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "status":
        return {"subcommand": "status"}

    if sub == "resume":
        product_id = rest.strip().split()[0] if rest.strip() else ""
        return {"subcommand": "resume", "product_id": product_id}

    if sub == "open":
        # Try to extract tier as first token of rest
        tier = 1
        summary = rest.strip()
        if summary:
            m = re.match(r"^([1-4])\s*(.*)", summary)
            if m:
                tier = int(m.group(1))
                summary = m.group(2).strip()
        return {"subcommand": "open", "tier": tier, "summary": summary}

    # Unknown subcommand
    return {"subcommand": "unknown", "error": f"unknown subcommand: {sub!r}"}
