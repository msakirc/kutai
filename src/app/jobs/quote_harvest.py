"""Z7 T4 A8 — Monthly quote harvest job.

Runs monthly (registered in beckman cron_seed as ``_executor='quote_harvest'``).

Pipeline
--------
1. Scan tickets with positive resolution (``sentiment='positive'``) from the
   last 30 days.
2. For each quotable ticket: emit a founder_action "request quote consent?".
3. On consent (caller invokes ``_on_consent_approved``): insert a row into
   ``press_kit_quotes`` with ``source_kind='ticket'``.

Public API
----------
- ``run_quote_harvest()``          — main entry point (mr_roboto executor).
- ``_fetch_positive_tickets(days)``  — DB query, testable.
- ``_emit_consent_request(ticket, product_id, mission_id)`` — founder_action emit.
- ``_on_consent_approved(product_id, ticket_id, speaker, body)`` — DB insert.
- ``_create_founder_action``         — monkeypatchable in tests.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.jobs.quote_harvest")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _fetch_positive_tickets(days: int = 30) -> list[dict]:
    """Return positively-resolved tickets from the last *days* days.

    Criteria: ``sentiment = 'positive'`` AND ``status = 'closed'``.
    """
    from src.infra.db import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT id, user_id, question, answer, confidence, sentiment, created_at "
            "FROM tickets "
            "WHERE sentiment = 'positive' "
            "  AND status = 'closed' "
            "  AND created_at >= datetime('now', ?) "
            "ORDER BY created_at DESC",
            (f"-{int(days)} days",),
        )
        cols = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        logger.warning("quote_harvest: _fetch_positive_tickets failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Founder action emission (monkeypatchable for tests)
# ---------------------------------------------------------------------------


async def _create_founder_action(**kwargs) -> Any:
    """Thin wrapper around ``src.founder_actions.create`` (monkeypatchable)."""
    from src.founder_actions import create as _create
    return await _create(**kwargs)


async def _emit_consent_request(
    *,
    ticket: dict,
    product_id: str,
    mission_id: int,
) -> Any:
    """Emit a founder_action requesting quote consent for a positive ticket.

    The action payload includes the ticket id so the on-approve handler can
    insert the quote into ``press_kit_quotes``.
    """
    ticket_id = ticket.get("id")
    user_id = ticket.get("user_id", "unknown")
    question = ticket.get("question", "")
    answer = ticket.get("answer", "")
    body_preview = answer[:120] + ("…" if len(answer) > 120 else "")

    title = f"Request quote consent from {user_id}?"
    why = (
        f"This support ticket ended positively. The user's response "
        f"may make a compelling press-kit quote: \"{body_preview}\""
    )
    instructions = [
        "Approve to insert this quote into press_kit_quotes (requires user consent).",
        "Consider reaching out to the user to confirm they consent to being quoted.",
        "Reject to skip this ticket.",
    ]
    payload = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "body": answer,
        "product_id": product_id,
        "_quote_consent_pending": True,
    }

    return await _create_founder_action(
        mission_id=mission_id,
        kind="generic",
        title=title,
        why=why,
        instructions=instructions,
        expected_output_kind="ack_only",
        expected_output_schema=payload,
        notify_telegram=True,
    )


# ---------------------------------------------------------------------------
# On-approve handler
# ---------------------------------------------------------------------------


async def _on_consent_approved(
    *,
    product_id: str,
    ticket_id: int,
    speaker: str,
    body: str,
) -> int | None:
    """Insert an approved quote into ``press_kit_quotes``.

    Called by the Telegram approve handler when the founder consents to using
    the quote. Returns the new ``quote_id`` or None on failure.
    """
    from src.infra.db import get_db
    db = await get_db()
    try:
        cur = await db.execute(
            "INSERT INTO press_kit_quotes "
            "(product_id, kit_id, source_kind, speaker, body, approved, created_at) "
            "VALUES (?, NULL, 'ticket', ?, ?, 1, strftime('%Y-%m-%d %H:%M:%S','now'))",
            (product_id, speaker, body),
        )
        await db.commit()
        quote_id = cur.lastrowid
        logger.info(
            "quote_harvest: inserted approved quote",
            product_id=product_id,
            ticket_id=ticket_id,
            quote_id=quote_id,
        )
        return quote_id
    except Exception as exc:
        logger.error(
            "quote_harvest: _on_consent_approved failed",
            product_id=product_id,
            ticket_id=ticket_id,
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_quote_harvest(days: int = 30) -> dict:
    """Monthly quote harvest entry point. Called by mr_roboto for ``quote_harvest`` executor.

    Returns ``{"ok": True, "candidates": N}`` on success.
    """
    try:
        tickets = await _fetch_positive_tickets(days=days)
        if not tickets:
            logger.info("quote_harvest: no positive tickets in window")
            return {"ok": True, "candidates": 0, "reason": "no_positive_tickets"}

        emitted = 0
        for ticket in tickets:
            product_id = ticket.get("mission_id") or "__unknown__"
            try:
                await _emit_consent_request(
                    ticket=ticket,
                    product_id=str(product_id),
                    mission_id=0,
                )
                emitted += 1
            except Exception as exc:
                logger.warning(
                    "quote_harvest: emit failed for ticket",
                    ticket_id=ticket.get("id"),
                    error=str(exc),
                )

        logger.info("quote_harvest: run complete", candidates=emitted)
        return {"ok": True, "candidates": emitted}

    except Exception as exc:
        logger.error("quote_harvest: failed", error=str(exc))
        return {"ok": False, "reason": str(exc)}
