"""Z7 T6 A11.r2 — internal_signal_poll mechanical executor.

Watches the ``tickets`` table for negative-sentiment + low-confidence
clusters as a proxy mention source until the Z6 product event stream ships.

A11.r2 spec:
  - Filter: sentiment='negative' AND confidence < 0.5
  - Cluster window: 1h (same window as the crisis threshold)
  - If >=3 such tickets in the window → ingest a synthetic mention into
    the ``mentions`` table with source='internal_signal' and trigger the
    standard crisis-threshold check.
  - Also ingest each qualifying ticket as a low-score mention (score=3
    to keep it below the immediate threshold but above silent so it lands
    in the digest).

Proxy notes:
  - This is explicitly a temporary proxy; see A11.r2 in
    docs/i2p-evolution/07-humanish-layers-v3.md.
  - When Z6 product event stream ships, replace with direct event-stream
    consumer; this module can be retired or repurposed.

Public API
----------
  run(payload) -> dict
      payload keys:
        product_id (str) — required
        window_hours (int) — default 1
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.internal_signal_poll")

_INTERNAL_SOURCE = "internal_signal"
_PROXY_SCORE = 3  # below immediate (7), below digest threshold (4) — silent but logged


async def _fetch_negative_tickets(
    product_id: str,
    window_hours: int,
) -> list[dict]:
    """Return low-confidence negative-sentiment tickets within window.

    Note: tickets table does NOT have product_id — it's a global support
    table. product_id is used for scoping the ingested mentions only.
    The sentiment column uses 'negative' (not 'neg') matching the Z8 schema.
    """
    try:
        from dabidabi import get_db
        db = await get_db()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=window_hours)
        ).strftime("%Y-%m-%d %H:%M:%S")
        # tickets schema: id, mission_id, user_id, question, answer,
        # confidence, status, escalated_to_founder, founder_action_id,
        # sentiment, created_at
        cur = await db.execute(
            "SELECT id, question, confidence, created_at "
            "FROM tickets "
            "WHERE sentiment = 'negative' "
            "AND confidence < 0.5 "
            "AND created_at >= ?",
            (cutoff,),
        )
        rows = await cur.fetchall()
        return [
            {
                "id": r[0],
                "text": r[1] or "",
                "confidence": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("internal_signal_poll: ticket query failed", error=str(exc))
        return []


async def _ingest_internal_mention(
    *,
    product_id: str,
    ticket_id: int,
    text: str,
) -> bool:
    """Insert a synthetic mention row for the ticket (INSERT OR IGNORE)."""
    try:
        from dabidabi import get_db
        db = await get_db()
        source_id = f"ticket:{ticket_id}"
        await db.execute(
            "INSERT OR IGNORE INTO mentions "
            "(product_id, source, source_id, url, canonical_url, author, "
            " author_followers, text, sentiment, signal_score) "
            "VALUES (?, ?, ?, NULL, NULL, 'support_user', 0, ?, 'neg', ?)",
            (product_id, _INTERNAL_SOURCE, source_id, text[:2000], _PROXY_SCORE),
        )
        await db.commit()
        return True
    except Exception as exc:
        logger.error("internal_signal_poll: ingest failed", error=str(exc))
        return False


async def _check_cluster_threshold(product_id: str, window_hours: int) -> bool:
    """Return True if >=3 internal_signal neg mentions in window."""
    try:
        from dabidabi import get_db
        db = await get_db()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=window_hours)
        ).strftime("%Y-%m-%d %H:%M:%S")
        cur = await db.execute(
            "SELECT COUNT(*) FROM mentions "
            "WHERE product_id = ? AND source = ? AND sentiment = 'neg' AND seen_at >= ?",
            (product_id, _INTERNAL_SOURCE, cutoff),
        )
        row = await cur.fetchone()
        return (row[0] if row else 0) >= 3
    except Exception:
        return False


async def _trigger_crisis_action(product_id: str, count: int) -> None:
    """Surface crisis_comms_draft founder_action for internal signal cluster."""
    try:
        from packages.general_beckman.src.general_beckman import enqueue  # type: ignore
    except ImportError:
        try:
            from general_beckman import enqueue  # type: ignore
        except ImportError:
            logger.debug("internal_signal_poll: general_beckman not available")
            return

    await enqueue(
        {
            "description": (
                f"[crisis_comms_draft] {product_id}: "
                f"{count} negative low-confidence support tickets in 1h"
            ),
            "agent_type": "mechanical",
            "context": {
                "payload": {
                    "action": "notify_user",
                    "message": (
                        f"Internal signal alert ({product_id}): "
                        f"{count} negative low-confidence support tickets in the last hour. "
                        "This may indicate a product issue. Review /mention_monitor status."
                    ),
                }
            },
        }
    )


async def run(payload: dict) -> dict[str, Any]:
    """mr_roboto entry point for ``action == 'internal_signal_poll'``."""
    product_id = str(payload.get("product_id") or "")
    window_hours = int(payload.get("window_hours") or 1)

    if not product_id:
        return {"status": "failed", "error": "internal_signal_poll: missing product_id"}

    tickets = await _fetch_negative_tickets(product_id, window_hours)

    ingested = 0
    for t in tickets:
        ok = await _ingest_internal_mention(
            product_id=product_id,
            ticket_id=int(t["id"]),
            text=t["text"],
        )
        if ok:
            ingested += 1

    crisis_triggered = False
    if ingested > 0:
        crisis_triggered = await _check_cluster_threshold(product_id, window_hours)
        if crisis_triggered:
            await _trigger_crisis_action(product_id, ingested)

    return {
        "status": "ok",
        "tickets_scanned": len(tickets),
        "ingested": ingested,
        "crisis_triggered": crisis_triggered,
        "proxy_note": (
            "internal_signal is a proxy until Z6 product event stream ships; "
            "source=tickets.sentiment='negative'+confidence<0.5"
        ),
    }
