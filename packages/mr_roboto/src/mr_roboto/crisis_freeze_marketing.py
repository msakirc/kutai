"""Z7 T3E — B6: crisis/freeze_marketing mr_roboto verb.

Writes a per-product freeze flag into the marketing_freeze table.
Downstream subsystems (A2/B1/A7 — not yet shipped) call
``is_marketing_frozen(product_id)`` before proceeding.

Reversible via ``crisis/resume`` (sets resumed_at=now on active freeze rows).

Payload::

    {
        "product_id": "prod-abc",   # required
        "event_id":   42,           # required — crisis_events row
    }

Returns ``{"status": "ok", "frozen": bool, "freeze_id": int|None}``.
"""
from __future__ import annotations

from yazbunu import get_logger

logger = get_logger("mr_roboto.crisis_freeze_marketing")


async def run(payload: dict) -> dict:
    """Execute crisis/freeze_marketing.

    Inserts a marketing_freeze row for the product+event pair if none exists.
    Idempotent: if an active freeze row already exists, returns ok with frozen=True.
    """
    product_id = payload.get("product_id") or ""
    event_id = payload.get("event_id")

    if not product_id:
        return {"status": "error", "error": "product_id is required"}
    if event_id is None:
        return {"status": "error", "error": "event_id is required"}

    from dabidabi import get_db

    db = await get_db()

    # Check for existing active freeze (idempotency guard)
    async with db.execute(
        "SELECT freeze_id FROM marketing_freeze "
        "WHERE product_id=? AND event_id=? AND resumed_at IS NULL",
        (product_id, int(event_id)),
    ) as cur:
        existing = await cur.fetchone()

    if existing:
        logger.info(
            "crisis_freeze_marketing: freeze already active",
            product_id=product_id,
            event_id=event_id,
            freeze_id=existing[0],
        )
        return {"status": "ok", "frozen": True, "freeze_id": existing[0]}

    # Insert new freeze row
    cursor = await db.execute(
        "INSERT INTO marketing_freeze (product_id, event_id) VALUES (?, ?)",
        (product_id, int(event_id)),
    )
    await db.commit()
    freeze_id = cursor.lastrowid

    logger.info(
        "crisis_freeze_marketing: freeze activated",
        product_id=product_id,
        event_id=event_id,
        freeze_id=freeze_id,
    )
    return {"status": "ok", "frozen": True, "freeze_id": freeze_id}


async def is_marketing_frozen(product_id: str) -> bool:
    """Check whether the product currently has an active marketing freeze.

    Called by A2/B1/A7 subsystems before proceeding with launches or sends.
    Returns True when at least one active (resumed_at IS NULL) freeze row exists.
    """
    if not product_id:
        return False
    try:
        from dabidabi import get_db

        db = await get_db()
        async with db.execute(
            "SELECT 1 FROM marketing_freeze "
            "WHERE product_id=? AND resumed_at IS NULL "
            "LIMIT 1",
            (product_id,),
        ) as cur:
            row = await cur.fetchone()
        return row is not None
    except Exception as exc:
        logger.warning(
            "is_marketing_frozen: DB check failed — defaulting to False",
            product_id=product_id,
            error=str(exc),
        )
        return False


async def resume_marketing_freeze(product_id: str) -> dict:
    """Clear all active freezes for *product_id* by setting resumed_at=now.

    Called via ``/crisis resume <product_id>``.
    Returns {"cleared": int} with count of rows updated.
    """
    if not product_id:
        return {"status": "error", "error": "product_id is required"}
    try:
        from dabidabi import get_db

        db = await get_db()
        cursor = await db.execute(
            "UPDATE marketing_freeze "
            "SET resumed_at=strftime('%Y-%m-%d %H:%M:%S','now') "
            "WHERE product_id=? AND resumed_at IS NULL",
            (product_id,),
        )
        await db.commit()
        count = cursor.rowcount

        logger.info(
            "crisis_freeze_marketing: resume cleared freezes",
            product_id=product_id,
            cleared=count,
        )
        return {"status": "ok", "cleared": count, "product_id": product_id}
    except Exception as exc:
        logger.error(
            "crisis_freeze_marketing: resume failed",
            product_id=product_id,
            error=str(exc),
        )
        return {"status": "error", "error": str(exc)}
