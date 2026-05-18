"""Z8 T3A — webhook event deduplication.

(integration_id, event_id) primary key in `webhook_events` makes replayed
deliveries a no-op. ``already_seen`` short-circuits; ``mark_seen`` stamps a
fresh row with the payload hash + the routing mission_id (NULL if no
mapping was found yet — T3E backfills via integration_mappings).
"""
from __future__ import annotations

from src.infra.db import get_db


async def already_seen(integration_id: str, event_id: str) -> bool:
    """Return True if (integration_id, event_id) is already in webhook_events."""
    conn = await get_db()
    async with conn.execute(
        "SELECT 1 FROM webhook_events WHERE integration_id=? AND event_id=?",
        (integration_id, event_id),
    ) as cur:
        return (await cur.fetchone()) is not None


async def mark_seen(
    integration_id: str,
    event_id: str,
    payload_hash: str,
    mission_id: int | None,
) -> None:
    """Insert a webhook_events row. Idempotent — INSERT OR IGNORE."""
    conn = await get_db()
    await conn.execute(
        "INSERT OR IGNORE INTO webhook_events "
        "(integration_id, event_id, received_at, payload_hash, mission_id) "
        "VALUES (?, ?, datetime('now'), ?, ?)",
        (integration_id, event_id, payload_hash, mission_id),
    )
