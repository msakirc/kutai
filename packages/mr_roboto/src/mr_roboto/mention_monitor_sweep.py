"""Z7 A11 — mention_monitor_sweep: the hourly mention-monitor cron entry.

The A11 mention monitor (``src/workflows/mention_monitor.json``) was never
wired: no ``/mention_monitor`` command registered products, the workflow had
no loader, and it was absent from ``cron_seed.INTERNAL_CADENCES``. This
executor supersedes that dead JSON workflow:

  1. read every enabled ``mention_monitors`` row (registered via the
     ``/mention_monitor`` Telegram command);
  2. for each, poll the founder-enabled channels via ``poll_source``;
  3. compose a digest of score 4-6 mentions not yet acted on and enqueue a
     single ``notify_user`` task per product (the ``mention_digest`` the
     workflow's broken ``skip_when`` step could never deliver).

High-score (>=7) mentions still surface their own immediate founder_action
inside ``poll_source`` — the digest only bundles the mid-tier 4-6 band.
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.mention_monitor_sweep")

#: Sources a monitor may enable. Twitter is intentionally excluded — it is
#: PAID and gated behind MENTION_TWITTER_ENABLED inside poll_source.
VALID_CHANNELS: frozenset[str] = frozenset({"hn", "reddit", "google", "discord"})

#: Digest band — mentions scored in [LO, HI] that have not been acted on.
DIGEST_SCORE_LO: int = 4
DIGEST_SCORE_HI: int = 6


async def _enabled_monitors(db) -> list[dict]:
    cur = await db.execute(
        "SELECT product_id, product_name, channels_json FROM mention_monitors "
        "WHERE enabled = 1"
    )
    rows = await cur.fetchall()
    await cur.close()
    out: list[dict] = []
    for product_id, product_name, channels_json in rows:
        try:
            channels = [c for c in json.loads(channels_json or "[]")
                        if c in VALID_CHANNELS]
        except (json.JSONDecodeError, TypeError):
            channels = []
        out.append({
            "product_id": product_id,
            "product_name": product_name or "",
            "channels": channels,
        })
    return out


async def _compose_and_enqueue_digest(product_id: str, db) -> int:
    """Bundle pending score 4-6 mentions into one notify_user task.

    Returns the number of mentions in the digest (0 → no task enqueued).
    """
    cur = await db.execute(
        "SELECT mention_id, source, author, text, signal_score, url "
        "FROM mentions "
        "WHERE product_id = ? AND acted_on = 0 "
        "  AND signal_score BETWEEN ? AND ? "
        "ORDER BY signal_score DESC, seen_at DESC LIMIT 20",
        (product_id, DIGEST_SCORE_LO, DIGEST_SCORE_HI),
    )
    rows = await cur.fetchall()
    await cur.close()
    if not rows:
        return 0

    lines = [f"\U0001f4ec Mention digest — {len(rows)} new mention(s):"]
    ids: list[int] = []
    for mention_id, source, author, text, score, url in rows:
        ids.append(mention_id)
        snippet = (text or "").strip().replace("\n", " ")[:120]
        lines.append(f"  [{source}] ({score}) @{author or '?'}: {snippet}")
        if url:
            lines.append(f"    {url}")

    from general_beckman import add_task
    await add_task(
        title=f"Mention digest: {product_id}",
        description="",
        agent_type="mechanical",
        mission_id=None,
        context={
            "executor": "mechanical",
            "payload": {"action": "notify_user", "message": "\n".join(lines)},
        },
        depends_on=[],
    )
    # Mark these digested mentions acted_on so the next sweep does not
    # re-surface them — acted_on now has a real consumer.
    await db.execute(
        f"UPDATE mentions SET acted_on = 1 WHERE mention_id IN "
        f"({','.join('?' * len(ids))})",
        ids,
    )
    await db.commit()
    return len(rows)


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Cron entry — poll every registered product, enqueue per-product digests."""
    from src.infra.db import get_db
    from mr_roboto.mention_polls import poll_source

    db = await get_db()
    monitors = await _enabled_monitors(db)
    summary = {
        "monitors": len(monitors),
        "polled": 0,
        "digested": 0,
        "errors": [],
    }
    for mon in monitors:
        product_id = mon["product_id"]
        product_name = mon["product_name"]
        for source in mon["channels"]:
            try:
                await poll_source(
                    source=source,
                    product_id=product_id,
                    product_name=product_name,
                    config={},
                )
                summary["polled"] += 1
            except Exception as exc:  # noqa: BLE001 — one bad source must not abort
                logger.warning("mention poll failed", product_id=product_id,
                                source=source, error=str(exc))
                summary["errors"].append(f"{product_id}/{source}: {exc}")
        try:
            summary["digested"] += await _compose_and_enqueue_digest(product_id, db)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mention digest failed", product_id=product_id,
                            error=str(exc))
            summary["errors"].append(f"{product_id}/digest: {exc}")
        try:
            await db.execute(
                "UPDATE mention_monitors "
                "SET last_run_at = strftime('%Y-%m-%d %H:%M:%S','now') "
                "WHERE product_id = ?",
                (product_id,),
            )
            await db.commit()
        except Exception:  # noqa: BLE001
            pass

    logger.info("mention_monitor_sweep complete", **{
        k: v for k, v in summary.items() if k != "errors"})
    return {"status": "ok", **summary}
