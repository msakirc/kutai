"""Z7 T5 B2 — changelog/publish mechanical executor.

On founder approval: marks the entry published, regenerates RSS + in-app banner
cache, and queues the B1 announcement email blast via trigger_sequence_by_kind
(degrades gracefully if no announcement sequence exists).

Public surface
--------------
  run(payload: dict) -> dict

  payload keys:
    entry_id    int  (required) — the changelog_entries row to publish
    product_id  TEXT (required)

  Returns:
    {"status": "ok", "entry_id": int, "email_blast_skipped": bool,
     "email_blast_result": dict}
  or
    {"status": "error", "error": str}
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.changelog_publish")


# ---------------------------------------------------------------------------
# Cache invalidation helper
# ---------------------------------------------------------------------------

def _invalidate_changelog_cache() -> None:
    """Bust the changelog page module-level cache so next request re-renders."""
    try:
        from src.app.changelog_page import invalidate_cache
        invalidate_cache()
    except Exception as exc:
        logger.debug("changelog_publish: cache invalidation failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Email blast helper
# ---------------------------------------------------------------------------

async def _queue_announcement_email(product_id: str, **kwargs) -> dict:
    """Queue a B1 announcement email blast for the product.

    Calls lifecycle_email.trigger_sequence_by_kind(product_id, user_id=None,
    trigger_kind='announcement'). With user_id=None + the 'announcement'
    trigger_kind, trigger_sequence_by_kind fans out as a broadcast: it
    enumerates every opted-in recipient in email_preferences and creates one
    email_sends row per subscribed user_token per step.

    Degrades gracefully when:
    - B1 lifecycle_email module is not yet installed.
    - No enabled announcement sequence exists for the product.
    - No subscribed recipients exist (ok=True, sends_created=0).

    Returns: {"ok": bool, "sends_created": int, "recipients": int,
              "reason": str | None}
    """
    try:
        from src.app.lifecycle_email import trigger_sequence_by_kind
        # Announcement blasts target all opted-in subscribers — user_id=None
        # triggers the broadcast fan-out over email_preferences.
        result = await trigger_sequence_by_kind(
            product_id=product_id,
            user_id=None,
            trigger_kind="announcement",
        )
        return result
    except ImportError as exc:
        logger.info(
            "changelog_publish: lifecycle_email not available — email blast skipped: %s", exc
        )
        return {"ok": False, "reason": f"lifecycle_email unavailable: {exc}"}
    except Exception as exc:
        logger.warning(
            "changelog_publish: trigger_sequence_by_kind raised: %s", exc
        )
        return {"ok": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """mr_roboto executor: changelog/publish.

    Marks the entry published, invalidates caches, and queues email blast.
    """
    entry_id_raw = payload.get("entry_id")
    product_id: str = str(payload.get("product_id") or "")

    if entry_id_raw is None:
        return {"status": "error", "error": "entry_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    try:
        entry_id = int(entry_id_raw)
    except (TypeError, ValueError):
        return {"status": "error", "error": f"invalid entry_id: {entry_id_raw!r}"}

    # 1. Mark as published in DB
    try:
        from src.infra.db import get_db
        from src.infra.times import db_now
        db = await get_db()
        now_str = db_now()
        await db.execute(
            "UPDATE changelog_entries "
            "SET published=1, released_at=COALESCE(released_at, ?), updated_at=? "
            "WHERE entry_id=? AND product_id=?",
            (now_str, now_str, entry_id, product_id),
        )
        await db.commit()
    except Exception as exc:
        logger.error("changelog_publish: DB update failed: %s", exc)
        return {"status": "error", "error": f"DB update failed: {exc}"}

    # 2. Invalidate in-app banner + page cache
    _invalidate_changelog_cache()

    # 3. Queue announcement email blast (B1) — best-effort
    email_result = await _queue_announcement_email(product_id)
    email_blast_ok = bool(email_result.get("ok"))
    email_blast_skipped = not email_blast_ok

    logger.info(
        "changelog_publish: entry published",
        entry_id=entry_id,
        product_id=product_id,
        email_blast_ok=email_blast_ok,
    )

    return {
        "status": "ok",
        "entry_id": entry_id,
        "product_id": product_id,
        "email_blast_skipped": email_blast_skipped,
        "email_blast_result": email_result,
    }
