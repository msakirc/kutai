"""Z7 T5 B8 — Reviews poll daily job.

Polls all configured platforms for each product, ingests new reviews
(dedup via UNIQUE(platform, external_id) constraint), and classifies
all unclassified reviews using the LLM classifier.

Called by mr_roboto as ``_executor='reviews_poll_daily'``.

Public API
----------
  run_reviews_poll_daily(config) -> dict
      ``config`` (optional): {"products": [{"product_id": str, "platforms": {...}}]}
      Falls back to env var ``REVIEWS_CONFIG_JSON`` if not provided.
      Returns {"ok": bool, "total_ingested": int, "total_classified": int, ...}

Platform detection strategy
----------------------------
- g2           : vecihi scrape (free, no auth)
- appstore     : Apple RSS feed (free, no auth)
- playstore    : google-play-scraper lib or vecihi fallback
- producthunt  : vecihi scrape (free, no auth)

All network/vecihi/LLM calls go through mr_roboto.reviews_poll and
mr_roboto.reviews_classify which are fully mocked in tests.
"""
from __future__ import annotations

import json
import os

from src.infra.logging_config import get_logger

logger = get_logger("app.jobs.reviews_poll_daily")


def _load_config(override: dict | None) -> dict:
    """Load the reviews poll config.

    Priority: override > env var REVIEWS_CONFIG_JSON > empty default.
    """
    if override:
        return override
    raw = os.getenv("REVIEWS_CONFIG_JSON", "")
    if raw:
        try:
            return json.loads(raw)
        except Exception as exc:
            logger.warning("reviews_poll_daily: bad REVIEWS_CONFIG_JSON: %s", exc)
    return {"products": []}


async def run_reviews_poll_daily(config: dict | None = None) -> dict:
    """Main entry point: poll all products + platforms, then enqueue a CPS
    classify producer per unclassified review (classification runs async).

    Returns:
        {
            "ok": True,
            "total_ingested": int,
            "total_enqueued": int,
            "errors": [str, ...],
        }
    """
    from mr_roboto.reviews_poll import poll_platform
    from src.reviews.producers import enqueue_classify
    from src.infra.db import get_db

    cfg = _load_config(config)
    products: list[dict] = cfg.get("products") or []

    total_ingested = 0
    total_enqueued = 0
    errors: list[str] = []

    # Phase 1: poll all platforms for all products
    for product in products:
        product_id = str(product.get("product_id") or "")
        if not product_id:
            errors.append("product missing product_id — skipped")
            continue

        platforms: dict = product.get("platforms") or {}

        for platform, platform_config in platforms.items():
            try:
                result = await poll_platform(
                    platform=platform,
                    product_id=product_id,
                    config=platform_config or {},
                )
                if result.get("error"):
                    errors.append(f"{product_id}/{platform}: {result['error']}")
                else:
                    total_ingested += result.get("ingested", 0)
            except Exception as exc:
                errors.append(f"{product_id}/{platform}: poll failed: {exc}")
                logger.error(
                    "reviews_poll_daily: poll failed product=%s platform=%s: %s",
                    product_id, platform, exc,
                )

    # Phase 2: enqueue a CPS classify producer per unclassified review.
    # Classification now happens asynchronously on the pump (the
    # reviews.classify.resume continuation persists + routes side-effects).
    try:
        db = await get_db()
        cur = await db.execute(
            "SELECT review_id, product_id FROM external_reviews "
            "WHERE sentiment IS NULL ORDER BY created_at ASC LIMIT 100"
        )
        unclassified = await cur.fetchall()
    except Exception as exc:
        errors.append(f"DB query for unclassified reviews failed: {exc}")
        unclassified = []

    for row in unclassified:
        review_id, product_id = row[0], row[1]
        try:
            tid = await enqueue_classify(review_id=review_id, product_id=product_id)
            if tid:
                total_enqueued += 1
            else:
                errors.append(f"classify review_id={review_id}: review not found")
        except Exception as exc:
            errors.append(f"classify review_id={review_id}: {exc}")
            logger.error("reviews_poll_daily: classify enqueue failed review_id=%d: %s", review_id, exc)

    logger.info(
        "reviews_poll_daily: done total_ingested=%d total_enqueued=%d errors=%d",
        total_ingested, total_enqueued, len(errors),
    )

    return {
        "ok": True,
        "total_ingested": total_ingested,
        "total_enqueued": total_enqueued,
        "errors": errors,
    }
