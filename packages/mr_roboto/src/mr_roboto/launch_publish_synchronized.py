"""Z7 T3A (A2) — publish_synchronized mr_roboto verb.

Publishes all approved channel drafts at T-0.

BEFORE publishing, checks ``is_marketing_frozen(product_id)`` (B6).
If frozen → abort with ``{"status": "aborted", "reason": "marketing_frozen"}``.

All channels are published in parallel using asyncio.gather.
Each channel publish is stubbed via ``_publish_channel`` (real adapters
added per-channel when the distribution layer is built in T6D).

Payload
-------
::

    {
        "product_id": "prod-abc",
        "launch_id": 1,
        "channels": ["hn", "twitter", "linkedin"],
        "drafts": {
            "hn": "Show HN: ...",
            "twitter": "We just launched ...",
            "linkedin": "Today we ...",
        },
    }

Returns
-------
``{"status": "published", "published": [...]}``  — all channels succeeded
``{"status": "partial", "published": [...], "failed": [...]}`` — some failed
``{"status": "aborted", "reason": "marketing_frozen"}`` — B6 freeze active
``{"status": "error", "error": str}`` — unexpected error
"""
from __future__ import annotations

import asyncio

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.launch_publish_synchronized")


async def _publish_channel(
    channel: str,
    draft: str,
    product_id: str,
) -> dict:
    """Stub channel publisher.

    In production this delegates to a real channel adapter:
    - HN: no API (manual post) → returns URL from operator
    - Product Hunt: PH API or Selenium
    - Twitter: Twitter v2 API
    - LinkedIn: LinkedIn API
    - Reddit: Reddit API

    Until real adapters are wired (Z7 T6D), this returns a no-op success.
    """
    logger.info(
        "publish_synchronized: stub publish (real adapter pending T6D)",
        channel=channel,
        product_id=product_id,
        draft_preview=draft[:80],
    )
    return {
        "channel": channel,
        "status": "published",
        "url": f"https://placeholder.example.com/{channel}/{product_id}",
        "note": "stub — real adapter pending T6D",
    }


async def run(payload: dict) -> dict:
    """Execute publish_synchronized.

    Check marketing freeze first, then publish all approved channel
    drafts in parallel.
    """
    product_id = payload.get("product_id") or ""
    launch_id = payload.get("launch_id") or 0
    channels = list(payload.get("channels") or [])
    drafts = dict(payload.get("drafts") or {})

    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    # ── B6 freeze check ───────────────────────────────────────────────────
    try:
        from mr_roboto.crisis_freeze_marketing import is_marketing_frozen
        if await is_marketing_frozen(product_id):
            logger.warning(
                "publish_synchronized: ABORTED — marketing frozen",
                product_id=product_id,
                launch_id=launch_id,
            )
            return {
                "status": "aborted",
                "reason": "marketing_frozen",
                "product_id": product_id,
                "launch_id": launch_id,
            }
    except Exception as exc:
        logger.error(
            "publish_synchronized: freeze check failed — aborting conservatively",
            product_id=product_id,
            error=str(exc),
        )
        return {
            "status": "error",
            "error": f"freeze_check_failed: {exc}",
        }

    if not channels:
        return {"status": "error", "error": "channels list is empty"}

    # ── Parallel publish ──────────────────────────────────────────────────
    tasks = []
    for ch in channels:
        draft = drafts.get(ch) or ""
        tasks.append(_publish_channel(ch, draft, product_id))

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    published = []
    failed = []
    for ch, res in zip(channels, results):
        if isinstance(res, Exception):
            failed.append({"channel": ch, "error": str(res)})
            logger.error(
                "publish_synchronized: channel publish failed",
                channel=ch,
                product_id=product_id,
                error=str(res),
            )
        elif isinstance(res, dict) and res.get("status") == "published":
            published.append(res)
        else:
            failed.append({"channel": ch, "result": res})

    if failed and not published:
        return {"status": "error", "error": "all channels failed", "failed": failed}
    if failed:
        return {
            "status": "partial",
            "published": published,
            "failed": failed,
            "product_id": product_id,
            "launch_id": launch_id,
        }

    logger.info(
        "publish_synchronized: all channels published",
        product_id=product_id,
        launch_id=launch_id,
        channels=channels,
    )
    return {
        "status": "published",
        "published": published,
        "product_id": product_id,
        "launch_id": launch_id,
    }
