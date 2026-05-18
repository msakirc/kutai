"""Z7 T3A (A2) — launch_lessons_writeback mr_roboto verb.

At T+7d, emits 3-5 mission_lessons rows capturing what worked and what
didn't during the launch, keyed for cross-mission reuse.

Lesson examples (dedup_key namespace ``launch.*``)
---------------------------------------------------
- ``launch.hn.timing.9am-est`` — HN timing lesson
- ``launch.ph.upvote_velocity`` — PH velocity pattern
- ``launch.twitter.thread_length`` — Twitter engagement shape
- ``launch.channel_priority.{product_id}`` — which channel drove most signups
- ``launch.readiness.payment_e2e`` — payment E2E gate result

Payload
-------
::

    {
        "product_id": "prod-abc",
        "launch_id": 1,
        "mission_id": 42,
        "channels": ["hn", "ph", "twitter"],
        "engagement_summary": {
            "hn": {"upvotes": 120, "comments": 34, "timing_utc": "09:00"},
            "ph": {"votes": 45, "comments": 12, "timing_utc": "09:00"},
            "twitter": {"likes": 200, "retweets": 30, "timing_utc": "09:00"},
        },
    }

Returns ``{"status": "ok", "lessons_written": int}``.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.launch_lessons_writeback")


async def upsert_mission_lesson(**kwargs) -> int:
    """Thin import wrapper — allows monkeypatching in tests."""
    from src.infra.mission_lessons import upsert_mission_lesson as _upsert
    return await _upsert(**kwargs)


def _top_channel(engagement: dict[str, dict]) -> str | None:
    """Return the channel with the highest combined engagement score."""
    if not engagement:
        return None
    scores: dict[str, int] = {}
    for ch, stats in engagement.items():
        score = 0
        score += int(stats.get("upvotes", 0) or 0)
        score += int(stats.get("votes", 0) or 0)
        score += int(stats.get("likes", 0) or 0)
        score += int(stats.get("comments", 0) or 0) * 2  # weight comments
        score += int(stats.get("retweets", 0) or 0) * 3
        scores[ch] = score
    return max(scores, key=lambda c: scores[c]) if scores else None


async def run(payload: dict) -> dict:
    """Write 3-5 mission_lessons rows from launch engagement data."""
    product_id = payload.get("product_id") or ""
    launch_id = payload.get("launch_id") or 0
    mission_id = payload.get("mission_id") or 0
    channels = list(payload.get("channels") or [])
    engagement = dict(payload.get("engagement_summary") or {})

    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    stack = f"launch/{product_id}"
    lessons_written = 0

    # Lesson 1 — top channel for signups
    top_ch = _top_channel(engagement)
    if top_ch:
        pattern = f"launch top channel: {top_ch} drove highest engagement"
        fix = (
            f"For next launch of '{product_id}': prioritize {top_ch} "
            "for pre-launch seeding. Prepare draft 48h in advance."
        )
        try:
            await upsert_mission_lesson(
                stack=stack,
                domain="launch_channel",
                pattern=pattern[:120],
                fix=fix[:300],
                severity="info",
                source_kind="launch_writeback",
                source_ref={
                    "product_id": product_id,
                    "launch_id": launch_id,
                    "mission_id": mission_id,
                    "top_channel": top_ch,
                    "engagement": engagement,
                },
            )
            lessons_written += 1
        except Exception as exc:
            logger.warning("launch_lessons_writeback: lesson 1 failed", error=str(exc))

    # Lesson 2 — timing pattern per channel
    for ch in channels[:3]:  # cap at 3 channels for this lesson
        stats = engagement.get(ch) or {}
        timing = stats.get("timing_utc") or "unknown"
        upvotes = stats.get("upvotes") or stats.get("votes") or stats.get("likes") or 0
        if upvotes:
            pattern = f"launch.{ch}.timing.{timing.replace(':', '').replace('-', '')}"[:80]
            fix = (
                f"{ch.upper()} launch at {timing} UTC produced {upvotes} "
                f"upvotes/likes. Use this as anchor for next {ch} launch."
            )
            try:
                await upsert_mission_lesson(
                    stack=stack,
                    domain=f"launch_{ch}",
                    pattern=pattern,
                    fix=fix[:300],
                    severity="info",
                    source_kind="launch_writeback",
                    source_ref={
                        "product_id": product_id,
                        "launch_id": launch_id,
                        "channel": ch,
                        "timing_utc": timing,
                        "engagement": stats,
                    },
                )
                lessons_written += 1
            except Exception as exc:
                logger.warning(
                    "launch_lessons_writeback: timing lesson failed",
                    channel=ch,
                    error=str(exc),
                )

    # Lesson 3 — readiness gate result
    pattern = f"launch.readiness.gate.{product_id}.launch{launch_id}"[:80]
    fix = (
        "Run launch_readiness_gate ≥1h before T-0 to catch blockers early. "
        "All 7 checks should pass before scheduling publish_synchronized."
    )
    try:
        await upsert_mission_lesson(
            stack=stack,
            domain="launch_readiness",
            pattern=pattern,
            fix=fix[:300],
            severity="info",
            source_kind="launch_writeback",
            source_ref={
                "product_id": product_id,
                "launch_id": launch_id,
                "mission_id": mission_id,
            },
        )
        lessons_written += 1
    except Exception as exc:
        logger.warning("launch_lessons_writeback: readiness lesson failed", error=str(exc))

    logger.info(
        "launch_lessons_writeback: complete",
        product_id=product_id,
        launch_id=launch_id,
        lessons_written=lessons_written,
    )
    return {
        "status": "ok",
        "lessons_written": lessons_written,
        "product_id": product_id,
        "launch_id": launch_id,
    }
