"""Z7 T3A (A2) — launch_response_monitor mr_roboto verb.

Sub-mission polling each channel for replies/comments/upvotes.

Surfaces:
  - Top engagement (top 3 comments/replies by score)
  - Negative-sentiment threads (sentiment < threshold)
  - Velocity anomalies (engagement rate > 2× baseline in 1h window)

All as founder_actions for review.

This verb enqueues a sub-mission (via Beckman) that polls the channels
at T+1h, T+4h, T+24h, T+72h intervals per the phase clock.

Payload
-------
::

    {
        "product_id": "prod-abc",
        "launch_id": 1,
        "channels": ["hn", "twitter"],
        "mission_id": 42,
    }

Returns ``{"status": "enqueued", "sub_mission_id": int|None}``.
"""
from __future__ import annotations

from yazbunu import get_logger

logger = get_logger("mr_roboto.launch_response_monitor")


async def run(payload: dict) -> dict:
    """Enqueue a response-monitoring sub-mission for the launch.

    The sub-mission polls each channel at the defined intervals and
    surfaces founder_actions for engagement highlights and alerts.
    """
    product_id = payload.get("product_id") or ""
    launch_id = payload.get("launch_id") or 0
    channels = list(payload.get("channels") or [])
    mission_id = payload.get("mission_id")

    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    try:
        from general_beckman import enqueue as beckman_enqueue

        task_id = await beckman_enqueue(
            spec={
                "title": (
                    f"Monitor launch responses for '{product_id}' "
                    f"launch #{launch_id}"
                ),
                "description": (
                    f"Poll channels {channels} for engagement, negative sentiment, "
                    f"and velocity anomalies. Surface top threads + alerts as "
                    f"founder_actions. Intervals: T+1h, T+4h, T+24h, T+72h."
                ),
                "agent_type": "researcher",
                "kind": "main_work",
                "context": {
                    "product_id": product_id,
                    "launch_id": launch_id,
                    "channels": channels,
                    "parent_mission_id": mission_id,
                    "monitor_kind": "launch_response",
                    "poll_intervals_h": [1, 4, 24, 72],
                    "sentiment_threshold": -0.3,
                    "velocity_multiplier": 2.0,
                },
            },
            parent_id=int(mission_id) if mission_id else None,
        )
        logger.info(
            "launch_response_monitor: sub-mission enqueued",
            product_id=product_id,
            launch_id=launch_id,
            task_id=task_id,
        )
        return {
            "status": "enqueued",
            "sub_mission_id": task_id,
            "product_id": product_id,
            "launch_id": launch_id,
        }
    except Exception as exc:
        logger.error(
            "launch_response_monitor: enqueue failed",
            product_id=product_id,
            error=str(exc),
        )
        return {"status": "error", "error": str(exc)}
