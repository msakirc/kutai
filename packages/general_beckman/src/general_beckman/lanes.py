"""Z8 T1B — admission lane policy.

Two lanes:
  * ``oneshot``  — terminal-state missions (default; the historical Beckman flow)
  * ``ongoing``  — alert_triage / cron / support_ticket etc. that stream
                   events into already-active ``kind='ongoing'`` missions.

Separate concurrency caps so a webhook burst on the ongoing lane does
not starve oneshot apply/coder work, and vice versa. The lane column
lives on ``tasks.lane`` (Z8 T1B migration).
"""
from __future__ import annotations

LANE_ONESHOT = "oneshot"
LANE_ONGOING = "ongoing"

# Caps are intentionally conservative for v1. The dispatcher's local /
# cloud pressure gates still apply inside each lane; these caps only
# bound the per-lane concurrent in-flight ceiling so the two lanes can't
# fully drown each other.
ONESHOT_CONCURRENCY = 4
ONGOING_CONCURRENCY = 8  # webhooks bursty; cron sparse

_ONGOING_TASK_TYPES = frozenset({
    "alert_triage",
    "cron_backup_verify",
    "cron_dep_hygiene",
    "cron_cve_scan",
    "cron_secret_scan",
    "cron_cost_pull",
    "cron_synthetic_check",
    "support_ticket",
})


def pick_lane(task_type: str | None) -> str:
    """Default lane resolution by task_type.

    Unknown / empty task_type defaults to ``oneshot``. Callers may
    override by passing ``lane=`` explicitly to :func:`enqueue`.
    """
    if task_type and task_type in _ONGOING_TASK_TYPES:
        return LANE_ONGOING
    return LANE_ONESHOT


async def count_in_flight(conn, lane: str) -> int:
    """How many tasks in this lane are currently processing/assigned."""
    async with conn.execute(
        "SELECT COUNT(*) FROM tasks "
        "WHERE lane=? AND status IN ('processing','assigned','in_progress')",
        (lane,),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return 0
    return int(row[0])


async def cap_for(lane: str) -> int:
    """Concurrency cap for the given lane."""
    return ONGOING_CONCURRENCY if lane == LANE_ONGOING else ONESHOT_CONCURRENCY
