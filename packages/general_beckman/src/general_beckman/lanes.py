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
    """In-flight LLM/overhead tasks on this lane.

    Mechanical tasks (mr_roboto: git commit / workspace snapshot /
    notify_user — CPU-only, no LLM, no GPU/cloud) are EXCLUDED. The lane
    cap bounds GPU/cloud contention, which mechanicals don't create.
    Counting them would let a burst of git commits eat the LLM admission
    budget; combined with :func:`has_ready_mechanical` (which lets them
    bypass the cap gate), the count must stay LLM-only on both sides.
    """
    async with conn.execute(
        "SELECT COUNT(*) FROM tasks "
        "WHERE lane=? AND status IN ('processing','assigned','in_progress') "
        "AND COALESCE(agent_type,'') != 'mechanical' "
        "AND COALESCE(runner,'') != 'mechanical'",
        (lane,),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return 0
    return int(row[0])


async def has_ready_mechanical(conn, lane: str) -> bool:
    """True if a pending MECHANICAL task is waiting on this lane.

    Mechanicals are exempt from the concurrency cap. next_task() uses this
    so a lane saturated by LLM work still admits a ready mechanical instead
    of returning None and stalling git commits / blackboard writes behind
    LLM load (bug 2026-05-26).

    Cheap existence check: mirrors get_ready_tasks' SQL-level readiness
    (status='pending' + retry-due + mission active) MINUS the in-Python
    dependency filter. A dependency-blocked mechanical can falsely pass
    here — harmless: it only costs one extra admission scan that tick (the
    candidate loop's real dep check then admits nothing).
    """
    async with conn.execute(
        """SELECT 1 FROM tasks t
           LEFT JOIN missions m ON t.mission_id = m.id
           WHERE t.status = 'pending'
             AND COALESCE(t.lane, 'oneshot') = ?
             AND (t.next_retry_at IS NULL OR t.next_retry_at <= datetime('now'))
             AND (m.id IS NULL OR m.lifecycle_state = 'active')
             AND (COALESCE(t.agent_type,'') = 'mechanical'
                  OR COALESCE(t.runner,'') = 'mechanical')
           LIMIT 1""",
        (lane,),
    ) as cur:
        row = await cur.fetchone()
    return row is not None


async def cap_for(lane: str) -> int:
    """Concurrency cap for the given lane."""
    return ONGOING_CONCURRENCY if lane == LANE_ONGOING else ONESHOT_CONCURRENCY
