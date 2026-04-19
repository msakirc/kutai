"""Task queue: eligibility filter + priority boost + paused-pattern filter.

Lane classification stays for Task 3; Task 4 deletes it.
"""
from __future__ import annotations

from src.infra.db import get_ready_tasks, claim_task, update_task
from src.infra.times import from_db, utc_now

from general_beckman.paused_patterns import is_paused


def classify_lane(task: dict) -> str:  # DELETED in Task 4
    """Return the dispatch lane for a task row.

    - ``mechanical`` — ``agent_type == "mechanical"``
    - ``cloud_llm``  — cloud-preferred agent types (researcher/planner/architect)
    - ``local_llm``  — everything else
    """
    if task.get("agent_type") == "mechanical":
        return "mechanical"
    if task.get("agent_type") in {"researcher", "planner", "architect"}:
        return "cloud_llm"
    return "local_llm"


def _effective_priority(task: dict) -> float:
    """Base priority + age boost (starvation prevention).

    +0.1 per hour waiting, capped at +1.0.
    Handles missing/malformed created_at gracefully.
    """
    base = float(task.get("priority", 5))
    created = task.get("created_at", "")
    if not created:
        return base
    try:
        age_h = (utc_now() - from_db(created)).total_seconds() / 3600
    except Exception:
        return base
    return base + min(age_h * 0.1, 1.0)


async def pick_ready_task(saturated_lanes: set[str]) -> dict | None:
    """Return one ready task eligible for dispatch, or None.

    ``saturated_lanes`` contains lanes where capacity snapshot says no room.
    Tasks bound to those lanes are skipped. Applies age-boost sort and
    paused-pattern filter internally (replaces orchestrator inline blocks).
    """
    rows = await get_ready_tasks(limit=8)
    # Age-boost sort (stable: preserves DB tie-break for equal boosts)
    rows.sort(key=_effective_priority, reverse=True)
    for row in rows:
        if is_paused(row):
            continue
        lane = classify_lane(row)
        if lane in saturated_lanes:
            continue
        claimed = await claim_task(row["id"])
        if claimed:
            return row
    return None


async def count_pending_cloud_tasks() -> int:  # DELETED in Task 4
    """Return the number of ready tasks classified as cloud_llm lane.

    Used by the look-ahead module to estimate quota pressure.
    """
    rows = await get_ready_tasks(limit=30)
    return sum(1 for r in rows if classify_lane(r) == "cloud_llm")


async def unclaim(task: dict) -> None:
    """Revert a claimed task back to pending so the caller can re-consider it
    on the next tick."""
    await update_task(task["id"], status="pending")
