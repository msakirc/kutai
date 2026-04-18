"""Task queue: eligibility filter + priority + lane classification.

Reads ready tasks from the DB via :func:`src.infra.db.get_ready_tasks`, skips
ones whose lane is saturated (per the snapshot passed by the caller), and
claims the first viable candidate.
"""
from __future__ import annotations

from src.infra.db import get_ready_tasks, claim_task


def classify_lane(task: dict) -> str:
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


async def pick_ready_task(saturated_lanes: set[str]) -> dict | None:
    """Return one ready task eligible for dispatch, or None.

    ``saturated_lanes`` contains lanes where the capacity snapshot says no
    room. Tasks bound to those lanes are skipped. ``get_ready_tasks`` is
    responsible for applying priority/eligibility filters (dependency gates,
    statuses != pending, etc.) — this function layers lane pre-gating on top.
    """
    rows = await get_ready_tasks(limit=8)
    for row in rows:
        lane = classify_lane(row)
        if lane in saturated_lanes:
            continue
        claimed = await claim_task(row["id"])
        if claimed:
            return row
    return None


async def count_pending_cloud_tasks() -> int:
    """Return the number of ready tasks classified as cloud_llm lane.

    Used by the look-ahead module to estimate quota pressure.
    """
    rows = await get_ready_tasks(limit=30)
    return sum(1 for r in rows if classify_lane(r) == "cloud_llm")


async def unclaim(task: dict) -> None:
    """Revert a claimed task back to pending so the caller can re-consider it
    on the next tick."""
    from src.infra.db import update_task
    await update_task(task["id"], status="pending")
