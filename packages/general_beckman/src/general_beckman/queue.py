"""Task queue: eligibility + priority boost + paused-pattern filter."""
from __future__ import annotations

from src.infra.db import get_ready_tasks, claim_task, update_task
from src.infra.times import from_db, utc_now

from general_beckman.admission import compute_urgency
from general_beckman.paused_patterns import is_paused


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


async def pick_ready_task() -> dict | None:
    """Return one ready task eligible for dispatch, or None.

    Admission-layer gates (pool-pressure, VRAM headroom) live in
    ``next_task()``. This helper just claims the top-priority ready row.
    """
    rows = await get_ready_tasks(limit=8)
    # Age-boost sort (stable: preserves DB tie-break for equal boosts)
    rows.sort(key=_effective_priority, reverse=True)
    for row in rows:
        if is_paused(row):
            continue
        claimed = await claim_task(row["id"])
        if claimed:
            return row
    return None


async def pick_ready_top_k(k: int = 5) -> list[dict]:
    """Return up to ``k`` ready tasks ordered by urgency descending.

    Unlike :func:`pick_ready_task`, this helper does NOT claim any rows.
    The admission loop inspects several candidates and claims at most one.
    Paused tasks are filtered out.
    """
    rows = await get_ready_tasks(limit=max(k * 2, 8))
    eligible = [r for r in rows if not is_paused(r)]
    eligible.sort(key=compute_urgency, reverse=True)
    return eligible[:k]


async def unclaim(task: dict) -> None:
    """Revert a claimed task back to pending so the caller can re-consider it
    on the next tick."""
    await update_task(task["id"], status="pending")
