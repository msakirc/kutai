"""Post-restart task-queue hygiene.

Extracted from `Orchestrator._startup_recovery` during the Task 13 trim.
Resets tasks left in ``processing`` by a crashed prior run, accelerates
overdue retry gates, and clears stale file locks so dispatch resumes cleanly.
"""
from __future__ import annotations

from src.infra.db import get_db
from src.infra.logging_config import get_logger

logger = get_logger("core.startup_recovery")


async def startup_recovery() -> None:
    """Post-restart: reset stuck tasks + clear retry backoffs + clear locks."""
    db = await get_db()
    summary: list[str] = []

    # 1. Reset tasks stuck in 'processing' (prior run didn't finish them).
    c = await db.execute(
        "SELECT id, infra_resets FROM tasks WHERE status = 'processing'"
    )
    interrupted = [dict(r) for r in await c.fetchall()]
    for t in interrupted:
        ir = (t.get("infra_resets") or 0) + 1
        await db.execute(
            "UPDATE tasks SET status='pending', infra_resets=?, "
            "retry_reason='infrastructure' WHERE id=?",
            (ir, t["id"])
        )
    if interrupted:
        await db.commit()
        summary.append(f"Reset {len(interrupted)} interrupted task(s)")

    # 2. Accelerate overdue retry gates.
    try:
        from src.infra.db import accelerate_retries
        if w := await accelerate_retries("startup"):
            summary.append(f"Accelerated {w} task(s)")
    except Exception as e:
        logger.debug(f"accelerate_retries failed: {e}")

    # 3. Clear future-dated next_retry_at on ready tasks so the queue picks
    # them up on the next cycle instead of waiting out a stale backoff.
    c = await db.execute(
        "SELECT id FROM tasks WHERE status IN ('pending','ungraded') "
        "AND next_retry_at IS NOT NULL AND next_retry_at > datetime('now')"
    )
    delayed = [dict(r) for r in await c.fetchall()]
    for t in delayed:
        await db.execute(
            "UPDATE tasks SET next_retry_at=NULL WHERE id=?", (t["id"],)
        )
    if delayed:
        await db.commit()
        summary.append(f"Cleared backoff for {len(delayed)} task(s)")

    # 4. Stale file locks from a prior crash.
    try:
        await db.execute("DELETE FROM file_locks")
        await db.commit()
    except Exception:
        pass

    logger.info(f"[Startup Recovery] {' | '.join(summary) or 'clean start'}")
