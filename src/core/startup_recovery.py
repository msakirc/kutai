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
    from general_beckman import recover_startup_tasks as _recover
    summary: list[str] = []

    result = await _recover()
    if result["interrupted"]:
        summary.append(f"Reset {result['interrupted']} interrupted task(s)")
    if result["backoff_cleared"]:
        summary.append(f"Cleared backoff for {result['backoff_cleared']} task(s)")

    # 2. Accelerate overdue retry gates.
    try:
        from src.infra.db import accelerate_retries
        if w := await accelerate_retries("startup"):
            summary.append(f"Accelerated {w} task(s)")
    except Exception as e:
        logger.debug(f"accelerate_retries failed: {e}")

    # 3. Stale file locks from a prior crash.
    try:
        db = await get_db()
        await db.execute("DELETE FROM file_locks")
        await db.commit()
    except Exception:
        pass

    logger.info(f"[Startup Recovery] {' | '.join(summary) or 'clean start'}")
