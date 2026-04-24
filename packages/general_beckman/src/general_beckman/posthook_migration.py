"""One-shot migration from legacy 'ungraded' rows to the post-hook shape.

Runs once per process via the `_migrated` sentinel. Safe to call
repeatedly; no-op after first successful run.
"""
from __future__ import annotations

import asyncio
import json

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthook_migration")

_migrated: bool = False
_lock: asyncio.Lock = asyncio.Lock()


async def run() -> None:
    """Migrate stale ungraded rows; drop the defunct pending_llm_summaries table."""
    global _migrated
    if _migrated:
        return
    async with _lock:
        if _migrated:
            return

        from src.infra.db import get_db, add_task

        db = await get_db()

        # Step 1: migrate ungraded rows without _pending_posthooks.
        # Materialise all rows first; we CANNOT hold a cursor open while
        # calling add_task (it starts its own transaction) or executing
        # UPDATEs — aiosqlite raises "cannot start a transaction within a
        # transaction" in that case.
        cursor = await db.execute(
            "SELECT id, mission_id, context FROM tasks WHERE status='ungraded'"
        )
        raw_rows = await cursor.fetchall()
        await cursor.close()
        rows = [dict(r) for r in raw_rows]
        await db.commit()  # close any implicit read txn
        migrated_count = 0
        for row in rows:
            try:
                ctx = json.loads(row["context"] or "{}")
                # Callers that already passed a JSON string get it wrapped
                # once more by add_task; peel a second layer when present.
                if isinstance(ctx, str):
                    try:
                        ctx = json.loads(ctx)
                    except (json.JSONDecodeError, TypeError):
                        ctx = {}
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            if not isinstance(ctx, dict):
                ctx = {}
            if ctx.get("_pending_posthooks"):
                continue
            ctx["_pending_posthooks"] = ["grade"]
            await db.execute(
                "UPDATE tasks SET context = ? WHERE id = ?",
                (json.dumps(ctx), row["id"]),
            )
            await db.commit()
            # Spawn the grader post-hook (add_task manages its own txn).
            await add_task(
                title=f"Grade task #{row['id']}",
                description="",
                agent_type="grader",
                mission_id=row["mission_id"],
                depends_on=[],
                context={
                    "source_task_id": row["id"],
                    "generating_model": ctx.get("generating_model", ""),
                },
            )
            migrated_count += 1

        # Step 2: drop orphaned llm_summary table. Swallow lock errors
        # so the migration flag still flips — the table is defunct, so
        # leaving it around is harmless, and retrying every tick only
        # thrashes on the same lock contention while re-migrating step 1.
        try:
            await db.execute("DROP TABLE IF EXISTS pending_llm_summaries")
            await db.commit()
        except Exception as exc:
            logger.warning(
                "posthook_migration: DROP pending_llm_summaries skipped",
                error=str(exc),
            )

        _migrated = True
        logger.info(
            "posthook_migration complete",
            ungraded_migrated=migrated_count,
        )
