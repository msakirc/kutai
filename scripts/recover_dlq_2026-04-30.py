"""DLQ recovery helper — 2026-04-30.

Resets tasks stuck in DLQ status with recoverable error categories back to
'pending' so they can be re-attempted after a pool-pressure stabilisation.

Categories reset: availability, no_model, timeout, swap_failed
(These indicate transient resource/routing failures, not permanent logic errors.)

Usage:
    python scripts/recover_dlq_2026-04-30.py [--dry-run]

Environment:
    DB_PATH  — path to kutai.db (defaults to data/kutai.db relative to repo root)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

RECOVERABLE_CATEGORIES = {"availability", "no_model", "timeout", "swap_failed"}


async def run(dry_run: bool = False) -> None:
    from src.app.config import DB_PATH
    import aiosqlite

    print(f"DB: {DB_PATH}")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        rows = []
        async with db.execute(
            "SELECT id, agent_type, last_error_category FROM tasks WHERE status='dlq'"
        ) as cur:
            async for row in cur:
                rows.append(dict(row))

        if not rows:
            print("No DLQ tasks found.")
            return

        print(f"Found {len(rows)} DLQ task(s) total:")
        recoverable = [r for r in rows if (r.get("last_error_category") or "") in RECOVERABLE_CATEGORIES]
        non_recoverable = [r for r in rows if r not in recoverable]

        print(f"  Recoverable ({', '.join(sorted(RECOVERABLE_CATEGORIES))}): {len(recoverable)}")
        print(f"  Non-recoverable (skipped): {len(non_recoverable)}")

        if not recoverable:
            print("Nothing to reset.")
            return

        if dry_run:
            print("[DRY RUN] Would reset:")
            for r in recoverable:
                print(f"  task #{r['id']} agent={r['agent_type']} category={r['last_error_category']}")
            return

        ids = [r["id"] for r in recoverable]
        placeholders = ",".join("?" * len(ids))
        await db.execute(
            f"""UPDATE tasks
                   SET status='pending',
                       next_retry_at=NULL,
                       worker_attempts=0
                 WHERE id IN ({placeholders})""",
            ids,
        )
        await db.commit()

        print(f"Reset {len(ids)} task(s) to 'pending':")
        for r in recoverable:
            print(f"  task #{r['id']} agent={r['agent_type']} category={r['last_error_category']}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(run(dry_run=dry_run))
