"""DLQ recovery — 2026-05-02 batch reset.

Today's commits land four selector / retry / dead-models fixes that
unblock the failure modes which DLQ'd 384 tasks during the prior run:

  - ff5f283: function-calling floor from agent profile
  - fa67b96: drop discovery-diff mark_dead (clears .dead_models.json
    from auto-killing every gemini + openrouter id)
  - 0781c55: 'no_model' transient pool exhaustion now defers
    (30/60/120/300/600 ladder, 10 attempts) instead of fast-DLQ
  - 4b73278: selector accepts needs_json_mode kwarg
    (constrained_emit no longer TypeErrors)

This script walks the dead_letter_tasks table, picks unresolved entries
in recoverable categories, and calls dead_letter.retry_dlq_task() on
each. That function is the canonical recovery path: phase-aware reset,
mechanical→parent redirect, exclusion clearing, checkpoint restart,
cascade-dependent reset, and DLQ entry mark resolved.

Usage:
    python scripts/recover_dlq_2026-05-02.py [--dry-run] [--all]

Default: reset categories {availability, no_model, timeout, swap_failed}
With --all: also include {quality, worker} (deterministic failures —
worth retrying with the new constrained_emit and selector floors).

Environment:
    DB_PATH  — path to kutai.db (defaults to ./kutai.db)

Run with KutAI stopped to avoid singleton-conn contention.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_RECOVERABLE = {"availability", "no_model", "timeout", "swap_failed"}
_ALL_RECOVERABLE = _DEFAULT_RECOVERABLE | {"quality", "worker", "unknown", ""}


async def run(*, dry_run: bool, include_all: bool) -> None:
    from src.app.config import DB_PATH
    print(f"DB: {DB_PATH}")

    cats = _ALL_RECOVERABLE if include_all else _DEFAULT_RECOVERABLE
    print(f"Recoverable categories: {sorted(c for c in cats if c)}")

    from src.infra.db import get_db
    db = await get_db()

    cur = await db.execute(
        """SELECT task_id, error_category
             FROM dead_letter_tasks
            WHERE resolved_at IS NULL"""
    )
    rows = [(r[0], r[1] or "") for r in await cur.fetchall()]
    if not rows:
        print("No unresolved DLQ tasks.")
        return

    by_cat: dict[str, list[int]] = {}
    for tid, cat in rows:
        by_cat.setdefault(cat, []).append(tid)

    print(f"\nUnresolved DLQ: {len(rows)} total")
    for cat, ids in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        marker = "RESET" if cat in cats else "skip "
        print(f"  [{marker}] {cat or '(none)':16s}  {len(ids)}")

    targets = [tid for tid, cat in rows if cat in cats]
    if not targets:
        print("\nNothing in recoverable categories.")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would call retry_dlq_task on {len(targets)} task(s).")
        for tid in targets[:20]:
            print(f"  #{tid}")
        if len(targets) > 20:
            print(f"  ... and {len(targets) - 20} more")
        return

    print(f"\nResetting {len(targets)} task(s) via retry_dlq_task()...")
    from src.infra.dead_letter import retry_dlq_task

    ok_count = 0
    fail_count = 0
    for tid in targets:
        try:
            success = await retry_dlq_task(tid)
            if success:
                ok_count += 1
            else:
                fail_count += 1
                print(f"  #{tid} retry returned False")
        except Exception as e:
            fail_count += 1
            print(f"  #{tid} raised: {type(e).__name__}: {str(e)[:100]}")

    print(f"\nDone. ok={ok_count} failed={fail_count}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    include_all = "--all" in sys.argv
    asyncio.run(run(dry_run=dry, include_all=include_all))
