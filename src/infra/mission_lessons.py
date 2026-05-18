"""mission_lessons — cross-mission failure memory.

Schema lives in src/infra/db.py (registered via apply_migration in init_db).
This module provides the public helpers:

  upsert_mission_lesson(...)  → int          (inserted-or-updated row id)
  top_mission_lessons(...)    → list[dict]   (top-N by weighted recency)
  suppress_mission_lesson(id) → None         (founder mute)
  emit_lessons_from_dlq_patterns()           (DLQ → lessons sink, manual CLI)

CLI entry:
  python -m src.infra.mission_lessons emit-dlq
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("infra.mission_lessons")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_pattern(pattern: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    p = pattern.lower()
    p = re.sub(r"\s+", " ", p).strip()
    p = p.rstrip(".,;:!?")
    return p


def _dedup_key(stack: str, domain: str, pattern: str) -> str:
    """sha256(stack\\ndomain\\nnormalized_pattern)[:32]."""
    raw = f"{stack}\n{domain}\n{_normalize_pattern(pattern)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def upsert_mission_lesson(
    stack: str,
    domain: str,
    pattern: str,
    *,
    fix: str = "",
    severity: str = "warning",
    source_kind: str,
    source_ref: Optional[dict] = None,
) -> int:
    """Insert or bump a lesson row. Returns the row id.

    On conflict (same dedup_key):
    - occurrences + 1
    - last_seen_at refreshed
    - fix updated only if new fix is non-empty
    """
    from src.infra.db import get_db

    db = await get_db()
    key = _dedup_key(stack, domain, pattern)
    ref_json = json.dumps(source_ref or {})

    cursor = await db.execute(
        """
        INSERT INTO mission_lessons
            (stack, domain, pattern, fix, severity, occurrences,
             dedup_key, source_kind, source_ref)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(dedup_key) DO UPDATE SET
            occurrences  = occurrences + 1,
            last_seen_at = CURRENT_TIMESTAMP,
            fix = CASE WHEN excluded.fix != '' THEN excluded.fix
                       ELSE fix
                  END
        """,
        (stack, domain, pattern, fix, severity, key, source_kind, ref_json),
    )
    await db.commit()

    # Fetch actual row id (may be the existing row on conflict).
    cur2 = await db.execute(
        "SELECT id FROM mission_lessons WHERE dedup_key = ?", (key,)
    )
    row = await cur2.fetchone()
    return int(row[0]) if row else (cursor.lastrowid or 0)


async def top_mission_lessons(
    stack: str,
    domain: Optional[str] = None,
    *,
    limit: int = 5,
) -> list[dict]:
    """Return top-N lessons ordered by a cheap recency-weighted score.

    Score proxy: ``occurrences * (1.0 / (1.0 + julianday('now') - julianday(last_seen_at)))``

    This is a monotone proxy for ``occurrences * exp(-age_days / 180)``:
    same sign, cheaper in SQLite (no exp). Documented choice: SQLite has
    no built-in exp(); the proxy is accurate enough for lesson ranking
    where sub-linear decay is fine.

    Suppressed rows (suppressed=1) are excluded.
    domain=None matches any domain.
    """
    from src.infra.db import get_db

    db = await get_db()

    if domain is not None:
        cursor = await db.execute(
            """
            SELECT id, stack, domain, pattern, fix, severity,
                   occurrences, source_kind, source_ref, created_at, last_seen_at,
                   occurrences * (1.0 / (1.0 + julianday('now') - julianday(last_seen_at)))
                       AS score
            FROM mission_lessons
            WHERE stack = ? AND domain = ? AND suppressed = 0
            ORDER BY score DESC
            LIMIT ?
            """,
            (stack, domain, limit),
        )
    else:
        cursor = await db.execute(
            """
            SELECT id, stack, domain, pattern, fix, severity,
                   occurrences, source_kind, source_ref, created_at, last_seen_at,
                   occurrences * (1.0 / (1.0 + julianday('now') - julianday(last_seen_at)))
                       AS score
            FROM mission_lessons
            WHERE stack = ? AND suppressed = 0
            ORDER BY score DESC
            LIMIT ?
            """,
            (stack, limit),
        )

    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        try:
            d["source_ref"] = json.loads(d.get("source_ref") or "{}")
        except Exception:
            pass
        result.append(d)
    return result


async def suppress_mission_lesson(lesson_id: int) -> None:
    """Founder-mute: mark lesson suppressed=1 (excluded from top_mission_lessons)."""
    from src.infra.db import get_db

    db = await get_db()
    await db.execute(
        "UPDATE mission_lessons SET suppressed = 1 WHERE id = ?",
        (lesson_id,),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# DLQ → lessons populator
# ---------------------------------------------------------------------------

async def emit_lessons_from_dlq_patterns() -> int:
    """Scan recent DLQ rows, group by (stack, error_category), upsert lessons.

    Only groups with occurrences >= 3 are emitted (avoids noise from
    one-off failures). Stack is read from mission.tech_stack_detected
    artifact; defaults to "unknown" if absent.

    Returns the number of lesson rows upserted.
    """
    from src.infra.db import get_db

    db = await get_db()

    # Fetch all unresolved DLQ rows from the last 30 days.
    cursor = await db.execute(
        """
        SELECT d.id, d.mission_id, d.error, d.error_category,
               d.task_id,
               COALESCE(t.retry_reason, '') AS feedback
        FROM dead_letter_tasks d
        LEFT JOIN tasks t ON t.id = d.task_id
        WHERE d.resolved_at IS NULL
          AND d.quarantined_at >= datetime('now', '-30 days')
        ORDER BY d.mission_id, d.error_category
        """
    )
    rows = await cursor.fetchall()

    if not rows:
        logger.info("emit_lessons_from_dlq_patterns: no unresolved DLQ rows")
        return 0

    # Group by (mission_id, error_category).
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row[1], row[3])  # (mission_id, error_category)
        groups[key].append({
            "dlq_id": row[0],
            "mission_id": row[1],
            "error": row[2] or "",
            "error_category": row[3] or "unknown",
            "task_id": row[4],
            "feedback": row[5] or "",
        })

    emitted = 0
    for (mission_id, error_category), entries in groups.items():
        if len(entries) < 3:
            continue

        # Resolve stack from mission artifact.
        stack = await _resolve_stack(db, mission_id)

        # Pattern = first non-empty error, truncated.
        pattern = ""
        fix = ""
        dlq_ids = []
        for e in entries:
            dlq_ids.append(e["dlq_id"])
            if not pattern and e["error"].strip():
                pattern = e["error"].strip()[:120]
            if not fix and e["feedback"].strip():
                fix = e["feedback"].strip()[:300]

        if not pattern:
            pattern = f"Repeated {error_category} failures"

        try:
            await upsert_mission_lesson(
                stack=stack,
                domain=error_category,
                pattern=pattern,
                fix=fix,
                severity="warning",
                source_kind="dlq_pattern",
                source_ref={"mission_id": mission_id, "dlq_ids": dlq_ids[:20]},
            )
            emitted += 1
        except Exception as exc:
            logger.warning(
                "emit_lessons_from_dlq_patterns: upsert failed",
                mission_id=mission_id,
                category=error_category,
                error=str(exc),
            )

    logger.info("emit_lessons_from_dlq_patterns: emitted %d lesson(s)", emitted)
    return emitted


async def _resolve_stack(db, mission_id: Optional[int]) -> str:
    """Pull tech_stack_detected from mission context; default 'unknown'."""
    if not mission_id:
        return "unknown"
    try:
        cursor = await db.execute(
            "SELECT context FROM missions WHERE id = ?", (mission_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return "unknown"
        ctx = json.loads(row[0] or "{}")
        return str(ctx.get("tech_stack_detected") or "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "emit-dlq":
        async def _main():
            # Bootstrap DB (creates tables if needed).
            from src.infra.db import init_db
            await init_db()
            n = await emit_lessons_from_dlq_patterns()
            print(f"Emitted {n} lesson(s) from DLQ patterns.")

        _asyncio.run(_main())
    else:
        print("Usage: python -m src.infra.mission_lessons emit-dlq")
        sys.exit(1)
