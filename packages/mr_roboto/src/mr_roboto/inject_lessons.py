"""inject_lessons — cross-mission lesson injector (Z2 T4C).

Reads the top-N lessons for a given stack/domain from ``mission_lessons``
(populated by the T4A/T4B sibling) and writes them into the mission's
``context`` JSON bucket as ``lessons_top_n``.  Coulson's prompt builder
reads the bucket and renders a "## Watch out for" block.

Public API
----------
inject_lessons(mission_id, stack, domain=None, limit=5) -> dict
    Returns ``{"ok": True, "lessons_count": N, "mission_id": mission_id}``.
    Skips gracefully when N == 0 or when the T4A helper is not yet on the
    branch (try/except stubs the call and returns ok with count 0).
"""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.inject_lessons")


async def inject_lessons(
    mission_id: int,
    stack: str,
    domain: str | None = None,
    limit: int = 5,
) -> dict:
    """Fetch top-N lessons for ``stack``/``domain`` and write to mission context.

    Parameters
    ----------
    mission_id:
        Target mission row in the ``missions`` table.
    stack:
        Stack identifier (e.g. ``"fastapi"``).  Passed directly to
        ``top_mission_lessons``; may be a ``+``-joined composite.
    domain:
        Optional domain filter (e.g. ``"auth"``).  ``None`` = all domains.
    limit:
        Maximum lessons to inject (default 5, ranked by occurrences × recency).

    Returns
    -------
    dict
        ``{"ok": True, "lessons_count": N, "mission_id": mission_id}``
    """
    # ── 1. Query top lessons (lazy import; T4A may not be on branch yet) ──
    lessons: list[dict] = []
    try:
        from src.infra.db import top_mission_lessons  # type: ignore[attr-defined]
        lessons = await top_mission_lessons(stack=stack, domain=domain, limit=limit)
    except (ImportError, AttributeError):
        # T4A/T4B helper not yet merged — degrade gracefully.
        logger.debug(
            "inject_lessons: top_mission_lessons not available "
            "(T4A not yet merged); returning ok with 0 lessons"
        )
        lessons = []
    except Exception as exc:
        logger.warning(f"inject_lessons: top_mission_lessons failed: {exc}")
        lessons = []

    if not lessons:
        return {"ok": True, "lessons_count": 0, "mission_id": mission_id}

    # ── 2. Read existing mission context ──
    try:
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT context FROM missions WHERE id = ?",
            (mission_id,),
        ) as cur:
            row = await cur.fetchone()
    except Exception as exc:
        logger.warning(f"inject_lessons: could not read mission context: {exc}")
        return {"ok": True, "lessons_count": 0, "mission_id": mission_id}

    if row is None:
        logger.warning(f"inject_lessons: mission {mission_id} not found")
        return {"ok": True, "lessons_count": 0, "mission_id": mission_id}

    raw_ctx = row[0] or "{}"
    if isinstance(raw_ctx, str):
        try:
            ctx = json.loads(raw_ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    elif isinstance(raw_ctx, dict):
        ctx = raw_ctx
    else:
        ctx = {}

    # ── 3. Idempotent write — only update if different ──
    lesson_items = [
        {
            "pattern": les.get("pattern", ""),
            "fix": les.get("fix", ""),
            "severity": les.get("severity", "info"),
            "stack": les.get("stack", stack),
            "domain": les.get("domain"),
            "occurrences": les.get("occurrences", 1),
        }
        for les in lessons
    ]

    existing = ctx.get("lessons_top_n")
    if existing == lesson_items:
        # Already up-to-date — skip the write.
        return {"ok": True, "lessons_count": len(lesson_items), "mission_id": mission_id}

    ctx["lessons_top_n"] = lesson_items
    new_ctx = json.dumps(ctx, ensure_ascii=False)

    try:
        from general_beckman import update_mission_fields as _umf
        await _umf(mission_id, context=new_ctx)
    except Exception as exc:
        logger.warning(f"inject_lessons: could not write mission context: {exc}")
        return {"ok": True, "lessons_count": 0, "mission_id": mission_id}

    logger.info(
        f"inject_lessons: wrote {len(lesson_items)} lessons "
        f"(stack={stack!r}, domain={domain!r}) to mission {mission_id}"
    )
    return {"ok": True, "lessons_count": len(lesson_items), "mission_id": mission_id}
