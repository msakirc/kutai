"""Yalayut Phase 4 — internal-hint auto-capture.

``capture_hint(task, outcome)`` is the post-hook body that replaces the old
``src/memory/skills.py`` auto-capture. When a task succeeded after 2+
iterations (i.e. something non-trivial was learned), it embeds the task
description and upserts an ``internal_hint`` artifact into ``yalayut_index``.
"""
from __future__ import annotations

import re

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

logger = get_logger("yalayut.capture")

#: Minimum iterations for a task to be worth capturing as a hint.
MIN_ITERATIONS_FOR_CAPTURE: int = 2


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:60] or "hint"


async def capture_hint(task: dict, outcome: dict) -> None:
    """Capture a successful 2+-iteration task as an internal_hint artifact.

    No-op when the task failed or completed in a single iteration — there is
    no transferable strategy to capture from a trivially-solved task.
    """
    if (outcome or {}).get("status") != "completed":
        return
    iterations = int((outcome or {}).get("iterations") or 0)
    if iterations < MIN_ITERATIONS_FOR_CAPTURE:
        return

    title = (task or {}).get("title") or ""
    description = (task or {}).get("description") or ""
    if not title and not description:
        return

    name = f"internal-{_slug(title)}"
    body = f"{title}\n\n{description}".strip()

    # Embed the body for vector matching (multilingual-e5-base, 768d).
    embedding_blob = None
    try:
        from src.memory.embeddings import embed_text
        vec = await embed_text(body)
        import struct
        embedding_blob = struct.pack(f"{len(vec)}f", *vec)
    except Exception as e:  # noqa: BLE001 — capture must never crash on_finish
        logger.debug("capture_hint embedding failed: %s", e)

    db = await get_db()
    now = to_db(utc_now())
    # Upsert: a repeated pattern refreshes the existing row, not a duplicate.
    cur = await db.execute(
        "SELECT id FROM yalayut_index WHERE source = 'internal' AND name = ?",
        (name,))
    existing = await cur.fetchone()
    await cur.close()
    if existing:
        await db.execute(
            "UPDATE yalayut_index SET body_excerpt = ?, embedding = ?, "
            "vetted_at = ? WHERE id = ?",
            (body[:500], embedding_blob, now, existing[0]))
        # Repeat capture of the same pattern — a reusable EXTERNAL skill would
        # beat re-deriving this internally. Record a reactive hint_miss signal.
        try:
            from yalayut.discovery.demand import record as _record_demand
            await _record_demand(
                source_step_pattern=f"hint_miss:{name}",
                intent_keywords=[w for w in title.split() if len(w) > 2][:12],
                signal_type="hint_miss",
                confidence=0.3,
            )
        except Exception as exc:  # noqa: BLE001 — capture must never crash
            logger.debug("hint_miss demand signal skipped: %s", exc)
    else:
        await db.execute(
            "INSERT INTO yalayut_index "
            "(artifact_type, kind, source, owner, name, name_original, "
            " version, body_excerpt, embedding, vet_tier, exposure_class, "
            " applies_to, mechanizable, enabled, created_at, vetted_at) "
            "VALUES ('skill', 'internal_hint', 'internal', 'kutai', ?, ?, "
            " '1.0.0', ?, ?, 0, 'inject', 'execution', 0, 1, ?, ?)",
            (name, title[:120], body[:500], embedding_blob, now, now))
    await db.commit()
    logger.info("internal_hint captured", name=name, iterations=iterations)
