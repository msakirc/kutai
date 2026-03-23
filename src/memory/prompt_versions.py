# memory/prompt_versions.py
"""
Phase 13.1 — Prompt Versioning

DB table for versioned agent prompts. Agents can load prompts from DB
instead of (or as override of) hardcoded strings. Tracks quality scores
per version and auto-promotes better versions after ≥10 tasks.
"""
from __future__ import annotations
from typing import Optional

from src.infra.logging_config import get_logger
from src.infra.db import get_db

logger = get_logger("memory.prompt_versions")

MIN_TASKS_FOR_PROMOTION = 10
PROMOTION_QUALITY_THRESHOLD = 7.0  # avg quality score to consider promotion


async def _ensure_table(db) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS prompt_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_type TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            prompt_text TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 0,
            task_count INTEGER DEFAULT 0,
            quality_sum REAL DEFAULT 0.0,
            avg_quality REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT DEFAULT ''
        )
    """)
    try:
        await db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pv_agent_version "
            "ON prompt_versions(agent_type, version)"
        )
        await db.commit()
    except Exception:
        pass


async def get_active_prompt(agent_type: str) -> Optional[str]:
    """Get the active prompt for an agent type, or None if using hardcoded."""
    try:
        db = await get_db()
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT prompt_text FROM prompt_versions "
            "WHERE agent_type = ? AND is_active = 1 "
            "ORDER BY version DESC LIMIT 1",
            (agent_type,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None
    except Exception as exc:
        logger.debug(f"get_active_prompt failed: {exc}")
        return None


async def save_prompt_version(
    agent_type: str,
    prompt_text: str,
    notes: str = "",
    activate: bool = False,
) -> int:
    """Save a new prompt version. Returns the new version number."""
    db = await get_db()
    await _ensure_table(db)

    # Get next version number
    cursor = await db.execute(
        "SELECT MAX(version) FROM prompt_versions WHERE agent_type = ?",
        (agent_type,),
    )
    row = await cursor.fetchone()
    next_version = (row[0] or 0) + 1

    if activate:
        # Deactivate existing active versions
        await db.execute(
            "UPDATE prompt_versions SET is_active = 0 WHERE agent_type = ?",
            (agent_type,),
        )

    await db.execute(
        """INSERT INTO prompt_versions (agent_type, version, prompt_text, is_active, notes)
           VALUES (?, ?, ?, ?, ?)""",
        (agent_type, next_version, prompt_text, 1 if activate else 0, notes),
    )
    await db.commit()
    logger.info(f"Saved prompt v{next_version} for {agent_type}", activate=activate)
    return next_version


async def record_prompt_quality(agent_type: str, quality_score: float) -> None:
    """Record a quality score for the active prompt version."""
    try:
        db = await get_db()
        await _ensure_table(db)
        await db.execute(
            """UPDATE prompt_versions
               SET task_count = task_count + 1,
                   quality_sum = quality_sum + ?,
                   avg_quality = (quality_sum + ?) / (task_count + 1)
               WHERE agent_type = ? AND is_active = 1""",
            (quality_score, quality_score, agent_type),
        )
        await db.commit()
        await _maybe_promote_candidate(agent_type)
    except Exception as exc:
        logger.debug(f"record_prompt_quality failed: {exc}")


async def _maybe_promote_candidate(agent_type: str) -> None:
    """Check if any non-active version has better quality and should be promoted."""
    try:
        db = await get_db()
        # Get active version stats
        c1 = await db.execute(
            "SELECT avg_quality, task_count FROM prompt_versions "
            "WHERE agent_type = ? AND is_active = 1",
            (agent_type,),
        )
        active = await c1.fetchone()
        if not active:
            return
        active_quality, active_count = active
        if active_count < MIN_TASKS_FOR_PROMOTION:
            return

        # Check if any candidate beats it significantly
        c2 = await db.execute(
            """SELECT id, version, avg_quality, task_count FROM prompt_versions
               WHERE agent_type = ? AND is_active = 0
               AND task_count >= ? AND avg_quality > ?
               ORDER BY avg_quality DESC LIMIT 1""",
            (agent_type, MIN_TASKS_FOR_PROMOTION, active_quality + 0.5),
        )
        candidate = await c2.fetchone()
        if candidate:
            cid, cversion, cquality, _ = candidate
            logger.info(
                f"Auto-promoting prompt v{cversion} for {agent_type}: "
                f"{cquality:.2f} > {active_quality:.2f}"
            )
            await db.execute(
                "UPDATE prompt_versions SET is_active = 0 WHERE agent_type = ?",
                (agent_type,),
            )
            await db.execute(
                "UPDATE prompt_versions SET is_active = 1 WHERE id = ?",
                (cid,),
            )
            await db.commit()
    except Exception as exc:
        logger.debug(f"Prompt promotion check failed: {exc}")


async def list_prompt_versions(agent_type: str) -> list[dict]:
    """List all prompt versions for an agent type."""
    try:
        db = await get_db()
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT id, version, is_active, task_count, avg_quality, created_at, notes "
            "FROM prompt_versions WHERE agent_type = ? ORDER BY version DESC",
            (agent_type,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


async def seed_from_agents() -> int:
    """
    Seed the prompt_versions table with current hardcoded prompts from all
    agent classes. Only seeds if no version exists yet for that agent type.
    Returns the number of prompts seeded.
    """
    seeded = 0
    try:
        from src.agents import get_agent, AGENT_REGISTRY
        # Create a dummy task for prompt extraction
        dummy_task = {"id": 0, "title": "seed", "description": "seed"}
        for agent_type in AGENT_REGISTRY:
            try:
                existing = await get_active_prompt(agent_type)
                if existing:
                    continue  # already has a versioned prompt
                agent = get_agent(agent_type)
                prompt = agent.get_system_prompt(dummy_task)
                if prompt and len(prompt) > 20:
                    await save_prompt_version(
                        agent_type=agent_type,
                        prompt_text=prompt,
                        notes="Auto-seeded from hardcoded prompt",
                        activate=True,
                    )
                    seeded += 1
            except Exception:
                continue
    except Exception as exc:
        logger.debug(f"seed_from_agents failed: {exc}")
    return seeded
