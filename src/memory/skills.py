# memory/skills.py
"""
Phase 13.2 — Skill Library

DB table of reusable task approaches. On successful novel tasks, extract
the approach and store it. Inject relevant skills as context on similar
future tasks.
"""
from __future__ import annotations
import re
from typing import Optional

from src.infra.logging_config import get_logger
from src.infra.db import get_db

logger = get_logger("memory.skills")


async def _ensure_table(db) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            trigger_pattern TEXT DEFAULT '',
            tool_sequence TEXT DEFAULT '',
            examples TEXT DEFAULT '',
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.commit()


async def add_skill(
    name: str,
    description: str,
    trigger_pattern: str = "",
    tool_sequence: str = "",
    examples: str = "",
) -> int:
    """Add or update a skill. Returns the skill ID."""
    db = await get_db()
    await _ensure_table(db)
    cursor = await db.execute(
        """INSERT INTO skills (name, description, trigger_pattern, tool_sequence, examples)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
           description=excluded.description,
           trigger_pattern=excluded.trigger_pattern,
           tool_sequence=excluded.tool_sequence,
           examples=excluded.examples,
           updated_at=CURRENT_TIMESTAMP""",
        (name, description, trigger_pattern, tool_sequence, examples),
    )
    await db.commit()
    logger.info(f"Skill saved: {name}")
    return cursor.lastrowid or 0


async def find_relevant_skills(task_text: str, limit: int = 3) -> list[dict]:
    """Find skills whose trigger_pattern matches the task text."""
    try:
        db = await get_db()
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT * FROM skills WHERE success_count > 0 ORDER BY success_count DESC"
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        all_skills = [dict(zip(cols, row)) for row in rows]

        task_lower = task_text.lower()
        matches = []
        for skill in all_skills:
            pattern = skill.get("trigger_pattern", "")
            if not pattern:
                continue
            if re.search(pattern, task_lower, re.IGNORECASE):
                matches.append(skill)
            if len(matches) >= limit:
                break
        return matches
    except Exception as exc:
        logger.debug(f"find_relevant_skills failed: {exc}")
        return []


async def record_skill_outcome(name: str, success: bool) -> None:
    """Record success/failure for a skill."""
    try:
        db = await get_db()
        await _ensure_table(db)
        if success:
            await db.execute(
                "UPDATE skills SET success_count = success_count + 1 WHERE name = ?",
                (name,),
            )
        else:
            await db.execute(
                "UPDATE skills SET failure_count = failure_count + 1 WHERE name = ?",
                (name,),
            )
        await db.commit()
    except Exception:
        pass


async def list_skills() -> list[dict]:
    """List all skills."""
    try:
        db = await get_db()
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT id, name, description, success_count, failure_count, created_at "
            "FROM skills ORDER BY success_count DESC"
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


def format_skills_for_prompt(skills: list[dict]) -> str:
    """Format relevant skills as a context block for agent prompts."""
    if not skills:
        return ""
    lines = ["## Relevant Skills from Library\n"]
    for s in skills:
        lines.append(f"### {s['name']}")
        lines.append(s.get("description", ""))
        if s.get("tool_sequence"):
            lines.append(f"**Approach:** {s['tool_sequence']}")
        if s.get("examples"):
            lines.append(f"**Example:** {s['examples'][:200]}")
        lines.append("")
    return "\n".join(lines)
