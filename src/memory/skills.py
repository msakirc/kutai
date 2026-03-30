# memory/skills.py
"""
Phase 13.2 — Skill Library

DB table of reusable task approaches. On successful novel tasks, extract
the approach and store it. Inject relevant skills as context on similar
future tasks.

Matching uses both regex trigger_pattern AND vector similarity search.
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
    """Add or update a skill. Returns the skill ID. Also embeds for vector search."""
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
    skill_id = cursor.lastrowid or 0
    logger.info(f"Skill saved: {name}")

    # Embed skill description for vector search
    try:
        from src.memory.vector_store import embed_and_store
        embed_text = f"{name}: {description}"
        if tool_sequence:
            embed_text += f" Approach: {tool_sequence}"
        await embed_and_store(
            text=embed_text,
            metadata={
                "type": "skill",
                "skill_name": name,
                "skill_id": skill_id,
            },
            collection="semantic",
            doc_id=f"skill:{name}",
        )
    except Exception as exc:
        logger.debug(f"Skill embedding failed (non-critical): {exc}")

    return skill_id


async def find_relevant_skills(task_text: str, limit: int = 3) -> list[dict]:
    """
    Find skills matching the task text.

    Uses both regex trigger_pattern matching AND vector similarity search,
    then merges and deduplicates results.
    """
    try:
        db = await get_db()
        await _ensure_table(db)

        # 1. Regex-based matching (original approach)
        cursor = await db.execute(
            "SELECT * FROM skills WHERE success_count > 0 ORDER BY success_count DESC"
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        all_skills = [dict(zip(cols, row)) for row in rows]

        task_lower = task_text.lower()
        regex_matches = []
        for skill in all_skills:
            pattern = skill.get("trigger_pattern", "")
            if not pattern:
                continue
            if re.search(pattern, task_lower, re.IGNORECASE):
                regex_matches.append(skill)

        # 2. Vector-based matching
        vector_matches = []
        try:
            from src.memory.vector_store import query as vquery
            results = await vquery(
                text=task_text,
                collection="semantic",
                top_k=limit * 2,
                where={"type": "skill"},
            )
            # Look up full skill data from DB for vector matches
            for r in results:
                # Filter out distant matches (distance > 0.7)
                if r.get("distance", 1.0) > 0.7:
                    continue
                meta = r.get("metadata", {})
                skill_name = meta.get("skill_name", "")
                if not skill_name:
                    continue
                # Skip if already in regex matches
                if any(s.get("name") == skill_name for s in regex_matches):
                    continue
                # Find full skill record
                for skill in all_skills:
                    if skill.get("name") == skill_name:
                        vector_matches.append(skill)
                        break
        except Exception as exc:
            logger.debug(f"Vector skill search failed (falling back to regex): {exc}")

        # Merge: regex matches first (stronger signal), then vector matches
        merged = regex_matches + vector_matches
        return merged[:limit]

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
