# memory/feedback.py
"""
Phase 13.3 — Feedback Loop

Track implicit and explicit feedback on task outcomes.
Score: accept=+1, partial=-0.5, reject=-1
Feed into model stats and prompt version quality scores.
"""
from __future__ import annotations
from typing import Optional

from src.infra.logging_config import get_logger
from src.infra.db import get_db

logger = get_logger("memory.feedback")

FEEDBACK_ACCEPT  = "good"
FEEDBACK_PARTIAL = "partial"
FEEDBACK_REJECT  = "bad"

SCORE_MAP = {
    FEEDBACK_ACCEPT:  1.0,
    FEEDBACK_PARTIAL: -0.5,
    FEEDBACK_REJECT:  -1.0,
}


async def _ensure_table(db) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS task_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            feedback_type TEXT NOT NULL,
            score REAL NOT NULL,
            reason TEXT DEFAULT '',
            model_used TEXT DEFAULT '',
            agent_type TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_task ON task_feedback(task_id)"
        )
        await db.commit()
    except Exception:
        pass


async def record_feedback(
    task_id: int,
    feedback_type: str,
    reason: str = "",
    model_used: str = "",
    agent_type: str = "",
) -> None:
    """Record feedback for a task. feedback_type: good | partial | bad"""
    score = SCORE_MAP.get(feedback_type, 0.0)
    db = await get_db()
    await _ensure_table(db)
    await db.execute(
        """INSERT INTO task_feedback (task_id, feedback_type, score, reason, model_used, agent_type)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (task_id, feedback_type, score, reason, model_used, agent_type),
    )
    await db.commit()
    logger.info(f"Feedback recorded: task #{task_id} = {feedback_type} ({score:+.1f})")

    # Feed into model stats
    if model_used:
        try:
            from src.infra.db import record_model_call
            await record_model_call(
                model=model_used,
                agent_type=agent_type,
                success=(score > 0),
                grade=max(0, min(10, int((score + 1) * 5))),
            )
        except Exception:
            pass


async def get_task_feedback(task_id: int) -> list[dict]:
    """Get all feedback for a task."""
    try:
        db = await get_db()
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT * FROM task_feedback WHERE task_id = ? ORDER BY created_at",
            (task_id,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


async def get_feedback_stats() -> dict:
    """Get aggregated feedback statistics."""
    try:
        db = await get_db()
        await _ensure_table(db)
        cursor = await db.execute(
            """SELECT feedback_type, COUNT(*) as count, AVG(score) as avg_score
               FROM task_feedback GROUP BY feedback_type"""
        )
        rows = await cursor.fetchall()
        return {row[0]: {"count": row[1], "avg_score": row[2]} for row in rows}
    except Exception:
        return {}
