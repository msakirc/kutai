# infra/progress.py
"""
Phase 7.2 — Progress Notes

DB table for milestone/blocker/decision/artifact/log notes linked to
projects, goals, and tasks.
"""
from __future__ import annotations
from typing import Optional

from .logging_config import get_logger
from .db import get_db

logger = get_logger("infra.progress")

NOTE_MILESTONE = "milestone"
NOTE_BLOCKER   = "blocker"
NOTE_DECISION  = "decision"
NOTE_ARTIFACT  = "artifact"
NOTE_LOG       = "log"

ALL_NOTE_TYPES = (NOTE_MILESTONE, NOTE_BLOCKER, NOTE_DECISION, NOTE_ARTIFACT, NOTE_LOG)


async def _ensure_table(db) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS progress_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            goal_id INTEGER,
            task_id INTEGER,
            note_type TEXT NOT NULL DEFAULT 'log',
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        await db.execute("CREATE INDEX IF NOT EXISTS idx_progress_project ON progress_notes(project_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_progress_goal ON progress_notes(goal_id)")
        await db.commit()
    except Exception:
        pass


async def add_note(
    content: str,
    note_type: str = NOTE_LOG,
    project_id: Optional[int] = None,
    goal_id: Optional[int] = None,
    task_id: Optional[int] = None,
) -> int:
    """Add a progress note. Returns the note ID."""
    if note_type not in ALL_NOTE_TYPES:
        note_type = NOTE_LOG
    db = await get_db()
    await _ensure_table(db)
    cursor = await db.execute(
        """INSERT INTO progress_notes (project_id, goal_id, task_id, note_type, content)
           VALUES (?, ?, ?, ?, ?)""",
        (project_id, goal_id, task_id, note_type, content),
    )
    await db.commit()
    return cursor.lastrowid


async def get_notes(
    project_id: Optional[int] = None,
    goal_id: Optional[int] = None,
    task_id: Optional[int] = None,
    note_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Query progress notes with optional filters."""
    db = await get_db()
    await _ensure_table(db)

    conditions = []
    params = []
    if project_id is not None:
        conditions.append("project_id = ?")
        params.append(project_id)
    if goal_id is not None:
        conditions.append("goal_id = ?")
        params.append(goal_id)
    if task_id is not None:
        conditions.append("task_id = ?")
        params.append(task_id)
    if note_type:
        conditions.append("note_type = ?")
        params.append(note_type)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)
    cursor = await db.execute(
        f"SELECT * FROM progress_notes {where} ORDER BY created_at DESC LIMIT ?",
        params,
    )
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


def format_notes_timeline(notes: list[dict]) -> str:
    """Format notes as a markdown timeline."""
    if not notes:
        return "_No progress notes._"

    icons = {
        NOTE_MILESTONE: "🏁",
        NOTE_BLOCKER:   "🚧",
        NOTE_DECISION:  "💡",
        NOTE_ARTIFACT:  "📎",
        NOTE_LOG:       "📝",
    }
    lines = []
    for note in reversed(notes):  # oldest first
        icon = icons.get(note.get("note_type", NOTE_LOG), "📝")
        ts = note.get("created_at", "")[:16]
        content = note.get("content", "")[:200]
        lines.append(f"{icon} `{ts}` {content}")
    return "\n".join(lines)
