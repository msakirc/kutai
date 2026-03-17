# user_inputs.py
"""
Tracks user-submitted bug reports, feature requests, notes, and feedback.
DB table: user_inputs
"""

from src.infra.logging_config import get_logger
from src.infra.db import get_db

logger = get_logger("infra.user_inputs")


async def init_user_inputs_table():
    """Create user_inputs table if it doesn't exist."""
    async with get_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,          -- 'bug' | 'feature' | 'ui_note' | 'review' | 'feedback'
                content TEXT NOT NULL,
                related_goal_id INTEGER,
                priority TEXT DEFAULT 'normal',  -- 'low' | 'normal' | 'high' | 'critical'
                status TEXT DEFAULT 'new',   -- 'new' | 'triaged' | 'in_progress' | 'done'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_user_inputs_type ON user_inputs(type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_user_inputs_status ON user_inputs(status)")
        await db.commit()


async def log_input(
    input_type: str,
    content: str,
    related_goal_id: int | None = None,
    priority: str = "normal",
) -> int:
    """Store a user input. Returns the new record id."""
    await init_user_inputs_table()
    async with get_db() as db:
        cursor = await db.execute(
            """INSERT INTO user_inputs (type, content, related_goal_id, priority)
               VALUES (?, ?, ?, ?)""",
            (input_type, content, related_goal_id, priority),
        )
        await db.commit()
        row_id = cursor.lastrowid
    logger.info("user input logged", type=input_type, id=row_id, priority=priority)
    return row_id


async def list_inputs(input_type: str | None = None, status: str | None = None) -> list[dict]:
    """List user inputs, optionally filtered by type or status."""
    await init_user_inputs_table()
    async with get_db() as db:
        query = "SELECT * FROM user_inputs WHERE 1=1"
        params = []
        if input_type:
            query += " AND type = ?"
            params.append(input_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT 100"
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def update_input_status(input_id: int, status: str) -> bool:
    """Update status of a user input."""
    await init_user_inputs_table()
    async with get_db() as db:
        await db.execute(
            "UPDATE user_inputs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, input_id),
        )
        await db.commit()
    logger.info("user input status updated", id=input_id, status=status)
    return True
