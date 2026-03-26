"""Shopping session memory — resume context across conversations."""

import json
import time
import uuid

import aiosqlite

from src.shopping.memory._db import get_memory_db
from src.infra.logging_config import get_logger

logger = get_logger("shopping.memory.session")

# ─── Schema ──────────────────────────────────────────────────────────────────


async def init_session_db() -> None:
    """Create session tables if they don't exist."""
    db = await get_memory_db()

    await db.execute("""
        CREATE TABLE IF NOT EXISTS shopping_sessions (
            session_id  TEXT    PRIMARY KEY,
            user_id     INTEGER NOT NULL,
            topic       TEXT    NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'active',
            summary     TEXT,
            created_at  REAL    NOT NULL,
            updated_at  REAL    NOT NULL
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_sess_user ON shopping_sessions(user_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_sess_user_topic ON shopping_sessions(user_id, topic)"
    )

    await db.execute("""
        CREATE TABLE IF NOT EXISTS session_products (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            product_json TEXT   NOT NULL,
            added_at    REAL   NOT NULL,
            FOREIGN KEY (session_id) REFERENCES shopping_sessions(session_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_sp_session ON session_products(session_id)"
    )

    await db.execute("""
        CREATE TABLE IF NOT EXISTS session_questions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            question    TEXT    NOT NULL,
            answer      TEXT,
            asked_at    REAL   NOT NULL,
            FOREIGN KEY (session_id) REFERENCES shopping_sessions(session_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_sq_session ON session_questions(session_id)"
    )

    await db.commit()
    logger.info("Session tables initialised")


# ─── Public API ──────────────────────────────────────────────────────────────


async def create_session(user_id: int, topic: str) -> str:
    """Create a new shopping session and return its session_id (uuid4)."""
    db = await get_memory_db()
    session_id = str(uuid.uuid4())
    now = time.time()
    await db.execute(
        """
        INSERT INTO shopping_sessions (session_id, user_id, topic, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, user_id, topic, now, now),
    )
    await db.commit()
    logger.info("Created session %s for user %s: %s", session_id[:8], user_id, topic)
    return session_id


async def get_session(session_id: str) -> dict:
    """Return the full session context including products and questions."""
    db = await get_memory_db()

    cur = await db.execute(
        "SELECT * FROM shopping_sessions WHERE session_id = ?",
        (session_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return {}

    session = dict(row)

    # Products discussed
    cur = await db.execute(
        "SELECT product_json, added_at FROM session_products WHERE session_id = ? ORDER BY added_at",
        (session_id,),
    )
    session["products"] = [
        {**json.loads(r["product_json"]), "_added_at": r["added_at"]}
        for r in await cur.fetchall()
    ]

    # Questions asked
    cur = await db.execute(
        "SELECT question, answer, asked_at FROM session_questions WHERE session_id = ? ORDER BY asked_at",
        (session_id,),
    )
    session["questions"] = [dict(r) for r in await cur.fetchall()]

    return session


async def update_session(session_id: str, **fields) -> None:
    """Update session fields.

    Supported fields: topic (str), status (str), summary (str).
    """
    db = await get_memory_db()

    sets: list[str] = []
    params: list = []

    for key in ("topic", "status", "summary"):
        if key in fields:
            sets.append(f"{key} = ?")
            params.append(fields[key])

    if not sets:
        return

    sets.append("updated_at = ?")
    params.append(time.time())
    params.append(session_id)

    await db.execute(
        f"UPDATE shopping_sessions SET {', '.join(sets)} WHERE session_id = ?",
        params,
    )
    await db.commit()


async def add_session_product(session_id: str, product: dict) -> None:
    """Track a product discussed during this session."""
    db = await get_memory_db()
    now = time.time()
    await db.execute(
        "INSERT INTO session_products (session_id, product_json, added_at) VALUES (?, ?, ?)",
        (session_id, json.dumps(product, ensure_ascii=False), now),
    )
    await db.execute(
        "UPDATE shopping_sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id),
    )
    await db.commit()


async def add_session_question(session_id: str, question: str, answer: str = None) -> None:
    """Record a question (and optionally its answer) within a session."""
    db = await get_memory_db()
    now = time.time()
    await db.execute(
        "INSERT INTO session_questions (session_id, question, answer, asked_at) VALUES (?, ?, ?, ?)",
        (session_id, question, answer, now),
    )
    await db.execute(
        "UPDATE shopping_sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id),
    )
    await db.commit()


async def get_recent_session(user_id: int, topic: str = None, hours: int = 24) -> dict | None:
    """Find the most recent active session for a user, optionally filtered by topic.

    Returns the full session context or None if no recent session exists.
    """
    db = await get_memory_db()
    cutoff = time.time() - hours * 3600

    if topic:
        cur = await db.execute(
            """
            SELECT session_id FROM shopping_sessions
            WHERE user_id = ? AND topic = ? AND status = 'active' AND updated_at > ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id, topic, cutoff),
        )
    else:
        cur = await db.execute(
            """
            SELECT session_id FROM shopping_sessions
            WHERE user_id = ? AND status = 'active' AND updated_at > ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id, cutoff),
        )

    row = await cur.fetchone()
    if row is None:
        return None
    return await get_session(row["session_id"])


async def clear_session(session_id: str) -> None:
    """Delete a session and all its associated products and questions."""
    db = await get_memory_db()
    await db.execute("DELETE FROM session_questions WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM session_products WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM shopping_sessions WHERE session_id = ?", (session_id,))
    await db.commit()
    logger.info("Cleared session %s", session_id[:8])
