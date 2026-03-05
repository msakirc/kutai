# db.py
import aiosqlite
import hashlib
import json
import logging
from datetime import datetime
from config import DB_PATH

logger = logging.getLogger(__name__)

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        # Check if we need to migrate by checking for old schema
        try:
            cursor = await db.execute("PRAGMA table_info(tasks)")
            columns = [row[1] for row in await cursor.fetchall()]

            if columns and "agent_type" not in columns:
                logger.warning("⚠️ Old database schema detected! Migrating...")
                await db.execute("ALTER TABLE tasks RENAME TO tasks_old")
                await db.execute("ALTER TABLE conversations RENAME TO conversations_old")
                logger.info("Old tables renamed. Creating new schema...")
        except Exception as e:
            logger.info(f"Fresh database, no migration needed: {e}")

        # Goals
        await db.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'active',
                priority INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                context JSON DEFAULT '{}'
            )
        """)

        # Tasks
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER,
                parent_task_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                agent_type TEXT DEFAULT 'executor',
                status TEXT DEFAULT 'pending',
                tier TEXT DEFAULT 'auto',
                priority INTEGER DEFAULT 5,
                requires_approval BOOLEAN DEFAULT 0,
                depends_on JSON DEFAULT '[]',
                result TEXT,
                error TEXT,
                context JSON DEFAULT '{}',
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (goal_id) REFERENCES goals(id),
                FOREIGN KEY (parent_task_id) REFERENCES tasks(id)
            )
        """)

        # Conversations
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                role TEXT,
                content TEXT,
                model_used TEXT,
                agent_type TEXT,
                cost_estimate REAL DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)

        # Memory
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Scheduled tasks
        await db.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                cron_expression TEXT,
                agent_type TEXT DEFAULT 'executor',
                tier TEXT DEFAULT 'cheap',
                enabled BOOLEAN DEFAULT 1,
                last_run TIMESTAMP,
                next_run TIMESTAMP,
                context JSON DEFAULT '{}'
            )
        """)

        await db.commit()

        # Verify schema
        cursor = await db.execute("PRAGMA table_info(tasks)")
        columns = [row[1] for row in await cursor.fetchall()]
        logger.info(f"📊 DB schema verified. Tasks columns: {columns}")

        # Migration: add task_hash column if not present
        if "task_hash" not in columns:
            try:
                await db.execute(
                    "ALTER TABLE tasks ADD COLUMN task_hash TEXT"
                )
                await db.commit()
                logger.info("📊 Added task_hash column to tasks table")
            except Exception as e:
                logger.debug(f"task_hash column migration skipped: {e}")

        # Migration: add task_state column for checkpointing
        if "task_state" not in columns:
            try:
                await db.execute(
                    "ALTER TABLE tasks ADD COLUMN task_state JSON DEFAULT NULL"
                )
                await db.commit()
                logger.info("📊 Added task_state column to tasks table")
            except Exception as e:
                logger.debug(f"task_state column migration skipped: {e}")

# --- Goal Operations ---

async def add_goal(title, description, priority=5, context=None):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO goals (title, description, priority, context) VALUES (?, ?, ?, ?)",
            (title, description, priority, json.dumps(context or {}))
        )
        await db.commit()
        return cursor.lastrowid

async def get_active_goals():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM goals WHERE status = 'active' ORDER BY priority DESC"
        )
        return [dict(row) for row in await cursor.fetchall()]

async def update_goal(goal_id, **kwargs):
    async with aiosqlite.connect(DB_PATH) as db:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [goal_id]
        await db.execute(f"UPDATE goals SET {sets} WHERE id = ?", values)
        await db.commit()


# --- Task Operations ---

def compute_task_hash(title: str, description: str, agent_type: str,
    goal_id=None, parent_task_id=None) -> str:
    """Compute a SHA-256 hash for task deduplication."""
    raw = f"{title or ''}|{description or ''}|{agent_type or ''}|{goal_id or ''}|{parent_task_id or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


async def find_duplicate_task(task_hash: str) -> dict | None:
    """Check for existing pending/processing tasks with the same hash."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT id, title, status FROM tasks
               WHERE task_hash = ?
                 AND status IN ('pending', 'processing')
               LIMIT 1""",
            (task_hash,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def add_task(title, description, goal_id=None, parent_task_id=None,
                   agent_type="executor", tier="auto", priority=5,
                   requires_approval=False, depends_on=None, context=None):
    # Deduplication check
    task_hash = compute_task_hash(title, description, agent_type, goal_id, parent_task_id)
    duplicate = await find_duplicate_task(task_hash)
    if duplicate:
        logger.info(
            f"⏭️ Task dedup: '{title[:50]}' matches pending task "
            f"#{duplicate['id']} — skipping creation"
        )
        return None

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO tasks 
               (goal_id, parent_task_id, title, description, agent_type, 
                tier, priority, requires_approval, depends_on, context,
                task_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (goal_id, parent_task_id, title, description, agent_type,
             tier, priority, requires_approval,
             json.dumps(depends_on or []), json.dumps(context or {}),
             task_hash)
        )
        await db.commit()
        return cursor.lastrowid

async def get_ready_tasks(limit=5):
    """Get pending tasks whose dependencies are all completed.

    IMPORTANT: We fetch ALL pending tasks first, then filter by deps,
    then apply the limit. Otherwise blocked high-priority tasks hide
    ready lower-priority ones.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Fetch ALL pending tasks — filter AFTER dep check
        cursor = await db.execute(
            """SELECT * FROM tasks
               WHERE status = 'pending'
               ORDER BY priority DESC, created_at ASC"""
        )
        all_pending = [dict(row) for row in await cursor.fetchall()]

        if not all_pending:
            return []

        ready = []
        blocked = []

        for task in all_pending:
            task_id = task.get("id")
            raw_deps = task.get("depends_on")

            # Parse depends_on safely
            try:
                if raw_deps is None or raw_deps == "" or raw_deps == "null":
                    deps = []
                elif isinstance(raw_deps, str):
                    parsed = json.loads(raw_deps)
                    # Ensure it's always a list
                    if isinstance(parsed, list):
                        deps = parsed
                    elif isinstance(parsed, int):
                        deps = [parsed]
                    else:
                        deps = []
                elif isinstance(raw_deps, (list, tuple)):
                    deps = list(raw_deps)
                elif isinstance(raw_deps, int):
                    deps = [raw_deps]
                else:
                    logger.warning(
                        f"Task #{task_id}: unexpected depends_on type "
                        f"{type(raw_deps)}: {raw_deps}"
                    )
                    deps = []
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Task #{task_id}: malformed depends_on '{raw_deps}': {e}")
                deps = []

            # No dependencies → immediately ready
            if not deps:
                ready.append(task)
                if len(ready) >= limit:
                    break
                continue

            # Check if ALL dependencies are completed
            placeholders = ",".join("?" * len(deps))
            dep_cursor = await db.execute(
                f"""SELECT COUNT(*) FROM tasks 
                    WHERE id IN ({placeholders}) AND status = 'completed'""",
                deps
            )
            completed_count = (await dep_cursor.fetchone())[0]

            # Also check if any dependency FAILED (unrecoverable block)
            fail_cursor = await db.execute(
                f"""SELECT COUNT(*) FROM tasks 
                    WHERE id IN ({placeholders}) AND status = 'failed'""",
                deps
            )
            failed_count = (await fail_cursor.fetchone())[0]

            if completed_count == len(deps):
                ready.append(task)
                if len(ready) >= limit:
                    break
            elif failed_count > 0:
                blocked.append((task_id, "has_failed_deps"))
                logger.debug(
                    f"Task #{task_id} blocked: {failed_count}/{len(deps)} deps failed"
                )
            else:
                blocked.append((task_id, "deps_pending"))

        logger.info(
            f"Task check: {len(all_pending)} pending, "
            f"{len(ready)} ready, {len(blocked)} blocked"
            + (f" | Ready: {[t['id'] for t in ready]}" if ready else "")
            + (f" | Blocked: {[b[0] for b in blocked[:5]]}" if blocked else "")
        )

        return ready[:limit]

async def get_task(task_id):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

async def get_tasks_for_goal(goal_id):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM tasks WHERE goal_id = ? ORDER BY created_at",
            (goal_id,)
        )
        return [dict(row) for row in await cursor.fetchall()]

async def get_completed_dependency_results(depends_on):
    """Fetch results of completed dependency tasks."""
    if not depends_on:
        return {}
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        results = {}
        for dep_id in depends_on:
            cursor = await db.execute(
                "SELECT id, title, result FROM tasks WHERE id = ?", (dep_id,)
            )
            row = await cursor.fetchone()
            if row:
                results[dep_id] = dict(row)
        return results

async def update_task(task_id, **kwargs):
    async with aiosqlite.connect(DB_PATH) as db:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [task_id]
        await db.execute(f"UPDATE tasks SET {sets} WHERE id = ?", values)
        await db.commit()


# --- Checkpoint Operations ---

async def save_task_checkpoint(task_id: int, state: dict) -> None:
    """Persist intermediate agent state for crash recovery."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tasks SET task_state = ? WHERE id = ?",
            (json.dumps(state), task_id)
        )
        await db.commit()


async def load_task_checkpoint(task_id: int) -> dict | None:
    """Load saved agent checkpoint state, or None if no checkpoint exists."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT task_state FROM tasks WHERE id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"[Checkpoint] Corrupt task_state for task #{task_id}, "
                    f"starting fresh"
                )
    return None


async def clear_task_checkpoint(task_id: int) -> None:
    """Clear checkpoint after successful completion."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tasks SET task_state = NULL WHERE id = ?",
            (task_id,)
        )
        await db.commit()

async def log_conversation(task_id, role, content, model_used=None,
                           agent_type=None, cost=0):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO conversations 
               (task_id, role, content, model_used, agent_type, cost_estimate)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (task_id, role, content, model_used, agent_type, cost)
        )
        await db.commit()

async def get_recent_completed_tasks(limit=5):
    """Get the most recently completed tasks with their results."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT id, title, result, agent_type, completed_at
               FROM tasks
               WHERE status = 'completed'
                 AND result IS NOT NULL
                 AND parent_task_id IS NULL
               ORDER BY completed_at DESC
               LIMIT ?""",
            (limit,)
        )
        return [dict(row) for row in await cursor.fetchall()]

# --- Memory Operations ---

async def store_memory(key, value, category="general", goal_id=None):
    async with aiosqlite.connect(DB_PATH) as db:
        # Upsert
        existing = await db.execute(
            "SELECT id FROM memory WHERE key = ? AND goal_id IS ?",
            (key, goal_id)
        )
        row = await existing.fetchone()
        if row:
            await db.execute(
                "UPDATE memory SET value = ?, updated_at = ? WHERE id = ?",
                (value, datetime.now().isoformat(), row[0])
            )
        else:
            await db.execute(
                "INSERT INTO memory (key, value, category, goal_id) VALUES (?, ?, ?, ?)",
                (key, value, category, goal_id)
            )
        await db.commit()

async def recall_memory(category=None, goal_id=None, limit=20):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        query = "SELECT * FROM memory WHERE 1=1"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if goal_id:
            query += " AND (goal_id = ? OR goal_id IS NULL)"
            params.append(goal_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        return [dict(row) for row in await cursor.fetchall()]

async def get_daily_stats():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        stats = {}
        for status in ["pending", "processing", "completed", "failed"]:
            cursor = await db.execute(
                "SELECT COUNT(*) as c FROM tasks WHERE status = ?", (status,)
            )
            row = await cursor.fetchone()
            stats[status] = row["c"]
        cost_cursor = await db.execute(
            """SELECT COALESCE(SUM(cost_estimate), 0) as total 
               FROM conversations 
               WHERE date(timestamp) = date('now')"""
        )
        cost_row = await cost_cursor.fetchone()
        stats["today_cost"] = cost_row["total"]
        return stats
