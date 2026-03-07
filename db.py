# db.py
import aiosqlite
import hashlib
import json
import logging
from datetime import datetime
from config import DB_PATH

logger = logging.getLogger(__name__)

# ─── Connection Pool (singleton) ────────────────────────────────────────────
# Instead of opening/closing a connection on every DB call, we maintain a
# single long-lived connection.  WAL mode allows concurrent reads from the
# same connection, and a single writer avoids contention.

_db_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    """Return the shared database connection, creating it on first call."""
    global _db_connection
    if _db_connection is None:
        _db_connection = await aiosqlite.connect(DB_PATH)
        _db_connection.row_factory = aiosqlite.Row
        # Enable WAL for concurrent reads + better write performance
        await _db_connection.execute("PRAGMA journal_mode=WAL")
        await _db_connection.execute("PRAGMA synchronous=NORMAL")
        await _db_connection.execute("PRAGMA busy_timeout=5000")
    return _db_connection


async def close_db() -> None:
    """Close the shared connection (call on shutdown)."""
    global _db_connection
    if _db_connection is not None:
        # Phase 9: checkpoint WAL before closing to consolidate writes
        try:
            await _db_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        await _db_connection.close()
        _db_connection = None
        logger.info("Database connection closed")


async def init_db():
    db = await get_db()

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

    # Blackboards (Phase 13.1)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS blackboards (
            goal_id INTEGER PRIMARY KEY,
            data JSON NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Model performance stats (Phase 4)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS model_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            avg_grade REAL DEFAULT 0.0,
            avg_cost REAL DEFAULT 0.0,
            avg_latency REAL DEFAULT 0.0,
            success_rate REAL DEFAULT 1.0,
            total_calls INTEGER DEFAULT 0,
            total_successes INTEGER DEFAULT 0,
            total_grade_sum REAL DEFAULT 0.0,
            total_cost_sum REAL DEFAULT 0.0,
            total_latency_sum REAL DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model, agent_type)
        )
    """)

    # Cost budget tracking (Phase 4)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS cost_budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            scope_id TEXT,
            daily_limit REAL DEFAULT 0.0,
            total_limit REAL DEFAULT 0.0,
            spent_today REAL DEFAULT 0.0,
            spent_total REAL DEFAULT 0.0,
            last_reset_date TEXT,
            UNIQUE(scope, scope_id)
        )
    """)

    # File locks (Phase 6)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS file_locks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL,
            goal_id INTEGER,
            task_id INTEGER,
            agent_type TEXT,
            acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(filepath)
        )
    """)

    # Workspace snapshots (Phase 6)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS workspace_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_id INTEGER NOT NULL,
            task_id INTEGER,
            file_hashes JSON NOT NULL DEFAULT '{}',
            branch_name TEXT,
            commit_sha TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    # Migration: add timeout_seconds column for per-task timeouts (Phase 3)
    if "timeout_seconds" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN timeout_seconds INTEGER DEFAULT NULL"
            )
            await db.commit()
            logger.info("📊 Added timeout_seconds column to tasks table")
        except Exception as e:
            logger.debug(f"timeout_seconds column migration skipped: {e}")

    # Migration: add quality_score column for response grading (Phase 4)
    if "quality_score" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN quality_score REAL DEFAULT NULL"
            )
            await db.commit()
            logger.info("📊 Added quality_score column to tasks table")
        except Exception as e:
            logger.debug(f"quality_score column migration skipped: {e}")

    # Migration: add error_category column for error taxonomy (Phase 9)
    if "error_category" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN error_category TEXT DEFAULT NULL"
            )
            await db.commit()
            logger.info("📊 Added error_category column to tasks table")
        except Exception as e:
            logger.debug(f"error_category column migration skipped: {e}")

    # Migration: add max_cost column for per-task budgets (Phase 9)
    if "max_cost" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN max_cost REAL DEFAULT NULL"
            )
            await db.commit()
            logger.info("📊 Added max_cost column to tasks table")
        except Exception as e:
            logger.debug(f"max_cost column migration skipped: {e}")

# --- Goal Operations ---

async def add_goal(title, description, priority=5, context=None):
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO goals (title, description, priority, context) VALUES (?, ?, ?, ?)",
        (title, description, priority, json.dumps(context or {}))
    )
    await db.commit()
    return cursor.lastrowid

async def get_active_goals():
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM goals WHERE status = 'active' ORDER BY priority DESC"
    )
    return [dict(row) for row in await cursor.fetchall()]

async def update_goal(goal_id, **kwargs):
    db = await get_db()
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
    db = await get_db()
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
    db = await get_db()

    # Dedup check + insert — single-connection serialises writes via WAL,
    # so the gap between SELECT and INSERT is safe within one async task.
    task_hash = compute_task_hash(title, description, agent_type, goal_id, parent_task_id)

    cursor = await db.execute(
        """SELECT id, title, status FROM tasks
           WHERE task_hash = ?
             AND status IN ('pending', 'processing')
           LIMIT 1""",
        (task_hash,)
    )
    duplicate = await cursor.fetchone()
    if duplicate:
        dup = dict(duplicate)
        logger.info(
            f"⏭️ Task dedup: '{title[:50]}' matches pending task "
            f"#{dup['id']} — skipping creation"
        )
        return None

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
    db = await get_db()

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
    db = await get_db()
    cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None

async def get_tasks_for_goal(goal_id):
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM tasks WHERE goal_id = ? ORDER BY created_at",
        (goal_id,)
    )
    return [dict(row) for row in await cursor.fetchall()]

async def get_completed_dependency_results(depends_on):
    """Fetch results of completed dependency tasks."""
    if not depends_on:
        return {}
    db = await get_db()
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
    db = await get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    await db.execute(f"UPDATE tasks SET {sets} WHERE id = ?", values)
    await db.commit()


# ─── Task Locking (atomic claim) ────────────────────────────────────────────

async def claim_task(task_id: int) -> bool:
    """Atomically claim a task for processing.

    Uses UPDATE ... WHERE status='pending' and checks rowcount.
    Returns True if this caller won the race, False if already taken.
    """
    db = await get_db()
    cursor = await db.execute(
        "UPDATE tasks SET status = 'processing', started_at = ? "
        "WHERE id = ? AND status = 'pending'",
        (datetime.now().isoformat(), task_id)
    )
    await db.commit()
    return cursor.rowcount > 0


# ─── Transaction-safe subtask creation ───────────────────────────────────────

async def add_subtasks_atomically(
    parent_task_id: int,
    subtasks: list[dict],
    goal_id: int | None = None,
    parent_status: str = "waiting_subtasks",
    parent_result: str | None = None,
) -> list[int]:
    """Add multiple subtasks and update parent in a single transaction.

    Each subtask dict should have: title, description, agent_type, tier,
    priority, depends_on (list of task IDs).

    Returns list of created task IDs (or -1 for deduped entries).
    """
    db = await get_db()
    created_ids: list[int] = []

    # aiosqlite auto-commits by default; disable temporarily for
    # explicit transaction control.
    old_isolation = db._conn.isolation_level if db._conn else None
    try:
        await db.execute("BEGIN")

        for st in subtasks:
            title = st.get("title", "Subtask")
            description = st.get("description", "")
            agent_type = st.get("agent_type", "executor")
            task_hash = compute_task_hash(
                title, description, agent_type, goal_id, parent_task_id
            )

            # Dedup check within transaction
            cursor = await db.execute(
                """SELECT id FROM tasks
                   WHERE task_hash = ? AND status IN ('pending', 'processing')
                   LIMIT 1""",
                (task_hash,)
            )
            dup = await cursor.fetchone()
            if dup:
                created_ids.append(-1)
                continue

            cursor = await db.execute(
                """INSERT INTO tasks
                   (goal_id, parent_task_id, title, description, agent_type,
                    tier, priority, requires_approval, depends_on, context,
                    task_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, '{}', ?)""",
                (goal_id, parent_task_id, title, description, agent_type,
                 st.get("tier", "auto"), st.get("priority", 5),
                 json.dumps(st.get("depends_on", [])),
                 task_hash)
            )
            created_ids.append(cursor.lastrowid)

        # Update parent task status
        update_fields = {"status": parent_status}
        if parent_result is not None:
            update_fields["result"] = parent_result
        sets = ", ".join(f"{k} = ?" for k in update_fields)
        values = list(update_fields.values()) + [parent_task_id]
        await db.execute(f"UPDATE tasks SET {sets} WHERE id = ?", values)

        await db.commit()
    except Exception:
        await db.execute("ROLLBACK")
        raise

    return created_ids


# --- Checkpoint Operations ---

async def save_task_checkpoint(task_id: int, state: dict) -> None:
    """Persist intermediate agent state for crash recovery."""
    db = await get_db()
    await db.execute(
        "UPDATE tasks SET task_state = ? WHERE id = ?",
        (json.dumps(state), task_id)
    )
    await db.commit()


async def load_task_checkpoint(task_id: int) -> dict | None:
    """Load saved agent checkpoint state, or None if no checkpoint exists."""
    db = await get_db()
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
    db = await get_db()
    await db.execute(
        "UPDATE tasks SET task_state = NULL WHERE id = ?",
        (task_id,)
    )
    await db.commit()

async def log_conversation(task_id, role, content, model_used=None,
                           agent_type=None, cost=0):
    db = await get_db()
    await db.execute(
        """INSERT INTO conversations
           (task_id, role, content, model_used, agent_type, cost_estimate)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (task_id, role, content, model_used, agent_type, cost)
    )
    await db.commit()

async def get_recent_completed_tasks(limit=5):
    """Get the most recently completed tasks with their results."""
    db = await get_db()
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
    db = await get_db()
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
    db = await get_db()
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
    db = await get_db()
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


# ─── Phase 3: Scheduler & Task Engine ─────────────────────────────────────

# --- Scheduled Tasks ---

async def get_due_scheduled_tasks() -> list[dict]:
    """Return enabled scheduled tasks whose next_run <= now."""
    db = await get_db()
    cursor = await db.execute(
        """SELECT * FROM scheduled_tasks
           WHERE enabled = 1
             AND (next_run IS NULL OR next_run <= datetime('now'))
           ORDER BY id"""
    )
    return [dict(row) for row in await cursor.fetchall()]


async def update_scheduled_task(sched_id: int, **kwargs) -> None:
    """Update fields on a scheduled task (e.g. last_run, next_run)."""
    db = await get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [sched_id]
    await db.execute(
        f"UPDATE scheduled_tasks SET {sets} WHERE id = ?", values
    )
    await db.commit()


async def add_scheduled_task(
    title: str,
    description: str = "",
    cron_expression: str = "0 * * * *",
    agent_type: str = "executor",
    tier: str = "cheap",
    context: dict | None = None,
) -> int:
    """Create a new scheduled task."""
    db = await get_db()
    cursor = await db.execute(
        """INSERT INTO scheduled_tasks
           (title, description, cron_expression, agent_type, tier, context)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (title, description, cron_expression, agent_type, tier,
         json.dumps(context or {}))
    )
    await db.commit()
    return cursor.lastrowid


async def get_scheduled_tasks() -> list[dict]:
    """Return all scheduled tasks."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM scheduled_tasks ORDER BY id"
    )
    return [dict(row) for row in await cursor.fetchall()]


# --- Task Cancellation ---

async def cancel_task(task_id: int) -> bool:
    """Cancel a task and all its pending/processing children.

    Returns True if the task was found and cancelled.
    """
    db = await get_db()

    # Cancel the task itself (only if not already completed/failed)
    cursor = await db.execute(
        """UPDATE tasks SET status = 'cancelled',
               completed_at = datetime('now')
           WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')""",
        (task_id,)
    )
    await db.commit()
    if cursor.rowcount == 0:
        return False

    # Propagate: cancel all pending/processing children recursively
    await _cancel_children(task_id)
    return True


async def _cancel_children(parent_id: int) -> None:
    """Recursively cancel all pending/processing children of a task."""
    db = await get_db()
    cursor = await db.execute(
        """SELECT id FROM tasks
           WHERE parent_task_id = ?
             AND status IN ('pending', 'processing', 'waiting_subtasks')""",
        (parent_id,)
    )
    children = [row["id"] for row in await cursor.fetchall()]
    if not children:
        return

    placeholders = ",".join("?" * len(children))
    await db.execute(
        f"""UPDATE tasks SET status = 'cancelled',
                completed_at = datetime('now')
            WHERE id IN ({placeholders})""",
        children
    )
    await db.commit()

    # Recurse into each child's children
    for child_id in children:
        await _cancel_children(child_id)


# --- Task Reprioritization ---

async def reprioritize_task(task_id: int, new_priority: int) -> bool:
    """Change priority of a pending task. Returns True if updated."""
    db = await get_db()
    cursor = await db.execute(
        """UPDATE tasks SET priority = ?
           WHERE id = ? AND status IN ('pending', 'processing')""",
        (new_priority, task_id)
    )
    await db.commit()
    return cursor.rowcount > 0


# --- Dependency Graph ---

async def get_task_tree(goal_id: int) -> list[dict]:
    """Get all tasks for a goal, including parent-child relationships."""
    db = await get_db()
    cursor = await db.execute(
        """SELECT id, parent_task_id, title, status, agent_type,
                  priority, depends_on
           FROM tasks
           WHERE goal_id = ?
           ORDER BY id""",
        (goal_id,)
    )
    return [dict(row) for row in await cursor.fetchall()]


# ─── Phase 4: Model Intelligence Layer ────────────────────────────────────

# --- Model Stats ---

async def record_model_call(
    model: str,
    agent_type: str,
    success: bool,
    cost: float = 0.0,
    latency: float = 0.0,
    grade: float | None = None,
) -> None:
    """Record a model call for performance tracking.

    Updates running averages in model_stats table.
    """
    db = await get_db()

    # Upsert: try to get existing row
    cursor = await db.execute(
        "SELECT * FROM model_stats WHERE model = ? AND agent_type = ?",
        (model, agent_type)
    )
    existing = await cursor.fetchone()

    if existing:
        row = dict(existing)
        total = row["total_calls"] + 1
        successes = row["total_successes"] + (1 if success else 0)
        cost_sum = row["total_cost_sum"] + cost
        latency_sum = row["total_latency_sum"] + latency
        grade_sum = row["total_grade_sum"] + (grade if grade else 0)

        avg_cost = cost_sum / total if total > 0 else 0
        avg_latency = latency_sum / total if total > 0 else 0
        graded_calls = row["total_calls"]  # approximate
        if grade is not None:
            graded_calls += 1
        avg_grade = grade_sum / graded_calls if graded_calls > 0 else 0
        success_rate = successes / total if total > 0 else 0

        await db.execute(
            """UPDATE model_stats SET
                   total_calls = ?, total_successes = ?,
                   total_cost_sum = ?, total_latency_sum = ?,
                   total_grade_sum = ?,
                   avg_cost = ?, avg_latency = ?,
                   avg_grade = ?, success_rate = ?,
                   updated_at = datetime('now')
               WHERE model = ? AND agent_type = ?""",
            (total, successes, cost_sum, latency_sum, grade_sum,
             avg_cost, avg_latency, avg_grade, success_rate,
             model, agent_type)
        )
    else:
        avg_grade = grade if grade else 0.0
        await db.execute(
            """INSERT INTO model_stats
               (model, agent_type, total_calls, total_successes,
                total_cost_sum, total_latency_sum, total_grade_sum,
                avg_cost, avg_latency, avg_grade, success_rate)
               VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model, agent_type, 1 if success else 0,
             cost, latency, grade if grade else 0,
             cost, latency, avg_grade,
             1.0 if success else 0.0)
        )
    await db.commit()


async def get_model_stats(
    model: str | None = None,
    agent_type: str | None = None,
) -> list[dict]:
    """Query model performance stats.

    Filter by model and/or agent_type.  Returns all if no filters.
    """
    db = await get_db()
    query = "SELECT * FROM model_stats WHERE 1=1"
    params: list = []
    if model:
        query += " AND model = ?"
        params.append(model)
    if agent_type:
        query += " AND agent_type = ?"
        params.append(agent_type)
    query += " ORDER BY avg_grade DESC, success_rate DESC"
    cursor = await db.execute(query, params)
    return [dict(row) for row in await cursor.fetchall()]


async def get_model_performance_ranking(agent_type: str) -> list[dict]:
    """Get models ranked by performance for a specific agent type.

    Returns models sorted by: success_rate * avg_grade (composite score).
    Only includes models with >= 3 calls for statistical significance.
    """
    db = await get_db()
    cursor = await db.execute(
        """SELECT model, avg_grade, avg_cost, avg_latency,
                  success_rate, total_calls
           FROM model_stats
           WHERE agent_type = ? AND total_calls >= 3
           ORDER BY (success_rate * avg_grade) DESC""",
        (agent_type,)
    )
    return [dict(row) for row in await cursor.fetchall()]


# --- Cost Budget ---

async def get_budget(scope: str, scope_id: str | None = None) -> dict | None:
    """Get budget info for a scope (e.g. 'daily', 'goal')."""
    db = await get_db()
    if scope_id:
        cursor = await db.execute(
            "SELECT * FROM cost_budgets WHERE scope = ? AND scope_id = ?",
            (scope, scope_id)
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM cost_budgets WHERE scope = ? AND scope_id IS NULL",
            (scope,)
        )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def set_budget(
    scope: str,
    scope_id: str | None = None,
    daily_limit: float = 0.0,
    total_limit: float = 0.0,
) -> None:
    """Create or update a cost budget."""
    db = await get_db()
    existing = await get_budget(scope, scope_id)
    today = datetime.now().strftime("%Y-%m-%d")

    if existing:
        await db.execute(
            """UPDATE cost_budgets
               SET daily_limit = ?, total_limit = ?
               WHERE scope = ? AND scope_id IS ?""",
            (daily_limit, total_limit, scope, scope_id)
        )
    else:
        await db.execute(
            """INSERT INTO cost_budgets
               (scope, scope_id, daily_limit, total_limit,
                spent_today, spent_total, last_reset_date)
               VALUES (?, ?, ?, ?, 0, 0, ?)""",
            (scope, scope_id, daily_limit, total_limit, today)
        )
    await db.commit()


async def record_cost(
    cost: float,
    scope: str = "daily",
    scope_id: str | None = None,
) -> None:
    """Add cost to a budget scope. Resets daily spend if date changed."""
    db = await get_db()
    budget = await get_budget(scope, scope_id)
    if not budget:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    last_reset = budget.get("last_reset_date", "")

    # Reset daily spend if new day
    if today != last_reset:
        await db.execute(
            """UPDATE cost_budgets
               SET spent_today = ?, spent_total = spent_total + ?,
                   last_reset_date = ?
               WHERE scope = ? AND scope_id IS ?""",
            (cost, cost, today, scope, scope_id)
        )
    else:
        await db.execute(
            """UPDATE cost_budgets
               SET spent_today = spent_today + ?,
                   spent_total = spent_total + ?
               WHERE scope = ? AND scope_id IS ?""",
            (cost, cost, scope, scope_id)
        )
    await db.commit()


async def check_budget(
    scope: str = "daily",
    scope_id: str | None = None,
) -> dict:
    """Check if budget is exceeded.

    Returns:
        {"ok": True/False, "reason": str, "budget": dict}
    """
    budget = await get_budget(scope, scope_id)
    if not budget:
        return {"ok": True, "reason": "no budget set", "budget": None}

    today = datetime.now().strftime("%Y-%m-%d")
    spent_today = budget["spent_today"]
    if budget.get("last_reset_date") != today:
        spent_today = 0.0  # new day, reset hasn't happened yet

    daily_limit = budget["daily_limit"]
    total_limit = budget["total_limit"]
    spent_total = budget["spent_total"]

    if daily_limit > 0 and spent_today >= daily_limit:
        return {
            "ok": False,
            "reason": f"Daily budget exceeded: ${spent_today:.4f} / ${daily_limit:.4f}",
            "budget": budget,
        }

    if total_limit > 0 and spent_total >= total_limit:
        return {
            "ok": False,
            "reason": f"Total budget exceeded: ${spent_total:.4f} / ${total_limit:.4f}",
            "budget": budget,
        }

    return {"ok": True, "reason": "within budget", "budget": budget}


# ─── Phase 6: File Locking ──────────────────────────────────────────────────

async def acquire_file_lock(
    filepath: str,
    goal_id: int | None = None,
    task_id: int | None = None,
    agent_type: str | None = None,
) -> bool:
    """Acquire an advisory lock on a file. Returns True if acquired."""
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO file_locks (filepath, goal_id, task_id, agent_type) "
            "VALUES (?, ?, ?, ?)",
            (filepath, goal_id, task_id, agent_type),
        )
        await db.commit()
        return True
    except Exception:
        # UNIQUE constraint violation → already locked
        return False


async def release_file_lock(filepath: str) -> None:
    """Release advisory lock on a file."""
    db = await get_db()
    await db.execute("DELETE FROM file_locks WHERE filepath = ?", (filepath,))
    await db.commit()


async def release_task_locks(task_id: int) -> None:
    """Release all locks held by a specific task."""
    db = await get_db()
    await db.execute("DELETE FROM file_locks WHERE task_id = ?", (task_id,))
    await db.commit()


async def release_goal_locks(goal_id: int) -> None:
    """Release all locks held by a specific goal."""
    db = await get_db()
    await db.execute("DELETE FROM file_locks WHERE goal_id = ?", (goal_id,))
    await db.commit()


async def get_file_lock(filepath: str) -> dict | None:
    """Check who holds the lock on a file."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM file_locks WHERE filepath = ?", (filepath,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_goal_locks(goal_id: int) -> list[dict]:
    """List all locks held by a goal."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM file_locks WHERE goal_id = ?", (goal_id,)
    )
    return [dict(r) for r in await cursor.fetchall()]


# ─── Phase 6: Workspace Snapshots ───────────────────────────────────────────

async def save_workspace_snapshot(
    goal_id: int,
    file_hashes: dict,
    task_id: int | None = None,
    branch_name: str | None = None,
    commit_sha: str | None = None,
) -> int:
    """Save a workspace state snapshot before a task runs."""
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO workspace_snapshots "
        "(goal_id, task_id, file_hashes, branch_name, commit_sha) "
        "VALUES (?, ?, ?, ?, ?)",
        (goal_id, task_id, json.dumps(file_hashes), branch_name, commit_sha),
    )
    await db.commit()
    return cursor.lastrowid


async def get_latest_snapshot(goal_id: int) -> dict | None:
    """Get the most recent snapshot for a goal."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM workspace_snapshots WHERE goal_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (goal_id,),
    )
    row = await cursor.fetchone()
    if row:
        d = dict(row)
        if isinstance(d.get("file_hashes"), str):
            d["file_hashes"] = json.loads(d["file_hashes"])
        return d
    return None


async def get_snapshot(snapshot_id: int) -> dict | None:
    """Get a specific snapshot by ID."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM workspace_snapshots WHERE id = ?", (snapshot_id,)
    )
    row = await cursor.fetchone()
    if row:
        d = dict(row)
        if isinstance(d.get("file_hashes"), str):
            d["file_hashes"] = json.loads(d["file_hashes"])
        return d
    return None


# ─── Phase 9: Per-Task Cost Tracking ─────────────────────────────────────────

async def get_task_cost(task_id: int) -> float:
    """Sum up all conversation costs for a task."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT COALESCE(SUM(cost_estimate), 0) as total FROM conversations "
        "WHERE task_id = ?",
        (task_id,),
    )
    row = await cursor.fetchone()
    return float(row["total"]) if row else 0.0


async def get_goal_total_cost(goal_id: int) -> float:
    """Sum up all conversation costs across all tasks in a goal."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT COALESCE(SUM(c.cost_estimate), 0) as total "
        "FROM conversations c "
        "JOIN tasks t ON c.task_id = t.id "
        "WHERE t.goal_id = ?",
        (goal_id,),
    )
    row = await cursor.fetchone()
    return float(row["total"]) if row else 0.0


async def check_task_budget(task_id: int, additional_cost: float = 0.0) -> dict:
    """
    Check if a task's cost budget would be exceeded.

    Returns: {"ok": bool, "reason": str, "spent": float, "limit": float|None}
    """
    from db import get_task
    task = await get_task(task_id)
    if not task:
        return {"ok": True, "reason": "task not found", "spent": 0, "limit": None}

    max_cost = task.get("max_cost")
    if not max_cost:
        return {"ok": True, "reason": "no task budget set", "spent": 0, "limit": None}

    spent = await get_task_cost(task_id)
    if spent + additional_cost > max_cost:
        return {
            "ok": False,
            "reason": (
                f"Task budget exceeded: ${spent + additional_cost:.4f} "
                f"> ${max_cost:.4f}"
            ),
            "spent": spent,
            "limit": max_cost,
        }
    return {"ok": True, "reason": "within budget", "spent": spent, "limit": max_cost}
