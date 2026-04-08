# db.py
import aiosqlite
import hashlib
import json
from datetime import datetime, timedelta, timezone

from src.app.config import DB_PATH
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, db_now, to_db, DB_FMT

logger = get_logger("infra.db")

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

    # Migration: goals → missions
    try:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='goals'")
        if await cursor.fetchone():
            await db.execute("ALTER TABLE goals RENAME TO missions")
            await db.commit()
            logger.info("Migrated: goals → missions")
    except Exception as e:
        logger.debug(f"goals→missions migration skipped: {e}")

    # Migration: goal_id → mission_id in all tables
    _GOAL_ID_TABLES = ["tasks", "memory", "file_locks", "approval_requests",
                        "workspace_snapshots", "workflow_checkpoints"]
    for tbl in _GOAL_ID_TABLES:
        try:
            await db.execute(f"ALTER TABLE {tbl} RENAME COLUMN goal_id TO mission_id")
        except Exception:
            pass
    try:
        await db.execute("ALTER TABLE blackboards RENAME COLUMN goal_id TO mission_id")
    except Exception:
        pass
    await db.commit()

    # Migration: Add project fields to missions table
    for col, default in [("workflow", ""), ("repo_path", ""), ("language", ""), ("framework", "")]:
        try:
            await db.execute(f"ALTER TABLE missions ADD COLUMN {col} TEXT DEFAULT '{default}'")
        except Exception:
            pass  # Column already exists

    # Migrate project data into missions (if projects table exists)
    try:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'")
        if await cursor.fetchone():
            await db.execute("""
                UPDATE missions SET
                    repo_path = COALESCE((SELECT p.repo_path FROM projects p WHERE p.id = missions.project_id), ''),
                    language = COALESCE((SELECT p.language FROM projects p WHERE p.id = missions.project_id), ''),
                    framework = COALESCE((SELECT p.framework FROM projects p WHERE p.id = missions.project_id), '')
                WHERE project_id IS NOT NULL
            """)
            await db.commit()
            logger.info("Migrated project data into missions table")
    except Exception as e:
        logger.debug(f"Project migration skipped: {e}")

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

    # Missions
    await db.execute("""
        CREATE TABLE IF NOT EXISTS missions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'active',
            priority INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            context JSON DEFAULT '{}',
            workflow TEXT DEFAULT '',
            repo_path TEXT DEFAULT '',
            language TEXT DEFAULT '',
            framework TEXT DEFAULT ''
        )
    """)

    # Tasks
    await db.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER,
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
            worker_attempts INTEGER DEFAULT 0,
            max_worker_attempts INTEGER DEFAULT 6,
            grade_attempts INTEGER DEFAULT 0,
            max_grade_attempts INTEGER DEFAULT 3,
            next_retry_at TIMESTAMP,
            retry_reason TEXT,
            failed_in_phase TEXT,
            infra_resets INTEGER DEFAULT 0,
            exhaustion_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (mission_id) REFERENCES missions(id),
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
            mission_id INTEGER,
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
            mission_id INTEGER PRIMARY KEY,
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
            mission_id INTEGER,
            task_id INTEGER,
            agent_type TEXT,
            acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(filepath)
        )
    """)

    # Approval requests (Resilience)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS approval_requests (
            task_id INTEGER PRIMARY KEY,
            mission_id INTEGER,
            title TEXT,
            details TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
        )
    """)

    # Workspace snapshots (Phase 6)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS workspace_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER NOT NULL,
            task_id INTEGER,
            file_hashes JSON NOT NULL DEFAULT '{}',
            branch_name TEXT,
            commit_sha TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Credentials (External Service Integration)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS credentials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_name TEXT UNIQUE NOT NULL,
            encrypted_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS workflow_checkpoints (
            mission_id INTEGER PRIMARY KEY,
            workflow_name TEXT NOT NULL,
            current_phase TEXT,
            completed_phases TEXT DEFAULT '[]',
            failed_step_id TEXT,
            checkpoint_at TEXT,
            metadata TEXT DEFAULT '{}'
        )
    """)

    # Todo items
    await db.execute("""
        CREATE TABLE IF NOT EXISTS todo_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            priority TEXT DEFAULT 'normal',
            due_date TIMESTAMP,
            status TEXT DEFAULT 'pending',
            source TEXT DEFAULT 'explicit',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            suggestion TEXT,
            suggestion_agent TEXT,
            suggestion_at TIMESTAMP
        )
    """)

    # Web source quality tracking
    await db.execute("""
        CREATE TABLE IF NOT EXISTS web_source_quality (
            domain TEXT PRIMARY KEY,
            success_count INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0,
            block_count INTEGER DEFAULT 0,
            avg_relevance REAL DEFAULT 0.0,
            last_success TIMESTAMP,
            last_failure TIMESTAMP,
            last_block TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # User preferences (key-value store for location, display settings, etc.)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Free API registry (auto-growth)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS free_api_registry (
            name TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            base_url TEXT NOT NULL,
            auth_type TEXT DEFAULT 'none',
            env_var TEXT,
            rate_limit TEXT,
            description TEXT,
            example_endpoint TEXT,
            source TEXT DEFAULT 'static',
            verified INTEGER DEFAULT 0,
            last_checked TIMESTAMP,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Skills — strategies and injection tracking
    await db.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            skill_type TEXT DEFAULT 'auto',
            strategies TEXT DEFAULT '[]',
            injection_count INTEGER DEFAULT 0,
            injection_success INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
        )
    """)

    # Skill injection A/B metrics
    await db.execute("""
        CREATE TABLE IF NOT EXISTS skill_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            skill_name TEXT NOT NULL,
            injected INTEGER DEFAULT 1,
            task_succeeded INTEGER DEFAULT 0,
            iterations_used INTEGER DEFAULT 0,
            agent_type TEXT DEFAULT '',
            task_duration_seconds REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Smart Resource Integration tables ──

    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_keywords (
            api_name TEXT NOT NULL,
            keyword TEXT NOT NULL,
            source TEXT DEFAULT 'description',
            UNIQUE(api_name, keyword)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_keywords_kw ON api_keywords(keyword)"
    )

    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_category_patterns (
            category TEXT PRIMARY KEY,
            pattern TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS smart_search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
            query TEXT NOT NULL,
            layer INTEGER NOT NULL,
            source TEXT,
            success INTEGER DEFAULT 1,
            response_ms INTEGER
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_reliability (
            api_name TEXT PRIMARY KEY,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            consecutive_failures INTEGER DEFAULT 0,
            last_success TEXT,
            last_failure TEXT,
            status TEXT DEFAULT 'active'
        )
    """)

    # Migration: add consecutive_failures column if missing
    try:
        await db.execute("ALTER TABLE api_reliability ADD COLUMN consecutive_failures INTEGER DEFAULT 0")
    except Exception:
        pass  # column already exists

    await db.commit()

    # Seed todo reminder (every 2h during Turkey daytime: 9,11,13,15,17,19,21 TR = 6,8,10,12,14,16,18 UTC)
    await db.execute("""
        INSERT OR IGNORE INTO scheduled_tasks (id, title, description, cron_expression, agent_type, enabled, context)
        VALUES (9999, 'Todo Reminder', 'Send pending todo items to user', '0 6,8,10,12,14,16,18 * * *', 'system', 1, '{"type": "todo_reminder"}')
    """)
    # Migrate existing row from Turkey-time cron to UTC cron
    await db.execute("""
        UPDATE scheduled_tasks SET cron_expression = '0 6,8,10,12,14,16,18 * * *'
        WHERE id = 9999 AND cron_expression = '0 9,11,13,15,17,19,21 * * *'
    """)

    # Seed price watch checker (daily at noon UTC = 15:00 Turkey time)
    await db.execute("""
        INSERT OR IGNORE INTO scheduled_tasks (id, title, description, cron_expression, agent_type, enabled, context)
        VALUES (9998, 'Price Watch Check', 'Re-scrape watched products and notify on price drops', '0 12 * * *', 'system', 1, '{"type": "price_watch_check"}')
    """)

    await db.commit()

    # Migration: fix next_run / last_run values that were stored with
    # Python's datetime.isoformat() format (2026-03-26T10:00:00+00:00)
    # instead of SQLite-compatible format (2026-03-26 10:00:00).
    # SQLite's datetime('now') returns the latter, so isoformat values
    # never compare as <= datetime('now'), causing scheduled tasks to
    # never fire.  Strip timezone suffix and replace the 'T' separator.
    try:
        await db.execute("""
            UPDATE scheduled_tasks
            SET next_run = REPLACE(SUBSTR(next_run, 1, 19), 'T', ' ')
            WHERE next_run LIKE '%T%'
        """)
        await db.execute("""
            UPDATE scheduled_tasks
            SET last_run = REPLACE(SUBSTR(last_run, 1, 19), 'T', ' ')
            WHERE last_run LIKE '%T%'
        """)
        await db.commit()
        logger.info("Migration: fixed isoformat timestamps in scheduled_tasks")
    except Exception as e:
        logger.debug(f"scheduled_tasks timestamp migration skipped: {e}")

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
            logger.info("Added max_cost column to tasks table")
        except Exception as e:
            logger.debug(f"max_cost column migration skipped: {e}")

    # Migration: add cost column for per-task cost tracking (Phase 9.4)
    if "cost" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN cost REAL DEFAULT 0"
            )
            await db.commit()
            logger.info("Added cost column to tasks table")
        except Exception as e:
            logger.debug(f"cost column migration skipped: {e}")

    # Migration: add sleep_state column for signal-based sleeping queue (S11)
    if "sleep_state" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN sleep_state TEXT"
            )
            await db.commit()
            logger.info("Added sleep_state column to tasks table")
        except Exception as e:
            logger.debug(f"sleep_state column migration skipped: {e}")

    # Migration: Unified Task Lifecycle — new retry columns
    if "worker_attempts" not in columns and "attempts" not in columns:
        for col_sql in [
            "ALTER TABLE tasks ADD COLUMN worker_attempts INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN max_worker_attempts INTEGER DEFAULT 6",
            "ALTER TABLE tasks ADD COLUMN grade_attempts INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN max_grade_attempts INTEGER DEFAULT 3",
            "ALTER TABLE tasks ADD COLUMN next_retry_at TIMESTAMP",
            "ALTER TABLE tasks ADD COLUMN retry_reason TEXT",
            "ALTER TABLE tasks ADD COLUMN failed_in_phase TEXT",
            "ALTER TABLE tasks ADD COLUMN infra_resets INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN exhaustion_reason TEXT",
        ]:
            try:
                await db.execute(col_sql)
                await db.commit()
            except Exception:
                pass  # column may already exist
        logger.info("Added unified task lifecycle columns")

        # Run data migration
        await _migrate_task_lifecycle(db)

    # Migration: Retry Pipeline Overhaul — rename columns, add new ones
    if "attempts" in columns and "worker_attempts" not in columns:
        for sql in [
            "ALTER TABLE tasks RENAME COLUMN attempts TO worker_attempts",
            "ALTER TABLE tasks RENAME COLUMN max_attempts TO max_worker_attempts",
            "ALTER TABLE tasks ADD COLUMN infra_resets INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN exhaustion_reason TEXT",
        ]:
            try:
                await db.execute(sql)
                await db.commit()
            except Exception:
                pass
        logger.info("Applied retry pipeline overhaul migration")

    # Migration: Add infra_resets/exhaustion_reason if missing (fresh DB with worker_attempts already)
    if "worker_attempts" in columns and "infra_resets" not in columns:
        for sql in [
            "ALTER TABLE tasks ADD COLUMN infra_resets INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN exhaustion_reason TEXT",
        ]:
            try:
                await db.execute(sql)
                await db.commit()
            except Exception:
                pass
        logger.info("Added infra_resets/exhaustion_reason columns")

    # Migration: add suggestion columns to todo_items
    for col in ["suggestion", "suggestion_agent", "suggestion_at"]:
        try:
            await db.execute(f"ALTER TABLE todo_items ADD COLUMN {col} TEXT")
        except Exception:
            pass  # already exists

    # ── Performance indexes on common query patterns ──
    _indexes = [
        ("idx_tasks_status", "tasks", "status"),
        ("idx_tasks_mission_id", "tasks", "mission_id"),
        ("idx_tasks_status_priority", "tasks", "status, priority DESC"),
        ("idx_tasks_mission_status", "tasks", "mission_id, status"),
        ("idx_tasks_hash", "tasks", "task_hash"),
        ("idx_tasks_parent", "tasks", "parent_task_id"),
        ("idx_tasks_created", "tasks", "created_at"),
        ("idx_missions_status", "missions", "status"),
        ("idx_conversations_task_id", "conversations", "task_id"),
        ("idx_memory_mission_category", "memory", "mission_id, category"),
        ("idx_model_stats_model_agent", "model_stats", "model, agent_type"),
        ("idx_blackboards_mission", "blackboards", "mission_id"),
        ("idx_credentials_service", "credentials", "service_name"),
        ("idx_todo_status", "todo_items", "status"),
        ("idx_todo_created", "todo_items", "created_at"),
        ("idx_web_source_quality_domain", "web_source_quality", "domain"),
        ("idx_free_api_registry_category", "free_api_registry", "category"),
        ("idx_free_api_registry_source", "free_api_registry", "source"),
        ("idx_skill_metrics_skill", "skill_metrics", "skill_name"),
    ]
    for idx_name, table, columns_str in _indexes:
        try:
            await db.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({columns_str})"
            )
        except Exception as e:
            logger.debug(f"Index {idx_name} creation skipped: {e}")

    # One-time cleanup: purge legacy skills without strategies
    try:
        await db.execute("DELETE FROM skills WHERE strategies = '[]' OR strategies IS NULL")
    except Exception:
        pass

    await db.commit()

async def _migrate_task_lifecycle(db) -> None:
    """One-time migration: sleeping/paused/rejected → unified model."""
    try:
        # Backfill worker_attempts from retry_count (legacy column)
        await db.execute(
            "UPDATE tasks SET worker_attempts = COALESCE(retry_count, 0) "
            "WHERE worker_attempts = 0 AND COALESCE(retry_count, 0) > 0"
        )
        # Backfill max_worker_attempts from max_retries (add 3 for grading headroom)
        await db.execute(
            "UPDATE tasks SET max_worker_attempts = COALESCE(max_retries, 3) + 3 "
            "WHERE max_worker_attempts = 6 AND max_retries IS NOT NULL AND max_retries != 3"
        )
        # Convert sleeping → pending with next_retry_at
        await db.execute(
            """UPDATE tasks SET status = 'pending',
               next_retry_at = json_extract(sleep_state, '$.next_timer_wake')
               WHERE status = 'sleeping'"""
        )
        # Convert paused → pending with 10-min delay
        await db.execute(
            """UPDATE tasks SET status = 'pending',
               next_retry_at = datetime('now', '+10 minutes')
               WHERE status = 'paused'"""
        )
        # Rename needs_clarification → waiting_human
        await db.execute(
            "UPDATE tasks SET status = 'waiting_human' "
            "WHERE status = 'needs_clarification'"
        )
        # Fix rejected → cancelled
        await db.execute(
            "UPDATE tasks SET status = 'cancelled' WHERE status = 'rejected'"
        )
        await db.commit()
        logger.info("Migrated task lifecycle data")
    except Exception as e:
        logger.warning(f"Task lifecycle data migration error: {e}")


# --- Mission Operations ---

async def add_mission(title, description, priority=5, context=None,
                      workflow=None, repo_path=None, language=None, framework=None):
    db = await get_db()
    cursor = await db.execute(
        """INSERT INTO missions (title, description, priority, context, workflow, repo_path, language, framework)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (title, description, priority, json.dumps(context or {}),
         workflow or "", repo_path or "", language or "", framework or "")
    )
    await db.commit()
    return cursor.lastrowid

async def get_mission(mission_id):
    """Fetch a single mission by ID."""
    db = await get_db()
    cursor = await db.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None

async def get_active_missions():
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM missions WHERE status = 'active' ORDER BY priority DESC"
    )
    return [dict(row) for row in await cursor.fetchall()]

# ─── Column whitelists (prevent SQL injection via dynamic kwargs) ─────────

_MISSION_COLUMNS = frozenset({
    "title", "description", "status", "priority",
    "completed_at", "context", "workflow", "repo_path", "language", "framework",
})

_TASK_COLUMNS = frozenset({
    "title", "description", "agent_type", "status", "tier", "priority",
    "requires_approval", "depends_on", "result", "error", "error_category",
    "context", "worker_attempts", "max_worker_attempts", "started_at", "completed_at",
    "task_hash", "max_cost", "cost", "quality_score", "sleep_state",
    "next_retry_at", "retry_reason",
    # Unified lifecycle columns
    "grade_attempts", "max_grade_attempts",
    "failed_in_phase", "infra_resets", "exhaustion_reason",
})


def _validate_columns(kwargs: dict, whitelist: frozenset, table: str) -> None:
    """Raise ValueError if any kwarg key is not in the column whitelist."""
    bad = set(kwargs.keys()) - whitelist
    if bad:
        raise ValueError(
            f"Invalid column(s) for {table}: {bad}. "
            f"Allowed: {sorted(whitelist)}"
        )


async def update_mission(mission_id, **kwargs):
    _validate_columns(kwargs, _MISSION_COLUMNS, "missions")
    db = await get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [mission_id]
    await db.execute(f"UPDATE missions SET {sets} WHERE id = ?", values)
    await db.commit()


# --- Task Operations ---

def compute_task_hash(title: str, description: str, agent_type: str,
    mission_id=None, parent_task_id=None) -> str:
    """Compute a SHA-256 hash for task deduplication."""
    raw = f"{title or ''}|{description or ''}|{agent_type or ''}|{mission_id or ''}|{parent_task_id or ''}"
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


async def add_task(title, description, mission_id=None, parent_task_id=None,
                   agent_type="executor", tier="auto", priority=5,
                   requires_approval=False, depends_on=None, context=None):
    db = await get_db()

    # Atomic dedup + insert — wrapped in explicit transaction to prevent
    # race conditions between concurrent async coroutines.
    task_hash = compute_task_hash(title, description, agent_type, mission_id, parent_task_id)

    try:
        await db.execute("BEGIN IMMEDIATE")

        cursor = await db.execute(
            """SELECT id, title, status, started_at FROM tasks
               WHERE task_hash = ?
                 AND status IN ('pending', 'processing')
               LIMIT 1""",
            (task_hash,)
        )
        duplicate = await cursor.fetchone()
        if duplicate:
            dup = dict(duplicate)
            # If the duplicate is stuck in 'processing' for >10 minutes,
            # reset it instead of blocking the new task creation.
            if dup["status"] == "processing" and dup.get("started_at"):
                stuck_cursor = await db.execute(
                    """SELECT 1 FROM tasks
                       WHERE id = ? AND status = 'processing'
                         AND started_at < datetime('now', '-10 minutes')""",
                    (dup["id"],)
                )
                if await stuck_cursor.fetchone():
                    logger.warning(
                        f"Task dedup: duplicate #{dup['id']} stuck in "
                        f"processing — resetting to pending"
                    )
                    await db.execute(
                        "UPDATE tasks SET status = 'pending', "
                        "worker_attempts = COALESCE(worker_attempts, 0) + 1 "
                        "WHERE id = ?",
                        (dup["id"],)
                    )
                    await db.commit()
                    return dup["id"]
            logger.info(
                f"⏭️ Task dedup: '{title[:50]}' matches pending task "
                f"#{dup['id']} — skipping creation"
            )
            await db.execute("ROLLBACK")
            return None

        cursor = await db.execute(
            """INSERT INTO tasks
               (mission_id, parent_task_id, title, description, agent_type,
                tier, priority, requires_approval, depends_on, context,
                task_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mission_id, parent_task_id, title, description, agent_type,
             tier, priority, requires_approval,
             json.dumps(depends_on or []), json.dumps(context or {}),
             task_hash)
        )
        await db.commit()
        return cursor.lastrowid
    except Exception:
        try:
            await db.execute("ROLLBACK")
        except Exception:
            pass
        raise

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
           AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))
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

        # Check dependency statuses (completed + skipped count as resolved)
        placeholders = ",".join("?" * len(deps))
        dep_cursor = await db.execute(
            f"""SELECT COUNT(*) FROM tasks
                WHERE id IN ({placeholders}) AND status = 'completed'""",
            deps
        )
        completed_count = (await dep_cursor.fetchone())[0]

        skip_cursor = await db.execute(
            f"""SELECT COUNT(*) FROM tasks
                WHERE id IN ({placeholders}) AND status = 'skipped'""",
            deps
        )
        skipped_count = (await skip_cursor.fetchone())[0]

        resolved_count = completed_count + skipped_count

        # Also check if any dependency FAILED (unrecoverable block)
        fail_cursor = await db.execute(
            f"""SELECT COUNT(*) FROM tasks
                WHERE id IN ({placeholders}) AND status = 'failed'""",
            deps
        )
        failed_count = (await fail_cursor.fetchone())[0]

        if resolved_count == len(deps) and completed_count > 0:
            # All deps resolved AND at least one completed → ready
            ready.append(task)
            if len(ready) >= limit:
                break
        elif resolved_count == len(deps) and completed_count == 0:
            # All deps resolved but ALL are skipped → auto-skip this task
            await db.execute(
                "UPDATE tasks SET status = 'skipped', error = 'dependency_skipped' WHERE id = ?",
                (task_id,)
            )
            await db.commit()
            logger.info(f"Task #{task_id} auto-skipped: all deps are skipped")
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


async def get_blocked_task_summary():
    """Return summary of blocked pending tasks: total count and top blocker task IDs."""
    db = await get_db()

    # Count all pending tasks
    cursor = await db.execute(
        "SELECT COUNT(*) FROM tasks WHERE status = 'pending'"
    )
    total_pending = (await cursor.fetchone())[0]

    # Count pending tasks that have unresolved dependencies
    cursor = await db.execute(
        """SELECT t.id, t.depends_on FROM tasks t
           WHERE t.status = 'pending' AND t.depends_on IS NOT NULL
           AND t.depends_on != '' AND t.depends_on != 'null' AND t.depends_on != '[]'"""
    )
    blocked_rows = await cursor.fetchall()

    blocked_count = 0
    blocker_counts = {}  # task_id -> how many tasks it blocks

    for row in blocked_rows:
        raw_deps = row[1]
        try:
            if isinstance(raw_deps, str):
                deps = json.loads(raw_deps)
                if isinstance(deps, int):
                    deps = [deps]
                elif not isinstance(deps, list):
                    deps = []
            elif isinstance(raw_deps, (list, tuple)):
                deps = list(raw_deps)
            elif isinstance(raw_deps, int):
                deps = [raw_deps]
            else:
                deps = []
        except (json.JSONDecodeError, TypeError):
            deps = []

        if not deps:
            continue

        # Check if any dep is NOT completed/skipped
        placeholders = ",".join("?" * len(deps))
        dep_cursor = await db.execute(
            f"""SELECT COUNT(*) FROM tasks
                WHERE id IN ({placeholders}) AND status IN ('completed', 'skipped')""",
            deps
        )
        resolved = (await dep_cursor.fetchone())[0]

        if resolved < len(deps):
            blocked_count += 1
            # Track which unresolved deps are blocking
            unresolved_cursor = await db.execute(
                f"""SELECT id FROM tasks
                    WHERE id IN ({placeholders}) AND status NOT IN ('completed', 'skipped')""",
                deps
            )
            for dep_row in await unresolved_cursor.fetchall():
                blocker_counts[dep_row[0]] = blocker_counts.get(dep_row[0], 0) + 1

    # Top blockers sorted by how many tasks they block
    top_blockers = sorted(blocker_counts.items(), key=lambda x: -x[1])[:3]

    return {
        "blocked_count": blocked_count,
        "total_pending": total_pending,
        "top_blockers": top_blockers,  # [(task_id, count), ...]
    }


async def get_task(task_id):
    db = await get_db()
    cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None

async def get_tasks_for_mission(mission_id):
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM tasks WHERE mission_id = ? ORDER BY created_at",
        (mission_id,)
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
    _validate_columns(kwargs, _TASK_COLUMNS, "tasks")
    db = await get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    await db.execute(f"UPDATE tasks SET {sets} WHERE id = ?", values)
    await db.commit()


async def update_task_by_context_field(
    mission_id: int, field: str, value: str, **kwargs
):
    """Update tasks matching a JSON context field within a mission.

    Uses SQLite's json_extract to find tasks where
    ``context->>'$.{field}' = value`` and applies the given updates.

    Example::

        await update_task_by_context_field(
            mission_id=30,
            field="workflow_step_id",
            value="1.3",
            status="skipped",
        )
    """
    _validate_columns(kwargs, _TASK_COLUMNS, "tasks")
    db = await get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [mission_id, value]
    await db.execute(
        f"UPDATE tasks SET {sets} "
        f"WHERE mission_id = ? AND json_extract(context, '$.{field}') = ?",
        values,
    )
    await db.commit()


async def accelerate_retries(reason: str) -> int:
    """Pull next_retry_at to now for tasks waiting on availability.

    Called from: model_swap, gpu_available, rate_limit_reset,
    quota_restored, circuit_breaker_reset.

    Resets last_avail_delay in context so backoff starts fresh.
    Covers both phases: pending (worker) and ungraded (grading).

    Returns number of tasks accelerated.
    """
    import json as _json

    db = await get_db()
    cursor = await db.execute(
        """SELECT id, context FROM tasks
           WHERE status IN ('pending', 'ungraded')
           AND next_retry_at > datetime('now')
           AND retry_reason IN ('availability', 'quality')"""
    )
    rows = [dict(r) for r in await cursor.fetchall()]

    for row in rows:
        try:
            ctx = _json.loads(row.get("context") or "{}")
        except (ValueError, TypeError):
            ctx = {}
        ctx["last_avail_delay"] = 0
        await db.execute(
            """UPDATE tasks SET next_retry_at = datetime('now'),
               context = ? WHERE id = ?""",
            (_json.dumps(ctx), row["id"]),
        )

    if rows:
        await db.commit()
        logger.info(f"Accelerated {len(rows)} task(s) | reason={reason}")
    return len(rows)


# ─── Task Locking (atomic claim) ────────────────────────────────────────────

async def claim_task(task_id: int) -> bool:
    """Atomically claim a task for processing.

    Uses UPDATE ... WHERE status='pending' and checks rowcount.
    Returns True if this caller won the race, False if already taken.
    """
    db = await get_db()
    # Use strftime format (space-separated, no TZ) so SQLite datetime()
    # comparisons in the watchdog work correctly.  isoformat() produces
    # a 'T' separator which breaks `started_at < datetime('now', ...)`.
    now_str = db_now()
    cursor = await db.execute(
        "UPDATE tasks SET status = 'processing', started_at = ? "
        "WHERE id = ? AND status = 'pending'",
        (now_str, task_id)
    )
    await db.commit()
    return cursor.rowcount > 0


# ─── Transaction-safe subtask creation ───────────────────────────────────────

async def add_subtasks_atomically(
    parent_task_id: int,
    subtasks: list[dict],
    mission_id: int | None = None,
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
                title, description, agent_type, mission_id, parent_task_id
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
                   (mission_id, parent_task_id, title, description, agent_type,
                    tier, priority, requires_approval, depends_on, context,
                    task_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, '{}', ?)""",
                (mission_id, parent_task_id, title, description, agent_type,
                 st.get("tier", "auto"), st.get("priority", 5),
                 json.dumps(st.get("depends_on", [])),
                 task_hash)
            )
            created_ids.append(cursor.lastrowid)

        # Update parent task status (safe: keys are hardcoded above)
        update_fields = {"status": parent_status}
        if parent_result is not None:
            update_fields["result"] = parent_result
        _validate_columns(update_fields, _TASK_COLUMNS, "tasks")
        sets = ", ".join(f"{k} = ?" for k in update_fields)
        values = list(update_fields.values()) + [parent_task_id]
        await db.execute(f"UPDATE tasks SET {sets} WHERE id = ?", values)

        await db.commit()
    except Exception:
        await db.execute("ROLLBACK")
        raise

    return created_ids


# ─── Transaction-safe task creation (no parent) ──────────────────────────────

async def insert_tasks_atomically(
    tasks: list[dict],
    mission_id: int,
) -> list[int]:
    """Insert multiple tasks for a mission in a single transaction.

    Each task dict may have: title, description, agent_type, tier, priority,
    depends_on (list of task IDs), context (dict).

    Uses compute_task_hash for dedup within the batch (and against existing
    pending/processing tasks).

    Returns list of created task IDs (or -1 for deduped entries).
    """
    db = await get_db()
    created_ids: list[int] = []

    try:
        await db.execute("BEGIN")

        seen_hashes: set[str] = set()

        for t in tasks:
            title = t.get("title", "Task")
            description = t.get("description", "")
            agent_type = t.get("agent_type", "executor")
            task_hash = compute_task_hash(
                title, description, agent_type, mission_id, None
            )

            # Dedup within this batch
            if task_hash in seen_hashes:
                created_ids.append(-1)
                continue
            seen_hashes.add(task_hash)

            # Dedup against existing DB tasks
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
                   (mission_id, parent_task_id, title, description, agent_type,
                    tier, priority, requires_approval, depends_on, context,
                    task_hash)
                   VALUES (?, NULL, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (mission_id, title, description, agent_type,
                 t.get("tier", "auto"), t.get("priority", 5),
                 json.dumps(t.get("depends_on", [])),
                 json.dumps(t.get("context", {})),
                 task_hash)
            )
            created_ids.append(cursor.lastrowid)

        await db.commit()
    except Exception:
        await db.execute("ROLLBACK")
        raise

    return created_ids


# ─── Skip propagation (transitive closure) ────────────────────────────────────

async def propagate_skips(mission_id: int) -> int:
    """Propagate skip status transitively through dependency chains.

    For each pending task whose depends_on contains a skipped dep:
    - If ALL deps are completed or skipped, AND at least one is completed,
      the task is still ready (do nothing).
    - If ALL deps are skipped (none completed), mark the task as skipped
      with error reason "dependency_skipped".

    Repeats until no more tasks are newly skipped (transitive closure).

    Returns count of newly skipped tasks.
    """
    db = await get_db()
    total_skipped = 0

    while True:
        # Collect all skipped task IDs for this mission
        cursor = await db.execute(
            "SELECT id FROM tasks WHERE mission_id = ? AND status = 'skipped'",
            (mission_id,)
        )
        skipped_ids = {row[0] for row in await cursor.fetchall()}

        if not skipped_ids:
            break

        # Get all pending tasks for this mission
        cursor = await db.execute(
            "SELECT id, depends_on FROM tasks WHERE mission_id = ? AND status = 'pending'",
            (mission_id,)
        )
        pending_tasks = await cursor.fetchall()

        newly_skipped = 0

        for row in pending_tasks:
            task_id = row[0]
            raw_deps = row[1]

            # Parse depends_on
            try:
                if raw_deps is None or raw_deps == "" or raw_deps == "null":
                    deps = []
                elif isinstance(raw_deps, str):
                    parsed = json.loads(raw_deps)
                    deps = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, int) else []
                elif isinstance(raw_deps, (list, tuple)):
                    deps = list(raw_deps)
                elif isinstance(raw_deps, int):
                    deps = [raw_deps]
                else:
                    deps = []
            except (json.JSONDecodeError, TypeError):
                deps = []

            if not deps:
                continue

            # Check if any dep is skipped
            has_skipped_dep = any(d in skipped_ids for d in deps)
            if not has_skipped_dep:
                continue

            # Check status of all deps
            placeholders = ",".join("?" * len(deps))
            dep_cursor = await db.execute(
                f"SELECT id, status FROM tasks WHERE id IN ({placeholders})",
                deps
            )
            dep_rows = await dep_cursor.fetchall()
            dep_statuses = {r[0]: r[1] for r in dep_rows}

            # All deps must be in {completed, skipped}
            all_resolved = all(
                dep_statuses.get(d) in ('completed', 'skipped')
                for d in deps
            )
            if not all_resolved:
                continue

            # If ALL are skipped (none completed), skip this task
            has_completed = any(
                dep_statuses.get(d) == 'completed' for d in deps
            )
            if not has_completed:
                await db.execute(
                    "UPDATE tasks SET status = 'skipped', error = 'dependency_skipped' WHERE id = ?",
                    (task_id,)
                )
                newly_skipped += 1

        if newly_skipped == 0:
            break

        await db.commit()
        total_skipped += newly_skipped

    if total_skipped > 0:
        await db.commit()
        logger.info(f"Skip propagation: {total_skipped} tasks skipped for mission #{mission_id}")

    return total_skipped


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

async def get_last_model_for_task(task_id: int) -> str | None:
    """Get the last model used for a task from conversation log."""
    db = await get_db()
    cursor = await db.execute(
        """SELECT model_used FROM conversations
           WHERE task_id = ? AND model_used IS NOT NULL
           ORDER BY id DESC LIMIT 1""",
        (task_id,),
    )
    row = await cursor.fetchone()
    return row[0] if row else None


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

async def store_memory(key, value, category="general", mission_id=None):
    db = await get_db()
    # Upsert in SQL
    existing = await db.execute(
        "SELECT id FROM memory WHERE key = ? AND mission_id IS ?",
        (key, mission_id)
    )
    row = await existing.fetchone()
    if row:
        await db.execute(
            "UPDATE memory SET value = ?, updated_at = ? WHERE id = ?",
            (value, db_now(), row[0])
        )
    else:
        await db.execute(
            "INSERT INTO memory (key, value, category, mission_id) VALUES (?, ?, ?, ?)",
            (key, value, category, mission_id)
        )
    await db.commit()

    # Also embed into vector store for semantic recall
    try:
        from src.memory.vector_store import embed_and_store
        await embed_and_store(
            text=f"{key}: {value}",
            metadata={
                "source": "memory_table",
                "key": key,
                "category": category,
                "mission_id": mission_id or "",
                "type": "memory",
            },
            collection="semantic",
            doc_id=f"mem:{hashlib.md5(f'{key}:{mission_id}'.encode()).hexdigest()[:16]}",
        )
    except Exception as exc:
        logger.debug(f"Vector embedding of memory failed (non-critical): {exc}")


async def recall_memory(category=None, mission_id=None, limit=20):
    db = await get_db()
    query = "SELECT * FROM memory WHERE 1=1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if mission_id:
        query += " AND mission_id = ?"
        params.append(mission_id)
    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)
    cursor = await db.execute(query, params)
    return [dict(row) for row in await cursor.fetchall()]


async def semantic_recall(query_text, category=None, mission_id=None, top_k=5):
    """Semantic search over stored memories using vector similarity."""
    try:
        from src.memory.vector_store import query as vquery

        conditions = [{"source": {"$eq": "memory_table"}}]
        if category:
            conditions.append({"category": {"$eq": category}})
        if mission_id:
            conditions.append({"mission_id": {"$eq": mission_id}})
        where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        results = await vquery(
            text=query_text,
            collection="semantic",
            top_k=top_k,
            where=where_filter,
        )

        # Convert to memory-table-like dicts
        memories = []
        for r in results:
            meta = r.get("metadata", {})
            memories.append({
                "key": meta.get("key", ""),
                "value": r.get("text", ""),
                "category": meta.get("category", ""),
                "mission_id": meta.get("mission_id", ""),
                "distance": r.get("distance", 1.0),
            })
        return memories
    except Exception as exc:
        logger.debug(f"semantic_recall failed: {exc}")
        return []

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
    """Create a new scheduled task.

    Deduplicates by title — if a task with the same title already exists,
    returns its ID without inserting a duplicate.  New tasks get ``next_run``
    computed from the cron expression so they don't fire immediately on the
    first scheduler check.
    """
    db = await get_db()

    # Deduplicate: skip if a task with this title already exists
    cursor = await db.execute(
        "SELECT id FROM scheduled_tasks WHERE title = ?", (title,)
    )
    existing = await cursor.fetchone()
    if existing:
        return existing[0]

    # Compute next_run so the task doesn't fire immediately (next_run=NULL bug).
    # Inline minimal cron parse to avoid circular import with orchestrator.
    next_run_str: str | None = None
    try:
        now = utc_now()
        parts = cron_expression.strip().split()
        if len(parts) == 5:
            minute, hour = parts[0], parts[1]
            m = int(minute) if minute != "*" else 0
            if minute != "*" and hour != "*" and "," not in hour and "/" not in hour:
                # Daily at H:M — covers most recurring tasks
                candidate = now.replace(
                    hour=int(hour), minute=m, second=0, microsecond=0
                )
                if candidate <= now:
                    candidate += timedelta(days=1)
                next_run_str = candidate.strftime("%Y-%m-%d %H:%M:%S")
            elif minute != "*" and "/" in hour:
                # e.g. */4 — every N hours at minute M
                interval = int(hour.split("/")[1])
                candidate = now.replace(minute=m, second=0, microsecond=0)
                if candidate <= now:
                    candidate += timedelta(hours=interval)
                next_run_str = candidate.strftime("%Y-%m-%d %H:%M:%S")
            elif minute != "*" and hour == "*":
                # Every hour at minute M
                candidate = now.replace(minute=m, second=0, microsecond=0)
                if candidate <= now:
                    candidate += timedelta(hours=1)
                next_run_str = candidate.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass  # Fall back to NULL — scheduler will compute on first fire

    cursor = await db.execute(
        """INSERT INTO scheduled_tasks
           (title, description, cron_expression, agent_type, tier, context, next_run)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (title, description, cron_expression, agent_type, tier,
         json.dumps(context or {}), next_run_str)
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


# ─── Todo Items ────────────────────────────────────────────────────────────

_TODO_COLUMNS = frozenset({
    "title", "description", "priority", "due_date",
    "status", "source", "completed_at",
    "suggestion", "suggestion_agent", "suggestion_at",
})


async def add_todo(
    title: str,
    description: str = "",
    priority: str = "normal",
    due_date: str | None = None,
    source: str = "explicit",
) -> int:
    """Create a new todo item. Returns the new row ID."""
    db = await get_db()
    cursor = await db.execute(
        """INSERT INTO todo_items (title, description, priority, due_date, source)
           VALUES (?, ?, ?, ?, ?)""",
        (title, description, priority, due_date, source),
    )
    await db.commit()
    return cursor.lastrowid


async def get_todos(status: str | None = None, limit: int = 50) -> list[dict]:
    """Return todo items, optionally filtered by status."""
    db = await get_db()
    if status:
        cursor = await db.execute(
            "SELECT * FROM todo_items WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM todo_items ORDER BY status ASC, created_at DESC LIMIT ?",
            (limit,),
        )
    return [dict(row) for row in await cursor.fetchall()]


async def get_todo(todo_id: int) -> dict | None:
    """Fetch a single todo item by ID."""
    db = await get_db()
    cursor = await db.execute("SELECT * FROM todo_items WHERE id = ?", (todo_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_todo(todo_id: int, **kwargs) -> None:
    """Update fields on a todo item."""
    _validate_columns(kwargs, _TODO_COLUMNS, "todo_items")
    db = await get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [todo_id]
    await db.execute(f"UPDATE todo_items SET {sets} WHERE id = ?", values)
    await db.commit()


async def toggle_todo(todo_id: int) -> str:
    """Toggle a todo between 'pending' and 'done'. Returns the new status."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT status FROM todo_items WHERE id = ?", (todo_id,)
    )
    row = await cursor.fetchone()
    if not row:
        return "pending"
    current = row["status"]
    if current == "done":
        new_status = "pending"
        await db.execute(
            """UPDATE todo_items SET status = 'pending', completed_at = NULL,
               suggestion = NULL, suggestion_agent = NULL, suggestion_at = NULL
               WHERE id = ?""",
            (todo_id,),
        )
    else:
        new_status = "done"
        await db.execute(
            "UPDATE todo_items SET status = 'done', completed_at = ? WHERE id = ?",
            (db_now(), todo_id),
        )
    await db.commit()
    return new_status


async def delete_todo(todo_id: int) -> None:
    """Delete a todo item."""
    db = await get_db()
    await db.execute("DELETE FROM todo_items WHERE id = ?", (todo_id,))
    await db.commit()


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

async def get_task_tree(mission_id: int) -> list[dict]:
    """Get all tasks for a mission, including parent-child relationships."""
    db = await get_db()
    cursor = await db.execute(
        """SELECT id, parent_task_id, title, status, agent_type,
                  priority, depends_on
           FROM tasks
           WHERE mission_id = ?
           ORDER BY id""",
        (mission_id,)
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

    This is the **single entry point** for recording model calls.  It persists
    stats to the ``model_stats`` DB table *and* updates the in-memory metric
    counters via ``track_model_call_metrics()`` so both systems stay in sync.
    """
    # Also update in-memory Prometheus-style counters
    try:
        from .metrics import track_model_call_metrics
        track_model_call_metrics(
            model=model,
            cost=cost,
            latency_ms=latency * 1000 if latency else 0.0,
        )
    except Exception:
        pass  # metrics are best-effort

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
    """Get budget info for a scope (e.g. 'daily', 'mission')."""
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
    today = utc_now().strftime("%Y-%m-%d")

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

    today = utc_now().strftime("%Y-%m-%d")
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

    today = utc_now().strftime("%Y-%m-%d")
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
    mission_id: int | None = None,
    task_id: int | None = None,
    agent_type: str | None = None,
) -> bool:
    """Acquire an advisory lock on a file. Returns True if acquired."""
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO file_locks (filepath, mission_id, task_id, agent_type) "
            "VALUES (?, ?, ?, ?)",
            (filepath, mission_id, task_id, agent_type),
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


async def release_mission_locks(mission_id: int) -> None:
    """Release all locks held by a specific mission."""
    db = await get_db()
    await db.execute("DELETE FROM file_locks WHERE mission_id = ?", (mission_id,))
    await db.commit()


async def get_file_lock(filepath: str) -> dict | None:
    """Check who holds the lock on a file."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM file_locks WHERE filepath = ?", (filepath,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_mission_locks(mission_id: int) -> list[dict]:
    """List all locks held by a mission."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM file_locks WHERE mission_id = ?", (mission_id,)
    )
    return [dict(r) for r in await cursor.fetchall()]


# ─── Phase 6: Workspace Snapshots ───────────────────────────────────────────

async def save_workspace_snapshot(
    mission_id: int,
    file_hashes: dict,
    task_id: int | None = None,
    branch_name: str | None = None,
    commit_sha: str | None = None,
) -> int:
    """Save a workspace state snapshot before a task runs."""
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO workspace_snapshots "
        "(mission_id, task_id, file_hashes, branch_name, commit_sha) "
        "VALUES (?, ?, ?, ?, ?)",
        (mission_id, task_id, json.dumps(file_hashes), branch_name, commit_sha),
    )
    await db.commit()
    return cursor.lastrowid


async def get_latest_snapshot(mission_id: int) -> dict | None:
    """Get the most recent snapshot for a mission."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM workspace_snapshots WHERE mission_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (mission_id,),
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


async def get_mission_total_cost(mission_id: int) -> float:
    """Sum up all conversation costs across all tasks in a mission."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT COALESCE(SUM(c.cost_estimate), 0) as total "
        "FROM conversations c "
        "JOIN tasks t ON c.task_id = t.id "
        "WHERE t.mission_id = ?",
        (mission_id,),
    )
    row = await cursor.fetchone()
    return float(row["total"]) if row else 0.0


async def check_task_budget(task_id: int, additional_cost: float = 0.0) -> dict:
    """
    Check if a task's cost budget would be exceeded.

    Returns: {"ok": bool, "reason": str, "spent": float, "limit": float|None}
    """
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


# ─── Approval Requests ────────────────────────────────────────────────────────

async def insert_approval_request(task_id: int, mission_id: int | None,
                                  title: str, details: str):
    """Persist an approval request to the DB."""
    db = await get_db()
    await db.execute(
        """INSERT OR REPLACE INTO approval_requests
           (task_id, mission_id, title, details, status, created_at)
           VALUES (?, ?, ?, ?, 'pending', ?)""",
        (task_id, mission_id, title, details, db_now()),
    )
    await db.commit()


async def update_approval_status(task_id: int, status: str):
    """Update an approval request status and set resolved_at."""
    db = await get_db()
    await db.execute(
        """UPDATE approval_requests
           SET status = ?, resolved_at = ?
           WHERE task_id = ?""",
        (status, db_now(), task_id),
    )
    await db.commit()


async def get_pending_approvals() -> list[dict]:
    """Return all pending approval requests."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM approval_requests WHERE status = 'pending' "
        "ORDER BY created_at"
    )
    return [dict(row) for row in await cursor.fetchall()]


# ── Workflow checkpoints ──────────────────────────────────────────────────


async def upsert_workflow_checkpoint(
    mission_id: int,
    workflow_name: str,
    current_phase: str = None,
    completed_phases: list = None,
    failed_step_id: str = None,
    metadata: dict = None,
) -> None:
    """Create or update a workflow checkpoint for the given mission."""
    db = await get_db()
    await db.execute(
        """INSERT OR REPLACE INTO workflow_checkpoints
           (mission_id, workflow_name, current_phase, completed_phases,
            failed_step_id, checkpoint_at, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            mission_id,
            workflow_name,
            current_phase,
            json.dumps(completed_phases or []),
            failed_step_id,
            db_now(),
            json.dumps(metadata or {}),
        ),
    )
    await db.commit()


async def get_workflow_checkpoint(mission_id: int) -> dict | None:
    """Get the workflow checkpoint for a mission, or None if not found."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM workflow_checkpoints WHERE mission_id = ?",
        (mission_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None

    result = dict(row)
    # JSON-decode fields
    for field in ("completed_phases", "metadata"):
        raw = result.get(field, "")
        try:
            result[field] = json.loads(raw) if raw else ([] if field == "completed_phases" else {})
        except (json.JSONDecodeError, TypeError):
            result[field] = [] if field == "completed_phases" else {}
    return result


async def update_model_stats(
    model: str,
    agent_type: str,
    success: bool,
    cost: float = 0.0,
    latency_ms: float = 0.0,
    grade: float = 0.0,
) -> None:
    """Record model performance stats for health monitoring."""
    try:
        db = await get_db()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS model_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                success INTEGER NOT NULL,
                cost REAL DEFAULT 0,
                latency_ms REAL DEFAULT 0,
                grade REAL DEFAULT 0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute(
            "INSERT INTO model_stats (model, agent_type, success, cost, latency_ms, grade) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (model, agent_type, int(success), cost, latency_ms, grade),
        )
        await db.commit()
    except Exception:
        pass  # best-effort


# ─── Web Source Quality Tracking ───────────────────────────────────────────


async def record_source_quality(
    domain: str, success: bool, blocked: bool = False, relevance: float = 0.0
) -> None:
    """Upsert domain quality record, incrementing counts and updating timestamps."""
    try:
        db = await get_db()
        now = db_now()

        # Try to insert first, then update on conflict
        if blocked:
            await db.execute("""
                INSERT INTO web_source_quality (domain, block_count, last_block, updated_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    block_count = block_count + 1,
                    last_block = excluded.last_block,
                    updated_at = excluded.updated_at
            """, (domain, now, now))
        elif success:
            # Update avg_relevance with incremental mean
            await db.execute("""
                INSERT INTO web_source_quality (domain, success_count, avg_relevance, last_success, updated_at)
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    success_count = success_count + 1,
                    avg_relevance = (avg_relevance * success_count + ?) / (success_count + 1),
                    last_success = excluded.last_success,
                    updated_at = excluded.updated_at
            """, (domain, relevance, now, now, relevance))
        else:
            await db.execute("""
                INSERT INTO web_source_quality (domain, fail_count, last_failure, updated_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    fail_count = fail_count + 1,
                    last_failure = excluded.last_failure,
                    updated_at = excluded.updated_at
            """, (domain, now, now))

        await db.commit()
    except Exception as e:
        logger.debug("record_source_quality failed: %s", e)


async def get_source_quality(domains: list[str]) -> dict[str, dict]:
    """Return quality info for a list of domains.

    Returns {domain: {success_count, fail_count, block_count, avg_relevance, ...}}.
    """
    if not domains:
        return {}
    try:
        db = await get_db()
        placeholders = ",".join("?" for _ in domains)
        cursor = await db.execute(
            f"SELECT * FROM web_source_quality WHERE domain IN ({placeholders})",
            domains,
        )
        rows = await cursor.fetchall()
        return {row["domain"]: dict(row) for row in rows}
    except Exception as e:
        logger.debug("get_source_quality failed: %s", e)
        return {}


# --- Free API Registry Operations ---

async def upsert_free_api(api_data: dict) -> bool:
    """Insert or update a free API in the registry. Returns True if newly inserted."""
    try:
        db = await get_db()
        cur = await db.execute(
            "SELECT 1 FROM free_api_registry WHERE name = ?", (api_data["name"],)
        )
        is_new = (await cur.fetchone()) is None
        await db.execute(
            """INSERT INTO free_api_registry
               (name, category, base_url, auth_type, env_var, rate_limit,
                description, example_endpoint, source, verified, last_checked)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   category = excluded.category,
                   base_url = excluded.base_url,
                   auth_type = excluded.auth_type,
                   env_var = excluded.env_var,
                   rate_limit = excluded.rate_limit,
                   description = excluded.description,
                   example_endpoint = excluded.example_endpoint,
                   source = excluded.source,
                   verified = excluded.verified,
                   last_checked = excluded.last_checked
            """,
            (
                api_data["name"],
                api_data.get("category", "misc"),
                api_data["base_url"],
                api_data.get("auth_type", "none"),
                api_data.get("env_var"),
                api_data.get("rate_limit", "unknown"),
                api_data.get("description", ""),
                api_data.get("example_endpoint", ""),
                api_data.get("source", "static"),
                api_data.get("verified", 0),
                api_data.get("last_checked"),
            ),
        )
        await db.commit()
        return is_new
    except Exception as e:
        logger.debug("upsert_free_api failed: %s", e)
        return False


async def get_all_free_apis() -> list[dict]:
    """Fetch all APIs from the registry."""
    try:
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM free_api_registry ORDER BY category, name"
        )
        return [dict(row) for row in await cursor.fetchall()]
    except Exception as e:
        logger.debug("get_all_free_apis failed: %s", e)
        return []


async def get_free_apis_by_category(category: str) -> list[dict]:
    """Fetch APIs matching a category (case-insensitive)."""
    try:
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM free_api_registry WHERE LOWER(category) = LOWER(?) ORDER BY name",
            (category,),
        )
        return [dict(row) for row in await cursor.fetchall()]
    except Exception as e:
        logger.debug("get_free_apis_by_category failed: %s", e)
        return []


# ─── User Preferences ──────────────────────────────────────────────────────

async def get_user_pref(key: str, default: str = "") -> str:
    """Get a user preference value."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT value FROM user_preferences WHERE key = ?", (key,)
    )
    row = await cursor.fetchone()
    return row["value"] if row else default


async def set_user_pref(key: str, value: str) -> None:
    """Set a user preference value."""
    db = await get_db()
    await db.execute(
        "INSERT INTO user_preferences (key, value, updated_at) VALUES (?, ?, strftime('%Y-%m-%d %H:%M:%S', 'now')) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value),
    )
    await db.commit()


async def get_all_user_prefs() -> dict:
    """Get all user preferences as a dict."""
    db = await get_db()
    cursor = await db.execute("SELECT key, value FROM user_preferences")
    rows = await cursor.fetchall()
    return {row["key"]: row["value"] for row in rows}


# ─── Skill Injection A/B Metrics ────────────────────────────────────────────


async def record_skill_metric(task_id: int, skill_name: str, succeeded: bool,
                               iterations: int = 0, agent_type: str = "",
                               duration: float = 0.0) -> None:
    """Record a skill injection outcome for A/B analysis."""
    db = await get_db()
    await db.execute(
        """INSERT INTO skill_metrics (task_id, skill_name, task_succeeded,
           iterations_used, agent_type, task_duration_seconds)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (task_id, skill_name, 1 if succeeded else 0, iterations, agent_type, duration),
    )
    await db.commit()


async def record_no_skill_metric(task_id: int, succeeded: bool,
                                  iterations: int = 0, agent_type: str = "",
                                  duration: float = 0.0) -> None:
    """Record a task that ran WITHOUT skill injection (baseline)."""
    db = await get_db()
    await db.execute(
        """INSERT INTO skill_metrics (task_id, skill_name, injected, task_succeeded,
           iterations_used, agent_type, task_duration_seconds)
           VALUES (?, '_baseline_', 0, ?, ?, ?, ?)""",
        (task_id, 1 if succeeded else 0, iterations, agent_type, duration),
    )
    await db.commit()


async def get_skill_metrics_summary() -> dict:
    """Get A/B comparison: tasks with skills vs without."""
    db = await get_db()

    # Overall: with skills vs baseline
    cursor = await db.execute("""
        SELECT
            injected,
            COUNT(*) as total,
            SUM(task_succeeded) as successes,
            AVG(iterations_used) as avg_iterations,
            AVG(task_duration_seconds) as avg_duration
        FROM skill_metrics
        GROUP BY injected
    """)
    overall = {}
    for row in await cursor.fetchall():
        key = "with_skills" if row["injected"] else "baseline"
        total = row["total"]
        successes = row["successes"] or 0
        overall[key] = {
            "total": total,
            "successes": successes,
            "success_rate": round(successes / total * 100, 1) if total > 0 else 0,
            "avg_iterations": round(row["avg_iterations"] or 0, 1),
            "avg_duration": round(row["avg_duration"] or 0, 1),
        }

    # Per-skill breakdown
    cursor2 = await db.execute("""
        SELECT
            skill_name,
            COUNT(*) as total,
            SUM(task_succeeded) as successes,
            AVG(iterations_used) as avg_iterations
        FROM skill_metrics
        WHERE injected = 1
        GROUP BY skill_name
        ORDER BY total DESC
        LIMIT 20
    """)
    per_skill = []
    for row in await cursor2.fetchall():
        total = row["total"]
        successes = row["successes"] or 0
        per_skill.append({
            "name": row["skill_name"],
            "total": total,
            "successes": successes,
            "success_rate": round(successes / total * 100, 1) if total > 0 else 0,
            "avg_iterations": round(row["avg_iterations"] or 0, 1),
        })

    return {"overall": overall, "per_skill": per_skill}


# ── Skills v2 helpers ──

async def upsert_skill(
    name: str,
    description: str,
    skill_type: str = "auto",
    strategies: list | None = None,
) -> int:
    """Insert or update a skill. Returns the skill row id."""
    db = await get_db()
    strategies_json = json.dumps(strategies or [])
    cursor = await db.execute(
        """INSERT INTO skills (name, description, skill_type, strategies)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
               description = excluded.description,
               skill_type = excluded.skill_type,
               strategies = excluded.strategies,
               updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now')""",
        (name, description, skill_type, strategies_json),
    )
    await db.commit()
    return cursor.lastrowid


async def get_all_skills() -> list[dict]:
    """Return all skills as a list of dicts."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, name, description, skill_type, strategies, injection_count, injection_success, created_at, updated_at FROM skills"
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_skill_by_name(name: str) -> dict | None:
    """Return a single skill by name, or None if not found."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, name, description, skill_type, strategies, injection_count, injection_success, created_at, updated_at FROM skills WHERE name = ?",
        (name,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def increment_skill_injection(name: str) -> None:
    """Increment injection_count for a skill."""
    db = await get_db()
    await db.execute(
        "UPDATE skills SET injection_count = injection_count + 1, updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now') WHERE name = ?",
        (name,),
    )
    await db.commit()


async def increment_skill_success(name: str) -> None:
    """Increment injection_success for a skill."""
    db = await get_db()
    await db.execute(
        "UPDATE skills SET injection_success = injection_success + 1, updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now') WHERE name = ?",
        (name,),
    )
    await db.commit()


async def prune_old_conversations(max_age_days: int = 30) -> int:
    """Delete conversations older than max_age_days. Returns count deleted."""
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM conversations WHERE created_at < datetime('now', ? || ' days')",
        (f"-{max_age_days}",),
    )
    await db.commit()
    return cursor.rowcount


# ── Smart Resource Integration queries ──

async def upsert_api_keyword(api_name: str, keyword: str, source: str = "description"):
    db = await get_db()
    await db.execute(
        "INSERT OR IGNORE INTO api_keywords (api_name, keyword, source) VALUES (?, ?, ?)",
        (api_name, keyword, source),
    )
    await db.commit()


async def bulk_upsert_api_keywords(rows: list[tuple[str, str, str]]):
    """rows = [(api_name, keyword, source), ...]"""
    db = await get_db()
    await db.executemany(
        "INSERT OR IGNORE INTO api_keywords (api_name, keyword, source) VALUES (?, ?, ?)",
        rows,
    )
    await db.commit()


async def find_apis_by_keywords(keywords: list[str], limit: int = 5) -> list[dict]:
    """Find APIs with the most keyword overlap. Returns [{api_name, match_count}, ...]."""
    if not keywords:
        return []
    db = await get_db()
    placeholders = ",".join("?" for _ in keywords)
    cur = await db.execute(
        f"""SELECT api_name, COUNT(*) as match_count
            FROM api_keywords
            WHERE keyword IN ({placeholders})
            GROUP BY api_name
            ORDER BY match_count DESC
            LIMIT ?""",
        (*keywords, limit),
    )
    rows = await cur.fetchall()
    return [{"api_name": r[0], "match_count": r[1]} for r in rows]


async def get_api_category_patterns() -> dict[str, str]:
    """Return {category: pattern} for Turkish localization patterns."""
    db = await get_db()
    cur = await db.execute("SELECT category, pattern FROM api_category_patterns")
    rows = await cur.fetchall()
    return {r[0]: r[1] for r in rows}


async def upsert_category_pattern(category: str, pattern: str):
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO api_category_patterns (category, pattern) VALUES (?, ?)",
        (category, pattern),
    )
    await db.commit()


async def log_smart_search(query: str, layer: int, source: str | None, success: bool, response_ms: int):
    db = await get_db()
    await db.execute(
        """INSERT INTO smart_search_log (query, layer, source, success, response_ms)
           VALUES (?, ?, ?, ?, ?)""",
        (query, layer, source, 1 if success else 0, response_ms),
    )
    await db.commit()


async def record_api_call(api_name: str, success: bool):
    """Update api_reliability counters and auto-demote if needed."""
    db = await get_db()
    now = "strftime('%Y-%m-%d %H:%M:%S', 'now')"
    if success:
        await db.execute(
            f"""INSERT INTO api_reliability (api_name, success_count, last_success, consecutive_failures)
                VALUES (?, 1, {now}, 0)
                ON CONFLICT(api_name) DO UPDATE SET
                    success_count = success_count + 1,
                    consecutive_failures = 0,
                    last_success = {now}""",
            (api_name,),
        )
    else:
        await db.execute(
            f"""INSERT INTO api_reliability (api_name, failure_count, last_failure, consecutive_failures)
                VALUES (?, 1, {now}, 1)
                ON CONFLICT(api_name) DO UPDATE SET
                    failure_count = failure_count + 1,
                    consecutive_failures = consecutive_failures + 1,
                    last_failure = {now}""",
            (api_name,),
        )
    # Auto-demote check
    cur = await db.execute(
        "SELECT success_count, failure_count, consecutive_failures FROM api_reliability WHERE api_name = ?",
        (api_name,),
    )
    row = await cur.fetchone()
    if row:
        total = row[0] + row[1]
        rate = row[0] / max(total, 1)
        consec = row[2]
        if total >= 20 and rate < 0.10:
            status = "suspended"
        elif total >= 15 and rate < 0.25:
            status = "demoted"
        elif total >= 15 and rate < 0.50:
            status = "warning"
        elif consec >= 3:
            status = "warning"
        else:
            status = "active"
        await db.execute(
            "UPDATE api_reliability SET status = ? WHERE api_name = ?",
            (status, api_name),
        )
    await db.commit()


async def get_api_reliability(api_name: str) -> dict | None:
    db = await get_db()
    cur = await db.execute(
        "SELECT api_name, success_count, failure_count, status, consecutive_failures FROM api_reliability WHERE api_name = ?",
        (api_name,),
    )
    row = await cur.fetchone()
    if not row:
        return None
    return {"api_name": row[0], "success_count": row[1], "failure_count": row[2], "status": row[3], "consecutive_failures": row[4]}


async def get_api_reliability_all() -> list[dict]:
    db = await get_db()
    cur = await db.execute(
        "SELECT api_name, success_count, failure_count, status, last_success, last_failure, consecutive_failures FROM api_reliability ORDER BY (success_count + failure_count) DESC"
    )
    rows = await cur.fetchall()
    return [
        {"api_name": r[0], "success_count": r[1], "failure_count": r[2], "status": r[3], "last_success": r[4], "last_failure": r[5], "consecutive_failures": r[6]}
        for r in rows
    ]


async def get_smart_search_stats(days: int = 7) -> dict:
    """Aggregate smart_search_log for observability menu."""
    db = await get_db()
    cutoff = f"datetime('now', '-{days} days')"

    # Layer breakdown
    cur = await db.execute(
        f"SELECT layer, COUNT(*), SUM(success) FROM smart_search_log WHERE timestamp > {cutoff} GROUP BY layer"
    )
    layers = {r[0]: {"count": r[1], "success": r[2]} for r in await cur.fetchall()}

    # Top sources
    cur = await db.execute(
        f"""SELECT source, COUNT(*), SUM(success) FROM smart_search_log
            WHERE timestamp > {cutoff} AND source IS NOT NULL
            GROUP BY source ORDER BY COUNT(*) DESC LIMIT 10"""
    )
    top_sources = [{"source": r[0], "count": r[1], "success": r[2]} for r in await cur.fetchall()]

    # Today count
    cur = await db.execute(
        "SELECT COUNT(*) FROM smart_search_log WHERE date(timestamp) = date('now')"
    )
    today = (await cur.fetchone())[0]

    return {"layers": layers, "top_sources": top_sources, "today": today}


async def unsuspend_api(api_name: str):
    db = await get_db()
    await db.execute(
        "UPDATE api_reliability SET status = 'active', success_count = 0, failure_count = 0, consecutive_failures = 0 WHERE api_name = ?",
        (api_name,),
    )
    await db.commit()
