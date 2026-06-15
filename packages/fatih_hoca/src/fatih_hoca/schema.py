"""Model-registry schema, owned by fatih_hoca. SINGLE source of truth for the
registry tables' DDL. Both the async engine registration (create_registry_schema)
and the sync registry_store._ensure_schema (a later task) execute
REGISTRY_DDL/REGISTRY_ALTERS, so there is exactly one place the schema is defined.

The DDL is copied byte-for-byte from packages/db/src/dabidabi/__init__.py
(init_db) so the registration path produces an identical schema to the engine's
own inline DDL. ``CREATE ... IF NOT EXISTS`` makes the overlap safe while the
tables still live in core.db this slice (no file-split yet).
"""
import dabidabi

REGISTRY_DDL = [
    # ── model_stats — Schema A (canonical aggregate). dabidabi init_db ──
    """
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
    """,
    # ── model_pick_log — CREATE (added cols covered by REGISTRY_ALTERS) ──
    """
        CREATE TABLE IF NOT EXISTS model_pick_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            task_name TEXT NOT NULL,
            agent_type TEXT,
            difficulty INTEGER,
            call_category TEXT,
            picked_model TEXT NOT NULL,
            picked_score REAL NOT NULL,
            picked_reasons TEXT,
            candidates_json TEXT NOT NULL,
            failures_json TEXT,
            snapshot_summary TEXT,
            pool TEXT,
            urgency REAL,
            success INTEGER,
            error_category TEXT,
            provider TEXT
        )
    """,
    # model_pick_log indexes
    "CREATE INDEX IF NOT EXISTS idx_pick_log_provider ON model_pick_log(provider)",
    "CREATE INDEX IF NOT EXISTS idx_pick_log_task ON model_pick_log(task_name, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_pick_log_model ON model_pick_log(picked_model, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_pick_log_task_id ON model_pick_log(task_id)",
    # ── providers ──
    """
        CREATE TABLE IF NOT EXISTS providers (
            name        TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'active',
            cause       TEXT,
            marked_at   TIMESTAMP,
            revived_at  TIMESTAMP,
            key_hash    TEXT
        )
    """,
    # ── models ──
    """
        CREATE TABLE IF NOT EXISTS models (
            litellm_name    TEXT PRIMARY KEY,
            provider        TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active',
            cause           TEXT,
            marked_at       TIMESTAMP,
            revived_at      TIMESTAMP,
            expires_at      TIMESTAMP,
            source          TEXT,
            first_seen_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    # ── registry_events — FULL shape: base 8 cols + migrated 4 cols inline ──
    # so fresh DBs get the migrated columns without ALTER. REGISTRY_ALTERS
    # cover existing DBs whose CREATE predates the migrated columns.
    """
        CREATE TABLE IF NOT EXISTS registry_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scope           TEXT NOT NULL,
            target          TEXT NOT NULL,
            event           TEXT NOT NULL,
            cause           TEXT,
            actor           TEXT,
            payload_json    TEXT,
            mission_id      INTEGER,
            task_id         INTEGER,
            verb            TEXT,
            reversibility   TEXT
        )
    """,
    # registry_events + models indexes
    "CREATE INDEX IF NOT EXISTS idx_registry_events_target_ts "
    "ON registry_events(target, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_models_status "
    "ON models(status, provider)",
]

REGISTRY_ALTERS = [
    # ALTERs for EXISTING DBs whose CREATE predates added columns. Each attempted
    # individually; "duplicate column name" is the expected/ignored error.
    # ── registry_events migrated cols (Z10 T1C action-scope audit) ──
    "ALTER TABLE registry_events ADD COLUMN mission_id INTEGER",
    "ALTER TABLE registry_events ADD COLUMN task_id INTEGER",
    "ALTER TABLE registry_events ADD COLUMN verb TEXT",
    "ALTER TABLE registry_events ADD COLUMN reversibility TEXT",
    # ── model_pick_log added cols (pre-Phase-2c / pre-Task-5 / pre-Task-15) ──
    "ALTER TABLE model_pick_log ADD COLUMN pool TEXT",
    "ALTER TABLE model_pick_log ADD COLUMN urgency REAL",
    "ALTER TABLE model_pick_log ADD COLUMN success INTEGER",
    "ALTER TABLE model_pick_log ADD COLUMN error_category TEXT",
    "ALTER TABLE model_pick_log ADD COLUMN provider TEXT",
    "ALTER TABLE model_pick_log ADD COLUMN outcome TEXT",
    "ALTER TABLE model_pick_log ADD COLUMN reinforce REAL",
    "ALTER TABLE model_pick_log ADD COLUMN task_id INTEGER",
]


def _is_dup_col(err: Exception) -> bool:
    return "duplicate column name" in str(err).lower()


async def create_registry_schema(db) -> None:
    """Async executor (engine registration path)."""
    for sql in REGISTRY_DDL:
        await db.execute(sql)
    for sql in REGISTRY_ALTERS:
        try:
            await db.execute(sql)
        except Exception as e:
            if not _is_dup_col(e):
                raise


def ensure_registry_schema_sync(conn) -> None:
    """Sync executor (registry_store path). conn = sqlite3.Connection."""
    for sql in REGISTRY_DDL:
        conn.execute(sql)
    for sql in REGISTRY_ALTERS:
        try:
            conn.execute(sql)
        except Exception as e:
            if not _is_dup_col(e):
                raise


dabidabi.register_schema("fatih_hoca_registry", create_registry_schema)
