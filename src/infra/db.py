# db.py
import asyncio
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
# Path the cached connection was opened against. When the module-level
# DB_PATH gets monkeypatched (test setup) without manually clearing the
# cache, get_db() would silently return the stale connection pointing at
# whatever DB the previous test or runtime opened — leaking writes into
# production. Tracking the path lets get_db / init_db detect the
# mismatch and rebuild the connection. Production 2026-05-04: stray test
# row #10325 ("llm_call:coder:000001-abcdef" / "code this") landed in
# kutai.db because test_beckman_writes_selected_model_to_task changed
# DB_PATH but get_db handed back the cached production connection.
_db_connection_path: str | None = None
# Global lock guarding explicit BEGIN/COMMIT regions. aiosqlite serialises
# individual SQL statements via its worker thread, but transactions span
# multiple awaits — if coroutine A is between its BEGIN and COMMIT, any
# concurrent BEGIN from B raises sqlite3.OperationalError "cannot start
# a transaction within a transaction" (single connection, single open
# tx). Production triage 2026-05-01: burst of on_task_finished calls
# fired add_task / add_subtasks_atomically / insert_tasks_atomically
# concurrently; one batch's BEGIN landed while another's was still open.
# All callers must acquire this lock for the duration of an explicit
# BEGIN ... COMMIT block.
_tx_lock: asyncio.Lock = asyncio.Lock()

# ─── Z10 T3C: per-mission tx-lock shard ──────────────────────────────────────
# Cross-mission interference (zone doc 10): the single global ``_tx_lock``
# above means Mission A's slow INSERT blocks Mission B's add_task until the
# 60s WAL busy_timeout fires → ``OperationalError "database is locked"``. The
# fix: shard the lock per mission_id so writes into mission-scoped tables
# (tasks / task_events / mission_events / artifact_provenance /
# mission_pacing_snapshots / mission_tradeoff_prompts / mission_budget_alerts
# / action_confirmations / cost_budgets WHERE scope='mission') do not contend
# across missions.
#
# Lock-shard table assignments
# ----------------------------
# mission-scoped (use ``_get_tx_lock(mission_id)``):
#   - tasks, task_events
#   - mission_events
#   - artifact_provenance
#   - mission_pacing_snapshots
#   - mission_tradeoff_prompts
#   - mission_budget_alerts
#   - action_confirmations
#   - cost_budgets rows WHERE scope='mission'
#   - mission_green_tags  (T3C: green-tag ledger)
#
# global / cross-mission (use ``_get_tx_lock(None)``):
#   - missions row writes (insert/update on the row itself)
#   - models, model_pick_log, model_call_tokens (cross-mission view)
#   - registry_events
#   - schema_migrations
#   - cost_budgets rows WHERE scope IN ('vendor:*', 'global')
#
# Combined writes (mission-scoped + global in one tx) MUST acquire both locks,
# global first then mission, via ``_get_combined_lock(mission_id)``.
_mission_tx_locks: dict["int | None", asyncio.Lock] = {}
_mission_tx_locks_meta_lock: asyncio.Lock = asyncio.Lock()


def _get_tx_lock(mission_id: "int | None") -> asyncio.Lock:
    """Return (or create) the tx lock for ``mission_id``.

    ``mission_id=None`` → global lock for cross-mission tables. A per-mission
    lock is created lazily on first request. Locks are never evicted — the
    cardinality is bounded by total missions (~hundreds) and each Lock is
    ~64 bytes. Eviction would race with concurrent acquirers.
    """
    # Fast path: no lock-creation race for already-cached entries.
    lock = _mission_tx_locks.get(mission_id)
    if lock is not None:
        return lock
    # Slow path: race-safe create. We can't use asyncio.Lock as a context
    # here without awaiting, so use a simple double-checked pattern with a
    # synchronous fallback — the only racer is a parallel _get_tx_lock for
    # the same mission_id, which is rare and ``setdefault`` on a dict is
    # atomic at the GIL level for non-async dict mutation. The
    # ``_mission_tx_locks_meta_lock`` is reserved for future invariants
    # (e.g. periodic eviction) and is intentionally unused on the hot path.
    fresh = asyncio.Lock()
    return _mission_tx_locks.setdefault(mission_id, fresh)


class _CombinedLock:
    """Async context manager that acquires (global, mission) locks in order.

    Order is fixed — global first, then mission — so two coroutines that
    each want both locks for different missions cannot deadlock. The
    global lock is always the same singleton, so the standard
    "acquire-in-canonical-order" rule is satisfied.

    Skips the mission lock when ``mission_id is None`` (equivalent to a
    pure-global tx).
    """

    def __init__(self, mission_id: "int | None"):
        self._mission_id = mission_id
        self._global = _get_tx_lock(None)
        self._mission = (
            _get_tx_lock(mission_id) if mission_id is not None else None
        )
        # Same Lock object would deadlock on double-acquire — collapse.
        if self._mission is self._global:
            self._mission = None

    async def __aenter__(self):
        await self._global.acquire()
        if self._mission is not None:
            try:
                await self._mission.acquire()
            except BaseException:
                self._global.release()
                raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._mission is not None:
            try:
                self._mission.release()
            except RuntimeError:
                pass
        try:
            self._global.release()
        except RuntimeError:
            pass


def _get_combined_lock(mission_id: "int | None") -> _CombinedLock:
    """Acquire-both helper for callers writing mission-scoped AND global rows.

    Deadlock-safe order: global FIRST, then mission. See ``_CombinedLock``.
    """
    return _CombinedLock(mission_id)



async def get_db() -> aiosqlite.Connection:
    """Return the shared database connection, creating it on first call.

    Uses ``isolation_level=None`` (autocommit). Default deferred mode
    auto-began an implicit transaction on every DML, which collided with
    explicit ``BEGIN IMMEDIATE`` blocks in add_task / add_subtasks /
    insert_tasks (raising "cannot start a transaction within a
    transaction"). With autocommit, each statement is its own tx unless
    wrapped in an explicit BEGIN/COMMIT region — those callers already
    use ``_tx_lock`` to serialize.

    Atomicity note: any caller that issues multiple DML statements
    expecting them to commit as one tx must wrap them in BEGIN/COMMIT
    AND acquire ``_tx_lock``. Plain ``await db.commit()`` after a series
    of inserts is now a no-op; the inserts auto-committed individually.
    """
    global _db_connection, _db_connection_path
    # Detect DB_PATH override (typically a test that monkeypatched the
    # module-level DB_PATH after a prior caller already opened the
    # singleton). Close the stale connection so the next aiosqlite.connect
    # opens against the new path. Without this, writes silently leak into
    # whatever DB the cached connection happens to point at.
    if _db_connection is not None and _db_connection_path != DB_PATH:
        try:
            await _db_connection.close()
        except Exception:
            pass
        _db_connection = None
        _db_connection_path = None
    if _db_connection is None:
        _db_connection = await aiosqlite.connect(DB_PATH, isolation_level=None)
        _db_connection.row_factory = aiosqlite.Row
        _db_connection_path = DB_PATH
        # Enable WAL for concurrent reads + better write performance
        await _db_connection.execute("PRAGMA journal_mode=WAL")
        await _db_connection.execute("PRAGMA synchronous=NORMAL")
        await _db_connection.execute("PRAGMA busy_timeout=60000")
    return _db_connection


async def close_db(checkpoint: bool = True) -> None:
    """Close the shared connection (call on shutdown).

    Args:
        checkpoint: If True, run WAL checkpoint before closing (for clean
                    stop). If False, skip it (for restarts — next instance
                    will use WAL mode anyway).
    """
    global _db_connection, _db_connection_path
    if _db_connection is not None:
        if checkpoint:
            try:
                await _db_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
        await _db_connection.close()
        _db_connection = None
        _db_connection_path = None
        logger.info("Database connection closed", checkpoint=checkpoint)


def _apply_pragmas_sync(conn) -> None:
    """Apply WAL + 60s busy_timeout + synchronous=NORMAL to a sync sqlite3 conn."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")


async def _apply_pragmas(db: aiosqlite.Connection) -> None:
    """Apply WAL + 60s busy_timeout + synchronous=NORMAL to an aiosqlite conn."""
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA busy_timeout=60000")


# Holder tracker: maps label → list of (entry_time, ctx_id) for active
# connect_aux blocks. When a slow region times out, dump current holders
# so we can see who's blocking. Module-level so we can read from any
# reporter coroutine.
import time as _time
_aux_active: dict[int, tuple[str, float]] = {}
_aux_active_seq = 0


def _aux_active_summary() -> str:
    """Return a one-line summary of currently-active aux regions.

    Each entry: ``label@<held_seconds>s``. Sorted longest-held first.
    Appends singleton state (``singleton.tx=1`` when an explicit BEGIN
    is open on the shared conn). Production triage: when slow regions
    fire with ``other_active=[(none)]``, the holder was either the
    singleton or a sibling process (nerd_herd sidecar). Without this
    flag the operator can't tell which.
    """
    now = _time.monotonic()
    entries = [
        (lab, now - t0) for (lab, t0) in _aux_active.values()
    ]
    entries.sort(key=lambda x: -x[1])
    parts = [f"{lab}@{held:.1f}s" for lab, held in entries[:8]]
    if _db_connection is not None:
        try:
            in_tx = bool(_db_connection._conn.in_transaction)
            if in_tx:
                parts.append("singleton.tx=1")
        except Exception:
            pass
    return ", ".join(parts) or "(none)"


def connect_aux(db_path, _label: str | None = None):
    """Async context manager wrapping aiosqlite.connect with WAL pragmas applied.

    Use for any DB write/read OUTSIDE the get_db() singleton. Ensures every
    auxiliary connection respects the 60s busy_timeout and WAL journal mode
    so it cannot fall back to default 0ms (immediate "database is locked").

    Uses ``isolation_level=None`` to match the singleton — auto-commit per
    statement, no implicit transactions. Wrap multi-statement work in
    explicit BEGIN/COMMIT if you need atomicity.

    Pass ``_label`` to tag this conn in slow-region telemetry. When the
    block holds the conn open for > AUX_HOLD_WARN_SEC (1s), a WARNING is
    emitted with the label AND a snapshot of currently-active aux
    regions so we can identify the writer-slot holder vs waiters.

    Usage:
        async with connect_aux(DB_PATH, _label="add_task") as db:
            await db.execute(...)
    """
    AUX_HOLD_WARN_SEC = 1.0  # any aux region > 1s is suspicious
    label = _label or "?"

    class _Ctx:
        def __init__(self, p):
            self._p = p
            self._db = None
            self._t0 = 0.0
            self._tok = -1

        async def __aenter__(self):
            global _aux_active_seq
            self._t0 = _time.monotonic()
            _aux_active_seq += 1
            self._tok = _aux_active_seq
            _aux_active[self._tok] = (label, self._t0)
            self._db = await aiosqlite.connect(self._p, isolation_level=None)
            await _apply_pragmas(self._db)
            return self._db

        async def __aexit__(self, exc_type, exc, tb):
            try:
                await self._db.close()
            finally:
                self._db = None
            held = _time.monotonic() - self._t0
            _aux_active.pop(self._tok, None)
            if held >= AUX_HOLD_WARN_SEC:
                # Snapshot of OTHER active regions while we logged.
                # Helps distinguish "I'm the holder" from "I waited on
                # someone else who is still active."
                other_holders = _aux_active_summary()
                logger.warning(
                    "connect_aux slow region: label=%s held=%.1fs "
                    "other_active=[%s]",
                    label, held, other_holders,
                )

    return _Ctx(db_path)


def connect_aux_sync(db_path, timeout: float = 60.0):
    """Sync analogue of connect_aux for sqlite3 (not aiosqlite) callsites.

    Returns a sqlite3.Connection with WAL pragmas applied + isolation_level=None
    for autocommit semantics. Caller is responsible for calling .close().
    """
    import sqlite3
    conn = sqlite3.connect(db_path, timeout=timeout, isolation_level=None)
    _apply_pragmas_sync(conn)
    return conn


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

    # Z1 Tier 3 design-tokens migration: gate the new phase 5.0 / 5.0a steps
    # so existing in-flight missions are not retroactively required to emit
    # taste_emphasis.json + design_tokens.json. New missions start at 0
    # (default); legacy rows get backfilled to 1.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN legacy_pre_design_tokens INTEGER DEFAULT 0"
        )
        # Column was just added → every existing row predates the cluster.
        await db.execute("UPDATE missions SET legacy_pre_design_tokens = 1")
    except Exception:
        pass  # Column already exists; skip backfill on subsequent boots.
    # Z1 T3B migration: gate the new phase-5 user-flow steps for older missions
    # by tagging existing rows as legacy_pre_user_flow=1; new missions default
    # to 0 so they receive 5.0b/5.0c/5.0d.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN legacy_pre_user_flow INTEGER DEFAULT 0"
        )
        # Backfill: every mission that existed before this column was added is
        # legacy. Rows inserted after this point use the column default of 0.
        await db.execute(
            "UPDATE missions SET legacy_pre_user_flow = 1 "
            "WHERE legacy_pre_user_flow IS NULL OR legacy_pre_user_flow = 0"
        )
        await db.commit()
        logger.info("Z1 T3B: added missions.legacy_pre_user_flow column and backfilled")
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
            framework TEXT DEFAULT '',
            legacy_pre_p7 INTEGER DEFAULT 0,
            legacy_pre_charter INTEGER DEFAULT 0,
            legacy_pre_adr INTEGER DEFAULT 0,
            legacy_pre_falsification INTEGER DEFAULT 0,
            legacy_pre_non_goals INTEGER DEFAULT 0,
            legacy_pre_competitive_positioning INTEGER DEFAULT 0,
            legacy_pre_per_screen_plans INTEGER DEFAULT 0,
            legacy_pre_html_oids INTEGER DEFAULT 0,
            legacy_pre_compliance INTEGER DEFAULT 0,
            founder_attention_budget_minutes INTEGER,
            legacy_pre_premortem INTEGER DEFAULT 0,
            legacy_pre_spec_alive INTEGER DEFAULT 0,
            legacy_pre_prior_art INTEGER DEFAULT 0,
            interview_skip_reason TEXT,
            phase_7_rework_loops INTEGER DEFAULT 0
        )
    """)

    # B10 migration (i2p Z1): rework-loop counter on missions. Counts
    # rollbacks from phase >=7 back to phase <=6 for spec-first bet
    # telemetry. See src/telemetry/rework.py + docs/i2p-evolution
    # /01-pre-code-master-synthesis.md §B10. Idempotent ALTER for
    # existing DBs that pre-date the inline column above.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN phase_7_rework_loops INTEGER DEFAULT 0"
        )
    except Exception:
        pass  # Column already exists

    # P7 migration: add `legacy_pre_p7` column to existing DBs and backfill
    # rows that predate P7. Only the ALTER-succeeded branch backfills —
    # subsequent runs see the column already exists and skip the UPDATE so
    # missions created post-P7 keep their default 0.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN legacy_pre_p7 INTEGER DEFAULT 0"
        )
        # Every mission that exists at migration time predates P7 (its
        # blackboard artifacts have no `_schema_version` field).
        await db.execute("UPDATE missions SET legacy_pre_p7 = 1")
        logger.info("P7 migration: legacy_pre_p7 added + existing rows backfilled to 1")
    except Exception:
        # Column already exists — no-op; new missions default to 0.
        pass

    # Z1 Tier 1 migration: add `legacy_pre_charter` column to existing DBs
    # and backfill rows that predate the charter consolidation (steps
    # 0.0z / 0.0a / 0.1 product_charter). Same idempotent pattern as P7.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN legacy_pre_charter INTEGER DEFAULT 0"
        )
        # Every mission existing at migration time predates the charter
        # reshape — its phase-0 produced the old micro-artifacts (idea_brief,
        # problem_statement, target_users, value_proposition).
        await db.execute("UPDATE missions SET legacy_pre_charter = 1")
        logger.info(
            "Z1 migration: legacy_pre_charter added + existing rows backfilled to 1"
        )
    except Exception:
        # Column already exists — no-op; new missions default to 0.
        pass

    # Z1 Tier 2 migration: add `legacy_pre_adr` column. Tier 2 reshapes
    # phase 4 around universal-shape ADRs (P3 + C7 + A8); pre-existing
    # missions emitted Nygard-5-field artifacts only and cannot be
    # retroactively upgraded. Same idempotent pattern as P7/charter.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN legacy_pre_adr INTEGER DEFAULT 0"
        )
        await db.execute("UPDATE missions SET legacy_pre_adr = 1")
        logger.info(
            "Z1 Tier 2 migration: legacy_pre_adr added + existing rows backfilled to 1"
        )
    except Exception:
        # Column already exists — no-op; new missions default to 0.
        pass

    # Z1 Tier 2 migration (P4): add `legacy_pre_falsification` column.
    # Backfilled to 1 for existing missions — they predate the falsification
    # triple (risk_if_wrong / validation_method / falsification_signal) on
    # phase-3 requirement-emitting steps (3.1 / 3.2 / 3.3 / 3.7).
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_falsification INTEGER DEFAULT 0"
        )
        await db.execute("UPDATE missions SET legacy_pre_falsification = 1")
        logger.info(
            "Z1 Tier 2 migration: legacy_pre_falsification added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        # Column already exists — no-op; new missions default to 0.
        pass

    # Z1 Tier 2 migration (A2): add `legacy_pre_non_goals` column.
    # Backfilled to 1 for existing missions — they predate the 0.6a
    # non_goals_lock mission-wide refusal artifact.
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_non_goals INTEGER DEFAULT 0"
        )
        await db.execute("UPDATE missions SET legacy_pre_non_goals = 1")
        logger.info(
            "Z1 Tier 2 migration: legacy_pre_non_goals added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        # Column already exists — no-op; new missions default to 0.
        pass

    # Z1 Tier 2 migration (C2): legacy_pre_competitive_positioning column.
    # Backfilled to 1 for existing missions — they predate the
    # 1.4a competitive_positioning_lock step.
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_competitive_positioning INTEGER DEFAULT 0"
        )
        await db.execute(
            "UPDATE missions SET legacy_pre_competitive_positioning = 1"
        )
        logger.info(
            "Z1 Tier 2 migration: legacy_pre_competitive_positioning added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        pass

    # Z1 Tier 3 migration (C3+A10+C9+A11+C14): add
    # `legacy_pre_per_screen_plans` column. Backfilled to 1 for existing
    # missions — they predate the per-screen plan + HTML prototype reshape
    # of phase 5 (steps `5.1 generate_per_screen_plans` and
    # `5.2 generate_html_prototypes`).
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_per_screen_plans INTEGER DEFAULT 0"
        )
        await db.execute(
            "UPDATE missions SET legacy_pre_per_screen_plans = 1"
        )
        logger.info(
            "Z1 Tier 3 migration: legacy_pre_per_screen_plans added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        pass

    # Z1 Tier 4 migration (T4B / C17+A20): add `legacy_pre_html_oids`
    # column. Backfilled to 1 for existing missions — they predate the
    # `annotate_html_oids` post-processor (step `5.30c`) that tags
    # semantic blocks with `data-oid` for the spec-patch proposer.
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_html_oids INTEGER DEFAULT 0"
        )
        await db.execute(
            "UPDATE missions SET legacy_pre_html_oids = 1"
        )
        logger.info(
            "Z1 Tier 4 migration: legacy_pre_html_oids added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        pass

    # Z1 Tier 4 migration (T4C / C10+A19): preview-URL surface columns +
    # legacy gate. `preview_url` holds the most recent tunnel URL (NULL when
    # no tunnel is active or hosting is deferred to Z2). `preview_started_at`
    # is a CURRENT_TIMESTAMP marker. `legacy_pre_preview_url` gates the new
    # phase-5 `5.40 emit_preview_url` step so existing missions don't
    # retroactively require a preview surface; backfilled to 1 for them.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN preview_url TEXT"
        )
    except Exception:
        pass  # Column already exists
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN preview_started_at TIMESTAMP"
        )
    except Exception:
        pass  # Column already exists
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_preview_url INTEGER DEFAULT 0"
        )
        await db.execute(
            "UPDATE missions SET legacy_pre_preview_url = 1"
        )
        logger.info(
            "Z1 Tier 4 migration: legacy_pre_preview_url added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        pass  # Column already exists

    # Z1 Tier 5B migration (A6 / premortem): gate the new `6.5z
    # failure_premortem` step so existing missions don't retroactively
    # require a premortem.md. Backfilled to 1 for older rows; new
    # missions default to 0.
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_premortem INTEGER DEFAULT 0"
        )
        await db.execute(
            "UPDATE missions SET legacy_pre_premortem = 1"
        )
        logger.info(
            "Z1 Tier 5B migration: legacy_pre_premortem added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        pass  # Column already exists

    # Z1 Tier 5B migration (B5 / spec_consistency_check): gate the new
    # phase-7+ wave-start `<N>.0z` mechanical steps so existing missions
    # don't retroactively run drift checks against an absent phase-≤6
    # spec. Backfilled to 1 for older rows; new missions default to 0.
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_spec_alive INTEGER DEFAULT 0"
        )
        await db.execute(
            "UPDATE missions SET legacy_pre_spec_alive = 1"
        )
        logger.info(
            "Z1 Tier 5B migration: legacy_pre_spec_alive added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        pass  # Column already exists

    # Z1 Tier 4 (T4C / C10): preview_log audit trail — one row per
    # emit/kill action so we can correlate Telegram /preview commands
    # with mission lifecycle state.
    try:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS preview_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id INTEGER NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('emit','kill')),
                url TEXT,
                exit_code INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    except Exception:
        pass

    # Z1 Tier 2 migration (A4): interview_skip_reason column. NULL for
    # all existing rows — only set by request_interview_data when the
    # founder explicitly opts SKIP.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN interview_skip_reason TEXT"
        )
        logger.info("Z1 Tier 2 migration: interview_skip_reason column added")
    except Exception:
        pass

    # Z1 Tier 4A migration (C11+A15+C19) — regen primitive telemetry.
    # Idempotent CREATE; one row per artifact-level regen (artifact or
    # bundle scope). The `prev_version` / `new_version` columns hold the
    # absolute paths to the versioned `.v{N}` siblings on disk so the
    # full lineage of an artifact can be reconstructed by walking
    # `regen_log` for a (mission_id, artifact_path) pair.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS regen_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER NOT NULL,
            artifact_path TEXT NOT NULL,
            change_description TEXT NOT NULL,
            prev_version TEXT NOT NULL,
            new_version TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'artifact',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mission_id) REFERENCES missions(id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_regen_log_mission_artifact "
        "ON regen_log (mission_id, artifact_path, created_at)"
    )

    # Z1 Tier 5A migration (P6): legacy_pre_compliance column. Backfilled
    # to 1 for existing missions — they predate the compliance fingerprint
    # collection step (0.4a) and overlay (1.11a), so skip_when on those
    # new steps reads this flag and lets old missions short-circuit.
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_compliance INTEGER DEFAULT 0"
        )
        await db.execute(
            "UPDATE missions SET legacy_pre_compliance = 1"
        )
        logger.info(
            "Z1 Tier 5A migration: legacy_pre_compliance added "
            "+ existing rows backfilled to 1"
        )
    except Exception:
        pass  # Column already exists

    # Z1 Tier 5A migration (A5): founder_attention_budget_minutes column.
    # NULL for existing rows = unbounded (treated as ok=True by
    # attention_check). z0 sets it on new missions.
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN founder_attention_budget_minutes INTEGER"
        )
        logger.info(
            "Z1 Tier 5A migration: founder_attention_budget_minutes added"
        )
    except Exception:
        pass  # Column already exists

    # Z0 minimal slice — ambition_tier + cost_ceiling per mission. The full
    # z0 preflight wizard is a multi-feature zone of its own; this slice
    # ships the two fields downstream gates need (attention-budget defaults
    # by tier, cost-ceiling enforcement in 06-real-world-bridge).
    # ambition_tier ∈ {prototype, private_beta, public_launch, revenue_product}.
    # NULL = legacy / undeclared → treated as 'private_beta' by callers.
    for sql, label in (
        ("ALTER TABLE missions ADD COLUMN ambition_tier TEXT", "ambition_tier"),
        (
            "ALTER TABLE missions ADD COLUMN cost_ceiling_usd REAL",
            "cost_ceiling_usd",
        ),
        (
            "ALTER TABLE missions ADD COLUMN branched_from_mission_id INTEGER",
            "branched_from_mission_id",
        ),
    ):
        try:
            await db.execute(sql)
            logger.info(f"Z0 minimal migration: {label} added")
        except Exception:
            pass

    # 2026-05-12 Z8 T1A — mission lifecycle columns.
    # ``kind`` ∈ {oneshot, ongoing}; ``lifecycle_state`` ∈ {pending, active,
    # terminal, revoked}. Existing rows backfill to ('oneshot','terminal') so
    # nothing legacy looks like a live ongoing subscription.
    # ``cursor`` is opaque JSON owned by the handler (webhook event id,
    # cron last-fire ts, etc.). ``product_id`` is a nullable placeholder —
    # Z0 may take ownership later; routing code treats NULL as "default
    # product." Index supports the resumption query
    # (kind='ongoing' AND lifecycle_state='active').
    for sql, label in (
        ("ALTER TABLE missions ADD COLUMN kind TEXT NOT NULL DEFAULT 'oneshot'", "kind"),
        (
            "ALTER TABLE missions ADD COLUMN lifecycle_state TEXT NOT NULL DEFAULT 'terminal'",
            "lifecycle_state",
        ),
        ("ALTER TABLE missions ADD COLUMN cursor TEXT", "cursor"),
        ("ALTER TABLE missions ADD COLUMN product_id TEXT", "product_id"),
        ("ALTER TABLE missions ADD COLUMN revoked_at TEXT", "revoked_at"),
    ):
        try:
            await db.execute(sql)
            logger.info(f"Z8 T1A migration: {label} added to missions")
        except Exception:
            pass
    try:
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_missions_kind_state "
            "ON missions(kind, lifecycle_state)"
        )
    except Exception:
        pass

    # Z1 Tier 5A (A5): founder_attention_log — one row per debit. Used by
    # attention_check to compute remaining budget.
    try:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS founder_attention_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id INTEGER NOT NULL,
                step_id TEXT NOT NULL,
                action TEXT NOT NULL,
                minutes_debited INTEGER NOT NULL,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_founder_attention_log_mission "
            "ON founder_attention_log (mission_id, ts)"
        )
    except Exception:
        pass

    # Z1 Tier 5A (P6) — founder_signoffs. One row per (mission, doc_type) the
    # founder has affirmed. Read by compliance_blocker_check to gate phase
    # boundaries when `founder_review_required=true` is set in the
    # compliance_overlay. `signature_hash` is sha256[:16] of the signed
    # doc body at sign time — lets a later check detect template drift.
    try:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS founder_signoffs (
                mission_id INTEGER NOT NULL,
                doc_type TEXT NOT NULL,
                signed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                signature_hash TEXT,
                PRIMARY KEY (mission_id, doc_type)
            )
            """
        )
    except Exception:
        pass

    # Z1 Tier 5C (B4) — Critic gate audit trail. One row per critic call
    # (pass or veto) on an irreversible action. `redacted_payload_hash`
    # is a sha256[:16] of the SECRET-REDACTED payload — usable for
    # de-dup/correlation without storing the payload itself.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS critic_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER NOT NULL,
            action_name TEXT NOT NULL,
            verdict TEXT NOT NULL CHECK(verdict IN ('pass','veto')),
            reasons_json TEXT NOT NULL DEFAULT '[]',
            redacted_payload_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_critic_log_mission "
        "ON critic_log (mission_id, created_at)"
    )

    # Z1 Tier 5C (B4) — legacy_pre_critic_gate column on missions.
    # Backfilled to 1 for existing rows so critic-gate post-hooks know
    # the mission predates the gate (treat as legacy = no veto attempt).
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN legacy_pre_critic_gate INTEGER DEFAULT 0"
        )
        await db.execute("UPDATE missions SET legacy_pre_critic_gate = 1")
        logger.info(
            "Z1 T5C migration: legacy_pre_critic_gate added + backfilled"
        )
    except Exception:
        pass

    # Z1 Tier 6 (C18) — github_repo_url + legacy_pre_github_init on missions.
    # `github_repo_url` holds the live GitHub URL (NULL when not yet initialised
    # or hosting deferred via fail-soft pending status). `legacy_pre_github_init`
    # gates the new phase-6 `6.7 init_github_repo` step so existing missions
    # don't retroactively try to push a repo for already-shipped specs.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN github_repo_url TEXT"
        )
        logger.info("Z1 T6C migration: github_repo_url column added")
    except Exception:
        pass
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_github_init INTEGER DEFAULT 0"
        )
        await db.execute("UPDATE missions SET legacy_pre_github_init = 1")
        logger.info(
            "Z1 T6C migration: legacy_pre_github_init added + backfilled"
        )
    except Exception:
        pass

    # Z1 Tier 5C (B3) — streaming-guard audit trail. One row per
    # `warn` or `halt` outcome from the streaming-guards pipeline.
    # `fix` outcomes are silent (token rewritten in place).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS streaming_guard_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER,
            task_id INTEGER,
            guard_name TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('warn','halt','fix')),
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_streaming_guard_log_task "
        "ON streaming_guard_log (task_id, created_at)"
    )

    # Z1 Tier 6A (A7) — legacy_pre_idea_dedup column.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN legacy_pre_idea_dedup INTEGER DEFAULT 0"
        )
        await db.execute("UPDATE missions SET legacy_pre_idea_dedup = 1")
        logger.info(
            "Z1 T6A migration: legacy_pre_idea_dedup added + backfilled"
        )
    except Exception:
        pass

    # Z1 Tier 6A (P9) — legacy_pre_inheritance column.
    try:
        await db.execute(
            "ALTER TABLE missions ADD COLUMN legacy_pre_inheritance INTEGER DEFAULT 0"
        )
        await db.execute("UPDATE missions SET legacy_pre_inheritance = 1")
        logger.info(
            "Z1 T6A migration: legacy_pre_inheritance added + backfilled"
        )
    except Exception:
        pass

    # Z1 Tier 6A (P9) — cross-mission artifact index.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS mission_artifacts_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER NOT NULL,
            artifact_name TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            schema_version TEXT,
            domain_keywords_json TEXT NOT NULL DEFAULT '[]',
            founder_id TEXT NOT NULL DEFAULT 'default',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(mission_id, artifact_name)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_mission_artifacts_index_mission "
        "ON mission_artifacts_index (mission_id, artifact_name)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_mission_artifacts_index_founder "
        "ON mission_artifacts_index (founder_id, created_at)"
    )

    # Z1 Tier 6B (P5) — prior_art_cache.
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS prior_art_cache (
            domain_keywords_hash TEXT PRIMARY KEY,
            results_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            ttl_hours INTEGER NOT NULL DEFAULT 168
        )
        """
    )

    # Z1 Tier 6B (P5) — legacy_pre_prior_art column.
    try:
        await db.execute(
            "ALTER TABLE missions "
            "ADD COLUMN legacy_pre_prior_art INTEGER DEFAULT 0"
        )
        await db.execute("UPDATE missions SET legacy_pre_prior_art = 1")
        logger.info(
            "Z1 Tier 6B migration: legacy_pre_prior_art added + backfilled"
        )
    except Exception:
        pass  # Column already exists

    # Z1 Tier 7B (C21) — paraflow bundle-quality regression log.
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS paraflow_diff_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER NOT NULL,
            archetype TEXT NOT NULL,
            verdict TEXT NOT NULL,
            score REAL,
            gaps_json TEXT NOT NULL DEFAULT '[]',
            coverage_json TEXT NOT NULL DEFAULT '{}',
            coherence_json TEXT NOT NULL DEFAULT '{}',
            design_fitness_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_paraflow_diff_log_mission "
        "ON paraflow_diff_log (mission_id, created_at DESC)"
    )

    # Tasks
    await db.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER,
            parent_task_id INTEGER,
            title TEXT NOT NULL,
            description TEXT,
            agent_type TEXT DEFAULT 'executor',
            runner TEXT NOT NULL DEFAULT 'react',
            status TEXT DEFAULT 'pending',
            tier TEXT DEFAULT 'auto',
            priority INTEGER DEFAULT 5,
            requires_approval BOOLEAN DEFAULT 0,
            depends_on JSON DEFAULT '[]',
            result TEXT,
            error TEXT,
            context JSON DEFAULT '{}',
            worker_attempts INTEGER DEFAULT 0,
            max_worker_attempts INTEGER DEFAULT 15,
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
            interval_seconds INTEGER,
            kind TEXT DEFAULT 'user',
            agent_type TEXT DEFAULT 'executor',
            tier TEXT DEFAULT 'cheap',
            enabled BOOLEAN DEFAULT 1,
            last_run TIMESTAMP,
            next_run TIMESTAMP,
            context JSON DEFAULT '{}'
        )
    """)

    for col_sql in (
        "ALTER TABLE scheduled_tasks ADD COLUMN interval_seconds INTEGER",
        "ALTER TABLE scheduled_tasks ADD COLUMN kind TEXT DEFAULT 'user'",
    ):
        try:
            await db.execute(col_sql)
        except Exception:
            pass  # column already exists

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

    # Per-call token telemetry (pool-pressure machinery)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS model_call_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL DEFAULT (datetime('now')),
            task_id INTEGER,
            agent_type TEXT,
            workflow_step_id TEXT,
            workflow_phase TEXT,
            call_category TEXT,
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            is_streaming INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            reasoning_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER NOT NULL,
            duration_ms INTEGER,
            iteration_n INTEGER,
            success INTEGER NOT NULL
        )
    """)
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mct_task ON model_call_tokens(task_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mct_step ON model_call_tokens(agent_type, workflow_step_id)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_mct_recent ON model_call_tokens(timestamp)")

    # Per-step token percentile rollup (pool-pressure machinery)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS step_token_stats (
            agent_type TEXT NOT NULL,
            workflow_step_id TEXT NOT NULL,
            workflow_phase TEXT NOT NULL,
            samples_n INTEGER NOT NULL,
            in_p50 INTEGER, in_p90 INTEGER, in_p99 INTEGER,
            out_p50 INTEGER, out_p90 INTEGER, out_p99 INTEGER,
            iters_p50 REAL, iters_p90 REAL, iters_p99 REAL,
            updated_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (agent_type, workflow_step_id, workflow_phase)
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
    # Z10 T1A: `expires_at` added so orphan locks (task crashed without
    # explicit release) get reaped by sweep_file_locks(). Migration below
    # adds the column on existing DBs.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS file_locks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL,
            mission_id INTEGER,
            task_id INTEGER,
            agent_type TEXT,
            acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            UNIQUE(filepath)
        )
    """)
    # Migrate older databases that pre-date `expires_at`.
    try:
        await db.execute("BEGIN")
        await db.execute(
            "ALTER TABLE file_locks ADD COLUMN expires_at TIMESTAMP"
        )
        await db.execute("COMMIT")
        # T1C will absorb this into schema_migrations; for now leave a
        # filesystem breadcrumb that doesn't depend on the as-yet-unborn
        # ledger table.
        try:
            import pathlib as _pl
            _pl.Path("_migrations_pending.txt").open("a", encoding="utf-8").write(
                "z10_t1a_file_locks_expires_at\t"
                "ALTER TABLE file_locks ADD COLUMN expires_at TIMESTAMP\n"
            )
        except Exception:
            pass
        logger.info(
            "Z10 T1A migration: file_locks.expires_at column added"
        )
    except Exception:
        # Column already exists (or txn already rolled back) — best-effort.
        try:
            await db.execute("ROLLBACK")
        except Exception:
            pass

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

    await db.execute("""
        CREATE TABLE IF NOT EXISTS pending_llm_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id TEXT NOT NULL,
            artifact_name TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
            attempts INTEGER DEFAULT 0
        )
    """)

    # Model pick telemetry (Phase 1 selection-intelligence plan)
    await db.execute("""
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
    """)
    # Idempotent column add for pre-Phase-2c / pre-Task-5 / pre-Task-15 databases.
    # ``reinforce`` (Z9 T4E) holds a confirmed-verdict score nudge — see
    # record_reinforce_nudge() / fatih_hoca.grading.reinforce_bonus().
    for col_name, col_type in (
        ("pool", "TEXT"),
        ("urgency", "REAL"),
        ("success", "INTEGER"),
        ("error_category", "TEXT"),
        ("provider", "TEXT"),
        ("outcome", "TEXT"),
        ("reinforce", "REAL"),
    ):
        try:
            await db.execute(f"ALTER TABLE model_pick_log ADD COLUMN {col_name} {col_type}")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise
            # column already exists — expected on re-init
    # Backfill legacy rows: pre-cloud era was 100% local picks. Idempotent —
    # re-running just no-ops because no NULL rows remain.
    await db.execute(
        "UPDATE model_pick_log SET provider='local' WHERE provider IS NULL"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_pick_log_provider ON model_pick_log(provider)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_pick_log_task ON model_pick_log(task_name, timestamp DESC)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_pick_log_model ON model_pick_log(picked_model, timestamp DESC)"
    )
    # ── Admission violations (Q1 forensics) ──────────────────────────────
    # Captures every "Beckman admitted, KDV/dispatcher rejected" event.
    # Three sites:
    #   1. caller.py KDV pre_call refusal post-admission (rate_limit, canary,
    #      circuit_breaker, daily_exhausted) — admission_time gate failed
    #   2. dispatcher.py pick is None during retry recursion — pool drained
    #      mid-task, pressure model failed to predict
    #   3. caller.py daily_exhausted at call-time — selector eligibility
    #      missed it
    # Forensic trail for offline pressure-model tuning. NOT for live
    # decisions; consumers query this table to identify saturation patterns,
    # not to gate admission.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS admission_violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            site TEXT NOT NULL,
            phase TEXT NOT NULL,
            task_id INTEGER,
            call_category TEXT,
            agent_type TEXT,
            difficulty INTEGER,
            model TEXT,
            provider TEXT,
            reason TEXT,
            wait_seconds REAL,
            scope TEXT,
            error_category TEXT,
            error_message TEXT,
            in_flight_n INTEGER,
            queue_total INTEGER,
            queue_hard INTEGER,
            snapshot_summary TEXT,
            extra_json TEXT
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_admission_viol_ts "
        "ON admission_violations(timestamp DESC)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_admission_viol_model "
        "ON admission_violations(model, timestamp DESC)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_admission_viol_site "
        "ON admission_violations(site, timestamp DESC)"
    )
    # ── KDV (kuleden_donen_var) persistent state ─────────────────────────
    # One row per (scope, scope_key). scope ∈ {"model","provider","breaker"}.
    # snapshot_json holds the dict from RateLimitState/CircuitBreaker
    # snapshot_state(). last_persisted is unix epoch; loader drops rows
    # older than 24h to avoid restoring stale 60s windows or stale
    # header reset times. KDV per-row design lets us do partial updates
    # without rewriting one giant blob each save.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS kdv_state (
            scope TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            last_persisted REAL NOT NULL,
            PRIMARY KEY (scope, scope_key)
        )
    """)
    # ── Provider/model registry ───────────────────────────────────────────
    # Replaces the flat .dead_models.json file. Three concerns:
    #   providers       — provider-level status (auth dead, key rotation hash)
    #   models          — model-level status (404, cause, TTL expiry)
    #   registry_events — append-only audit trail (mark_dead/revive/probe)
    # Reads from packages/fatih_hoca/registry.py via src/infra/registry_store.
    # Writes from caller (404/auth), discovery (revive), telegram (/revive),
    # orchestrator boot probe.
    #
    # status='dead' AND (expires_at IS NULL OR expires_at > now) = effectively
    # dead. Per-cause TTL stored explicitly on the row (not derived) so policy
    # tweaks don't retroactively shift live entries.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            name        TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'active',
            cause       TEXT,
            marked_at   TIMESTAMP,
            revived_at  TIMESTAMP,
            key_hash    TEXT
        )
    """)
    await db.execute("""
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
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS registry_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scope           TEXT NOT NULL,
            target          TEXT NOT NULL,
            event           TEXT NOT NULL,
            cause           TEXT,
            actor           TEXT,
            payload_json    TEXT
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_registry_events_target_ts "
        "ON registry_events(target, timestamp DESC)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_models_status "
        "ON models(status, provider)"
    )

    # ── Z10 T1C: schema_migrations ledger ─────────────────────────────────
    # Records every DDL migration applied to this database. apply_migration()
    # is the only sanctioned way to evolve the schema from this point on:
    # all DDL in T1C (and future tiers) runs through it inside a single
    # BEGIN/COMMIT so a partial failure leaves no half-applied state.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version       TEXT PRIMARY KEY,
            applied_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sql           TEXT NOT NULL,
            reversal_sql  TEXT,
            description   TEXT
        )
    """)
    await db.commit()

    # Absorb the T1A breadcrumb (if present): the file_locks.expires_at
    # ALTER ran before this ledger existed. Record-only (already applied).
    try:
        import pathlib as _pl
        bc = _pl.Path("_migrations_pending.txt")
        if bc.exists():
            for line in bc.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                version = parts[0]
                sql_text = parts[1] if len(parts) > 1 else ""
                cur = await db.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (version,),
                )
                exists = await cur.fetchone()
                if not exists:
                    await db.execute(
                        "INSERT INTO schema_migrations "
                        "(version, sql, reversal_sql, description) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            version,
                            sql_text,
                            None,
                            "Absorbed from _migrations_pending.txt breadcrumb",
                        ),
                    )
            await db.commit()
            try:
                bc.unlink()
            except Exception:
                pass
            logger.info(
                "Z10 T1C: absorbed _migrations_pending.txt into schema_migrations"
            )
    except Exception as e:
        logger.debug(f"breadcrumb absorption skipped: {e}")

    # ── Z10 T1C migrations ────────────────────────────────────────────────
    # 1. tasks: confidence_categorical / confidence_numeric / reasoning / reversibility
    await apply_migration(
        version="2026-05-10-tasks-confidence-reversibility",
        sql=(
            "ALTER TABLE tasks ADD COLUMN confidence_categorical TEXT;\n"
            "ALTER TABLE tasks ADD COLUMN confidence_numeric REAL;\n"
            "ALTER TABLE tasks ADD COLUMN reasoning TEXT;\n"
            "ALTER TABLE tasks ADD COLUMN reversibility TEXT;\n"
        ),
        reversal_sql=(
            "ALTER TABLE tasks DROP COLUMN confidence_categorical;\n"
            "ALTER TABLE tasks DROP COLUMN confidence_numeric;\n"
            "ALTER TABLE tasks DROP COLUMN reasoning;\n"
            "ALTER TABLE tasks DROP COLUMN reversibility;\n"
        ),
        description=(
            "T1C provenance: tasks confidence (categorical/numeric), "
            "reasoning, reversibility columns"
        ),
    )

    # 2. artifact_provenance table + indexes
    await apply_migration(
        version="2026-05-10-artifact-provenance",
        sql=(
            "CREATE TABLE artifact_provenance ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " path TEXT NOT NULL,"
            " task_id INTEGER,"
            " step_id TEXT,"
            " model_id TEXT,"
            " retry_n INTEGER DEFAULT 0,"
            " reviewer_verdict_id INTEGER,"
            " mission_id INTEGER,"
            " written_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " FOREIGN KEY (task_id) REFERENCES tasks(id)"
            ");\n"
            "CREATE INDEX idx_artifact_provenance_path "
            "ON artifact_provenance(path);\n"
            "CREATE INDEX idx_artifact_provenance_mission "
            "ON artifact_provenance(mission_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_artifact_provenance_mission;\n"
            "DROP INDEX IF EXISTS idx_artifact_provenance_path;\n"
            "DROP TABLE IF EXISTS artifact_provenance;\n"
        ),
        description=(
            "T1C provenance: artifact_provenance table joining writes to "
            "task/step/model/mission/reviewer"
        ),
    )

    # 3. registry_events: per-action audit columns
    await apply_migration(
        version="2026-05-10-registry-events-action-scope",
        sql=(
            "ALTER TABLE registry_events ADD COLUMN mission_id INTEGER;\n"
            "ALTER TABLE registry_events ADD COLUMN task_id INTEGER;\n"
            "ALTER TABLE registry_events ADD COLUMN verb TEXT;\n"
            "ALTER TABLE registry_events ADD COLUMN reversibility TEXT;\n"
        ),
        reversal_sql=(
            "ALTER TABLE registry_events DROP COLUMN mission_id;\n"
            "ALTER TABLE registry_events DROP COLUMN task_id;\n"
            "ALTER TABLE registry_events DROP COLUMN verb;\n"
            "ALTER TABLE registry_events DROP COLUMN reversibility;\n"
        ),
        description=(
            "T1C provenance: registry_events extended for scope='action' "
            "audit rows (mission_id/task_id/verb/reversibility)"
        ),
    )

    # 4. action_confirmations table + index
    await apply_migration(
        version="2026-05-10-action-confirmations",
        sql=(
            "CREATE TABLE action_confirmations ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " task_id INTEGER NOT NULL,"
            " verb TEXT NOT NULL,"
            " reversibility TEXT NOT NULL,"
            " payload_summary TEXT,"
            " requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " responded_at TIMESTAMP,"
            " verdict TEXT"
            ");\n"
            "CREATE INDEX idx_action_confirmations_task "
            "ON action_confirmations(task_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_action_confirmations_task;\n"
            "DROP TABLE IF EXISTS action_confirmations;\n"
        ),
        description=(
            "T1C confirmation flow: action_confirmations skeleton table "
            "(Telegram wire deferred to T2B)"
        ),
    )

    # ── Z10 T2A migrations (cost transparency wiring) ─────────────────────
    # 5. cost_budgets: add budget_ceiling_usd + helpful index
    await apply_migration(
        version="2026-05-10-cost-budgets-mission-scope",
        sql=(
            "ALTER TABLE cost_budgets ADD COLUMN budget_ceiling_usd REAL;\n"
            "CREATE INDEX IF NOT EXISTS idx_cost_budgets_scope "
            "ON cost_budgets(scope, scope_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_cost_budgets_scope;\n"
            "ALTER TABLE cost_budgets DROP COLUMN budget_ceiling_usd;\n"
        ),
        description=(
            "T2A cost wiring: cost_budgets.budget_ceiling_usd + scope index"
        ),
    )

    # 6. model_call_tokens: add cost_usd column for per-call cost capture
    await apply_migration(
        version="2026-05-10-model-call-tokens-cost-column",
        sql=(
            "ALTER TABLE model_call_tokens ADD COLUMN cost_usd REAL;\n"
        ),
        reversal_sql=(
            "ALTER TABLE model_call_tokens DROP COLUMN cost_usd;\n"
        ),
        description=(
            "T2A cost wiring: model_call_tokens.cost_usd for per-call USD"
        ),
    )

    # 7. cost_by_iteration view
    await apply_migration(
        version="2026-05-10-cost-by-iteration-view",
        sql=(
            "DROP VIEW IF EXISTS cost_by_iteration;\n"
            "CREATE VIEW cost_by_iteration AS\n"
            "SELECT\n"
            "  t.mission_id AS mission_id,\n"
            "  mct.iteration_n AS iteration_n,\n"
            "  SUM(COALESCE(mct.prompt_tokens, 0)) AS prompt_tokens,\n"
            "  SUM(COALESCE(mct.completion_tokens, 0)) AS completion_tokens,\n"
            "  SUM(COALESCE(mct.total_tokens, 0)) AS total_tokens,\n"
            "  SUM(COALESCE(mct.cost_usd, 0)) AS cost_usd,\n"
            "  COUNT(*) AS calls\n"
            "FROM model_call_tokens mct\n"
            "JOIN tasks t ON t.id = mct.task_id\n"
            "GROUP BY t.mission_id, mct.iteration_n;\n"
        ),
        reversal_sql=(
            "DROP VIEW IF EXISTS cost_by_iteration;\n"
        ),
        description=(
            "T2A cost wiring: cost_by_iteration view aggregating per-mission "
            "× iteration_n token + cost"
        ),
    )

    # 8. mission_budget_alerts table (cron writes; T2B drains)
    await apply_migration(
        version="2026-05-10-mission-budget-alerts",
        sql=(
            "CREATE TABLE IF NOT EXISTS mission_budget_alerts ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER NOT NULL,"
            " threshold REAL NOT NULL,"
            " total_usd REAL NOT NULL,"
            " posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " drained_at TIMESTAMP"
            ");\n"
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mission_budget_alerts_mt "
            "ON mission_budget_alerts(mission_id, threshold);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_mission_budget_alerts_mt;\n"
            "DROP TABLE IF EXISTS mission_budget_alerts;\n"
        ),
        description=(
            "T2A cost wiring: mission_budget_alerts (50/75/90% threshold "
            "rows; UNIQUE per mission+threshold = idempotent cron writes)"
        ),
    )

    # 9. tasks.estimated_cost_usd + actual_cost_usd
    await apply_migration(
        version="2026-05-10-tasks-estimated-cost",
        sql=(
            "ALTER TABLE tasks ADD COLUMN estimated_cost_usd REAL;\n"
            "ALTER TABLE tasks ADD COLUMN actual_cost_usd REAL;\n"
        ),
        reversal_sql=(
            "ALTER TABLE tasks DROP COLUMN estimated_cost_usd;\n"
            "ALTER TABLE tasks DROP COLUMN actual_cost_usd;\n"
        ),
        description=(
            "T2A cost wiring: tasks.estimated_cost_usd + actual_cost_usd"
        ),
    )

    # 10. missions.cost_decision_threshold_usd
    await apply_migration(
        version="2026-05-10-missions-cost-threshold",
        sql=(
            "ALTER TABLE missions ADD COLUMN cost_decision_threshold_usd REAL;\n"
        ),
        reversal_sql=(
            "ALTER TABLE missions DROP COLUMN cost_decision_threshold_usd;\n"
        ),
        description=(
            "T2A cost wiring: missions.cost_decision_threshold_usd "
            "(per-mission cost-at-decision floor; NULL → $1.00 default)"
        ),
    )

    # 11. missions.quality_mode (quick / balanced / thorough)
    await apply_migration(
        version="2026-05-10-missions-quality-mode",
        sql=(
            "ALTER TABLE missions ADD COLUMN quality_mode TEXT DEFAULT 'balanced';\n"
        ),
        reversal_sql=(
            "ALTER TABLE missions DROP COLUMN quality_mode;\n"
        ),
        description=(
            "T2A cost wiring: missions.quality_mode dial "
            "(quick|balanced|thorough)"
        ),
    )

    # ── Z10 T2B: per-mission Telegram forum topics ───────────────────
    await apply_migration(
        version="2026-05-10-missions-telegram-thread",
        sql=(
            "ALTER TABLE missions ADD COLUMN telegram_thread_id INTEGER;\n"
            "ALTER TABLE missions ADD COLUMN telegram_thread_archived "
            "INTEGER DEFAULT 0;\n"
        ),
        reversal_sql=(
            "ALTER TABLE missions DROP COLUMN telegram_thread_id;\n"
            "ALTER TABLE missions DROP COLUMN telegram_thread_archived;\n"
        ),
        description=(
            "T2B per-mission forum topics: missions.telegram_thread_id "
            "(NULL = unallocated / fallback-flat-mode) + archived flag"
        ),
    )

    # ── Z10 T2B: mission_events typed event log ──────────────────────
    await apply_migration(
        version="2026-05-10-mission-events",
        sql=(
            "CREATE TABLE mission_events ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER NOT NULL,"
            " kind TEXT NOT NULL,"
            " payload TEXT NOT NULL,"
            " telegram_message_id INTEGER,"
            " posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " resolved_at TIMESTAMP,"
            " resolution TEXT"
            ");\n"
            "CREATE INDEX idx_mission_events_mission "
            "ON mission_events(mission_id);\n"
            "CREATE INDEX idx_mission_events_msgid "
            "ON mission_events(telegram_message_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_mission_events_msgid;\n"
            "DROP INDEX IF EXISTS idx_mission_events_mission;\n"
            "DROP TABLE IF EXISTS mission_events;\n"
        ),
        description=(
            "T2B typed mission events: milestone/blocker/asking/"
            "confirmation_required/cost_alert rendered to mission thread"
        ),
    )

    # ── Z10 T2B: link action_confirmations to Telegram events ─────────
    await apply_migration(
        version="2026-05-10-action-confirmations-telegram",
        sql=(
            "ALTER TABLE action_confirmations "
            "ADD COLUMN telegram_event_id INTEGER;\n"
        ),
        reversal_sql=(
            "ALTER TABLE action_confirmations "
            "DROP COLUMN telegram_event_id;\n"
        ),
        description=(
            "T2B confirmation drain: action_confirmations.telegram_event_id "
            "links a pending confirmation to its posted mission_event row"
        ),
    )

    # ── Z10 T3A: time awareness ─────────────────────────────────────
    # 12. missions: target_launch + time_budget_hours + phase_budget_json
    await apply_migration(
        version="2026-05-10-missions-time-awareness",
        sql=(
            "ALTER TABLE missions ADD COLUMN target_launch DATE;\n"
            "ALTER TABLE missions ADD COLUMN time_budget_hours REAL;\n"
            "ALTER TABLE missions ADD COLUMN phase_budget_json TEXT;\n"
        ),
        reversal_sql=(
            "ALTER TABLE missions DROP COLUMN target_launch;\n"
            "ALTER TABLE missions DROP COLUMN time_budget_hours;\n"
            "ALTER TABLE missions DROP COLUMN phase_budget_json;\n"
        ),
        description=(
            "T3A time awareness: missions.target_launch + time_budget_hours "
            "+ phase_budget_json (per-phase hour budgets)"
        ),
    )

    # 13. tasks: step_started_at + phase_id
    await apply_migration(
        version="2026-05-10-tasks-step-timing",
        sql=(
            "ALTER TABLE tasks ADD COLUMN step_started_at TIMESTAMP;\n"
            "ALTER TABLE tasks ADD COLUMN phase_id TEXT;\n"
        ),
        reversal_sql=(
            "ALTER TABLE tasks DROP COLUMN step_started_at;\n"
            "ALTER TABLE tasks DROP COLUMN phase_id;\n"
        ),
        description=(
            "T3A time awareness: tasks.step_started_at + tasks.phase_id "
            "(populated from workflow context at expansion time)"
        ),
    )

    # 14. mission_pacing_snapshots
    await apply_migration(
        version="2026-05-10-mission-pacing-snapshots",
        sql=(
            "CREATE TABLE mission_pacing_snapshots ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER NOT NULL,"
            " taken_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " elapsed_hours REAL,"
            " remaining_budget_hours REAL,"
            " projected_finish_at TIMESTAMP,"
            " percent_burn REAL,"
            " scope_remaining_pct REAL"
            ");\n"
            "CREATE INDEX idx_mission_pacing_snapshots_mt "
            "ON mission_pacing_snapshots(mission_id, taken_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_mission_pacing_snapshots_mt;\n"
            "DROP TABLE IF EXISTS mission_pacing_snapshots;\n"
        ),
        description=(
            "T3A time awareness: mission_pacing_snapshots time-series for "
            "elapsed / remaining / burn / projected finish"
        ),
    )

    # 15. mission_tradeoff_prompts (idempotent guard for 75/25 cron)
    await apply_migration(
        version="2026-05-10-mission-tradeoff-prompts",
        sql=(
            "CREATE TABLE mission_tradeoff_prompts ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER NOT NULL,"
            " posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " mission_event_id INTEGER,"
            " resolution TEXT"
            ");\n"
            "CREATE UNIQUE INDEX idx_mission_tradeoff_prompts_md "
            "ON mission_tradeoff_prompts(mission_id, DATE(posted_at));\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_mission_tradeoff_prompts_md;\n"
            "DROP TABLE IF EXISTS mission_tradeoff_prompts;\n"
        ),
        description=(
            "T3A time awareness: mission_tradeoff_prompts log "
            "(UNIQUE(mission_id, DATE(posted_at)) → idempotent daily "
            "tradeoff prompt at 75%/25%)"
        ),
    )

    # ── Z10 T3B: sandboxing per mission ──────────────────────────────
    # 16. missions.sandbox_resource_overrides_json
    await apply_migration(
        version="2026-05-10-missions-sandbox-overrides",
        sql=(
            "ALTER TABLE missions ADD COLUMN sandbox_resource_overrides_json TEXT;\n"
        ),
        reversal_sql=(
            "ALTER TABLE missions DROP COLUMN sandbox_resource_overrides_json;\n"
        ),
        description=(
            "T3B per-mission container: optional JSON dict of resource caps "
            "{memory, cpus, pids_limit} overriding env defaults"
        ),
    )

    # 17. missions.sandbox_mode (per-mission docker/local opt-in)
    await apply_migration(
        version="2026-05-10-missions-sandbox-mode",
        sql=(
            "ALTER TABLE missions ADD COLUMN sandbox_mode TEXT DEFAULT 'docker';\n"
        ),
        reversal_sql=(
            "ALTER TABLE missions DROP COLUMN sandbox_mode;\n"
        ),
        description=(
            "T3B per-mission sandbox mode opt-in: docker (default) | local. "
            "When mission requests local AND system default isn't local, "
            "dispatcher opens a sandbox_local_mode confirmation."
        ),
    )

    # ── Z10 T3C: mission green-tag ledger ───────────────────────────
    # Records every successful "green" checkpoint per mission so the
    # rollback_mission verb can restore workspace/git + mission DB rows +
    # Chroma snapshot atomically.
    await apply_migration(
        version="2026-05-10-mission-green-tags",
        sql=(
            "CREATE TABLE mission_green_tags ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER NOT NULL,"
            " task_id INTEGER NOT NULL,"
            " git_tag TEXT NOT NULL,"
            " db_snapshot_path TEXT NOT NULL,"
            " chroma_snapshot_path TEXT NOT NULL,"
            " schema_migrations_at TEXT,"
            " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " UNIQUE(mission_id, task_id)"
            ");\n"
            "CREATE INDEX idx_mission_green_tags_mission "
            "ON mission_green_tags(mission_id, created_at DESC);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_mission_green_tags_mission;\n"
            "DROP TABLE IF EXISTS mission_green_tags;\n"
        ),
        description=(
            "T3C reset-to-green: ledger of mission green checkpoints with "
            "paired git tag + DB snapshot + Chroma snapshot paths + schema "
            "version at green time for rewind-on-rollback"
        ),
    )

    # ── Z10 T4A: demo deliverable strict-mode flag ────────────────────
    # When demo_required=1 (default), missions without e2e specs post a
    # [blocker] mission event. When demo_required=0, the demo step is
    # allowed to skip silently. Founder controls per-mission strictness.
    await apply_migration(
        version="2026-05-10-missions-demo-required",
        sql=(
            "ALTER TABLE missions ADD COLUMN demo_required INTEGER DEFAULT 1;\n"
        ),
        reversal_sql=(
            "ALTER TABLE missions DROP COLUMN demo_required;\n"
        ),
        description=(
            "T4A end-of-mission demo strictness: 1 (default) posts [blocker] "
            "when no e2e specs found; 0 allows silent skip"
        ),
    )

    # ── Z10 wire-fixes F3: per-mission demo scenario path ────────────────
    # record_demo previously hardcoded tests/e2e/golden_path.spec.ts. This
    # column lets add_mission seed a per-mission default (auto-set when the
    # mission's stack hints "web"); record_demo's resolution order is:
    # payload.scenario_path > missions.demo_scenario_path > newest
    # tests/e2e/*.spec.[tj]s > no_e2e_specs skip path.
    await apply_migration(
        version="2026-05-11-missions-demo-scenario-path",
        sql=(
            "ALTER TABLE missions ADD COLUMN demo_scenario_path TEXT;\n"
        ),
        reversal_sql=(
            "ALTER TABLE missions DROP COLUMN demo_scenario_path;\n"
        ),
        description=(
            "z10-wire-fixes F3: per-mission demo scenario path so record_demo "
            "isn't hardcoded to tests/e2e/golden_path.spec.ts. Non-web missions "
            "leave it NULL → record_demo falls through to no_e2e_specs path."
        ),
    )

    # ── Z10 T4B: confidence_outcomes (trust calibration loop) ────────────
    # Each row attributes a confidence claim on a task to an actual outcome
    # (reviewer-approved, downstream pass, regression). The nightly job
    # in cron_seed/cron rolls these up into ``confidence_reliability_scores``
    # per-(model, task_kind, bucket) which the prompt builder consults to
    # nudge well/poorly calibrated agents.
    await apply_migration(
        version="2026-05-11-confidence-outcomes",
        sql=(
            "CREATE TABLE confidence_outcomes ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " task_id INTEGER NOT NULL,"
            " mission_id INTEGER,"
            " agent_type TEXT,"
            " task_kind TEXT,"
            " model_id TEXT NOT NULL,"
            " picked_at TIMESTAMP NOT NULL,"
            " confidence_categorical TEXT,"
            " confidence_numeric REAL,"
            " outcome_correct INTEGER,"
            " outcome_resolved_at TIMESTAMP,"
            " resolution_source TEXT,"
            " reviewer_verdict_id INTEGER,"
            " notes TEXT"
            ");\n"
            "CREATE INDEX idx_confidence_outcomes_model_kind "
            "ON confidence_outcomes(model_id, task_kind);\n"
            "CREATE INDEX idx_confidence_outcomes_unresolved "
            "ON confidence_outcomes(outcome_correct) "
            "WHERE outcome_correct IS NULL;\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_confidence_outcomes_unresolved;\n"
            "DROP INDEX IF EXISTS idx_confidence_outcomes_model_kind;\n"
            "DROP TABLE IF EXISTS confidence_outcomes;\n"
        ),
        description=(
            "T4B trust calibration: confidence_outcomes attribution table "
            "(per-task confidence claim + later outcome resolution)"
        ),
    )

    # 2. confidence_reliability_scores — aggregated nightly by recompute
    await apply_migration(
        version="2026-05-11-confidence-reliability-scores",
        sql=(
            "CREATE TABLE confidence_reliability_scores ("
            " model_id TEXT NOT NULL,"
            " task_kind TEXT NOT NULL,"
            " confidence_bucket TEXT NOT NULL,"
            " sample_n INTEGER NOT NULL,"
            " correct_n INTEGER NOT NULL,"
            " reliability REAL NOT NULL,"
            " updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " PRIMARY KEY (model_id, task_kind, confidence_bucket)"
            ");\n"
        ),
        reversal_sql=(
            "DROP TABLE IF EXISTS confidence_reliability_scores;\n"
        ),
        description=(
            "T4B trust calibration: per-(model, task_kind, bucket) "
            "reliability rollup. Recomputed by cron every 6h."
        ),
    )

    # ── Z2 T4A: mission_lessons — cross-mission failure memory ───────────
    await apply_migration(
        version="2026-05-10-mission-lessons",
        sql=(
            "CREATE TABLE IF NOT EXISTS mission_lessons ("
            " id           INTEGER PRIMARY KEY AUTOINCREMENT,"
            " stack        TEXT    NOT NULL,"
            " domain       TEXT    NOT NULL,"
            " pattern      TEXT    NOT NULL,"
            " fix          TEXT    NOT NULL DEFAULT '',"
            " severity     TEXT    NOT NULL DEFAULT 'warning',"
            " occurrences  INTEGER NOT NULL DEFAULT 1,"
            " dedup_key    TEXT    NOT NULL UNIQUE,"
            " source_kind  TEXT    NOT NULL,"
            " source_ref   TEXT    NOT NULL DEFAULT '{}',"
            " suppressed   INTEGER NOT NULL DEFAULT 0,"
            " created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_mission_lessons_stack_domain "
            "ON mission_lessons(stack, domain);\n"
            "CREATE INDEX IF NOT EXISTS idx_mission_lessons_occurrences "
            "ON mission_lessons(occurrences DESC, last_seen_at DESC);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_mission_lessons_occurrences;\n"
            "DROP INDEX IF EXISTS idx_mission_lessons_stack_domain;\n"
            "DROP TABLE IF EXISTS mission_lessons;\n"
        ),
        description=(
            "Z2 T4A cross-mission memory: mission_lessons table + indexes. "
            "Dedup via sha256(stack\\ndomain\\nnormalized_pattern)[:32]. "
            "Populated by DLQ analyst + posthook-fail populators."
        ),
    )

    # ── Z2 T5A: recipe_pin_log — recipe pinning ledger ───────────────────────
    await apply_migration(
        version="2026-05-11-recipe-pin-log",
        sql=(
            "CREATE TABLE IF NOT EXISTS recipe_pin_log ("
            " id          INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id  INTEGER NOT NULL,"
            " recipe_name TEXT    NOT NULL,"
            " version     TEXT    NOT NULL,"
            " fit_score   REAL    NOT NULL DEFAULT 1.0,"
            " pinned_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            " UNIQUE(mission_id, recipe_name)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_recipe_pin_log_mission "
            "ON recipe_pin_log(mission_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_recipe_pin_log_mission;\n"
            "DROP TABLE IF EXISTS recipe_pin_log;\n"
        ),
        description=(
            "Z2 T5A recipe library substrate: recipe_pin_log table + index. "
            "Tracks which recipe version was pinned to each mission at mission start. "
            "UNIQUE(mission_id, recipe_name) prevents silent upgrades mid-mission."
        ),
    )

    # ── Z6 T1B: founder_actions table (real-world bridge queue) ───────────
    # A founder_action is "agent surfaces a real-world task to founder,
    # mission parks until founder marks done with optional output". One
    # row per pending real-world handoff. Lifecycle: pending → in_progress
    # → {done|blocked|cancelled}. Status transitions are gated by the repo
    # module in src/founder_actions/ — DDL only here.
    await apply_migration(
        version="2026-05-11-founder-actions",
        sql=(
            "CREATE TABLE IF NOT EXISTS founder_actions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER NOT NULL,"
            " blocking_task_id INTEGER,"
            " blocking_step_id TEXT,"
            " kind TEXT NOT NULL,"
            " title TEXT NOT NULL,"
            " why TEXT NOT NULL,"
            " instructions_json TEXT NOT NULL,"
            " expected_output_kind TEXT,"
            " expected_output_schema_json TEXT,"
            " cost_estimate_usd REAL,"
            " reversibility TEXT,"
            " status TEXT NOT NULL DEFAULT 'pending',"
            " response_payload_json TEXT,"
            " created_at TEXT NOT NULL,"
            " updated_at TEXT NOT NULL,"
            " resolved_at TEXT,"
            " FOREIGN KEY (mission_id) REFERENCES missions(id),"
            " FOREIGN KEY (blocking_task_id) REFERENCES tasks(id)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_founder_actions_mission "
            "ON founder_actions(mission_id);\n"
            "CREATE INDEX IF NOT EXISTS idx_founder_actions_status "
            "ON founder_actions(status);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_founder_actions_status;\n"
            "DROP INDEX IF EXISTS idx_founder_actions_mission;\n"
            "DROP TABLE IF EXISTS founder_actions;\n"
        ),
        description=(
            "Z6 T1B: founder_actions table + mission/status indexes. "
            "Surfaces real-world tasks to founder; mission blocks while "
            "any row is pending/in_progress."
        ),
    )

    # ── Z6 polish P1: founder_actions.urgent flag for DM bypass ─────────────
    # When urgent=1, the Telegram notifier DMs the admin chat directly
    # instead of posting to the mission thread (disputes, payment failures,
    # security incidents, expired credentials). Idempotent ADD COLUMN.
    await apply_migration(
        version="2026-05-12-founder-actions-urgent",
        sql=(
            "ALTER TABLE founder_actions "
            "ADD COLUMN urgent INTEGER DEFAULT 0;\n"
        ),
        reversal_sql=(
            "ALTER TABLE founder_actions DROP COLUMN urgent;\n"
        ),
        description=(
            "Z6 polish P1: founder_actions.urgent flag drives DM-to-admin "
            "bypass for disputes, expired creds, security incidents."
        ),
    )

    # ── Z6 T1A: hoist needs_real_tools from task.context to indexed column ──
    # reversibility was added by 2026-05-10-tasks-confidence-reversibility
    # (Z10 T1C); needs_real_tools is new. Idempotent ADD COLUMN.
    await apply_migration(
        version="2026-05-11-tasks-needs-real-tools",
        sql=(
            "ALTER TABLE tasks ADD COLUMN needs_real_tools INTEGER DEFAULT 0;\n"
        ),
        reversal_sql=(
            "ALTER TABLE tasks DROP COLUMN needs_real_tools;\n"
        ),
        description=(
            "Z6 T1A: tasks.needs_real_tools indexed column hoisted from "
            "task.context JSON so beckman admission can gate without parsing."
        ),
    )

    # ── Z8 T1B: tasks.lane admission column ─────────────────────────
    # Lane is the admission pool: 'oneshot' (default) vs 'ongoing'.
    # Ongoing covers alert_triage / cron / support_ticket tasks attached
    # to ``kind='ongoing'`` missions. Caps live in
    # ``general_beckman.lanes``. Backfill to 'oneshot' so legacy rows
    # keep flowing through the historical pool.
    await apply_migration(
        version="2026-05-12-tasks-lane",
        sql=(
            "ALTER TABLE tasks ADD COLUMN lane TEXT NOT NULL DEFAULT 'oneshot';\n"
            "CREATE INDEX IF NOT EXISTS idx_tasks_lane_status "
            "ON tasks(lane, status);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_tasks_lane_status;\n"
            "ALTER TABLE tasks DROP COLUMN lane;\n"
        ),
        description=(
            "Z8 T1B: tasks.lane admission column for ongoing-mission "
            "lane separation (cap=8 ongoing / cap=4 oneshot)."
        ),
    )

    # ── Z6 T2A: credentials hardening — scope, rotated_at, expires_at,
    # key_version, schema_id columns. expires_at is promoted from inside the
    # encrypted envelope to an indexable column (still also kept inside the
    # envelope for tamper-proofing). scope defaults to 'read_write' for
    # legacy rows so they remain usable. key_version=1 means the row was
    # encrypted with KUTAY_MASTER_KEY (or KUTAY_MASTER_KEY_v1).
    await apply_migration(
        version="2026-05-11-credentials-hardening",
        sql=(
            "ALTER TABLE credentials "
            "ADD COLUMN scope TEXT DEFAULT 'read_write';\n"
            "ALTER TABLE credentials ADD COLUMN rotated_at TEXT;\n"
            "ALTER TABLE credentials ADD COLUMN expires_at TEXT;\n"
            "ALTER TABLE credentials "
            "ADD COLUMN key_version INTEGER DEFAULT 1;\n"
            "ALTER TABLE credentials ADD COLUMN schema_id TEXT;\n"
        ),
        reversal_sql=(
            "ALTER TABLE credentials DROP COLUMN schema_id;\n"
            "ALTER TABLE credentials DROP COLUMN key_version;\n"
            "ALTER TABLE credentials DROP COLUMN expires_at;\n"
            "ALTER TABLE credentials DROP COLUMN rotated_at;\n"
            "ALTER TABLE credentials DROP COLUMN scope;\n"
        ),
        description=(
            "Z6 T2A: credentials.scope/rotated_at/expires_at/key_version/"
            "schema_id columns hoisted from encrypted envelope so vault "
            "metadata is queryable without decryption."
        ),
    )

    # ── Z6 T2C: credential_access_log audit table ──────────────────────────
    # One row per get/store/rotate/delete on credential_store. Captures
    # mission/task/agent/model context via ContextVar so we can prove who
    # pulled a secret when. `success=0` rows include `error` text.
    await apply_migration(
        version="2026-05-11-credential-access-log",
        sql=(
            "CREATE TABLE IF NOT EXISTS credential_access_log ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " service_name TEXT NOT NULL,"
            " mission_id INTEGER,"
            " task_id INTEGER,"
            " agent TEXT,"
            " model_id TEXT,"
            " action TEXT NOT NULL,"
            " scope TEXT,"
            " success INTEGER NOT NULL,"
            " error TEXT,"
            " accessed_at TEXT NOT NULL"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_cred_log_service "
            "ON credential_access_log(service_name);\n"
            "CREATE INDEX IF NOT EXISTS idx_cred_log_mission "
            "ON credential_access_log(mission_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_cred_log_mission;\n"
            "DROP INDEX IF EXISTS idx_cred_log_service;\n"
            "DROP TABLE IF EXISTS credential_access_log;\n"
        ),
        description=(
            "Z6 T2C: credential_access_log audit trail. "
            "Every get/store/rotate/delete is logged with mission/task/"
            "agent/model context from ContextVar."
        ),
    )

    # ── Z8 T3A: webhook_events dedup table ────────────────────────────────
    # Tracks vendor webhook deliveries. Primary key (integration_id, event_id)
    # makes duplicate POSTs a no-op. payload_hash captured for forensics.
    await apply_migration(
        version="2026-05-12-webhook-events",
        sql=(
            "CREATE TABLE IF NOT EXISTS webhook_events ("
            " integration_id TEXT NOT NULL,"
            " event_id TEXT NOT NULL,"
            " received_at TEXT NOT NULL,"
            " payload_hash TEXT NOT NULL,"
            " mission_id INTEGER,"
            " processed_at TEXT,"
            " PRIMARY KEY (integration_id, event_id)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_webhook_received "
            "ON webhook_events(received_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_webhook_received;\n"
            "DROP TABLE IF EXISTS webhook_events;\n"
        ),
        description=(
            "Z8 T3A: webhook_events dedup table — (integration_id, event_id) "
            "PK + received_at index. Records every vendor webhook delivery."
        ),
    )

    # ── Z8 T3E: integration_mappings — webhook → mission routing ─────────
    # Maps an (integration_id, product_id) pair to a long-lived ongoing
    # mission. Product_id NULL = catch-all for the integration. The
    # webhook listener prefers the most specific match (non-NULL
    # product_id) when both rows exist.
    await apply_migration(
        version="2026-05-12-integration-mappings",
        sql=(
            "CREATE TABLE IF NOT EXISTS integration_mappings ("
            " integration_id TEXT NOT NULL,"
            " product_id TEXT,"
            " mission_id INTEGER NOT NULL,"
            " created_at TEXT DEFAULT CURRENT_TIMESTAMP,"
            " PRIMARY KEY (integration_id, product_id, mission_id),"
            " FOREIGN KEY (mission_id) REFERENCES missions(id)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_integration_mappings_integ "
            "ON integration_mappings(integration_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_integration_mappings_integ;\n"
            "DROP TABLE IF EXISTS integration_mappings;\n"
        ),
        description=(
            "Z8 T3E: integration_mappings table — vendor webhooks route "
            "to an ongoing mission scoped by optional product_id. NULL "
            "product_id is the catch-all; specific product wins."
        ),
    )

    # ── Z3 T1C: review_density_json — founder dials per mission ───────────────
    # Stores a JSON blob of ReviewDensityDials fields.  NULL = all defaults
    # (conservative: standard/off/False/standard).  No backfill needed — NULL
    # rows just return defaults from get_dials().
    await apply_migration(
        version="2026-05-12-missions-review-density",
        sql="ALTER TABLE missions ADD COLUMN review_density_json TEXT;\n",
        reversal_sql="ALTER TABLE missions DROP COLUMN review_density_json;\n",
        description=(
            "Z3 T1C: missions.review_density_json stores founder-dial JSON "
            "(qa_dial, accessibility_dial, multi_file_expansion, "
            "integration_replay).  NULL = all defaults."
        ),
    )

    # Z8 T4B — action_cooldowns: per-(mission, verb) rate-limit ledger for
    # the on-call agent (rollback ≤ 2/hr, key-rotate ≤ 1/24h, etc.).
    await apply_migration(
        version="2026-05-12-action-cooldowns",
        sql=(
            "CREATE TABLE action_cooldowns ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER NOT NULL,"
            " verb TEXT NOT NULL,"
            " invoked_at TIMESTAMP NOT NULL,"
            " outcome TEXT"
            ");\n"
            "CREATE INDEX idx_cooldown_lookup ON action_cooldowns"
            "(mission_id, verb, invoked_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_cooldown_lookup;\n"
            "DROP TABLE IF EXISTS action_cooldowns;\n"
        ),
        description=(
            "Z8 T4B: action_cooldowns ledger for oncall_agent per-verb "
            "rate limiting (Mr. Roboto pre-execute enforcement)."
        ),
    )

    # Z8 T4D — escalation_policy: per-mission tier→channel routing
    # with quiet-hours support.
    await apply_migration(
        version="2026-05-12-escalation-policy",
        sql=(
            "CREATE TABLE escalation_policy ("
            " mission_id INTEGER PRIMARY KEY,"
            " quiet_hours_start TEXT,"
            " quiet_hours_end TEXT,"
            " tier1_channel TEXT NOT NULL DEFAULT 'telegram',"
            " tier2_channel TEXT NOT NULL DEFAULT 'telegram',"
            " tier3_channel TEXT NOT NULL DEFAULT 'sms',"
            " tz TEXT DEFAULT 'UTC'"
            ");\n"
        ),
        reversal_sql="DROP TABLE IF EXISTS escalation_policy;\n",
        description=(
            "Z8 T4D: escalation_policy table for per-mission tier-routing + "
            "quiet-hours-aware channel dispatch."
        ),
    )

    # ── Z8 T5E: tickets table for tier-1 support RAG ────────────────────────
    # One row per user question. ``confidence`` is the agent's calibrated
    # confidence ∈ [0.0, 1.0]. ``escalated_to_founder`` flips when confidence
    # < 0.7 OR sentiment is angry/urgent — the inlet writes a
    # ``founder_action(kind='support_escalation')`` and points its
    # ``founder_action_id`` here for follow-up.
    await apply_migration(
        version="2026-05-12-tickets",
        sql=(
            "CREATE TABLE IF NOT EXISTS tickets ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER,"
            " user_id TEXT NOT NULL,"
            " question TEXT NOT NULL,"
            " answer TEXT,"
            " confidence REAL,"
            " status TEXT NOT NULL DEFAULT 'open',"
            " escalated_to_founder INTEGER NOT NULL DEFAULT 0,"
            " founder_action_id INTEGER,"
            " sentiment TEXT,"
            " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);\n"
            "CREATE INDEX IF NOT EXISTS idx_tickets_user_id ON tickets(user_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_tickets_user_id;\n"
            "DROP INDEX IF EXISTS idx_tickets_status;\n"
            "DROP TABLE IF EXISTS tickets;\n"
        ),
        description=(
            "Z8 T5E: tickets table for tier-1 support RAG; confidence + "
            "escalation tracking."
        ),
    )

    # ── Z8 T5F: perf_baselines for synthetic-check regression diff ──────────
    # One row per release_tag + metric. Synthetic checks compare current
    # p50/p95/p99 against the last green baseline; >10% delta marks the
    # mission task as ``regression_detected`` and surfaces a rollback verb.
    await apply_migration(
        version="2026-05-12-perf-baselines",
        sql=(
            "CREATE TABLE IF NOT EXISTS perf_baselines ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id INTEGER NOT NULL,"
            " release_tag TEXT NOT NULL,"
            " metric TEXT NOT NULL,"
            " p50 REAL,"
            " p95 REAL,"
            " p99 REAL,"
            " recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_perf_baselines_mission_metric "
            "ON perf_baselines(mission_id, metric, recorded_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_perf_baselines_mission_metric;\n"
            "DROP TABLE IF EXISTS perf_baselines;\n"
        ),
        description=(
            "Z8 T5F: perf_baselines for synthetic-check regression diff."
        ),
    )

    # ── Z9 T1A: growth zone foundation — hypotheses / experiment_variants /
    # growth_events ──────────────────────────────────────────────────────────
    # Three tables backing the Z9 growth zone (analytics, hypothesis verdict
    # loop, A/B experiments). Kept separate from registry_events so Z8 oncall
    # queries stay clean. Timestamp columns store SQLite space-separated
    # datetime ('YYYY-MM-DD HH:MM:SS') — never datetime.isoformat() (T-form);
    # see CLAUDE.md scheduled_tasks pitfall.
    #
    # hypotheses: prediction-side store. predicted_json={metric,direction,
    #   magnitude}; actual_json gets {..,p_value} once a verdict is recorded.
    #   dedup_key = feature_slug + metric_name; refuted verdicts set
    #   suppressed_until = now + 90d so the same pair isn't re-predicted.
    await apply_migration(
        version="2026-05-15-z9-hypotheses",
        sql=(
            "CREATE TABLE IF NOT EXISTS hypotheses ("
            " id              INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id      INTEGER,"
            " feature         TEXT,"
            " predicted_json  TEXT,"
            " actual_json     TEXT,"
            " verdict         TEXT    NOT NULL DEFAULT 'pending',"
            " window_seconds  INTEGER,"
            " measured_at     TEXT,"
            " dedup_key       TEXT,"
            " suppressed_until TEXT,"
            " created_at      TEXT    DEFAULT (datetime('now')),"
            " FOREIGN KEY (mission_id) REFERENCES missions(id)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_hypotheses_mission "
            "ON hypotheses(mission_id);\n"
            "CREATE INDEX IF NOT EXISTS idx_hypotheses_dedup_key "
            "ON hypotheses(dedup_key);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_hypotheses_dedup_key;\n"
            "DROP INDEX IF EXISTS idx_hypotheses_mission;\n"
            "DROP TABLE IF EXISTS hypotheses;\n"
        ),
        description=(
            "Z9 T1A: hypotheses table — prediction-side store for the growth "
            "hypothesis/verdict loop. verdict pending|confirmed|refuted|"
            "inconclusive; refuted pairs suppressed_until now+90d."
        ),
    )

    # experiment_variants: A/B variant ledger (populated from T5). status
    #   active|winner|loser|stopped; retired_at set when status leaves active.
    await apply_migration(
        version="2026-05-15-z9-experiment-variants",
        sql=(
            "CREATE TABLE IF NOT EXISTS experiment_variants ("
            " id              INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id      INTEGER,"
            " hypothesis_id   INTEGER,"
            " variant_name    TEXT,"
            " assignment_rule TEXT,"
            " status          TEXT    NOT NULL DEFAULT 'active',"
            " shipped_at      TEXT,"
            " retired_at      TEXT,"
            " created_at      TEXT    DEFAULT (datetime('now')),"
            " FOREIGN KEY (mission_id) REFERENCES missions(id),"
            " FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_experiment_variants_mission "
            "ON experiment_variants(mission_id);\n"
            "CREATE INDEX IF NOT EXISTS idx_experiment_variants_hypothesis "
            "ON experiment_variants(hypothesis_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_experiment_variants_hypothesis;\n"
            "DROP INDEX IF EXISTS idx_experiment_variants_mission;\n"
            "DROP TABLE IF EXISTS experiment_variants;\n"
        ),
        description=(
            "Z9 T1A: experiment_variants table — A/B variant ledger. "
            "status active|winner|loser|stopped; populated by Z9 T5."
        ),
    )

    # growth_events: append-only growth telemetry log. kind is free-form
    #   ('metric_emit', 'backlog_candidate', 'dlq_pattern', 'verdict', ...);
    #   properties_json holds the kind-specific payload; segment is nullable.
    await apply_migration(
        version="2026-05-15-z9-growth-events",
        sql=(
            "CREATE TABLE IF NOT EXISTS growth_events ("
            " id              INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mission_id      INTEGER,"
            " kind            TEXT,"
            " properties_json TEXT,"
            " segment         TEXT,"
            " occurred_at     TEXT    DEFAULT (datetime('now')),"
            " FOREIGN KEY (mission_id) REFERENCES missions(id)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_growth_events_mission_occurred "
            "ON growth_events(mission_id, occurred_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_growth_events_kind "
            "ON growth_events(kind);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_growth_events_kind;\n"
            "DROP INDEX IF EXISTS idx_growth_events_mission_occurred;\n"
            "DROP TABLE IF EXISTS growth_events;\n"
        ),
        description=(
            "Z9 T1A: growth_events table — append-only growth telemetry log "
            "(metric_emit / backlog_candidate / dlq_pattern / verdict). "
            "Separate from registry_events to keep Z8 oncall queries clean."
        ),
    )

    # ── Z7 T1.0: humanish-layers foundation schema ────────────────────────────
    # product_id convention: TEXT column = root mission_id of the i2p mission
    # that produced the product. No formal product entity exists; this is an
    # app-level FK (TEXT, not INTEGER, to future-proof non-sequential IDs).

    # 1. mission_briefings — daily/completion briefings summarising what the
    #    system did so the founder can stay informed without reading raw logs.
    await apply_migration(
        version="2026-05-15-z7-mission-briefings",
        sql=(
            "CREATE TABLE IF NOT EXISTS mission_briefings ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            # product_id = root i2p mission id (TEXT; app-level FK to missions.id)
            " product_id TEXT NOT NULL,"
            " mission_id TEXT,"
            " kind TEXT NOT NULL,"        # 'completion' | 'daily'
            " body_md TEXT,"
            " founder_minutes_saved_estimate INTEGER,"
            " prepared_at TIMESTAMP,"
            " read_at TIMESTAMP,"         # nullable — NULL = not yet read
            " acted_on TEXT"              # JSON blob, nullable
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_mission_briefings_product_prepared "
            "ON mission_briefings(product_id, prepared_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_mission_briefings_product_prepared;\n"
            "DROP TABLE IF EXISTS mission_briefings;\n"
        ),
        description=(
            "Z7 T1.0: mission_briefings table (kind=completion|daily). "
            "product_id = root i2p mission id (TEXT app-level FK). "
            "founder_minutes_saved_estimate tracks ROI of automation."
        ),
    )

    # 2. external_comms_log — immutable audit trail for every outbound
    #    communication (email, tweet, blog post, press release, etc.).
    #    content_md stores gzip+base64 of the body to keep the row compact.
    #    reversibility mirrors founder_actions (full|partial|irreversible).
    await apply_migration(
        version="2026-05-15-z7-external-comms-log",
        sql=(
            "CREATE TABLE IF NOT EXISTS external_comms_log ("
            " log_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id TEXT NOT NULL,"
            " sent_at TIMESTAMP,"
            " channel TEXT NOT NULL,"     # see VALID_CHANNELS in app code
            " recipient TEXT,"            # nullable (broadcast channels)
            " recipient_count INTEGER DEFAULT 1,"
            " content_hash TEXT,"
            " content_md TEXT,"           # gzip+base64 of body
            " source_mission_id INTEGER,"  # FK missions.id (app-level)
            " source_action_id INTEGER,"
            " vendor_call_id INTEGER,"    # FK action_confirmations.id (app-level)
            " reversibility TEXT,"        # full|partial|irreversible
            " revoked_at TIMESTAMP,"
            " revoke_reason TEXT"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_external_comms_log_product_sent "
            "ON external_comms_log(product_id, sent_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_external_comms_log_content_hash "
            "ON external_comms_log(content_hash);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_external_comms_log_content_hash;\n"
            "DROP INDEX IF EXISTS idx_external_comms_log_product_sent;\n"
            "DROP TABLE IF EXISTS external_comms_log;\n"
        ),
        description=(
            "Z7 T1.0: external_comms_log — immutable audit trail for outbound "
            "comms (email/tweet/reddit_post/hn_post/ph_post/linkedin_post/"
            "press_release/blog_post/sms/webhook). content_md=gzip+b64 body."
        ),
    )

    # 3. mission_events: add founder_minutes_saved column.
    await apply_migration(
        version="2026-05-15-z7-mission-events-minutes-saved",
        sql=(
            "ALTER TABLE mission_events "
            "ADD COLUMN founder_minutes_saved INTEGER;\n"
        ),
        reversal_sql=(
            "ALTER TABLE mission_events "
            "DROP COLUMN founder_minutes_saved;\n"
        ),
        description=(
            "Z7 T1.0: mission_events.founder_minutes_saved (nullable INTEGER). "
            "Set by the briefing_compose posthook after analysing what was done."
        ),
    )

    # 4. founder_actions: add priority / defer_until / expires_at columns.
    #    priority valid values: p0_blocking / p1_today / p2_this_week /
    #    p3_when_idle — enforced by app code, not DB constraint (consistent
    #    with repo style; see founder_actions module for the validation layer).
    await apply_migration(
        version="2026-05-15-z7-founder-actions-priority-defer",
        sql=(
            "ALTER TABLE founder_actions "
            "ADD COLUMN priority TEXT DEFAULT 'p2_this_week';\n"
            "ALTER TABLE founder_actions "
            "ADD COLUMN defer_until TIMESTAMP;\n"
            "ALTER TABLE founder_actions "
            "ADD COLUMN expires_at TIMESTAMP;\n"
        ),
        reversal_sql=(
            "ALTER TABLE founder_actions DROP COLUMN expires_at;\n"
            "ALTER TABLE founder_actions DROP COLUMN defer_until;\n"
            "ALTER TABLE founder_actions DROP COLUMN priority;\n"
        ),
        description=(
            "Z7 T1.0: founder_actions.priority (p0_blocking|p1_today|"
            "p2_this_week|p3_when_idle, default p2_this_week) + defer_until "
            "(snooze timestamp) + expires_at (auto-cancel deadline). "
            "Priority validated by app code, not CHECK constraint."
        ),
    )

    # 5. founder_attention_log: add card_id / surfaced_at / acted_at /
    #    deferred_to / attention_minutes columns.
    #    card_id is an app-level FK to founder_actions.id; INTEGER, nullable
    #    (legacy rows pre-date this column and have NULL card_id).
    await apply_migration(
        version="2026-05-15-z7-founder-attention-log-card",
        sql=(
            "ALTER TABLE founder_attention_log "
            "ADD COLUMN card_id INTEGER;\n"           # FK founder_actions.id
            "ALTER TABLE founder_attention_log "
            "ADD COLUMN surfaced_at TIMESTAMP;\n"
            "ALTER TABLE founder_attention_log "
            "ADD COLUMN acted_at TIMESTAMP;\n"        # nullable
            "ALTER TABLE founder_attention_log "
            "ADD COLUMN deferred_to TIMESTAMP;\n"     # nullable
            "ALTER TABLE founder_attention_log "
            "ADD COLUMN attention_minutes INTEGER;\n"  # nullable
        ),
        reversal_sql=(
            "ALTER TABLE founder_attention_log DROP COLUMN attention_minutes;\n"
            "ALTER TABLE founder_attention_log DROP COLUMN deferred_to;\n"
            "ALTER TABLE founder_attention_log DROP COLUMN acted_at;\n"
            "ALTER TABLE founder_attention_log DROP COLUMN surfaced_at;\n"
            "ALTER TABLE founder_attention_log DROP COLUMN card_id;\n"
        ),
        description=(
            "Z7 T1.0: founder_attention_log extended for attention-UX: "
            "card_id (FK founder_actions), surfaced_at, acted_at, deferred_to, "
            "attention_minutes. Legacy rows remain valid (NULL in new columns)."
        ),
    )

    # ── Z7 T2A: email-send shared service tables ─────────────────────────────

    # product_email_config: per-product email provider configuration.
    #   provider: 'brevo' | 'resend' | 'postmark' | 'ses'
    #   from_domain: sender domain (e.g. 'example.com')
    #   api_key_ref: key name in the credential_store (not the key itself)
    #   monthly_quota: max emails/month (NULL = unlimited)
    #   tier: 'free' | 'paid'  — default 'free' per founder decision 2026-05-15
    await apply_migration(
        version="2026-05-15-z7-product-email-config",
        sql=(
            "CREATE TABLE IF NOT EXISTS product_email_config ("
            " id          INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id  TEXT NOT NULL UNIQUE,"
            " provider    TEXT NOT NULL DEFAULT 'brevo',"
            " from_domain TEXT,"
            " api_key_ref TEXT,"
            " monthly_quota INTEGER,"
            " tier        TEXT NOT NULL DEFAULT 'free',"
            " created_at  TIMESTAMP DEFAULT (datetime('now')),"
            " updated_at  TIMESTAMP DEFAULT (datetime('now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_product_email_config_product "
            "ON product_email_config(product_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_product_email_config_product;\n"
            "DROP TABLE IF EXISTS product_email_config;\n"
        ),
        description=(
            "Z7 T2A: product_email_config — per-product email provider config "
            "(provider=brevo|resend|postmark|ses, tier=free|paid, api_key_ref → "
            "credential_store, monthly_quota guard)."
        ),
    )

    # email_events: immutable log of every email event (sent / open / click /
    #   bounce / unsub / complaint / delivery).  Quota counting queries this
    #   table for event_type='sent' within the current calendar month.
    await apply_migration(
        version="2026-05-15-z7-email-events",
        sql=(
            "CREATE TABLE IF NOT EXISTS email_events ("
            " id          INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id  TEXT NOT NULL,"
            " event_type  TEXT NOT NULL,"   # sent|open|click|bounce|unsub|complaint|delivery
            " recipient   TEXT NOT NULL,"
            " provider    TEXT,"
            " message_id  TEXT,"
            " occurred_at TIMESTAMP DEFAULT (datetime('now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_email_events_product_type_occurred "
            "ON email_events(product_id, event_type, occurred_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_email_events_recipient "
            "ON email_events(recipient);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_email_events_recipient;\n"
            "DROP INDEX IF EXISTS idx_email_events_product_type_occurred;\n"
            "DROP TABLE IF EXISTS email_events;\n"
        ),
        description=(
            "Z7 T2A: email_events — append-only log of email send + webhook "
            "events (sent/open/click/bounce/unsub/complaint/delivery). "
            "Quota guard counts event_type='sent' for the current month."
        ),
    )

    # email_suppression: per-product suppression list.
    #   Recipients added here are silently skipped by service.send_email.
    #   Populated automatically from bounce / complaint / unsub webhooks.
    await apply_migration(
        version="2026-05-15-z7-email-suppression",
        sql=(
            "CREATE TABLE IF NOT EXISTS email_suppression ("
            " id         INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id TEXT NOT NULL,"
            " email      TEXT NOT NULL,"
            " reason     TEXT NOT NULL,"   # bounce|complaint|unsub|manual
            " added_at   TIMESTAMP DEFAULT (datetime('now')),"
            " UNIQUE(product_id, email)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_email_suppression_product_email "
            "ON email_suppression(product_id, email);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_email_suppression_product_email;\n"
            "DROP TABLE IF EXISTS email_suppression;\n"
        ),
        description=(
            "Z7 T2A: email_suppression — per-product suppression list "
            "(bounce/complaint/unsub/manual). UNIQUE(product_id, email) prevents "
            "duplicates. service.send_email checks before any send."
        ),
    )

    # ── Z7 T3C: press_kits — versioned binary store (A4 + A4.r1) ────────────
    # One row per assembled press kit. manifest_json carries the full manifest
    # including 4 audience variants (investor/journalist/partner/candidate).
    # published_url is the canonical permanent URL prefix for this version;
    # per-audience URLs are derived as published_url + "/{audience}/".
    # product_id is NOT NULL per-product scoping (founder decision 2026-05-15).
    await apply_migration(
        version="2026-05-15-z7-press-kits",
        sql=(
            "CREATE TABLE IF NOT EXISTS press_kits ("
            " kit_id      INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id  TEXT NOT NULL,"
            " version     INTEGER NOT NULL,"
            " mission_id  INTEGER,"
            " manifest_json TEXT NOT NULL,"
            " published_url TEXT,"
            " created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_press_kits_product_version "
            "ON press_kits(product_id, version);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_press_kits_product_version;\n"
            "DROP TABLE IF EXISTS press_kits;\n"
        ),
        description=(
            "Z7 T3C (A4): press_kits versioned binary store. One row per assembled "
            "kit; manifest_json holds 4 audience variants (investor/journalist/"
            "partner/candidate). published_url = permanent URL prefix; audience "
            "URLs = published_url/{audience}/. product_id NOT NULL (per-product "
            "scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T3C: press_kit_quotes — harvested quotes for press kits ───────────
    # Quotes sourced from customer interviews (B7), reviews (B8), past mentions
    # (A11), or manually entered. approved flag controls inclusion in kit zip.
    # source_kind: 'interview' | 'review' | 'mention' | 'manual'.
    # kit_id is nullable — quotes harvested before assembly have kit_id=NULL.
    await apply_migration(
        version="2026-05-15-z7-press-kit-quotes",
        sql=(
            "CREATE TABLE IF NOT EXISTS press_kit_quotes ("
            " quote_id    INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id  TEXT NOT NULL,"
            " kit_id      INTEGER,"
            " source_kind TEXT NOT NULL,"
            " speaker     TEXT,"
            " body        TEXT NOT NULL,"
            " approved    INTEGER NOT NULL DEFAULT 0,"
            " created_at  TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_press_kit_quotes_product "
            "ON press_kit_quotes(product_id, approved);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_press_kit_quotes_product;\n"
            "DROP TABLE IF EXISTS press_kit_quotes;\n"
        ),
        description=(
            "Z7 T3C (A4): press_kit_quotes — harvested quotes for press kit variants. "
            "source_kind=interview|review|mention|manual. approved=1 means included "
            "in kit zip. kit_id nullable (NULL = not yet assigned to a version)."
        ),
    )

    # ── Z7 T3D: incidents — customer-facing incident tracking (B3) ───────────
    # product_id NOT NULL per-product scoping (founder decision 2026-05-15).
    # severity: 'critical' | 'major' | 'minor'
    # resolved_at NULL means incident is still open.
    # postmortem_url filled in by founder after publishing the postmortem (7d SLA).
    await apply_migration(
        version="2026-05-15-z7-incidents",
        sql=(
            "CREATE TABLE IF NOT EXISTS incidents ("
            " incident_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id  TEXT NOT NULL,"
            " opened_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " resolved_at TEXT,"
            " severity    TEXT NOT NULL DEFAULT 'minor',"
            " affected_components_json TEXT NOT NULL DEFAULT '[]',"
            " customer_impact_summary TEXT,"
            " current_status_md TEXT,"
            " postmortem_url TEXT"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_incidents_product_open "
            "ON incidents(product_id, resolved_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_incidents_product_open;\n"
            "DROP TABLE IF EXISTS incidents;\n"
        ),
        description=(
            "Z7 T3D (B3): incidents — per-product customer-facing incident log. "
            "resolved_at NULL = open. severity in critical|major|minor. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T3D: status_updates — per-incident public status update feed (B3) ─
    # status_kind: 'investigating' | 'identified' | 'monitoring' | 'resolved'
    # Each row is one customer-facing status update, reviewed by founder
    # before publish via incident/publish_status verb.
    await apply_migration(
        version="2026-05-15-z7-status-updates",
        sql=(
            "CREATE TABLE IF NOT EXISTS status_updates ("
            " update_id   INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id  TEXT NOT NULL,"
            " incident_id INTEGER NOT NULL,"
            " posted_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " body_md     TEXT NOT NULL,"
            " status_kind TEXT NOT NULL DEFAULT 'investigating'"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_status_updates_incident "
            "ON status_updates(incident_id, posted_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_status_updates_product "
            "ON status_updates(product_id, posted_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_status_updates_incident;\n"
            "DROP INDEX IF EXISTS idx_status_updates_product;\n"
            "DROP TABLE IF EXISTS status_updates;\n"
        ),
        description=(
            "Z7 T3D (B3): status_updates — per-incident customer-facing status "
            "update log. status_kind in investigating|identified|monitoring|resolved. "
            "Each row published after founder review via incident/publish_status. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T3E (B6): crisis_events ────────────────────────────────────────────
    await apply_migration(
        version="2026-05-15-z7-crisis-events",
        sql=(
            "CREATE TABLE IF NOT EXISTS crisis_events ("
            " event_id      INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id    TEXT NOT NULL,"
            " opened_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " tier          INTEGER NOT NULL CHECK(tier IN (1,2,3,4)),"
            " source        TEXT NOT NULL DEFAULT 'manual',"
            " summary       TEXT NOT NULL DEFAULT '',"
            " status        TEXT NOT NULL DEFAULT 'active',"
            " resolved_at   TEXT,"
            " postmortem_url TEXT"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_crisis_events_product "
            "ON crisis_events(product_id, status);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_crisis_events_product;\n"
            "DROP TABLE IF EXISTS crisis_events;\n"
        ),
        description=(
            "Z7 T3E (B6): crisis_events — per-product crisis lifecycle table. "
            "tier in 1-4 (brand misstep / outage / security breach / existential). "
            "source in mention_monitor|incident|manual. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15). "
            "Freeze scoped per-product; other products keep running."
        ),
    )

    # ── Z7 T3E (B6): marketing_freeze ────────────────────────────────────────
    await apply_migration(
        version="2026-05-15-z7-marketing-freeze",
        sql=(
            "CREATE TABLE IF NOT EXISTS marketing_freeze ("
            " freeze_id   INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id  TEXT NOT NULL,"
            " event_id    INTEGER NOT NULL,"
            " frozen_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " resumed_at  TEXT"
            ");\n"
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_marketing_freeze_product_event "
            "ON marketing_freeze(product_id, event_id) WHERE resumed_at IS NULL;\n"
            "CREATE INDEX IF NOT EXISTS idx_marketing_freeze_product "
            "ON marketing_freeze(product_id, resumed_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_marketing_freeze_product_event;\n"
            "DROP INDEX IF EXISTS idx_marketing_freeze_product;\n"
            "DROP TABLE IF EXISTS marketing_freeze;\n"
        ),
        description=(
            "Z7 T3E (B6): marketing_freeze — per-product freeze flag for A2/B1/A7. "
            "Downstream subsystems check is_marketing_frozen(product_id) before "
            "proceeding with launches, announcement sends, or outreach. "
            "resumed_at=NULL means currently frozen. Set by crisis/freeze_marketing, "
            "cleared by crisis/resume (sets resumed_at=now). "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T3A (A2): launches — per-product launch mission state ─────────────
    await apply_migration(
        version="2026-05-16-z7-launches",
        sql=(
            "CREATE TABLE IF NOT EXISTS launches ("
            " launch_id            INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id           TEXT NOT NULL,"
            " scheduled_publish_at TEXT NOT NULL,"
            " status               TEXT NOT NULL DEFAULT 'planned',"
            " channels_json        TEXT NOT NULL DEFAULT '[]',"
            " mission_id           INTEGER,"
            " created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_launches_product "
            "ON launches(product_id, status);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_launches_product;\n"
            "DROP TABLE IF EXISTS launches;\n"
        ),
        description=(
            "Z7 T3A (A2): launches — per-product launch mission state. "
            "Tracks scheduled_publish_at (phase clock anchor), status "
            "(planned|live|concluded), channels as JSON list, and the "
            "mission_id of the spawned launch_playbook mission. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T4 A10: relationships — CRM contact directory ─────────────────────
    # Telegram-native interaction log. NOT a relationship graph; no email
    # integration. category valid values enforced at app level (see crm.py).
    # product_id NOT NULL per-product scoping (founder decision 2026-05-15).
    await apply_migration(
        version="2026-05-16-z7-relationships",
        sql=(
            "CREATE TABLE IF NOT EXISTS relationships ("
            " contact_id   INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id   TEXT NOT NULL,"
            " handle       TEXT NOT NULL,"             # @telegram_handle or name-slug
            " display_name TEXT NOT NULL,"
            " category     TEXT NOT NULL DEFAULT 'other',"
            # customer|prospect|investor|journalist|partner|advisor|candidate|vendor|other
            " email        TEXT,"
            " links_json   TEXT,"                      # JSON array of URLs
            " notes_md     TEXT,"
            " created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_product_handle "
            "ON relationships(product_id, handle);\n"
            "CREATE INDEX IF NOT EXISTS idx_relationships_product_category "
            "ON relationships(product_id, category);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_relationships_product_handle;\n"
            "DROP INDEX IF EXISTS idx_relationships_product_category;\n"
            "DROP TABLE IF EXISTS relationships;\n"
        ),
        description=(
            "Z7 T4 A10: relationships — Telegram-native contact directory. "
            "contact_id PK, product_id NOT NULL, handle UNIQUE per product, "
            "category: customer|prospect|investor|journalist|partner|advisor|"
            "candidate|vendor|other. B4 (meeting brief) and B7 (interviews) "
            "write interactions rows; this table is the contact registry."
        ),
    )

    # ── Z7 T4 A10: interactions — structured interaction log ─────────────────
    # kind valid values: call|email|meeting|message|event|interview|other.
    # follow_up_at nullable; done=0 means pending, done=1 means completed.
    # B4 writes rows at meeting end; B7 writes rows with kind='interview'.
    await apply_migration(
        version="2026-05-16-z7-interactions",
        sql=(
            "CREATE TABLE IF NOT EXISTS interactions ("
            " interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id     TEXT NOT NULL,"
            " contact_id     INTEGER NOT NULL,"
            " kind           TEXT NOT NULL DEFAULT 'other',"
            " summary        TEXT NOT NULL DEFAULT '',"
            " next_action    TEXT,"
            " follow_up_at   TEXT,"                    # nullable; "%Y-%m-%d %H:%M:%S"
            " logged_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " mission_id     INTEGER,"                 # nullable FK to missions.id
            " done           INTEGER NOT NULL DEFAULT 0"  # 0=pending, 1=done
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_interactions_product_contact "
            "ON interactions(product_id, contact_id);\n"
            "CREATE INDEX IF NOT EXISTS idx_interactions_follow_up "
            "ON interactions(product_id, follow_up_at, done) "
            "WHERE follow_up_at IS NOT NULL;\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_interactions_product_contact;\n"
            "DROP INDEX IF EXISTS idx_interactions_follow_up;\n"
            "DROP TABLE IF EXISTS interactions;\n"
        ),
        description=(
            "Z7 T4 A10: interactions — structured interaction log. "
            "kind: call|email|meeting|message|event|interview|other. "
            "follow_up_at nullable parsed from relative windows (2w/3d/1m). "
            "done flag for follow-up reminder sweep. "
            "B4 meeting-brief creates rows at outcome-log time; "
            "B7 interview pipeline creates rows with kind='interview'."
        ),
    )

    # ── Z7 T4 A10.r1: consent_records — per-purpose consent ledger ───────────
    # purpose valid values: quote_use|data_processing|marketing_email|
    #   interview_recording|case_study.
    # revoked_at NULL = consent active (if not expired).
    # expires_at NULL = no expiry (permanent unless revoked).
    await apply_migration(
        version="2026-05-16-z7-consent-records",
        sql=(
            "CREATE TABLE IF NOT EXISTS consent_records ("
            " consent_id          INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id          TEXT NOT NULL,"
            " contact_id          INTEGER NOT NULL,"
            " purpose             TEXT NOT NULL,"
            # quote_use|data_processing|marketing_email|interview_recording|case_study
            " granted_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " expires_at          TEXT,"                # nullable; "%Y-%m-%d %H:%M:%S"
            " source_evidence_url TEXT,"               # URL proving consent was collected
            " revoked_at          TEXT"                # nullable; set on revoke
            ");\n"
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_consent_records_unique "
            "ON consent_records(product_id, contact_id, purpose);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_consent_records_unique;\n"
            "DROP TABLE IF EXISTS consent_records;\n"
        ),
        description=(
            "Z7 T4 A10.r1: consent_records — per-purpose consent ledger. "
            "purpose: quote_use|data_processing|marketing_email|"
            "interview_recording|case_study. "
            "revoked_at NULL + not expired = valid consent (has_consent()). "
            "Every Z7 surface checks has_consent() before touching contact data. "
            "B7 interview pipeline checks interview_recording consent before transcribing."
        ),
    )

    # ── Z7 T4 B4: meetings — meeting brief auto-generation ───────────────────
    # meeting_id PK, product_id NOT NULL per-product scoping.
    # contact_id references relationships(contact_id) at app level (no FK constraint
    # — consistent with repo style).
    # brief_generated_at: set when the brief job fires; NULL means not yet generated.
    # brief_md: the structured Markdown brief surfaced to founder.
    # outcome_logged_interaction_id: FK to interactions; NULL until founder logs outcome.
    await apply_migration(
        version="2026-05-16-z7-meetings",
        sql=(
            "CREATE TABLE IF NOT EXISTS meetings ("
            " meeting_id                     INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id                     TEXT NOT NULL,"
            " contact_id                     INTEGER,"
            " scheduled_for                  TEXT NOT NULL,"
            " purpose                        TEXT NOT NULL DEFAULT '',"
            " brief_generated_at             TEXT,"   # NULL = not yet generated
            " brief_md                       TEXT,"   # populated after brief/brief verb
            " outcome_logged_interaction_id  INTEGER"  # FK interactions; NULL = not logged
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_meetings_product_scheduled "
            "ON meetings(product_id, scheduled_for);\n"
            "CREATE INDEX IF NOT EXISTS idx_meetings_brief_pending "
            "ON meetings(scheduled_for, brief_generated_at) "
            "WHERE brief_generated_at IS NULL;\n"
            "CREATE INDEX IF NOT EXISTS idx_meetings_outcome_pending "
            "ON meetings(scheduled_for, outcome_logged_interaction_id) "
            "WHERE outcome_logged_interaction_id IS NULL;\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_meetings_product_scheduled;\n"
            "DROP INDEX IF EXISTS idx_meetings_brief_pending;\n"
            "DROP INDEX IF EXISTS idx_meetings_outcome_pending;\n"
            "DROP TABLE IF EXISTS meetings;\n"
        ),
        description=(
            "Z7 T4 B4: meetings — per-product meeting schedule + brief state. "
            "brief_generated_at NULL = brief not yet dispatched. "
            "outcome_logged_interaction_id NULL = outcome not yet logged. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15). "
            "contact_id references relationships at app level."
        ),
    )

    # ── Z7 T4 B7: interview_notes — customer interview / call notes pipeline ─────
    # Transcript stored as plain text (Markdown). Quotes / action items as JSON.
    # product_id NOT NULL per-product scoping (founder decision 2026-05-15).
    await apply_migration(
        version="2026-05-16-z7-interview-notes",
        sql=(
            "CREATE TABLE IF NOT EXISTS interview_notes ("
            " note_id            INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id         TEXT NOT NULL,"
            " contact_id         INTEGER,"
            " started_at         TEXT,"
            " duration_minutes   REAL,"
            " transcript_md      TEXT,"
            " summary_md         TEXT,"
            " quotes_json        TEXT,"
            " insights_md        TEXT,"
            " action_items_json  TEXT,"
            " audio_path         TEXT"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_interview_notes_product "
            "ON interview_notes(product_id, started_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_interview_notes_contact "
            "ON interview_notes(contact_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_interview_notes_contact;\n"
            "DROP INDEX IF EXISTS idx_interview_notes_product;\n"
            "DROP TABLE IF EXISTS interview_notes;\n"
        ),
        description=(
            "Z7 T4 B7: interview_notes — per-product customer interview pipeline. "
            "Transcript (Markdown), structured summary (bullets/quotes/insights/"
            "action_items as JSON), audio_path for source recording. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15). "
            "contact_id FK to relationships at app level."
        ),
    )

    # ── Z7 T4 A8: docs_gap_log — documentation gap tracker ──────────────────────
    # One row per escalation where no matching support_doc was found via semantic
    # search. Weekly faq_regen digest surfaces gap clusters; A0 briefing appends
    # a summary when gaps exist. product_id NOT NULL per-product scoping.
    # matched_doc_id nullable — NULL means no match found (the gap case).
    await apply_migration(
        version="2026-05-16-z7-docs-gap-log",
        sql=(
            "CREATE TABLE IF NOT EXISTS docs_gap_log ("
            " gap_id       INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id   TEXT NOT NULL,"
            " question     TEXT NOT NULL,"
            " matched_doc_id TEXT,"          # NULL = no match (the gap)
            " logged_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_docs_gap_log_product_logged "
            "ON docs_gap_log(product_id, logged_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_docs_gap_log_product_logged;\n"
            "DROP TABLE IF EXISTS docs_gap_log;\n"
        ),
        description=(
            "Z7 T4 A8: docs_gap_log — documentation gap tracker. One row per "
            "escalation where semantic search found no matching support_doc. "
            "matched_doc_id NULL = gap. Weekly faq_regen surfaces clusters; "
            "A0 briefing appends digest. product_id NOT NULL (per-product scoping)."
        ),
    )

    # ── Z7 T5 B1: email_templates — lifecycle email template store ─────────────
    # One row per email template. kind: onboarding|retention|churn_rescue|
    # transactional|announcement. status: draft|approved|archived.
    # brand_voice_lint_pass + copy_compliance_pass must both be 1 before founder
    # can approve (status → 'approved'). product_id NOT NULL (per-product scoping).
    await apply_migration(
        version="2026-05-16-z7-email-templates",
        sql=(
            "CREATE TABLE IF NOT EXISTS email_templates ("
            " template_id  INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id   TEXT NOT NULL,"
            " kind         TEXT NOT NULL DEFAULT 'onboarding',"
            " subject      TEXT NOT NULL DEFAULT '',"
            " body_md      TEXT NOT NULL DEFAULT '',"
            " variants_json TEXT NOT NULL DEFAULT '[]',"
            " status       TEXT NOT NULL DEFAULT 'draft',"
            " brand_voice_lint_pass INTEGER NOT NULL DEFAULT 0,"
            " copy_compliance_pass  INTEGER NOT NULL DEFAULT 0,"
            " created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_email_templates_product_kind "
            "ON email_templates(product_id, kind);\n"
            "CREATE INDEX IF NOT EXISTS idx_email_templates_status "
            "ON email_templates(product_id, status);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_email_templates_status;\n"
            "DROP INDEX IF EXISTS idx_email_templates_product_kind;\n"
            "DROP TABLE IF EXISTS email_templates;\n"
        ),
        description=(
            "Z7 T5 B1: email_templates — lifecycle email template store. "
            "kind=onboarding|retention|churn_rescue|transactional|announcement. "
            "status=draft|approved|archived. brand_voice_lint_pass + "
            "copy_compliance_pass must both=1 before founder approval. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T5 B1: email_sequences — ordered drip sequence definitions ──────────
    # steps_json: ordered list of {template_id, delay_hours}.  enabled=1 means
    # trigger_sequence will expand it; enabled=0 silently skips.
    # trigger_kind: signup|first_action|inactivity_7d|cancellation|payment_failed|
    #               announcement|manual (matches Z6 product event types).
    await apply_migration(
        version="2026-05-16-z7-email-sequences",
        sql=(
            "CREATE TABLE IF NOT EXISTS email_sequences ("
            " sequence_id  INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id   TEXT NOT NULL,"
            " name         TEXT NOT NULL DEFAULT '',"
            " trigger_kind TEXT NOT NULL DEFAULT 'manual',"
            " steps_json   TEXT NOT NULL DEFAULT '[]',"
            " enabled      INTEGER NOT NULL DEFAULT 1,"
            " created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_email_sequences_product_trigger "
            "ON email_sequences(product_id, trigger_kind);\n"
            "CREATE INDEX IF NOT EXISTS idx_email_sequences_enabled "
            "ON email_sequences(product_id, enabled);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_email_sequences_enabled;\n"
            "DROP INDEX IF EXISTS idx_email_sequences_product_trigger;\n"
            "DROP TABLE IF EXISTS email_sequences;\n"
        ),
        description=(
            "Z7 T5 B1: email_sequences — drip sequence definitions. "
            "steps_json = ordered list of {template_id, delay_hours}. "
            "trigger_kind = signup|first_action|inactivity_7d|cancellation|"
            "payment_failed|announcement|manual. product_id NOT NULL."
        ),
    )

    # ── Z7 T5 B1: email_sends — per-user per-step send schedule ───────────────
    # Populated by trigger_sequence from a sequence's steps_json.
    # lifecycle_email_send cron picks rows where scheduled_for <= now AND
    # sent_at IS NULL, calls send_email, marks sent_at on success.
    # Bounce/unsub/open/click event timestamps recorded from email webhooks.
    await apply_migration(
        version="2026-05-16-z7-email-sends",
        sql=(
            "CREATE TABLE IF NOT EXISTS email_sends ("
            " send_id       INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id    TEXT NOT NULL,"
            " user_id       TEXT NOT NULL,"
            " sequence_id   INTEGER,"
            " template_id   INTEGER,"
            " scheduled_for TEXT NOT NULL,"
            " sent_at       TEXT,"
            " opened_at     TEXT,"
            " clicked_at    TEXT,"
            " bounced_at    TEXT,"
            " unsubscribed_at TEXT,"
            " created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_email_sends_due "
            "ON email_sends(product_id, scheduled_for) WHERE sent_at IS NULL;\n"
            "CREATE INDEX IF NOT EXISTS idx_email_sends_sequence "
            "ON email_sends(sequence_id, user_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_email_sends_sequence;\n"
            "DROP INDEX IF EXISTS idx_email_sends_due;\n"
            "DROP TABLE IF EXISTS email_sends;\n"
        ),
        description=(
            "Z7 T5 B1: email_sends — per-user per-step send schedule. "
            "Populated by trigger_sequence. Cron picks scheduled_for<=now "
            "AND sent_at IS NULL. Timestamps for sent/opened/clicked/bounced/"
            "unsubscribed recorded from email webhooks. product_id NOT NULL."
        ),
    )

    # ── Z7 T5 B1: email_preferences — per-user preference center data ─────────
    # user_token is an opaque string (URL-safe, per-product unique) embedded in
    # unsubscribe links.  subscriptions_json: {"<sequence_id>": true|false}.
    # UNIQUE(product_id, user_token) so set_preferences can use INSERT OR REPLACE.
    await apply_migration(
        version="2026-05-16-z7-email-preferences",
        sql=(
            "CREATE TABLE IF NOT EXISTS email_preferences ("
            " pref_id           INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id        TEXT NOT NULL,"
            " user_token        TEXT NOT NULL,"
            " subscriptions_json TEXT NOT NULL DEFAULT '{}',"
            " updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " UNIQUE(product_id, user_token)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_email_preferences_product_token "
            "ON email_preferences(product_id, user_token);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_email_preferences_product_token;\n"
            "DROP TABLE IF EXISTS email_preferences;\n"
        ),
        description=(
            "Z7 T5 B1: email_preferences — per-user preference center. "
            "user_token = opaque URL-safe string embedded in unsubscribe links. "
            "subscriptions_json = {sequence_id: bool}. UNIQUE(product_id,user_token) "
            "supports upsert. product_id NOT NULL (per-product scoping)."
        ),
    )

    # ── Z7 T5 B2: changelog_entries — public changelog artifact ──────────────
    # One row per changelog entry. published=0 means draft; published=1 means
    # public (rendered on /changelog, /changelog.rss, in-app banner, email blast).
    # kind_breakdown_json: {"added": [...], "changed": [...], "fixed": [...],
    #                       "deprecated": [...], "removed": [...]}.
    # related_mission_ids_json: list of mission IDs whose commits are in this entry.
    # product_id NOT NULL (per-product scoping, founder decision 2026-05-15).
    await apply_migration(
        version="2026-05-16-z7-b2-changelog-entries",
        sql=(
            "CREATE TABLE IF NOT EXISTS changelog_entries ("
            " entry_id               INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id             TEXT    NOT NULL,"
            " version                TEXT    NOT NULL DEFAULT '',"
            " released_at            TEXT,"
            " title                  TEXT    NOT NULL DEFAULT '',"
            " body_md                TEXT    NOT NULL DEFAULT '',"
            " kind_breakdown_json    TEXT    NOT NULL DEFAULT '{}',"
            " shipped_features_json  TEXT    NOT NULL DEFAULT '[]',"
            " related_mission_ids_json TEXT  NOT NULL DEFAULT '[]',"
            " external_url           TEXT,"
            " published              INTEGER NOT NULL DEFAULT 0,"
            " created_at             TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " updated_at             TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_changelog_entries_product_published "
            "ON changelog_entries(product_id, published, released_at DESC);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_changelog_entries_product_published;\n"
            "DROP TABLE IF EXISTS changelog_entries;\n"
        ),
        description=(
            "Z7 T5 B2: changelog_entries — public changelog artifact table. "
            "published=0=draft, published=1=public. kind_breakdown_json holds "
            "Keep-A-Changelog buckets (added/changed/fixed/deprecated/removed). "
            "related_mission_ids_json links to source missions. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T5 B8: external_reviews — reviews harvest table ───────────────────
    # One row per review per platform. UNIQUE(platform, external_id) for dedup.
    # sentiment: 'positive' | 'negative' | 'neutral' | NULL (pre-classify).
    # theme_tag: 'UX' | 'pricing' | 'bug' | 'feature-request' | 'support' |
    #            'generic-positive' | 'generic-negative' | NULL (pre-classify).
    # replied_at / reply_body_md: NULL until founder approves + posts reply
    # (NEVER auto-replied — draft_reply produces drafts only).
    # product_id NOT NULL (per-product scoping, founder decision 2026-05-15).
    await apply_migration(
        version="2026-05-16-z7-b8-external-reviews",
        sql=(
            "CREATE TABLE IF NOT EXISTS external_reviews ("
            " review_id    INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id   TEXT    NOT NULL,"
            " platform     TEXT    NOT NULL,"
            " external_id  TEXT    NOT NULL,"
            " posted_at    TEXT,"
            " author       TEXT,"
            " rating       INTEGER,"
            " body_md      TEXT    NOT NULL DEFAULT '',"
            " sentiment    TEXT,"
            " theme_tag    TEXT,"
            " replied_at   TEXT,"
            " reply_body_md TEXT,"
            " created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))"
            ");\n"
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_external_reviews_platform_id "
            "ON external_reviews(platform, external_id);\n"
            "CREATE INDEX IF NOT EXISTS idx_external_reviews_product_platform "
            "ON external_reviews(product_id, platform, posted_at DESC);\n"
            "CREATE INDEX IF NOT EXISTS idx_external_reviews_rating "
            "ON external_reviews(product_id, rating);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_external_reviews_rating;\n"
            "DROP INDEX IF EXISTS idx_external_reviews_product_platform;\n"
            "DROP INDEX IF EXISTS uq_external_reviews_platform_id;\n"
            "DROP TABLE IF EXISTS external_reviews;\n"
        ),
        description=(
            "Z7 T5 B8: external_reviews — reviews harvest table. "
            "Polls G2/AppStore/PlayStore/ProductHunt/etc. daily; "
            "UNIQUE(platform, external_id) for dedup. sentiment + theme_tag "
            "set by reviews/classify. replied_at/reply_body_md NULL until "
            "founder manually approves reply — never auto-replied. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T6 A7: outreach_warmup — ramp-curve enforcement ───────────────────────
    # One row per (product_id, domain, day) combination. Warmup curve: day1=50/day
    # ramping to day14=500/day. outreach/send checks sent_count < target_count before
    # dispatching. product_id NOT NULL (per-product scoping, founder decision 2026-05-15).
    await apply_migration(
        version="2026-05-16-z7-outreach-warmup",
        sql=(
            "CREATE TABLE IF NOT EXISTS outreach_warmup ("
            " warmup_id    INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id   TEXT    NOT NULL,"
            " domain       TEXT    NOT NULL,"
            " day          INTEGER NOT NULL,"       # 1-based day in warmup ramp
            " sent_count   INTEGER NOT NULL DEFAULT 0,"
            " target_count INTEGER NOT NULL DEFAULT 50,"
            " created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " UNIQUE(product_id, domain, day)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_outreach_warmup_product_domain "
            "ON outreach_warmup(product_id, domain);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_outreach_warmup_product_domain;\n"
            "DROP TABLE IF EXISTS outreach_warmup;\n"
        ),
        description=(
            "Z7 T6 A7: outreach_warmup — domain warm-up ramp table. "
            "UNIQUE(product_id, domain, day). sent_count < target_count required "
            "for outreach/send to proceed. Ramp: day1=50 → day14=500/day. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T6 A7: outreach_sends — cold outreach send log ────────────────────────
    # One row per outbound cold outreach email. send_id is the canonical handle
    # passed in Reply-To + X-Send-ID headers so handle_reply can match webhook events.
    # replied_at set by handle_reply; opened_at/bounced_at set by webhook receivers.
    # product_id NOT NULL (per-product scoping, founder decision 2026-05-15).
    await apply_migration(
        version="2026-05-16-z7-outreach-sends",
        sql=(
            "CREATE TABLE IF NOT EXISTS outreach_sends ("
            " send_id      INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id   TEXT    NOT NULL,"
            " list_id      TEXT,"
            " target_email TEXT    NOT NULL,"
            " template_id  TEXT,"
            " sent_at      TEXT,"
            " opened_at    TEXT,"
            " replied_at   TEXT,"
            " bounced_at   TEXT"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_outreach_sends_product_sent "
            "ON outreach_sends(product_id, sent_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_outreach_sends_list "
            "ON outreach_sends(list_id, product_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_outreach_sends_list;\n"
            "DROP INDEX IF EXISTS idx_outreach_sends_product_sent;\n"
            "DROP TABLE IF EXISTS outreach_sends;\n"
        ),
        description=(
            "Z7 T6 A7: outreach_sends — cold outreach send log. "
            "send_id matched in Reply-To / X-Send-ID headers for A7.r1 reply-handling. "
            "opened_at/replied_at/bounced_at recorded from ESP webhooks. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 T6 A11: mentions — mention monitor ingestion table ────────────────────
    # One row per unique mention. UNIQUE(source, source_id) for within-source dedup.
    # Cross-source dedup via canonical_url + 24h window is enforced in the poll verb.
    # sentiment: 'pos' | 'neg' | 'neu'
    # signal_score: 0-10 (0=noise, 4-7=digest, >=7=immediate founder_action)
    # acted_on: 1 when a founder_action has been surfaced for this mention.
    # product_id NOT NULL (per-product scoping, founder decision 2026-05-15).
    await apply_migration(
        version="2026-05-16-z7-a11-mentions",
        sql=(
            "CREATE TABLE IF NOT EXISTS mentions ("
            " mention_id    INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id    TEXT    NOT NULL,"
            " source        TEXT    NOT NULL,"
            " source_id     TEXT    NOT NULL,"
            " url           TEXT,"
            " canonical_url TEXT,"
            " author        TEXT,"
            " author_followers INTEGER NOT NULL DEFAULT 0,"
            " text          TEXT    NOT NULL DEFAULT '',"
            " sentiment     TEXT    NOT NULL DEFAULT 'neu',"  # 'pos' | 'neg' | 'neu'
            " signal_score  INTEGER NOT NULL DEFAULT 0,"  # 0-10
            " seen_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " acted_on      INTEGER NOT NULL DEFAULT 0,"
            " UNIQUE(source, source_id)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_product_seen "
            "ON mentions(product_id, seen_at DESC);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_product_sentiment_score "
            "ON mentions(product_id, sentiment, signal_score);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_canonical_url "
            "ON mentions(canonical_url, seen_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_acted_on "
            "ON mentions(product_id, acted_on, signal_score DESC);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_mentions_acted_on;\n"
            "DROP INDEX IF EXISTS idx_mentions_canonical_url;\n"
            "DROP INDEX IF EXISTS idx_mentions_product_sentiment_score;\n"
            "DROP INDEX IF EXISTS idx_mentions_product_seen;\n"
            "DROP TABLE IF EXISTS mentions;\n"
        ),
        description=(
            "Z7 T6 A11: mentions — mention monitor ingestion table. "
            "UNIQUE(source, source_id) for within-source dedup. "
            "sentiment='pos'|'neg'|'neu'; signal_score 0-10. "
            "score<4 silent, 4-7 daily digest, >=7 immediate founder_action. "
            "acted_on=1 after founder_action surfaced. "
            "canonical_url + 24h window used for cross-source dedup in poll verbs. "
            "product_id NOT NULL (per-product scoping, founder decision 2026-05-15)."
        ),
    )

    # ── Z7 wiring-sweep A11: mention_monitors — registry of monitored products ───
    # The mention monitor only polls products the founder has explicitly
    # registered via /mention_monitor add. One row per (product_id). channels
    # is a JSON list of enabled source names ('hn','reddit','google','discord').
    await apply_migration(
        version="2026-05-18-z7-a11-mention-monitors",
        sql=(
            "CREATE TABLE IF NOT EXISTS mention_monitors ("
            " product_id    TEXT    PRIMARY KEY,"
            " product_name  TEXT    NOT NULL DEFAULT '',"
            " channels_json TEXT    NOT NULL DEFAULT '[]',"
            " enabled       INTEGER NOT NULL DEFAULT 1,"
            " created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " last_run_at   TEXT"
            ");\n"
        ),
        reversal_sql="DROP TABLE IF EXISTS mention_monitors;\n",
        description=(
            "Z7 wiring-sweep A11: mention_monitors — registry of products the "
            "founder opted into mention monitoring for. channels_json is the "
            "enabled source list; the hourly mention_monitor_sweep cron polls "
            "only the rows here. Without this the A11 monitor never runs."
        ),
    )

    # ── Z7 #4: outreach_prospects — uploaded cold-outreach prospect lists ────────
    # /outreach upload persists prospects here as 'pending'; on founder
    # approval of the batch card they flip to 'approved' and an outreach/draft
    # task is dispatched per prospect. Without this table /outreach upload was
    # a pure stub with no list storage.
    await apply_migration(
        version="2026-05-18-z7-a7-outreach-prospects",
        sql=(
            "CREATE TABLE IF NOT EXISTS outreach_prospects ("
            " prospect_id  INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id   TEXT    NOT NULL,"
            " list_id      TEXT    NOT NULL,"
            " target_email TEXT    NOT NULL,"
            " name         TEXT    NOT NULL DEFAULT '',"
            " status       TEXT    NOT NULL DEFAULT 'pending',"
            " created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " UNIQUE(product_id, list_id, target_email)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_outreach_prospects_list "
            "ON outreach_prospects(product_id, list_id, status);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_outreach_prospects_list;\n"
            "DROP TABLE IF EXISTS outreach_prospects;\n"
        ),
        description=(
            "Z7 #4 A7: outreach_prospects — uploaded cold-outreach prospect "
            "lists. status pending->approved on founder batch approval; "
            "UNIQUE(product_id,list_id,target_email) dedups re-uploads."
        ),
    )

    # ── Z7 #7: backfill missions.product_id = id for legacy rows ─────────────────
    # add_mission now defaults product_id to the mission id; backfill the rows
    # created before that so investor_bullets' product-scoped JOINs see them.
    await apply_migration(
        version="2026-05-18-z7-product-id-backfill",
        sql=(
            "UPDATE missions SET product_id = CAST(id AS TEXT) "
            "WHERE product_id IS NULL;\n"
        ),
        reversal_sql=None,  # data backfill — no structural reversal
        description=(
            "Z7 #7: backfill missions.product_id with the mission id for rows "
            "predating the add_mission default. Unblocks investor_bullets' "
            "product-scoped metric JOINs."
        ),
    )

    # ── Z7 fix-pass: product-scope the mentions UNIQUE constraint ────────────────
    # The original 2026-05-16-z7-a11-mentions migration used
    # UNIQUE(source, source_id). Two *different* products mentioned in the same
    # HN/Reddit thread share (source, source_id) and would collide — the second
    # product's mention is silently dropped by INSERT OR IGNORE. The correct
    # within-source dedup key is per-product: UNIQUE(product_id, source, source_id).
    # This corrective migration rebuilds the table preserving any existing rows
    # (the table is normally empty in practice — A11 polls had not run live).
    await apply_migration(
        version="2026-05-17-z7-a11-mentions-product-scope",
        sql=(
            "ALTER TABLE mentions RENAME TO mentions_old;\n"
            "CREATE TABLE mentions ("
            " mention_id    INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id    TEXT    NOT NULL,"
            " source        TEXT    NOT NULL,"
            " source_id     TEXT    NOT NULL,"
            " url           TEXT,"
            " canonical_url TEXT,"
            " author        TEXT,"
            " author_followers INTEGER NOT NULL DEFAULT 0,"
            " text          TEXT    NOT NULL DEFAULT '',"
            " sentiment     TEXT    NOT NULL DEFAULT 'neu',"
            " signal_score  INTEGER NOT NULL DEFAULT 0,"
            " seen_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " acted_on      INTEGER NOT NULL DEFAULT 0,"
            " UNIQUE(product_id, source, source_id)"
            ");\n"
            "INSERT OR IGNORE INTO mentions "
            "(mention_id, product_id, source, source_id, url, canonical_url, "
            " author, author_followers, text, sentiment, signal_score, "
            " seen_at, acted_on) "
            "SELECT mention_id, product_id, source, source_id, url, canonical_url, "
            " author, author_followers, text, sentiment, signal_score, "
            " seen_at, acted_on FROM mentions_old;\n"
            "DROP TABLE mentions_old;\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_product_seen "
            "ON mentions(product_id, seen_at DESC);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_product_sentiment_score "
            "ON mentions(product_id, sentiment, signal_score);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_canonical_url "
            "ON mentions(canonical_url, seen_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_acted_on "
            "ON mentions(product_id, acted_on, signal_score DESC);\n"
        ),
        reversal_sql=(
            # Reverse to the original (source, source_id) unique key.
            "DROP INDEX IF EXISTS idx_mentions_acted_on;\n"
            "DROP INDEX IF EXISTS idx_mentions_canonical_url;\n"
            "DROP INDEX IF EXISTS idx_mentions_product_sentiment_score;\n"
            "DROP INDEX IF EXISTS idx_mentions_product_seen;\n"
            "ALTER TABLE mentions RENAME TO mentions_new;\n"
            "CREATE TABLE mentions ("
            " mention_id    INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id    TEXT    NOT NULL,"
            " source        TEXT    NOT NULL,"
            " source_id     TEXT    NOT NULL,"
            " url           TEXT,"
            " canonical_url TEXT,"
            " author        TEXT,"
            " author_followers INTEGER NOT NULL DEFAULT 0,"
            " text          TEXT    NOT NULL DEFAULT '',"
            " sentiment     TEXT    NOT NULL DEFAULT 'neu',"
            " signal_score  INTEGER NOT NULL DEFAULT 0,"
            " seen_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " acted_on      INTEGER NOT NULL DEFAULT 0,"
            " UNIQUE(source, source_id)"
            ");\n"
            "INSERT OR IGNORE INTO mentions "
            "(mention_id, product_id, source, source_id, url, canonical_url, "
            " author, author_followers, text, sentiment, signal_score, "
            " seen_at, acted_on) "
            "SELECT mention_id, product_id, source, source_id, url, canonical_url, "
            " author, author_followers, text, sentiment, signal_score, "
            " seen_at, acted_on FROM mentions_new;\n"
            "DROP TABLE mentions_new;\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_product_seen "
            "ON mentions(product_id, seen_at DESC);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_product_sentiment_score "
            "ON mentions(product_id, sentiment, signal_score);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_canonical_url "
            "ON mentions(canonical_url, seen_at);\n"
            "CREATE INDEX IF NOT EXISTS idx_mentions_acted_on "
            "ON mentions(product_id, acted_on, signal_score DESC);\n"
        ),
        description=(
            "Z7 fix-pass: re-scope mentions UNIQUE from (source, source_id) to "
            "(product_id, source, source_id). The original key collided when two "
            "different products were mentioned in the same external thread, "
            "silently dropping the second via INSERT OR IGNORE. Table rebuilt "
            "preserving existing rows."
        ),
    )

    # ── Z7 fix-pass A6: outreach_pauses — real campaign-pause flag ───────────────
    # outreach_deliverability_check used to only *emit a founder_action* saying
    # "campaign paused" — it set no flag, so outreach/send kept sending. This
    # table is the real pause flag: one row per paused (product_id, list_id).
    # outreach/send refuses to send when an un-cleared row exists.
    await apply_migration(
        version="2026-05-17-z7-a6-outreach-pauses",
        sql=(
            "CREATE TABLE IF NOT EXISTS outreach_pauses ("
            " pause_id   INTEGER PRIMARY KEY AUTOINCREMENT,"
            " product_id TEXT    NOT NULL,"
            " list_id    TEXT    NOT NULL,"
            " reason     TEXT    NOT NULL DEFAULT '',"
            " paused_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),"
            " cleared_at TEXT,"
            " UNIQUE(product_id, list_id)"
            ");\n"
            "CREATE INDEX IF NOT EXISTS idx_outreach_pauses_active "
            "ON outreach_pauses(product_id, list_id, cleared_at);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_outreach_pauses_active;\n"
            "DROP TABLE IF EXISTS outreach_pauses;\n"
        ),
        description=(
            "Z7 fix-pass A6: outreach_pauses — campaign-pause flag. "
            "outreach_deliverability_check INSERTs a row (cleared_at NULL) when "
            "bounce/complaint thresholds are exceeded; outreach/send refuses to "
            "send while an un-cleared row exists for (product_id, list_id). "
            "Founder clears the pause by stamping cleared_at."
        ),
    )

    # ── Z7 B9 fix: external_comms_log.task_id correlation key ─────────────
    # external_comms_log.vendor_call_id (action_confirmations.id) is never
    # populated by the production publish path (log_publish_action does not
    # have the confirmation id), so pending_audit_gaps' NOT EXISTS join on
    # vendor_call_id never matched and flagged EVERY irreversible confirmation
    # as an un-logged gap. The dispatching task_id IS available at every
    # publish-log call site AND on action_confirmations.task_id, so it is the
    # honest correlation key. log_publish_action now threads task_id into
    # external_comms_log, and pending_audit_gaps correlates on
    # (task_id, verb).
    await apply_migration(
        version="2026-05-17-z7-external-comms-log-task-id",
        sql=(
            "ALTER TABLE external_comms_log "
            "ADD COLUMN task_id INTEGER;\n"   # FK tasks.id (app-level)
            "CREATE INDEX IF NOT EXISTS idx_external_comms_log_task "
            "ON external_comms_log(task_id);\n"
        ),
        reversal_sql=(
            "DROP INDEX IF EXISTS idx_external_comms_log_task;\n"
            "ALTER TABLE external_comms_log DROP COLUMN task_id;\n"
        ),
        description=(
            "Z7 fix-pass B9: external_comms_log.task_id — the dispatching "
            "task that produced the send. Real correlation key for "
            "pending_audit_gaps (action_confirmations.task_id = "
            "external_comms_log.task_id + verb match); vendor_call_id was "
            "never populated in production so the old join always missed."
        ),
    )

    # Legacy 'Todo Reminder' (id=9999) and 'Price Watch Check' (id=9998) seeds
    # were removed — beckman cron_seed.INTERNAL_CADENCES now owns these via
    # mr_roboto mechanical executors. Clean up any stale rows from earlier runs.
    await db.execute("DELETE FROM scheduled_tasks WHERE id IN (9998, 9999)")
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
    logger.info(f"DB schema verified. Tasks columns: {columns}")

    # Migration: add task_hash column if not present
    if "task_hash" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN task_hash TEXT"
            )
            await db.commit()
            logger.info("Added task_hash column to tasks table")
        except Exception as e:
            logger.debug(f"task_hash column migration skipped: {e}")

    # Migration: add task_state column for checkpointing
    if "task_state" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN task_state JSON DEFAULT NULL"
            )
            await db.commit()
            logger.info("Added task_state column to tasks table")
        except Exception as e:
            logger.debug(f"task_state column migration skipped: {e}")

    # Migration: add timeout_seconds column for per-task timeouts (Phase 3)
    if "timeout_seconds" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN timeout_seconds INTEGER DEFAULT NULL"
            )
            await db.commit()
            logger.info("Added timeout_seconds column to tasks table")
        except Exception as e:
            logger.debug(f"timeout_seconds column migration skipped: {e}")

    # Migration: add quality_score column for response grading (Phase 4)
    if "quality_score" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN quality_score REAL DEFAULT NULL"
            )
            await db.commit()
            logger.info("Added quality_score column to tasks table")
        except Exception as e:
            logger.debug(f"quality_score column migration skipped: {e}")

    # Migration: add error_category column for error taxonomy (Phase 9)
    if "error_category" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN error_category TEXT DEFAULT NULL"
            )
            await db.commit()
            logger.info("Added error_category column to tasks table")
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
            "ALTER TABLE tasks ADD COLUMN max_worker_attempts INTEGER DEFAULT 15",
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

    # Migration: add kind column for sub-task admission (Beckman Phase 1)
    if "kind" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'main_work'"
            )
            await db.commit()
            logger.info("Added kind column to tasks table")
        except Exception as e:
            logger.debug(f"kind column migration skipped: {e}")

    # Migration: add runner column for orchestrator lane dispatch
    # (Phase D — runtime extraction, 2026-05-05). Three lanes:
    #   'mechanical' — mr_roboto sub-tasks (no LLM)
    #   'direct'     — single-call OVERHEAD (graders, structured_emit, classifier)
    #   'react'      — multi-call ReAct loop with tools (default)
    # Backfill matches what the orchestrator's lane decision derives today:
    # mechanical wins if agent_type='mechanical'; else 'direct' for overhead
    # kind; else 'react'. After D.3 the orchestrator dispatches by runner
    # column directly instead of inferring from agent_type/kind.
    if "runner" not in columns:
        try:
            await db.execute(
                "ALTER TABLE tasks ADD COLUMN runner TEXT NOT NULL DEFAULT 'react'"
            )
            await db.execute("""
                UPDATE tasks
                SET runner = CASE
                    WHEN agent_type = 'mechanical' THEN 'mechanical'
                    WHEN kind = 'overhead' THEN 'direct'
                    ELSE 'react'
                END
                WHERE runner IS NULL OR runner = 'react'
            """)
            await db.commit()
            logger.info("Added runner column to tasks table + backfilled")
        except Exception as e:
            logger.debug(f"runner column migration skipped: {e}")

    # Migration: add estimated_cost_usd column for per-task cost estimates (Z0-T9)
    try:
        await db.execute("ALTER TABLE tasks ADD COLUMN estimated_cost_usd REAL DEFAULT 0")
        await db.commit()
        logger.info("Added estimated_cost_usd column to tasks table")
    except Exception as e:
        logger.debug(f"tasks.estimated_cost_usd migration skipped: {e}")

    # Z0 mission preflight columns (2026-05-05). NOTE: lifecycle_state is
    # intentionally NOT added here — it is owned by the Z8 T1A migration
    # above (NOT NULL DEFAULT 'terminal'). Z0 lifecycle (pause/resume/kill)
    # shares that single column; its state set extends Z8's with
    # {paused, killed}. Legacy rows backfill to 'terminal', which is
    # correct: a pre-existing mission is not running.
    for ddl in (
        "ALTER TABLE missions ADD COLUMN cost_ceiling_usd REAL",
        "ALTER TABLE missions ADD COLUMN spent_usd REAL DEFAULT 0",
        "ALTER TABLE missions ADD COLUMN message_thread_id INTEGER",
    ):
        try:
            await db.execute(ddl)
            await db.commit()
        except Exception as e:
            logger.debug(f"Z0 column migration skipped (already present): {e}")

    await db.execute("""
        CREATE TABLE IF NOT EXISTS mission_lifecycle_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id INTEGER NOT NULL,
            from_state TEXT,
            to_state TEXT NOT NULL,
            reason TEXT,
            triggered_by TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mission_id) REFERENCES missions(id)
        )
    """)
    await db.commit()

    # Backfill any pre-existing NULL rows (older installs). lifecycle_state
    # is NOT NULL via the Z8 migration, so only spent_usd needs backfill.
    await db.execute(
        "UPDATE missions SET spent_usd = 0 WHERE spent_usd IS NULL"
    )
    await db.commit()

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

    await db.commit()

    # Yalayut tables (catalog, demand signals, sources, index, etc.)
    try:
        from yalayut.schema import ensure_yalayut_schema
        await ensure_yalayut_schema(db)
    except Exception as e:
        logger.debug(f"yalayut schema skipped: {e}")

    # Yalayut seed data — owners / sources / disabled-imports / the ~20 seed
    # manifests. run_full_migration() is fully idempotent (INSERT OR IGNORE +
    # ON CONFLICT DO UPDATE), but install_seed_manifests() embeds every seed
    # on each call, so we gate it on an empty yalayut_index: the expensive
    # path runs once on a fresh deploy, every later boot is just one COUNT.
    try:
        cur = await db.execute("SELECT COUNT(*) FROM yalayut_index")
        row = await cur.fetchone()
        index_count = row[0] if row else 0
        if index_count == 0:
            from yalayut.migration import run_full_migration
            result = await run_full_migration(db)
            logger.info(
                "yalayut seed loaded: %s seeds, %s skills, %s owners, "
                "%s sources",
                result.get("seeds_indexed"), result.get("skills_migrated"),
                result.get("owners_seeded"), result.get("sources_seeded"),
            )
    except Exception as e:
        logger.debug(f"yalayut seed-load skipped: {e}")

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
        # Lift max_worker_attempts cap from 6 → 10 on still-active rows.
        # The retry policy + backoff ladder were sized for 10 attempts
        # (apply.py:264 / sweep.py:403 both default `or 10`), but the DB
        # column DEFAULT had been left at 6 — so existing rows DLQ'd at
        # att=5/6 on availability cascades that should have ridden out
        # the longer ladder. Only touch live rows; failed/completed/
        # cancelled history stays as-is.
        await db.execute(
            "UPDATE tasks SET max_worker_attempts = 10 "
            "WHERE max_worker_attempts = 6 "
            "AND status IN ('pending','ready','processing','ungraded','waiting_human')"
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
                      workflow=None, repo_path=None, language=None, framework=None,
                      budget_ceiling_usd: float | None = None):
    db = await get_db()
    cursor = await db.execute(
        """INSERT INTO missions (title, description, priority, context, workflow, repo_path, language, framework)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (title, description, priority, json.dumps(context or {}),
         workflow or "", repo_path or "", language or "", framework or "")
    )
    await db.commit()
    mission_id = cursor.lastrowid
    # Z7 #7: default product_id to the mission's own id. The column was a
    # nullable placeholder no code ever populated, so investor_bullets'
    # product-scoped JOINs (WHERE m.product_id=?) always matched zero rows.
    # One mission == one product unless something explicitly re-groups it.
    try:
        if mission_id is not None:
            await db.execute(
                "UPDATE missions SET product_id = ? "
                "WHERE id = ? AND product_id IS NULL",
                (str(mission_id), mission_id),
            )
            await db.commit()
    except Exception as e:
        logger.debug(f"product_id default at add_mission skipped: {e}")
    # Z10 T2A: seed the per-mission cost_budgets row so token accounting
    # has a target to increment into. budget_ceiling_usd remains NULL
    # (= unlimited) unless the caller supplies one.
    try:
        if mission_id is not None:
            await ensure_mission_cost_row(
                int(mission_id), budget_ceiling_usd=budget_ceiling_usd
            )
    except Exception as e:
        logger.debug(f"ensure_mission_cost_row at add_mission skipped: {e}")
    # Z10-T3B: provision the per-mission Docker container + network on
    # mission creation. Best-effort — failures are warning-logged but
    # don't block mission creation (e.g. docker daemon down → callers
    # fall back to host-local execution when shell is invoked). Skip
    # entirely when SANDBOX_MODE is ``none`` or ``local`` (no docker
    # path is going to be taken anyway).
    try:
        if mission_id is not None:
            from src.tools import shell as _shell_mod
            if _shell_mod.SANDBOX_MODE not in ("none", "local"):
                await _shell_mod.ensure_mission_container(int(mission_id))
    except Exception as e:
        logger.debug(f"ensure_mission_container at add_mission skipped: {e}")
    return mission_id

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
    "phase_7_rework_loops",
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
    # Z6 T1A: real-world bridge gate inputs (hoisted from context for index).
    "needs_real_tools", "reversibility",
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


async def increment_mission_rework_loops(mission_id: int) -> int:
    """Atomically bump missions.phase_7_rework_loops; return the new count.

    Used by src/telemetry/rework.record_rollback() — the single source of
    truth for B10 rework telemetry. Returns 0 if the mission row is
    missing (defensive: telemetry must never crash the caller).
    """
    db = await get_db()
    await db.execute(
        "UPDATE missions SET phase_7_rework_loops = "
        "COALESCE(phase_7_rework_loops, 0) + 1 WHERE id = ?",
        (mission_id,),
    )
    cursor = await db.execute(
        "SELECT phase_7_rework_loops FROM missions WHERE id = ?",
        (mission_id,),
    )
    row = await cursor.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


async def get_mission_rework_summary(
    limit: int = 20,
) -> list[dict]:
    """Return per-mission rework counts + reasons for the most recent missions.

    Reasons are derived from yazbunu phase_rollback events stored in
    logs/kutai.jsonl. If the log file is missing, only counts are returned.
    """
    db = await get_db()
    cursor = await db.execute(
        """SELECT id, title, status, phase_7_rework_loops, created_at
           FROM missions
           ORDER BY id DESC
           LIMIT ?""",
        (limit,),
    )
    rows = [dict(r) for r in await cursor.fetchall()]
    return rows


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
                   requires_approval=False, depends_on=None, context=None,
                   kind="main_work", runner=None,
                   needs_real_tools=None, reversibility=None,
                   lane=None):
    """Atomic dedup + insert.

    Uses an isolated connection (connect_aux) for the BEGIN/COMMIT
    region. Sharing the singleton's connection with concurrent
    coroutines is unsafe even with _tx_lock — another coroutine on
    the same conn can issue ``db.commit()`` (= "COMMIT" SQL) and
    close OUR explicit tx, leaving ROLLBACK with nothing to roll back
    ("cannot rollback - no transaction is active").
    """
    task_hash = compute_task_hash(title, description, agent_type, mission_id, parent_task_id)

    # Phase D — orchestrator dispatches by ``runner``. Producers can pass it
    # explicitly; otherwise derive from agent_type/kind matching the
    # backfill rule applied at migration time so behaviour matches for
    # legacy callers.
    if runner is None:
        if agent_type == "mechanical":
            runner = "mechanical"
        elif kind == "overhead":
            runner = "direct"
        else:
            runner = "react"

    # Z10 T3C: lock per mission_id so concurrent missions don't serialize.
    # Writes to `tasks` are mission-scoped — fall back to the global slot for
    # the rare mission_id=None case (orphan tasks).
    async with _get_tx_lock(mission_id), connect_aux(DB_PATH, _label="add_task") as db:
        db.row_factory = aiosqlite.Row
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
                        if db._conn.in_transaction:
                            await db.execute("COMMIT")
                        else:
                            logger.warning(
                                f"add_task: tx vanished before COMMIT "
                                f"(stuck-reset path, dup #{dup['id']}); "
                                f"work likely auto-rolled-back"
                            )
                        return dup["id"]
                logger.info(
                    f"⏭️ Task dedup: '{title[:50]}' matches pending task "
                    f"#{dup['id']} — skipping creation"
                )
                if db._conn.in_transaction:
                    await db.execute("ROLLBACK")
                return None

            # Z10 T3A: derive phase_id from workflow context if present.
            # Expander writes 'workflow_phase' (e.g. 'phase_5'); legacy
            # callers may pass 'phase_id' directly. Either populates the
            # new tasks.phase_id column for the pacing breakdown.
            _phase_id = None
            if isinstance(context, dict):
                _phase_id = (
                    context.get("phase_id")
                    or context.get("workflow_phase")
                )
            # Z6 T1A: hoist needs_real_tools / reversibility from context if
            # not passed explicitly. Expander writes them on the step JSON;
            # callers like _apply_subtasks fan that into task.context. Pulling
            # them up here means a single source of truth (the indexed column)
            # for beckman admission.
            _nrt = needs_real_tools
            _rev = reversibility
            if isinstance(context, dict):
                if _nrt is None and context.get("needs_real_tools"):
                    _nrt = 1
                if _rev is None and context.get("reversibility"):
                    _rev = str(context.get("reversibility"))
            _nrt_int = 1 if _nrt else 0
            # Z8 T1B: lane defaults to 'oneshot' so legacy callers keep
            # flowing through the historical pool. Callers that want the
            # ongoing pool (alert_triage / cron) pass lane='ongoing'.
            _lane = lane or "oneshot"
            cursor = await db.execute(
                """INSERT INTO tasks
                   (mission_id, parent_task_id, title, description, agent_type,
                    tier, priority, requires_approval, depends_on, context,
                    task_hash, kind, runner, phase_id,
                    needs_real_tools, reversibility, lane)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mission_id, parent_task_id, title, description, agent_type,
                 tier, priority, requires_approval,
                 json.dumps(depends_on or []), json.dumps(context or {}),
                 task_hash, kind, runner, _phase_id,
                 _nrt_int, _rev, _lane)
            )
            row_id = cursor.lastrowid
            if db._conn.in_transaction:
                await db.execute("COMMIT")
            else:
                logger.warning(
                    f"add_task: tx vanished before COMMIT (insert path, "
                    f"new row {row_id}); work likely auto-rolled-back"
                )
            return row_id
        except Exception:
            try:
                if db._conn.in_transaction:
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

    # Fetch ALL pending tasks — filter AFTER dep check.
    # LEFT JOIN missions so standalone tasks (mission_id IS NULL) still pass through.
    # Tasks whose mission is paused or killed are excluded at the SQL level.
    cursor = await db.execute(
        """SELECT t.* FROM tasks t
           LEFT JOIN missions m ON t.mission_id = m.id
           WHERE t.status = 'pending'
           AND (t.next_retry_at IS NULL OR t.next_retry_at <= datetime('now'))
           AND (m.id IS NULL OR m.lifecycle_state = 'active')
           ORDER BY t.priority DESC, t.created_at ASC"""
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
            title = task.get("title", "?")
            await db.execute(
                "UPDATE tasks SET status = 'skipped', error = 'dependency_skipped' WHERE id = ?",
                (task_id,)
            )
            await db.commit()
            logger.warning(
                "TASK SKIPPED (all deps skipped)",
                task_id=task_id,
                title=title,
                deps=deps,
            )
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

async def _retry_on_locked(coro_factory, *, label: str, max_attempts: int = 4):
    """Retry an awaitable factory on 'database is locked' OperationalError.

    The singleton's busy_timeout=60s waits for the writer slot, then raises.
    On contention bursts (production 2026-05-01: held=132s on add_task aux
    conns while singleton update_task fired ModelCallFailed via 'database
    is locked' on on_task_finished's posthook verdict), the singleton's one
    pass is not enough — by the time we surface OperationalError, the
    writer-slot is usually freed and a quick retry succeeds.

    Retry budget: 4 attempts, exponential backoff
    (0.2s → 0.5s → 1.0s → fail). With 60s busy_timeout per attempt, total
    wall-clock cap is ~241s. Bounded by `_aux_active_summary()` evidence
    that long holds typically cluster within ~140s, then quiet down.

    Caller passes a coroutine FACTORY (zero-arg callable) so each attempt
    creates a fresh awaitable — re-awaiting a consumed coroutine raises.
    """
    import sqlite3
    delays = [0.2, 0.5, 1.0]
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except sqlite3.OperationalError as e:
            last_exc = e
            msg = str(e).lower()
            if "database is locked" not in msg and "database is busy" not in msg:
                raise
            if attempt >= len(delays):
                logger.error(
                    "%s: lock retry exhausted attempts=%d holders=[%s]",
                    label, attempt + 1, _aux_active_summary(),
                )
                raise
            logger.warning(
                "%s: lock retry %d/%d backoff=%.1fs holders=[%s]",
                label, attempt + 1, max_attempts, delays[attempt],
                _aux_active_summary(),
            )
            await asyncio.sleep(delays[attempt])
    if last_exc is not None:
        raise last_exc


async def update_task(task_id, **kwargs):
    _validate_columns(kwargs, _TASK_COLUMNS, "tasks")
    if kwargs.get("status") == "skipped":
        logger.warning(
            "TASK STATUS → SKIPPED",
            task_id=task_id,
            error=kwargs.get("error", "NO_ERROR_SET"),
        )
    db = await get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    sql = f"UPDATE tasks SET {sets} WHERE id = ?"

    async def _do():
        await db.execute(sql, values)
        await db.commit()

    await _retry_on_locked(_do, label=f"update_task #{task_id}")


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
    sql = (
        f"UPDATE tasks SET {sets} "
        f"WHERE mission_id = ? AND json_extract(context, '$.{field}') = ?"
    )

    async def _do():
        await db.execute(sql, values)
        await db.commit()

    await _retry_on_locked(
        _do, label=f"update_task_by_context_field mission={mission_id}",
    )


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


# Strong-reference set so fire-and-forget accelerate_retries tasks are not GC'd.
_pending_accelerate_retry_tasks: set[asyncio.Task] = set()


# Co-located with accelerate_retries() to avoid a circular import
# between a dedicated asyncio-utils module and this DB module.
def schedule_accelerate_retries(reason: str) -> None:
    """Fire-and-forget accelerate_retries with strong-ref retention.

    Safe to call from any sync context that has a running event loop.
    No-ops silently when no loop is running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(accelerate_retries(reason))
    _pending_accelerate_retry_tasks.add(task)
    task.add_done_callback(_pending_accelerate_retry_tasks.discard)


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
    # Z10 T3A: stamp step_started_at alongside started_at on transition to
    # 'processing'. step_started_at is reset per attempt; started_at marks
    # the first pickup. For the claim path the two are equal.
    cursor = await db.execute(
        "UPDATE tasks SET status = 'processing', started_at = ?, "
        "                  step_started_at = ? "
        "WHERE id = ? AND status = 'pending'",
        (now_str, now_str, task_id)
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
    created_ids: list[int] = []

    # Isolated connection for the BEGIN/COMMIT region — see add_task
    # docstring for why _tx_lock alone cannot protect the singleton.
    # Z10 T3C: mission-scoped lock — concurrent missions don't contend here.
    async with _get_tx_lock(mission_id), connect_aux(
        DB_PATH, _label="add_subtasks_atomically"
    ) as db:
        db.row_factory = aiosqlite.Row
        try:
            await db.execute("BEGIN IMMEDIATE")

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

                # Phase D — orchestrator dispatches by runner. Producers
                # may pass it explicitly via st["runner"]; otherwise derive
                # from agent_type (mechanical lane is the only special-case
                # — sub-tasks default to react).
                runner = st.get("runner") or (
                    "mechanical" if agent_type == "mechanical" else "react"
                )

                # Z10 T3A: derive phase_id from context if present.
                _sub_ctx = st.get("context") or {}
                _sub_phase_id = (
                    (_sub_ctx.get("phase_id")
                     or _sub_ctx.get("workflow_phase"))
                    if isinstance(_sub_ctx, dict) else None
                )
                # Z6 T1A: hoist from sub-context.
                _sub_nrt = st.get("needs_real_tools")
                _sub_rev = st.get("reversibility")
                if isinstance(_sub_ctx, dict):
                    if _sub_nrt is None and _sub_ctx.get("needs_real_tools"):
                        _sub_nrt = 1
                    if _sub_rev is None and _sub_ctx.get("reversibility"):
                        _sub_rev = str(_sub_ctx.get("reversibility"))
                _sub_nrt_int = 1 if _sub_nrt else 0
                cursor = await db.execute(
                    """INSERT INTO tasks
                       (mission_id, parent_task_id, title, description, agent_type,
                        tier, priority, requires_approval, depends_on, context,
                        task_hash, runner, phase_id,
                        needs_real_tools, reversibility)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)""",
                    (mission_id, parent_task_id, title, description, agent_type,
                     st.get("tier", "auto"), st.get("priority", 5),
                     json.dumps(st.get("depends_on", [])),
                     json.dumps(st.get("context", {})),
                     task_hash, runner, _sub_phase_id,
                     _sub_nrt_int, _sub_rev)
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

            if db._conn.in_transaction:
                await db.execute("COMMIT")
            else:
                logger.warning(
                    "add_subtasks_atomically: tx vanished before COMMIT; "
                    "work likely auto-rolled-back"
                )
        except Exception:
            try:
                if db._conn.in_transaction:
                    await db.execute("ROLLBACK")
            except Exception:
                pass
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
    created_ids: list[int] = []

    # Isolated connection for the BEGIN/COMMIT region.
    # Z10 T3C: mission-scoped lock — concurrent missions don't contend here.
    async with _get_tx_lock(mission_id), connect_aux(
        DB_PATH, _label="insert_tasks_atomically"
    ) as db:
        db.row_factory = aiosqlite.Row
        try:
            await db.execute("BEGIN IMMEDIATE")

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

                # Phase D — orchestrator dispatches by runner. Producer
                # may pass it explicitly; otherwise derive from agent_type.
                runner = t.get("runner") or (
                    "mechanical" if agent_type == "mechanical" else "react"
                )

                # Z10 T3A: derive phase_id from context if present.
                _t_ctx = t.get("context") or {}
                _t_phase_id = (
                    (_t_ctx.get("phase_id") or _t_ctx.get("workflow_phase"))
                    if isinstance(_t_ctx, dict) else None
                )
                # Z6 T1A: hoist from sub-context.
                _t_nrt = t.get("needs_real_tools")
                _t_rev = t.get("reversibility")
                if isinstance(_t_ctx, dict):
                    if _t_nrt is None and _t_ctx.get("needs_real_tools"):
                        _t_nrt = 1
                    if _t_rev is None and _t_ctx.get("reversibility"):
                        _t_rev = str(_t_ctx.get("reversibility"))
                _t_nrt_int = 1 if _t_nrt else 0
                cursor = await db.execute(
                    """INSERT INTO tasks
                       (mission_id, parent_task_id, title, description, agent_type,
                        tier, priority, requires_approval, depends_on, context,
                        task_hash, runner, phase_id,
                        needs_real_tools, reversibility)
                       VALUES (?, NULL, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)""",
                    (mission_id, title, description, agent_type,
                     t.get("tier", "auto"), t.get("priority", 5),
                     json.dumps(t.get("depends_on", [])),
                     json.dumps(t.get("context", {})),
                     task_hash, runner, _t_phase_id,
                     _t_nrt_int, _t_rev)
                )
                created_ids.append(cursor.lastrowid)

            if db._conn.in_transaction:
                await db.execute("COMMIT")
            else:
                logger.warning(
                    "insert_tasks_atomically: tx vanished before COMMIT; "
                    "work likely auto-rolled-back"
                )
        except Exception:
            try:
                if db._conn.in_transaction:
                    await db.execute("ROLLBACK")
            except Exception:
                pass
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
            "SELECT id, depends_on, title FROM tasks WHERE mission_id = ? AND status = 'pending'",
            (mission_id,)
        )
        pending_tasks = await cursor.fetchall()

        newly_skipped = 0

        for row in pending_tasks:
            task_id = row[0]
            raw_deps = row[1]
            task_title = row[2] if len(row) > 2 else "?"

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
                logger.warning(
                    "TASK SKIPPED (propagated — all deps skipped)",
                    task_id=task_id,
                    title=task_title,
                    deps=deps,
                    dep_statuses=dep_statuses,
                )

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

    Persists per-(model, agent_type) aggregates to the ``model_stats`` DB
    table. Phase C.5 (2026-05-05): in-memory Prometheus counters are
    emitted by ``hallederiz_kadir.caller`` directly — this function used
    to ALSO emit them which inflated counters ~2.5× because every ReAct
    iter passed through both paths. Audit:
    ``docs/handoff/2026-05-04-record-model-call-audit.md``.
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


async def record_call_tokens(
    *,
    task_id: int | None,
    agent_type: str | None,
    workflow_step_id: str | None,
    workflow_phase: str | None,
    call_category: str,
    model: str,
    provider: str,
    is_streaming: bool,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    total_tokens: int,
    duration_ms: int,
    iteration_n: int,
    success: bool,
) -> None:
    """Persist per-call token usage. Single INSERT, no upsert.

    Feeds step_token_stats rollup (Beckman cron) and offline calibration.
    """
    db = await get_db()
    await db.execute(
        """INSERT INTO model_call_tokens
           (task_id, agent_type, workflow_step_id, workflow_phase, call_category,
            model, provider, is_streaming, prompt_tokens, completion_tokens,
            reasoning_tokens, total_tokens, duration_ms, iteration_n, success)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (task_id, agent_type, workflow_step_id, workflow_phase, call_category,
         model, provider, int(is_streaming), prompt_tokens, completion_tokens,
         reasoning_tokens, total_tokens, duration_ms, iteration_n, int(success)),
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
    ttl_seconds: int = 3600,
) -> bool:
    """Acquire an advisory lock on a file. Returns True if acquired.

    Z10 T1A: every acquire now stamps ``expires_at = now + ttl_seconds`` so
    crashed-task orphans get reaped by :func:`sweep_file_locks`. Default
    TTL is 1 hour, mirroring the original spec column default.
    """
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO file_locks "
            "(filepath, mission_id, task_id, agent_type, expires_at) "
            "VALUES (?, ?, ?, ?, datetime('now', ?))",
            (filepath, mission_id, task_id, agent_type, f"+{int(ttl_seconds)} seconds"),
        )
        await db.commit()
        return True
    except Exception:
        # UNIQUE constraint violation → already locked
        return False


async def sweep_file_locks() -> int:
    """Release orphan file_locks. Z10 T1A.

    A row is orphan if either:
      * ``expires_at`` is in the past, OR
      * its owning ``task_id`` is no longer pending or running (crashed,
        cancelled, completed without explicit release).

    Returns the number of rows released. Wired into Beckman's cron via the
    ``file_locks_sweep`` internal cadence (60s).
    """
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM file_locks "
        "WHERE (expires_at IS NOT NULL AND expires_at < datetime('now')) "
        "   OR (task_id IS NOT NULL "
        "       AND task_id NOT IN ("
        "           SELECT id FROM tasks WHERE status IN ('pending','running')"
        "       )"
        "   )"
    )
    released = cursor.rowcount or 0
    await db.commit()
    if released:
        logger.info(f"sweep_file_locks: released {released} orphan lock(s)")
    return int(released)


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


async def record_streaming_guard_outcome(
    *,
    guard_name: str,
    action: str,
    note: str | None = None,
    task_id: int | None = None,
    mission_id: int | None = None,
) -> None:
    """Z1 T5C (B3) — insert one streaming_guard_log row.

    Fire-and-forget from the streaming pipeline's sink. Errors are
    swallowed: a broken telemetry table must never abort LLM streaming.
    Schema CHECK enforces ``action IN ('warn','halt','fix')``.
    """
    if action not in ("warn", "halt", "fix"):
        return
    try:
        db = await get_db()
        await db.execute(
            "INSERT INTO streaming_guard_log "
            "(mission_id, task_id, guard_name, action, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (mission_id, task_id, guard_name, action, (note or "")[:500]),
        )
    except Exception:
        pass


async def record_founder_signoff(
    *,
    mission_id: int,
    doc_type: str,
    signature_hash: str | None = None,
) -> dict:
    """Z1 T5A (P6) — upsert a founder_signoffs row.

    PRIMARY KEY (mission_id, doc_type) — re-signing replaces the prior row
    so signature_hash always reflects the latest signed body.
    """
    db = await get_db()
    await db.execute(
        """
        INSERT INTO founder_signoffs (mission_id, doc_type, signature_hash)
        VALUES (?, ?, ?)
        ON CONFLICT(mission_id, doc_type) DO UPDATE SET
            signed_at = CURRENT_TIMESTAMP,
            signature_hash = excluded.signature_hash
        """,
        (int(mission_id), str(doc_type), signature_hash),
    )
    await db.commit()
    return {"ok": True, "mission_id": int(mission_id), "doc_type": str(doc_type)}


async def get_founder_signoffs(mission_id: int) -> set[str]:
    """Return the set of `doc_type`s that have a founder_signoffs row."""
    try:
        db = await get_db()
        cur = await db.execute(
            "SELECT doc_type FROM founder_signoffs WHERE mission_id = ?",
            (int(mission_id),),
        )
        rows = await cur.fetchall()
        return {r[0] for r in rows if r and r[0]}
    except Exception:
        return set()


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


# ───────────────────────────────────────────────────────────────────────────
# Z10 T1C — schema migration ledger + provenance + audit + confirmation APIs
# ───────────────────────────────────────────────────────────────────────────


async def apply_migration(
    version: str,
    sql: str,
    reversal_sql: str | None,
    description: str,
) -> bool:
    """Apply a DDL migration inside a single transaction and record it.

    Idempotent: if ``version`` is already in ``schema_migrations`` this is a
    no-op and returns False. On a fresh apply, runs ``sql`` and inserts the
    ledger row inside one BEGIN/COMMIT block — any failure rolls back the
    DDL *and* the ledger insert so the next call retries cleanly.

    ``sql`` may contain multiple statements separated by ``;`` — they are
    executed via ``executescript`` after BEGIN.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)
    )
    if await cur.fetchone():
        return False

    try:
        await db.execute("BEGIN")
        # Execute each statement individually so the transaction is honoured
        # (executescript() issues its own COMMIT mid-flight).
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            await db.execute(stmt)
        await db.execute(
            "INSERT INTO schema_migrations "
            "(version, sql, reversal_sql, description) "
            "VALUES (?, ?, ?, ?)",
            (version, sql, reversal_sql, description),
        )
        await db.execute("COMMIT")
        logger.info(f"Applied migration {version}: {description}")
        return True
    except Exception as e:
        try:
            await db.execute("ROLLBACK")
        except Exception:
            pass
        logger.error(f"Migration {version} failed (rolled back): {e}")
        raise


async def record_artifact_write(
    path: str,
    task_id: int | None = None,
    step_id: str | None = None,
    model_id: str | None = None,
    retry_n: int = 0,
    reviewer_verdict_id: int | None = None,
    mission_id: int | None = None,
) -> int:
    """Insert a row into ``artifact_provenance``. Returns the new row id."""
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO artifact_provenance "
        "(path, task_id, step_id, model_id, retry_n, "
        " reviewer_verdict_id, mission_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            path,
            task_id,
            step_id,
            model_id,
            retry_n,
            reviewer_verdict_id,
            mission_id,
        ),
    )
    await db.commit()
    return cur.lastrowid or 0


async def get_artifact_provenance(path: str) -> list[dict]:
    """Return chronological writes to ``path`` (most recent first).

    Each row carries the provenance core plus, when available, the owning
    task's ``agent_type`` and the summed token counts from
    ``model_call_tokens`` for that task.
    """
    db = await get_db()
    cur = await db.execute(
        """
        SELECT
            ap.written_at,
            ap.task_id,
            ap.step_id,
            ap.model_id,
            ap.retry_n,
            ap.mission_id,
            ap.reviewer_verdict_id,
            t.agent_type AS agent_type,
            (SELECT COALESCE(SUM(mct.prompt_tokens), 0)
             FROM model_call_tokens mct
             WHERE mct.task_id = ap.task_id) AS prompt_tokens,
            (SELECT COALESCE(SUM(mct.completion_tokens), 0)
             FROM model_call_tokens mct
             WHERE mct.task_id = ap.task_id) AS completion_tokens
        FROM artifact_provenance ap
        LEFT JOIN tasks t ON t.id = ap.task_id
        WHERE ap.path = ?
        ORDER BY ap.written_at DESC, ap.id DESC
        """,
        (path,),
    )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


async def record_action_event(
    verb: str,
    reversibility: str,
    mission_id: int | None,
    task_id: int | None,
    payload: dict | None,
    status: str,
) -> int:
    """Append a per-action audit row to ``registry_events`` (scope='action').

    The existing ``event`` column carries the verb (mirrors ``verb`` for
    backward-compatible target/event scans); ``payload_json`` carries the
    JSON-serialized payload + status. Returns the new row id.
    """
    db = await get_db()
    try:
        payload_json = json.dumps(
            {"payload": payload or {}, "status": status},
            default=str,
        )
    except Exception:
        payload_json = json.dumps(
            {"payload": str(payload), "status": status}
        )
    cur = await db.execute(
        "INSERT INTO registry_events "
        "(scope, target, event, payload_json, "
        " mission_id, task_id, verb, reversibility) "
        "VALUES ('action', ?, ?, ?, ?, ?, ?, ?)",
        (
            verb,
            verb,
            payload_json,
            mission_id,
            task_id,
            verb,
            reversibility,
        ),
    )
    await db.commit()
    return cur.lastrowid or 0


async def request_confirmation(
    task_id: int,
    verb: str,
    reversibility: str,
    payload_summary: str | None = None,
) -> int:
    """Open a confirmation request. Returns id with verdict='pending'."""
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO action_confirmations "
        "(task_id, verb, reversibility, payload_summary, verdict) "
        "VALUES (?, ?, ?, ?, 'pending')",
        (task_id, verb, reversibility, payload_summary),
    )
    await db.commit()
    return cur.lastrowid or 0


async def check_confirmation(confirmation_id: int) -> dict:
    """Return ``{'verdict': str, 'responded_at': ts | None}`` for a request.

    Verdict is one of ``pending`` / ``approved`` / ``rejected``. If the row
    does not exist returns ``{'verdict': 'missing', 'responded_at': None}``.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT verdict, responded_at FROM action_confirmations WHERE id = ?",
        (confirmation_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return {"verdict": "missing", "responded_at": None}
    return {"verdict": row[0], "responded_at": row[1]}


async def resolve_confirmation(
    confirmation_id: int, verdict: str
) -> None:
    """Stamp a confirmation as ``approved`` or ``rejected`` with now()."""
    if verdict not in ("approved", "rejected"):
        raise ValueError(
            f"verdict must be 'approved' or 'rejected', got {verdict!r}"
        )
    db = await get_db()
    await db.execute(
        "UPDATE action_confirmations "
        "SET verdict = ?, responded_at = CURRENT_TIMESTAMP "
        "WHERE id = ?",
        (verdict, confirmation_id),
    )
    await db.commit()


# ───────────────────────────────────────────────────────────────────────────
# Z9 T1A — growth zone CRUD helpers (hypotheses / experiment_variants /
# growth_events)
# ───────────────────────────────────────────────────────────────────────────
#
# Schema lives in init_db() (apply_migration 2026-05-15-z9-*). These async
# helpers mirror the recipe_pin_log / mission_lessons conventions: JSON blobs
# are json.dumps()'d on the way in and json.loads()'d on the way out; all
# timestamp columns store SQLite space-separated datetime via datetime('now')
# or strftime("%Y-%m-%d %H:%M:%S") — never datetime.isoformat() (T-form).

_HYP_SUPPRESSION_DAYS = 90


async def insert_hypothesis(
    mission_id: int | None,
    feature: str,
    predicted: dict,
    window_seconds: int,
    dedup_key: str,
) -> int:
    """Insert a new pending hypothesis row. Returns the new row id.

    Refuses (returns -1) if a row with the same ``dedup_key`` is still
    suppressed — i.e. has ``suppressed_until`` in the future. A refuted
    feature/metric pair is not re-predicted until its 90-day cool-off ends.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM hypotheses "
        "WHERE dedup_key = ? AND suppressed_until IS NOT NULL "
        "AND suppressed_until > datetime('now') LIMIT 1",
        (dedup_key,),
    )
    if await cur.fetchone():
        logger.info(
            "insert_hypothesis: refused (dedup_key %s still suppressed)",
            dedup_key,
        )
        return -1

    cur = await db.execute(
        "INSERT INTO hypotheses "
        "(mission_id, feature, predicted_json, verdict, window_seconds, "
        " dedup_key) "
        "VALUES (?, ?, ?, 'pending', ?, ?)",
        (
            mission_id,
            feature,
            json.dumps(predicted or {}),
            window_seconds,
            dedup_key,
        ),
    )
    await db.commit()
    return cur.lastrowid or 0


async def record_hypothesis_verdict(
    hypothesis_id: int,
    actual: dict,
    verdict: str,
) -> None:
    """Record a verdict on a hypothesis.

    Sets ``actual_json``, ``verdict`` and stamps ``measured_at`` with the
    current SQLite datetime. When ``verdict == 'refuted'`` also sets
    ``suppressed_until`` to now + 90 days so the feature/metric pair is
    not re-predicted during the cool-off window.
    """
    db = await get_db()
    measured_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suppressed_until = None
    if verdict == "refuted":
        suppressed_until = (
            datetime.now() + timedelta(days=_HYP_SUPPRESSION_DAYS)
        ).strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "UPDATE hypotheses "
        "SET actual_json = ?, verdict = ?, measured_at = ?, "
        "    suppressed_until = ? "
        "WHERE id = ?",
        (
            json.dumps(actual or {}),
            verdict,
            measured_at,
            suppressed_until,
            hypothesis_id,
        ),
    )
    await db.commit()


async def get_pending_hypotheses(
    mission_id: int | None = None,
) -> list[dict]:
    """Return hypotheses with ``verdict='pending'`` (most recent first).

    ``mission_id=None`` returns pending hypotheses across all missions.
    ``predicted_json`` / ``actual_json`` are decoded back to dicts.
    """
    db = await get_db()
    if mission_id is not None:
        cur = await db.execute(
            "SELECT * FROM hypotheses "
            "WHERE verdict = 'pending' AND mission_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (mission_id,),
        )
    else:
        cur = await db.execute(
            "SELECT * FROM hypotheses "
            "WHERE verdict = 'pending' "
            "ORDER BY created_at DESC, id DESC"
        )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        for k in ("predicted_json", "actual_json"):
            try:
                d[k] = json.loads(d[k]) if d.get(k) else None
            except Exception:
                pass
        result.append(d)
    return result


async def insert_growth_event(
    mission_id: int | None,
    kind: str,
    properties: dict,
    segment: str | None = None,
) -> int:
    """Append a row to ``growth_events``. Returns the new row id.

    ``properties`` is JSON-serialized into ``properties_json``;
    ``occurred_at`` defaults to the current SQLite datetime.
    """
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO growth_events "
        "(mission_id, kind, properties_json, segment) "
        "VALUES (?, ?, ?, ?)",
        (mission_id, kind, json.dumps(properties or {}), segment),
    )
    await db.commit()
    return cur.lastrowid or 0


async def get_growth_events(
    mission_id: int | None = None,
    kind: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return growth_events rows filtered by mission / kind / time.

    ``since`` is an inclusive SQLite datetime string ('YYYY-MM-DD HH:MM:SS').
    Any combination of filters may be omitted. ``limit`` caps the row count
    (most recent first). Each row exposes the decoded payload under BOTH
    ``properties`` and ``properties_json`` keys. Most recent first.
    """
    db = await get_db()
    clauses = []
    params: list = []
    if mission_id is not None:
        clauses.append("mission_id = ?")
        params.append(mission_id)
    if kind is not None:
        clauses.append("kind = ?")
        params.append(kind)
    if since is not None:
        clauses.append("occurred_at >= ?")
        params.append(since)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT * FROM growth_events" + where
        + " ORDER BY occurred_at DESC, id DESC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    cur = await db.execute(sql, tuple(params))
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        decoded = None
        try:
            decoded = (
                json.loads(d["properties_json"])
                if d.get("properties_json")
                else None
            )
        except Exception:
            decoded = d.get("properties_json")
        d["properties_json"] = decoded
        d["properties"] = decoded
        result.append(d)
    return result


# Z9 T4E — reinforce loop. A confirmed hypothesis verdict bumps the model
# that built the winning feature: a small, decaying nudge so old wins fade
# and the loop never locks in an early winner. The nudge is written as a
# dedicated ``model_pick_log`` row (call_category='reinforce', reinforce
# column carries the +bonus) — fatih_hoca.grading.reinforce_bonus() reads
# these rows and folds a time-decayed sum into the model's perf_score.
REINFORCE_NUDGE: float = 0.05  # founder-decided: +0.05 per confirmed verdict


async def record_reinforce_nudge(
    model: str,
    *,
    task_name: str = "hypothesis_verdict",
    amount: float = REINFORCE_NUDGE,
    provider: str = "local",
    hypothesis_id: int | None = None,
) -> None:
    """Write a confirmed-verdict reinforce nudge for ``model``.

    Fire-and-forget telemetry — never raises into the caller. The row is a
    ``model_pick_log`` entry tagged ``call_category='reinforce'`` with the
    bonus stored in the ``reinforce`` column. Fatih Hoca's grading layer
    sums these with a 50%-per-30-day decay so the influence fades.
    """
    if not model:
        return
    try:
        db = await get_db()
        snapshot = f"hypothesis_id={hypothesis_id}" if hypothesis_id else ""
        await db.execute(
            "INSERT INTO model_pick_log "
            "(task_name, picked_model, picked_score, call_category, "
            " candidates_json, snapshot_summary, success, provider, "
            " outcome, reinforce) "
            "VALUES (?, ?, ?, 'reinforce', '[]', ?, 1, ?, 'reinforce', ?)",
            (
                task_name,
                model,
                0.0,
                snapshot,
                provider,
                float(amount),
            ),
        )
        await db.commit()
        logger.info(
            "reinforce nudge recorded model=%s amount=%.3f hyp=%s",
            model, amount, hypothesis_id,
        )
    except Exception as e:  # noqa: BLE001 — telemetry must never propagate
        logger.warning("record_reinforce_nudge failed: %s", e)


async def insert_variant(
    mission_id: int | None,
    hypothesis_id: int | None,
    variant_name: str,
    assignment_rule: str,
) -> int:
    """Insert an ``experiment_variants`` row (status='active'). Returns id."""
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO experiment_variants "
        "(mission_id, hypothesis_id, variant_name, assignment_rule, status) "
        "VALUES (?, ?, ?, ?, 'active')",
        (mission_id, hypothesis_id, variant_name, assignment_rule),
    )
    await db.commit()
    return cur.lastrowid or 0


async def update_variant_status(variant_id: int, status: str) -> None:
    """Update an experiment variant's status.

    When ``status`` is a terminal state ('winner' | 'loser' | 'stopped')
    ``retired_at`` is stamped with the current SQLite datetime. Setting it
    back to 'active' clears ``retired_at``.
    """
    db = await get_db()
    if status in ("winner", "loser", "stopped"):
        retired_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "UPDATE experiment_variants "
            "SET status = ?, retired_at = ? WHERE id = ?",
            (status, retired_at, variant_id),
        )
    else:
        await db.execute(
            "UPDATE experiment_variants "
            "SET status = ?, retired_at = NULL WHERE id = ?",
            (status, variant_id),
        )
    await db.commit()


async def get_variants(
    mission_id: int | None = None,
    hypothesis_id: int | None = None,
    status: str | None = None,
) -> list[dict]:
    """Return ``experiment_variants`` rows filtered by mission / hypothesis.

    Z9 T5D — read surface for the A/B harness (verdict evaluation,
    retire_variant, /experiment Telegram command). Any combination of
    filters may be omitted; ``status`` narrows to one lifecycle state.
    Most recent first.
    """
    db = await get_db()
    clauses: list = []
    params: list = []
    if mission_id is not None:
        clauses.append("mission_id = ?")
        params.append(mission_id)
    if hypothesis_id is not None:
        clauses.append("hypothesis_id = ?")
        params.append(hypothesis_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    cur = await db.execute(
        "SELECT * FROM experiment_variants" + where
        + " ORDER BY created_at DESC, id DESC",
        tuple(params),
    )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows]


# ───────────────────────────────────────────────────────────────────────────
# Z10 T2A — cost transparency wiring
# ───────────────────────────────────────────────────────────────────────────
#
# Layout:
#   D1 — ensure_mission_cost_row + token-write hook (record_call_cost)
#   D2 — get_cost_by_iteration, get_mission_cost_breakdown
#   D4 — get_pending_cost_alerts + check_and_write_mission_budget_alerts
#   D5 — estimate_task_cost, finalize_task_actual_cost
#   D8 — record_vendor_cost
# D3/D6/D7 wire from helpers in cost_wiring.py + coulson + beckman cron.


# Conservative per-task-kind cost fallbacks (USD) when <5 historical samples.
_TASK_KIND_DEFAULT_COST_USD: dict[str, float] = {
    "coder": 0.05,
    "implementer": 0.05,
    "fixer": 0.04,
    "test_generator": 0.04,
    "reviewer": 0.03,
    "code_reviewer": 0.03,
    "visual_reviewer": 0.03,
    "planner": 0.04,
    "architect": 0.05,
    "researcher": 0.03,
    "analyst": 0.03,
    "shopping_advisor": 0.02,
    "product_researcher": 0.02,
    "deal_analyst": 0.02,
    "classifier": 0.005,
    "grader": 0.005,
    "mechanical": 0.0,
}
_TASK_KIND_DEFAULT_FALLBACK_USD = 0.02


async def ensure_mission_cost_row(
    mission_id: int,
    budget_ceiling_usd: float | None = None,
) -> None:
    """Idempotent: insert/update the mission's row in ``cost_budgets``.

    Inserts ``scope='mission', scope_id=str(mission_id)`` if absent.
    When ``budget_ceiling_usd`` is provided, the ceiling is set/updated.
    """
    db = await get_db()
    scope_id = str(mission_id)
    today = utc_now().strftime("%Y-%m-%d")
    cur = await db.execute(
        "SELECT id FROM cost_budgets WHERE scope = ? AND scope_id = ?",
        ("mission", scope_id),
    )
    existing = await cur.fetchone()
    if existing is None:
        await db.execute(
            "INSERT INTO cost_budgets "
            "(scope, scope_id, daily_limit, total_limit, "
            " spent_today, spent_total, last_reset_date, "
            " budget_ceiling_usd) "
            "VALUES (?, ?, 0, 0, 0, 0, ?, ?)",
            ("mission", scope_id, today, budget_ceiling_usd),
        )
    elif budget_ceiling_usd is not None:
        await db.execute(
            "UPDATE cost_budgets SET budget_ceiling_usd = ? "
            "WHERE scope = ? AND scope_id = ?",
            (budget_ceiling_usd, "mission", scope_id),
        )
    await db.commit()


async def record_call_cost(
    task_id: int | None,
    cost_usd: float,
) -> None:
    """Stamp ``cost_usd`` onto the most recent ``model_call_tokens`` row
    for ``task_id`` AND increment the matching mission's ``cost_budgets``.

    Called from the token-accounting writer right after
    ``record_call_tokens``. Best-effort: any failure leaves accounting
    silent rather than blocking the LLM hot path.
    """
    if not task_id or cost_usd <= 0:
        return
    db = await get_db()
    # Stamp cost on the most-recent matching token row.
    try:
        await db.execute(
            "UPDATE model_call_tokens SET cost_usd = ? "
            "WHERE id = (SELECT id FROM model_call_tokens "
            "            WHERE task_id = ? "
            "            ORDER BY id DESC LIMIT 1)",
            (cost_usd, task_id),
        )
    except Exception:
        pass

    # Resolve mission_id for this task and increment the mission budget.
    try:
        cur = await db.execute(
            "SELECT mission_id FROM tasks WHERE id = ?", (task_id,)
        )
        row = await cur.fetchone()
        mission_id = row[0] if row else None
        if mission_id is None:
            await db.commit()
            return
        await ensure_mission_cost_row(int(mission_id))
        today = utc_now().strftime("%Y-%m-%d")
        scope_id = str(int(mission_id))
        cur = await db.execute(
            "SELECT last_reset_date FROM cost_budgets "
            "WHERE scope = ? AND scope_id = ?",
            ("mission", scope_id),
        )
        row = await cur.fetchone()
        last_reset = row[0] if row else None
        if last_reset != today:
            await db.execute(
                "UPDATE cost_budgets "
                "SET spent_today = ?, spent_total = spent_total + ?, "
                "    last_reset_date = ? "
                "WHERE scope = ? AND scope_id = ?",
                (cost_usd, cost_usd, today, "mission", scope_id),
            )
        else:
            await db.execute(
                "UPDATE cost_budgets "
                "SET spent_today = spent_today + ?, "
                "    spent_total = spent_total + ? "
                "WHERE scope = ? AND scope_id = ?",
                (cost_usd, cost_usd, "mission", scope_id),
            )
        await db.commit()
    except Exception as e:
        logger.debug(f"record_call_cost mission rollup skipped: {e}")
        try:
            await db.commit()
        except Exception:
            pass


async def get_cost_by_iteration(mission_id: int) -> list[dict]:
    """Return per-iteration breakdown for a mission.

    Rows ordered by ``iteration_n`` ascending. Each row carries
    ``iteration_n``, ``prompt_tokens``, ``completion_tokens``,
    ``total_tokens``, ``cost_usd`` and ``calls``.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT iteration_n, prompt_tokens, completion_tokens, "
        "       total_tokens, cost_usd, calls "
        "FROM cost_by_iteration "
        "WHERE mission_id = ? "
        "ORDER BY iteration_n ASC",
        (mission_id,),
    )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


async def get_mission_cost_breakdown(mission_id: int) -> dict:
    """Return ``{first_pass_usd, retry_usd, vendor_usd, total_usd}``.

    ``first_pass = iteration_n == 0``, ``retry = iteration_n >= 1``.
    ``vendor`` sums ``cost_budgets`` rows scoped ``vendor:<name>`` for
    this mission.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT iteration_n, COALESCE(SUM(cost_usd), 0) AS c "
        "FROM cost_by_iteration "
        "WHERE mission_id = ? "
        "GROUP BY (iteration_n == 0)",
        (mission_id,),
    )
    rows = await cur.fetchall()
    first_pass = 0.0
    retry = 0.0
    for r in rows:
        iter_n = r[0]
        c = float(r[1] or 0.0)
        if iter_n == 0:
            first_pass += c
        else:
            retry += c

    # Sum vendor:* scopes for this mission.
    cur = await db.execute(
        "SELECT COALESCE(SUM(spent_total), 0) AS c "
        "FROM cost_budgets "
        "WHERE scope LIKE 'vendor:%' AND scope_id = ?",
        (str(mission_id),),
    )
    row = await cur.fetchone()
    vendor = float(row[0] or 0.0)

    total = first_pass + retry + vendor
    return {
        "first_pass_usd": first_pass,
        "retry_usd": retry,
        "vendor_usd": vendor,
        "total_usd": total,
    }


async def get_pending_cost_alerts() -> list[dict]:
    """Return rows in ``mission_budget_alerts`` with ``drained_at IS NULL``."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, mission_id, threshold, total_usd, posted_at "
        "FROM mission_budget_alerts "
        "WHERE drained_at IS NULL "
        "ORDER BY posted_at ASC"
    )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


async def mark_cost_alert_drained(alert_id: int) -> None:
    """Stamp an alert as drained (T2B drain hook)."""
    db = await get_db()
    await db.execute(
        "UPDATE mission_budget_alerts SET drained_at = CURRENT_TIMESTAMP "
        "WHERE id = ?",
        (alert_id,),
    )
    await db.commit()


async def check_and_write_mission_budget_alerts() -> int:
    """Sweep missions with a ceiling; insert threshold rows.

    Returns number of new alert rows written. Idempotent: the
    ``UNIQUE(mission_id, threshold)`` index prevents duplicates so callers
    can run this every 5 minutes without flooding.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT id, status FROM missions "
        "WHERE status IN ('running', 'pending', 'active')"
    )
    missions = [(int(r[0]), r[1]) for r in await cur.fetchall()]

    written = 0
    THRESHOLDS = (0.5, 0.75, 0.9)
    for mission_id, _status in missions:
        # Find ceiling for this mission.
        cur = await db.execute(
            "SELECT budget_ceiling_usd FROM cost_budgets "
            "WHERE scope = ? AND scope_id = ?",
            ("mission", str(mission_id)),
        )
        row = await cur.fetchone()
        if not row or row[0] is None or row[0] <= 0:
            continue
        ceiling = float(row[0])
        breakdown = await get_mission_cost_breakdown(mission_id)
        total = breakdown["total_usd"]
        if total <= 0:
            continue
        ratio = total / ceiling
        for t in THRESHOLDS:
            if ratio >= t:
                try:
                    await db.execute(
                        "INSERT INTO mission_budget_alerts "
                        "(mission_id, threshold, total_usd) "
                        "VALUES (?, ?, ?)",
                        (mission_id, t, total),
                    )
                    written += 1
                except Exception:
                    # UNIQUE constraint hit — already alerted at this threshold.
                    pass
    if written:
        await db.commit()
    return written


async def estimate_task_cost(
    model_id: str | None,
    task_kind: str | None,
) -> float:
    """Return an estimated USD cost for a task on a model+kind.

    Methodology: average historical ``cost_usd`` from
    ``model_call_tokens × tasks`` where ``model = model_id`` and
    ``tasks.agent_type = task_kind``. Requires >=5 samples; otherwise
    falls back to ``_TASK_KIND_DEFAULT_COST_USD``.
    """
    if not task_kind:
        return _TASK_KIND_DEFAULT_FALLBACK_USD
    if not model_id:
        return _TASK_KIND_DEFAULT_COST_USD.get(
            task_kind, _TASK_KIND_DEFAULT_FALLBACK_USD
        )
    db = await get_db()
    cur = await db.execute(
        "SELECT COUNT(*) AS n, AVG(COALESCE(mct.cost_usd, 0)) AS avg_c "
        "FROM model_call_tokens mct "
        "JOIN tasks t ON t.id = mct.task_id "
        "WHERE mct.model = ? AND t.agent_type = ? "
        "  AND mct.cost_usd IS NOT NULL AND mct.cost_usd > 0",
        (model_id, task_kind),
    )
    row = await cur.fetchone()
    n = int(row[0] or 0) if row else 0
    avg_c = float(row[1] or 0.0) if row else 0.0
    if n >= 5 and avg_c > 0:
        return avg_c
    return _TASK_KIND_DEFAULT_COST_USD.get(
        task_kind, _TASK_KIND_DEFAULT_FALLBACK_USD
    )


async def set_task_estimated_cost(task_id: int, cost_usd: float) -> None:
    """Stamp ``estimated_cost_usd`` on a task row."""
    db = await get_db()
    await db.execute(
        "UPDATE tasks SET estimated_cost_usd = ? WHERE id = ?",
        (cost_usd, task_id),
    )
    await db.commit()


async def finalize_task_actual_cost(task_id: int) -> float:
    """Sum ``model_call_tokens.cost_usd`` for a task → ``tasks.actual_cost_usd``.

    Returns the computed actual cost.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM model_call_tokens "
        "WHERE task_id = ?",
        (task_id,),
    )
    row = await cur.fetchone()
    actual = float(row[0] or 0.0)
    await db.execute(
        "UPDATE tasks SET actual_cost_usd = ? WHERE id = ?",
        (actual, task_id),
    )
    await db.commit()
    return actual


async def record_vendor_cost(
    mission_id: int,
    vendor: str,
    usd: float,
    line_item: str,
) -> None:
    """Append vendor cost to ``cost_budgets`` row ``scope='vendor:{vendor}'``.

    Increments ``spent_today`` + ``spent_total``. Idempotently creates the
    row on first call. ``line_item`` is currently recorded only via the
    audit log helper (best-effort) — schema-wise the row carries totals.
    """
    if usd < 0:
        raise ValueError("usd must be non-negative")
    db = await get_db()
    scope = f"vendor:{vendor}"
    scope_id = str(mission_id)
    today = utc_now().strftime("%Y-%m-%d")
    cur = await db.execute(
        "SELECT id, last_reset_date FROM cost_budgets "
        "WHERE scope = ? AND scope_id = ?",
        (scope, scope_id),
    )
    row = await cur.fetchone()
    if row is None:
        await db.execute(
            "INSERT INTO cost_budgets "
            "(scope, scope_id, daily_limit, total_limit, "
            " spent_today, spent_total, last_reset_date) "
            "VALUES (?, ?, 0, 0, ?, ?, ?)",
            (scope, scope_id, usd, usd, today),
        )
    else:
        last_reset = row[1]
        if last_reset != today:
            await db.execute(
                "UPDATE cost_budgets "
                "SET spent_today = ?, spent_total = spent_total + ?, "
                "    last_reset_date = ? "
                "WHERE scope = ? AND scope_id = ?",
                (usd, usd, today, scope, scope_id),
            )
        else:
            await db.execute(
                "UPDATE cost_budgets "
                "SET spent_today = spent_today + ?, "
                "    spent_total = spent_total + ? "
                "WHERE scope = ? AND scope_id = ?",
                (usd, usd, scope, scope_id),
            )
    await db.commit()
    # Lightweight audit breadcrumb; best-effort.
    try:
        await record_action_event(
            verb="record_vendor_cost",
            reversibility="irreversible",
            mission_id=mission_id,
            task_id=None,
            payload={"vendor": vendor, "usd": usd, "line_item": line_item},
            status="ok",
        )
    except Exception:
        pass


async def get_mission_quality_mode(mission_id: int) -> str:
    """Return the mission's ``quality_mode`` (``quick``/``balanced``/``thorough``).

    Defaults to ``balanced`` when the row is missing or the column is NULL.
    """
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT quality_mode FROM missions WHERE id = ?", (mission_id,)
        )
        row = await cur.fetchone()
        if row is None or row[0] is None:
            return "balanced"
        mode = str(row[0]).strip().lower()
        if mode in ("quick", "balanced", "thorough"):
            return mode
    except Exception:
        pass
    return "balanced"


async def set_mission_quality_mode(mission_id: int, mode: str) -> None:
    """Set ``missions.quality_mode``. Raises on invalid mode."""
    if mode not in ("quick", "balanced", "thorough"):
        raise ValueError(
            f"quality_mode must be quick/balanced/thorough, got {mode!r}"
        )
    db = await get_db()
    await db.execute(
        "UPDATE missions SET quality_mode = ? WHERE id = ?",
        (mode, mission_id),
    )
    await db.commit()


# ───────────────────────────────────────────────────────────────────────────
# Z10 T3C — mission green-tag ledger + mission-scoped row snapshot
# ───────────────────────────────────────────────────────────────────────────

# Tables whose rows belong to a single mission. Used by the snapshot/restore
# helpers in the rollback_mission verb. Each entry: (table_name, mission_fk).
MISSION_SCOPED_TABLES: list[tuple[str, str]] = [
    ("tasks", "mission_id"),
    ("task_events", "mission_id"),
    ("mission_events", "mission_id"),
    ("artifact_provenance", "mission_id"),
    ("mission_pacing_snapshots", "mission_id"),
    ("mission_tradeoff_prompts", "mission_id"),
    ("mission_budget_alerts", "mission_id"),
]


async def record_green_tag(
    mission_id: int,
    task_id: int,
    git_tag: str,
    db_snapshot_path: str,
    chroma_snapshot_path: str,
    schema_migrations_at: str | None = None,
) -> int:
    """Insert a row into ``mission_green_tags``.

    Idempotent: returns the existing rowid when ``(mission_id, task_id)`` is
    already present.
    """
    db = await get_db()
    async with _get_tx_lock(mission_id):
        cur = await db.execute(
            "SELECT id FROM mission_green_tags "
            "WHERE mission_id = ? AND task_id = ?",
            (mission_id, task_id),
        )
        row = await cur.fetchone()
        if row is not None:
            return int(row[0])
        cur = await db.execute(
            "INSERT INTO mission_green_tags "
            "(mission_id, task_id, git_tag, db_snapshot_path, "
            " chroma_snapshot_path, schema_migrations_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                mission_id,
                task_id,
                git_tag,
                db_snapshot_path,
                chroma_snapshot_path,
                schema_migrations_at,
            ),
        )
        await db.commit()
        return int(cur.lastrowid or 0)


async def get_latest_green_tag(
    mission_id: int,
    task_id: int | None = None,
) -> dict | None:
    """Return the most recent green-tag row for ``mission_id`` (or specific task)."""
    db = await get_db()
    if task_id is not None:
        cur = await db.execute(
            "SELECT id, mission_id, task_id, git_tag, db_snapshot_path, "
            "       chroma_snapshot_path, schema_migrations_at, created_at "
            "FROM mission_green_tags "
            "WHERE mission_id = ? AND task_id = ?",
            (mission_id, task_id),
        )
    else:
        cur = await db.execute(
            "SELECT id, mission_id, task_id, git_tag, db_snapshot_path, "
            "       chroma_snapshot_path, schema_migrations_at, created_at "
            "FROM mission_green_tags "
            "WHERE mission_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (mission_id,),
        )
    row = await cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "mission_id": row[1],
        "task_id": row[2],
        "git_tag": row[3],
        "db_snapshot_path": row[4],
        "chroma_snapshot_path": row[5],
        "schema_migrations_at": row[6],
        "created_at": row[7],
    }


async def snapshot_mission_db_rows(mission_id: int) -> dict:
    """Serialize all mission-scoped rows for ``mission_id`` into a dict.

    Shape: ``{"<table>": [row_dict, ...], ...}`` + ``"_meta"`` with the max
    schema_migrations version applied at snapshot time.
    """
    db = await get_db()
    out: dict = {}
    for table, fk in MISSION_SCOPED_TABLES:
        try:
            cur = await db.execute(
                f"SELECT * FROM {table} WHERE {fk} = ?", (mission_id,)
            )
            rows = await cur.fetchall()
        except Exception as e:
            logger.debug(f"snapshot: skip {table} ({e})")
            out[table] = []
            continue
        out[table] = [dict(r) for r in rows]

    try:
        cur = await db.execute(
            "SELECT version FROM schema_migrations "
            "ORDER BY applied_at DESC, version DESC LIMIT 1"
        )
        row = await cur.fetchone()
        out["_meta"] = {"schema_migrations_at": (row[0] if row else None)}
    except Exception:
        out["_meta"] = {"schema_migrations_at": None}
    return out


async def restore_mission_db_rows(mission_id: int, snapshot: dict) -> dict:
    """DELETE-then-INSERT mission-scoped rows from ``snapshot``.

    Held under the mission's tx-lock so concurrent writers for the same
    mission cannot interleave. Returns per-table insert counts.
    """
    db = await get_db()
    counts: dict = {}
    async with _get_tx_lock(mission_id):
        for table, fk in MISSION_SCOPED_TABLES:
            rows = snapshot.get(table) or []
            try:
                await db.execute(
                    f"DELETE FROM {table} WHERE {fk} = ?", (mission_id,)
                )
            except Exception as e:
                logger.debug(f"restore: DELETE {table} skipped: {e}")
                counts[table] = 0
                continue
            inserted = 0
            for r in rows:
                cols = list(r.keys())
                qmarks = ",".join(["?"] * len(cols))
                col_list = ",".join(cols)
                try:
                    await db.execute(
                        f"INSERT OR REPLACE INTO {table} ({col_list}) "
                        f"VALUES ({qmarks})",
                        tuple(r[c] for c in cols),
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning(f"restore: INSERT {table} row failed: {e}")
            counts[table] = inserted
        await db.commit()
    return counts


async def rewind_migrations_to(target_version: str | None) -> dict:
    """Run reversal_sql for every migration applied AFTER ``target_version``.

    Best-effort. Rows with NULL ``reversal_sql`` are SKIPPED and counted in
    ``skipped`` — the snapshot rollback will run against a newer schema and
    may not match the snapshot row shape exactly. Returned dict shape:

        {"rewound": [versions], "skipped": [versions], "failed": [versions]}

    ``target_version=None`` is a no-op.
    """
    out = {"rewound": [], "skipped": [], "failed": []}
    if target_version is None:
        return out
    db = await get_db()
    # Compare by rowid + applied_at so migrations applied within the same
    # CURRENT_TIMESTAMP second still order correctly. Anything with a
    # strictly higher (applied_at, rowid) tuple than the target landed AFTER
    # the snapshot was taken.
    cur = await db.execute(
        "SELECT version, reversal_sql FROM schema_migrations "
        "WHERE (applied_at, rowid) > ("
        "    SELECT applied_at, rowid FROM schema_migrations WHERE version = ?"
        ") "
        "ORDER BY applied_at DESC, rowid DESC",
        (target_version,),
    )
    rows = await cur.fetchall()
    for version, reversal in rows:
        if not reversal:
            out["skipped"].append(version)
            logger.warning(
                f"rewind_migrations_to: NULL reversal_sql for {version!r} — "
                f"cannot rewind; rollback will run against newer schema"
            )
            continue
        try:
            await db.execute("BEGIN")
            for stmt in [s.strip() for s in reversal.split(";") if s.strip()]:
                await db.execute(stmt)
            await db.execute(
                "DELETE FROM schema_migrations WHERE version = ?", (version,)
            )
            await db.execute("COMMIT")
            out["rewound"].append(version)
        except Exception as e:
            try:
                await db.execute("ROLLBACK")
            except Exception:
                pass
            logger.error(f"rewind failed for {version}: {e}")
            out["failed"].append(version)
    return out


async def purge_mission_chroma_collections_via_db(mission_id: int) -> int:
    """Convenience proxy to vector_store.purge_mission_chroma_collections."""
    try:
        from src.memory.vector_store import (
            purge_mission_chroma_collections as _purge,
        )
        return await _purge(mission_id)
    except Exception as e:
        logger.warning(f"purge_mission_chroma_collections proxy failed: {e}")
        return 0


# ───────────────────────────────────────────────────────────────────────────
# Z10 T4B — confidence outcomes + reliability scores (trust calibration)
# ───────────────────────────────────────────────────────────────────────────


async def record_confidence_claim(task_id: int) -> int | None:
    """Record a confidence claim row for ``task_id``.

    Reads ``tasks.confidence_categorical / confidence_numeric / agent_type /
    mission_id`` and the most recent ``model_pick_log`` row for the task
    (matched by task_name=tasks.title). Inserts a row with
    ``outcome_correct=NULL`` and returns its id.

    Returns None when the task lacks any confidence signal — there's
    nothing to attribute and we'd just pollute the table.

    ``task_kind`` is derived from ``tasks.agent_type`` (v1 proxy for
    domain). Later iterations can refine via workflow_step_id or a
    dedicated tag.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT id, title, mission_id, agent_type, confidence_categorical, "
        "       confidence_numeric "
        "FROM tasks WHERE id = ?",
        (task_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return None
    (_, title, mission_id, agent_type, conf_cat, conf_num) = row
    if conf_cat is None and conf_num is None:
        # No confidence claim to record — bail out quietly.
        return None

    # Pick the most recent model_pick_log row for this task. task_id isn't a
    # column on model_pick_log so we use task_name=title (best proxy today).
    picked_model: str | None = None
    picked_at: str | None = None
    if title:
        cur2 = await db.execute(
            "SELECT picked_model, timestamp FROM model_pick_log "
            "WHERE task_name = ? ORDER BY timestamp DESC LIMIT 1",
            (title,),
        )
        prow = await cur2.fetchone()
        await cur2.close()
        if prow:
            picked_model, picked_at = prow[0], prow[1]

    if not picked_model:
        # Fall back: use agent_type as model_id placeholder so the row is
        # still attributable. (Rare — only when pick log was scrubbed.)
        picked_model = f"unknown::{agent_type or 'unknown'}"
    if not picked_at:
        picked_at = utc_now_str() if "utc_now_str" in globals() else None
        if picked_at is None:
            from src.infra.times import utc_now, to_db as _to_db
            picked_at = _to_db(utc_now())

    cur3 = await db.execute(
        "INSERT INTO confidence_outcomes "
        "(task_id, mission_id, agent_type, task_kind, model_id, picked_at,"
        " confidence_categorical, confidence_numeric, outcome_correct) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        (
            task_id,
            mission_id,
            agent_type,
            agent_type,  # task_kind = agent_type (v1 proxy)
            picked_model,
            picked_at,
            conf_cat,
            conf_num,
        ),
    )
    await db.commit()
    return cur3.lastrowid or 0


async def resolve_confidence_outcome(
    claim_id: int,
    correct: bool,
    source: str,
    reviewer_verdict_id: int | None = None,
    notes: str | None = None,
) -> bool:
    """Resolve an outstanding confidence claim. Idempotent: returns False
    if the row was already resolved (outcome_correct IS NOT NULL).
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT outcome_correct FROM confidence_outcomes WHERE id = ?",
        (claim_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return False
    if row[0] is not None:
        return False  # already resolved — idempotent

    await db.execute(
        "UPDATE confidence_outcomes "
        "SET outcome_correct = ?, "
        "    outcome_resolved_at = CURRENT_TIMESTAMP, "
        "    resolution_source = ?, "
        "    reviewer_verdict_id = ?, "
        "    notes = ? "
        "WHERE id = ?",
        (
            1 if correct else 0,
            source,
            reviewer_verdict_id,
            notes,
            claim_id,
        ),
    )
    await db.commit()
    return True


async def outstanding_confidence_claims(
    older_than_hours: int = 24,
) -> list[dict]:
    """Return claims older than ``older_than_hours`` still NULL.

    Used by the reaper job: anything dangling that long either needs a
    downstream signal or should be marked notes='timeout' so calibration
    isn't skewed by hanging tasks.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT id, task_id, mission_id, agent_type, task_kind, model_id,"
        "       picked_at, confidence_categorical, confidence_numeric "
        "FROM confidence_outcomes "
        "WHERE outcome_correct IS NULL "
        "  AND picked_at <= datetime('now', ?)",
        (f"-{int(older_than_hours)} hours",),
    )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    await cur.close()
    return [dict(zip(cols, r)) for r in rows]


async def recompute_reliability_scores() -> int:
    """Aggregate resolved confidence_outcomes into reliability_scores.

    Group by (model_id, task_kind, confidence_categorical); compute
    correct_n / sample_n per bucket; upsert into
    ``confidence_reliability_scores``. Returns the number of rows written.

    Skips rows with NULL outcome_correct or NULL confidence_categorical.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT model_id, task_kind, confidence_categorical, "
        "       COUNT(*) AS sample_n, "
        "       SUM(CASE WHEN outcome_correct=1 THEN 1 ELSE 0 END) AS correct_n "
        "FROM confidence_outcomes "
        "WHERE outcome_correct IS NOT NULL "
        "  AND confidence_categorical IS NOT NULL "
        "  AND task_kind IS NOT NULL "
        "GROUP BY model_id, task_kind, confidence_categorical"
    )
    rows = await cur.fetchall()
    await cur.close()
    written = 0
    for (model_id, task_kind, bucket, sample_n, correct_n) in rows:
        sample_n = int(sample_n or 0)
        correct_n = int(correct_n or 0)
        reliability = (correct_n / sample_n) if sample_n else 0.0
        await db.execute(
            "INSERT INTO confidence_reliability_scores "
            "(model_id, task_kind, confidence_bucket, sample_n, correct_n, "
            " reliability, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(model_id, task_kind, confidence_bucket) DO UPDATE SET "
            "  sample_n=excluded.sample_n, "
            "  correct_n=excluded.correct_n, "
            "  reliability=excluded.reliability, "
            "  updated_at=CURRENT_TIMESTAMP",
            (model_id, task_kind, bucket, sample_n, correct_n, reliability),
        )
        written += 1
    await db.commit()
    return written


async def get_reliability(
    model_id: str, task_kind: str, confidence_bucket: str,
) -> dict | None:
    """Lookup a single reliability row. Returns None when absent."""
    db = await get_db()
    cur = await db.execute(
        "SELECT model_id, task_kind, confidence_bucket, sample_n, "
        "       correct_n, reliability, updated_at "
        "FROM confidence_reliability_scores "
        "WHERE model_id = ? AND task_kind = ? AND confidence_bucket = ?",
        (model_id, task_kind, confidence_bucket),
    )
    row = await cur.fetchone()
    cols = [d[0] for d in cur.description]
    await cur.close()
    if not row:
        return None
    return dict(zip(cols, row))


async def calibration_matrix() -> list[dict]:
    """Full reliability table dump for the /calibration command.

    Sorted by (model_id, task_kind, confidence_bucket).
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT model_id, task_kind, confidence_bucket, sample_n, "
        "       correct_n, reliability, updated_at "
        "FROM confidence_reliability_scores "
        "ORDER BY model_id, task_kind, confidence_bucket"
    )
    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    await cur.close()
    return [dict(zip(cols, r)) for r in rows]

