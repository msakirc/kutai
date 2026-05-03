"""SQLite-backed provider/model registry. Replaces .dead_models.json.

Sync sqlite3 module — selector eligibility filter is on the hot sync
path (`fatih_hoca/selector.py:377`), called multiple times per
candidate per task. Going through the aiosqlite singleton would force
a thread-pool round-trip per check; that's a 30+ check ms-scale tax
per selection. A dedicated sync conn with WAL + 60s busy_timeout
matches the connect_aux pragmas (db.py:_apply_pragmas_sync) so it
plays nicely with the async writer.

Public API mirrors what `fatih_hoca/registry.py` exposed via the JSON
file (mark_dead, is_dead, revive) plus the new provider-level helpers
(mark_provider_dead, is_provider_dead, revive_provider) and per-cause
policy lookup. All writes append a row to `registry_events` for audit.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

from src.infra.logging_config import get_logger

logger = get_logger("infra.registry_store")


# ── Cause policy ─────────────────────────────────────────────────────────
# Per-cause TTL (seconds) and revival policy. Rows store explicit
# expires_at so tweaking these constants doesn't retroactively shift
# already-marked entries.
#
#   ttl_seconds=None  → no auto-expiry; manual /revive only
#   ttl_seconds=N     → expires_at = marked_at + N
#   manual_revive=True → /revive command honored, auto-revive blocked
#
# Tuned defaults follow handoff guidance:
#   auth/manual: never auto-expire (bad key needs operator action)
#   404_permanent: 24h (Gemini *-preview-MM-DD slug retirements)
#   404_transient: 5min (openrouter "no endpoints found" upstream rotations)
#   server_error: 10min (transient backend; let circuit breaker handle short-cycle)
CAUSE_POLICY: dict[str, dict] = {
    "auth":          {"ttl_seconds": None,  "manual_revive": True},
    "manual":        {"ttl_seconds": None,  "manual_revive": True},
    # 404_permanent flipped to manual_revive=True (2026-05-03): runtime
    # call returned `code=model_not_found` from the provider — that's an
    # account/access-level signal, stronger than discovery's "id appears
    # in /v1/models" listing. Without manual_revive the discovery loop
    # cycled revive→call→404→re-mark-dead every refresh on cerebras's
    # gpt-oss-120b and zai-glm-4.7 (free tier lists them but rejects
    # inference). 24h TTL is unchanged — auto-revive still fires when
    # the cause window elapses, so a one-off bad 404 still recovers
    # without operator action; only the discovery-driven early revive
    # is suppressed.
    "404_permanent": {"ttl_seconds": 86400, "manual_revive": True},
    "404_transient": {"ttl_seconds": 300,   "manual_revive": False},
    "server_error":  {"ttl_seconds": 600,   "manual_revive": False},
}


# ── Connection management ────────────────────────────────────────────────
_conn: sqlite3.Connection | None = None
_conn_lock = threading.RLock()
_db_path: str | None = None


def set_db_path(path: str) -> None:
    """Override DB path. Closes any existing connection. Tests use this
    to point at a tmp_path. In production the default (DB_PATH) wins."""
    global _conn, _db_path
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
        _db_path = path


def close() -> None:
    """Close the singleton connection. Tests use between cases."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None


def _resolve_db_path() -> str:
    if _db_path is not None:
        return _db_path
    # Lazy import — avoids hard dep on src.app.config at import time so
    # tests that set_db_path before any production import path stay clean.
    from src.app.config import DB_PATH
    return DB_PATH


def _get_conn() -> sqlite3.Connection:
    """Lazy singleton. WAL + 60s busy_timeout match connect_aux pragmas."""
    global _conn
    with _conn_lock:
        if _conn is None:
            path = _resolve_db_path()
            _conn = sqlite3.connect(
                path,
                isolation_level=None,  # autocommit; explicit BEGIN where needed
                check_same_thread=False,
                timeout=60.0,
            )
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")
            _conn.execute("PRAGMA busy_timeout=60000")
            _ensure_schema(_conn)
        return _conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent CREATE — mirrors src/infra/db.py init_db. Lets this
    module work in tests against a fresh tmp DB without forcing the
    test to call init_db() (async)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            name        TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'active',
            cause       TEXT,
            marked_at   TIMESTAMP,
            revived_at  TIMESTAMP,
            key_hash    TEXT
        )
    """)
    conn.execute("""
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
    conn.execute("""
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_registry_events_target_ts "
        "ON registry_events(target, timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_models_status "
        "ON models(status, provider)"
    )


# ── Time helpers ─────────────────────────────────────────────────────────
def _now_iso() -> str:
    """SQLite datetime() format — space separator, no timezone, UTC.
    Matches CURRENT_TIMESTAMP default used elsewhere in db.py so string
    comparisons against now() work without TZ surprises."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def _expires_at_for(cause: str, marked_at_epoch: float) -> str | None:
    policy = CAUSE_POLICY.get(cause, {})
    ttl = policy.get("ttl_seconds")
    if ttl is None:
        return None
    return time.strftime(
        "%Y-%m-%d %H:%M:%S", time.gmtime(marked_at_epoch + float(ttl))
    )


# ── Audit log ────────────────────────────────────────────────────────────
def _emit_event(
    conn: sqlite3.Connection,
    scope: str,
    target: str,
    event: str,
    cause: str | None = None,
    actor: str = "auto",
    payload: dict | None = None,
) -> None:
    payload_json = json.dumps(payload) if payload else None
    conn.execute(
        "INSERT INTO registry_events "
        "(scope, target, event, cause, actor, payload_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (scope, target, event, cause, actor, payload_json),
    )


# ── Model API ────────────────────────────────────────────────────────────
def register_model(
    litellm_name: str, provider: str, source: str = "yaml"
) -> None:
    """Idempotent register. Inserts as active=alive on first sight.
    Subsequent calls are no-ops — does NOT revive a dead row."""
    if not litellm_name or not provider:
        return
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO models "
        "(litellm_name, provider, status, source) "
        "VALUES (?, ?, 'active', ?)",
        (litellm_name, provider, source),
    )


def mark_dead(
    litellm_name: str,
    cause: str = "404_permanent",
    actor: str = "auto",
    payload: dict | None = None,
) -> None:
    """Mark a model dead with explicit cause. expires_at computed from
    CAUSE_POLICY at write time. Re-marking refreshes marked_at + expires_at
    (sustained failures keep the entry out longer; one-off failures heal
    after TTL).

    Unknown causes treated as '404_permanent' for safety (24h TTL,
    auto-revivable). Caller should always pass a known cause.
    """
    if not litellm_name:
        return
    if cause not in CAUSE_POLICY:
        logger.warning("mark_dead: unknown cause '%s', treating as 404_permanent", cause)
        cause = "404_permanent"
    now_epoch = time.time()
    now_iso = _now_iso()
    expires_at = _expires_at_for(cause, now_epoch)
    conn = _get_conn()
    # Provider field required on the row. If model unknown to registry,
    # derive provider from the litellm_name's first path segment — it's
    # the same convention used everywhere else in the codebase
    # (provider/model[/sub-vendor]).
    provider = litellm_name.split("/")[0] if "/" in litellm_name else "unknown"
    # UPSERT preserves first_seen_at + provider on existing rows.
    conn.execute(
        "INSERT INTO models (litellm_name, provider, status, cause, "
        "marked_at, expires_at, source) "
        "VALUES (?, ?, 'dead', ?, ?, ?, 'runtime') "
        "ON CONFLICT(litellm_name) DO UPDATE SET "
        "status='dead', cause=excluded.cause, "
        "marked_at=excluded.marked_at, expires_at=excluded.expires_at",
        (litellm_name, provider, cause, now_iso, expires_at),
    )
    _emit_event(
        conn, "model", litellm_name, "mark_dead",
        cause=cause, actor=actor, payload=payload,
    )
    logger.warning(
        "registry: marked dead %s — cause=%s, expires_at=%s, actor=%s",
        litellm_name, cause, expires_at or "never", actor,
    )


def is_dead(identifier: str) -> bool:
    """True iff the model is in the dead-set AND its expiry hasn't passed.
    Auto-revives expired rows (status='active', revived_at=now, cause='auto_expiry')
    and emits a revive event so the audit trail captures the transition.
    """
    if not identifier:
        return False
    conn = _get_conn()
    row = conn.execute(
        "SELECT status, expires_at FROM models WHERE litellm_name = ?",
        (identifier,),
    ).fetchone()
    if row is None:
        return False
    if row["status"] != "dead":
        return False
    expires_at = row["expires_at"]
    if expires_at is None:
        # No TTL = manual revive only. Stays dead until /revive.
        return True
    if expires_at > _now_iso():
        return True
    # TTL elapsed — auto-revive in place.
    _auto_revive(conn, identifier)
    return False


def _auto_revive(conn: sqlite3.Connection, litellm_name: str) -> None:
    now_iso = _now_iso()
    conn.execute(
        "UPDATE models SET status='active', revived_at=?, cause=NULL, "
        "expires_at=NULL WHERE litellm_name=?",
        (now_iso, litellm_name),
    )
    _emit_event(
        conn, "model", litellm_name, "revive",
        cause="auto_expiry", actor="auto",
    )


def revive(litellm_name: str, actor: str = "auto") -> None:
    """Mark a model alive. Called by discovery refresh on /v1/models hit
    or by operator /revive command. Idempotent — no-op if already active.

    Honors CAUSE_POLICY[cause].manual_revive: when True (auth, manual,
    404_permanent), an `actor=auto` revive (typically discovery) is
    refused — only `actor=manual` (operator /revive) overrides. Lets
    the runtime call's 404 evidence outweigh discovery's weaker
    "appears in /v1/models" hint until the TTL elapses naturally.
    """
    if not litellm_name:
        return
    conn = _get_conn()
    row = conn.execute(
        "SELECT status, cause FROM models WHERE litellm_name = ?",
        (litellm_name,),
    ).fetchone()
    if row is None or row["status"] == "active":
        return
    cause = row["cause"]
    policy = CAUSE_POLICY.get(cause or "", {})
    # actor=="auto" is reserved for unattended sources (discovery refresh,
    # TTL expiry sweeps). manual_revive=True blocks those — only an
    # operator-initiated revive (any non-"auto" actor: telegram passes
    # "user", direct CLI passes "manual") overrides. Lets the runtime
    # call's 404_permanent evidence outweigh discovery's weaker /v1/models
    # listing until the 24h TTL elapses naturally.
    if policy.get("manual_revive", False) and actor == "auto":
        logger.debug(
            "registry: revive(%s) ignored — cause=%s requires non-auto actor "
            "(got actor=auto); operator /revive overrides",
            litellm_name, cause,
        )
        return
    conn.execute(
        "UPDATE models SET status='active', revived_at=?, cause=NULL, "
        "expires_at=NULL WHERE litellm_name=?",
        (_now_iso(), litellm_name),
    )
    _emit_event(conn, "model", litellm_name, "revive", actor=actor)
    logger.info("registry: revived %s (actor=%s)", litellm_name, actor)


def get_model_cause(litellm_name: str) -> str | None:
    """Diagnostic — return cause for currently-dead model, else None."""
    if not litellm_name:
        return None
    conn = _get_conn()
    row = conn.execute(
        "SELECT status, cause FROM models WHERE litellm_name = ?",
        (litellm_name,),
    ).fetchone()
    if row is None or row["status"] != "dead":
        return None
    return row["cause"]


def list_dead() -> list[dict]:
    """All currently-dead models (post auto-expiry filter). Used by
    /revive Telegram command listing and /diag introspection."""
    conn = _get_conn()
    now_iso = _now_iso()
    rows = conn.execute(
        "SELECT litellm_name, provider, cause, marked_at, expires_at "
        "FROM models WHERE status='dead' "
        "AND (expires_at IS NULL OR expires_at > ?) "
        "ORDER BY marked_at DESC",
        (now_iso,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Provider API ─────────────────────────────────────────────────────────
def register_provider(name: str, key_hash: str | None = None) -> None:
    """Idempotent register. Records key_hash so rotation detection can
    notice a changed key on next boot."""
    if not name:
        return
    conn = _get_conn()
    if key_hash is None:
        conn.execute(
            "INSERT OR IGNORE INTO providers (name, status) VALUES (?, 'active')",
            (name,),
        )
    else:
        conn.execute(
            "INSERT INTO providers (name, status, key_hash) VALUES (?, 'active', ?) "
            "ON CONFLICT(name) DO UPDATE SET key_hash=excluded.key_hash",
            (name, key_hash),
        )


def mark_provider_dead(
    provider: str,
    cause: str = "auth",
    actor: str = "auto",
    payload: dict | None = None,
) -> None:
    """Mark provider dead. Auth failures land here instead of mass-marking
    every model on the provider — single row replaces 30+. Selector
    eligibility checks both is_dead(model) and is_provider_dead(provider).

    Auth/manual causes have no TTL — operator must /revive after fixing
    the key. .env mtime watcher (separate concern) can also auto-trigger
    revive on key change.
    """
    if not provider:
        return
    if cause not in CAUSE_POLICY:
        logger.warning(
            "mark_provider_dead: unknown cause '%s', treating as 'auth'",
            cause,
        )
        cause = "auth"
    now_iso = _now_iso()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO providers (name, status, cause, marked_at) "
        "VALUES (?, 'dead', ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "status='dead', cause=excluded.cause, marked_at=excluded.marked_at",
        (provider, cause, now_iso),
    )
    _emit_event(
        conn, "provider", provider, "mark_dead",
        cause=cause, actor=actor, payload=payload,
    )
    logger.warning(
        "registry: marked PROVIDER dead %s — cause=%s, actor=%s",
        provider, cause, actor,
    )


def is_provider_dead(provider: str) -> bool:
    """True iff provider row status='dead'. Provider-level dead has no
    TTL by design — auth/manual causes need operator action."""
    if not provider:
        return False
    conn = _get_conn()
    row = conn.execute(
        "SELECT status FROM providers WHERE name = ?",
        (provider,),
    ).fetchone()
    return row is not None and row["status"] == "dead"


def list_dead_providers() -> list[dict]:
    """All currently-dead providers. Used by /dead Telegram command."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT name, cause, marked_at FROM providers WHERE status='dead' "
        "ORDER BY marked_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def revive_provider(provider: str, actor: str = "auto") -> None:
    """Mark provider alive. Idempotent."""
    if not provider:
        return
    conn = _get_conn()
    row = conn.execute(
        "SELECT status FROM providers WHERE name = ?", (provider,),
    ).fetchone()
    if row is None or row["status"] == "active":
        return
    conn.execute(
        "UPDATE providers SET status='active', revived_at=?, cause=NULL "
        "WHERE name=?",
        (_now_iso(), provider),
    )
    _emit_event(conn, "provider", provider, "revive", actor=actor)
    logger.info("registry: revived PROVIDER %s (actor=%s)", provider, actor)


def hash_key(key: str) -> str:
    """SHA256 first 8 hex of provider API key. Used to detect rotation
    across restarts (compare persisted vs current at boot → mismatch =
    auto-revive provider, force re-probe)."""
    import hashlib
    if not key:
        return ""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def get_provider_key_hash(provider: str) -> str | None:
    """Return persisted key_hash for provider, or None if absent."""
    if not provider:
        return None
    conn = _get_conn()
    row = conn.execute(
        "SELECT key_hash FROM providers WHERE name = ?", (provider,),
    ).fetchone()
    if row is None:
        return None
    return row["key_hash"]


# ── Diagnostic events query ──────────────────────────────────────────────
def recent_events(target: str | None = None, limit: int = 50) -> list[dict]:
    """Return registry_events ordered newest-first. Used by /diag
    Telegram command and tests."""
    conn = _get_conn()
    if target:
        rows = conn.execute(
            "SELECT timestamp, scope, target, event, cause, actor, payload_json "
            "FROM registry_events WHERE target = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (target, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT timestamp, scope, target, event, cause, actor, payload_json "
            "FROM registry_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
