"""Yalayut DB schema — all 13 tables + MCP extras.

Idempotent. Folds the spec's ALTER-COLUMN additions (env_status, MCP health
columns) directly into the CREATE TABLE so a fresh DB needs no migration step.
"""
import aiosqlite

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS yalayut_index (
      id INTEGER PRIMARY KEY,
      artifact_type TEXT NOT NULL,
      kind TEXT,
      source TEXT NOT NULL,
      owner TEXT,
      name TEXT NOT NULL,
      name_original TEXT,
      version TEXT NOT NULL,
      manifest_path TEXT,
      body_excerpt TEXT,
      embedding BLOB,
      vet_tier INTEGER,
      exposure_class TEXT,
      applies_to TEXT,
      vet_state TEXT,
      vet_hash TEXT,
      source_max INTEGER,
      check_max_json TEXT,
      signature TEXT,
      mechanizable BOOLEAN,
      model_hint TEXT,
      env_status TEXT DEFAULT 'ready',
      enabled BOOLEAN DEFAULT 1,
      created_at TIMESTAMP,
      vetted_at TIMESTAMP,
      UNIQUE(source, name, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_usage (
      id INTEGER PRIMARY KEY,
      artifact_id INTEGER REFERENCES yalayut_index(id),
      task_id TEXT,
      exposure_class TEXT,
      bind_args_json TEXT,
      exposed BOOLEAN,
      called BOOLEAN,
      succeeded BOOLEAN,
      latency_ms INTEGER,
      conflict_loser BOOLEAN,
      would_have_used INTEGER,
      escape_reason TEXT,
      occurred_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_sources (
      id INTEGER PRIMARY KEY,
      source_id TEXT UNIQUE NOT NULL,
      source_type TEXT,
      endpoint TEXT,
      auth_env TEXT,
      trust_score REAL DEFAULT 0.3,
      pin_policy TEXT DEFAULT 'minor',
      discovery_mode TEXT DEFAULT 'on_demand',
      trusted BOOLEAN,
      enabled BOOLEAN DEFAULT 1,
      last_run_at TIMESTAMP,
      min_interval_s INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_owners (
      owner_id TEXT PRIMARY KEY,
      trust_score REAL DEFAULT 0.3,
      allowed_artifact_types TEXT,
      source_count INTEGER,
      rolling_success_rate REAL,
      notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_disabled_imports (
      id INTEGER PRIMARY KEY,
      source TEXT NOT NULL,
      artifact_name TEXT NOT NULL,
      reason TEXT,
      added_at TIMESTAMP,
      UNIQUE(source, artifact_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_bind_cache (
      id INTEGER PRIMARY KEY,
      manifest_id INTEGER REFERENCES yalayut_index(id),
      ctx_embedding BLOB,
      bound_args_json TEXT,
      hit_count INTEGER DEFAULT 0,
      created_at TIMESTAMP,
      last_used_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_mcp_processes (
      artifact_id INTEGER PRIMARY KEY REFERENCES yalayut_index(id),
      pid INTEGER,
      port INTEGER,
      started_at TIMESTAMP,
      last_used_at TIMESTAMP,
      idle_timeout_s INTEGER DEFAULT 300,
      health TEXT DEFAULT 'starting',
      last_probe_at TIMESTAMP,
      consecutive_probe_fails INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_mcp_tools (
      id INTEGER PRIMARY KEY,
      artifact_id INTEGER REFERENCES yalayut_index(id),
      tool_name TEXT NOT NULL,
      description TEXT,
      description_embedding BLOB,
      input_schema_json TEXT,
      first_seen_at TIMESTAMP,
      UNIQUE(artifact_id, tool_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_secrets (
      id INTEGER PRIMARY KEY,
      key_name TEXT UNIQUE NOT NULL,
      encrypted_value BLOB,
      added_at TIMESTAMP,
      last_used_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_policy (
      id INTEGER PRIMARY KEY,
      check_name TEXT NOT NULL,
      key TEXT NOT NULL,
      value TEXT,
      added_by TEXT,
      added_at TIMESTAMP,
      UNIQUE(check_name, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_policy_proposals (
      id INTEGER PRIMARY KEY,
      check_name TEXT NOT NULL,
      key TEXT NOT NULL,
      proposed_value TEXT,
      evidence_json TEXT,
      state TEXT DEFAULT 'pending',
      proposed_at TIMESTAMP,
      decided_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_source_candidates (
      id INTEGER PRIMARY KEY,
      candidate_source_id TEXT,
      source_type TEXT,
      endpoint TEXT,
      metadata_json TEXT,
      state TEXT DEFAULT 'pending',
      proposed_at TIMESTAMP,
      decided_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yalayut_demand_signals (
      id INTEGER PRIMARY KEY,
      source_step_pattern TEXT,
      intent_keywords_json TEXT,
      signal_type TEXT,
      confidence REAL,
      fired_at TIMESTAMP,
      resulted_in_discovery BOOLEAN
    )
    """,
]


async def ensure_yalayut_schema(db: aiosqlite.Connection) -> None:
    """Create every yalayut table if absent. Idempotent — safe on every boot."""
    for ddl in _DDL:
        await db.execute(ddl)
    await db.commit()
