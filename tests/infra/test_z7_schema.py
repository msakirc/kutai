"""Z7 T1.0 — foundation schema migration tests.

Covers:
  - mission_briefings table and index exist on fresh DB
  - external_comms_log table and indexes exist on fresh DB
  - mission_events.founder_minutes_saved column added
  - founder_actions priority / defer_until / expires_at columns added
  - founder_attention_log card_id / surfaced_at / acted_at / deferred_to /
    attention_minutes columns added
  - Re-running init_db is idempotent (no crash on second call)
  - posthook registry contains all 4 new Z7 kinds
"""
from __future__ import annotations

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

async def _setup_fresh(tmp_path, monkeypatch):
    """Open a fresh temp DB, run init_db, return (db_path, db_mod)."""
    db_path = tmp_path / "z7_schema.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_path, db_mod


async def _columns_of(db, table: str) -> set[str]:
    cur = await db.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in await cur.fetchall()}


async def _indexes_of(db, table: str) -> set[str]:
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table,),
    )
    return {row[0] for row in await cur.fetchall()}


# ── mission_briefings ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mission_briefings_table_exists(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='mission_briefings'"
    )
    assert await cur.fetchone() is not None, "mission_briefings table missing"


@pytest.mark.asyncio
async def test_mission_briefings_columns(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cols = await _columns_of(db, "mission_briefings")
    expected = {
        "id", "product_id", "mission_id", "kind",
        "body_md", "founder_minutes_saved_estimate",
        "prepared_at", "read_at", "acted_on",
    }
    missing = expected - cols
    assert not missing, f"mission_briefings missing columns: {missing}"


@pytest.mark.asyncio
async def test_mission_briefings_index_exists(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    idxs = await _indexes_of(db, "mission_briefings")
    assert "idx_mission_briefings_product_prepared" in idxs


# ── external_comms_log ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_external_comms_log_table_exists(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='external_comms_log'"
    )
    assert await cur.fetchone() is not None, "external_comms_log table missing"


@pytest.mark.asyncio
async def test_external_comms_log_columns(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cols = await _columns_of(db, "external_comms_log")
    expected = {
        "log_id", "product_id", "sent_at", "channel",
        "recipient", "recipient_count", "content_hash",
        "content_md", "source_mission_id", "source_action_id",
        "vendor_call_id", "reversibility", "revoked_at", "revoke_reason",
    }
    missing = expected - cols
    assert not missing, f"external_comms_log missing columns: {missing}"


@pytest.mark.asyncio
async def test_external_comms_log_indexes_exist(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    idxs = await _indexes_of(db, "external_comms_log")
    assert "idx_external_comms_log_product_sent" in idxs
    assert "idx_external_comms_log_content_hash" in idxs


# ── mission_events.founder_minutes_saved ──────────────────────────────────────

@pytest.mark.asyncio
async def test_mission_events_founder_minutes_saved(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cols = await _columns_of(db, "mission_events")
    assert "founder_minutes_saved" in cols, (
        "mission_events.founder_minutes_saved column missing"
    )


# ── founder_actions new columns ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_founder_actions_priority_defer_expires(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cols = await _columns_of(db, "founder_actions")
    for col in ("priority", "defer_until", "expires_at"):
        assert col in cols, f"founder_actions.{col} column missing"


# ── founder_attention_log new columns ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_founder_attention_log_new_columns(tmp_path, monkeypatch):
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cols = await _columns_of(db, "founder_attention_log")
    for col in ("card_id", "surfaced_at", "acted_at", "deferred_to", "attention_minutes"):
        assert col in cols, f"founder_attention_log.{col} column missing"


# ── idempotency (re-run init_db) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_init_db_idempotent(tmp_path, monkeypatch):
    """Calling init_db twice must not raise."""
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    # Second call — must be idempotent
    await db_mod.init_db()


# ── posthook registry ─────────────────────────────────────────────────────────

def test_z7_posthook_kinds_registered():
    """All 4 Z7 kinds must be present in POST_HOOK_REGISTRY and POST_HOOK_KINDS."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY, POST_HOOK_KINDS

    z7_kinds = {
        "briefing_compose",
        "brand_voice_lint",
        "copy_compliance_review",
        "audit_completeness_check",
    }
    for kind in z7_kinds:
        assert kind in POST_HOOK_REGISTRY, f"{kind} missing from POST_HOOK_REGISTRY"
        assert kind in POST_HOOK_KINDS, f"{kind} missing from POST_HOOK_KINDS"


def test_z7_posthook_stubs_importable():
    """All 4 stub handler modules must be importable."""
    from general_beckman.posthook_handlers import briefing_compose
    from general_beckman.posthook_handlers import brand_voice_lint
    from general_beckman.posthook_handlers import copy_compliance_review
    from general_beckman.posthook_handlers import audit_completeness_check

    for mod in (
        briefing_compose,
        brand_voice_lint,
        copy_compliance_review,
        audit_completeness_check,
    ):
        assert callable(getattr(mod, "handle", None)), (
            f"{mod.__name__} has no callable 'handle'"
        )


@pytest.mark.asyncio
async def test_z7_posthook_handlers_degrade_gracefully():
    """Real handlers, called with minimal input and no DB/metadata, must not
    crash and must return a valid status (graceful degradation)."""
    from general_beckman.posthook_handlers import briefing_compose
    from general_beckman.posthook_handlers import brand_voice_lint
    from general_beckman.posthook_handlers import copy_compliance_review
    from general_beckman.posthook_handlers import audit_completeness_check

    dummy_task = {"id": 1, "mission_id": 42}
    dummy_result = {}
    valid = {"ok", "skip", "warning", "failed", "fail"}
    for mod in (
        briefing_compose,
        brand_voice_lint,
        copy_compliance_review,
        audit_completeness_check,
    ):
        result = await mod.handle(dummy_task, dummy_result)
        assert isinstance(result, dict), f"{mod.__name__}.handle must return a dict"
        assert result.get("status") in valid, (
            f"{mod.__name__}.handle returned unexpected status: {result}"
        )


# ── schema_migrations ledger ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_z7_migrations_recorded_in_ledger(tmp_path, monkeypatch):
    """All 5 Z7 migration versions must appear in schema_migrations after init."""
    _, db_mod = await _setup_fresh(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT version FROM schema_migrations WHERE version LIKE '2026-05-15-z7-%'"
    )
    versions = {row[0] for row in await cur.fetchall()}
    expected = {
        "2026-05-15-z7-mission-briefings",
        "2026-05-15-z7-external-comms-log",
        "2026-05-15-z7-mission-events-minutes-saved",
        "2026-05-15-z7-founder-actions-priority-defer",
        "2026-05-15-z7-founder-attention-log-card",
    }
    missing = expected - versions
    assert not missing, f"Z7 migrations not recorded in ledger: {missing}"
