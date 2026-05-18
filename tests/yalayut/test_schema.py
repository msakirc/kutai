"""Schema creation tests."""
import pytest

pytestmark = pytest.mark.asyncio

EXPECTED_TABLES = {
    "yalayut_index", "yalayut_usage", "yalayut_sources", "yalayut_owners",
    "yalayut_disabled_imports", "yalayut_bind_cache", "yalayut_mcp_processes",
    "yalayut_mcp_tools", "yalayut_secrets", "yalayut_policy",
    "yalayut_policy_proposals", "yalayut_source_candidates",
    "yalayut_demand_signals",
}


async def test_all_tables_created(yalayut_db):
    cur = await yalayut_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    rows = await cur.fetchall()
    names = {r["name"] for r in rows}
    assert EXPECTED_TABLES.issubset(names), EXPECTED_TABLES - names


async def test_index_has_env_status_column(yalayut_db):
    cur = await yalayut_db.execute("PRAGMA table_info(yalayut_index)")
    cols = {r["name"] for r in await cur.fetchall()}
    assert "env_status" in cols
    assert "name_original" in cols
    assert "embedding" in cols


async def test_mcp_processes_has_health_columns(yalayut_db):
    cur = await yalayut_db.execute("PRAGMA table_info(yalayut_mcp_processes)")
    cols = {r["name"] for r in await cur.fetchall()}
    assert {"health", "last_probe_at", "consecutive_probe_fails"} <= cols


async def test_idempotent(yalayut_db):
    # second call must not raise
    from yalayut.schema import ensure_yalayut_schema
    await ensure_yalayut_schema(yalayut_db)
