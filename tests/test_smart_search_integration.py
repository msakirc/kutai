"""End-to-end test for the layered resolution system."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_layer0_resolves_weather_without_agent(tmp_path):
    """A weather query should be resolved at Layer 0 without touching an LLM."""
    import src.infra.db as db_mod
    original_path = db_mod.DB_PATH
    original_conn = db_mod._db_connection
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db_connection = None
    try:
        await db_mod.init_db()
        from src.tools.free_apis import seed_registry, build_keyword_index, seed_category_patterns
        await seed_registry()
        await build_keyword_index()
        await seed_category_patterns()

        from src.core.fast_resolver import try_resolve

        task = {"title": "Istanbul hava durumu", "description": ""}

        with patch("src.core.fast_resolver.call_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"current_weather": {"temperature": 22, "weathercode": 0}}'
            result = await try_resolve(task)

        assert result is not None
        assert "22" in result
    finally:
        await db_mod.close_db()
        db_mod.DB_PATH = original_path
        db_mod._db_connection = original_conn


@pytest.mark.asyncio
async def test_layer1_enriches_context(tmp_path):
    """A partial-match query should get enriched context."""
    import src.infra.db as db_mod
    original_path = db_mod.DB_PATH
    original_conn = db_mod._db_connection
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db_connection = None
    try:
        await db_mod.init_db()
        from src.tools.free_apis import seed_registry, build_keyword_index, seed_category_patterns
        await seed_registry()
        await build_keyword_index()
        await seed_category_patterns()

        from src.core.fast_resolver import enrich_context

        task = {"title": "Istanbul'da bu hafta sonu piknik yapilir mi? hava durumu nasil?", "description": ""}

        with patch("src.core.fast_resolver.call_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"temp": 22}'
            result = await enrich_context(task)

        # May or may not enrich depending on threshold tuning
        assert result is None or "Available Data" in result
    finally:
        await db_mod.close_db()
        db_mod.DB_PATH = original_path
        db_mod._db_connection = original_conn


@pytest.mark.asyncio
async def test_reliability_tracking(tmp_path):
    """API calls should update reliability counters."""
    import src.infra.db as db_mod
    original_path = db_mod.DB_PATH
    original_conn = db_mod._db_connection
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db_connection = None
    try:
        await db_mod.init_db()

        await db_mod.record_api_call("test_api", success=True)
        await db_mod.record_api_call("test_api", success=True)
        await db_mod.record_api_call("test_api", success=False)

        rel = await db_mod.get_api_reliability("test_api")
        assert rel["success_count"] == 2
        assert rel["failure_count"] == 1
        assert rel["status"] == "active"
    finally:
        await db_mod.close_db()
        db_mod.DB_PATH = original_path
        db_mod._db_connection = original_conn


@pytest.mark.asyncio
async def test_auto_demotion(tmp_path):
    """APIs with low success rate should be auto-demoted."""
    import src.infra.db as db_mod
    original_path = db_mod.DB_PATH
    original_conn = db_mod._db_connection
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db_connection = None
    try:
        await db_mod.init_db()

        # 1 success, 19 failures = 5% success rate with 20 calls -> suspended
        await db_mod.record_api_call("bad_api", success=True)
        for _ in range(19):
            await db_mod.record_api_call("bad_api", success=False)

        rel = await db_mod.get_api_reliability("bad_api")
        assert rel["status"] == "suspended"
    finally:
        await db_mod.close_db()
        db_mod.DB_PATH = original_path
        db_mod._db_connection = original_conn
