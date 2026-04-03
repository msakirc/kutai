import pytest

from src.tools.free_apis import tokenize_api_description, TURKISH_CATEGORY_PATTERNS


def test_tokenize_extracts_meaningful_keywords():
    desc = "Weather forecasts in plain text or JSON. No API key needed."
    keywords = tokenize_api_description(desc)
    assert "weather" in keywords
    assert "forecasts" in keywords
    # Stop words excluded
    assert "in" not in keywords
    assert "or" not in keywords
    assert "no" not in keywords


def test_tokenize_handles_empty():
    assert tokenize_api_description("") == []
    assert tokenize_api_description(None) == []


def test_turkish_patterns_cover_key_categories():
    assert "weather" in TURKISH_CATEGORY_PATTERNS
    assert "currency" in TURKISH_CATEGORY_PATTERNS
    assert "pharmacy" in TURKISH_CATEGORY_PATTERNS
    assert "earthquake" in TURKISH_CATEGORY_PATTERNS


@pytest.mark.asyncio
async def test_build_keyword_index_populates_db(tmp_path):
    """build_keyword_index should create entries in api_keywords table."""
    import src.infra.db as db_mod
    original = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db = None
    try:
        await db_mod.init_db()
        from src.tools.free_apis import build_keyword_index, seed_registry
        await seed_registry()
        count = await build_keyword_index()
        assert count > 50

        results = await db_mod.find_apis_by_keywords(["weather", "forecast"])
        assert len(results) > 0
        assert results[0]["api_name"] in ("wttr.in", "Open-Meteo")
    finally:
        db_mod.DB_PATH = original
        db_mod._db = None
