"""Public API surface tests — every function has a real body."""
import pytest

import yalayut

pytestmark = pytest.mark.asyncio


def test_public_api_surface():
    for fn in ("query", "daily_discovery", "source_scout_scan",
               "on_demand_discovery", "capture_hint", "run_recipe"):
        assert hasattr(yalayut, fn), fn
        assert callable(getattr(yalayut, fn))


async def test_source_scout_scan_proposes_unknown_sources(
    yalayut_db, monkeypatch
):
    # an indexed artifact from a source NOT in yalayut_sources
    await yalayut_db.execute(
        "INSERT INTO yalayut_index "
        "(artifact_type, kind, source, name, version, vet_tier, enabled) "
        "VALUES ('skill','prompt_skill','github:new/repo@/x','a','1',0,1)"
    )
    await yalayut_db.commit()

    async def fake_get_db():
        return yalayut_db
    monkeypatch.setattr("src.infra.db.get_db", fake_get_db)

    result = await yalayut.source_scout_scan()
    assert result["candidates_proposed"] == 1
    cur = await yalayut_db.execute(
        "SELECT candidate_source_id FROM yalayut_source_candidates"
    )
    assert (await cur.fetchone())["candidate_source_id"] == "github:new/repo@/x"


async def test_run_recipe_unknown_id(yalayut_db, monkeypatch):
    async def fake_get_db():
        return yalayut_db
    monkeypatch.setattr("src.infra.db.get_db", fake_get_db)
    result = await yalayut.run_recipe("9999", {})
    assert result["ok"] is False
