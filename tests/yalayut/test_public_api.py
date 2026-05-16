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


# ── H2 regression ──────────────────────────────────────────────────────────────

async def test_on_demand_discovery_reverts_source_promotion(
    yalayut_db, monkeypatch
):
    """H2: on_demand_discovery() must restore discovery_mode and trusted to
    their original values after the temporary cron-style run finishes.
    Before the fix the source is permanently promoted to cron/trusted=1."""
    from yalayut.schema import ensure_yalayut_schema
    from yalayut.vetting.policy import seed_policy

    await seed_policy(yalayut_db)

    # Insert an on_demand / untrusted source with a github_path adapter
    source_id = "github:testowner/testrepo@/skills"
    await yalayut_db.execute(
        "INSERT OR IGNORE INTO yalayut_sources "
        "(source_id, source_type, endpoint, discovery_mode, trusted, "
        " trust_score, min_interval_s) "
        "VALUES (?, 'github_path', 'https://github.com/testowner/testrepo', "
        "        'on_demand', 0, 0.5, 86400)",
        (source_id,),
    )
    await yalayut_db.commit()

    # Stub out get_db and run_cron_discovery so no real network or cron run
    async def fake_get_db():
        return yalayut_db
    monkeypatch.setattr("src.infra.db.get_db", fake_get_db)

    async def fake_run_cron(db):
        # cron finds no rows because the source is enabled=1 but has no
        # enabled flag set — we just want to test the revert logic
        return {"sources_run": 0, "artifacts_indexed": 0,
                "skipped_disabled": 0, "errors": []}
    monkeypatch.setattr(
        "yalayut.discovery.cron.run_cron_discovery", fake_run_cron
    )

    await yalayut.on_demand_discovery({"source_id": source_id})

    # After on_demand_discovery returns, the source must be back to original
    cur = await yalayut_db.execute(
        "SELECT discovery_mode, trusted FROM yalayut_sources WHERE source_id=?",
        (source_id,),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row["discovery_mode"] == "on_demand", (
        f"source must revert to on_demand; got {row['discovery_mode']!r}"
    )
    assert row["trusted"] == 0, (
        f"source must revert to trusted=0; got {row['trusted']}"
    )
