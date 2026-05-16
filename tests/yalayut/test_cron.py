"""daily_discovery / cron tests — must ACTUALLY fetch + index."""
import pytest

from yalayut.contracts import ArtifactRef
from yalayut.discovery.cron import run_cron_discovery
from yalayut.discovery.sources import github_path
from yalayut.seed.seed_data import seed_owners, seed_sources
from yalayut.vetting.policy import seed_policy

pytestmark = pytest.mark.asyncio

_FIXTURE_BODY = (
    b"---\nname: brainstorming\ndescription: Use this before creative work, "
    b"design and requirements gathering.\nlicense: MIT\n---\n\nBody text here."
)


async def test_cron_fetches_and_indexes(yalayut_db, monkeypatch, tmp_path):
    await seed_policy(yalayut_db)
    await seed_owners(yalayut_db)
    await seed_sources(yalayut_db)

    # mock the adapter network calls
    async def fake_discover(self, cfg):
        return [ArtifactRef(
            source_id=cfg.source_id, name="brainstorming",
            fetch_url="https://raw/x", owner="obra",
        )]

    async def fake_fetch(self, ref):
        d = tmp_path / ref.name
        d.mkdir(exist_ok=True)
        f = d / "SKILL.md"
        f.write_bytes(_FIXTURE_BODY)
        return f

    async def fake_embed(text, is_query=False):
        return [1.0] + [0.0] * 767

    monkeypatch.setattr(github_path.GithubPathAdapter, "discover",
                        fake_discover)
    monkeypatch.setattr(github_path.GithubPathAdapter, "fetch", fake_fetch)
    monkeypatch.setattr("yalayut.discovery.cron._embed", fake_embed)

    result = await run_cron_discovery(yalayut_db)
    # 4 trusted cron sources each discover the one fixture artifact
    assert result["sources_run"] == 4
    assert result["artifacts_indexed"] >= 1

    cur = await yalayut_db.execute(
        "SELECT name, vet_tier, enabled FROM yalayut_index "
        "WHERE name_original='brainstorming'"
    )
    rows = await cur.fetchall()
    assert rows, "cron must populate the index"
    # obra owner trust 0.9 + trusted source -> T0
    assert rows[0]["vet_tier"] == 0
    assert rows[0]["enabled"] == 1


async def test_cron_honors_disabled_imports(yalayut_db, monkeypatch, tmp_path):
    await seed_policy(yalayut_db)
    await seed_owners(yalayut_db)
    await seed_sources(yalayut_db)
    await yalayut_db.execute(
        "INSERT INTO yalayut_disabled_imports (source, artifact_name, reason) "
        "VALUES ('github:obra/superpowers@/skills', 'brainstorming', 'test')"
    )
    await yalayut_db.commit()

    async def fake_discover(self, cfg):
        if "superpowers" not in cfg.source_id:
            return []
        return [ArtifactRef(source_id=cfg.source_id, name="brainstorming",
                            fetch_url="x", owner="obra")]

    async def fake_fetch(self, ref):
        f = tmp_path / "SKILL.md"
        f.write_bytes(_FIXTURE_BODY)
        return f

    async def fake_embed(text, is_query=False):
        return [1.0] + [0.0] * 767

    monkeypatch.setattr(github_path.GithubPathAdapter, "discover",
                        fake_discover)
    monkeypatch.setattr(github_path.GithubPathAdapter, "fetch", fake_fetch)
    monkeypatch.setattr("yalayut.discovery.cron._embed", fake_embed)

    result = await run_cron_discovery(yalayut_db)
    assert result["skipped_disabled"] >= 1
    cur = await yalayut_db.execute(
        "SELECT COUNT(*) c FROM yalayut_index WHERE name_original='brainstorming'"
    )
    assert (await cur.fetchone())["c"] == 0
