"""daily_discovery / cron tests — must ACTUALLY fetch + index."""
import pytest

from yalayut.contracts import ArtifactRef, Manifest
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


# ── H3 regression ──────────────────────────────────────────────────────────────

async def test_cron_writes_manifest_yaml(yalayut_db, monkeypatch, tmp_path):
    """H3: after fetch+synthesize, a manifest.yaml must exist next to SKILL.md
    so that _to_artifact can load intent_keywords and inputs_schema from it."""
    await seed_policy(yalayut_db)
    await seed_owners(yalayut_db)
    await seed_sources(yalayut_db)

    skill_dir = tmp_path / "brainstorming"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_bytes(_FIXTURE_BODY)

    async def fake_discover(self, cfg):
        if "superpowers" not in cfg.source_id:
            return []
        return [ArtifactRef(
            source_id=cfg.source_id, name="brainstorming",
            fetch_url="https://raw/x", owner="obra",
        )]

    async def fake_fetch(self, ref):
        # return the pre-created SKILL.md (no network)
        return skill_file

    async def fake_embed(text, is_query=False):
        return [1.0] + [0.0] * 767

    monkeypatch.setattr(github_path.GithubPathAdapter, "discover", fake_discover)
    monkeypatch.setattr(github_path.GithubPathAdapter, "fetch", fake_fetch)
    monkeypatch.setattr("yalayut.discovery.cron._embed", fake_embed)

    result = await run_cron_discovery(yalayut_db)
    assert result["artifacts_indexed"] >= 1

    manifest_yaml = skill_dir / "manifest.yaml"
    assert manifest_yaml.exists(), (
        "cron must write manifest.yaml next to SKILL.md so _to_artifact "
        "can load intent_keywords/inputs_schema"
    )

    # Round-trip: the written YAML must be parseable and carry the synthesized
    # intent_keywords (non-empty, since brainstorming + description yield tokens)
    import yaml
    data = yaml.safe_load(manifest_yaml.read_text(encoding="utf-8"))
    assert "name" in data, "manifest.yaml must contain the artifact name"
    assert "intent_keywords" in data, "manifest.yaml must carry intent_keywords"


async def test_cron_skips_invalid_manifest_and_records_error(
    yalayut_db, monkeypatch, tmp_path
):
    """validate_manifest errors must be recorded in errors[] and artifact skipped."""
    await seed_policy(yalayut_db)
    await seed_owners(yalayut_db)
    await seed_sources(yalayut_db)

    skill_dir = tmp_path / "badskill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_bytes(_FIXTURE_BODY)

    # Return a Manifest that fails validate_manifest (missing name + version)
    bad_manifest = Manifest(
        name="", name_original="badskill", version="",
        artifact_type="bogus_type",
    )

    async def fake_discover(self, cfg):
        if "superpowers" not in cfg.source_id:
            return []
        return [ArtifactRef(
            source_id=cfg.source_id, name="badskill",
            fetch_url="https://raw/x", owner="obra",
        )]

    async def fake_fetch(self, ref):
        return skill_file

    async def fake_embed(text, is_query=False):
        return [1.0] + [0.0] * 767

    import yalayut.discovery.cron as _cron_mod
    monkeypatch.setattr(github_path.GithubPathAdapter, "discover", fake_discover)
    monkeypatch.setattr(github_path.GithubPathAdapter, "fetch", fake_fetch)
    monkeypatch.setattr("yalayut.discovery.cron._embed", fake_embed)
    # Patch synthesize so it returns the bad manifest
    monkeypatch.setattr(
        _cron_mod, "synthesize", lambda ref, raw: (bad_manifest, "body text")
    )

    result = await run_cron_discovery(yalayut_db)

    # Artifact must be skipped — not indexed
    assert result["artifacts_indexed"] == 0, (
        "invalid manifest must not be indexed"
    )
    # Error must be recorded in the errors list
    assert any("validate" in e and "badskill" in e for e in result["errors"]), (
        f"validate error must be in errors: {result['errors']}"
    )
    # DB must be empty for this artifact
    cur = await yalayut_db.execute(
        "SELECT COUNT(*) c FROM yalayut_index WHERE name_original='badskill'"
    )
    assert (await cur.fetchone())["c"] == 0


# ── Fix 1 regression: _ingest_source must run a REAL adapter pipeline ───────


async def test_ingest_source_invokes_real_adapter_and_stores(
    yalayut_db, monkeypatch, tmp_path
):
    """`_ingest_source(source_row)` must actually run the adapter ->
    fetch -> synthesize -> vet -> classify -> store pipeline (not call a
    non-existent fetch.ingest_all). Proven by registering a fake adapter in
    `_ADAPTERS` and asserting a row lands in yalayut_index with correct
    enabled/quarantined counts."""
    import yalayut.discovery.cron as _cron_mod
    from yalayut.contracts import ArtifactRef
    from yalayut.discovery.sources import github_path

    await seed_policy(yalayut_db)
    await seed_owners(yalayut_db)
    await seed_sources(yalayut_db)

    reached = {"fetch": False}

    class _FakeAdapter:
        source_type = "fake_src"

        async def discover(self, cfg):
            return [ArtifactRef(
                source_id=cfg.source_id, name="brainstorming",
                fetch_url="https://raw/x", owner="obra",
            )]

        async def fetch(self, ref):
            reached["fetch"] = True
            d = tmp_path / ref.name
            d.mkdir(exist_ok=True)
            f = d / "SKILL.md"
            f.write_bytes(_FIXTURE_BODY)
            return f

    async def fake_embed(text, is_query=False):
        return [1.0] + [0.0] * 767

    # Register the fake adapter as a real entry in the registry.
    monkeypatch.setitem(_cron_mod._ADAPTERS, "fake_src", _FakeAdapter())
    monkeypatch.setattr("yalayut.discovery.cron._embed", fake_embed)

    # _ingest_source uses get_db() — route it to the in-memory test DB.
    async def _fake_get_db():
        return yalayut_db

    monkeypatch.setattr(_cron_mod, "get_db", _fake_get_db)

    source_row = {
        "source_id": "github:obra/superpowers@/skills",
        "source_type": "fake_src",
        "endpoint": "https://example.com",
        "auth_env": None,
        "trusted": True,
        "discovery_mode": "cron",
        "min_interval_s": 0,
    }
    res = await _cron_mod._ingest_source(source_row)

    assert reached["fetch"], "_ingest_source must reach the adapter fetch path"
    assert res["ingested"] == 1, f"expected 1 ingested, got {res}"
    # obra owner trust 0.9 + brainstorming -> T0 -> enabled
    assert res["enabled"] == 1
    assert res["quarantined"] == 0

    cur = await yalayut_db.execute(
        "SELECT name, enabled FROM yalayut_index "
        "WHERE name_original='brainstorming'"
    )
    rows = await cur.fetchall()
    assert rows, "_ingest_source must actually populate yalayut_index"
    assert rows[0]["enabled"] == 1


async def test_ingest_source_unknown_type_returns_error_never_raises(
    yalayut_db, monkeypatch
):
    """A source_type with no adapter must yield a zero-count dict with an
    error string — never raise."""
    import yalayut.discovery.cron as _cron_mod

    async def _fake_get_db():
        return yalayut_db

    monkeypatch.setattr(_cron_mod, "get_db", _fake_get_db)

    res = await _cron_mod._ingest_source({
        "source_id": "x", "source_type": "no_such_adapter",
        "endpoint": "", "auth_env": None,
    })
    assert res["ingested"] == 0
    assert res["enabled"] == 0
    assert res["quarantined"] == 0
    assert any("no adapter" in e for e in res["errors"])


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
