"""End-to-end: discovery cron -> query, mocked github_path source."""
import pytest

from yalayut.contracts import ArtifactRef
from yalayut.discovery.cron import run_cron_discovery
from yalayut.discovery.sources import github_path
from yalayut.migration import run_full_migration
from yalayut._query_engine import query_db
from yalayut.contracts import TaskContext

pytestmark = pytest.mark.asyncio

_PDF = (
    b"---\nname: pdf\ndescription: Extract text merge split pdf files and "
    b"fill forms.\nlicense: proprietary\n---\n\nPDF body."
)
_BRAINSTORM = (
    b"---\nname: brainstorming\ndescription: Design requirements gathering "
    b"before creative work.\nlicense: MIT\n---\n\nBrainstorm body."
)


async def test_full_pipeline(yalayut_db, monkeypatch, tmp_path):
    # full migration installs schema + seeds
    async def fake_embed(text, is_query=False):
        # cheap deterministic embedding: bag-of-chars buckets
        vec = [0.0] * 768
        for ch in text.lower():
            vec[ord(ch) % 768] += 1.0
        return vec

    monkeypatch.setattr("yalayut.migration._embed", fake_embed)
    monkeypatch.setattr("yalayut.discovery.cron._embed", fake_embed)

    mig = await run_full_migration(yalayut_db)
    assert mig["sources_seeded"] == 4

    bodies = {"pdf": _PDF, "brainstorming": _BRAINSTORM}

    async def fake_discover(self, cfg):
        if "anthropics" in cfg.source_id:
            return [ArtifactRef(cfg.source_id, "pdf", "x", "anthropics")]
        if "superpowers" in cfg.source_id:
            return [ArtifactRef(cfg.source_id, "brainstorming", "x", "obra")]
        return []

    async def fake_fetch(self, ref):
        d = tmp_path / ref.name
        d.mkdir(exist_ok=True)
        f = d / "SKILL.md"
        f.write_bytes(bodies[ref.name])
        return f

    monkeypatch.setattr(github_path.GithubPathAdapter, "discover",
                        fake_discover)
    monkeypatch.setattr(github_path.GithubPathAdapter, "fetch", fake_fetch)

    result = await run_cron_discovery(yalayut_db)
    assert result["artifacts_indexed"] == 2

    # both artifacts indexed at T0 (trusted source + trusted owner)
    cur = await yalayut_db.execute(
        "SELECT name, vet_tier FROM yalayut_index WHERE source != 'internal'"
    )
    rows = await cur.fetchall()
    assert {r["name"] for r in rows} == {"anthropics-pdf",
                                         "obra-brainstorming"}
    assert all(r["vet_tier"] == 0 for r in rows)

    # query for a pdf task ranks the pdf skill first
    q_emb = await fake_embed("extract text from pdf and fill forms")
    results = await query_db(
        yalayut_db, TaskContext(title="pdf form extraction"),
        query_embedding=q_emb,
    )
    assert results[0].name == "anthropics-pdf"


async def test_migration_skills_then_query(yalayut_db, monkeypatch):
    await yalayut_db.execute("""
        CREATE TABLE skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL, skill_type TEXT, strategies TEXT,
            injection_count INTEGER, injection_success INTEGER,
            created_at TEXT, updated_at TEXT
        )
    """)
    await yalayut_db.execute(
        "INSERT INTO skills (name, description, strategies) "
        "VALUES ('route-shop', 'route shopping to advisor agent', '[]')"
    )
    await yalayut_db.commit()

    async def fake_embed(text, is_query=False):
        vec = [0.0] * 768
        for ch in text.lower():
            vec[ord(ch) % 768] += 1.0
        return vec

    monkeypatch.setattr("yalayut.migration._embed", fake_embed)
    await run_full_migration(yalayut_db)

    q_emb = await fake_embed("route shopping to advisor agent")
    results = await query_db(
        yalayut_db, TaskContext(title="shopping route"), query_embedding=q_emb,
    )
    # migrated internal_hint is queryable, exposure_class inject
    hit = next(r for r in results if r.name == "route-shop")
    assert hit.kind == "internal_hint"
    assert hit.exposure_class == "inject"
