"""github_path adapter tests — frontmatter parse + discover (mocked HTTP)."""
from pathlib import Path

import pytest

from yalayut.contracts import ArtifactRef, SourceConfig
from yalayut.discovery.sources.github_path import (
    GithubPathAdapter, parse_skill_md,
)
from yalayut.discovery.fetch import stage_dir, promote

pytestmark = pytest.mark.asyncio

FIXTURE = Path(__file__).parent / "fixtures" / "sample_skill.md"


def test_parse_skill_md_frontmatter():
    raw = FIXTURE.read_bytes()
    meta, body = parse_skill_md(raw)
    assert meta["name"] == "pdf"
    assert "PDF files" in meta["description"]
    assert meta["license"] == "Proprietary"
    assert "pypdf" in body


def test_adapter_satisfies_protocol():
    from yalayut.contracts import SourceAdapter
    assert isinstance(GithubPathAdapter(), SourceAdapter)


async def test_discover_lists_skill_dirs(monkeypatch):
    adapter = GithubPathAdapter()

    async def fake_list_tree(self, owner, repo, path):
        return ["skills/pdf/SKILL.md", "skills/docx/SKILL.md",
                "skills/pdf/scripts/helper.py"]

    monkeypatch.setattr(GithubPathAdapter, "_list_tree", fake_list_tree)
    cfg = SourceConfig(
        source_id="github:anthropics/skills@/skills",
        source_type="github_path",
        endpoint="https://github.com/anthropics/skills",
        trusted=True,
    )
    refs = await adapter.discover(cfg)
    names = {r.name for r in refs}
    assert names == {"pdf", "docx"}
    assert all(r.owner == "anthropics" for r in refs)


async def test_fetch_writes_to_staging(monkeypatch, tmp_path):
    adapter = GithubPathAdapter()

    async def fake_get(self, url):
        return FIXTURE.read_bytes()

    monkeypatch.setattr(GithubPathAdapter, "_http_get", fake_get)
    monkeypatch.setattr(
        "yalayut.discovery.fetch._VENDOR_ROOT", tmp_path / "vendor"
    )
    ref = ArtifactRef(
        source_id="github:anthropics/skills@/skills", name="pdf",
        fetch_url="https://raw/anthropics/skills/skills/pdf/SKILL.md",
        owner="anthropics",
    )
    path = await adapter.fetch(ref)
    assert path.exists()
    assert b"PDF files" in path.read_bytes()


def test_promote_moves_staging_to_versioned(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yalayut.discovery.fetch._VENDOR_ROOT", tmp_path / "vendor"
    )
    staging = stage_dir("github:anthropics/skills", "pdf")
    (staging / "SKILL.md").write_text("body")
    final = promote(staging, "github:anthropics/skills", "pdf", "1.0.0")
    assert final.exists()
    assert (final / "SKILL.md").read_text() == "body"
    assert "v1.0.0" in str(final)
