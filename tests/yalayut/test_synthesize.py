"""Manifest synthesis (parser path) tests."""
from pathlib import Path
from unittest.mock import patch

import pytest

from yalayut.contracts import ArtifactRef
from yalayut.discovery.synthesize import synthesize

FIXTURE = Path(__file__).parent / "fixtures" / "sample_skill.md"


def test_synthesize_from_frontmatter():
    ref = ArtifactRef(
        source_id="github:anthropics/skills@/skills", name="pdf",
        fetch_url="x", owner="anthropics",
    )
    m, body = synthesize(ref, FIXTURE.read_bytes())
    assert m.name == "anthropics-pdf"        # canonical
    assert m.name_original == "pdf"
    assert m.artifact_type == "skill"
    assert m.kind == "prompt_skill"
    assert m.owner == "anthropics"
    assert m.license == "Proprietary"
    # intent keywords extracted mechanically from the description
    assert "pdf" in [k.lower() for k in m.intent_keywords]
    assert "PDF files" in body


def test_synthesize_marks_shell_recipe_mechanizable_false_by_default():
    ref = ArtifactRef(
        source_id="github:anthropics/skills@/skills", name="pdf",
        fetch_url="x", owner="anthropics",
    )
    m, _ = synthesize(ref, FIXTURE.read_bytes())
    # parser path never asserts mechanizable=true; only seed manifests do
    assert m.mechanizable is False


@pytest.mark.asyncio
async def test_llm_synthesize_uses_husam_and_parses_manifest():
    from yalayut.discovery import synthesize as syn

    async def fake_run(spec):
        return {"content": '{"intent_keywords": ["pdf"], "mechanizable": true, "kind": "shell_recipe", "install_cmd": null, "auth_env": null}'}

    with patch("husam.run", fake_run):
        out = await syn.llm_synthesize(
            raw_text="Extract text from PDF files.",
            source_meta={"name_original": "pdf"},
        )
    assert out["kind"] == "shell_recipe"
    assert "pdf" in out["intent_keywords"]
    assert out["mechanizable"] is True


@pytest.mark.asyncio
async def test_llm_synthesize_empty_manifest_when_husam_raises():
    from yalayut.discovery import synthesize as syn

    async def fake_run(spec):
        raise RuntimeError("no model")

    with patch("husam.run", fake_run):
        out = await syn.llm_synthesize(raw_text="y", source_meta={"name_original": "x"})
    assert out["intent_keywords"] == []
    assert out["kind"] == "prompt_skill"  # default


def test_synthesize_no_await_inline_in_module():
    import pathlib
    _root = pathlib.Path(__file__).resolve().parents[2]
    src = (_root / "packages" / "yalayut" / "src" / "yalayut"
           / "discovery" / "synthesize.py").read_text(encoding="utf-8")
    offenders = [ln for ln in src.splitlines()
                 if "await_inline=True" in ln and not ln.lstrip().startswith("#")]
    assert not offenders, f"synthesize.py still uses await_inline: {offenders}"
