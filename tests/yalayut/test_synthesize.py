"""Manifest synthesis (parser path) tests."""
from pathlib import Path

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
