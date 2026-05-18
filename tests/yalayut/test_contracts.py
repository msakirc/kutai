"""Contract dataclass + protocol shape tests."""
from pathlib import Path

from yalayut.contracts import (
    Manifest, Issue, ArtifactRef, SourceConfig, IndexRow, TaskContext,
    SkillApplication, Artifact, Result,
    DiscoveryPlugin, AccessPlugin, SourceAdapter,
)


def test_manifest_minimal_construction():
    m = Manifest(
        name="anthropics-pdf", name_original="pdf", version="1.0.0",
        artifact_type="skill", kind="prompt_skill",
        source="github:anthropics/skills@/skills", owner="anthropics",
    )
    assert m.mechanizable is False
    assert m.intent_keywords == []
    assert m.inputs_schema == {}


def test_issue_carries_tier():
    i = Issue(check="shell_allowlist", max_tier=2, detail="unknown bin: foo")
    assert i.max_tier == 2


def test_artifact_ref_roundtrip():
    r = ArtifactRef(
        source_id="github:anthropics/skills@/skills", name="pdf",
        fetch_url="https://example/SKILL.md", owner="anthropics",
        raw_meta={"path": "skills/pdf/SKILL.md"},
    )
    assert r.owner == "anthropics"


def test_artifact_is_query_result_shape():
    a = Artifact(
        artifact_id=1, name="anthropics-pdf", name_original="pdf",
        artifact_type="skill", kind="prompt_skill", vet_tier=0,
        score=0.81, exposure_class="inject", applies_to="execution",
        mechanizable=False, body_excerpt="...", payload={},
    )
    assert a.score == 0.81


def test_protocols_are_runtime_checkable():
    # Protocols must be importable and usable as type hints; a concrete
    # object with the right attrs satisfies isinstance when runtime_checkable.
    class FakeAdapter:
        source_type = "github_path"
        async def discover(self, source_cfg): return []
        async def fetch(self, ref): return Path(".")
    assert isinstance(FakeAdapter(), SourceAdapter)
