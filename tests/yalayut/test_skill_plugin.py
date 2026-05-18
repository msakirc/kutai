"""Skill artifact plugin tests."""
from pathlib import Path

import pytest

from yalayut.contracts import IndexRow, Manifest, TaskContext
from yalayut.plugins.skill import SkillPlugin

FIXTURE = Path(__file__).parent / "fixtures" / "sample_skill.md"


def _row(**o):
    base = dict(
        id=1, artifact_type="skill", kind="prompt_skill",
        source="github:anthropics/skills@/skills", owner="anthropics",
        name="anthropics-pdf", name_original="pdf", version="1.0.0",
        manifest_path=None, body_excerpt="A skill for PDFs.", vet_tier=0,
        exposure_class="inject", applies_to="execution", mechanizable=False,
        model_hint=None, enabled=True,
    )
    base.update(o)
    return IndexRow(**base)


def test_plugin_satisfies_protocols():
    from yalayut.contracts import DiscoveryPlugin, AccessPlugin
    p = SkillPlugin()
    assert isinstance(p, DiscoveryPlugin)
    assert isinstance(p, AccessPlugin)


def test_parse_manifest_from_skill_md():
    p = SkillPlugin()
    m = p.parse_manifest(
        FIXTURE.read_bytes(),
        {"source_id": "github:anthropics/skills@/skills", "owner": "anthropics",
         "name": "pdf"},
    )
    assert m.name == "anthropics-pdf"
    assert m.kind == "prompt_skill"


def test_vet_checks_flags_oversize(tmp_path):
    p = SkillPlugin()
    big = tmp_path / "SKILL.md"
    big.write_text("x" * (60 * 1024))
    m = Manifest(name="x", name_original="x", version="1", artifact_type="skill",
                 kind="prompt_skill", license="MIT")
    issues = p.vet_checks(m, big)
    assert any(i.check == "body_size" for i in issues)


def test_to_application_is_inject_for_prompt_skill():
    p = SkillPlugin()
    app = p.to_application(_row(), TaskContext(title="convert a pdf"))
    assert app.exposure_class == "inject"
    assert app.applies_to == "execution"
    assert app.artifact_id == 1


def test_bind_args_none_for_non_parametric():
    p = SkillPlugin()
    # prompt_skill has no inputs_schema -> nothing to bind
    assert p.bind_args(_row(), TaskContext(title="x")) is None


def test_execute_refuses_non_mechanizable():
    p = SkillPlugin()
    res = p.execute(_row(mechanizable=False), TaskContext(), {})
    assert res.ok is False
    assert "not mechanizable" in res.detail
