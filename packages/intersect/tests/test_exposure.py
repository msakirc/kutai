"""Unit tests for the exposure-class decision matrix."""
import pytest

from intersect import exposure


def test_t3_always_quarantine(fake_artifact):
    art = fake_artifact(vet_tier=3, score=1.0)
    assert exposure.classify(art, confidence=0.99) == "quarantine"


def test_t2_quarantined_in_phase2(fake_artifact):
    # Phase 2: no sandbox — T2 stays quarantined-until-founder-promotes.
    art = fake_artifact(vet_tier=2, score=1.0)
    assert exposure.classify(art, confidence=0.99) == "quarantine"


def test_below_theta_min_dropped(fake_artifact):
    art = fake_artifact(vet_tier=0)
    assert exposure.classify(art, confidence=0.10) == "quarantine"


def test_t0_mechanizable_shell_recipe_high_conf_is_preempt(fake_artifact):
    art = fake_artifact(
        vet_tier=0, kind="shell_recipe", mechanizable=True,
        artifact_type="skill",
    )
    assert exposure.classify(art, confidence=0.90) == "preempt"


def test_t1_shell_recipe_never_preempt(fake_artifact):
    # T1 ceiling excludes preempt even for mechanizable shell recipes.
    art = fake_artifact(
        vet_tier=1, kind="shell_recipe", mechanizable=True,
    )
    assert exposure.classify(art, confidence=0.95) == "inject"


def test_api_artifact_is_tool(fake_artifact):
    art = fake_artifact(vet_tier=0, artifact_type="api", kind=None)
    assert exposure.classify(art, confidence=0.60) == "tool"


def test_mcp_artifact_is_tool(fake_artifact):
    art = fake_artifact(vet_tier=0, artifact_type="mcp", kind=None)
    assert exposure.classify(art, confidence=0.60) == "tool"


def test_prompt_skill_is_inject(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="prompt_skill")
    assert exposure.classify(art, confidence=0.70) == "inject"


def test_agent_config_is_inject(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="agent_config")
    assert exposure.classify(art, confidence=0.70) == "inject"


def test_render_prebind_when_parametric_and_bound(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="shell_recipe", mechanizable=True,
                        inputs_schema={"name": {"type": "string"}})
    # below θ_preempt but parametric + fully bound → prebind inject
    assert exposure.render_variant(art, bound_args={"name": "x"}) == "prebind"


def test_render_prose_when_unbound(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="shell_recipe",
                        inputs_schema={"name": {"type": "string"}})
    assert exposure.render_variant(art, bound_args=None) == "prose"


def test_render_prose_for_non_parametric(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="prompt_skill", inputs_schema={})
    assert exposure.render_variant(art, bound_args={}) == "prose"
