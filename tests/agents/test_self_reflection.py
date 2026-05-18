import pytest
from src.agents import get_agent

@pytest.mark.parametrize("name", ["coder", "implementer", "fixer", "test_generator"])
def test_reflection_enabled(name):
    """Code workhorses must have self-reflection on."""
    assert get_agent(name).enable_self_reflection is True, f"{name} should enable_self_reflection"

def test_build_reflection_prompt_exists():
    from coulson.reflection import build_reflection_prompt
    p = build_reflection_prompt(agent_name="coder", iteration=3)
    assert isinstance(p, str)
    assert len(p) > 30

def test_coder_reflection_checklist():
    """Reflection should ask: ran code? tests pass? TODOs left? imports verified?"""
    from coulson.reflection import build_reflection_prompt
    p = build_reflection_prompt(agent_name="coder", iteration=3).lower()
    for keyword in ["run", "test", "todo", "import"]:
        assert keyword in p, f"coder reflection missing '{keyword}'"

def test_implementer_reflection_checklist():
    from coulson.reflection import build_reflection_prompt
    p = build_reflection_prompt(agent_name="implementer", iteration=3).lower()
    for keyword in ["lint", "syntax", "spec", "interface"]:
        assert keyword in p, f"implementer reflection missing '{keyword}'"

def test_fixer_reflection_checklist():
    from coulson.reflection import build_reflection_prompt
    p = build_reflection_prompt(agent_name="fixer", iteration=3).lower()
    for keyword in ["feedback", "test", "delete"]:
        assert keyword in p, f"fixer reflection missing '{keyword}'"

def test_test_generator_reflection_checklist():
    from coulson.reflection import build_reflection_prompt
    p = build_reflection_prompt(agent_name="test_generator", iteration=3).lower()
    for keyword in ["coverage", "flak", "assert"]:
        assert keyword in p, f"test_generator reflection missing '{keyword}'"

def test_unknown_agent_falls_back_to_generic():
    """Default reflection block for agents without a specific checklist."""
    from coulson.reflection import build_reflection_prompt
    p = build_reflection_prompt(agent_name="researcher", iteration=2)
    assert isinstance(p, str)
    assert len(p) > 30
