import pytest
from src.agents import get_agent, AGENT_REGISTRY
from prompt_foundry import PROFILE_REGISTRY

# Build the union of Foundry-data profiles and class-backed agents,
# excluding carve-outs that have dynamic prompts (oncall_agent, writer).
_CARVE_OUTS = {"oncall_agent", "writer"}

ALL_AGENTS = sorted((set(PROFILE_REGISTRY) | set(AGENT_REGISTRY)) - _CARVE_OUTS)

_TASK_STUB = {"id": 0, "title": "x", "description": "x"}


def _get_prompt(name: str) -> str:
    return get_agent(name).get_system_prompt(_TASK_STUB)


@pytest.mark.parametrize("name", ALL_AGENTS)
def test_prompt_has_role_primer(name):
    """Prompt should start with explicit role identity."""
    p = _get_prompt(name)
    first_line = p.strip().split("\n")[0].lower()
    assert "you are" in first_line, f"{name} prompt missing role primer"

@pytest.mark.parametrize("name", ALL_AGENTS)
def test_prompt_has_dos_and_donts(name):
    p = _get_prompt(name).lower()
    assert ("don't" in p or "never" in p or "do not" in p), f"{name} missing negative guardrails"
    assert ("must" in p or "always" in p), f"{name} missing positive directives"

@pytest.mark.parametrize("name", ALL_AGENTS)
def test_prompt_has_final_answer_schema(name):
    p = _get_prompt(name)
    assert "final_answer" in p
    assert "```json" in p, f"{name} missing JSON schema example"
