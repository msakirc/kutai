import pytest
from src.agents import get_agent

ACTIVE_AGENTS = [
    "implementer", "test_generator", "executor", "planner",  # Task 2 (already polished)
    "reviewer", "coder", "fixer", "researcher",
    "analyst", "writer", "summarizer", "architect",
]

LOW_TRAFFIC_AGENTS = [
    "assistant", "grader", "code_reviewer", "visual_reviewer",
    "shopping_advisor", "shopping_clarifier", "deal_analyst",
    "product_researcher", "artifact_summarizer", "integration_reviewer",
    "growth_digest_synthesizer",  # Z9 T2C — weekly growth digest synthesis
]

ALL_AGENTS = ACTIVE_AGENTS + LOW_TRAFFIC_AGENTS  # 22 total

@pytest.mark.parametrize("name", ALL_AGENTS)
def test_prompt_has_role_primer(name):
    """Prompt should start with explicit role identity."""
    p = get_agent(name).get_system_prompt({"description": "x"})
    first_line = p.strip().split("\n")[0].lower()
    assert "you are" in first_line, f"{name} prompt missing role primer"

@pytest.mark.parametrize("name", ALL_AGENTS)
def test_prompt_has_dos_and_donts(name):
    p = get_agent(name).get_system_prompt({"description": "x"}).lower()
    assert ("don't" in p or "never" in p or "do not" in p), f"{name} missing negative guardrails"
    assert ("must" in p or "always" in p), f"{name} missing positive directives"

@pytest.mark.parametrize("name", ALL_AGENTS)
def test_prompt_has_final_answer_schema(name):
    p = get_agent(name).get_system_prompt({"description": "x"})
    assert "final_answer" in p
    assert "```json" in p, f"{name} missing JSON schema example"
