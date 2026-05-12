"""Tests for IntegrationReviewerAgent — Z3 T2B."""
import pytest

from src.agents import get_agent, AGENT_REGISTRY
from general_beckman.posthooks import _NO_POSTHOOKS_AGENT_TYPES
from coulson.reflection import REFLECTION_BLOCKS


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return get_agent("integration_reviewer")


@pytest.fixture
def prompt(agent):
    return agent.get_system_prompt({"description": "check cross-module signatures"})


# ─── Prompt quality invariants (mirror test_prompt_quality.py) ────────────

def test_prompt_role_primer(prompt):
    first_line = prompt.strip().split("\n")[0].lower()
    assert "you are" in first_line, "integration_reviewer prompt missing role primer"


def test_prompt_dos_and_donts(prompt):
    p = prompt.lower()
    assert ("don't" in p or "never" in p or "do not" in p), (
        "integration_reviewer missing negative guardrails"
    )
    assert ("must" in p or "always" in p), (
        "integration_reviewer missing positive directives"
    )


def test_prompt_has_final_answer_schema(prompt):
    assert "final_answer" in prompt
    assert "```json" in prompt, "integration_reviewer missing JSON schema example"


# ─── Output schema shape ─────────────────────────────────────────────────

def test_prompt_verdict_field(prompt):
    assert '"verdict"' in prompt, "Output schema must declare verdict field"


def test_prompt_findings_field(prompt):
    assert '"findings"' in prompt, "Output schema must declare findings field"


def test_prompt_cross_file_field(prompt):
    assert '"file_b"' in prompt, "Output schema must show optional file_b for cross-file findings"


# ─── allowed_tools ────────────────────────────────────────────────────────

def test_allowed_tools_exact(agent):
    assert set(agent.allowed_tools) == {"read_file", "file_tree", "ast_signatures"}, (
        f"Unexpected allowed_tools: {agent.allowed_tools}"
    )


def test_no_write_tools(agent):
    for tool in agent.allowed_tools:
        assert "write" not in tool, f"Write tool found in allowed_tools: {tool}"


# ─── No-posthooks carve-out ───────────────────────────────────────────────

def test_in_no_posthooks_set():
    assert "integration_reviewer" in _NO_POSTHOOKS_AGENT_TYPES, (
        "integration_reviewer must be in _NO_POSTHOOKS_AGENT_TYPES"
    )


# ─── Reflection block ─────────────────────────────────────────────────────

def test_reflection_block_present():
    assert "integration_reviewer" in REFLECTION_BLOCKS, (
        "No reflection block for integration_reviewer in REFLECTION_BLOCKS"
    )


def test_reflection_block_content():
    block = REFLECTION_BLOCKS["integration_reviewer"]
    assert "file" in block.lower(), "Reflection must mention file checking"
    assert "signature" in block.lower(), "Reflection must mention signatures"
    assert "line" in block.lower() or "path" in block.lower(), (
        "Reflection must mention line numbers or paths"
    )


# ─── Registry ─────────────────────────────────────────────────────────────

def test_registered_in_agent_registry():
    assert "integration_reviewer" in AGENT_REGISTRY, (
        "integration_reviewer not registered in AGENT_REGISTRY"
    )


def test_get_agent_returns_correct_type():
    from src.agents.integration_reviewer import IntegrationReviewerAgent
    agent = get_agent("integration_reviewer")
    assert isinstance(agent, IntegrationReviewerAgent)


# ─── Classifier coverage ─────────────────────────────────────────────────

def test_classifier_contains_integration_reviewer_entry():
    from src.core.task_classifier import CLASSIFIER_PROMPT
    assert "integration_reviewer" in CLASSIFIER_PROMPT, (
        "task_classifier CLASSIFIER_PROMPT must mention integration_reviewer"
    )


def test_classifier_keywords_present():
    from src.core.task_classifier import CLASSIFIER_PROMPT
    keywords = ["cross-file", "cross-module", "signatures match", "boundary"]
    for kw in keywords:
        assert kw in CLASSIFIER_PROMPT, (
            f"Expected keyword '{kw}' in CLASSIFIER_PROMPT for integration_reviewer"
        )


# ─── Signatures context mention ───────────────────────────────────────────

def test_prompt_mentions_signatures_context(prompt):
    assert "signatures" in prompt.lower(), (
        "Prompt must mention the optional signatures context field"
    )


def test_prompt_signatures_authoritative(prompt):
    assert "authoritative" in prompt.lower(), (
        "Prompt must describe signatures context as authoritative"
    )
