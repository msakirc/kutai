"""Z8 T4A — oncall_agent prompt-quality + reflection-block invariants."""
from __future__ import annotations

import pytest

from src.agents import get_agent


def test_oncall_agent_registered():
    """Registry must resolve oncall_agent to a non-fallback agent."""
    agent = get_agent("oncall_agent")
    assert agent.name == "oncall_agent", (
        "oncall_agent missing from registry — would silently fallback to executor"
    )


def test_oncall_agent_prompt_role_primer():
    p = get_agent("oncall_agent").get_system_prompt({"description": "x"})
    first = p.strip().split("\n")[0].lower()
    assert "you are" in first


def test_oncall_agent_prompt_dos_and_donts():
    p = get_agent("oncall_agent").get_system_prompt({"description": "x"}).lower()
    assert ("must" in p or "always" in p), "missing positive directive"
    assert ("don't" in p or "never" in p or "do not" in p), "missing negative guardrail"


def test_oncall_agent_prompt_final_answer_schema():
    p = get_agent("oncall_agent").get_system_prompt({"description": "x"})
    assert "final_answer" in p
    assert "```json" in p


@pytest.mark.parametrize(
    "verb",
    [
        "restart_service",
        "rollback_to_last_green",
        "scale_up",
        "scale_down",
        "drain_traffic",
        "rotate_failed_key",
        "archive_flake_test",
        "escalate_to_founder",
    ],
)
def test_oncall_agent_prompt_lists_whitelisted_verb(verb: str):
    """Every whitelisted verb must appear in the prompt so the LLM sees it."""
    p = get_agent("oncall_agent").get_system_prompt({"description": "x"})
    assert verb in p, f"prompt missing whitelisted verb: {verb}"


def test_oncall_reflection_block_present():
    """Coulson reflection block must exist and cover key checks."""
    from coulson.reflection import build_reflection_prompt

    block = build_reflection_prompt(agent_name="oncall_agent", iteration=1).lower()
    for keyword in ["playbook", "whitelist", "cooldown", "reversible", "escalate"]:
        assert keyword in block, (
            f"oncall reflection block missing '{keyword}' — "
            "agent will not catch this class of mistake"
        )
