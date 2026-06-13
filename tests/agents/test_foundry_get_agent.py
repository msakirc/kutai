# tests/agents/test_foundry_get_agent.py
"""Task 6: summarizer is served from the Foundry as a data Profile via get_agent.

get_agent precedence: Foundry profile > legacy class instance > executor.
"""
from src.agents import get_agent


def test_summarizer_served_from_foundry():
    p = get_agent("summarizer")
    assert p.name == "summarizer"
    assert p.max_iterations == 3
    assert p.allowed_tools == ["read_file", "file_tree", "web_search"]
    # data-backed Profile, NOT the old SummarizerAgent class:
    assert type(p).__module__.startswith("finch")


def test_summarizer_singleton_identity():
    assert get_agent("summarizer") is get_agent("summarizer")


def test_other_agents_still_work():
    # coder still class-backed in Phase 1; identity still holds
    assert get_agent("coder") is get_agent("coder")
    assert get_agent("coder").name == "coder"


def test_unknown_falls_back_to_executor():
    assert get_agent("nonexistent").name == "executor"
