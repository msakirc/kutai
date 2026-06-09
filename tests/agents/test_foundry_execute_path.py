# tests/agents/test_foundry_execute_path.py
"""Task 7: execute-path smoke — coulson consumes the data Profile unchanged.

build_system_prompt must build off the data Profile (no class methods needed),
and the DB prompt override (carried on _prompt_version_override) must win over
the in-package YAML seed.
"""
from src.agents import get_agent
from coulson.context import build_system_prompt


def test_build_system_prompt_consumes_data_profile():
    p = get_agent("summarizer")
    # coulson sets these per-execution; data Profile must tolerate them:
    p._prompt_version_override = None
    msg = build_system_prompt(p, {"id": 1, "title": "t", "description": "d"})
    assert "summarization specialist" in msg


def test_db_override_wins_over_seed():
    p = get_agent("summarizer")
    p._prompt_version_override = "OVERRIDDEN PROMPT TEXT"
    msg = build_system_prompt(p, {"id": 1, "title": "t", "description": "d"})
    assert msg.startswith("OVERRIDDEN PROMPT TEXT")
    p._prompt_version_override = None  # reset
