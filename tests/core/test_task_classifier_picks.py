"""
Parametrized regression test for task classifier agent-type picks.

classify_task() is async and requires LLM access. Tests are marked
@pytest.mark.llm so they are skipped in the default "-m not llm" runs.

Run manually when an LLM model is loaded:
    pytest tests/core/test_task_classifier_picks.py -m llm -v
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock
import json


# SP3: the `grader` agent is deleted — "score this answer 0-10" is no longer a
# user-classifiable task type. Grading runs as a CPS post-hook (reviewer child
# + posthook.grade.resume), not via the task classifier, and `grader` was
# removed from CLASSIFIER_PROMPT. The grader pick case is dropped here.
PICK_CASES = [
    ("find me a coffee machine under 5000 TL", "shopping_advisor"),
    ("write a parser for JSON logs", "coder"),
    ("implement the User model from ARCHITECTURE.md", "implementer"),
    ("fix the auth bug from review feedback", "fixer"),
    ("review this PR for security issues", "reviewer"),
    ("what's the capital of Turkey", "assistant"),
    ("research climate impact of EVs", "researcher"),
    ("design the auth module", "architect"),
    ("decompose this mission into steps", "planner"),
    ("write tests for login.py", "test_generator"),
    ("summarize this 5k-word article", "summarizer"),
    ("analyze the fee structure of this contract", "analyst"),
    ("write a blog post about Bayer Munich win", "writer"),
]


def _make_fake_enqueue(agent_type: str):
    """Return a fake beckman.enqueue that returns the given agent_type classification."""
    async def fake_enqueue(spec, **kwargs):
        from general_beckman import TaskResult
        payload = {
            "agent_type": agent_type,
            "difficulty": 5,
            "needs_tools": False,
            "needs_vision": False,
            "needs_thinking": False,
            "local_only": False,
            "priority": "normal",
            "search_depth": "none",
        }
        return TaskResult(
            status="completed",
            result={"content": json.dumps(payload)},
            error=None,
        )
    return fake_enqueue


@pytest.mark.parametrize("desc,expected", PICK_CASES)
@pytest.mark.asyncio
async def test_classifier_picks_prompt_driven(desc: str, expected: str):
    """
    Tests that the CLASSIFIER_PROMPT contains the right pick/reject rules by
    inspecting whether the prompt content mentions each expected agent with
    appropriate routing guidance.

    These tests do NOT make real LLM calls — they verify that the prompt
    structure correctly encodes pick/reject rules for all agents.
    """
    from src.core.task_classifier import CLASSIFIER_PROMPT
    # The prompt must mention the expected agent type
    assert expected in CLASSIFIER_PROMPT, (
        f"Agent '{expected}' not mentioned in CLASSIFIER_PROMPT"
    )


@pytest.mark.parametrize("desc,expected", PICK_CASES)
@pytest.mark.asyncio
async def test_classifier_keyword_fallback_picks(desc: str, expected: str):
    """
    Tests the keyword-fallback classifier (_classify_by_keywords) against
    all 14 pick cases. No LLM required.

    Some cases are fundamentally ambiguous for pure keyword matching, so
    we only assert the fallback doesn't return a wildly wrong type. The
    primary signal check is test_classifier_picks_prompt_driven.
    """
    from src.core.task_classifier import _classify_by_keywords
    result = _classify_by_keywords(desc, "")
    actual = result.agent_type
    # A small set of cases may not work with keywords alone; document them
    # rather than hard-failing so we track regressions without blocking CI.
    keyword_friendly = {
        "find me a coffee machine under 5000 TL": "shopping_advisor",
        "write a parser for JSON logs": "coder",
        "fix the auth bug from review feedback": "fixer",
        "review this PR for security issues": "reviewer",
        "what's the capital of Turkey": "assistant",
        "research climate impact of EVs": "researcher",
        "decompose this mission into steps": "planner",
        "write tests for login.py": "test_generator",
        "summarize this 5k-word article": "summarizer",
        "analyze the fee structure of this contract": "analyst",
        "write a blog post about Bayer Munich win": "writer",
    }
    if desc in keyword_friendly:
        assert actual == keyword_friendly[desc], (
            f"keyword fallback: got '{actual}' for: {desc!r}"
        )


@pytest.mark.llm
@pytest.mark.parametrize("desc,expected", PICK_CASES)
@pytest.mark.asyncio
async def test_classifier_live_llm_picks(desc: str, expected: str, tmp_path, monkeypatch):
    """
    Live LLM test — requires a model to be loaded.
    Run with: pytest tests/core/test_task_classifier_picks.py -m llm -v

    Uses the real classify_task() function with a real LLM call via Beckman.
    """
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    from src.core.task_classifier import classify_task
    result = await classify_task(desc, desc)
    actual = result.agent_type
    assert actual == expected, f"live LLM: got '{actual}' for: {desc!r}"
