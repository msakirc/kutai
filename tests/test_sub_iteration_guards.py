"""Tests for sub-iteration guard logic in BaseAgent."""
import pytest
from src.agents.base import BaseAgent, GuardCorrection


def _make_agent(**kwargs):
    """Create a BaseAgent with optional overrides."""
    agent = BaseAgent()
    for k, v in kwargs.items():
        setattr(agent, k, v)
    return agent


# ── Hallucination guard ──

def test_hallucination_guard_returns_correction():
    agent = _make_agent(allowed_tools=["shell", "read_file"])
    parsed = {"action": "final_answer", "result": "I would run ls"}
    task = {"title": "List files in /tmp"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task=task,
        search_depth="none", suppress_guards=False,
    )
    assert c is not None
    assert c.guard_name == "hallucination"
    assert "STOP" in c.message


def test_no_hallucination_guard_after_iteration_2():
    agent = _make_agent(allowed_tools=["shell"])
    parsed = {"action": "final_answer", "result": "x"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=3, tools_used=False,
        tools_used_names=set(), task={"title": "Run deploy script"},
        search_depth="none", suppress_guards=False,
    )
    assert c is None  # iteration >= 2, guard doesn't fire


def test_no_hallucination_guard_when_tools_used():
    agent = _make_agent(allowed_tools=["shell"])
    parsed = {"action": "final_answer", "result": "x"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=True,
        tools_used_names={"shell"}, task={"title": "Run deploy script"},
        search_depth="none", suppress_guards=False,
    )
    assert c is None  # tools were used, guard doesn't fire


def test_no_hallucination_guard_for_question_task():
    """Question tasks (e.g. 'What is ...') are not action tasks."""
    agent = _make_agent(allowed_tools=["shell"])
    parsed = {"action": "final_answer", "result": "It is a fruit"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "What is an apple?"},
        search_depth="none", suppress_guards=False,
    )
    assert c is None  # question task, not an action task


def test_no_hallucination_guard_when_no_tools():
    """Agent with empty tools list should not trigger hallucination guard."""
    agent = _make_agent(allowed_tools=[])
    parsed = {"action": "final_answer", "result": "x"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "Run deploy script"},
        search_depth="none", suppress_guards=False,
    )
    assert c is None  # no tools available


# ── Search-required guard ──

def test_search_guard_fires():
    agent = _make_agent(allowed_tools=["web_search", "shell"])
    parsed = {"action": "final_answer", "result": "answer"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "Tokyo time"},
        search_depth="standard", suppress_guards=False,
    )
    assert c is not None
    assert c.guard_name == "search_required"
    assert "web_search" in c.message


def test_search_guard_no_fire_after_iteration_3():
    agent = _make_agent(allowed_tools=["web_search"])
    parsed = {"action": "final_answer", "result": "answer"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=3, tools_used=False,
        tools_used_names=set(), task={"title": "Tokyo time"},
        search_depth="standard", suppress_guards=False,
    )
    assert c is None


def test_search_guard_no_fire_when_data_fetched():
    agent = _make_agent(allowed_tools=["web_search"])
    parsed = {"action": "final_answer", "result": "answer"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=True,
        tools_used_names={"web_search"}, task={"title": "Tokyo time"},
        search_depth="standard", suppress_guards=False,
    )
    assert c is None  # web_search was used


def test_search_guard_no_fire_when_depth_none():
    agent = _make_agent(allowed_tools=["web_search"])
    parsed = {"action": "final_answer", "result": "answer"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "Tokyo time"},
        search_depth="none", suppress_guards=False,
    )
    assert c is None


# ── Blocked clarification guard ──

def test_blocked_clarification_guard():
    agent = _make_agent(_suppress_clarification=True)
    parsed = {"action": "clarify", "question": "what?"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "t"},
        search_depth="none", suppress_guards=False,
    )
    assert c is not None
    assert c.guard_name == "blocked_clarification"
    assert "final_answer" in c.message


def test_clarification_allowed_when_not_suppressed():
    agent = _make_agent(_suppress_clarification=False)
    parsed = {"action": "clarify", "question": "what?"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "t"},
        search_depth="none", suppress_guards=False,
    )
    assert c is None  # should not block


# ── Tool call ──

def test_no_guard_on_tool_call():
    agent = _make_agent(allowed_tools=["shell"])
    parsed = {"action": "tool_call", "tool": "shell", "args": {}}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "t"},
        search_depth="none", suppress_guards=False,
    )
    assert c is None


# ── suppress_guards flag ──

def test_suppress_guards_flag():
    agent = _make_agent(allowed_tools=["shell"])
    parsed = {"action": "final_answer", "result": "x"}
    c = agent._check_sub_iteration_guards(
        parsed=parsed, iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "Run deploy script"},
        search_depth="none", suppress_guards=True,
    )
    assert c is None  # guards suppressed


def test_suppress_guards_blocks_all_guards():
    """Even search guard and clarification guard should be suppressed."""
    agent = _make_agent(
        allowed_tools=["web_search"],
        _suppress_clarification=True,
    )
    # Search guard would fire
    c = agent._check_sub_iteration_guards(
        parsed={"action": "final_answer", "result": "x"},
        iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "search me"},
        search_depth="deep", suppress_guards=True,
    )
    assert c is None

    # Clarification guard would fire
    c = agent._check_sub_iteration_guards(
        parsed={"action": "clarify", "question": "what?"},
        iteration=0, tools_used=False,
        tools_used_names=set(), task={"title": "t"},
        search_depth="none", suppress_guards=True,
    )
    assert c is None


# ── GuardCorrection dataclass ──

def test_guard_correction_dataclass():
    gc = GuardCorrection(guard_name="test", message="hello")
    assert gc.guard_name == "test"
    assert gc.message == "hello"
